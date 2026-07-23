from functools import partial

import torch
import torch.nn as nn

import OWT_models
from OrganEmbed import AHER


class PaddedAttentionLA(OWT_models.AttentionLA):
    """Linear attention that excludes padded token positions."""

    def forward(self, x, valid_token_mask=None):
        if valid_token_mask is None:
            return super().forward(x)

        batch_size, token_count, channels = x.shape
        if valid_token_mask.shape != (batch_size, token_count):
            raise ValueError("valid_token_mask shape must match x[:2]")
        if torch.any(valid_token_mask.sum(dim=1) == 0):
            raise ValueError("each sample must retain at least one token")

        qkv = self.qkv(x).reshape(
            batch_size, token_count, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        valid = valid_token_mask[:, None, :, None]
        q = q.softmax(dim=-1)
        q = q * valid.to(q.dtype)
        k = k.masked_fill(~valid, torch.finfo(k.dtype).min)
        k = k.softmax(dim=-2)
        k = k * valid.to(k.dtype)
        v = v * valid.to(v.dtype)

        q = q * q.shape[-1] ** -0.5
        context = torch.einsum("bhnd,bhne->bhde", k, v)
        x = torch.einsum("bhnd,bhde->bhne", q, context)
        x = x.transpose(1, 2).reshape(batch_size, token_count, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x * valid_token_mask.unsqueeze(-1).to(x.dtype)


class PaddedBlockLA(OWT_models.BlockLA):
    """OWT linear-attention block with padding-safe residual paths."""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=False,
        proj_drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        mlp_layer=OWT_models.Mlp,
    ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            mlp_layer=mlp_layer,
        )
        padded_attention = PaddedAttentionLA(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        padded_attention.load_state_dict(self.attn.state_dict())
        self.attn = padded_attention

    def forward(self, x, valid_token_mask=None):
        if valid_token_mask is None:
            return super().forward(x)

        valid = valid_token_mask.unsqueeze(-1).to(x.dtype)
        x = x * valid
        x = x + self.drop_path1(
            self.ls1(self.attn(self.norm1(x), valid_token_mask))
        )
        x = x * valid
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x * valid


class PaddedAHER(AHER):
    """AHER whose softmax ignores padded organ tokens."""

    def forward(self, input_x, valid_token_mask=None):
        if valid_token_mask is None:
            return super().forward(input_x)
        if valid_token_mask.shape != input_x.shape[:2]:
            raise ValueError("valid_token_mask shape must match input_x[:2]")
        if torch.any(valid_token_mask.sum(dim=1) == 0):
            raise ValueError("each sample must retain at least one token")

        attn_ids = self.sp_linear1(input_x).permute(0, 2, 1)
        valid = valid_token_mask.unsqueeze(1)
        attn_ids = attn_ids.masked_fill(
            ~valid, torch.finfo(attn_ids.dtype).min
        )
        attention_probs = self.softmax(attn_ids)
        attention_probs = attention_probs * valid.to(attention_probs.dtype)

        values = self.sp_linear2(input_x)
        values = values * valid_token_mask.unsqueeze(-1).to(values.dtype)
        outputs = torch.einsum("...si,...id->...sd", attention_probs, values)
        return outputs, attention_probs


class PSEMMaskedAutoencoderViT(OWT_models.MaskedAutoencoderViT):
    """OWT with per-sample variable-length organ-token masking."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        model_args=None,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            norm_pix_loss=norm_pix_loss,
            model_args=model_args,
        )
        if not self.model_args.LA:
            raise ValueError("PSEM-v1 currently requires OWT linear attention")
        if not self.model_args.arch_version.startswith("v1"):
            raise ValueError("PSEM-v1 currently supports OWT v1 architectures")

        padded_blocks = nn.ModuleList(
            [
                PaddedBlockLA(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(int(depth / 2))
            ]
        )
        padded_blocks.load_state_dict(self.blocks2.state_dict())
        self.blocks2 = padded_blocks

        padded_aher = PaddedAHER(
            embed_dim,
            self.decoder_embed.sp_linear2.out_features,
            self.model_args.organ_token_total,
            self.decoder_embed.sp_linear1.out_features,
        )
        padded_aher.load_state_dict(self.decoder_embed.state_dict())
        self.decoder_embed = padded_aher

    def pack_kept_tokens(self, tokens, class_keep_mask):
        batch_size, _, channels = tokens.shape
        class_count = self.model_args.num_classes_with_bg
        if class_keep_mask.shape != (batch_size, class_count):
            raise ValueError(
                f"class_keep_mask must be {(batch_size, class_count)}, "
                f"got {tuple(class_keep_mask.shape)}"
            )

        token_keep_mask = class_keep_mask.repeat_interleave(
            self.model_args.token_factor, dim=1
        )
        if token_keep_mask.shape[1] != tokens.shape[1]:
            raise ValueError("class mask does not match organ token count")

        lengths = token_keep_mask.sum(dim=1)
        if torch.any(lengths == 0):
            raise ValueError("each sample must retain at least one token group")
        max_length = int(lengths.max().item())

        valid_token_mask = (
            torch.arange(max_length, device=tokens.device).unsqueeze(0)
            < lengths.unsqueeze(1)
        )
        packed = tokens.new_zeros((batch_size, max_length, channels))
        packed[valid_token_mask] = tokens[token_keep_mask]
        return packed, valid_token_mask, token_keep_mask

    def forward_encoder(self, x, mask_ratio, middle=None):
        class_keep_mask = middle["class_keep_mask"].bool()

        x = self.patch_embed(x)
        if self.model_args.dataset_type == "3D":
            self.pos_embed = self.pos_embed_spatial.repeat(
                1, self.patch_embed.grid_size[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.patch_embed.grid_size[1] * self.patch_embed.grid_size[2],
                dim=1,
            )
            self.pos_embed = torch.cat([self.cls_token, self.pos_embed], 1)
            self.pos_embed = self.pos_embed[:, : x.shape[1] + 1, :]

        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for block in self.blocks1:
            x = block(x)
        x = self.norm(x)
        cls_tokens = x[:, :1, :]
        x = x[:, 1:, :]

        x, _ = self.organ_embed(x)
        if self.model_args.text_encoding != "None":
            x = x + self.text_linear(middle["text_features"])

        x_masked_b, valid_token_mask, full_token_mask = self.pack_kept_tokens(
            x, class_keep_mask
        )
        x_encoded = x_masked_b
        for block in self.blocks2:
            x_encoded = block(x_encoded, valid_token_mask)
        x_encoded = self.norm(x_encoded)
        x_masked = (x_masked_b + x_encoded) * valid_token_mask.unsqueeze(-1)

        middle_output = {
            "x_masked": x_masked,
            "x_masked_b": x_masked_b,
            "valid_token_mask": valid_token_mask,
            "full_token_mask": full_token_mask,
        }
        return x_masked, cls_tokens, middle_output

    def forward_decoder(self, x_restored, cls_tokens, middle_output):
        x, _ = self.decoder_embed(
            x_restored, middle_output["valid_token_mask"]
        )
        if self.model_args.dataset_type == "3D":
            pos_embed = self.decoder_pos_embed_spatial.repeat(
                1, self.patch_embed.grid_size[0], 1
            ) + torch.repeat_interleave(
                self.decoder_pos_embed_temporal,
                self.patch_embed.grid_size[1] * self.patch_embed.grid_size[2],
                dim=1,
            )
            x = x + pos_embed

        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return self.sigmoid(x)

    def forward(self, imgs, mask_ratio=0.75, middle=None):
        image_target = middle["image_target"]
        x_restored, cls_tokens, middle_output = self.forward_encoder(
            imgs, mask_ratio, middle=middle
        )
        pred = self.forward_decoder(x_restored, cls_tokens, middle_output)
        loss, p_loss = self.forward_loss(image_target, pred)
        if "LPIPS" in self.model_args.loss_version:
            middle_output["p_loss"] = p_loss

        if self.model_args.dataset_type == "2D":
            pred = self.unpatchify(pred)
        else:
            pred = self.unpatchify3D(pred)
        return loss, pred, middle_output


def mae_vit_base_patch16_dec512d8b(**kwargs):
    return PSEMMaskedAutoencoderViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=768,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b
mae_vit_basefix16_patch16 = mae_vit_base_patch16_dec512d8b
