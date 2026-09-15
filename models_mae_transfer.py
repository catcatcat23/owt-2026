"""Architecture-matched masked image model for AutoPET pretraining."""

from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed

from OWT_models import BlockLA
from util.patch_embed import PatchEmbed3Dfix
from util.pos_embed import get_2d_sincos_pos_embed


class ArchitectureMatchedMAE(nn.Module):
    """Six-layer LA encoder and shared decoder used by OrganSlot Arm B."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        encoder_depth=6,
        encoder_heads=12,
        decoder_dim=768,
        decoder_depth=8,
        decoder_heads=16,
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_pix_loss=False,
        frames=None,
        temporal_patch_size=1,
    ):
        super().__init__()
        if embed_dim != decoder_dim:
            raise ValueError("transfer MAE requires equal encoder/decoder dimensions")
        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.in_chans = int(in_chans)
        self.encoder_depth = int(encoder_depth)
        self.decoder_depth = int(decoder_depth)
        self.norm_pix_loss = bool(norm_pix_loss)
        self.frames = None if frames is None else int(frames)
        self.temporal_patch_size = int(temporal_patch_size)
        if self.frames is not None and self.frames % self.temporal_patch_size:
            raise ValueError("frames must be divisible by temporal_patch_size")

        if self.frames is None:
            self.patch_embed = PatchEmbed(
                self.img_size, self.patch_size, self.in_chans, embed_dim
            )
        else:
            self.patch_embed = PatchEmbed3Dfix(
                self.img_size, self.patch_size, self.in_chans, embed_dim,
                num_frames=self.frames, temp_stride=self.temporal_patch_size,
            )
        patch_count = int(self.patch_embed.num_patches)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if self.frames is None:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, patch_count + 1, embed_dim), requires_grad=False
            )
        else:
            temporal, height, width = self.patch_embed.grid_size
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(1, height * width, embed_dim), requires_grad=False
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, temporal, embed_dim)
            )
        self.blocks = nn.ModuleList(
            [
                BlockLA(
                    embed_dim,
                    encoder_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(self.encoder_depth)
            ]
        )
        self.encoder_norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        if self.frames is None:
            self.decoder_pos_embed = nn.Parameter(
                torch.zeros(1, patch_count, decoder_dim), requires_grad=False
            )
        else:
            temporal, height, width = self.patch_embed.grid_size
            self.decoder_pos_embed_spatial = nn.Parameter(
                torch.zeros(1, height * width, decoder_dim), requires_grad=False
            )
            self.decoder_pos_embed_temporal = nn.Parameter(
                torch.zeros(1, temporal, decoder_dim)
            )
        self.decoder_blocks = nn.ModuleList(
            [
                BlockLA(
                    decoder_dim,
                    decoder_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(self.decoder_depth)
            ]
        )
        self.decoder_norm = norm_layer(decoder_dim)
        patch_values = (
            self.temporal_patch_size * self.patch_size ** 2 * self.in_chans
        )
        if self.frames is None:
            self.decoder_pred = nn.Linear(decoder_dim, patch_values)
        else:
            spatial_values = self.patch_size ** 2 * self.in_chans
            self.decoder_pred = nn.Sequential(
                nn.Linear(decoder_dim, spatial_values),
                nn.Tanh(),
                nn.Linear(spatial_values, patch_values),
            )
        self.initialize_weights()

    def initialize_weights(self):
        spatial_size = self.img_size // self.patch_size
        if self.frames is None:
            encoder_pos = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1], spatial_size, cls_token=True
            )
            self.pos_embed.data.copy_(
                torch.from_numpy(encoder_pos).float().unsqueeze(0)
            )
            decoder_pos = get_2d_sincos_pos_embed(
                self.decoder_pos_embed.shape[-1], spatial_size, cls_token=False
            )
            self.decoder_pos_embed.data.copy_(
                torch.from_numpy(decoder_pos).float().unsqueeze(0)
            )
        else:
            encoder_pos = get_2d_sincos_pos_embed(
                self.pos_embed_spatial.shape[-1], spatial_size, cls_token=False
            )
            self.pos_embed_spatial.data.copy_(
                torch.from_numpy(encoder_pos).float().unsqueeze(0)
            )
            decoder_pos = get_2d_sincos_pos_embed(
                self.decoder_pos_embed_spatial.shape[-1],
                spatial_size,
                cls_token=False,
            )
            self.decoder_pos_embed_spatial.data.copy_(
                torch.from_numpy(decoder_pos).float().unsqueeze(0)
            )
            nn.init.normal_(self.pos_embed_temporal, std=0.02)
            nn.init.normal_(self.decoder_pos_embed_temporal, std=0.02)
        weight = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(weight.view(weight.shape[0], -1))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def patchify(self, images):
        patch = self.patch_size
        if self.frames is not None:
            if images.shape[-3:] != (self.frames, self.img_size, self.img_size):
                raise ValueError("3D input size does not match MAE configuration")
            temporal_patch = self.temporal_patch_size
            temporal = self.frames // temporal_patch
            height = width = self.img_size // patch
            values = images.reshape(
                images.shape[0],
                self.in_chans,
                temporal,
                temporal_patch,
                height,
                patch,
                width,
                patch,
            )
            values = values.permute(0, 2, 4, 6, 3, 5, 7, 1)
            return values.reshape(
                images.shape[0],
                temporal * height * width,
                temporal_patch * patch * patch * self.in_chans,
            )
        if images.shape[-2:] != (self.img_size, self.img_size):
            raise ValueError("input size does not match MAE configuration")
        height = width = self.img_size // patch
        values = images.reshape(
            images.shape[0], self.in_chans, height, patch, width, patch
        )
        values = torch.einsum("nchpwq->nhwpqc", values)
        return values.reshape(
            images.shape[0], height * width, patch * patch * self.in_chans
        )

    @staticmethod
    def random_masking(tokens, mask_ratio):
        batch, length, channels = tokens.shape
        keep_count = int(length * (1.0 - float(mask_ratio)))
        if keep_count <= 0 or keep_count >= length:
            raise ValueError("mask_ratio must retain at least one but not all patches")
        noise = torch.rand(batch, length, device=tokens.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :keep_count]
        visible = torch.gather(
            tokens, 1, ids_keep.unsqueeze(-1).expand(-1, -1, channels)
        )
        mask = torch.ones(batch, length, device=tokens.device)
        mask[:, :keep_count] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return visible, mask, ids_restore

    def forward_encoder(self, images, mask_ratio):
        if self.frames is None:
            position = self.pos_embed[:, 1:]
            cls_position = self.pos_embed[:, :1]
        else:
            temporal, height, width = self.patch_embed.grid_size
            position = self.pos_embed_spatial.repeat(1, temporal, 1)
            position = position + torch.repeat_interleave(
                self.pos_embed_temporal, height * width, dim=1
            )
            cls_position = torch.zeros_like(self.cls_token)
        tokens = self.patch_embed(images) + position
        tokens, mask, ids_restore = self.random_masking(tokens, mask_ratio)
        cls_token = (self.cls_token + cls_position).expand(
            tokens.shape[0], -1, -1
        )
        tokens = torch.cat((cls_token, tokens), dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return self.encoder_norm(tokens), mask, ids_restore

    def forward_decoder(self, latent, ids_restore):
        visible = self.decoder_embed(latent[:, 1:])
        mask_tokens = self.mask_token.expand(
            visible.shape[0], ids_restore.shape[1] - visible.shape[1], -1
        )
        tokens = torch.cat((visible, mask_tokens), dim=1)
        tokens = torch.gather(
            tokens,
            1,
            ids_restore.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]),
        )
        if self.frames is None:
            decoder_position = self.decoder_pos_embed
        else:
            temporal, height, width = self.patch_embed.grid_size
            decoder_position = self.decoder_pos_embed_spatial.repeat(
                1, temporal, 1
            ) + torch.repeat_interleave(
                self.decoder_pos_embed_temporal, height * width, dim=1
            )
        tokens = tokens + decoder_position
        for block in self.decoder_blocks:
            tokens = block(tokens)
        return self.decoder_pred(self.decoder_norm(tokens))

    def forward_loss(self, images, prediction, mask):
        target = self.patchify(images)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            variance = target.var(dim=-1, keepdim=True)
            target = (target - mean) / torch.sqrt(variance + 1e-6)
        loss = (prediction.float() - target.float()).pow(2).mean(dim=-1)
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)

    def forward(self, images, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(images, mask_ratio)
        prediction = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(images, prediction, mask)
        return loss, prediction, mask


def mae_transfer_base_patch16(**kwargs):
    return ArchitectureMatchedMAE(**kwargs)


def mae_transfer_3d_base_patch16(**kwargs):
    return ArchitectureMatchedMAE(frames=4, temporal_patch_size=1, **kwargs)
