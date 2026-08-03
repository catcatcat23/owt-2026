"""Minimal E-OWT-Seg model with explicit appendable organ slots."""

import copy
from functools import partial
import math

import torch
import torch.nn as nn

import OWT_models
from OrganSlotEmbed import OrganSlot, OrganSlotBank


class OrganSlotMaskedAutoencoderViT(OWT_models.MaskedAutoencoderViT):
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
        slot_specs=None,
        slot_tg_depth=1,
        fusion_mode="post_layernorm",
    ):
        if slot_specs is None:
            raise ValueError("slot_specs are required")
        if not model_args.LA:
            raise ValueError("OrganSlotBank v0 requires linear attention")
        if not model_args.arch_version.startswith("v1"):
            raise ValueError("OrganSlotBank v0 supports v1 OWT architectures")
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

        # Remove the original joint token path while preserving the original
        # encoder/decoder implementation as a controlled baseline dependency.
        del self.organ_embed
        del self.blocks2
        del self.decoder_embed
        del self.mask_token
        self.encoder_norm = self.norm
        del self.norm

        if model_args.dataset_type == "2D":
            # The pinned timm version exposes num_patches but not grid_size.
            spatial = int(img_size // patch_size)
            grid_size = (spatial, spatial)
        else:
            grid_size = tuple(int(v) for v in self.patch_embed.grid_size)
        output_patch_count = int(math.prod(grid_size))

        self.slot_bank = OrganSlotBank()
        self.slot_tg_depth = int(slot_tg_depth)
        self._slot_factory = {
            "embed_dim": embed_dim,
            "decoder_dim": decoder_embed_dim,
            "token_factor": int(model_args.token_factor),
            "output_patch_count": output_patch_count,
            "grid_size": grid_size,
            "num_heads": num_heads,
            "mlp_ratio": mlp_ratio,
            "slot_tg_depth": self.slot_tg_depth,
            "norm_layer": norm_layer,
            "dataset_type": model_args.dataset_type,
            "model_args": model_args,
        }
        for spec in slot_specs:
            self.append_slot(
                spec["name"],
                int(spec["raw_class_id"]),
                init_from=None,
            )

        if fusion_mode not in {"post_layernorm", "linear_sqrt"}:
            raise ValueError(
                "fusion_mode must be post_layernorm or linear_sqrt"
            )
        self.fusion_mode = fusion_mode
        # Keep this module in both variants so checkpoints remain structurally
        # compatible. The linear-sqrt ablation bypasses and freezes it.
        self.fusion_norm = norm_layer(decoder_embed_dim)
        if self.fusion_mode == "linear_sqrt":
            for parameter in self.fusion_norm.parameters():
                parameter.requires_grad = False

    @property
    def slot_names(self):
        return self.slot_bank.names()

    def _new_slot(self, name, raw_class_id):
        slot = OrganSlot(
            name=name,
            raw_class_id=raw_class_id,
            **self._slot_factory,
        )
        slot.apply(self._init_weights)
        return slot

    def append_slot(self, name, raw_class_id, init_from="background"):
        old_state = {
            key: value.detach().clone()
            for key, value in self.state_dict().items()
        }
        slot = self._new_slot(name, raw_class_id)
        if init_from is not None:
            slot.copy_from(self.slot_bank.get_slot(init_from))
            slot.semantic_name = str(name)
            slot.raw_class_id = int(raw_class_id)
            with torch.no_grad():
                slot.calibration_scale.fill_(1.0)
                slot.calibration_bias.zero_()
        self.slot_bank.add_slot(name, raw_class_id, slot)
        for key, value in old_state.items():
            if not torch.equal(self.state_dict()[key], value):
                raise RuntimeError(f"appending slot modified existing tensor {key}")
        return slot

    def forward_encoder(self, x):
        x = self.patch_embed(x)
        if self.model_args.dataset_type == "3D":
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.patch_embed.grid_size[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.patch_embed.grid_size[1] * self.patch_embed.grid_size[2],
                dim=1,
            )
            pos_embed = torch.cat([self.cls_token, pos_embed], dim=1)
        else:
            pos_embed = self.pos_embed
        pos_embed = pos_embed[:, : x.shape[1] + 1]

        x = x + pos_embed[:, 1:]
        cls_tokens = (self.cls_token + pos_embed[:, :1]).expand(
            x.shape[0], -1, -1
        )
        x = torch.cat([cls_tokens, x], dim=1)
        for block in self.blocks1:
            x = block(x)
        x = self.encoder_norm(x)
        return x[:, 1:], x[:, :1]

    def fuse_canvases(self, canvases, slot_keep_mask, slot_names=None):
        if slot_names is None:
            slot_names = tuple(canvases)
        if not slot_names:
            raise ValueError("at least one slot canvas is required")
        first = canvases[slot_names[0]]
        batch_size = first.shape[0]
        if slot_keep_mask.shape != (batch_size, len(slot_names)):
            raise ValueError("slot_keep_mask has incompatible shape")
        slot_keep_mask = slot_keep_mask.to(device=first.device, dtype=torch.bool)
        retained_count = slot_keep_mask.sum(dim=1)
        if torch.any(retained_count == 0):
            raise ValueError("every sample must retain at least one slot")

        canvas_sum = torch.zeros_like(first)
        for index, name in enumerate(slot_names):
            canvas = canvases[name]
            if canvas.shape != first.shape:
                raise ValueError("all slot canvases must share a shape")
            weight = slot_keep_mask[:, index, None, None].to(canvas.dtype)
            canvas_sum = canvas_sum + canvas * weight
        fused = canvas_sum / retained_count.sqrt()[:, None, None].to(
            canvas_sum.dtype
        )
        if self.fusion_mode == "post_layernorm":
            return self.fusion_norm(fused)
        return fused

    def forward_slots(
        self,
        z,
        output_size,
        slot_keep_mask=None,
        return_diagnostics=False,
    ):
        slot_names = self.slot_names
        batch_size = z.shape[0]
        if slot_keep_mask is None:
            slot_keep_mask = torch.ones(
                batch_size,
                len(slot_names),
                dtype=torch.bool,
                device=z.device,
            )
        if slot_keep_mask.shape != (batch_size, len(slot_names)):
            raise ValueError("slot_keep_mask has incompatible shape")
        if torch.any(slot_keep_mask.sum(dim=1) == 0):
            raise ValueError("every sample must retain at least one slot")

        slot_logits = {}
        calibrated_logits = {}
        canvases_for_fusion = {}
        diagnostics = {
            "slot_tokens": {},
            "slot_canvases": {},
            "collector_attention": {},
            "aher_attention": {},
        }
        for name in slot_names:
            result = self.slot_bank(
                name,
                z,
                output_size,
                return_attention=return_diagnostics,
            )
            slot_logits[name] = result["logits"]
            calibrated_logits[name] = result["calibrated_logits"]
            canvases_for_fusion[name] = result["canvas"]
            if return_diagnostics:
                diagnostics["slot_tokens"][name] = result["tokens"]
                diagnostics["slot_canvases"][name] = result["canvas"]
                diagnostics["collector_attention"][name] = result[
                    "collector_attention"
                ]
                diagnostics["aher_attention"][name] = result["aher_attention"]

        fused = self.fuse_canvases(
            canvases_for_fusion, slot_keep_mask, slot_names
        )
        output = {
            "canvas": fused,
            "slot_logits": slot_logits,
            "calibrated_logits": calibrated_logits,
            "slot_keep_mask": slot_keep_mask,
        }
        if return_diagnostics:
            output.update(diagnostics)
        return output

    def forward_decoder(self, canvas):
        x = canvas
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
        x = self.sigmoid(self.decoder_pred(x))
        if self.model_args.dataset_type == "2D":
            return self.unpatchify(x)
        return self.unpatchify3D(x)

    def forward(
        self,
        images,
        slot_keep_mask=None,
        return_diagnostics=False,
    ):
        z, _ = self.forward_encoder(images)
        output_size = tuple(images.shape[2:])
        slot_output = self.forward_slots(
            z,
            output_size,
            slot_keep_mask=slot_keep_mask,
            return_diagnostics=return_diagnostics,
        )
        slot_output["reconstruction"] = self.forward_decoder(
            slot_output["canvas"]
        )
        return slot_output

    def freeze_for_incremental(
        self,
        old_slots,
        new_slots,
        background_policy="frozen",
        train_calibration=False,
    ):
        old_slots = set(old_slots)
        new_slots = set(new_slots)
        known_slots = set(self.slot_names)
        unknown = (old_slots | new_slots) - known_slots
        if unknown:
            raise KeyError(f"unknown incremental slots: {sorted(unknown)}")
        if old_slots & new_slots:
            raise ValueError("old_slots and new_slots must be disjoint")
        for parameter in self.parameters():
            parameter.requires_grad = False
        for name in self.slot_names:
            slot = self.slot_bank.get_slot(name)
            if name in new_slots:
                slot.set_trainable(True)
                if not train_calibration:
                    slot.calibration_scale.requires_grad = False
                    slot.calibration_bias.requires_grad = False
            elif name in old_slots:
                slot.set_trainable(False)
            elif name == "background":
                if background_policy == "frozen":
                    slot.set_trainable(False)
                elif background_policy == "plastic":
                    slot.set_trainable(False)
                    for module in (slot.collector, slot.aher, slot.head):
                        for parameter in module.parameters():
                            parameter.requires_grad = True
                    slot.calibration_scale.requires_grad = True
                    slot.calibration_bias.requires_grad = True
                else:
                    raise ValueError(
                        f"unsupported background_policy: {background_policy}"
                    )
        return self.parameter_report()

    def freeze_for_head_only(self, new_slot, train_calibration=False):
        """Freeze the complete network except the new slot's binary head."""
        if new_slot not in self.slot_names:
            raise KeyError(f"unknown slot: {new_slot}")
        for parameter in self.parameters():
            parameter.requires_grad = False
        slot = self.slot_bank.get_slot(new_slot)
        for parameter in slot.head.parameters():
            parameter.requires_grad = True
        if train_calibration:
            slot.calibration_scale.requires_grad = True
            slot.calibration_bias.requires_grad = True
        return self.parameter_report()

    def unfreeze_all(self):
        """Enable the controlled sequential-finetuning baseline."""
        for parameter in self.parameters():
            parameter.requires_grad = True
        return self.parameter_report()

    def parameter_report(self):
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        per_slot = {}
        for name in self.slot_names:
            slot = self.slot_bank.get_slot(name)
            per_slot[name] = {
                "total": sum(p.numel() for p in slot.parameters()),
                "trainable": sum(
                    p.numel() for p in slot.parameters() if p.requires_grad
                ),
            }
        return {
            "total": total,
            "trainable": trainable,
            "fusion_mode": self.fusion_mode,
            "per_slot": per_slot,
        }


def mae_vit_base_patch16_dec512d8b(**kwargs):
    return OrganSlotMaskedAutoencoderViT(
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
