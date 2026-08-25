"""Minimal E-OWT-Seg model with explicit appendable organ slots."""

import copy
from functools import partial
import math

import torch
import torch.nn as nn

import OWT_models
from OrganSlotEmbed import (
    OrganSlot,
    OrganSlotBank,
    SharedPixelQueryDecoder2D,
)


FUSION_MODES = (
    "post_layernorm",
    "linear_sqrt",
    "linear_fixed_sqrt",
)


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
        fusion_reference_count=None,
        slot_head_type="linear",
        slot_head_channels=128,
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
            "head_type": slot_head_type,
            "head_channels": slot_head_channels,
        }
        self.pixel_query_decoder = None
        if slot_head_type == "query_dot":
            if model_args.dataset_type != "2D":
                raise ValueError("query_dot head currently supports 2D only")
            self.pixel_query_decoder = SharedPixelQueryDecoder2D(
                embed_dim,
                grid_size,
                channels=slot_head_channels,
            )

        for spec in slot_specs:
            self.append_slot(
                spec["name"],
                int(spec["raw_class_id"]),
                init_from=None,
            )

        if fusion_mode not in FUSION_MODES:
            raise ValueError(f"fusion_mode must be one of {FUSION_MODES}")
        self.fusion_mode = fusion_mode
        if fusion_reference_count is None:
            fusion_reference_count = len(self.slot_names)
        self.fusion_reference_count = int(fusion_reference_count)
        if self.fusion_reference_count <= 0:
            raise ValueError("fusion_reference_count must be positive")
        # Keep this module in both variants so checkpoints remain structurally
        # compatible. Linear fusion ablations bypass and freeze it.
        self.fusion_norm = norm_layer(decoder_embed_dim)
        if self.fusion_mode != "post_layernorm":
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
        if self.fusion_mode == "linear_fixed_sqrt":
            fused = canvas_sum / math.sqrt(self.fusion_reference_count)
        else:
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
        head_compute_mask=None,
        pixel_features=None,
        decode_heads=True,
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
        if head_compute_mask is None:
            head_compute_mask = torch.ones_like(slot_keep_mask)
        if head_compute_mask.shape != slot_keep_mask.shape:
            raise ValueError("head_compute_mask has incompatible shape")
        head_compute_mask = head_compute_mask.to(device=z.device, dtype=torch.bool)

        slot_logits = {}
        calibrated_logits = {}
        canvases_for_fusion = {}
        diagnostics = {
            "slot_tokens": {},
            "slot_canvases": {},
            "collector_attention": {},
            "aher_attention": {},
        }
        slot_compute_mask = slot_keep_mask.clone()
        if decode_heads:
            slot_compute_mask |= head_compute_mask

        for slot_index, name in enumerate(slot_names):
            slot = self.slot_bank.get_slot(name)
            active_slot_rows = torch.nonzero(
                slot_compute_mask[:, slot_index], as_tuple=False
            ).flatten()
            if active_slot_rows.numel():
                active_canvas, active_tokens, active_collector, active_aher = (
                    slot.forward_canvas(z.index_select(0, active_slot_rows))
                )
                canvas = active_canvas.new_zeros(
                    (batch_size,) + tuple(active_canvas.shape[1:])
                ).index_copy(0, active_slot_rows, active_canvas)
                tokens = active_tokens.new_zeros(
                    (batch_size,) + tuple(active_tokens.shape[1:])
                ).index_copy(0, active_slot_rows, active_tokens)
                collector_attention = active_collector.new_zeros(
                    (batch_size,) + tuple(active_collector.shape[1:])
                ).index_copy(0, active_slot_rows, active_collector)
                aher_attention = active_aher.new_zeros(
                    (batch_size,) + tuple(active_aher.shape[1:])
                ).index_copy(0, active_slot_rows, active_aher)
            else:
                decoder_dim = slot.aher.sp_linear2.out_features
                canvas = z.new_zeros(
                    batch_size, slot.output_patch_count, decoder_dim
                )
                tokens = z.new_zeros(
                    batch_size, slot.token_factor, z.shape[-1]
                )
                collector_attention = z.new_zeros(
                    batch_size, slot.token_factor, z.shape[1]
                )
                aher_attention = z.new_zeros(
                    batch_size, slot.output_patch_count, slot.token_factor
                )
            canvases_for_fusion[name] = canvas

            if decode_heads:
                active_head_rows = torch.nonzero(
                    head_compute_mask[:, slot_index], as_tuple=False
                ).flatten()
                if active_head_rows.numel():
                    if slot.head_type == "query_dot":
                        if (
                            pixel_features is None
                            or self.pixel_query_decoder is None
                        ):
                            raise RuntimeError(
                                "query_dot requires shared pixel features"
                            )
                        active_raw, active_calibrated = slot.forward_query_head(
                            tokens.index_select(0, active_head_rows),
                            pixel_features.index_select(0, active_head_rows),
                            self.pixel_query_decoder,
                            output_size,
                        )
                    else:
                        active_raw, active_calibrated = slot.forward_head(
                            canvas.index_select(0, active_head_rows), output_size
                        )
                    full_shape = (batch_size,) + tuple(active_raw.shape[1:])
                    raw = active_raw.new_zeros(full_shape).index_copy(
                        0, active_head_rows, active_raw
                    )
                    calibrated = active_calibrated.new_zeros(full_shape).index_copy(
                        0, active_head_rows, active_calibrated
                    )
                else:
                    full_shape = (
                        batch_size, 1, *tuple(int(v) for v in output_size)
                    )
                    raw = canvas.new_zeros(full_shape)
                    calibrated = canvas.new_zeros(full_shape)
                slot_logits[name] = raw
                calibrated_logits[name] = calibrated
            if return_diagnostics:
                diagnostics["slot_tokens"][name] = tokens
                diagnostics["slot_canvases"][name] = canvas
                diagnostics["collector_attention"][name] = collector_attention
                diagnostics["aher_attention"][name] = aher_attention

        fused = self.fuse_canvases(
            canvases_for_fusion, slot_keep_mask, slot_names
        )
        output = {
            "canvas": fused,
            "slot_logits": slot_logits,
            "calibrated_logits": calibrated_logits,
            "slot_keep_mask": slot_keep_mask,
            "slot_compute_mask": slot_compute_mask,
            "head_compute_mask": head_compute_mask,
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
        decode_reconstruction=True,
        head_compute_mask=None,
        decode_heads=True,
    ):
        z, _ = self.forward_encoder(images)
        output_size = tuple(images.shape[2:])
        pixel_features = None
        if decode_heads and self.pixel_query_decoder is not None:
            pixel_features = self.pixel_query_decoder.forward_pixels(z)
        slot_output = self.forward_slots(
            z,
            output_size,
            slot_keep_mask=slot_keep_mask,
            head_compute_mask=head_compute_mask,
            pixel_features=pixel_features,
            decode_heads=decode_heads,
            return_diagnostics=return_diagnostics,
        )
        if decode_reconstruction:
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

    def freeze_for_all_heads(self, train_calibration=True):
        """Freeze the feature/reconstruction path and train every binary head."""
        for parameter in self.parameters():
            parameter.requires_grad = False
        for name in self.slot_names:
            slot = self.slot_bank.get_slot(name)
            for parameter in slot.head.parameters():
                parameter.requires_grad = True
            if train_calibration:
                slot.calibration_scale.requires_grad = True
                slot.calibration_bias.requires_grad = True
        if self.pixel_query_decoder is not None:
            for parameter in self.pixel_query_decoder.parameters():
                parameter.requires_grad = True
        return self.parameter_report()

    def unfreeze_all(self):
        """Enable the controlled sequential-finetuning baseline."""
        for parameter in self.parameters():
            parameter.requires_grad = True
        if self.fusion_mode != "post_layernorm":
            for parameter in self.fusion_norm.parameters():
                parameter.requires_grad = False
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
            "fusion_reference_count": self.fusion_reference_count,
            "slot_head_type": self._slot_factory["head_type"],
            "slot_head_channels": self._slot_factory["head_channels"],
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
