"""Explicit organ-wise slot modules for E-OWT-Seg."""

import copy
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

import OWT_models
from OrganEmbed import AHER, OrganCollector, OrganCollector3D


def sanitize_slot_name(name):
    sanitized = re.sub(r"[^0-9a-zA-Z_]+", "_", str(name)).strip("_").lower()
    if not sanitized:
        raise ValueError("slot name must contain an alphanumeric character")
    if sanitized[0].isdigit():
        sanitized = f"slot_{sanitized}"
    return sanitized


class PatchBinaryHead(nn.Module):
    """Minimal LayerNorm+Linear head on an AHER patch canvas."""

    def __init__(self, dim, grid_size):
        super().__init__()
        self.grid_size = tuple(int(value) for value in grid_size)
        if len(self.grid_size) not in (2, 3):
            raise ValueError("grid_size must be 2D or 3D")
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, 1)

    def forward(self, canvas, output_size):
        batch_size, patch_count, _ = canvas.shape
        if patch_count != int(torch.tensor(self.grid_size).prod().item()):
            raise ValueError("canvas patch count does not match head grid")
        logits = self.proj(self.norm(canvas)).squeeze(-1)
        logits = logits.reshape(batch_size, 1, *self.grid_size)
        output_size = tuple(int(value) for value in output_size)
        if len(output_size) != len(self.grid_size):
            raise ValueError("output_size dimensionality does not match grid")
        mode = "bilinear" if len(output_size) == 2 else "trilinear"
        return F.interpolate(
            logits, size=output_size, mode=mode, align_corners=False
        )


class OrganSlot(nn.Module):
    """One collector, local TGEnc, AHER, head, and calibration pair."""

    def __init__(
        self,
        name,
        raw_class_id,
        embed_dim,
        decoder_dim,
        token_factor,
        output_patch_count,
        grid_size,
        num_heads,
        mlp_ratio,
        slot_tg_depth,
        norm_layer,
        dataset_type,
        model_args,
    ):
        super().__init__()
        self.semantic_name = str(name)
        self.raw_class_id = int(raw_class_id)
        self.token_factor = int(token_factor)
        self.output_patch_count = int(output_patch_count)
        self.dataset_type = dataset_type

        if dataset_type == "2D":
            self.collector = OrganCollector(
                embed_dim,
                embed_dim,
                self.token_factor,
                int(grid_size[-1]),
            )
        elif dataset_type == "3D":
            self.collector = OrganCollector3D(
                embed_dim,
                embed_dim,
                self.token_factor,
                int(grid_size[-1]),
                model_args,
            )
        else:
            raise ValueError(f"unsupported dataset_type: {dataset_type}")

        self.tg_encoder = nn.ModuleList(
            [
                OWT_models.BlockLA(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(int(slot_tg_depth))
            ]
        )
        self.token_norm = norm_layer(embed_dim)
        self.aher = AHER(
            embed_dim,
            decoder_dim,
            self.token_factor,
            self.output_patch_count,
        )
        self.head = PatchBinaryHead(decoder_dim, grid_size)
        self.calibration_scale = nn.Parameter(torch.ones(()))
        self.calibration_bias = nn.Parameter(torch.zeros(()))

    def forward_tokens(self, z):
        tokens, collector_attention = self.collector(z)
        encoded = tokens
        for block in self.tg_encoder:
            encoded = block(encoded)
        if self.tg_encoder:
            encoded = tokens + self.token_norm(encoded)
        return encoded, collector_attention

    def forward_canvas(self, z):
        tokens, collector_attention = self.forward_tokens(z)
        canvas, aher_attention = self.aher(tokens)
        return canvas, tokens, collector_attention, aher_attention

    def forward(self, z, output_size, return_attention=False):
        canvas, tokens, collector_attention, aher_attention = (
            self.forward_canvas(z)
        )
        raw_logits = self.head(canvas, output_size)
        calibrated_logits = (
            self.calibration_scale * raw_logits + self.calibration_bias
        )
        output = {
            "tokens": tokens,
            "canvas": canvas,
            "logits": raw_logits,
            "calibrated_logits": calibrated_logits,
        }
        if return_attention:
            output["collector_attention"] = collector_attention
            output["aher_attention"] = aher_attention
        return output

    def set_trainable(self, enabled):
        for parameter in self.parameters():
            parameter.requires_grad = bool(enabled)
        return self

    def copy_from(self, other_slot, reset_head_bias=False):
        if not isinstance(other_slot, OrganSlot):
            raise TypeError("other_slot must be an OrganSlot")
        self.load_state_dict(copy.deepcopy(other_slot.state_dict()), strict=True)
        if reset_head_bias and self.head.proj.bias is not None:
            nn.init.zeros_(self.head.proj.bias)
        return self


class OrganSlotBank(nn.Module):
    def __init__(self):
        super().__init__()
        self.slots = nn.ModuleDict()
        self._semantic_to_key = {}
        self._metadata = {}

    def add_slot(self, name, raw_class_id, slot):
        key = sanitize_slot_name(name)
        if name in self._semantic_to_key or key in self.slots:
            raise ValueError(f"duplicate slot: {name}")
        if not isinstance(slot, OrganSlot):
            raise TypeError("slot must be an OrganSlot")
        self.slots[key] = slot
        self._semantic_to_key[str(name)] = key
        self._metadata[str(name)] = {
            "key": key,
            "raw_class_id": int(raw_class_id),
        }
        return slot

    def get_slot(self, name):
        try:
            return self.slots[self._semantic_to_key[str(name)]]
        except KeyError as error:
            raise KeyError(f"unknown slot: {name}") from error

    def names(self):
        return tuple(self._semantic_to_key)

    def metadata(self):
        return copy.deepcopy(self._metadata)

    def forward(self, name, z, output_size, return_attention=False):
        return self.get_slot(name)(
            z, output_size, return_attention=return_attention
        )
