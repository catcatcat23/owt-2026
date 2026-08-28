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


def _group_count(channels):
    """Use the largest small GroupNorm divisor for tiny and full models."""
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class MultiScaleBinaryHead2D(nn.Module):
    """Lightweight local decoder for a retained 2D slot canvas."""

    def __init__(self, dim, grid_size, channels=128, upsample_stages=4):
        super().__init__()
        self.grid_size = tuple(int(value) for value in grid_size)
        if len(self.grid_size) != 2:
            raise ValueError("MultiScaleBinaryHead2D requires a 2D grid")
        self.channels = min(int(channels), int(dim))
        if self.channels <= 0:
            raise ValueError("head channels must be positive")
        self.upsample_stages = int(upsample_stages)
        if self.upsample_stages < 1:
            raise ValueError("upsample_stages must be positive")
        self.norm = nn.LayerNorm(dim)
        self.input_proj = nn.Linear(dim, self.channels)
        blocks = []
        current_channels = self.channels
        for _ in range(self.upsample_stages):
            next_channels = max(16, current_channels // 2)
            blocks.append(nn.Sequential(
                nn.Conv2d(current_channels, current_channels, 3, padding=1,
                          groups=current_channels, bias=False),
                nn.Conv2d(current_channels, next_channels, 1, bias=False),
                nn.GroupNorm(_group_count(next_channels), next_channels),
                nn.GELU(),
            ))
            current_channels = next_channels
        self.blocks = nn.ModuleList(blocks)
        self.proj = nn.Conv2d(current_channels, 1, kernel_size=1)

    def forward(self, canvas, output_size):
        batch_size, patch_count, _ = canvas.shape
        if patch_count != int(torch.tensor(self.grid_size).prod().item()):
            raise ValueError("canvas patch count does not match head grid")
        output_size = tuple(int(value) for value in output_size)
        if len(output_size) != 2:
            raise ValueError("MultiScaleBinaryHead2D output must be 2D")
        features = self.input_proj(self.norm(canvas))
        features = features.transpose(1, 2).reshape(
            batch_size, self.channels, *self.grid_size
        )
        for block in self.blocks:
            features = F.interpolate(
                features, scale_factor=2.0, mode="bilinear", align_corners=False
            )
            features = block(features)
        logits = self.proj(features)
        if tuple(logits.shape[-2:]) != output_size:
            logits = F.interpolate(
                logits, size=output_size, mode="bilinear", align_corners=False
            )
        return logits


class MultiScaleBinaryHead3D(nn.Module):
    """Anisotropic 3D counterpart of Arm B's local convolutional head.

    The ViT patch grid already has the requested temporal/depth resolution.
    Each stage therefore doubles only the in-plane resolution while a 3x3x3
    depthwise convolution exchanges information between adjacent CT slices.
    A final trilinear resize handles temporal patch strides greater than one.
    """

    def __init__(self, dim, grid_size, channels=128, upsample_stages=4):
        super().__init__()
        self.grid_size = tuple(int(value) for value in grid_size)
        if len(self.grid_size) != 3:
            raise ValueError("MultiScaleBinaryHead3D requires a 3D grid")
        self.channels = min(int(channels), int(dim))
        if self.channels <= 0:
            raise ValueError("head channels must be positive")
        self.upsample_stages = int(upsample_stages)
        if self.upsample_stages < 1:
            raise ValueError("upsample_stages must be positive")
        self.norm = nn.LayerNorm(dim)
        self.input_proj = nn.Linear(dim, self.channels)
        blocks = []
        current_channels = self.channels
        for _ in range(self.upsample_stages):
            next_channels = max(16, current_channels // 2)
            blocks.append(nn.Sequential(
                nn.Conv3d(
                    current_channels,
                    current_channels,
                    kernel_size=3,
                    padding=1,
                    groups=current_channels,
                    bias=False,
                ),
                nn.Conv3d(current_channels, next_channels, 1, bias=False),
                nn.GroupNorm(_group_count(next_channels), next_channels),
                nn.GELU(),
            ))
            current_channels = next_channels
        self.blocks = nn.ModuleList(blocks)
        self.proj = nn.Conv3d(current_channels, 1, kernel_size=1)

    def forward(self, canvas, output_size):
        batch_size, patch_count, _ = canvas.shape
        if patch_count != int(torch.tensor(self.grid_size).prod().item()):
            raise ValueError("canvas patch count does not match head grid")
        output_size = tuple(int(value) for value in output_size)
        if len(output_size) != 3:
            raise ValueError("MultiScaleBinaryHead3D output must be 3D")
        features = self.input_proj(self.norm(canvas))
        features = features.transpose(1, 2).reshape(
            batch_size, self.channels, *self.grid_size
        )
        for block in self.blocks:
            features = F.interpolate(
                features,
                scale_factor=(1.0, 2.0, 2.0),
                mode="trilinear",
                align_corners=False,
            )
            features = block(features)
        logits = self.proj(features)
        if tuple(logits.shape[-3:]) != output_size:
            logits = F.interpolate(
                logits,
                size=output_size,
                mode="trilinear",
                align_corners=False,
            )
        return logits


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
        head_type="linear",
        head_channels=128,
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
        self.head_type = str(head_type)
        if self.head_type == "linear":
            self.head = PatchBinaryHead(decoder_dim, grid_size)
        elif self.head_type == "multiscale_conv":
            head_class = (
                MultiScaleBinaryHead2D
                if dataset_type == "2D"
                else MultiScaleBinaryHead3D
            )
            self.head = head_class(decoder_dim, grid_size, channels=head_channels)
        else:
            raise ValueError("unsupported slot head type: {}".format(head_type))
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

    def forward_head(self, canvas, output_size):
        raw_logits = self.head(canvas, output_size)
        calibrated_logits = (
            self.calibration_scale * raw_logits + self.calibration_bias
        )
        return raw_logits, calibrated_logits

    def forward(self, z, output_size, return_attention=False):
        canvas, tokens, collector_attention, aher_attention = (
            self.forward_canvas(z)
        )
        raw_logits, calibrated_logits = self.forward_head(canvas, output_size)
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
