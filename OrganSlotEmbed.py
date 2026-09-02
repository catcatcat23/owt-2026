"""Explicit organ-wise slot modules for E-OWT-Seg."""

import copy
import math
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


class SlotQueryEmbedding(nn.Module):
    """Per-slot identity used by the shared query-mask decoder."""

    def __init__(self, dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(int(dim)))
        nn.init.normal_(self.embedding, std=0.02)


class SharedPixelQueryDecoder2D(nn.Module):
    """Shared spatial decoder and token-to-mask query projection."""

    def __init__(
        self,
        embed_dim,
        grid_size,
        channels=128,
        upsample_stages=2,
        multi_query=False,
    ):
        super().__init__()
        self.grid_size = tuple(int(value) for value in grid_size)
        if len(self.grid_size) != 2:
            raise ValueError("SharedPixelQueryDecoder2D requires a 2D grid")
        self.channels = int(channels)
        if self.channels <= 0:
            raise ValueError("query-mask channels must be positive")
        self.upsample_stages = int(upsample_stages)
        if self.upsample_stages < 1:
            raise ValueError("upsample_stages must be positive")
        self.multi_query = bool(multi_query)

        self.pixel_norm = nn.LayerNorm(embed_dim)
        self.pixel_proj = nn.Linear(embed_dim, self.channels)
        self.query_norm = nn.LayerNorm(embed_dim)
        self.query_proj = nn.Linear(embed_dim, self.channels)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    self.channels,
                    self.channels,
                    kernel_size=3,
                    padding=1,
                    groups=self.channels,
                    bias=False,
                ),
                nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=False),
                nn.GroupNorm(_group_count(self.channels), self.channels),
                nn.GELU(),
            )
            for _ in range(self.upsample_stages)
        ])

    def forward_pixels(self, z):
        batch_size, patch_count, _ = z.shape
        if patch_count != int(torch.tensor(self.grid_size).prod().item()):
            raise ValueError("encoder patch count does not match pixel grid")
        features = self.pixel_proj(self.pixel_norm(z))
        features = features.transpose(1, 2).reshape(
            batch_size, self.channels, *self.grid_size
        )
        for block in self.blocks:
            features = F.interpolate(
                features, scale_factor=2.0, mode="bilinear", align_corners=False
            )
            features = block(features)
        return features

    def forward_mask(self, pixel_features, tokens, slot_identity, output_size):
        if pixel_features.shape[0] != tokens.shape[0]:
            raise ValueError("pixel features and tokens must share a batch size")
        pixel_features = F.normalize(pixel_features, dim=1)
        normalized_tokens = self.query_norm(tokens)
        if self.multi_query:
            queries = self.query_proj(normalized_tokens)
            queries = queries + slot_identity[None, None, :]
            queries = F.normalize(queries, dim=-1)
            part_logits = torch.einsum(
                "bdhw,bkd->bkhw", pixel_features, queries
            ).mul(math.sqrt(self.channels))
            logits = torch.logsumexp(part_logits, dim=1)
            logits = logits - math.log(tokens.shape[1])
        else:
            pooled_tokens = normalized_tokens.mean(dim=1)
            query = self.query_proj(pooled_tokens) + slot_identity.unsqueeze(0)
            query = F.normalize(query, dim=-1)
            logits = torch.einsum(
                "bdhw,bd->bhw", pixel_features, query
            ).mul(math.sqrt(self.channels))
        logits = logits.unsqueeze(1)
        output_size = tuple(int(value) for value in output_size)
        return F.interpolate(
            logits, size=output_size, mode="bilinear", align_corners=False
        )


class SpatialStem2D(nn.Module):
    """Lightweight image encoder providing genuine stride-4/8 features."""

    def __init__(self, in_channels, channels):
        super().__init__()
        channels = int(channels)
        hidden_channels = max(16, channels // 2)
        self.to_p2 = nn.Sequential(
            nn.Conv2d(
                int(in_channels), hidden_channels, 3, stride=2, padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.GELU(),
        )
        self.to_p4 = nn.Sequential(
            nn.Conv2d(
                hidden_channels, channels, 3, stride=2, padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(channels), channels),
            nn.GELU(),
        )
        self.to_p8 = nn.Sequential(
            nn.Conv2d(
                channels, channels, 3, stride=2, padding=1, bias=False
            ),
            nn.GroupNorm(_group_count(channels), channels),
            nn.GELU(),
        )

    def forward(self, images):
        p2 = self.to_p2(images)
        p4 = self.to_p4(p2)
        p8 = self.to_p8(p4)
        return p4, p8


class MultiScalePixelQueryDecoder2D(SharedPixelQueryDecoder2D):
    """Arm E pixel path: ViT P16 semantics fused with image P8/P4 detail."""

    def __init__(self, in_channels, embed_dim, grid_size, channels=128):
        super().__init__(
            embed_dim,
            grid_size,
            channels=channels,
            upsample_stages=2,
        )
        self.spatial_stem = SpatialStem2D(in_channels, self.channels)
        self.p8_proj = nn.Conv2d(self.channels, self.channels, 1, bias=False)
        self.p4_proj = nn.Conv2d(self.channels, self.channels, 1, bias=False)

    def forward_pixels(self, images, z, return_intermediates=False):
        if images.ndim != 4:
            raise ValueError("Arm E spatial stem requires BCHW images")
        batch_size, patch_count, _ = z.shape
        expected_patches = int(torch.tensor(self.grid_size).prod().item())
        if patch_count != expected_patches:
            raise ValueError("encoder patch count does not match pixel grid")
        if images.shape[0] != batch_size:
            raise ValueError("images and encoder features must share a batch size")

        p4_raw, p8_raw = self.spatial_stem(images)
        p16 = self.pixel_proj(self.pixel_norm(z))
        p16 = p16.transpose(1, 2).reshape(
            batch_size, self.channels, *self.grid_size
        )
        p8_fused = F.interpolate(
            p16, size=p8_raw.shape[-2:], mode="bilinear", align_corners=False
        )
        p8_fused = self.blocks[0](p8_fused + self.p8_proj(p8_raw))
        p4_fused = F.interpolate(
            p8_fused,
            size=p4_raw.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        p4_fused = self.blocks[1](p4_fused + self.p4_proj(p4_raw))
        if return_intermediates:
            return p4_fused, {
                "p4_raw": p4_raw,
                "p8_raw": p8_raw,
                "p16": p16,
                "p8_fused": p8_fused,
                "p4_fused": p4_fused,
            }
        return p4_fused

    def forward_query(self, tokens, slot_identity):
        pooled_tokens = self.query_norm(tokens).mean(dim=1)
        return self.query_proj(pooled_tokens) + slot_identity.unsqueeze(0)

    def forward_mask_from_query(self, pixel_features, query, output_size):
        if pixel_features.shape[0] != query.shape[0]:
            raise ValueError("pixel features and queries must share a batch size")
        if query.ndim not in (2, 3):
            raise ValueError("queries must have shape [B,D] or [B,K,D]")
        pixel_features = F.normalize(pixel_features.float(), dim=1)
        query = F.normalize(query.float(), dim=-1)
        if query.ndim == 2:
            logits = torch.einsum("bdhw,bd->bhw", pixel_features, query)
            logits = logits.unsqueeze(1)
        else:
            logits = torch.einsum("bdhw,bkd->bkhw", pixel_features, query)
        logits = logits.mul(math.sqrt(self.channels))
        output_size = tuple(int(value) for value in output_size)
        return F.interpolate(
            logits, size=output_size, mode="bilinear", align_corners=False
        )

    def forward_mask(self, pixel_features, tokens, slot_identity, output_size):
        if pixel_features.shape[0] != tokens.shape[0]:
            raise ValueError("pixel features and tokens must share a batch size")
        query = self.forward_query(tokens, slot_identity)
        return self.forward_mask_from_query(pixel_features, query, output_size)


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
            if dataset_type != "2D":
                raise ValueError("multiscale_conv head currently supports 2D only")
            self.head = MultiScaleBinaryHead2D(
                decoder_dim, grid_size, channels=head_channels
            )
        elif self.head_type in (
            "query_dot", "multi_query_dot", "arm_e_multiscale_query"
        ):
            if dataset_type != "2D":
                raise ValueError("query heads currently support 2D only")
            self.head = SlotQueryEmbedding(head_channels)
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
        if self.head_type in (
            "query_dot", "multi_query_dot", "arm_e_multiscale_query"
        ):
            raise RuntimeError(
                "query heads require tokens and shared pixel features"
            )
        raw_logits = self.head(canvas, output_size)
        calibrated_logits = (
            self.calibration_scale * raw_logits + self.calibration_bias
        )
        return raw_logits, calibrated_logits

    def forward_query_head(self, tokens, pixel_features, decoder, output_size):
        if self.head_type not in (
            "query_dot", "multi_query_dot", "arm_e_multiscale_query"
        ):
            raise RuntimeError("forward_query_head requires a query slot")
        raw_logits = decoder.forward_mask(
            pixel_features,
            tokens,
            self.head.embedding,
            output_size,
        )
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
