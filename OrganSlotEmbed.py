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


def _interpolate_bf16_safe(tensor, **kwargs):
    """Use FP32 for BF16 bilinear resize; preserve dtype and gradients."""
    if tensor.dtype == torch.bfloat16 and kwargs.get("mode") == "bilinear":
        with torch.cuda.amp.autocast(enabled=False):
            return F.interpolate(tensor.float(), **kwargs).to(dtype=tensor.dtype)
    return F.interpolate(tensor, **kwargs)


def _trilinear_interpolate_fp32(tensor, **kwargs):
    """Run 3D trilinear interpolation in FP32 for AMP compatibility.

    The PyTorch build on XEC does not implement the CUDA BF16
    ``upsample_trilinear3d`` kernel. Keeping only this parameter-free resize
    in FP32 preserves BF16 convolutions and a fully differentiable path while
    avoiding a runtime failure before the first resumed optimizer step.
    """
    original_dtype = tensor.dtype
    if original_dtype in (torch.float16, torch.bfloat16):
        resized = _interpolate_bf16_safe(
            tensor.float(), mode="trilinear", align_corners=False, **kwargs
        )
        return resized.to(dtype=original_dtype)
    return _interpolate_bf16_safe(
        tensor, mode="trilinear", align_corners=False, **kwargs
    )


class FP32CompatibleDepthwiseConv3d(nn.Conv3d):
    """Run low-precision depthwise Conv3d in FP32 when kernels are missing.

    Some supported cluster PyTorch/CUDA builds do not implement BF16
    depthwise Conv3d. The cast remains inside the autograd graph, so both
    input and convolution weights still receive gradients while the rest of
    the model continues to use AMP.
    """

    def forward(self, tensor):
        original_dtype = tensor.dtype
        with torch.cuda.amp.autocast(enabled=False):
            bias = None if self.bias is None else self.bias.float()
            output = self._conv_forward(tensor.float(), self.weight.float(), bias)
        return output.to(dtype=original_dtype)


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
        if len(output_size) == 3:
            return _trilinear_interpolate_fp32(logits, size=output_size)
        return _interpolate_bf16_safe(
            logits, size=output_size, mode="bilinear", align_corners=False
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
            features = _interpolate_bf16_safe(
                features, scale_factor=2.0, mode="bilinear", align_corners=False
            )
            features = block(features)
        logits = self.proj(features)
        if tuple(logits.shape[-2:]) != output_size:
            logits = _interpolate_bf16_safe(
                logits, size=output_size, mode="bilinear", align_corners=False
            )
        return logits


class MultiScaleBinaryHead3D(nn.Module):
    """Anisotropic 3D counterpart of Arm B's local convolutional head."""

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
                FP32CompatibleDepthwiseConv3d(
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
            features = _trilinear_interpolate_fp32(
                features,
                scale_factor=(1.0, 2.0, 2.0),
            )
            features = block(features)
        logits = self.proj(features)
        if tuple(logits.shape[-3:]) != output_size:
            logits = _trilinear_interpolate_fp32(
                logits,
                size=output_size,
            )
        return logits


class SlotQueryEmbedding(nn.Module):
    """Per-slot identity used by the shared query-mask decoder."""

    def __init__(self, dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(int(dim)))
        nn.init.normal_(self.embedding, std=0.02)


class SharedPixelQueryDecoder(nn.Module):
    """Shared 2D/3D spatial decoder and token-to-mask query projection."""

    def __init__(
        self,
        embed_dim,
        grid_size,
        channels=128,
        upsample_stages=2,
        multi_query=False,
        pixel_pe="none",
    ):
        super().__init__()
        self.grid_size = tuple(int(value) for value in grid_size)
        if len(self.grid_size) not in (2, 3):
            raise ValueError("query decoder grid must be 2D or 3D")
        self.spatial_dims = len(self.grid_size)
        self.channels = int(channels)
        if self.channels <= 0:
            raise ValueError("query-mask channels must be positive")
        self.upsample_stages = int(upsample_stages)
        if self.upsample_stages < 1:
            raise ValueError("upsample_stages must be positive")
        self.multi_query = bool(multi_query)
        self.pixel_pe = pixel_pe
        if pixel_pe not in ("none", "spatial", "temporal", "spatiotemporal"):
            raise ValueError("unknown pixel PE mode")
        if self.spatial_dims == 2 and pixel_pe in ("temporal", "spatiotemporal"):
            raise ValueError("temporal pixel PE requires 3D")
        if pixel_pe in ("spatial", "spatiotemporal"):
            from util.pos_embed import get_2d_sincos_pos_embed
            h, w = self.grid_size[-2:]
            if h != w or self.channels % 4:
                raise ValueError("spatial PE requires square grid and channels divisible by 4")
            pe = torch.from_numpy(get_2d_sincos_pos_embed(self.channels, h)).float()
            self.spatial_pe = nn.Parameter(pe.T.reshape(1, self.channels, h, w))
        if pixel_pe in ("temporal", "spatiotemporal"):
            # Local slab coordinates, matching OWT's temporal PE convention.
            # fork_rng preserves initialization of all shared baseline parameters.
            self.temporal_pe = nn.Parameter(torch.empty(1, self.channels, self.grid_size[0], 1, 1))
            with torch.random.fork_rng(devices=[]):
                nn.init.normal_(self.temporal_pe, std=0.02)

        self.pixel_norm = nn.LayerNorm(embed_dim)
        self.pixel_proj = nn.Linear(embed_dim, self.channels)
        self.query_norm = nn.LayerNorm(embed_dim)
        self.query_proj = nn.Linear(embed_dim, self.channels)
        conv_layer = nn.Conv2d if self.spatial_dims == 2 else nn.Conv3d
        depthwise_layer = (
            nn.Conv2d
            if self.spatial_dims == 2
            else FP32CompatibleDepthwiseConv3d
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(
                depthwise_layer(
                    self.channels,
                    self.channels,
                    kernel_size=3,
                    padding=1,
                    groups=self.channels,
                    bias=False,
                ),
                conv_layer(self.channels, self.channels, kernel_size=1, bias=False),
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
        if hasattr(self, "spatial_pe"):
            pe = self.spatial_pe if self.spatial_dims == 2 else self.spatial_pe.unsqueeze(2)
            features = features + pe.to(dtype=features.dtype)
        if hasattr(self, "temporal_pe"):
            features = features + self.temporal_pe.to(dtype=features.dtype)
        for block in self.blocks:
            if self.spatial_dims == 2:
                features = _interpolate_bf16_safe(
                    features, scale_factor=2.0, mode="bilinear",
                    align_corners=False,
                )
            else:
                features = _trilinear_interpolate_fp32(
                    features, scale_factor=(1.0, 2.0, 2.0)
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
            if self.spatial_dims == 2:
                part_logits = torch.einsum(
                    "bdhw,bkd->bkhw", pixel_features, queries
                ).mul(math.sqrt(self.channels))
            else:
                part_logits = torch.einsum(
                    "bdthw,bkd->bkthw", pixel_features, queries
                ).mul(math.sqrt(self.channels))
            # Smooth maximum without token-count bias. If all token queries
            # are identical, this exactly matches the single-query score.
            logits = torch.logsumexp(part_logits, dim=1)
            logits = logits - math.log(tokens.shape[1])
        else:
            pooled_tokens = normalized_tokens.mean(dim=1)
            query = self.query_proj(pooled_tokens) + slot_identity.unsqueeze(0)
            query = F.normalize(query, dim=-1)
            if self.spatial_dims == 2:
                logits = torch.einsum(
                    "bdhw,bd->bhw", pixel_features, query
                ).mul(math.sqrt(self.channels))
            else:
                logits = torch.einsum(
                    "bdthw,bd->bthw", pixel_features, query
                ).mul(math.sqrt(self.channels))
        logits = logits.unsqueeze(1)
        output_size = tuple(int(value) for value in output_size)
        if self.spatial_dims == 3:
            return _trilinear_interpolate_fp32(logits, size=output_size)
        return _interpolate_bf16_safe(
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


class QueryCrossAttention(nn.Module):
    """One pre-norm query-to-image block; FP32 attention, no extra PE."""

    def __init__(self, channels, heads=4):
        super().__init__()
        if channels % heads:
            raise ValueError("cross-attention channels must divide heads")
        self.heads = heads
        self.q_norm = nn.LayerNorm(channels)
        self.memory_norm = nn.LayerNorm(channels)
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2), nn.GELU(),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, query, pixels, mode="normal"):
        if mode not in ("normal", "uniform", "bypass"):
            raise ValueError("unknown attention intervention")
        # This is pooled fused P4, NOT the separate P8 lateral feature.
        with torch.cuda.amp.autocast(enabled=False):
            memory_map = F.avg_pool2d(pixels.float(), 2)
            memory = self.memory_norm(memory_map.flatten(2).transpose(1, 2))
            batch, locations, channels = memory.shape
            head_dim = channels // self.heads
            q = self.q_proj(self.q_norm(query.float())).reshape(
                batch, self.heads, 1, head_dim
            )
            k = self.k_proj(memory).reshape(
                batch, locations, self.heads, head_dim
            ).transpose(1, 2)
            v = self.v_proj(memory).reshape(
                batch, locations, self.heads, head_dim
            ).transpose(1, 2)
            weights = torch.softmax(
                (q @ k.transpose(-1, -2)) / math.sqrt(head_dim), dim=-1
            )
            if mode == "uniform":
                weights = torch.ones_like(weights) / locations
            context = (weights @ v).reshape(batch, channels)
            updated = query.float() + self.out_proj(context)
            updated = updated + self.ffn(self.ffn_norm(updated))
            if mode == "bypass":
                updated = query.float()
            attention = weights.reshape(batch, self.heads, *memory_map.shape[-2:])
        return updated, attention


class MultiScalePixelQueryDecoder2D(SharedPixelQueryDecoder):
    """Arm E pixel path: ViT P16 semantics fused with image P8/P4 detail."""

    def __init__(self, in_channels, embed_dim, grid_size, channels=128,
                 query_refinement="none"):
        super().__init__(
            embed_dim,
            grid_size,
            channels=channels,
            upsample_stages=2,
        )
        self.spatial_stem = SpatialStem2D(in_channels, self.channels)
        self.p8_proj = nn.Conv2d(self.channels, self.channels, 1, bias=False)
        self.p4_proj = nn.Conv2d(self.channels, self.channels, 1, bias=False)
        if query_refinement not in ("none", "cross_attn"):
            raise ValueError("unknown query_refinement")
        self.query_refinement = query_refinement
        if query_refinement == "cross_attn":
            # Preserve RNG for subsequently initialized organ slots.
            with torch.random.fork_rng(devices=[]):
                self.query_cross_attention = QueryCrossAttention(self.channels)

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
        p8_fused = _interpolate_bf16_safe(
            p16, size=p8_raw.shape[-2:], mode="bilinear", align_corners=False
        )
        p8_fused = self.blocks[0](p8_fused + self.p8_proj(p8_raw))
        p4_fused = _interpolate_bf16_safe(
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
        return _interpolate_bf16_safe(
            logits, size=output_size, mode="bilinear", align_corners=False
        )

    def forward_mask(self, pixel_features, tokens, slot_identity, output_size):
        if pixel_features.shape[0] != tokens.shape[0]:
            raise ValueError("pixel features and tokens must share a batch size")
        query = self.forward_query(tokens, slot_identity)
        if self.query_refinement == "cross_attn":
            query, _ = self.query_cross_attention(query, pixel_features)
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
            head_class = (
                MultiScaleBinaryHead2D
                if dataset_type == "2D"
                else MultiScaleBinaryHead3D
            )
            self.head = head_class(decoder_dim, grid_size, channels=head_channels)
        elif self.head_type in (
            "query_dot", "multi_query_dot", "arm_e_multiscale_query"
        ):
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
