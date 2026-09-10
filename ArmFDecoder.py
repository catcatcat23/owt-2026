"""Organ-conditioned two-stage attention, with one mask per organ."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from OrganSlotEmbed import MultiScalePixelQueryDecoder2D, _interpolate_bf16_safe


def position_encoding(shape, channels, device):
    """Fixed axial sin/cos PE: y/x in 2D, t/y/x in 3D (no token PE)."""
    pairs = channels // (2 * len(shape))
    if pairs < 1:
        raise ValueError("channels too small for axial position encoding")
    frequency = torch.exp(-math.log(10000.) * torch.arange(pairs, device=device).float() / pairs)
    coords = torch.meshgrid(*[torch.linspace(0, 2 * math.pi, n, device=device) for n in shape], indexing="ij")
    axes = []
    for coord in coords:
        phase = coord.reshape(-1, 1) * frequency
        axes.extend([phase.sin(), phase.cos()])
    return F.pad(torch.cat(axes, dim=-1), (0, channels - 2 * pairs * len(shape))).unsqueeze(0)


def ffn(channels):
    return nn.Sequential(nn.Linear(channels, 4 * channels), nn.GELU(), nn.Linear(4 * channels, channels))


class TokenBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.cross = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.self_attn = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.norms = nn.ModuleList([nn.LayerNorm(channels) for _ in range(4)])
        self.ffn = ffn(channels)

    def forward(self, x, memory, pe):
        memory = self.norms[1](memory)
        x = x + self.cross(self.norms[0](x), memory + pe, memory, need_weights=False)[0]
        normalized = self.norms[2](x)
        x = x + self.self_attn(normalized, normalized, normalized, need_weights=False)[0]
        return x + self.ffn(self.norms[3](x))


class ArmFDecoder(nn.Module):
    def __init__(self, in_channels, embed_dim, grid_size, channels=128, token_count=20, readout="attention"):
        super().__init__()
        self.grid_size = tuple(grid_size)
        self.channels = channels
        self.readout = readout
        # Reuse the exact Arm E spatial stem/fusion weights and architecture.
        # In 3D it runs slice-wise; no added temporal convolution confound.
        self.pixels = MultiScalePixelQueryDecoder2D(in_channels, embed_dim, self.grid_size[-2:], channels)
        del self.pixels.query_norm
        del self.pixels.query_proj
        self.input_proj = nn.Linear(embed_dim, channels)
        self.blocks = nn.ModuleList([TokenBlock(channels) for _ in range(3)])
        if readout == "attention":
            self.reverse = nn.MultiheadAttention(channels, 4, batch_first=True)
            self.pixel_norm = nn.LayerNorm(channels)
            self.token_norm = nn.LayerNorm(channels)
            self.ffn_norm = nn.LayerNorm(channels)
            self.ffn = ffn(channels)
            self.classifier = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, 1))
        elif readout == "linear":
            self.mask_mlp = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels))
            self.token_fusion = nn.Linear(token_count, 1)
        else:
            raise ValueError("unknown Arm F readout")

    def forward_pixels(self, images, z):
        if images.ndim == 5:
            batch, cin, time, height, width = images.shape
            # Encoder temporal patch stride can differ from input slice count.
            if time != self.grid_size[0]:
                images = F.adaptive_avg_pool3d(images, (self.grid_size[0], height, width))
                time = self.grid_size[0]
            images = images.permute(0, 2, 1, 3, 4).reshape(batch * time, cin, height, width)
            z = z.reshape(batch * time, -1, z.shape[-1])
        _, features = self.pixels.forward_pixels(images, z, return_intermediates=True)
        maps = [features[key] for key in ("p16", "p8_fused", "p4_fused")]
        if len(self.grid_size) == 3:
            maps = [p.reshape(batch, time, self.channels, *p.shape[-2:]).permute(0, 2, 1, 3, 4) for p in maps]
        return maps

    def forward_mask(self, maps, tokens, slot_identity, output_size):
        # Local FP32 attention/FFN for the older cluster PyTorch builds.
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            x = self.input_proj(tokens.float()) + slot_identity.float()[None, None, :]
            for block, feature in zip(self.blocks, maps):
                memory = feature.float().flatten(2).transpose(1, 2)
                pe = position_encoding(feature.shape[2:], self.channels, feature.device)
                x = block(x, memory, pe)
            p = maps[-1].float().flatten(2).transpose(1, 2)
            if self.readout == "attention":
                kv = self.token_norm(x)
                z = p + self.reverse(self.pixel_norm(p) + pe, kv, kv, need_weights=False)[0]
                z = z + self.ffn(self.ffn_norm(z))
                logits = self.classifier(z)
            else:
                logits = self.token_fusion(p @ self.mask_mlp(x).transpose(1, 2) / math.sqrt(self.channels))
            logits = logits.transpose(1, 2).reshape(tokens.shape[0], 1, *maps[-1].shape[2:])
            return _interpolate_bf16_safe(logits, size=tuple(output_size), mode="bilinear" if logits.ndim == 4 else "trilinear", align_corners=False)
