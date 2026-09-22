"""Organ-conditioned attention, with one mask per organ."""
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

    def forward(self, x, memory, pe, attention_bias=None):
        memory = self.norms[1](memory)
        x = x + self.cross(self.norms[0](x), memory + pe, memory,
                           attn_mask=attention_bias, need_weights=False)[0]
        normalized = self.norms[2](x)
        x = x + self.self_attn(normalized, normalized, normalized, need_weights=False)[0]
        return x + self.ffn(self.norms[3](x))


class SAMStyleMaskTail(nn.Module):
    """One two-way block plus final token read after F multiscale refinement.

    OrganSlot adaptation, not original/pretrained SAM: shared mask token,
    no external prompts, IoU token, multimask output or extra upsampling.
    """

    def __init__(self, channels):
        super().__init__()
        self.mask_token = nn.Parameter(torch.empty(1, 1, channels))
        nn.init.normal_(self.mask_token, std=0.02)
        self.self_attn = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.token_to_pixel = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.pixel_to_token = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.final_token_to_pixel = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.norms = nn.ModuleList([nn.LayerNorm(channels) for _ in range(5)])
        self.token_ffn = ffn(channels)
        self.mask_mlp = nn.Sequential(
            nn.Linear(channels, channels), nn.GELU(),
            nn.Linear(channels, channels), nn.GELU(),
            nn.Linear(channels, channels))

    def forward(self, tokens, pixels, pixel_pe):
        tokens = torch.cat((self.mask_token.expand(tokens.shape[0], -1, -1), tokens), dim=1)
        # Fixed sparse input reference, NOT geometric coordinates for organ tokens.
        token_reference = tokens
        q = tokens + token_reference
        tokens = self.norms[0](tokens + self.self_attn(q, q, tokens, need_weights=False)[0])
        tokens = self.norms[1](tokens + self.token_to_pixel(
            tokens + token_reference, pixels + pixel_pe, pixels, need_weights=False)[0])
        tokens = self.norms[2](tokens + self.token_ffn(tokens))
        pixels = self.norms[3](pixels + self.pixel_to_token(
            pixels + pixel_pe, tokens + token_reference, tokens, need_weights=False)[0])
        tokens = self.norms[4](tokens + self.final_token_to_pixel(
            tokens + token_reference, pixels + pixel_pe, pixels, need_weights=False)[0])
        weights = self.mask_mlp(tokens[:, 0])
        # Ordinary dynamic dot product, not the historical cosine readout.
        return torch.bmm(pixels, weights.unsqueeze(-1))


class ArmFDecoder(nn.Module):
    def __init__(self, in_channels, embed_dim, grid_size, channels=128, token_count=20, readout="attention"):
        super().__init__()
        self.grid_size = tuple(grid_size)
        self.channels = channels
        self.soft_mask = readout == "reverse_dot_p2_softmask"
        if self.soft_mask:
            readout = "reverse_dot_p2"
        # Fixed ablation recipe; no extra learned parameters or auxiliary loss.
        self.soft_mask_alpha = 1.0
        self.soft_mask_epsilon = 0.2
        self.readout = readout
        if readout == "reverse_dot_p2" and len(self.grid_size) != 2:
            raise ValueError("reverse_dot_p2 currently supports only 2D")
        # Reuse the exact Arm E spatial stem/fusion weights and architecture.
        # In 3D it runs slice-wise; no added temporal convolution confound.
        self.pixels = MultiScalePixelQueryDecoder2D(in_channels, embed_dim, self.grid_size[-2:], channels)
        del self.pixels.query_norm
        del self.pixels.query_proj
        self.input_proj = nn.Linear(embed_dim, channels)
        self.blocks = nn.ModuleList([TokenBlock(channels) for _ in range(3)])
        if readout in ("attention", "reverse_dot", "reverse_dot_p2"):
            self.reverse = nn.MultiheadAttention(channels, 4, batch_first=True)
            self.pixel_norm = nn.LayerNorm(channels)
            self.token_norm = nn.LayerNorm(channels)
            self.ffn_norm = nn.LayerNorm(channels)
            self.ffn = ffn(channels)
            if readout == "attention":
                self.classifier = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, 1))
        elif readout == "linear":
            self.mask_mlp = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels))
            self.token_fusion = nn.Linear(token_count, 1)
        elif readout == "sam_tail":
            with torch.random.fork_rng(devices=[]):
                self.sam_tail = SAMStyleMaskTail(channels)
        elif readout != "query_dot":
            raise ValueError("unknown Arm F readout")
        if readout == "reverse_dot_p2":
            # Do not shift initialization of organ slots constructed afterwards.
            with torch.random.fork_rng(devices=[]):
                self.p2_proj = nn.Conv2d(max(16, channels // 2), channels, 1, bias=False)
                self.p2_fusion = nn.Sequential(
                    nn.Conv2d(2 * channels, channels, 3, padding=1, bias=False),
                    nn.GroupNorm(1, channels), nn.GELU(),
                    nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                    nn.GroupNorm(1, channels), nn.GELU(),
                )

    def forward_pixels(self, images, z):
        if images.ndim == 5:
            batch, cin, time, height, width = images.shape
            # Encoder temporal patch stride can differ from input slice count.
            if time != self.grid_size[0]:
                images = F.adaptive_avg_pool3d(images, (self.grid_size[0], height, width))
                time = self.grid_size[0]
            images = images.permute(0, 2, 1, 3, 4).reshape(batch * time, cin, height, width)
            z = z.reshape(batch * time, -1, z.shape[-1])
        _, features = self.pixels.forward_pixels(
            images, z, return_intermediates=True,
            return_p2=getattr(self, "readout", None) == "reverse_dot_p2")
        maps = [features[key] for key in ("p16", "p8_fused", "p4_fused")]
        if getattr(self, "readout", None) == "reverse_dot_p2":
            maps.append(features["p2_raw"])
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
                bias = None
                if self.soft_mask and self.soft_mask_alpha != 0:
                    bias = self.soft_attention_bias(memory, x, block.cross.num_heads)
                x = block(x, memory, pe, attention_bias=bias)
            mask_shape = maps[2].shape[2:]
            p = maps[2].float().flatten(2).transpose(1, 2)
            if self.readout in ("attention", "reverse_dot", "reverse_dot_p2"):
                kv = self.token_norm(x)
                z = p + self.reverse(self.pixel_norm(p) + pe, kv, kv, need_weights=False)[0]
                z = z + self.ffn(self.ffn_norm(z))
                if self.readout == "reverse_dot_p2":
                    s2 = maps[3].float()
                    z = z.transpose(1, 2).reshape(tokens.shape[0], self.channels, *mask_shape)
                    z = _interpolate_bf16_safe(
                        z, size=s2.shape[2:], mode="bilinear", align_corners=False)
                    z = self.p2_fusion(torch.cat((z, self.p2_proj(s2)), dim=1))
                    mask_shape = z.shape[2:]
                    z = z.flatten(2).transpose(1, 2)
                if self.readout == "attention":
                    logits = self.classifier(z)
                else:
                    logits = self.dot_readout(z, x)
            elif self.readout == "sam_tail":
                logits = self.sam_tail(x, p, pe)
            elif self.readout == "query_dot":
                logits = self.dot_readout(p, x)
            else:
                logits = self.token_fusion(p @ self.mask_mlp(x).transpose(1, 2) / math.sqrt(self.channels))
            logits = logits.transpose(1, 2).reshape(tokens.shape[0], 1, *mask_shape)
            return _interpolate_bf16_safe(logits, size=tuple(output_size), mode="bilinear" if logits.ndim == 4 else "trilinear", align_corners=False)

    def dot_readout(self, pixels, tokens):
        query = F.normalize(tokens.float().mean(dim=1), dim=-1)
        pixels = F.normalize(pixels.float(), dim=-1)
        return (pixels * query[:, None, :]).sum(dim=-1, keepdim=True) * math.sqrt(self.channels)

    @torch.no_grad()
    def soft_attention_bias(self, pixels, tokens, num_heads):
        """Shared organ prior, not 20 unsupervised token-specific masks.

        Recomputed from the current tokens at each scale. Detaching deliberately
        tests routing only. A positive floor avoids hard exclusion of missed GT.
        MHA expects [batch * heads, token_count, spatial_positions].
        """
        probability = self.dot_readout(pixels, tokens).squeeze(-1).sigmoid()
        gate = self.soft_mask_epsilon + (1 - self.soft_mask_epsilon) * probability
        bias = self.soft_mask_alpha * gate.log()
        return bias[:, None, None, :].expand(
            -1, num_heads, tokens.shape[1], -1).reshape(
                pixels.shape[0] * num_heads, tokens.shape[1], pixels.shape[1])


class ArmEStyleDecoder3D(ArmFDecoder):
    """E mean-query readout on F's slice-wise multiscale spatial pathway."""

    def __init__(self, in_channels, embed_dim, grid_size, channels=128):
        nn.Module.__init__(self)
        if len(grid_size) != 3:
            raise ValueError('arm_e_multiscale_query_3d requires a 3D grid')
        self.grid_size = tuple(grid_size)
        self.channels = channels
        self.pixels = MultiScalePixelQueryDecoder2D(
            in_channels, embed_dim, self.grid_size[-2:], channels)

    def forward_mask(self, maps, tokens, slot_identity, output_size):
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            query = self.pixels.forward_query(tokens.float(), slot_identity.float())
            pixels = F.normalize(maps[-1].float(), dim=1)
            query = F.normalize(query, dim=-1)
            logits = torch.einsum('bdthw,bd->bthw', pixels, query).unsqueeze(1)
            logits = logits * math.sqrt(self.channels)
            return _interpolate_bf16_safe(logits, size=tuple(output_size),
                                         mode='trilinear', align_corners=False)
