"""New-stage spatial residuals; old maps are never modified in place."""
import torch
from torch import nn
from torch.nn import functional as F


class ScaleResidual(nn.Module):
    def __init__(self, channels, lateral=False):
        super().__init__()
        self.down = nn.Conv2d(channels, 32, 1, bias=False)
        self.lateral = nn.Conv2d(channels, 32, 1, bias=False) if lateral else None
        self.refine = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
            nn.GroupNorm(8, 32), nn.GELU())
        self.up = nn.Conv2d(32, channels, 1, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, pixels, z, lateral=None):
        hidden = self.down(pixels) + F.interpolate(
            z, size=pixels.shape[-2:], mode="bilinear", align_corners=False)
        if self.lateral is not None:
            hidden = hidden + self.lateral(lateral)
        return pixels + self.up(self.refine(hidden))


class NewStageSpatialAdapter(nn.Module):
    def __init__(self, embed_dim, channels):
        super().__init__()
        self.z_down = nn.Conv2d(embed_dim, 32, 1, bias=False)
        self.scales = nn.ModuleList([
            ScaleResidual(channels), ScaleResidual(channels, True),
            ScaleResidual(channels, True)])

    def forward(self, maps, z, features):
        # FP32 avoids old-cluster BF16 interpolation incompatibilities.
        with torch.autocast(device_type=z.device.type, enabled=False):
            z = z.float().transpose(1, 2).reshape(z.shape[0], z.shape[-1], *maps[0].shape[-2:])
            z = self.z_down(z)
            lateral = [None, features["p8_raw"].float(), features["p4_raw"].float()]
            return [block(p.float(), z, s).to(p.dtype)
                    for block, p, s in zip(self.scales, maps, lateral)]
