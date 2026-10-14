"""Encoder-only 2D adaptations for the OrganSlot E backbone ablation.

References: nnU-Net ResEnc BasicBlockD and MedNeXt-v1 B. These are not full
segmentation models. See docs/CONFIGURATION.md for sources and deviations.
No pretrained weight download or new package dependency is required.
"""

import torch.nn as nn


CNN_BACKBONES = ("nnunet_resenc_s16", "mednext_b_s16")


def _instance_norm(channels):
    return nn.InstanceNorm2d(channels, eps=1e-5, affine=True)


class ResEncBlock2D(nn.Module):
    """BasicBlockD topology: pooled skip and two 3x3 convolutions."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=True),
            _instance_norm(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True),
            _instance_norm(out_channels),
        )
        skip = []
        if stride != 1:
            skip.append(nn.AvgPool2d(stride, stride))
        if in_channels != out_channels:
            skip.extend((nn.Conv2d(in_channels, out_channels, 1, bias=False),
                         _instance_norm(out_channels)))
        self.skip = nn.Sequential(*skip)
        self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        return self.activation(self.main(x) + self.skip(x))


class MedNeXtBlock2D(nn.Module):
    """MedNeXt-v1 block, including residual downsampling."""

    def __init__(self, in_channels, out_channels, expansion, stride=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels),
            nn.GroupNorm(in_channels, in_channels),
            nn.Conv2d(in_channels, expansion * in_channels, 1),
            nn.GELU(),
            nn.Conv2d(expansion * in_channels, out_channels, 1),
        )
        self.skip = (nn.Conv2d(in_channels, out_channels, 1, stride)
                     if stride != 1 or in_channels != out_channels
                     else nn.Identity())

    def forward(self, x):
        return self.main(x) + self.skip(x)


class CNNTokenEncoder2D(nn.Module):
    """S16 tokens with optional S4/S8 skips; metadata retained for reconstruction."""

    channels = (32, 64, 128, 256, 512)

    def __init__(self, kind, img_size, in_chans, embed_dim):
        super().__init__()
        if kind not in CNN_BACKBONES:
            raise ValueError("unknown CNN backbone: " + str(kind))
        if img_size < 32 or img_size % 16:
            raise ValueError("CNN input must be >=32 and divisible by 16")
        self.kind = kind
        self.img_size = (img_size, img_size)
        self.patch_size = (16, 16)
        self.grid_size = (img_size // 16, img_size // 16)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        stages = []
        if kind == "nnunet_resenc_s16":
            self.block_counts = (1, 3, 4, 6, 6)
            self.stem = nn.Sequential(
                nn.Conv2d(in_chans, 32, 3, padding=1, bias=True),
                _instance_norm(32), nn.LeakyReLU(0.01, inplace=True),
            )
            previous = 32
            for i, (channels, count) in enumerate(zip(self.channels, self.block_counts)):
                blocks = [ResEncBlock2D(previous, channels, 1 if i == 0 else 2)]
                blocks.extend(ResEncBlock2D(channels, channels) for _ in range(count - 1))
                stages.append(nn.Sequential(*blocks))
                previous = channels
        else:
            self.block_counts = (2, 2, 2, 2, 2)
            self.expansion = (2, 3, 4, 4, 4)
            self.stem = nn.Conv2d(in_chans, 32, 1)
            previous = 32
            for i, channels in enumerate(self.channels):
                blocks = []
                if i:
                    blocks.append(MedNeXtBlock2D(previous, channels, self.expansion[i], 2))
                blocks.extend(MedNeXtBlock2D(channels, channels, self.expansion[i])
                              for _ in range(self.block_counts[i]))
                stages.append(nn.Sequential(*blocks))
                previous = channels
        self.stages = nn.ModuleList(stages)
        self.proj = nn.Conv2d(self.channels[-1], embed_dim, 1)
        if kind == "nnunet_resenc_s16":
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, a=0.01)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            for module in self.modules():
                if isinstance(module, ResEncBlock2D):
                    nn.init.zeros_(module.main[-1].weight)

    def forward(self, images, return_features=False):
        if images.ndim != 4 or tuple(images.shape[-2:]) != self.img_size:
            raise ValueError("CNN input must match the configured 2D image grid")
        x = self.stem(images)
        features = {}
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if return_features and i in (2, 3):
                features['s4' if i == 2 else 's8'] = x
        tokens = self.proj(x).flatten(2).transpose(1, 2)
        return (tokens, features) if return_features else tokens
