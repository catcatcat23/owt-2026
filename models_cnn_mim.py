"""Dense CNN masked-image pretraining; only stem/stages transfer downstream."""
import hashlib

import torch
from torch import nn

from CNNBackbone import CNNTokenEncoder2D


class CNNMaskedImageModel(nn.Module):
    def __init__(self, backbone, img_size=224):
        super().__init__()
        self.encoder = CNNTokenEncoder2D(backbone, img_size, 3, 512)
        self.encoder.proj = nn.Identity()
        self.decoder = nn.Linear(512, 16 * 16 * 3)
        self.img_size = img_size

    def forward(self, images, mask_ratio=0.75):
        if not 0 < mask_ratio < 1:
            raise ValueError("mask_ratio must be in (0,1)")
        b, _, h, w = images.shape
        n = (h // 16) * (w // 16)
        count = min(n - 1, max(1, round(mask_ratio * n)))
        order = torch.rand(b, n, device=images.device).argsort(dim=1)
        mask = images.new_zeros(b, n)
        mask.scatter_(1, order[:, :count], 1)
        pixels = mask.reshape(b, 1, h // 16, w // 16).repeat_interleave(16, 2).repeat_interleave(16, 3)
        # Mask BEFORE any convolution. Target pixels never enter the encoder.
        z = self.encoder(images * (1 - pixels))
        prediction = self.decoder(z).reshape(b, h // 16, w // 16, 16, 16, 3)
        prediction = prediction.permute(0, 5, 1, 3, 2, 4).reshape(b, 3, h, w)
        loss = ((prediction.float() - images.float()).square() * pixels.float()).sum()
        loss = loss / (pixels.sum().float() * 3).clamp_min(1)
        return loss, prediction, mask


def load_cnn_encoder_checkpoint(model, path):
    """Strictly transfer all convolutional stem/stage weights, never the head."""
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("backbone") != model.backbone or model.backbone == "vit":
        raise ValueError("CNN checkpoint backbone does not match downstream model")
    if not checkpoint.get("pretraining_complete", False):
        raise ValueError("CNN transfer requires a completed pretraining checkpoint")
    state = checkpoint["model"]
    source = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
    expected = {k: v for k, v in model.patch_embed.state_dict().items()
                if k.startswith(("stem.", "stages."))}
    if set(source) != set(expected):
        raise ValueError("CNN encoder keys do not match exactly")
    if any(source[k].shape != expected[k].shape or not torch.isfinite(source[k]).all()
           for k in expected):
        raise ValueError("CNN encoder has invalid shapes or non-finite weights")
    merged = model.patch_embed.state_dict()
    merged.update(source)
    model.patch_embed.load_state_dict(merged, strict=True)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"backbone": model.backbone, "checkpoint": str(path),
            "sha256": digest.hexdigest(), "loaded_keys": sorted(source),
            "optimizer_updates": checkpoint["optimizer_updates"],
            "scope": "stem_and_stages_only", "projection_loaded": False,
            "decoder_loaded": False}
