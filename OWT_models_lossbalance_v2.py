"""OWT model variant with area- and sample-frequency-balanced ROI loss."""

from functools import partial

import torch
import torch.nn as nn

from OWT_models import MaskedAutoencoderViT
from util.frequency_balanced_loss import frequency_balanced_roi_l2


class LossBalancedMaskedAutoencoderViT(MaskedAutoencoderViT):
    """Preserve the OWT architecture and replace only loss accounting."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        class_weights = torch.as_tensor(
            self.model_args.roi_class_weights, dtype=torch.float32
        )
        if class_weights.numel() != self.model_args.num_classes_with_bg:
            raise ValueError(
                "roi_class_weights must contain background plus every foreground class"
            )
        # Derived dataset statistics are not trainable state. Keeping this
        # buffer non-persistent preserves compatibility with original OWT
        # checkpoints while still moving it with the model/device.
        self.register_buffer(
            "roi_class_weights", class_weights, persistent=False
        )

    def forward_loss(self, image_target, pred, label):
        if self.model_args.arch_version.startswith('v1'):
            if self.model_args.dataset_type == "2D":
                pred = self.unpatchify(pred)
            elif self.model_args.dataset_type == "3D":
                pred = self.unpatchify3D(pred)

        if "L2" in self.model_args.loss_version:
            global_recon_loss = (pred - image_target).pow(2).mean()
        elif "L1" in self.model_args.loss_version:
            global_recon_loss = torch.abs(pred - image_target).mean()
        else:
            raise ValueError("LossBalance requires L2 or L1 in loss_version")

        (
            roi_loss,
            class_losses,
            class_counts,
            valid_samples,
            weighted_mass,
        ) = frequency_balanced_roi_l2(
            pred,
            image_target,
            label,
            class_weights=self.roi_class_weights,
        )
        loss = global_recon_loss + self.model_args.roi_loss_weight * roi_loss

        if "LPIPS" in self.model_args.loss_version:
            if self.model_args.dataset_type == "2D":
                p_loss = self.perceptual_loss(
                    image_target.contiguous(), pred.contiguous()
                ).mean()
            else:
                slice_losses = []
                for slice_index in range(image_target.shape[2]):
                    slice_losses.append(self.perceptual_loss(
                        image_target[:, :, slice_index].contiguous(),
                        pred[:, :, slice_index].contiguous(),
                    ).mean())
                p_loss = torch.stack(slice_losses).mean()
        else:
            p_loss = pred.new_zeros(())

        return loss, p_loss, {
            "global_recon_loss": global_recon_loss,
            "roi_loss": roi_loss,
            "roi_class_losses": class_losses,
            "roi_class_counts": class_counts,
            "roi_valid_samples": valid_samples,
            "roi_weighted_mass": weighted_mass,
            "roi_class_weights": self.roi_class_weights,
        }

    def forward(self, imgs, mask_ratio=0.75, middle=None):
        image_target = middle["image_target"]
        label = middle["label"]
        x_restored, cls_tokens, middle_output = self.forward_encoder(
            imgs, mask_ratio, middle=middle
        )
        pred = self.forward_decoder(x_restored, cls_tokens, middle_output)
        loss, p_loss, loss_stats = self.forward_loss(image_target, pred, label)
        middle_output.update(loss_stats)
        if "LPIPS" in self.model_args.loss_version:
            middle_output["p_loss"] = p_loss

        if self.model_args.arch_version.startswith('v1'):
            if self.model_args.dataset_type == "2D":
                pred = self.unpatchify(pred)
            elif self.model_args.dataset_type == "3D":
                pred = self.unpatchify3D(pred)
        return loss, pred, middle_output


def mae_vit_base_patch16_dec512d8b(**kwargs):
    return LossBalancedMaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=768, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )


def mae_vit_large_patch16_dec512d8b(**kwargs):
    return LossBalancedMaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=1024, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    return LossBalancedMaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=1280, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )


mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b
mae_vit_basefix16_patch16 = mae_vit_base_patch16_dec512d8b
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b
