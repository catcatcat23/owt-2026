from functools import partial

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_channels=3, embed_dim=1024):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # Input: [batch, 3, 224, 224]
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),  # [batch, 32, 112, 112]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # [batch, 64, 56, 56]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # [batch, 128, 28, 28]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # [batch, 256, 14, 14]
            nn.ReLU(),
            nn.Conv2d(256, embed_dim, kernel_size=4, stride=2, padding=1),  # [batch, embed_dim, 7, 7]
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, embed_dim=1024, out_channels=3):
        super().__init__()

        self.decoder = nn.Sequential(
            # Input: [batch, embed_dim, 7, 7]
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=4, stride=2, padding=1),  # [batch, 256, 14, 14]
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [batch, 128, 28, 28]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [batch, 64, 56, 56]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # [batch, 32, 112, 112]
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),  # [batch, 3, 224, 224]
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)


class VQVAE(nn.Module):
    def __init__(self, img_size=224, in_chans=3, embed_dim=1024, model_args=None):
        super().__init__()
        
        self.model_args = model_args
        self.encoder = Encoder(in_chans, embed_dim)
        self.vq = VectorQuantizer(num_embeddings=getattr(model_args, 'num_tokens', 8192), embedding_dim=embed_dim)
        self.decoder = Decoder(embed_dim, in_chans)

        if "LPIPS" in self.model_args.loss_version:
            from VQ.lpips import LPIPS
            self.perceptual_loss = LPIPS().eval()

    def forward_loss(self, imgs, pred, vq_loss):
        """
        imgs: [N, 3, H, W]
        pred: [N, 3, H, W]
        """
        # Reconstruction loss
        recon_loss = torch.mean((pred - imgs) ** 2)

        # Total loss
        loss = recon_loss + vq_loss

        if "LPIPS" in self.model_args.loss_version:
            p_loss = self.perceptual_loss(imgs.contiguous(), pred.contiguous())
            p_loss = torch.mean(p_loss)
        else:
            p_loss = torch.tensor([0.0])

        return loss, p_loss

    def forward(self, imgs, mask_ratio=0.0):
        middle_output = {}

        # Encode
        z = self.encoder(imgs)
        
        # Vector Quantize
        z_q, vq_loss, perplexity = self.vq(z)
        middle_output["vq_loss"] = vq_loss
        middle_output["perplexity"] = perplexity
        
        # Decode
        pred = self.decoder(z_q)
        
        # Calculate loss
        loss, p_loss = self.forward_loss(imgs, pred, vq_loss)

        if "LPIPS" in self.model_args.loss_version:
            middle_output["p_loss"] = p_loss

        return loss, pred, middle_output


def vqvae_cnn_base(**kwargs):
    model = VQVAE(
        img_size=224, in_chans=3, embed_dim=1024, **kwargs)
    return model


def vqvae_cnn_large(**kwargs):
    model = VQVAE(
        img_size=224, in_chans=3, embed_dim=2048, **kwargs)
    return model


def vqvae_cnn_huge(**kwargs):
    model = VQVAE(
        img_size=224, in_chans=3, embed_dim=4096, **kwargs)
    return model


# set recommended archs
vqvae_cnn_base = vqvae_cnn_base  
vqvae_cnn_large = vqvae_cnn_large
vqvae_cnn_huge = vqvae_cnn_huge
