from functools import partial

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=1024):
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
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # [batch, 512, 7, 7]
            nn.ReLU(),
            nn.Flatten(),  # [batch, 512*7*7]
        )
        
        self.fc_mu = nn.Linear(512*7*7, latent_dim)
        self.fc_var = nn.Linear(512*7*7, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim=1024, out_channels=3):
        super().__init__()
        
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dim, 512*7*7),
            nn.ReLU()
        )

        self.decoder_conv = nn.Sequential(
            # Input: [batch, 512, 7, 7]
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # [batch, 256, 14, 14]
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
        x = self.decoder_linear(x)
        x = x.view(-1, 512, 7, 7)
        x = self.decoder_conv(x)
        return x


class VAE(nn.Module):
    def __init__(self, img_size=224, in_chans=3, embed_dim=1024, model_args=None):
        super().__init__()
        
        self.model_args = model_args
        self.encoder = Encoder(in_chans, embed_dim)
        self.decoder = Decoder(embed_dim, in_chans)

        if "LPIPS" in self.model_args.loss_version:
            from VQ.lpips import LPIPS
            self.perceptual_loss = LPIPS().eval()

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_loss(self, imgs, pred, mu, logvar):
        """
        imgs: [N, 3, H, W]
        pred: [N, 3, H, W]
        mu: [N, D]
        logvar: [N, D]
        """
        # Reconstruction loss
        recon_loss = torch.mean((pred - imgs) ** 2)

        # KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss
        loss = recon_loss + kl_loss

        if "LPIPS" in self.model_args.loss_version:
            p_loss = self.perceptual_loss(imgs.contiguous(), pred.contiguous())
            p_loss = torch.mean(p_loss)
        else:
            p_loss = torch.tensor([0.0])

        return loss, p_loss

    def forward(self, imgs, mask_ratio=0.0):
        middle_output = {}

        # Encode
        mu, logvar = self.encoder(imgs)
        
        # Sample latent vector
        z = self.reparameterize(mu, logvar)
        
        # Decode
        pred = self.decoder(z)
        
        # Calculate loss
        loss, p_loss = self.forward_loss(imgs, pred, mu, logvar)

        if "LPIPS" in self.model_args.loss_version:
            middle_output["p_loss"] = p_loss

        return loss, pred, middle_output


def vae_cnn_base(**kwargs):
    model = VAE(
        img_size=224, in_chans=3, embed_dim=1024, **kwargs)
    return model


def vae_cnn_large(**kwargs):
    model = VAE(
        img_size=224, in_chans=3, embed_dim=2048, **kwargs)
    return model


def vae_cnn_huge(**kwargs):
    model = VAE(
        img_size=224, in_chans=3, embed_dim=4096, **kwargs)
    return model


# set recommended archs
vae_cnn_base = vae_cnn_base  
vae_cnn_large = vae_cnn_large
vae_cnn_huge = vae_cnn_huge
