from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Any, Tuple

class BaseVAE(nn.Module):
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        raise NotImplementedError

    def decode(self, input: torch.Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    # @abstractmethod
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        pass

    # @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> torch.Tensor:
        pass

class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=1024):
        super().__init__()
        
        # Define hidden dimensions
        hidden_dims = [32, 64, 128, 256, 512]
        
        # Build Encoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        
        # Calculate flattened dimension
        self.flatten_dim = hidden_dims[-1] * 7 * 7  # For 224x224 input
        
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_var = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim=1024, out_channels=3):
        super().__init__()
        
        hidden_dims = [512, 256, 128, 64, 32]
        
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[0] * 7 * 7)  # For 224x224 output
        
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels=out_channels,
                                      kernel_size=3, padding=1),
                            nn.Tanh())

    def forward(self, x):
        x = self.decoder_input(x)
        x = x.view(-1, 512, 7, 7)  # Reshape to match encoder output dimensions
        x = self.decoder(x)
        x = self.final_layer(x)
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
