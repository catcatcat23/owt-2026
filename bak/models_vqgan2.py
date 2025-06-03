from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, inputs):
        # Convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = torch.mean((quantized.detach() - inputs)**2)
        q_latent_loss = torch.mean((quantized - inputs.detach())**2)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Convert quantized from BHWC -> BCHW
        return quantized.permute(0, 3, 1, 2).contiguous(), loss, perplexity


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        
        self.discriminator = nn.Sequential(
            # Input: [batch, 3, 224, 224]
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),  # [batch, 32, 112, 112]
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # [batch, 64, 56, 56]
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # [batch, 128, 28, 28]
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # [batch, 256, 14, 14]
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # [batch, 512, 7, 7]
            nn.LeakyReLU(0.2),
            nn.Flatten(),  # [batch, 512*7*7]
            nn.Linear(512*7*7, 1)  # [batch, 1]
        )

    def forward(self, x):
        return self.discriminator(x)


class VQGAN(nn.Module):
    def __init__(self, img_size=224, in_chans=3, embed_dim=1024, model_args=None):
        super().__init__()

        self.model_args = model_args
        self.use_discriminator = getattr(model_args, 'use_discriminator', True)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_chans, 32, kernel_size=4, stride=2, padding=1),  # [batch, 32, 112, 112]
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

        # VQ Layer
        self.vq = VectorQuantizer(num_embeddings=getattr(model_args, 'num_tokens', 8192), embedding_dim=embed_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=4, stride=2, padding=1),  # [batch, 256, 14, 14]
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [batch, 128, 28, 28]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [batch, 64, 56, 56]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # [batch, 32, 112, 112]
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_chans, kernel_size=4, stride=2, padding=1),  # [batch, 3, 224, 224]
            nn.Sigmoid()
        )

        # Discriminator
        if self.use_discriminator:
            self.discriminator = Discriminator(in_chans)

        if "LPIPS" in self.model_args.loss_version:
            from VQ.lpips import LPIPS
            self.perceptual_loss = LPIPS().eval()

    def forward_encoder(self, x):
        x = self.encoder(x)
        z_q, vq_loss, perplexity = self.vq(x)
        return z_q, vq_loss, perplexity

    def forward_decoder(self, z):
        return self.decoder(z)

    def forward_loss(self, imgs, pred, vq_loss):
        # Reconstruction loss
        recon_loss = torch.mean((pred - imgs) ** 2)

        # Total loss
        loss = recon_loss + vq_loss

        if "LPIPS" in self.model_args.loss_version:
            p_loss = self.perceptual_loss(imgs.contiguous(), pred.contiguous())
            p_loss = torch.mean(p_loss)
        else:
            p_loss = torch.tensor([0.0])

        # Adversarial loss
        if self.use_discriminator:
            # Discriminator loss
            real_logits = self.discriminator(imgs)
            fake_logits = self.discriminator(pred.detach())
            
            d_loss_real = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
            d_loss_fake = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
            d_loss = d_loss_real + d_loss_fake
            
            # Generator loss
            g_loss = F.binary_cross_entropy_with_logits(self.discriminator(pred), torch.ones_like(fake_logits))
            
            # Add adversarial loss to total loss
            loss = loss + g_loss
        else:
            d_loss = torch.tensor([0.0])
            g_loss = torch.tensor([0.0])

        return loss, p_loss, d_loss, g_loss

    def forward(self, imgs, mask_ratio=0.0):
        middle_output = {}

        # Encode and quantize
        z_q, vq_loss, perplexity = self.forward_encoder(imgs)
        middle_output["vq_loss"] = vq_loss
        middle_output["perplexity"] = perplexity
        
        # Decode
        pred = self.forward_decoder(z_q)
        
        # Calculate loss
        loss, p_loss, d_loss, g_loss = self.forward_loss(imgs, pred, vq_loss)

        if "LPIPS" in self.model_args.loss_version:
            middle_output["p_loss"] = p_loss
            
        if self.use_discriminator:
            middle_output["d_loss"] = d_loss
            middle_output["g_loss"] = g_loss

        return loss, pred, middle_output


def vqgan_cnn_base(**kwargs):
    model = VQGAN(
        img_size=224, in_chans=3, embed_dim=1024, **kwargs)
    return model


def vqgan_cnn_large(**kwargs):
    model = VQGAN(
        img_size=224, in_chans=3, embed_dim=2048, **kwargs)
    return model


def vqgan_cnn_huge(**kwargs):
    model = VQGAN(
        img_size=224, in_chans=3, embed_dim=4096, **kwargs)
    return model


# set recommended archs
vqgan_cnn_base = vqgan_cnn_base  
vqgan_cnn_large = vqgan_cnn_large
vqgan_cnn_huge = vqgan_cnn_huge
