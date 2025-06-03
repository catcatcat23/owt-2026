from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed


class VAE(nn.Module):
    """ Variational Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, model_args=None):
        super().__init__()

        self.model_args = model_args

        # --------------------------------------------------------------------------
        # VAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # VAE latent space
        self.avgpool = nn.AdaptiveAvgPool1d(49)  # Pool to 49 tokens
        self.fc_mu = nn.Linear(49 * embed_dim, 1024)  # Flatten 49 tokens then project
        self.fc_var = nn.Linear(49 * embed_dim, 1024)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # VAE decoder specifics
        self.decoder_embed = nn.Linear(1024, decoder_embed_dim * 49, bias=True)  # Project to sequence length

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

        if "LPIPS" in self.model_args.loss_version:
            from VQ.lpips import LPIPS
            self.perceptual_loss = LPIPS().eval()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # Get cls token and patch tokens
        cls_token = x[:, :1, :]  # [B, 1, D]
        patch_tokens = x[:, 1:, :]  # [B, L, D]
        
        # Apply adaptive average pooling to patch tokens
        patch_tokens = patch_tokens.transpose(1, 2)  # [B, D, L]
        pooled = self.avgpool(patch_tokens)  # [B, D, 49]
        pooled = pooled.transpose(1, 2)  # [B, 49, D]
        pooled_flat = pooled.reshape(pooled.size(0), -1)  # [B, 49*D]

        # Get mu and logvar from pooled tokens
        mu = self.fc_mu(pooled_flat)
        logvar = self.fc_var(pooled_flat)

        return mu, logvar, cls_token

    def forward_decoder(self, z, cls_token):
        # Project latent vector to sequence
        x = self.decoder_embed(z)  # [B, (L+1)*D]
        
        # Reshape to [B, 49, D]
        batch_size = z.shape[0]
        x = x.view(batch_size, 49, -1)  # [B, 49, D]
        
        # Interpolate to 196 tokens
        x = x.transpose(1, 2)  # [B, D, 49]
        x = F.interpolate(x.unsqueeze(3).reshape(batch_size, -1, 7, 7), 
                         size=(14, 14), mode='bilinear', align_corners=False)
        x = x.reshape(batch_size, -1, 196).transpose(1, 2)  # [B, 196, D]
        
        # Append cls token
        # x = torch.cat([cls_token, x], dim=1)  # [B, 197, D]
        
        # add pos embed
        x = x + self.decoder_pos_embed[:, 1:, :]

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        # x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mu, logvar):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mu: [N, D]
        logvar: [N, D]
        """
        target = self.patchify(imgs)
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        # Reconstruction loss
        recon_loss = torch.mean((pred - target) ** 2)

        # KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss
        loss = recon_loss + kl_loss

        pred = self.unpatchify(pred)
        image_target = self.unpatchify(target)

        if "LPIPS" in self.model_args.loss_version:
            p_loss = self.perceptual_loss(image_target.contiguous(), pred.contiguous())
            p_loss = torch.mean(p_loss)
        else:
            p_loss = torch.tensor([0.0])

        return loss, p_loss

    def forward(self, imgs, mask_ratio=0.0):
        middle_output = {}

        # Encode
        mu, logvar, cls_token = self.forward_encoder(imgs)
        
        # Sample latent vector
        z = self.reparameterize(mu, logvar)
        
        # Decode
        pred = self.forward_decoder(z, cls_token)
        
        # Calculate loss
        loss, p_loss = self.forward_loss(imgs, pred, mu, logvar)

        if "LPIPS" in self.model_args.loss_version:
            middle_output["p_loss"] = p_loss

        pred = self.unpatchify(pred)

        return loss, pred, middle_output


def vae_vit_base_patch16_dec512d8b(**kwargs):
    model = VAE(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vae_vit_large_patch16_dec512d8b(**kwargs):
    model = VAE(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vae_vit_huge_patch14_dec512d8b(**kwargs):
    model = VAE(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
vae_vit_base_patch16 = vae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
vae_vit_large_patch16 = vae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
vae_vit_huge_patch14 = vae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
