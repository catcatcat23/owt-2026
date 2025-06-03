from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed


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
        inputs = inputs.permute(0, 2, 1).contiguous()
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
        return quantized.permute(0, 2, 1).contiguous(), loss, perplexity


class Discriminator(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384, depth=6, num_heads=6):
        super().__init__()
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio=4., qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, 1)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        torch.nn.init.normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        x = self.patch_embed(x)
        
        x = x + self.pos_embed[:, 1:, :]
        
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        x = self.head(x[:, 0])
        return x


class VQGAN(nn.Module):
    """ VQGAN with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, model_args=None):
        super().__init__()

        self.model_args = model_args
        self.use_discriminator = getattr(model_args, 'use_discriminator', False)  # Changed default to True

        # --------------------------------------------------------------------------
        # VQGAN encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # VQGAN codebook
        self.vq = VectorQuantizer(num_embeddings=getattr(model_args, 'num_tokens', 8192), embedding_dim=embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # VQGAN decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # Discriminator
        if self.use_discriminator:
            self.discriminator = Discriminator(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads
            )
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

        # Get quantized tokens
        z_q, vq_loss, perplexity = self.vq(x[:, 1:])  # Remove cls token

        return z_q, vq_loss, perplexity

    def forward_decoder(self, z):
        # add cls token
        cls_token = torch.zeros(z.shape[0], 1, z.shape[2], device=z.device)
        z = torch.cat([cls_token, z], dim=1)
        
        # embed tokens
        x = self.decoder_embed(z)

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, vq_loss):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        """
        target = self.patchify(imgs)
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        # Reconstruction loss
        recon_loss = torch.mean((pred - target) ** 2)

        # Total loss
        loss = recon_loss + vq_loss

        pred = self.unpatchify(pred)
        image_target = self.unpatchify(target)

        if "LPIPS" in self.model_args.loss_version:
            p_loss = self.perceptual_loss(image_target.contiguous(), pred.contiguous())
            p_loss = torch.mean(p_loss)
        else:
            p_loss = torch.tensor([0.0])

        # Adversarial loss
        if self.use_discriminator:
            # Discriminator loss
            real_logits = self.discriminator(image_target)
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

        pred = self.unpatchify(pred)

        return loss, pred, middle_output


def vqgan_vit_base_patch16_dec512d8b(**kwargs):
    model = VQGAN(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vqgan_vit_large_patch16_dec512d8b(**kwargs):
    model = VQGAN(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vqgan_vit_huge_patch14_dec512d8b(**kwargs):
    model = VQGAN(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
vqgan_vit_base_patch16 = vqgan_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
vqgan_vit_large_patch16 = vqgan_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
vqgan_vit_huge_patch14 = vqgan_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
