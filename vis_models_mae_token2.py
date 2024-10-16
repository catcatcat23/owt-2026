# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
from torch.nn import Sigmoid

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed
from OrganEmbed import OrganEmbed, SpatialRestore

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, model_args=None):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.model_args = model_args
        # print("self.model_args", self.model_args)
        ### self.blocks = nn.ModuleList([
        ###     Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
        ###     for i in range(depth)])
        self.blocks1 = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(int(depth/2))])

        self.blocks2 = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(int(depth/2))])

        self.organ_embed = OrganEmbed(embed_dim, embed_dim, self.model_args.organ_token_total, img_size//patch_size)

        self.norm = norm_layer(embed_dim)
        self.sigmoid = Sigmoid()
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        ###  self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_embed = SpatialRestore(embed_dim, decoder_embed_dim, self.model_args.organ_token_total, img_size//patch_size)
        self.decoder_embed_cls = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        ###  self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # print("self.mask_token init", self.mask_token)

        ###  self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        ### decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        ### self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
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

    def random_masking(self, x, selected_classes):
        """
        Mask the indexed part of tokens based on selected classes.
        x: [B, 20*classes, c], sequence
        selected_classes: list of selected class indices
        """
        B, L, C = x.shape  # batch, length, channels
        tokens_per_class = self.model_args.token_factor

        mask = torch.ones([B, L], device=x.device)  # initialize mask with ones

        for cls in selected_classes:
            start_idx = cls * tokens_per_class
            end_idx = (cls + 1) * tokens_per_class
            mask[:, start_idx:end_idx] = 0  # mask the selected class tokens

        # Only save tokens that mask is not 0
        x_masked = x[mask == 1].reshape(B, -1, C)

        # print("x_masked.shape", x_masked.shape, x_masked) # torch.Size([64, 100, 768])
        # print("mask.shape", mask.shape, mask) # torch.Size([64, 200])

        return x_masked, mask

    def token_restore(self, x_masked, mask):
        # print("x_masked.shape before restore", x_masked.shape, x_masked) # torch.Size([64, 100, 768])
        B, L, C = x_masked.shape
        x_restored = torch.zeros((B, self.model_args.organ_token_total, C), device=x_masked.device, dtype=x_masked.dtype)
        # print("self.mask_token 2", self.mask_token, self.mask_token.shape) # torch.Size([1, 1, 768])

        masked_indices = torch.nonzero(mask == 0, as_tuple=False)[:, 1].reshape(B, -1)
        mask_tokens = self.mask_token.repeat(x_restored.shape[0], x_restored.shape[1] - x_masked.shape[1], 1)
        x_restored = x_restored.scatter(1, masked_indices.unsqueeze(-1).expand(-1, -1, C), mask_tokens.to(x_restored.dtype))

        # print("mask_tokens", mask_tokens, mask_tokens.shape) # torch.Size([64, 100, 768])
        # print("x_restored.shape1", x_restored.shape, x_restored) # torch.Size([64, 200, 768])

        unmasked_indices = torch.nonzero(mask == 1, as_tuple=False)[:, 1].reshape(B, -1)
        x_restored = x_restored.scatter(1, unmasked_indices.unsqueeze(-1).expand(-1, -1, C), x_masked)

        # print("x_restored.shape2", x_restored.shape, x_restored) # torch.Size([64, 200, 768])

        return x_restored

    def token_restore_sup(self, x_masked, mask, x_masked2):
        # print("x_masked.shape before restore", x_masked.shape, x_masked) # torch.Size([64, 100, 768])
        B, L, C = x_masked.shape
        x_restored = torch.zeros((B, self.model_args.organ_token_total, C), device=x_masked.device, dtype=x_masked.dtype)
        # print("self.mask_token 2", self.mask_token, self.mask_token.shape) # torch.Size([1, 1, 768])

        masked_indices = torch.nonzero(mask == 0, as_tuple=False)[:, 1].reshape(B, -1)
        # mask_tokens = self.mask_token.repeat(x_restored.shape[0], x_restored.shape[1] - x_masked.shape[1], 1)
        mask_tokens = x_masked2
        x_restored = x_restored.scatter(1, masked_indices.unsqueeze(-1).expand(-1, -1, C), mask_tokens.to(x_restored.dtype))

        # print("mask_tokens", mask_tokens, mask_tokens.shape) # torch.Size([64, 100, 768])
        # print("x_restored.shape1", x_restored.shape, x_restored) # torch.Size([64, 200, 768])

        unmasked_indices = torch.nonzero(mask == 1, as_tuple=False)[:, 1].reshape(B, -1)
        x_restored = x_restored.scatter(1, unmasked_indices.unsqueeze(-1).expand(-1, -1, C), x_masked)

        # print("x_restored.shape2", x_restored.shape, x_restored) # torch.Size([64, 200, 768])

        return x_restored

    def forward_encoder(self, x, mask_ratio, middle = None):
        image_target, random_selected_class = middle["image_target"], middle["random_selected_class"]

        # print("encoder, x.shape", x.shape) # torch.Size([64, 3, 224, 224])
        # embed patches
        x = self.patch_embed(x)
        # print("encoder, x.shape2", x.shape) # torch.Size([64, 196, 768])

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        # print("encoder, x.shape3", x.shape) # torch.Size([64, 196, 768])

        # masking: length -> length * mask_ratio
        ### x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token ## token2: save slice info here
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        # print("cls_token.shape1", cls_token.shape) # torch.Size([1, 1, 768])
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        # print("cls_tokens.shape2", cls_tokens.shape) # torch.Size([64, 1, 768])
        x = torch.cat((cls_tokens, x), dim=1)
        # print("x.shape4", x.shape) # torch.Size([64, 50, 768]) ## token2 torch.Size([64, 197, 768])

        # apply Transformer blocks
        for bi, blk in enumerate(self.blocks1):
            # print("bi", bi)
            x = blk(x) ## token2 torch.Size([64, 197, 768])
        x = self.norm(x)

        cls_tokens = x[:,:1,:]
        x = x[:,1:,:]
        x, _ = self.organ_embed(x) ## torch.Size([64, 200, 768])

        # print("x.shape before random", x.shape, x) # torch.Size([64, 200, 768])
        ## random mask organ tokens
        x_masked, mask = self.random_masking(x, random_selected_class) ## torch.Size([64, 100, 768])
        x_masked_ = torch.cat((cls_tokens, x_masked), dim=1) ## torch.Size([64, 101, 768])
        # print("x_masked_.shape after cls token", x_masked_.shape, x_masked_) # torch.Size([64, 101, 768])

        ## encoder2 (forward middle)
        for bi, blk in enumerate(self.blocks2):
            # print("bi", bi)
            x_masked_ = blk(x_masked_) ## token2 torch.Size([64, 101, 768])
        x_masked_ = self.norm(x_masked_)
        # print("x_masked_.shape after encoder2", x_masked_.shape) # torch.Size([64, 101, 768])

        cls_tokens = x_masked_[:,:1,:]
        x_masked = x_masked_[:,1:,:]

        x_restored = self.token_restore(x_masked, mask) ## torch.Size([64, 200, 768])
        ### x_restored = torch.cat((cls_tokens, x_restored), dim=1) ## torch.Size([64, 201, 768])
        # print("x_restored.shape final", x_restored.shape)

        middle_output = {"x_masked":x_masked, "mask":mask}

        return x_restored, cls_tokens, middle_output
        ### return x, mask, ids_restore

    def forward_decoder(self, x_restored, cls_tokens):
        # print("x_restored.shape, cls_tokens.shape decoder", x_restored.shape, cls_tokens.shape)
        # embed tokens # torch.Size([64, 200, 768])
        x, _ = self.decoder_embed(x_restored)
        # print("x.shape token1", x.shape) # torch.Size([64, 196, 512])
        # print("decoder, x.shape", x.shape) # torch.Size([64, 196, 512])

        cls_tokens = self.decoder_embed_cls(cls_tokens)
        x = torch.cat((cls_tokens, x), dim=1) ## 197

        ### append mask tokens to sequence
        ### mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        ### print("decoder, mask_tokens.shape", mask_tokens.shape) # torch.Size([64, 147, 512])
        ### x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        ### print("decoder, x_.shape1", x_.shape) # torch.Size([64, 196, 512])
        ### x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        ### print("decoder, x_.shape2", x_.shape) # torch.Size([64, 196, 512])
        ### x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        ### print("decoder, x.shape3", x.shape) # torch.Size([64, 197, 512])

        ### add pos embed
        ### x = x + self.decoder_pos_embed
        ### print("decoder, x.shape4", x.shape) # torch.Size([64, 197, 512])

        ## shouldn't have class tokens for generation
        ### x = torch.cat((cls_tokens, x), dim=1) # torch.Size([64, 197, 512])

        # apply Transformer blocks
        ## v2 for vqgan decoder, for better generation results (maybe VQ tokens?)
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        # print("x.shape token2", x.shape)
        # print("decoder, x.shape5", x.shape) # torch.Size([64, 197, 512])

        # predictor projection
        x = self.decoder_pred(x)
        # print("x.shape token3", x.shape)
        # print("decoder, x.shape6", x.shape) # torch.Size([64, 197, 768])
        #remove cls token
        x = x[:, 1:, :]
        ### print("decoder, x.shape7", x.shape) # torch.Size([64, 196, 768])
        x = self.sigmoid(x)
        return x

    def forward_loss(self, image_target, pred):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        # print("loss, imgs.shape", imgs.shape) # torch.Size([64, 3, 224, 224])
        # target = self.patchify(image_target)
        # # print("loss, target.shape", target.shape) # torch.Size([64, 196, 768])
        # if self.norm_pix_loss:
        #     mean = target.mean(dim=-1, keepdim=True)
        #     var = target.var(dim=-1, keepdim=True)
        #     target = (target - mean) / (var + 1.e-6)**.5

        # # print("loss, pred.shape, target.shape", pred.shape, target.shape) # torch.Size([64, 196, 768]) torch.Size([64, 196, 768])
        # target = self.unpatchify(target)
        pred = self.unpatchify(pred)
        # loss = (pred - target) ** 2
        loss = (pred - image_target) ** 2
        ### loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = loss.mean()

        # print("loss", loss)

        ### loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75, middle=None): ## imgs == combined latent, when args.decoder_only
        image_target, random_selected_class = middle["image_target"], middle["random_selected_class"]
        # middle["organ_token_total"] = 1*args.token_factor*1 + args.token_factor*args.num_classes ## 20+180 = 200
        # middle["organ_token_selet"] = args.token_factor*len(random_selected_class) ## 100
        # print('middle["organ_token_total"], middle["organ_token_selet"]', middle["organ_token_total"], middle["organ_token_selet"])
        # print("imgs.shape, mask_ratio", imgs.shape, mask_ratio) ## torch.Size([64, 3, 224, 224]) 0.75
        # if args.decoder_only == False:
        ### latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, middle = middle)
        x_restored, cls_tokens, _ = self.forward_encoder(imgs, mask_ratio, middle = middle)
        # print("latent.shape, mask.shape", latent.shape, mask.shape) # torch.Size([64, 50, 768]) torch.Size([64, 196])
        ### pred = self.forward_decoder(latent) #, ids_restore)  # [N, L, p*p*3] ## this latent should be 197,768 after OrganFuse (TokenFuse)
        pred = self.forward_decoder(x_restored, cls_tokens)
        # print("pred.shape", pred.shape) # torch.Size([64, 196, 768])
        loss = self.forward_loss(image_target, pred) #, mask)
        return loss, self.unpatchify(pred), None


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=768, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=1024, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=1280, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
