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

from timm.models.vision_transformer import PatchEmbed, Block #, Attention
from util.patch_embed import PatchEmbed3D, PatchEmbed3Dv2

from util.pos_embed import get_2d_sincos_pos_embed
from OrganEmbed import OrganEmbed, OrganEmbed2, SpatialRestore
from einops import rearrange

from timm.models.layers import Mlp, DropPath
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List

# class AttentionLA(Attention):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, norm_layer=nn.LayerNorm, attn_drop=0., proj_drop=0.):
#         super().__init__(dim, num_heads, qkv_bias, norm_layer, attn_drop, proj_drop)
#         self.attn_drop = nn.Identity()
#         self.proj_drop = nn.Identity()

class AttentionLA(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_norm=False, norm_layer=nn.LayerNorm, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # print("x.shape attn1", x.shape)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # print("qkv.shape attn1", qkv.shape)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        # print("q.shape attn1", q.shape)

        ## LA
        dim = q.shape[-1]
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)
        q = q * dim ** -0.5
        context = torch.einsum('bhnd,bhne->bhde', k, v)
        # print("context.shape attn1", context.shape)
        x = torch.einsum('bhnd,bhde->bhne', q, context)
        # x = attn.reshape(*q.shape)
        # print("x.shape attn2", x.shape)

        x = x.transpose(1, 2).reshape(B, N, C)
        # print("x.shape attn3", x.shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        # print("x.shape attn4", x.shape)

        return x

# class BlockLA(Block):
#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, norm_layer=nn.LayerNorm, **kwargs):
#         super().__init__(dim, num_heads, mlp_ratio, qkv_bias, norm_layer, **kwargs)
#         self.attn = AttentionLA(dim, num_heads=num_heads, qkv_bias=qkv_bias, norm_layer=norm_layer)

class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class BlockLA(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AttentionLA(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, model_args=None):
        super().__init__()

        self.model_args = model_args
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.img_size = img_size
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        if self.model_args.arch_version.startswith("v1") or self.model_args.arch_version.startswith("v2"):
            self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
            if self.model_args.dataset_type == "3D":
                if self.model_args.model == "mae_vit_base16_patch16":
                    self.patch_embed = PatchEmbed3D(img_size, patch_size//4, in_chans, embed_dim)
                elif self.model_args.model == "mae_vit_base16v2_patch16":
                    self.patch_embed = PatchEmbed3Dv2(img_size, patch_size, in_chans, embed_dim)
                else: ## normal
                    self.patch_embed = PatchEmbed3D(img_size, patch_size, in_chans, embed_dim)
            num_patches = self.patch_embed.num_patches
    
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            if self.model_args.dataset_type == "2D":
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
            elif self.model_args.dataset_type == "3D":
                self.pos_embed_spatial = nn.Parameter(torch.zeros(1, self.patch_embed.grid_size[1] * self.patch_embed.grid_size[2], embed_dim))
                self.pos_embed_temporal = nn.Parameter(torch.zeros(1, self.patch_embed.grid_size[0], embed_dim))
    
            # print("self.model_args", self.model_args)
            ### self.blocks = nn.ModuleList([
            ###     Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            ###     for i in range(depth)])
            if self.model_args.LA:
                self.blocks1 = nn.ModuleList([
                    BlockLA(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) 
                    for i in range(int(depth/2))]) ## delete all qk_scale=None, for H100
            else:
                self.blocks1 = nn.ModuleList([
                    Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) 
                    for i in range(int(depth/2))]) ## delete all qk_scale=None, for H100
        elif self.model_args.arch_version.startswith("v3"):
            from VQ.VQ_model import Encoder
            self.encoder = Encoder(ch=128, out_ch=3, ch_mult=[1,1,2,2,4], num_res_blocks=2, attn_resolutions=[16], dropout=0.0, resamp_with_conv=True, in_channels=3, resolution=224, z_channels=embed_dim, double_z=False, give_pre_end=False)
            self.avg_pool = nn.AdaptiveAvgPool2d((int(self.model_args.cls_num**.5),int(self.model_args.cls_num**.5)))
            self.encoder_embed = nn.Linear(embed_dim, embed_dim, bias=False)

        if self.model_args.LA:
            self.blocks2 = nn.ModuleList([
                BlockLA(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) 
                for i in range(int(depth/2))]) ## delete all qk_scale=None for H100
        else:
            self.blocks2 = nn.ModuleList([
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) 
                for i in range(int(depth/2))]) ## delete all qk_scale=None for H100

        self.embed_dim = embed_dim
        self.hw_size = img_size//patch_size
        self.organ_embed = OrganEmbed(embed_dim, embed_dim, self.model_args.organ_token_total, img_size//patch_size)
        if self.model_args.dataset_type == "3D":
            self.organ_embed = OrganEmbed2(embed_dim, embed_dim, self.model_args.organ_token_total, img_size//patch_size)

        self.norm = norm_layer(embed_dim)
        self.sigmoid = Sigmoid()
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        ###  self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_embed = SpatialRestore(embed_dim, decoder_embed_dim, self.model_args.organ_token_total, (img_size//patch_size)*(img_size//patch_size))
        if self.model_args.dataset_type == "3D":
            # if self.model_args.model == "mae_vit_base16_patch16" or self.model_args.model == "mae_vit_base16v2_patch16":
            #     self.decoder_embed = nn.ModuleList([
            #         SpatialRestore(embed_dim, decoder_embed_dim, self.model_args.organ_token_total, (self.patch_embed.grid_size[0]*self.patch_embed.grid_size[1]*self.patch_embed.grid_size[2])//36)
            #         for i in range(int(36))
            #         ])
            # else: ## normal
            self.decoder_embed = SpatialRestore(embed_dim, decoder_embed_dim, self.model_args.organ_token_total, self.patch_embed.grid_size[0]*self.patch_embed.grid_size[1]*self.patch_embed.grid_size[2])

        if self.model_args.arch_version == 'v11' or self.model_args.arch_version == 'v21' or self.model_args.arch_version == 'v31': ## only v11 no decoder cls token
            pass
        else:
            self.decoder_embed_cls = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        ###  self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # print("self.mask_token init", self.mask_token)

        ###  self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        if not self.model_args.arch_version.startswith("v1"): ## v2, v3...
            decoder_depth = int(decoder_depth/2)
            from VQ.VQ_model import Decoder
            self.decoder = Decoder(ch=128, out_ch=3, ch_mult=[1,1,2,2,4], num_res_blocks=2, attn_resolutions=[16], dropout=0.0, resamp_with_conv=True, in_channels=3, resolution=224, z_channels=decoder_embed_dim, give_pre_end=False)

        if self.model_args.LA:
            self.decoder_blocks = nn.ModuleList([
                BlockLA(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(decoder_depth)]) ## delete all qk_scale=None for H100
        else:
            self.decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(decoder_depth)]) ## delete all qk_scale=None for H100

        self.decoder_norm = norm_layer(decoder_embed_dim)
        if not self.model_args.arch_version.startswith("v1"): ## v2, v3...
            pass
        else: ## only v1
            self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
            if self.model_args.dataset_type == "3D":
                # self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans * 16, bias=True)
                if self.model_args.model == "mae_vit_base4_patch16":
                    self.decoder_pred = nn.Sequential(
                            nn.Linear(decoder_embed_dim, patch_size**2 * in_chans * 4, bias=True),
                            nn.Tanh(),
                            nn.Linear(patch_size**2 * in_chans * 4, patch_size**2 * in_chans * 16, bias=True))
                elif self.model_args.model == "mae_vit_base16_patch16" or self.model_args.model == "mae_vit_base16v2_patch16":
                    self.decoder_pred = nn.Sequential(
                            nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True),
                            nn.Tanh(),
                            nn.Linear(patch_size**2 * in_chans, patch_size**2 * in_chans, bias=True))
                else:
                    self.decoder_pred = nn.Sequential(
                            nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True),
                            nn.Tanh(),
                            nn.Linear(patch_size**2 * in_chans, patch_size**2 * in_chans * 4, bias=True),
                            nn.Tanh(),
                            nn.Linear(patch_size**2 * in_chans * 4, patch_size**2 * in_chans * 16, bias=True))
        # --------------------------------------------------------------------------
        if self.model_args.vq_version != None:
            self.n_embed = self.model_args.vq_n_token ## for each organ

            if self.model_args.vq_version.startswith('v0'):
                from VQ.VQ_model import VectorQuantizer2_OWC as VectorQuantizer

                if self.model_args.lib_version == None:
                    self.n_embed = self.model_args.vq_n_token * self.model_args.num_classes_with_bg
                    self.quantize = VectorQuantizer(self.n_embed, decoder_embed_dim, beta=0.25, model_args=self.model_args, legacy=False)
                else:
                    if self.model_args.lib_version == "v0":
                        self.library = nn.ModuleList([VectorQuantizer(self.n_embed, decoder_embed_dim, beta=0.25, model_args=self.model_args, legacy=False) for _ in range(self.model_args.num_classes_with_bg)])
                    elif self.model_args.lib_version == "v01":
                        library_ = [VectorQuantizer(self.n_embed*4, decoder_embed_dim, beta=0.25, model_args=self.model_args, legacy=False)]
                        self.library = nn.ModuleList(library_ + [VectorQuantizer(self.n_embed, decoder_embed_dim, beta=0.25, model_args=self.model_args, legacy=False) for _ in range(self.model_args.num_classes)])
                if self.model_args.vq_version == 'v0':
                    self.quant_lin = nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=False)
                    self.post_quant_lin = nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=False)
                elif self.model_args.vq_version == 'v01':
                    self.quant_lin = nn.Sequential(
                        nn.Linear(decoder_embed_dim, decoder_embed_dim),
                        nn.Tanh(),
                        nn.Linear(decoder_embed_dim, decoder_embed_dim))
                    self.post_quant_lin = nn.Sequential(
                        nn.Linear(decoder_embed_dim, decoder_embed_dim),
                        nn.Tanh(),
                        nn.Linear(decoder_embed_dim, decoder_embed_dim))
                    self.decoder_pred = nn.Sequential(
                        nn.Linear(decoder_embed_dim, decoder_embed_dim),
                        nn.Tanh(),
                        nn.Linear(decoder_embed_dim, patch_size**2 * in_chans))
                    self.quant_lin.apply(self._init_weights)
                    self.post_quant_lin.apply(self._init_weights)
                    self.decoder_pred.apply(self._init_weights)
            elif self.model_args.vq_version == 'v1':
                from VQ.norm_ema_quantizer import NormEMAVectorQuantizer_OWC as VectorQuantizer

                if self.model_args.lib_version == None:
                    self.n_embed = self.model_args.vq_n_token * self.model_args.num_classes_with_bg
                    self.quantize = VectorQuantizer(self.n_embed, decoder_embed_dim, beta=1.0, model_args=self.model_args, kmeans_init=True, decay=0.99)
                else:
                    if self.model_args.lib_version == "v0":
                        self.library = nn.ModuleList([VectorQuantizer(self.n_embed, decoder_embed_dim, beta=1.0, model_args=self.model_args, kmeans_init=True, decay=0.99) for _ in range(self.model_args.num_classes_with_bg)])
                    elif self.model_args.lib_version == "v01":
                        library_ = [VectorQuantizer(self.n_embed*4, decoder_embed_dim, beta=1.0, model_args=self.model_args, kmeans_init=True, decay=0.99)]
                        self.library = nn.ModuleList(library_ + [VectorQuantizer(self.n_embed, decoder_embed_dim, beta=1.0, model_args=self.model_args, kmeans_init=True, decay=0.99) for _ in range(self.model_args.num_classes)])
                # self.quant_lin = nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=False)
                # self.post_quant_lin = nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=False)
                self.quant_lin = nn.Sequential(
                    nn.Linear(decoder_embed_dim, decoder_embed_dim),
                    nn.Tanh(),
                    nn.Linear(decoder_embed_dim, decoder_embed_dim))
                self.post_quant_lin = nn.Sequential(
                    nn.Linear(decoder_embed_dim, decoder_embed_dim),
                    nn.Tanh(),
                    nn.Linear(decoder_embed_dim, decoder_embed_dim))
                self.decoder_pred = nn.Sequential(
                    nn.Linear(decoder_embed_dim, decoder_embed_dim),
                    nn.Tanh(),
                    nn.Linear(decoder_embed_dim, patch_size**2 * in_chans))
                self.quant_lin.apply(self._init_weights)
                self.post_quant_lin.apply(self._init_weights)
                self.decoder_pred.apply(self._init_weights)

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

        if "LPIPS" in self.model_args.loss_version:
            from VQ.lpips import LPIPS
            self.perceptual_loss = LPIPS().eval()
            # self.perceptual_weight = 1.0

    def initialize_weights(self):
        # initialization
        if self.model_args.arch_version.startswith("v1") or self.model_args.arch_version.startswith("v2"):
            # initialize (and freeze) pos_embed by sin-cos embedding
            if self.model_args.dataset_type == "2D":
                pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
                self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
            elif self.model_args.dataset_type == "3D":
                torch.nn.init.normal_(self.pos_embed_spatial, std=.02)
                torch.nn.init.normal_(self.pos_embed_temporal, std=.02)

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

    # def patchify3D(self, imgs):
    #     """
    #     imgs: (N, 3, T, H, W)
    #     x: (N, L, patch_size**2 *3 *temp_stride)
    #     """
    #     x = rearrange(imgs, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=self.temp_stride, p1=self.patch_embed.patch_size[1], p2=self.patch_embed.patch_size[2])
    #     x = rearrange(x, 'b n p c -> b n (p c)')
    #     return x

    def unpatchify3D(self, x):
        """
        x: (N, L, patch_size**2 *3 *temp_stride)
        imgs: (N, 3, T, H, W)
        """
        x = rearrange(x, 'b (t h w) (p0 p1 p2 c) -> b c (t p0) (h p1) (w p2)', p1=self.patch_size, p2=self.patch_size, c=self.in_chans, h=self.img_size//self.patch_size, w=self.img_size//self.patch_size)
        return x

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
        if self.model_args.arch_version.startswith("v1") or self.model_args.arch_version.startswith("v2"):
            # print("encoder, x.shape", x.shape) # torch.Size([64, 3, 224, 224]) # ([1, 3, 192, 224, 224])
            # embed patches
            x = self.patch_embed(x)
            # print("encoder, x.shape2", x.shape) # torch.Size([64, 196, 768]) # torch.Size([1, 2352, 768])
    
            # add pos embed w/o cls token
            if self.model_args.dataset_type == "3D":
                # add pos embed w/o cls token
                self.pos_embed = self.pos_embed_spatial.repeat(1, self.patch_embed.grid_size[0], 1) + \
                            torch.repeat_interleave(self.pos_embed_temporal, self.patch_embed.grid_size[1] * self.patch_embed.grid_size[2], dim=1)
                # print("self.pos_embed.shape 1", self.pos_embed.shape) ## torch.Size([1, 19600, 768])
                self.pos_embed = torch.cat([self.cls_token, self.pos_embed], 1)
                # print("self.pos_embed.shape 2", self.pos_embed.shape) ## torch.Size([1, 19601, 768])
                self.pos_embed = self.pos_embed[:,:x.shape[1]+1,:]
                shape1=x.shape[1]
                # print("self.pos_embed.shape 3", self.pos_embed.shape) ## torch.Size([1, 2353, 768])
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
            x = x[:,1:,:] ## torch.Size([64, 196, 768])
        elif self.model_args.arch_version.startswith("v3"):
            x_ = self.encoder(x)
            # print("v3 encoder final x_.shape", x_.shape) # torch.Size([32, 768, 14, 14])
            cls_tokens = self.avg_pool(x_)
            cls_tokens = cls_tokens.flatten(2)
            cls_tokens = cls_tokens.transpose(-1, -2)
            # print("cls_tokens.shape", cls_tokens.shape)
            x_ = x_.flatten(2) # torch.Size([32, 768, 196])
            x_ = x_.transpose(-1, -2) # torch.Size([32, 196, 768])
            x = self.encoder_embed(x_)
            # print("x.shape encoder finalfinal", x.shape) # torch.Size([32, 196, 768])

        x, _ = self.organ_embed(x) ## torch.Size([64, 200, 768])

        # print("x.shape before random", x.shape) # torch.Size([64, 200, 768])
        ## random mask organ tokens
        x_masked_b, mask = self.random_masking(x, random_selected_class) ## torch.Size([64, 100, 768])

        if self.model_args.arch_version == 'v11' or self.model_args.arch_version == 'v21' or self.model_args.arch_version == 'v31': ## only v11
            # x_masked_b = torch.cat((cls_tokens, x_masked_b), dim=1) ## torch.Size([64, 101, 768])
            # print("x_masked_.shape after cls token", x_masked_.shape) # torch.Size([64, 101, 768])
            ## encoder2 (forward middle)
            x_masked_ = self.blocks2[0](x_masked_b)
            for bi, blk in enumerate(self.blocks2):
                if bi > 0:
                    x_masked_ = blk(x_masked_) ## token2 torch.Size([64, 101, 768])
            x_masked_ = self.norm(x_masked_)
            x_masked_ = x_masked_b + x_masked_
            x_masked = x_masked_
        else:
            x_masked_ = torch.cat((cls_tokens, x_masked_b), dim=1) ## torch.Size([64, 101, 768])
            # print("x_masked_.shape after cls token", x_masked_.shape) # torch.Size([64, 101, 768])
            ## encoder2 (forward middle)
            for bi, blk in enumerate(self.blocks2):
                # print("bi", bi)
                x_masked_ = blk(x_masked_) ## token2 torch.Size([64, 101, 768])
            x_masked_ = self.norm(x_masked_)
            # print("x_masked_.shape after encoder2", x_masked_.shape) # torch.Size([64, 101, 768])

            cls_tokens = x_masked_[:,:self.model_args.cls_num,:]
            x_masked = x_masked_[:,self.model_args.cls_num:,:]

        # print("cls_tokens.shape after split", cls_tokens.shape)
        # print("x_masked.shape after split", x_masked.shape)

        ## VQ here
        if self.model_args.vq_version != None:
            if self.model_args.vq_version == 'v0':
                x_masked = self.quant_lin(x_masked)
                # x_masked_vq, vq_loss, (_, _, min_encoding_indices) = self.quantize(x_masked)
                # print("min_encoding_indices", min_encoding_indices)
                # x_masked = self.post_quant_lin(x_masked_vq)
            elif self.model_args.vq_version == 'v01':
                with torch.cuda.amp.autocast(enabled=False):
                    x_masked = self.quant_lin(x_masked.type_as(self.quant_lin[-1].weight))
                # x_masked_vq, vq_loss, (_, _, min_encoding_indices) = self.quantize(x_masked)
                # x_masked = self.post_quant_lin(x_masked_vq)
            elif self.model_args.vq_version == 'v1':
                with torch.cuda.amp.autocast(enabled=False):
                    x_masked = self.quant_lin(x_masked.type_as(self.quant_lin[-1].weight))
                # x_masked_vq, vq_loss, (_, _, min_encoding_indices) = self.quantize(x_masked)
                # x_masked = self.post_quant_lin(x_masked_vq)

            if self.model_args.lib_version == None:
                x_masked_vq, vq_loss, (_, _, min_encoding_indices) = self.quantize(x_masked)
            else:
                min_encoding_indices = []
                x_masked_vq_list = []
                x_masked_split = torch.split(x_masked, self.model_args.token_factor, dim=1) ## tuple of (32, 20, 768)
                vq_loss = 0
                loss_count = 0
                for sp_i, sp_j in enumerate(x_masked_split):
                    x_masked_vq_i, vq_loss_i, (_, _, min_encoding_indices_i) = self.library[sp_i](sp_j)
                    x_masked_vq_list.append(x_masked_vq_i)
                    vq_loss+=vq_loss_i
                    loss_count+=1
                    min_encoding_indices.append(min_encoding_indices_i)
                    # print("loss_count, vq_loss, vq_loss_i", loss_count, vq_loss, vq_loss_i)
                vq_loss = vq_loss/loss_count
                x_masked_vq = torch.cat(x_masked_vq_list, dim = 1)

            x_masked = self.post_quant_lin(x_masked_vq)

        if self.model_args.training_version.startswith('v01'):
            x_restored = x_masked ## torch.Size([64, 200, 768])
            # print("no mask token restore", x_restored.shape)
        else:
            x_restored = self.token_restore(x_masked, mask) ## torch.Size([64, 200, 768])
        ### x_restored = torch.cat((cls_tokens, x_restored), dim=1) ## torch.Size([64, 201, 768])
        # print("x_restored.shape final", x_restored.shape)

        middle_output = {"x_masked":x_masked, "mask":mask, "x_masked_b":x_masked_b}
        if self.model_args.vq_version != None:
            middle_output = {"x_masked":x_masked_vq, "mask":mask, "x_masked_b":x_masked_b, "min_encoding_indices":min_encoding_indices}
            middle_output["vq_loss"] = vq_loss

        if self.model_args.dataset_type == "3D":
            middle_output["shape1"] = shape1

        return x_restored, cls_tokens, middle_output
        ### return x, mask, ids_restore

    def forward_decoder(self, x_restored, cls_tokens, middle_output):
        # print("x_restored.shape, cls_tokens.shape decoder", x_restored.shape, cls_tokens.shape)
        # embed tokens # torch.Size([64, 200, 768])
        # if self.model_args.model == "mae_vit_base16_patch16" or self.model_args.model == "mae_vit_base16v2_patch16":
        #     x = [self.decoder_embed[i](x_restored)[0] for i in range(36)]
        #     x = torch.cat(x, 1)
        # else:
        x, _ = self.decoder_embed(x_restored)
        # print("x.shape token1", x.shape) # torch.Size([64, 196, 512])
        if self.model_args.dataset_type == "3D":
            x = x[:,:middle_output["shape1"],:]
        # print("decoder, x.shape", x.shape) # torch.Size([64, 196, 512])

        if self.model_args.arch_version == 'v11' or self.model_args.arch_version == 'v21' or self.model_args.arch_version == 'v31': ## only v11
            pass
        else:
            cls_tokens = self.decoder_embed_cls(cls_tokens)
            x = torch.cat((cls_tokens, x), dim=1) ## 197
            # print("x.shape in decoder 1", x.shape)

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
        if self.model_args.arch_version.startswith("v1"):
            for blk in self.decoder_blocks:
                x = blk(x)
            x = self.decoder_norm(x)
        elif self.model_args.arch_version.startswith("v2") or self.model_args.arch_version.startswith("v3"):
            x_ = self.decoder_blocks[0](x)
            for bi, blk in enumerate(self.decoder_blocks):
                if bi > 0:
                    x_ = blk(x_)
            x_ = self.decoder_norm(x_)
            x = x_+x
        # print("x.shape token2", x.shape)
        # print("decoder, x.shape5", x.shape) # torch.Size([64, 197, 512])

        if self.model_args.arch_version.startswith('v1'): ## only v1
            # predictor projection
            x = self.decoder_pred(x)
        else:                                             ## v2, v3...
            pass

        # print("x.shape token3", x.shape)
        # print("decoder, x.shape6", x.shape) # torch.Size([64, 197, 768])
        if self.model_args.arch_version == 'v11' or self.model_args.arch_version == 'v21' or self.model_args.arch_version == 'v31': ## only v11
            pass
        else:                                             ## v2, v3...
            #remove cls token
            x = x[:, self.model_args.cls_num:, :]
        # print("decoder, x.shape7", x.shape) # torch.Size([64, 196, 768])

        if self.model_args.arch_version.startswith('v1'): ## only v1
            pass
        else:                                             ## v2, v3...
            ##VQGAN decoder
            x = x.permute(0,2,1).contiguous().view(x.shape[0], self.embed_dim, self.hw_size, self.hw_size)
            # print("before VQ decoder, x.shape", x.shape) # torch.Size([64, 768, 14, 14])
            x = self.decoder(x)

        x = self.sigmoid(x)
        # print("decoder, x.shape finalfianl", x.shape) # torch.Size([64, 196, 768])
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
        if self.model_args.arch_version.startswith('v1'): ## only v1
            if self.model_args.dataset_type == "2D":
                pred = self.unpatchify(pred)
            elif self.model_args.dataset_type == "3D":
                # print("before unpatch pred.shape", pred.shape)
                pred = self.unpatchify3D(pred)
                # print("after unpatch pred.shape", pred.shape)
        else:                                             ## v2, v3...
            pass

        if "L2" in self.model_args.loss_version:
            # loss = (pred - target) ** 2
            # print("pred.shape, image_target.shape", pred.shape, image_target.shape)
            if self.model_args.model == "mae_vit_base16v2_patch16":
                image_target = image_target[:,:,:pred.shape[2],:,:]
                # print("pred.shape 2, image_target.shape 2", pred.shape, image_target.shape)
            loss = (pred - image_target) ** 2
            ### loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
            loss = loss.mean()

        if "L1" in self.model_args.loss_version:
            loss = torch.abs(pred - image_target)
            loss = loss.mean()

        if "LPIPS" in self.model_args.loss_version:
            if self.model_args.dataset_type == "2D":
                p_loss = self.perceptual_loss(image_target.contiguous(), pred.contiguous())
                p_loss = torch.mean(p_loss)
            elif self.model_args.dataset_type == "3D":
                for i_sl in range(image_target.shape[2]):
                    p_loss = 0
                    loss_count = 0
                    image_target_i = image_target[:,:,i_sl,:,:]
                    pred_i = pred[:,:,i_sl,:,:]
                    p_loss = p_loss + torch.mean(self.perceptual_loss(image_target_i.contiguous(), pred_i.contiguous()))
                    loss_count+=1
                p_loss = p_loss/loss_count

            # print("p_loss", p_loss.shape)
        else:
            p_loss = torch.tensor([0.0])
        # print("loss", loss)

        ### loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss, (p_loss)

    def forward(self, imgs, mask_ratio=0.75, middle=None): ## imgs == combined latent, when args.decoder_only
        image_target, random_selected_class = middle["image_target"], middle["random_selected_class"]
        # middle["organ_token_total"] = 1*args.token_factor*1 + args.token_factor*args.num_classes ## 20+180 = 200
        # middle["organ_token_selet"] = args.token_factor*len(random_selected_class) ## 100
        # print('middle["organ_token_total"], middle["organ_token_selet"]', middle["organ_token_total"], middle["organ_token_selet"])
        # print("imgs.shape, mask_ratio", imgs.shape, mask_ratio) ## torch.Size([64, 3, 224, 224]) 0.75
        # if args.decoder_only == False:
        ### latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, middle = middle)
        x_restored, cls_tokens, middle_output = self.forward_encoder(imgs, mask_ratio, middle = middle)
        # print("latent.shape, mask.shape", latent.shape, mask.shape) # torch.Size([64, 50, 768]) torch.Size([64, 196])
        ### pred = self.forward_decoder(latent) #, ids_restore)  # [N, L, p*p*3] ## this latent should be 197,768 after OrganFuse (TokenFuse)
        pred = self.forward_decoder(x_restored, cls_tokens, middle_output)
        # print("pred.shape", pred.shape) # torch.Size([64, 196, 768])
        loss, (p_loss) = self.forward_loss(image_target, pred) #, mask)

        if "LPIPS" in self.model_args.loss_version:
            middle_output["p_loss"] = p_loss

        if self.model_args.arch_version.startswith('v1'): ## only v1
            if self.model_args.dataset_type == "2D":
                pred = self.unpatchify(pred)
            elif self.model_args.dataset_type == "3D":
                pred = self.unpatchify3D(pred)
        else:                                             ## v2, v3...
            pass
        return loss, pred, middle_output


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=768, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base4_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768*4, depth=12, num_heads=12,
        decoder_embed_dim=768*4, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# def mae_vit_base16_patch16_dec512d8b(**kwargs):
#     model = MaskedAutoencoderViT(
#         patch_size=16, embed_dim=768, depth=12, num_heads=12,
#         decoder_embed_dim=768, decoder_depth=8, decoder_num_heads=16,
#         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model

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
mae_vit_base4_patch16 = mae_vit_base4_patch16_dec512d8b  # decoder: 512 dim, 8 blocks ## only 3D
mae_vit_base16_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks ## only 3D
mae_vit_base16v2_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks ## only 3D
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
