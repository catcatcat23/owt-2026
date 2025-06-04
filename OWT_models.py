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
from util.patch_embed import PatchEmbed3Dfix

from util.pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid
from OrganEmbed import OrganCollector, OrganCollector3D, AHER
from einops import rearrange

from timm.models.layers import DropPath ## for A100 and H100
# ~~from timm.models.layers import Mlp, DropPath ## for H100~~ deleted
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

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
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        ## LA
        dim = q.shape[-1]
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)
        q = q * dim ** -0.5
        context = torch.einsum('bhnd,bhne->bhde', k, v)
        x = torch.einsum('bhnd,bhde->bhne', q, context)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

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
        self.temp_stride = self.model_args.temp_stride
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        if self.model_args.arch_version.startswith("v1"):
            self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
            if self.model_args.dataset_type == "3D":
                self.patch_embed = PatchEmbed3Dfix(img_size, patch_size, in_chans, embed_dim, num_frames=self.model_args.fix_frame, temp_stride=self.model_args.temp_stride)
            num_patches = self.patch_embed.num_patches
    
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            if self.model_args.dataset_type == "2D":
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
            elif self.model_args.dataset_type == "3D":
                self.pos_embed_spatial = nn.Parameter(torch.zeros(1, self.patch_embed.grid_size[1] * self.patch_embed.grid_size[2], embed_dim))
                self.pos_embed_temporal = nn.Parameter(torch.zeros(1, self.patch_embed.grid_size[0], embed_dim))
    
                # Separable encoder positional embeddings
                self.decoder_pos_embed_class = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
                self.decoder_pos_embed_spatial = nn.Parameter(torch.zeros(1, self.patch_embed.grid_size[1] * self.patch_embed.grid_size[2], decoder_embed_dim))
                self.decoder_pos_embed_temporal = nn.Parameter(torch.zeros(1, self.patch_embed.grid_size[0], decoder_embed_dim))

            if self.model_args.LA:
                self.blocks1 = nn.ModuleList([
                    BlockLA(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) 
                    for i in range(int(depth/2))])
            else:
                self.blocks1 = nn.ModuleList([
                    Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) 
                    for i in range(int(depth/2))])
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
        if self.model_args.dataset_type == "2D":
            self.organ_embed = OrganCollector(embed_dim, embed_dim, self.model_args.organ_token_total, img_size//patch_size)
        elif self.model_args.dataset_type == "3D":
            self.organ_embed = OrganCollector3D(embed_dim, embed_dim, self.model_args.organ_token_total, img_size//patch_size, self.model_args)

        self.norm = norm_layer(embed_dim)
        self.sigmoid = Sigmoid()
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        if self.model_args.dataset_type == "2D":
            self.decoder_embed = AHER(embed_dim, decoder_embed_dim, self.model_args.organ_token_total, (img_size//patch_size)*(img_size//patch_size))
        elif self.model_args.dataset_type == "3D":
            if self.model_args.arch_version.startswith("v1"):
                self.decoder_embed = AHER(embed_dim, decoder_embed_dim, self.model_args.organ_token_total, self.patch_embed.grid_size[0]*self.patch_embed.grid_size[1]*self.patch_embed.grid_size[2])
            elif self.model_args.arch_version.startswith("v3"):
                self.decoder_embed = AHER(embed_dim, decoder_embed_dim, self.model_args.organ_token_total, (self.model_args.fix_frame//patch_size)*(img_size//patch_size)*(img_size//patch_size))

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if self.model_args.arch_version.startswith("v1"):
            pass
        else:
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
        if self.model_args.arch_version.startswith("v1"): 
            self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
            if self.model_args.dataset_type == "3D":
                self.decoder_pred = nn.Sequential(
                        nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True),
                        nn.Tanh(),
                        nn.Linear(patch_size**2 * in_chans, patch_size**2 * in_chans * self.temp_stride, bias=True))
        else:
            pass ## v3...

        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

        if "LPIPS" in self.model_args.loss_version:
            from VQ.lpips import LPIPS
            self.perceptual_loss = LPIPS().eval()
            # self.perceptual_weight = 1.0

        if self.model_args.text_encoding != "None":
            self.text_linear = nn.Linear(768, embed_dim)

    def initialize_weights(self):
        # initialization
        if self.model_args.arch_version.startswith("v1"):
            # initialize (and freeze) pos_embed by sin-cos embedding
            if self.model_args.dataset_type == "2D":
                pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
                self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
            elif self.model_args.dataset_type == "3D":
                # torch.nn.init.normal_(self.pos_embed_spatial, std=.02)
                pos_embed_spatial = get_2d_sincos_pos_embed(self.pos_embed_spatial.shape[-1], int(self.patch_embed.grid_size[1]), cls_token=False)
                self.pos_embed_spatial.data.copy_(torch.from_numpy(pos_embed_spatial).float().unsqueeze(0))
                torch.nn.init.normal_(self.pos_embed_temporal, std=.02)

                torch.nn.init.normal_(self.decoder_pos_embed_class, std=.02)
                # torch.nn.init.normal_(self.decoder_pos_embed_spatial, std=.02)
                decoder_pos_embed_spatial = get_2d_sincos_pos_embed(self.decoder_pos_embed_spatial.shape[-1], int(self.patch_embed.grid_size[1]), cls_token=False)
                self.decoder_pos_embed_spatial.data.copy_(torch.from_numpy(decoder_pos_embed_spatial).float().unsqueeze(0))
                torch.nn.init.normal_(self.decoder_pos_embed_temporal, std=.02)

                print("self.pos_embed_spatial.shape, self.pos_embed_temporal.shape, self.decoder_pos_embed_spatial.shape, self.decoder_pos_embed_temporal.shape", self.pos_embed_spatial.shape, self.pos_embed_temporal.shape, self.decoder_pos_embed_spatial.shape, self.decoder_pos_embed_temporal.shape)

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
        if self.model_args.dataset_type == "2D":
            p = self.patch_embed.patch_size[0]
        elif self.model_args.dataset_type == "3D":
            p = self.patch_size
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

        return x_masked, mask

    def token_restore(self, x_masked, mask):
        B, L, C = x_masked.shape
        x_restored = torch.zeros((B, self.model_args.organ_token_total, C), device=x_masked.device, dtype=x_masked.dtype)

        masked_indices = torch.nonzero(mask == 0, as_tuple=False)[:, 1].reshape(B, -1)
        mask_tokens = self.mask_token.repeat(x_restored.shape[0], x_restored.shape[1] - x_masked.shape[1], 1)
        x_restored = x_restored.scatter(1, masked_indices.unsqueeze(-1).expand(-1, -1, C), mask_tokens.to(x_restored.dtype))

        unmasked_indices = torch.nonzero(mask == 1, as_tuple=False)[:, 1].reshape(B, -1)
        x_restored = x_restored.scatter(1, unmasked_indices.unsqueeze(-1).expand(-1, -1, C), x_masked)

        return x_restored

    def token_restore_sup(self, x_masked, mask, x_masked2):
        B, L, C = x_masked.shape
        x_restored = torch.zeros((B, self.model_args.organ_token_total, C), device=x_masked.device, dtype=x_masked.dtype)

        masked_indices = torch.nonzero(mask == 0, as_tuple=False)[:, 1].reshape(B, -1)
        mask_tokens = x_masked2
        x_restored = x_restored.scatter(1, masked_indices.unsqueeze(-1).expand(-1, -1, C), mask_tokens.to(x_restored.dtype))

        unmasked_indices = torch.nonzero(mask == 1, as_tuple=False)[:, 1].reshape(B, -1)
        x_restored = x_restored.scatter(1, unmasked_indices.unsqueeze(-1).expand(-1, -1, C), x_masked)

        return x_restored

    def forward_encoder(self, x, mask_ratio, middle = None):
        image_target, random_selected_class = middle["image_target"], middle["random_selected_class"]
        #### Encoder ####
        if self.model_args.arch_version.startswith("v1"):
            x = self.patch_embed(x)
    
            if self.model_args.dataset_type == "3D":
                self.pos_embed = self.pos_embed_spatial.repeat(1, self.patch_embed.grid_size[0], 1) + \
                            torch.repeat_interleave(self.pos_embed_temporal, self.patch_embed.grid_size[1] * self.patch_embed.grid_size[2], dim=1)
                self.pos_embed = torch.cat([self.cls_token, self.pos_embed], 1)
                self.pos_embed = self.pos_embed[:,:x.shape[1]+1,:]
            x = x + self.pos_embed[:, 1:, :]
    
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
    
            # apply Transformer blocks
            for bi, blk in enumerate(self.blocks1):
                x = blk(x)
            x = self.norm(x)
    
            cls_tokens = x[:,:1,:]
            x = x[:,1:,:]
        elif self.model_args.arch_version.startswith("v3"):
            x_ = self.encoder(x)
            cls_tokens = self.avg_pool(x_)
            cls_tokens = cls_tokens.flatten(2)
            cls_tokens = cls_tokens.transpose(-1, -2)
            x_ = x_.flatten(2) 
            x_ = x_.transpose(-1, -2)
            x = self.encoder_embed(x_)
        
        #### OrganCollector ####
        x, _ = self.organ_embed(x)

        if self.model_args.text_encoding != "None":
            text_features = self.text_linear(middle["text_features"])
            x = x + text_features

        ## random mask organ tokens
        x_masked_b, mask = self.random_masking(x, random_selected_class) 

        #### Token Group Encoder ####
        x_masked_ = self.blocks2[0](x_masked_b)
        for bi, blk in enumerate(self.blocks2):
            if bi > 0:
                x_masked_ = blk(x_masked_)
        x_masked_ = self.norm(x_masked_)
        x_masked_ = x_masked_b + x_masked_
        x_masked = x_masked_
        x_restored = x_masked 

        middle_output = {"x_masked":x_masked, "mask":mask, "x_masked_b":x_masked_b}
        return x_restored, cls_tokens, middle_output

    def forward_decoder(self, x_restored, cls_tokens, middle_output):

        #### AHER ####
        x, _ = self.decoder_embed(x_restored)

        #### Decoder ####
        if self.model_args.arch_version.startswith("v1"): ## not for CNN decoder
            if self.model_args.dataset_type == "3D":
                # add pos embed
                pos_embed = self.decoder_pos_embed_spatial.repeat(1, self.patch_embed.grid_size[0], 1) + \
                            torch.repeat_interleave(self.decoder_pos_embed_temporal, self.patch_embed.grid_size[1] * self.patch_embed.grid_size[2], dim=1)
                x = x + pos_embed

        # apply Transformer blocks
        if self.model_args.arch_version.startswith("v1"):
            for blk in self.decoder_blocks:
                x = blk(x)
            x = self.decoder_norm(x)
        elif self.model_args.arch_version.startswith("v3"):
            x_ = self.decoder_blocks[0](x)
            for bi, blk in enumerate(self.decoder_blocks):
                if bi > 0:
                    x_ = blk(x_)
            x_ = self.decoder_norm(x_)
            x = x_+x

        if self.model_args.arch_version.startswith('v1'): ## only v1
            # predictor projection
            x = self.decoder_pred(x)
        else:                                             ## v3...
            pass

        if self.model_args.arch_version.startswith('v1'): ## only v1
            pass
        else:                                             ## v3...
            ##VQGAN decoder
            x = x.permute(0,2,1).contiguous().view(x.shape[0], self.embed_dim, self.hw_size, self.hw_size)
            x = self.decoder(x)

        x = self.sigmoid(x)
        return x

    def forward_loss(self, image_target, pred):

        if self.model_args.arch_version.startswith('v1'): ## only v1
            if self.model_args.dataset_type == "2D":
                pred = self.unpatchify(pred)
            elif self.model_args.dataset_type == "3D":
                pred = self.unpatchify3D(pred)
        else:                                             ## v3...
            pass

        if "L2" in self.model_args.loss_version:
            loss = (pred - image_target) ** 2
            loss = loss.mean()

        if "L1" in self.model_args.loss_version:
            loss = torch.abs(pred - image_target)
            loss = loss.mean()

        if "LPIPS" in self.model_args.loss_version:
            if self.model_args.dataset_type == "2D":
                p_loss = self.perceptual_loss(image_target.contiguous(), pred.contiguous())
                p_loss = torch.mean(p_loss)
            elif self.model_args.dataset_type == "3D":
                p_loss = 0
                loss_count = 0
                for i_sl in range(image_target.shape[2]):
                    image_target_i = image_target[:,:,i_sl,:,:]
                    pred_i = pred[:,:,i_sl,:,:]
                    p_loss = p_loss + torch.mean(self.perceptual_loss(image_target_i.contiguous(), pred_i.contiguous()))
                    loss_count+=1
                p_loss = p_loss/loss_count
        else:
            p_loss = torch.tensor([0.0])
        return loss, (p_loss)

    def forward(self, imgs, mask_ratio=0.75, middle=None):
        image_target, random_selected_class = middle["image_target"], middle["random_selected_class"]
        x_restored, cls_tokens, middle_output = self.forward_encoder(imgs, mask_ratio, middle = middle)
        pred = self.forward_decoder(x_restored, cls_tokens, middle_output)
        loss, (p_loss) = self.forward_loss(image_target, pred) #, mask)

        if "LPIPS" in self.model_args.loss_version:
            middle_output["p_loss"] = p_loss

        if self.model_args.arch_version.startswith('v1'): ## only v1
            if self.model_args.dataset_type == "2D":
                pred = self.unpatchify(pred)
            elif self.model_args.dataset_type == "3D":
                pred = self.unpatchify3D(pred)
        else:                                             ## v3...
            pass
        return loss, pred, middle_output


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
mae_vit_basefix16_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks ## only 3D

mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
