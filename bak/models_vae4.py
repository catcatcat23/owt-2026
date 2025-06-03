from functools import partial
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.dynunet_block import get_conv_layer as Conv
from monai.networks.layers.utils import get_norm_layer, get_dropout_layer
from monai.utils import ensure_tuple_rep
from monai.networks.blocks.basic_block import UnetBasicBlock, UnetResBlock, BasicUp, BasicDown
from monai.networks.blocks.selfattention import compute_attention, zero_module

class GEGLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels) 
        self.proj = nn.Linear(in_channels, out_channels*2, bias=True)

    def forward(self, x):
        # x expected to be [B, C, *] 
        # Workaround as layer norm can't currently be applied on arbitrary dimension: https://github.com/pytorch/pytorch/issues/71465
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1).transpose(1, 2) # -> [B, C, N] -> [B, N, C]
        x = self.norm(x)
        x, gate = self.proj(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        return x.transpose(1, 2).reshape(b, -1, *spatial) # -> [B, C, N] -> [B, C, *]

class BasicTransformerBlock(nn.Module):
    def __init__(
        self, 
        spatial_dims,
        in_channels, 
        out_channels, # WARNING: if out_channels != in_channels, skip connection is disabled 
        num_heads, 
        ch_per_head=32,
        norm_name=("GROUP", {'num_groups':32, "affine": True}),
        dropout=None, 
        emb_dim=None
    ):
        super().__init__()
        self.self_atn = LinearTransformer(spatial_dims, in_channels, in_channels, num_heads, ch_per_head, norm_name, dropout, None)
        if emb_dim is not None:  
            self.cros_atn = LinearTransformer(spatial_dims, in_channels, in_channels, num_heads, ch_per_head, norm_name, dropout, emb_dim)
        self.proj_out = nn.Sequential(
            GEGLU(in_channels, in_channels*4),
            nn.Identity() if dropout is None else get_dropout_layer(name=dropout, dropout_dim=spatial_dims),
            Conv["conv", spatial_dims](in_channels*4, out_channels, 1, bias=True)
        )
        
 
    def forward(self, x, embedding=None):
        # x expected to be [B, C, *]  and embedding is None or [B, C*] or [B, C*, *]
        x = self.self_atn(x)
        if embedding is not None:
            x = self.cros_atn(x, embedding=embedding)
        out = self.proj_out(x)
        if out.shape[1] == x.shape[1]:
            return out + x
        return x


class DownBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        downsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str],
        dropout: Optional[Union[Tuple, str, float]] = None,
        use_res_block: bool = False,
        learnable_interpolation: bool = True,
        use_attention: str = 'linear',
        emb_channels: int = None
    ):
        super(DownBlock, self).__init__()
        enable_down = ensure_tuple_rep(stride, spatial_dims) != ensure_tuple_rep(1, spatial_dims)
        down_out_channels = out_channels if learnable_interpolation and enable_down else in_channels
      
        # -------------- Down ----------------------
        self.down_op = BasicDown(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=downsample_kernel_size,
            stride=stride,
            learnable_interpolation=learnable_interpolation,
            use_res=False
        ) if enable_down else nn.Identity()
       

        # ---------------- Attention -------------
        self.attention = Attention(
            spatial_dims=spatial_dims,
            in_channels=down_out_channels,
            out_channels=down_out_channels,
            num_heads=8,
            ch_per_head=down_out_channels//8,
            depth=1,
            norm_name=norm_name,
            dropout=dropout,
            emb_dim=emb_channels,
            attention_type=use_attention
        )
       
        # -------------- Convolution ----------------------
        ConvBlock = UnetResBlock if use_res_block else UnetBasicBlock
        self.conv_block = ConvBlock(
            spatial_dims,
            down_out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            norm_name=norm_name,
            act_name=act_name,
            emb_channels=emb_channels 
        )

        
    def forward(self, x, emb=None):  
        # ----------- Down ---------
        x = self.down_op(x)
        
        # ----------- Attention -------------
        if self.attention is not None: 
            x = self.attention(x, emb) 

        # ------------- Convolution --------------
        x = self.conv_block(x, emb)

        return x


class UpBlock(nn.Module):
    def __init__(
        self, 
        spatial_dims, 
        in_channels: int, 
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str],
        dropout: Optional[Union[Tuple, str, float]] = None,
        use_res_block: bool = False,
        learnable_interpolation: bool = True,
        use_attention: str = 'linear',
        emb_channels: int = None, 
        skip_channels: int = 0
    ):
        super(UpBlock, self).__init__()
        enable_up = ensure_tuple_rep(stride, spatial_dims) != ensure_tuple_rep(1, spatial_dims)
        skip_out_channels = out_channels if learnable_interpolation and enable_up else in_channels+skip_channels
        self.learnable_interpolation = learnable_interpolation     
        

        # -------------- Up ----------------------
        self.up_op = BasicUp(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=upsample_kernel_size,
            stride=stride,
            learnable_interpolation=learnable_interpolation,
            use_res=False
        ) if enable_up else nn.Identity()

        # ---------------- Attention -------------
        self.attention = Attention(
            spatial_dims=spatial_dims,
            in_channels=skip_out_channels,
            out_channels=skip_out_channels,
            num_heads=8,
            ch_per_head=skip_out_channels//8,
            depth=1,
            norm_name=norm_name,
            dropout=dropout,
            emb_dim=emb_channels,
            attention_type=use_attention
        )

    
        # -------------- Convolution ----------------------
        ConvBlock = UnetResBlock if use_res_block else UnetBasicBlock
        self.conv_block = ConvBlock(
            spatial_dims,
            skip_out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            norm_name=norm_name,
            act_name=act_name,
            emb_channels=emb_channels
        )



    def forward(self, x_enc, x_skip=None, emb=None): 
        # ----------- Up -------------
        x = self.up_op(x_enc)

        # ----------- Skip Connection ------------
        if x_skip is not None:
            if self.learnable_interpolation: # Channel of x_enc and x_skip are equal and summation is possible 
                x = x+x_skip
            else:
                x = torch.cat((x, x_skip), dim=1)

        # ----------- Attention -------------
        if self.attention is not None: 
            x = self.attention(x, emb)      

        # ----------- Convolution ------------
        x = self.conv_block(x, emb)

        return x


class LinearTransformer(nn.Module):
    """ See LinearTransformer, however this implementation is fixed to Conv1d/Linear"""
    def __init__(
        self,
        spatial_dims,
        in_channels, 
        out_channels, # WARNING: if out_channels != in_channels, skip connection is disabled 
        num_heads, 
        ch_per_head=32, # rule of thumb: 32 or 64 channels per head (see stable-diffusion / diffusion models beat GANs)
        norm_name=("GROUP", {'num_groups':32, "affine": True}),
        dropout=None,
        emb_dim=None
    ):
        super().__init__()
        hid_channels = num_heads*ch_per_head
        self.num_heads = num_heads
        self.scale = ch_per_head**-0.25 # Should be 1/sqrt("queries and keys of dimension"), Note: additional sqrt needed as it follows OpenAI:  (q * scale) * (k * scale) instead of (q *k) * scale 
        
        self.norm_x = get_norm_layer(norm_name, spatial_dims=spatial_dims, channels=in_channels)
        emb_dim = in_channels if emb_dim is None else emb_dim

        # Note: Conv1d and Linear are interchangeable but order of input changes [B, C, N] <-> [B, N, C]
        self.to_q = nn.Conv1d(in_channels, hid_channels, 1) 
        self.to_k = nn.Conv1d(emb_dim, hid_channels, 1) 
        self.to_v = nn.Conv1d(emb_dim, hid_channels, 1)
        # self.to_qkv = nn.Conv1d(emb_dim, hid_channels*3, 1)

        self.to_out = nn.Sequential(
            zero_module(nn.Conv1d(hid_channels, out_channels, 1)),
            nn.Identity() if dropout is None else get_dropout_layer(name=dropout, dropout_dim=spatial_dims)
        )

    def forward(self, x, embedding=None):
        # x expected to be [B, C, *]  and embedding is None or [B, C*] or [B, C*, *]
        # if no embedding is given, cross-attention defaults to self-attention 
        
        # Normalize  
        b, c, *spatial = x.shape
        x_n = self.norm_x(x)

        # Attention:  embedding (cross-attention) or x (self-attention)
        if embedding is None:
            embedding = x_n  # WARNING: This assumes that emb_dim==in_channels 
        else:
            if embedding.ndim == 2:
                embedding = embedding.reshape(*embedding.shape[:2], *[1]*(x.ndim-2)) # [B, C*] -> [B, C*, *] 
            # Why no normalization for embedding here?

        # Flatten 
        x_n = x_n.reshape(b, c, -1)                              # [B, C,  *] -> [B, C,  N] 
        embedding = embedding.reshape(*embedding.shape[:2], -1)  # [B, C*, *] -> [B, C*, N'] 

        # Convolution 
        q = self.to_q(x_n)       # -> [B, (Heads x Dim_per_head), N] 
        k = self.to_k(embedding) # -> [B, (Heads x Dim_per_head), N'] 
        v = self.to_v(embedding) # -> [B, (Heads x Dim_per_head), N'] 
        # qkv = self.to_qkv(x_n)
        # q,k,v = qkv.split(qkv.shape[1]//3, dim=1)

        # Apply attention
        out = compute_attention(q, k, v, self.num_heads, self.scale)
        
        out = self.to_out(out)                      # -> [B, C', N]
        out = out.reshape(*out.shape[:2], *spatial) # -> [B, C', *]

        if x.shape == out.shape:
            out = x + out 
        return out # [B, C', *]


class SpatialTransformer(nn.Module):
    """ Proposed here: https://github.com/CompVis/stable-diffusion/blob/69ae4b35e0a0f6ee1af8bb9a5d0016ccb27e36dc/ldm/modules/attention.py#L218 
        Unrelated to: https://arxiv.org/abs/1506.02025  
    """
    def __init__(
        self, 
        spatial_dims,
        in_channels, 
        out_channels, # WARNING: if out_channels != in_channels, skip connection is disabled 
        num_heads, 
        ch_per_head=32, # rule of thumb: 32 or 64 channels per head (see stable-diffusion / diffusion models beat GANs)
        norm_name = ("GROUP", {'num_groups':32, "affine": True}),
        dropout=None, 
        emb_dim=None,
        depth=1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.norm = get_norm_layer(norm_name, spatial_dims=spatial_dims, channels=in_channels)
        conv_class = Conv["conv", spatial_dims]
        hid_channels = num_heads*ch_per_head

        self.proj_in = conv_class(
            in_channels,
            hid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(spatial_dims, hid_channels, hid_channels, num_heads, ch_per_head, norm_name, dropout=dropout, emb_dim=emb_dim) 
            for _ in range(depth)]
        )

        self.proj_out = conv_class( # Note: zero_module is used in original code  
            hid_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x, embedding=None):
        # x expected to be [B, C, *]  and embedding is None or [B, C*] or [B, C*, *]
        # Note: if no embedding is given, cross-attention is disabled
        h = self.norm(x)
        h = self.proj_in(h) 

        for block in self.transformer_blocks:
            h = block(h, embedding=embedding)

        h = self.proj_out(h) # -> [B, C'', *] 
        if h.shape == x.shape:
            return h + x
        return h 


class Attention(nn.Module):
    def __init__(
        self, 
        spatial_dims,
        in_channels, 
        out_channels,  
        num_heads=8, 
        ch_per_head=32, # rule of thumb: 32 or 64 channels per head (see stable-diffusion / diffusion models beat GANs) 
        norm_name = ("GROUP", {'num_groups':32, "affine": True}),
        dropout=0, 
        emb_dim=None,
        depth=1,
        attention_type='linear'
    ) -> None:
        super().__init__()
        if attention_type == 'spatial':
            self.attention = SpatialTransformer(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                num_heads=num_heads,
                ch_per_head=ch_per_head,
                depth=depth,
                norm_name=norm_name,
                dropout=dropout,
                emb_dim=emb_dim 
            )
        elif attention_type == 'linear':
            self.attention = LinearTransformer(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                num_heads=num_heads,
                ch_per_head=ch_per_head,
                norm_name=norm_name,
                dropout=dropout,
                emb_dim=emb_dim
            )
       
    
    def forward(self, x, emb=None):
        if hasattr(self, 'attention'):
            return self.attention(x, emb)
        else:
            return x 


class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=1024):
        super().__init__()
        
        hid_chs = [32, 64, 128, 256, 512]
        kernel_sizes = [4, 4, 4, 4, 4]
        strides = [2, 2, 2, 2, 2]
        norm_name = ("GROUP", {'num_groups':8, "affine": True})
        act_name = ("ReLU", {})
        
        self.inc = DownBlock(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=hid_chs[0],
            kernel_size=kernel_sizes[0],
            stride=strides[0],
            downsample_kernel_size=kernel_sizes[0],
            norm_name=norm_name,
            act_name=act_name,
            use_res_block=True,
            learnable_interpolation=True,
            use_attention='linear',
            emb_channels=None
        )

        self.encoders = nn.ModuleList([
            DownBlock(
                spatial_dims=2,
                in_channels=hid_chs[i],
                out_channels=hid_chs[i+1],
                kernel_size=kernel_sizes[i+1],
                stride=strides[i+1],
                downsample_kernel_size=kernel_sizes[i+1],
                norm_name=norm_name,
                act_name=act_name,
                use_res_block=True,
                learnable_interpolation=True,
                use_attention='linear',
                emb_channels=None
            )
            for i in range(len(hid_chs)-1)
        ])

        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(512*7*7, latent_dim)
        self.fc_var = nn.Linear(512*7*7, latent_dim)

    def forward(self, x):
        x = self.inc(x)
        for encoder in self.encoders:
            x = encoder(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim=1024, out_channels=3):
        super().__init__()
        
        hid_chs = [512, 256, 128, 64, 32]
        kernel_sizes = [4, 4, 4, 4, 4] 
        strides = [2, 2, 2, 2, 2]
        norm_name = ("GROUP", {'num_groups':8, "affine": True})
        act_name = ("ReLU", {})

        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dim, 512*7*7),
            nn.ReLU()
        )

        self.decoders = nn.ModuleList([
            UpBlock(
                spatial_dims=2,
                in_channels=hid_chs[i],
                out_channels=hid_chs[i+1] if i < len(hid_chs)-1 else out_channels,
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                upsample_kernel_size=strides[i],
                norm_name=norm_name,
                act_name=act_name,
                use_res_block=True,
                learnable_interpolation=True,
                use_attention='linear',
                emb_channels=None,
                skip_channels=0
            )
            for i in range(len(hid_chs))
        ])

        self.final_act = nn.Sigmoid()

    def forward(self, x):
        x = self.decoder_linear(x)
        x = x.view(-1, 512, 7, 7)
        for decoder in self.decoders:
            x = decoder(x)
        x = self.final_act(x)
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
