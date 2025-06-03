import torch
import torch.nn as nn
from thop import profile
from thop import clever_format
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
import time

from OWC2_LIB import BlockLA
from timm.models.vision_transformer import PatchEmbed, Block

def cal_model_params_flops_memory(model, input_shape=(1,196,768)):
    """Calculate parameters, FLOPs and memory usage for a model
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (batch_size, channels, height, width)
        
    Returns:
        params: Number of parameters (in M)
        flops: Number of GFLOPs
        memory: Memory usage in MB
    """
    # Create dummy input
    x = torch.randn(input_shape)
    
    # Calculate parameters
    params = sum(p.numel() for p in model.parameters()) / (10**6) # Convert to M
    
    # Calculate FLOPs
    flops, _ = profile(model, inputs=(x,))
    flops = flops / (10**9) # Convert to GFLOPs
    
    # Calculate memory usage
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    if torch.cuda.is_available():
        x = x.cuda()
        model = model.cuda()
        
    # Warm up
    with torch.no_grad():
        for _ in range(2):
            _ = model(x)
            
    # Measure memory
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    with torch.no_grad():
        _ = model(x)
    
    torch.cuda.synchronize()
    memory = torch.cuda.max_memory_allocated() / (1024**2) # Convert to MB
    
    # Format numbers
    params = f"{params:.3f}M"
    flops = f"{flops:.3f}G"
    
    return params, flops, memory

class BlockWrapper(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class OrganEmbed(nn.Module):
    def __init__(self, embed_dim, hidden_dim, organ_token_total, hw_size):
        super(OrganEmbed, self).__init__()
        self.organ_token_total = organ_token_total
        self.hw_size = hw_size
        self.embed_dim = embed_dim
        token_num = organ_token_total

        self.conv1 = nn.Conv2d(embed_dim, hidden_dim, 1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(hidden_dim, token_num, 1, padding=0, bias=False)
        self.softmax = Softmax(dim=-1)
        self.conv3 = nn.Conv2d(embed_dim, hidden_dim, 1, padding=0, bias=False)

    def forward(self, input_x):
        x = input_x
        print("----------------x.shape OE1", x.shape)
        x = x.permute(0,2,1).contiguous().view(x.shape[0], self.embed_dim, self.hw_size, self.hw_size)
        print("x.shape OE2", x.shape)
        B, c, h, w = x.shape

        attn_ids = self.conv2(self.conv1(x))
        print("attn_ids.shape OE3", attn_ids.shape)
        attn_ids = attn_ids.flatten(2)
        print("attn_ids.shape OE4", attn_ids.shape)
        attention_probs = self.softmax(attn_ids)
        print("attention_probs.shape OE5", attention_probs.shape)

        x = self.conv3(x)
        print("x.shape OE6", x.shape)
        x = x.flatten(2)
        print("x.shape OE7", x.shape)
        x = x.transpose(-1, -2)
        print("x.shape OE8", x.shape)

        outputs = torch.einsum("...si,...id->...sd", attention_probs, x)
        print("outputs.shape OE9", outputs.shape)

        return outputs, attention_probs

class SpatialRestore(nn.Module):
    # def __init__(self, config, step_num, vis = None):
    def __init__(self, input_dim, output_dim, organ_token_total, output_hw_size):
        super(SpatialRestore, self).__init__()

        self.sp_linear1 = nn.Linear(input_dim, output_hw_size, bias=False)
        self.sp_linear2 = nn.Linear(input_dim, output_dim, bias=False)
        self.softmax = Softmax(dim=-1)

    def forward(self, input_x):
        x = input_x # torch.Size([64, 200, 768])
        print("--------------------------------x.shape SR1", x.shape)
        attn_ids = self.sp_linear1(x) # torch.Size([64, 200, 196])
        attn_ids = attn_ids.permute(0, 2, 1)
        print("attn_ids.shape SR2", attn_ids.shape) # torch.Size([64, 196, 200])
        attention_probs = self.softmax(attn_ids) # torch.Size([64, 196, 200])

        x = self.sp_linear2(x) # torch.Size([64, 200, 512])
        print("x.shape2 SR", x.shape)

        outputs = torch.einsum("...si,...id->...sd", attention_probs, x)
        print("outputs.shape SR", outputs.shape) ## torch.Size([64, 196, 512])

        return outputs, attention_probs

if __name__ == "__main__":
    # Example usage for BlockLA
    embed_dim = 768
    num_heads = 12 # 12 (encoder), 16 (decoder)
    mlp_ratio = 4
    depth = 6 # 12 (encoder), 6 (decoder), 8 (decoder)
    norm_layer = nn.LayerNorm
    print("embed_dim, num_heads, mlp_ratio, depth", embed_dim, num_heads, mlp_ratio, depth)
    blocks = nn.ModuleList([
        BlockLA(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
        for i in range(int(depth))
    ])
    # blocks = nn.ModuleList([
    #     Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
    #     for i in range(int(depth))
    # ])
    
    model = BlockWrapper(blocks)
    
    # Calculate metrics
    params, flops, memory = cal_model_params_flops_memory(model, input_shape=(1,20,768))
    
    print(f"Parameters: {params}")
    print(f"FLOPs: {flops}") 
    print(f"Memory: {memory:.2f} MB")

    # Example usage for OrganEmbed
    organ_embed = OrganEmbed(
        embed_dim=768,
        hidden_dim=768,
        organ_token_total=200,
        hw_size=14
    )
    
    # Calculate metrics for OrganEmbed
    params2, flops2, memory2 = cal_model_params_flops_memory(organ_embed, input_shape=(1,196,768))
    
    print("\nOrganEmbed Metrics:")
    print(f"Parameters: {params2}")
    print(f"FLOPs: {flops2}")
    print(f"Memory: {memory2:.2f} MB")

    # Example usage for SpatialRestore
    spatial_restore = SpatialRestore(
        input_dim=768,
        output_dim=768,
        organ_token_total=200,
        output_hw_size=196
    )
    
    # Calculate metrics for SpatialRestore
    params3, flops3, memory3 = cal_model_params_flops_memory(spatial_restore, input_shape=(1,200,768))
    
    print("\nSpatialRestore Metrics:")
    print(f"Parameters: {params3}")
    print(f"FLOPs: {flops3}")
    print(f"Memory: {memory3:.2f} MB")
