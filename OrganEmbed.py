import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm

class OrganEmbed(nn.Module):
    # def __init__(self, config, step_num, vis = None):
    def __init__(self, embed_dim, hidden_dim, organ_token_total, hw_size):
        super(OrganEmbed, self).__init__()
        self.organ_token_total = organ_token_total
        self.hw_size = hw_size
        self.embed_dim = embed_dim
        token_num = organ_token_total

        # print("self.organ_token_total", self.organ_token_total)
        # print("self.hw_size", self.hw_size)

        self.conv1 = nn.Conv2d(embed_dim,hidden_dim,1,padding=0, bias=False) #groups=4, bias=False)
        self.conv2 = nn.Conv2d(hidden_dim,token_num,1,padding=0, bias=False) #groups=4, bias=False)
        self.softmax = Softmax(dim=-1)
        self.conv3 = nn.Conv2d(hidden_dim,hidden_dim,1,padding=0, bias=False) #groups=4, bias=False)

    def forward(self, input_x):
        x = input_x

        x = x.permute(0,2,1).contiguous().view(x.shape[0], self.embed_dim, self.hw_size, self.hw_size)
        # print("x.shape", x.shape) # torch.Size([64, 768, 14, 14])

        B, c, h, w = x.shape ## 2, 128, 128, 128 ; torch.Size([64, 197, 768])

        attn_ids = self.conv2(self.conv1(x)) ## 2, token_num(64), 128, 128 ; torch.Size([64, 200, 14, 14])
        # print("attn_ids.shape", attn_ids.shape)
        # forward the image model for token embeddings
        attn_ids = attn_ids.flatten(2) ## 2, token_num(64), 128*128 ; torch.Size([64, 200, 196])
        # print("attn_ids.shape", attn_ids.shape)
        attention_probs = self.softmax(attn_ids) ## 2, token_num(64), 128*128 ; 64, 200, 196

        x = self.conv3(x) ## 2, channel(128), 128, 128 ; torch.Size([64, 768, 14, 14])
        # print("x.shape2", x.shape)
        x = x.flatten(2) ## 2, channel(128), 128*128 ; torch.Size([64, 768, 196])
        # print("x.shape3", x.shape)
        x = x.transpose(-1, -2) ## 2, 128*128, channel(128) ; torch.Size([64, 196, 768])
        # print("x.shape4", x.shape)

        outputs = torch.einsum("...si,...id->...sd", attention_probs, x) ## 2, token_num(64), channel(128)
        # print("outputs.shape", outputs.shape) ## torch.Size([64, 200, 768])

        return outputs, attention_probs

# class OrganEmbed(nn.Module):
#     # def __init__(self, config, step_num, vis = None):
#     def __init__(self, input_dim, output_dim, organ_token_total, hw_size):
#         super(OrganEmbed, self).__init__()
#         token_num = organ_token_total

#         self.oe_linear1 = nn.Linear(input_dim, token_num, bias=False)
#         self.oe_linear2 = nn.Linear(input_dim, output_dim, bias=False)
#         self.softmax = Softmax(dim=-1)

#     def forward(self, input_x):
#         x = input_x ## 64, 196, 768

#         attn_ids = self.oe_linear1(x) ## 64, 196, 200
#         print("attn_ids.shape", attn_ids.shape)
#         attn_ids = attn_ids.permute(0,2,1) ## 64, 200, 196
#         print("attn_ids.shape", attn_ids.shape)
#         attention_probs = self.softmax(attn_ids) ## 2, token_num(64), 128*128 ; 64, 200, 196

#         x = self.oe_linear2(x) ## 64, 196, 768
#         print("x.shape2", x.shape)

#         outputs = torch.einsum("...si,...id->...sd", attention_probs, x) ## 2, token_num(64), channel(128)
#         print("outputs.shape", outputs.shape) ## torch.Size([64, 200, 768])

#         return outputs, attention_probs

class SpatialRestore(nn.Module):
    # def __init__(self, config, step_num, vis = None):
    def __init__(self, input_dim, output_dim, organ_token_total, hw_size):
        super(SpatialRestore, self).__init__()

        self.sp_linear1 = nn.Linear(input_dim, hw_size*hw_size, bias=False)
        self.sp_linear2 = nn.Linear(input_dim, output_dim, bias=False)
        self.softmax = Softmax(dim=-1)

    def forward(self, input_x):
        x = input_x # torch.Size([64, 200, 768])
        # print("x.shape SR1", x.shape)
        attn_ids = self.sp_linear1(x) # torch.Size([64, 200, 196])
        attn_ids = attn_ids.permute(0, 2, 1)
        # print("attn_ids.shape SR2", attn_ids.shape) # torch.Size([64, 196, 200])
        attention_probs = self.softmax(attn_ids) # torch.Size([64, 196, 200])

        x = self.sp_linear2(x) # torch.Size([64, 200, 512])
        # print("x.shape2 SR", x.shape)

        outputs = torch.einsum("...si,...id->...sd", attention_probs, x)
        # print("outputs.shape SR", outputs.shape) ## torch.Size([64, 196, 512])

        return outputs, attention_probs











