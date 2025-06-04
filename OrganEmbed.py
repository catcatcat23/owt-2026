import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm

class OrganCollector(nn.Module):
    def __init__(self, embed_dim, hidden_dim, organ_token_total, hw_size):
        super(OrganCollector, self).__init__()
        self.organ_token_total = organ_token_total
        self.hw_size = hw_size
        self.embed_dim = embed_dim
        token_num = organ_token_total

        self.conv1 = nn.Conv2d(embed_dim,hidden_dim,1,padding=0, bias=False)
        self.conv2 = nn.Conv2d(hidden_dim,token_num,1,padding=0, bias=False)
        self.softmax = Softmax(dim=-1)
        self.conv3 = nn.Conv2d(embed_dim,hidden_dim,1,padding=0, bias=False)

    def forward(self, input_x):
        x = input_x
        x = x.permute(0,2,1).contiguous().view(x.shape[0], self.embed_dim, self.hw_size, self.hw_size)
        B, c, h, w = x.shape

        attn_ids = self.conv2(self.conv1(x))
        attn_ids = attn_ids.flatten(2)
        attention_probs = self.softmax(attn_ids)

        x = self.conv3(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)

        outputs = torch.einsum("...si,...id->...sd", attention_probs, x) 
        return outputs, attention_probs

class OrganCollector3D(nn.Module):
    def __init__(self, embed_dim, hidden_dim, organ_token_total, hw_size, model_args):
        super(OrganCollector3D, self).__init__()
        self.organ_token_total = organ_token_total
        self.hw_size = hw_size
        self.embed_dim = embed_dim
        self.model_args = model_args
        token_num = organ_token_total

        self.conv1 = nn.Conv3d(embed_dim,hidden_dim,(self.model_args.fix_frame//self.model_args.temp_stride,1,1),padding=(0,0,0), stride=(self.model_args.fix_frame//self.model_args.temp_stride,1,1), bias=False)
        self.conv2 = nn.Conv3d(hidden_dim,token_num,(1,1,1),padding=(0,0,0), stride=(1,1,1), bias=False)
        self.softmax = Softmax(dim=-1)
        self.conv3 = nn.Conv3d(embed_dim,hidden_dim,(self.model_args.fix_frame//self.model_args.temp_stride,1,1),padding=(0,0,0), stride=(self.model_args.fix_frame//self.model_args.temp_stride,1,1), bias=False)

    def forward(self, input_x):
        x = input_x

        x = x.permute(0,2,1).contiguous().view(x.shape[0], self.embed_dim, self.model_args.fix_frame//self.model_args.temp_stride, self.hw_size, self.hw_size)

        attn_ids = self.conv2(self.conv1(x))
        attn_ids = attn_ids.flatten(2)
        attention_probs = self.softmax(attn_ids)

        x = self.conv3(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2) 

        outputs = torch.einsum("...si,...id->...sd", attention_probs, x) 
        return outputs, attention_probs

class AHER(nn.Module):
    def __init__(self, input_dim, output_dim, organ_token_total, output_hw_size):
        super(AHER, self).__init__()

        self.sp_linear1 = nn.Linear(input_dim, output_hw_size, bias=False)
        self.sp_linear2 = nn.Linear(input_dim, output_dim, bias=False)
        self.softmax = Softmax(dim=-1)

    def forward(self, input_x):
        x = input_x
        attn_ids = self.sp_linear1(x)
        attn_ids = attn_ids.permute(0, 2, 1)

        attention_probs = self.softmax(attn_ids)

        x = self.sp_linear2(x)

        outputs = torch.einsum("...si,...id->...sd", attention_probs, x)
        return outputs, attention_probs
