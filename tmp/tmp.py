import torch
import random

image = torch.Tensor([[[[1,2,3,4], [5,4,3,2], [5,6,0,1]], [[4,3,2,1], [2,3,9,5], [1,0,6,5]]], [[[2,3,4,5], [6,4,9,3], [6,7,9,0]], [[5,4,3,2], [9,3,9,5], [9,0,6,5]]]])

label = torch.Tensor([[[[1,2,3,4], [5,4,3,2], [5,6,0,1]], [[4,3,2,1], [2,3,9,5], [1,0,6,5]]], [[[2,3,4,5], [6,4,9,3], [6,7,9,0]], [[5,4,3,2], [9,3,9,5], [9,0,6,5]]]])
print("label.shape", label.shape)

num_classes = 9
mask_ratio = 0.5

mask_ratio_class = mask_ratio
class_list = list(range(num_classes+1))
random.shuffle(class_list)
print(class_list)
random_selected_class = class_list[:int((num_classes+1)*mask_ratio_class)]
print(random_selected_class)

# Generate random class index for each sample in the batch
for ms in random_selected_class:
    image[label==ms] = 0


import torch
import torch.nn as nn
import numpy as np

## ori mae method
mask_ratio = 0.5
# x=torch.rand([3,6,9])
x = torch.Tensor(np.array([i for i in range(216)]).reshape((3,8,9)))
N, L, D = x.shape
len_keep = int(L * (1 - mask_ratio))
noise = torch.rand(N, L)

ids_shuffle = torch.argsort(noise, dim=1)
ids_restore = torch.argsort(ids_shuffle, dim=1)

ids_keep = ids_shuffle[:, :len_keep]
x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

# generate the binary mask: 0 is keep, 1 is remove
mask = torch.ones([N, L], device=x.device)
mask[:, :len_keep] = 0
# unshuffle to get the binary mask
mask = torch.gather(mask, dim=1, index=ids_restore)

x=x_masked
mask_token = nn.Parameter(torch.zeros(1, 1, 9))
mask_tokens = mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)

## token_mae method
x = torch.Tensor(np.array([i for i in range(216)]).reshape((3,8,9)))
B, L, C = x.shape  # batch, length, channels
num_classes = 3
selected_classes = [0,2]
tokens_per_class = L // num_classes

mask = torch.ones([B, L], device=x.device)  # initialize mask with ones

for cls in selected_classes:
    start_idx = cls * tokens_per_class
    end_idx = (cls + 1) * tokens_per_class
    mask[:, start_idx:end_idx] = 0  # mask the selected class tokens

# Create ids_restore to restore the original order
# ids_restore = torch.argsort(mask, dim=1, descending=True)

# Keep the unmasked tokens
x_masked = x[mask == 1].reshape(B, -1, C)
ids_restore = torch.nonzero(mask == 1, as_tuple=False)#.reshape(B, -1)

x_restored = torch.zeros((B, L, C), device=x_masked.device)
unmasked_indices = torch.nonzero(mask == 1, as_tuple=False)[:, 1].reshape(B, -1)
x_restored = x_restored.scatter(1, unmasked_indices.unsqueeze(-1).expand(-1, -1, C), x_masked)







