import torch
import random

image = torch.Tensor([[[[1,2,3,4], [5,4,3,2], [5,6,0,1]], [[4,3,2,1], [2,3,10,5], [1,0,6,5]]], [[[2,3,4,5], [6,4,11,3], [6,7,12,0]], [[5,4,3,2], [12,3,11,5], [10,0,6,5]]]])

label = torch.Tensor([[[[1,2,3,4], [5,4,3,2], [5,6,0,1]], [[4,3,2,1], [2,3,10,5], [1,0,6,5]]], [[[2,3,4,5], [6,4,11,3], [6,7,12,0]], [[5,4,3,2], [12,3,11,5], [10,0,6,5]]]])
print("label.shape", label.shape)

num_classes = 12
mask_ratio = 0.5

mask_ratio_class = mask_ratio
class_list = list(range(num_classes))
random.shuffle(class_list)
print(class_list)
random_selected_class = class_list[:int(num_classes*mask_ratio_class)]
print(random_selected_class)

# Generate random class index for each sample in the batch
for ms in random_selected_class:
    image[label==ms] = 0