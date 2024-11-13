import os
import random
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from einops import repeat
from icecream import ic
import pandas as pd
import pickle
import PIL.Image
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
import cv2
import re
from einops import rearrange

def sorted_nicely( l ): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


def read_image(path):
    with open(path, 'rb') as file:
        img = pickle.load(file)
        return img

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    # image = np.rot90(image, k, axes=(0, 1))
    # label = np.rot90(label, k, axes=(0, 1))
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-15, 15)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

def convert_to_PIL(img: np.array) -> PIL.Image:
    '''
    img should be normalized between 0 and 1
    '''
    img = np.clip(img, 0, 1)
    return PIL.Image.fromarray((img * 255).astype(np.uint8))

def convert_to_PIL_label(label):
    return PIL.Image.fromarray(label.astype(np.uint8))

def convert_to_np(img: PIL.Image) -> np.array:
    return np.array(img).astype(np.float32) / 255

def convert_to_np_label(label):
    return np.array(label).astype(np.float32)

def random_erasing(
    imgs,
    label,
    scale_z=(0.02, 0.33),
    scale=(0.02, 0.05),
    ratio=(0.3, 3.3),
    apply_all: int = 0,
    rng: np.random.Generator = np.random.default_rng(0),
):

    # determine the box
    imgshape = imgs.shape
    
    # nx and ny
    while True:
        se = rng.uniform(scale[0], scale[1]) * imgshape[0] * imgshape[1]
        re = rng.uniform(ratio[0], ratio[1])
        nx = int(np.sqrt(se * re))
        ny = int(np.sqrt(se / re))
        if nx < imgshape[1] and ny < imgshape[0]:
            break

    # determine the position of the box
    sy = rng.integers(0, imgshape[0] - ny + 1)
    sx = rng.integers(0, imgshape[1] - nx + 1)

    # print(nz, ny, nx, sz, sy, sx)
    filling = rng.uniform(0, 1, size=[ny, nx])
    filling = filling[:,:,np.newaxis]
    filling = np.repeat(filling, imgshape[-1], axis=-1)

    # erase
    imgs[sy:sy + ny, sx:sx + nx, :] = filling
    label[sy:sy + ny, sx:sx + nx, :] = 0.

    return imgs, label

def posterize(img, label, v):
    '''
    4 < v < 8
    '''
    v = int(v)
    
    for slice_indx in range(img.shape[2]):
        img_curr = img[:,:,slice_indx]
        img_curr = convert_to_PIL(img_curr)
        img_curr = PIL.ImageOps.posterize(img_curr, bits=v)
        img_curr = convert_to_np(img_curr)
        img[:,:,slice_indx] = img_curr

    return img, label

def contrast(img, label, v):
    '''
    0.1 < v < 1.9
    '''
    
    for slice_indx in range(img.shape[2]):
        img_curr = img[:,:,slice_indx]
        img_curr = convert_to_PIL(img_curr)
        img_curr = PIL.ImageEnhance.Contrast(img_curr).enhance(v)
        img_curr = convert_to_np(img_curr)
        img[:,:,slice_indx] = img_curr

    return img, label

def brightness(img, label, v):
    '''
    0.1 < v < 1.9
    '''
    
    for slice_indx in range(img.shape[2]):
        img_curr = img[:,:,slice_indx]
        img_curr = convert_to_PIL(img_curr)
        img_curr = PIL.ImageEnhance.Brightness(img_curr).enhance(v)
        img_curr = convert_to_np(img_curr)
        img[:,:,slice_indx] = img_curr

    return img, label

def sharpness(img, label, v):
    '''
    0.1 < v < 1.9
    '''
    for slice_indx in range(img.shape[2]):
        img_curr = img[:,:,slice_indx]
        img_curr = convert_to_PIL(img_curr)
        img_curr = PIL.ImageEnhance.Sharpness(img_curr).enhance(v)
        img_curr = convert_to_np(img_curr)
        img[:,:,slice_indx] = img_curr

    return img, label

def identity(img, label, v):
    return img, label

def adjust_light(image, label):
    image = image*255.0
    gamma = random.random() * 3 + 0.5
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    for slice_indx in range(image.shape[2]):
        img_curr = image[:,:,slice_indx]
        img_curr = cv2.LUT(np.array(img_curr).astype(np.uint8), table).astype(np.uint8)
        image[:,:,slice_indx] = img_curr
    image = image/255.0

    return image, label

def shear_x(img, label, v):
    '''
    -0.3 < v < 0.3
    '''
    shear_mat = [1, v, -v * img.shape[1] / 2, 0, 1, 0]  # center the transform
    
    for slice_indx in range(img.shape[2]):
        img_curr = img[:,:,slice_indx]
        img_curr = convert_to_PIL(img_curr)
        img_curr = img_curr.transform(img_curr.size, PIL.Image.AFFINE, shear_mat, resample=PIL.Image.BILINEAR)
        img_curr = convert_to_np(img_curr)
        img[:,:,slice_indx] = img_curr
        # print(img.shape)

    for slice_indx in range(label.shape[2]):
        label_curr = label[:,:,slice_indx]
        label_curr = convert_to_PIL_label(label_curr)
        label_curr = label_curr.transform(label_curr.size, PIL.Image.AFFINE, shear_mat, resample=PIL.Image.NEAREST)
        label_curr = convert_to_np_label(label_curr)
        label[:,:,slice_indx] = label_curr
        # print(label.shape)

    return img, label

def shear_y(img, label, v):
    '''
    -0.3 < v < 0.3
    '''
    shear_mat = [1, 0, 0, v, 1, -v * img.shape[0] / 2]  # center the transform

    for slice_indx in range(img.shape[2]):
        img_curr = img[:,:,slice_indx]
        img_curr = convert_to_PIL(img_curr)
        img_curr = img_curr.transform(img_curr.size, PIL.Image.AFFINE, shear_mat, resample=PIL.Image.BILINEAR)
        img_curr = convert_to_np(img_curr)
        img[:,:,slice_indx] = img_curr
        # print(img.shape)

    for slice_indx in range(label.shape[2]):
        label_curr = label[:,:,slice_indx]
        label_curr = convert_to_PIL_label(label_curr)
        label_curr = label_curr.transform(label_curr.size, PIL.Image.AFFINE, shear_mat, resample=PIL.Image.NEAREST)
        label_curr = convert_to_np_label(label_curr)
        label[:,:,slice_indx] = label_curr
        # print(label.shape)

    return img, label

def translate_x(img, label, v):
    '''
    -0.45 < v < 0.45
    '''
    translate_mat = [1, 0, v * img.shape[1], 0, 1, 0]

    for slice_indx in range(img.shape[2]):
        img_curr = img[:,:,slice_indx]
        img_curr = convert_to_PIL(img_curr)
        img_curr = img_curr.transform(img_curr.size, PIL.Image.AFFINE, translate_mat, resample=PIL.Image.BILINEAR)
        img_curr = convert_to_np(img_curr)
        img[:,:,slice_indx] = img_curr

    for slice_indx in range(label.shape[2]):
        label_curr = label[:,:,slice_indx]
        label_curr = convert_to_PIL_label(label_curr)
        label_curr = label_curr.transform(label_curr.size, PIL.Image.AFFINE, translate_mat, resample=PIL.Image.NEAREST)
        label_curr = convert_to_np_label(label_curr)
        label[:,:,slice_indx] = label_curr

    return img, label

def translate_y(img, label, v):
    '''
    -0.45 < v < 0.45
    '''
    translate_mat = [1, 0, 0, 0, 1, v * img.shape[0]]

    for slice_indx in range(img.shape[2]):
        img_curr = img[:,:,slice_indx]
        img_curr = convert_to_PIL(img_curr)
        img_curr = img_curr.transform(img_curr.size, PIL.Image.AFFINE, translate_mat, resample=PIL.Image.BILINEAR)
        img_curr = convert_to_np(img_curr)
        img[:,:,slice_indx] = img_curr

    for slice_indx in range(label.shape[2]):
        label_curr = label[:,:,slice_indx]
        label_curr = convert_to_PIL_label(label_curr)
        label_curr = label_curr.transform(label_curr.size, PIL.Image.AFFINE, translate_mat, resample=PIL.Image.NEAREST)
        label_curr = convert_to_np_label(label_curr)
        label[:,:,slice_indx] = label_curr

    return img, label

def scale(img, label, v):
    '''
    0.6 < v < 1.4
    '''
    for slice_indx in range(img.shape[2]):
        img_curr = img[:,:,slice_indx]
        img_curr = convert_to_PIL(img_curr)
        img_curr = img_curr.transform(img_curr.size, PIL.Image.AFFINE, [v, 0, 0, 0, v, 0], resample=PIL.Image.BILINEAR)
        img_curr = convert_to_np(img_curr)
        img[:,:,slice_indx] = img_curr

    for slice_indx in range(label.shape[2]):
        label_curr = label[:,:,slice_indx]
        label_curr = convert_to_PIL_label(label_curr)
        label_curr = label_curr.transform(label_curr.size, PIL.Image.AFFINE, [v, 0, 0, 0, v, 0], resample=PIL.Image.NEAREST)
        label_curr = convert_to_np_label(label_curr)
        label[:,:,slice_indx] = label_curr

    return img, label

class RandomGenerator(object):
    def __init__(self, output_size, low_res):
        self.output_size = output_size
        self.low_res = low_res
        seed = 42
        self.rng = np.random.default_rng(seed)
        self.p = 0.5
        self.n = 2
        self.scale = (0.8, 1.2, 2)
        self.translate = (-0.2, 0.2, 2)
        self.shear = (-0.3, 0.3, 2)
        self.posterize = (4, 8.99, 2)
        self.contrast = (0.7, 1.3, 2)
        self.brightness = (0.7, 1.3, 2)
        self.sharpness = (0.1, 1.9, 2)

        self.create_ops()

    def create_ops(self):
        ops = [
            # (shear_x, self.shear),
            # (shear_y, self.shear),
            (scale, self.scale),
            (translate_x, self.translate),
            (translate_y, self.translate),
            (posterize, self.posterize),
            (contrast, self.contrast),
            (brightness, self.brightness),
            (sharpness, self.sharpness),
            (identity, (0, 1, 1)),
        ]

        self.ops = [op for op in ops if op[1][2] != 0]

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        if random.random() > 0.5:
            image, label = random_rotate(image, label)
        if random.random() > 0.5:
            image, label = adjust_light(image, label)
        # if random.random() > 0.5:
        #     image, label = random_erasing(imgs=image, label=label, rng=self.rng)
        
        inds = self.rng.choice(len(self.ops), size=self.n, replace=False)
        for i in inds:
            op = self.ops[i]
            aug_func = op[0]
            aug_params = op[1]
            v = self.rng.uniform(aug_params[0], aug_params[1])

            image, label = aug_func(image, label, v)

        x, y, z = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1.0), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y, 1.0), order=0)
        label_h, label_w, label_d = label.shape
        # low_res_label = zoom(label, (self.low_res[0] / label_h, self.low_res[1] / label_w, 1.0), order=0)
        
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        # low_res_label = torch.from_numpy(low_res_label.astype(np.float32))
        image = image.permute(2, 0, 1)
        label = label.permute(2, 0, 1)
        # low_res_label = low_res_label.permute(2, 0, 1)
        
        sample = {'image': image, 'label': label.long()}#, 'low_res_label': low_res_label.long()}
        return sample


class dataset_reader(Dataset):
    def __init__(self, base_dir, split, num_classes, transform=None, model_args=None):
        self.transform = transform 
        self.split = split
        
        self.data_dir = base_dir
        self.model_args = model_args

        if split=="train":
            self.num_classes = num_classes
            # df = pd.read_csv(base_dir+'/training.csv')
            df = pd.read_csv(base_dir)
            # if self.num_classes == 56: ## for TS MR only
            self.sample_list = [sample_pth for sample_pth in df["image_pth"]]
            self.masks_list = [sample_pth for sample_pth in df["mask_pth"]]
            print("len(self.sample_list 2D)", len(self.sample_list))
            print("self.sample_list[0] 2D", self.sample_list[0])

            ## 3D
            if self.model_args.dataset_type == "3D":
                self.sample_list = sorted_nicely(list(set(["/".join(i.split('/')[:-1]) for i in self.sample_list])))
                self.masks_list = sorted_nicely(list(set(["/".join(i.split('/')[:-1]) for i in self.masks_list])))
                # print("self.sample_list", self.sample_list)
                print("len(self.sample_list)", len(self.sample_list))
                print("self.sample_list[0]", self.sample_list[0])

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":

            if self.model_args.dataset_type == "2D":
                if self.sample_list[idx].endswith(".pkl"):
                    data = read_image(self.sample_list[idx])
                    # print("self.sample_list[idx]", self.sample_list[idx]) # /mnt/weka/wekafs/rad-megtron/cchen/prostateD/2D_all_5slice//0027/ images/2Dimage_0015.pkl
                elif self.sample_list[idx].endswith(".png") or self.sample_list[idx].endswith(".jpg"):
                    data = cv2.imread(self.sample_list[idx], cv2.IMREAD_UNCHANGED)
                    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                
                # if 'synapseCT' in self.sample_list[idx]:
                if self.num_classes==12:
                    HU_min, HU_max = -200, 250
                    data_mean = 50.21997497685108
                    data_std = 68.47153712416372
                    data = np.clip(data, HU_min, HU_max)
                    data = (data-HU_min)/(HU_max-HU_min)*255.0
                    data = np.float32(data)
                    data = (data - data_mean) / data_std
                    # elif 'prostate' in self.sample_list[idx]: 
                elif self.num_classes==1: ## only prostate not pancreas 
                    data = np.float32(data)
                elif self.num_classes==9: ## only abaltas 
                    data = np.float32(data)
                elif self.num_classes==56 or self.num_classes==78 or self.num_classes==131:
                    data = np.float32(data)
                else:
                    exit(1)
                data = (data-data.min())/(data.max()-data.min()+0.00000001)
                h, w, d = data.shape
    
                ## 2d mae
                if self.sample_list[idx].endswith(".pkl"): ## only pkl 5 slices needs
                    data = data[:,:,2]
                    data = np.expand_dims(data, axis=2)
                    data = np.repeat(data, 3, axis=2)
                elif self.sample_list[idx].endswith(".png") or self.sample_list[idx].endswith(".jpg"):
                    pass
                
                if self.sample_list[idx].endswith(".pkl"):
                    mask = read_image(self.masks_list[idx])
                    ## 2d mae
                    mask = mask[:,:,2]
                    mask = np.expand_dims(mask, axis=2)
                    mask = np.repeat(mask, 3, axis=2)
                elif self.sample_list[idx].endswith(".png") or self.sample_list[idx].endswith(".jpg"):
                    mask = cv2.imread(self.masks_list[idx], cv2.IMREAD_UNCHANGED)
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                
                if self.num_classes==12:
                    mask[mask==13] = 12
    
                image = np.float32(data)
                label = np.float32(mask)

            elif self.model_args.dataset_type == "3D":
                data_path = self.sample_list[idx]
                # print("data_path", data_path)
                samp_list = sorted_nicely([i_s for i_s in os.listdir(data_path) if not i_s.startswith(".")])
                # print("len(samp_list)", len(samp_list))
                images = [cv2.imread(data_path+"/"+i_i, cv2.IMREAD_GRAYSCALE) for i_i in samp_list]
                image = np.stack(images)
                # print("image.shape", image.shape)
                image = np.transpose(image, (1, 2, 0))
                # print("image.shape 2", image.shape)
                image = np.float32(image)
                image = (image-image.min())/(image.max()-image.min()+0.00000001)

                mask_path = self.masks_list[idx]
                mask_list = sorted_nicely([i_s for i_s in os.listdir(mask_path) if not i_s.startswith(".")])
                masks = [cv2.imread(mask_path+"/"+i_i, cv2.IMREAD_GRAYSCALE) for i_i in mask_list]
                mask = np.stack(masks)
                mask = np.transpose(mask, (1, 2, 0))
                label = np.float32(mask)

        sample = {'image': image, 'label': label}
        
        if self.transform:
            sample = self.transform(sample)

        if self.model_args.dataset_type == "3D":
            image = sample['image'].unsqueeze(3)
            # print("image.shape", image.shape)
            d, h, w, _ = image.shape
            sample['image'] = image.expand(d, h, w, 3)
            # print("sample['image'].shape 2", sample['image'].shape)

            label = sample['label'].unsqueeze(3)
            # print("label.shape", label.shape)
            sample['label'] = label.expand(d, h, w, 3)   
            # print("sample['label'].shape 2", sample['label'].shape)     
            # print("torch.unique(sample['label'])", torch.unique(sample['label']))

            if d % 16 != 0:
                pad_size = 16 - (d % 16)
                pad_tensor = torch.zeros(pad_size, h, w, 3, dtype=sample['image'].dtype)
                pad_tensor2 = torch.zeros(pad_size, h, w, 3, dtype=sample['label'].dtype)
                sample['image'] = torch.cat((sample['image'], pad_tensor), dim=0)
                sample['label'] = torch.cat((sample['label'], pad_tensor2), dim=0)
            # sample['image'] = rearrange(sample['image'], 'n h w c -> c n h w').contiguous()
            # sample['label'] = rearrange(sample['label'], 'n h w c -> c n h w').contiguous()
            sample['image'] = sample['image'].permute(3,0,1,2)
            sample['label'] = sample['label'].permute(3,0,1,2)
            # print("sample['image'].shape 3", sample['image'].shape)
            # print("sample['label'].shape 3", sample['label'].shape)
            # print("torch.unique(sample['label'])", torch.unique(sample['label']))

        sample['case_name'] = [self.sample_list[idx].strip('\n'), self.masks_list[idx].strip('\n')]
        # print(sample['case_name'])
        return sample
