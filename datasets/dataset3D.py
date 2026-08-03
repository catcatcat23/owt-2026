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
    def __init__(self, output_size, low_res, renormalize_after_resize=True):
        self.output_size = output_size
        self.low_res = low_res
        self.renormalize_after_resize = renormalize_after_resize
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
            if self.renormalize_after_resize:
                image = (image-image.min())/(image.max()-image.min()+0.00000001)
            else:
                image = np.clip(image, 0.0, 1.0)
        label_h, label_w, label_d = label.shape
        
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        image = image.permute(2, 0, 1)
        label = label.permute(2, 0, 1)
        
        sample = {'image': image, 'label': label.long()}
        return sample


def _parse_present_classes(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [int(item) for item in str(value).split('|') if item]


def crop_around_class(image, label, crop_size, class_id=None, jitter=0):
    """Crop image and label together in XY; 3D depth is kept unchanged."""
    height, width = image.shape[:2]
    if crop_size > height or crop_size > width:
        return image, label, False

    if class_id is not None:
        label_map = np.any(label == class_id, axis=2) if label.ndim == 3 else label == class_id
        coordinates = np.argwhere(label_map)
    else:
        coordinates = np.empty((0, 2))
    if coordinates.size:
        center_y = int((coordinates[:, 0].min() + coordinates[:, 0].max()) // 2)
        center_x = int((coordinates[:, 1].min() + coordinates[:, 1].max()) // 2)
        if jitter > 0:
            center_y += random.randint(-jitter, jitter)
            center_x += random.randint(-jitter, jitter)
        top = min(max(center_y - crop_size // 2, 0), height - crop_size)
        left = min(max(center_x - crop_size // 2, 0), width - crop_size)
    else:
        top = random.randint(0, height - crop_size)
        left = random.randint(0, width - crop_size)

    return (
        image[top:top + crop_size, left:left + crop_size, ...],
        label[top:top + crop_size, left:left + crop_size, ...],
        True,
    )


class dataset_reader(Dataset):
    def __init__(self, base_dir, split, num_classes, transform=None, model_args=None):
        self.transform = transform 
        self.split = split
        self.data_dir = base_dir
        self.model_args = model_args

        self.num_classes = num_classes
        df = pd.read_csv(base_dir)
        self.sample_list = [sample_pth for sample_pth in df["image_pth"]]
        self.masks_list = [sample_pth for sample_pth in df["mask_pth"]]
        self.present_classes = (
            [_parse_present_classes(value) for value in df["present_classes"]]
            if "present_classes" in df.columns
            else [[] for _ in self.sample_list]
        )
        self.crop_mix_prob = float(getattr(model_args, "crop_mix_prob", 0.0))
        self.crop_focus_sample_prob = float(getattr(model_args, "crop_focus_sample_prob", 0.0))
        self.crop_focus_classes = set(getattr(model_args, "crop_focus_classes", []))
        self.crop_size = int(getattr(model_args, "crop_size", 0) or model_args.input_size)
        self.crop_jitter = int(getattr(model_args, "crop_jitter", 0))
        self.focus_indices = [
            index for index, classes in enumerate(self.present_classes)
            if self.crop_focus_classes.intersection(classes)
        ]
        print("len(self.sample_list 2D)", len(self.sample_list))
        print("self.sample_list[0] 2D", self.sample_list[0])
        if self.crop_mix_prob > 0:
            print(
                "CropMix enabled:",
                f"crop_prob={self.crop_mix_prob}",
                f"focus_sample_prob={self.crop_focus_sample_prob}",
                f"focus_classes={sorted(self.crop_focus_classes)}",
                f"focus_rows={len(self.focus_indices)}",
                f"crop_size={self.crop_size}",
                f"jitter={self.crop_jitter}",
            )

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        crop_requested = self.crop_mix_prob > 0 and random.random() < self.crop_mix_prob
        actual_idx = idx
        if (
            crop_requested
            and self.focus_indices
            and random.random() < self.crop_focus_sample_prob
        ):
            actual_idx = random.choice(self.focus_indices)

        if self.model_args.dataset_type == "2D":
            data = cv2.imread(self.sample_list[actual_idx], cv2.IMREAD_UNCHANGED)
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

            data = np.float32(data)
            if getattr(self.model_args, "intensity_norm", "per_sample") == "fixed_255":
                data = data / 255.0
            else:
                data = (data-data.min())/(data.max()-data.min()+0.00000001)
            h, w, d = data.shape

            mask = cv2.imread(self.masks_list[actual_idx], cv2.IMREAD_UNCHANGED)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

            image = np.float32(data)
            label = np.float32(mask)

        elif self.model_args.dataset_type == "3D":
            data_path = self.sample_list[actual_idx]
            start_frame = int(data_path.split(".jpg")[0].split("_")[-1])
            base_data_path = "_".join(data_path.split(".jpg")[0].split("_")[:-1])
            images = [cv2.imread(base_data_path+"_"+str(i_i)+".jpg", cv2.IMREAD_GRAYSCALE) for i_i in range(start_frame, start_frame+self.model_args.fix_frame)]
            image = np.stack(images)
            image = np.transpose(image, (1, 2, 0))
            image = np.float32(image)
            if getattr(self.model_args, "intensity_norm", "per_sample") == "fixed_255":
                image = image / 255.0
            else:
                image = (image-image.min())/(image.max()-image.min()+0.00000001)

            mask_path = self.masks_list[actual_idx]
            start_frame = int(mask_path.split(".png")[0].split("_")[-1])
            base_mask_path = "_".join(mask_path.split(".png")[0].split("_")[:-1])
            masks = [cv2.imread(base_mask_path+"_"+str(i_i)+".png", cv2.IMREAD_GRAYSCALE) for i_i in range(start_frame, start_frame+self.model_args.fix_frame)]
            mask = np.stack(masks)
            mask = np.transpose(mask, (1, 2, 0))
            label = np.float32(mask)

        crop_focus_class = 0
        crop_applied = False
        if crop_requested:
            present = {int(value) for value in np.unique(label) if value > 0}
            focus_candidates = sorted(present.intersection(self.crop_focus_classes))
            candidates = focus_candidates or sorted(present)
            selected_class = random.choice(candidates) if candidates else None
            image, label, crop_applied = crop_around_class(
                image,
                label,
                self.crop_size,
                class_id=selected_class,
                jitter=self.crop_jitter,
            )
            crop_focus_class = selected_class or 0

        sample = {'image': image, 'label': label}
        
        if self.transform:
            sample = self.transform(sample)

        if self.model_args.dataset_type == "3D":
            image = sample['image'].unsqueeze(3)
            d, h, w, _ = image.shape
            sample['image'] = image.expand(d, h, w, 3)

            label = sample['label'].unsqueeze(3)
            sample['label'] = label.expand(d, h, w, 3)   

            sample['image'] = sample['image'].permute(3,0,1,2)
            sample['label'] = sample['label'].permute(3,0,1,2)

        if crop_focus_class and not torch.any(sample['label'] == crop_focus_class):
            crop_focus_class = 0
        sample['crop_applied'] = int(crop_applied)
        sample['crop_focus_class'] = int(crop_focus_class)
        sample['case_name'] = [
            self.sample_list[actual_idx].strip('\n'),
            self.masks_list[actual_idx].strip('\n'),
        ]
        return sample
