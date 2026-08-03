"""Paired native-resolution transforms for OrganSlotBank crop experiments."""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as vision_f


class TransformDataset(Dataset):
    """Apply a sample transform without exposing labels outside the wrapper."""

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
        self.case_ids = getattr(dataset, "case_ids", ())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.transform(self.dataset[index])


class OrganAwareMixDataset(Dataset):
    """Mix uniform global samples with class-balanced positive ROI samples."""

    def __init__(
        self,
        dataset,
        transform,
        roi_indices,
        focus_class_ids=(4, 5, 6),
        roi_probability=0.2,
    ):
        self.dataset = dataset
        self.transform = transform
        self.focus_class_ids = tuple(int(value) for value in focus_class_ids)
        self.roi_probability = float(roi_probability)
        self.case_ids = getattr(dataset, "case_ids", ())
        if not 0.0 <= self.roi_probability <= 1.0:
            raise ValueError("roi_probability must be in [0,1]")
        if not self.focus_class_ids:
            raise ValueError("focus_class_ids must not be empty")
        self.roi_indices = {}
        for class_id in self.focus_class_ids:
            pool = tuple(
                int(index)
                for index in roi_indices.get(str(class_id), roi_indices.get(class_id, ()))
                if int(index) < len(dataset)
            )
            if not pool:
                raise ValueError("empty ROI index pool for class {}".format(class_id))
            self.roi_indices[class_id] = pool

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        use_roi = bool(torch.rand(()) < self.roi_probability)
        focus_class_id = -1
        source_index = int(index)
        if use_roi:
            focus_position = int(
                torch.randint(0, len(self.focus_class_ids), ()).item()
            )
            focus_class_id = self.focus_class_ids[focus_position]
            pool = self.roi_indices[focus_class_id]
            pool_position = int(torch.randint(0, len(pool), ()).item())
            source_index = pool[pool_position]
        sample = dict(self.dataset[source_index])
        sample["focus_class_id"] = focus_class_id
        sample["roi_applied"] = use_roi
        return self.transform(sample)


def _as_spatial_batch(value):
    if value.ndim == 3:  # [C,H,W]
        return value.unsqueeze(0), lambda result: result.squeeze(0)
    if value.ndim == 4:  # [C,T,H,W] -> [T,C,H,W]
        return value.permute(1, 0, 2, 3), lambda result: result.permute(1, 0, 2, 3)
    raise ValueError("expected [C,H,W] or [C,T,H,W], got {}".format(value.shape))


def _resize_image(image, output_size):
    batch, restore = _as_spatial_batch(image)
    resized = F.interpolate(
        batch,
        size=(output_size, output_size),
        mode="bicubic",
        align_corners=False,
    )
    return restore(resized).clamp_(0.0, 1.0)


def _resize_label(label, output_size):
    batch, restore = _as_spatial_batch(label.float())
    resized = F.interpolate(
        batch,
        size=(output_size, output_size),
        mode="nearest",
    )
    return restore(resized).long()


def _apply_to_spatial_batch(value, function):
    batch, restore = _as_spatial_batch(value)
    return restore(function(batch))


class OrganSlotHighResTransform:
    """Paired augmentation followed by global or organ-aware physical crop.

    Global samples use a deterministic centered 448x448 window. ROI samples
    use a label-guided 384x384 window with bounded center jitter. The label is
    used only inside this training transform and is hidden by the subsequent
    strict visibility wrapper.
    """

    def __init__(
        self,
        output_size=448,
        training=True,
        global_crop_size=448,
        roi_crop_size=384,
        roi_center_jitter=0.1,
        flip_probability=0.5,
        rotation_probability=0.5,
        max_rotation_degrees=15.0,
        gamma_probability=0.5,
        photometric_operations=2,
    ):
        self.output_size = int(output_size)
        self.training = bool(training)
        self.global_crop_size = int(global_crop_size)
        self.roi_crop_size = int(roi_crop_size)
        self.roi_center_jitter = float(roi_center_jitter)
        self.flip_probability = float(flip_probability)
        self.rotation_probability = float(rotation_probability)
        self.max_rotation_degrees = float(max_rotation_degrees)
        self.gamma_probability = float(gamma_probability)
        self.photometric_operations = int(photometric_operations)
        if self.output_size <= 0:
            raise ValueError("output_size must be positive")
        if self.global_crop_size <= 0 or self.roi_crop_size <= 0:
            raise ValueError("crop sizes must be positive")
        if self.roi_crop_size > self.global_crop_size:
            raise ValueError("roi_crop_size must not exceed global_crop_size")
        if not 0.0 <= self.roi_center_jitter <= 0.5:
            raise ValueError("roi_center_jitter must be in [0,0.5]")
        for name, value in (
            ("flip_probability", self.flip_probability),
            ("rotation_probability", self.rotation_probability),
            ("gamma_probability", self.gamma_probability),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("{} must be in [0,1]".format(name))
        if not 0 <= self.photometric_operations <= 5:
            raise ValueError("photometric_operations must be between 0 and 5")

    @staticmethod
    def _random_event(probability):
        return bool(torch.rand(()) < probability)

    def _paired_flip(self, image, label):
        if not self._random_event(self.flip_probability):
            return image, label
        dimension = -1 if bool(torch.randint(0, 2, ()).item()) else -2
        return torch.flip(image, (dimension,)), torch.flip(label, (dimension,))

    def _paired_rotate(self, image, label):
        if not self._random_event(self.rotation_probability):
            return image, label
        angle = (
            float(torch.rand(())) * 2.0 - 1.0
        ) * self.max_rotation_degrees
        image = _apply_to_spatial_batch(
            image,
            lambda batch: vision_f.rotate(
                batch,
                angle,
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            ),
        )
        label = _apply_to_spatial_batch(
            label.float(),
            lambda batch: vision_f.rotate(
                batch,
                angle,
                interpolation=InterpolationMode.NEAREST,
                fill=0.0,
            ),
        ).long()
        return image, label

    def _gamma(self, image):
        if not self._random_event(self.gamma_probability):
            return image
        sampled_gamma = 0.5 + 3.0 * float(torch.rand(()))
        return image.clamp(0.0, 1.0).pow(1.0 / sampled_gamma)

    @staticmethod
    def _photometric(image, operation):
        batch, restore = _as_spatial_batch(image)
        if operation == 0:
            factor = 0.7 + 0.6 * float(torch.rand(()))
            batch = vision_f.adjust_contrast(batch, factor)
        elif operation == 1:
            factor = 0.7 + 0.6 * float(torch.rand(()))
            batch = vision_f.adjust_brightness(batch, factor)
        elif operation == 2:
            factor = 0.1 + 1.8 * float(torch.rand(()))
            batch = vision_f.adjust_sharpness(batch, factor)
        elif operation == 3:
            bits = int(torch.randint(4, 9, ()).item())
            uint8 = torch.round(batch.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
            batch = vision_f.posterize(uint8, bits).float().div_(255.0)
        elif operation != 4:
            raise ValueError("unknown photometric operation")
        return restore(batch).clamp_(0.0, 1.0)

    def _apply_photometric(self, image):
        if self.photometric_operations == 0:
            return image
        order = torch.randperm(5)[: self.photometric_operations]
        for operation in order.tolist():
            image = self._photometric(image, operation)
        return image

    @staticmethod
    def _check_crop_fits(crop_size, source_height, source_width):
        if source_height < crop_size or source_width < crop_size:
            raise ValueError(
                "fixed crop {} does not fit native matrix {}x{}".format(
                    crop_size, source_height, source_width
                )
            )

    def _global_crop(self, source_height, source_width):
        crop_size = self.global_crop_size
        self._check_crop_fits(crop_size, source_height, source_width)
        return (
            (source_height - crop_size) // 2,
            (source_width - crop_size) // 2,
            crop_size,
            crop_size,
        )

    def _organ_crop(self, label, focus_class_id, source_height, source_width):
        crop_size = self.roi_crop_size
        self._check_crop_fits(crop_size, source_height, source_width)
        if label.ndim == 3:
            focus_map = label[0].eq(int(focus_class_id))
        elif label.ndim == 4:
            focus_map = label[0].eq(int(focus_class_id)).any(dim=0)
        else:
            raise ValueError("label must be [1,H,W] or [1,T,H,W]")
        coordinates = torch.nonzero(focus_map, as_tuple=False)
        if coordinates.numel() == 0:
            raise ValueError(
                "ROI sample does not contain focus class {}".format(focus_class_id)
            )
        y_min = int(coordinates[:, 0].min())
        y_max = int(coordinates[:, 0].max())
        x_min = int(coordinates[:, 1].min())
        x_max = int(coordinates[:, 1].max())
        if y_max - y_min + 1 > crop_size or x_max - x_min + 1 > crop_size:
            raise ValueError("focus-organ bounding box exceeds ROI crop")

        jitter = int(round(crop_size * self.roi_center_jitter))
        jitter_y = int(torch.randint(-jitter, jitter + 1, ()).item()) if jitter else 0
        jitter_x = int(torch.randint(-jitter, jitter + 1, ()).item()) if jitter else 0
        desired_top = int(round((y_min + y_max + 1 - crop_size) / 2)) + jitter_y
        desired_left = int(round((x_min + x_max + 1 - crop_size) / 2)) + jitter_x

        top_lower = max(0, y_max - crop_size + 1)
        top_upper = min(y_min, source_height - crop_size)
        left_lower = max(0, x_max - crop_size + 1)
        left_upper = min(x_min, source_width - crop_size)
        if top_lower > top_upper or left_lower > left_upper:
            raise RuntimeError("could not place ROI crop while retaining focus organ")
        top = min(max(desired_top, top_lower), top_upper)
        left = min(max(desired_left, left_lower), left_upper)
        return top, left, crop_size, crop_size

    def __call__(self, sample):
        output = dict(sample)
        image = output["image"].float()
        label = output["label"].long()
        if label.shape[-2:] != image.shape[-2:]:
            raise ValueError("image and label spatial shapes differ")
        source_height, source_width = image.shape[-2:]
        if source_height <= 0 or source_width <= 0:
            raise ValueError("image spatial dimensions must be positive")

        if self.training:
            image, label = self._paired_flip(image, label)
            image, label = self._paired_rotate(image, label)
            image = self._gamma(image)
            image = self._apply_photometric(image)

        focus_class_id = int(output.get("focus_class_id", -1))
        roi_applied = bool(output.get("roi_applied", False))
        if roi_applied:
            if not self.training or focus_class_id < 0:
                raise ValueError("ROI crop requires training and a focus class")
            top, left, crop_height, crop_width = self._organ_crop(
                label, focus_class_id, source_height, source_width
            )
        else:
            top, left, crop_height, crop_width = self._global_crop(
                source_height, source_width
            )
        image = image[
            ..., top : top + crop_height, left : left + crop_width
        ]
        label = label[
            ..., top : top + crop_height, left : left + crop_width
        ]
        output["image"] = _resize_image(image, self.output_size)
        output["label"] = _resize_label(label, self.output_size)
        output["crop_box"] = torch.tensor(
            [
                top,
                left,
                crop_height,
                crop_width,
                source_height,
                source_width,
            ],
            dtype=torch.int64,
        )
        output["crop_applied"] = True
        output["crop_randomized"] = bool(roi_applied)
        output["roi_applied"] = bool(roi_applied)
        output["focus_class_id"] = int(focus_class_id)
        return output
