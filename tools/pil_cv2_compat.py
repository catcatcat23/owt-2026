"""Small Pillow-backed subset of cv2 used by Common8 preprocessing."""

from pathlib import Path

import numpy as np
from PIL import Image


IMREAD_COLOR = 1
IMREAD_UNCHANGED = -1
COLOR_GRAY2BGR = 8
IMWRITE_JPEG_QUALITY = 1
INTER_NEAREST = 0
INTER_LINEAR = 1


def imread(path, flag=IMREAD_COLOR):
    try:
        with Image.open(path) as image:
            if flag == IMREAD_COLOR:
                return np.array(image.convert("RGB"))[:, :, ::-1]
            return np.array(image)
    except (FileNotFoundError, OSError):
        return None


def cvtColor(array, code):
    if code != COLOR_GRAY2BGR or array.ndim != 2:
        raise ValueError(f"Unsupported color conversion code={code}, shape={array.shape}")
    return np.repeat(array[:, :, None], 3, axis=2)


def resize(array, size, interpolation=INTER_LINEAR):
    resample = Image.NEAREST if interpolation == INTER_NEAREST else Image.BILINEAR
    if np.issubdtype(array.dtype, np.floating):
        image = Image.fromarray(array.astype(np.float32), mode="F")
        return np.asarray(image.resize(size, resample=resample), dtype=np.float32)
    image = Image.fromarray(array)
    return np.asarray(image.resize(size, resample=resample), dtype=array.dtype)


def imwrite(path, array, params=None):
    output = Path(path)
    try:
        if array.ndim == 3 and array.shape[2] == 3:
            image = Image.fromarray(array[:, :, ::-1], mode="RGB")
        else:
            image = Image.fromarray(array)
        save_args = {}
        if output.suffix.lower() in {".jpg", ".jpeg"} and params:
            for index in range(0, len(params) - 1, 2):
                if params[index] == IMWRITE_JPEG_QUALITY:
                    save_args["quality"] = int(params[index + 1])
        image.save(output, **save_args)
        return True
    except OSError:
        return False
