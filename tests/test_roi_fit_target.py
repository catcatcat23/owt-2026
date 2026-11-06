"""CPU regression: opt-in large-organ ROI must retain the target bounding box."""
import torch
from datasets.orgslot_highres import OrganSlotHighResTransform, crop_with_padding


def test_roi_fit_target():
    base = OrganSlotHighResTransform(roi_center_jitter=0)
    fit = OrganSlotHighResTransform(roi_center_jitter=0, roi_fit_target=True)
    small = torch.zeros(1, 512, 512, dtype=torch.long)
    small[:, 180:230, 200:240] = 7
    assert base._organ_crop(small, 7, 512, 512) == fit._organ_crop(small, 7, 512, 512)
    for height, width, box in [(512, 512, (10, 450, 20, 460)), (400, 600, (0, 390, 20, 580))]:
        label = torch.zeros(1, height, width, dtype=torch.long)
        y0, y1, x0, x1 = box
        label[:, y0:y1, x0:x1] = 7
        try:
            base._organ_crop(label, 7, height, width)
        except ValueError:
            pass
        else:
            raise AssertionError('Legacy oversized-target rejection changed')
        top, left, h, w = fit._organ_crop(label, 7, height, width)
        cropped = crop_with_padding(label, top, left, h, w)
        assert cropped.shape[-2:] == (h, w)
        assert int(cropped.eq(7).sum()) == int(label.eq(7).sum())
    print('PASS: unchanged small ROI; complete oversized and padded ROI; legacy rejection')

if __name__ == '__main__':
    test_roi_fit_target()
