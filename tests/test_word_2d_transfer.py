import torch
import torch.nn.functional as F
from util.word_2d_transfer import inflate_conv, resize_canvas


def test_temporal_inflation_preserves_identical_slices():
    torch.manual_seed(1)
    w = torch.randn(3, 2, 1, 1)
    x = torch.randn(1, 2, 5, 5)
    inflated = inflate_conv(w, (3, 2, 4, 1, 1))
    actual = F.conv3d(x.unsqueeze(2).repeat(1, 1, 4, 1, 1), inflated).squeeze(2)
    torch.testing.assert_close(actual, F.conv2d(x, w))


def test_canvas_preserves_constants_and_time_order():
    w = torch.tensor([[1., 2.]]).repeat(4, 1)
    result = resize_canvas(w, (4, 3, 3))
    torch.testing.assert_close(result, torch.tensor([[1., 2.]]).repeat(36, 1))
