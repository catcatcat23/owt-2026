import torch
from losses_orgslot import base_segmentation_loss, small_organ_segmentation_loss


def compute(x, y, diagnostics=None, keep=None):
    if keep is None:
        keep = torch.ones(x.shape[0], 1, dtype=torch.bool)
    return base_segmentation_loss(
        {"pancreas": x}, {"pancreas": y}, ["pancreas"], keep,
        loss_type="small_organ", diagnostics=diagnostics,
        segmentation_unit="slice",
    )[0]


def test_mixed_slab_separate_means_and_finite_backward():
    torch.manual_seed(7)
    x = torch.randn(1, 1, 4, 8, 8, requires_grad=True)
    y = torch.zeros_like(x)
    y[:, :, 0, 2:5, 2:5] = 1
    diag = {}
    actual = compute(x, y, diag)
    losses = [small_organ_segmentation_loss(x[:, :, t], y[:, :, t])[0]
              for t in range(4)]
    expected = losses[0] + torch.stack(losses[1:]).mean()
    torch.testing.assert_close(actual, expected)
    assert diag["pancreas"]["positive_samples"].item() == 1
    assert diag["pancreas"]["negative_samples"].item() == 3
    actual.backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad[:, :, 1:].abs().sum() > 0


def test_2d_unchanged_and_equivalent_to_temporal_slices():
    x = torch.randn(4, 1, 8, 8, requires_grad=True)
    y = torch.zeros_like(x)
    y[0, :, 1:4, 2:5] = 1
    expected = small_organ_segmentation_loss(x[:1], y[:1])[0]
    expected = expected + torch.stack([
        small_organ_segmentation_loss(x[i:i+1], y[i:i+1])[0]
        for i in range(1, 4)
    ]).mean()
    torch.testing.assert_close(compute(x, y), expected)
    torch.testing.assert_close(compute(x.permute(1, 0, 2, 3)[None],
                                       y.permute(1, 0, 2, 3)[None]), expected)


def test_excluded_slab_has_zero_gradient():
    x = torch.randn(2, 1, 4, 8, 8, requires_grad=True)
    y = torch.zeros_like(x)
    loss = compute(x, y, keep=torch.tensor([[True], [False]]))
    loss.backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad[1].abs().sum() == 0
