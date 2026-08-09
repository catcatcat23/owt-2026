from types import SimpleNamespace

import torch
import torch.nn as nn

from OWT_models_psem_lossbalance_v3 import (
    StateSeparatedLossMaskedAutoencoderViT,
)
from util.per_sample_mask_schedule import masked_reconstruction_target
from util.triplet_query_loss import triplet_delta_loss
from util.triplet_query_schedule import triplet_query_schedule


def _model():
    args = SimpleNamespace(
        arch_version="v11",
        dataset_type="2D",
        temp_stride=0,
        fix_frame=0,
        LA=True,
        organ_token_total=6,
        token_factor=2,
        num_classes_with_bg=3,
        loss_version=["L2"],
        text_encoding="None",
        roi_class_weights=[0.0, 1.0, 2.0],
        positive_roi_loss_weight=0.25,
    )
    return StateSeparatedLossMaskedAutoencoderViT(
        img_size=32,
        patch_size=16,
        in_chans=3,
        embed_dim=32,
        depth=2,
        num_heads=4,
        decoder_embed_dim=32,
        decoder_depth=1,
        decoder_num_heads=4,
        mlp_ratio=2,
        norm_layer=nn.LayerNorm,
        model_args=args,
    )


def _triplet_batch(image, label, sample_indices, epoch=0):
    schedule = triplet_query_schedule(sample_indices, epoch, 3)
    masks = (
        schedule["direct_mask"],
        schedule["context_mask"],
        schedule["plus_mask"],
    )
    targets = tuple(
        masked_reconstruction_target(image, label, mask) for mask in masks
    )
    return schedule, masks, targets


def test_schedule_builds_anchor_context_edges_without_gt():
    sample_indices = torch.arange(16)
    schedule = triplet_query_schedule(sample_indices, epoch=3, num_classes_with_bg=5)
    direct = schedule["direct_mask"]
    context = schedule["context_mask"]
    plus = schedule["plus_mask"]
    anchors = schedule["anchor_classes"]
    rows = torch.arange(sample_indices.numel())

    assert torch.all((anchors >= 1) & (anchors < 5))
    assert torch.all(direct.sum(dim=1) == 1)
    assert torch.all(context.sum(dim=1) > 0)
    assert not torch.any(context[rows, anchors])
    assert torch.all(plus == (context | direct))
    assert torch.all(plus.sum(dim=1) == context.sum(dim=1) + 1)
    assert schedule["full_context"].sum().item() == 8


def test_anchor_cycle_covers_every_foreground_class():
    observed = []
    for epoch in range(4):
        schedule = triplet_query_schedule(
            torch.tensor([7]), epoch, num_classes_with_bg=5
        )
        observed.append(schedule["anchor_classes"].item())
    assert set(observed) == {1, 2, 3, 4}


def test_targets_obey_additive_identity_for_present_and_absent_anchors():
    image = torch.rand(2, 3, 4, 4)
    label = torch.zeros(2, 3, 4, 4, dtype=torch.long)
    label[0, :, :2] = 1
    direct = torch.tensor([[False, True, False], [False, True, False]])
    context = torch.tensor([[True, False, True], [True, False, True]])
    plus = context | direct

    target_direct = masked_reconstruction_target(image, label, direct)
    target_context = masked_reconstruction_target(image, label, context)
    target_plus = masked_reconstruction_target(image, label, plus)

    assert torch.allclose(target_plus - target_context, target_direct)
    assert target_direct[0].abs().sum() > 0
    assert target_direct[1].abs().sum() == 0
    assert torch.allclose(target_plus[1], target_context[1])


def test_delta_loss_supervises_present_and_absent_queries():
    image = torch.rand(2, 3, 4, 4)
    label = torch.zeros(2, 3, 4, 4, dtype=torch.long)
    label[0, :, :2] = 1
    direct = torch.tensor([[False, True, False], [False, True, False]])
    target_direct = masked_reconstruction_target(image, label, direct)
    pred_context = torch.rand_like(image, requires_grad=True)
    pred_plus = (pred_context.detach() + target_direct).requires_grad_(True)

    perfect = triplet_delta_loss(
        pred_context,
        pred_plus,
        target_direct,
        label,
        direct,
        class_weights=torch.tensor([0.0, 1.0, 2.0]),
        positive_roi_loss_weight=0.25,
    )
    assert torch.allclose(perfect["loss"], torch.zeros_like(perfect["loss"]), atol=1e-7)
    assert perfect["positive_class_counts"].tolist() == [0, 1, 0]

    perturbed_plus = pred_plus.detach().clone()
    perturbed_plus[0, :, :2] += 0.5
    perturbed_plus[1] += 0.25
    perturbed_plus.requires_grad_(True)
    perturbed = triplet_delta_loss(
        pred_context,
        perturbed_plus,
        target_direct,
        label,
        direct,
        class_weights=torch.tensor([0.0, 1.0, 2.0]),
        positive_roi_loss_weight=0.25,
    )
    assert perturbed["global_loss"] > 0
    assert perturbed["positive_roi_loss"] > 0
    perturbed["loss"].backward()
    assert torch.isfinite(perturbed_plus.grad).all()


def test_triplet_model_forward_backward():
    torch.manual_seed(4)
    model = _model()
    image = torch.rand(2, 3, 32, 32)
    label = torch.zeros(2, 3, 32, 32, dtype=torch.long)
    label[0, :, :16] = 1
    label[1, :, 16:] = 2
    _, masks, targets = _triplet_batch(
        image, label, torch.tensor([0, 1]), epoch=0
    )

    triplet_image = torch.cat((image, image, image), dim=0)
    triplet_label = torch.cat((label, label, label), dim=0)
    triplet_mask = torch.cat(masks, dim=0)
    triplet_target = torch.cat(targets, dim=0)
    branch_loss, pred, output = model(
        triplet_image,
        middle={
            "image_target": triplet_target,
            "label": triplet_label,
            "class_keep_mask": triplet_mask,
        },
    )
    pred_direct, pred_context, pred_plus = pred.chunk(3, dim=0)
    delta = triplet_delta_loss(
        pred_context,
        pred_plus,
        targets[0],
        label,
        masks[0],
        output["roi_class_weights"],
        positive_roi_loss_weight=0.25,
    )
    loss = branch_loss + 0.1 * delta["loss"]
    loss.backward()

    assert pred.shape == triplet_image.shape
    assert torch.isfinite(loss)
    assert model.patch_embed.proj.weight.grad is not None
    assert torch.isfinite(model.patch_embed.proj.weight.grad).all()


def test_triplet_tiny_batch_overfit():
    torch.manual_seed(5)
    model = _model()
    image = torch.rand(1, 3, 32, 32)
    label = torch.zeros(1, 3, 32, 32, dtype=torch.long)
    label[:, :, :16] = 1
    _, masks, targets = _triplet_batch(
        image, label, torch.tensor([0]), epoch=0
    )
    triplet_image = torch.cat((image, image, image), dim=0)
    triplet_label = torch.cat((label, label, label), dim=0)
    triplet_mask = torch.cat(masks, dim=0)
    triplet_target = torch.cat(targets, dim=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0)
    losses = []
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        branch_loss, pred, output = model(
            triplet_image,
            middle={
                "image_target": triplet_target,
                "label": triplet_label,
                "class_keep_mask": triplet_mask,
            },
        )
        _, pred_context, pred_plus = pred.chunk(3, dim=0)
        delta = triplet_delta_loss(
            pred_context,
            pred_plus,
            targets[0],
            label,
            masks[0],
            output["roi_class_weights"],
            positive_roi_loss_weight=0.25,
        )
        loss = branch_loss + 0.1 * delta["loss"]
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.8, (losses[0], losses[-1])


if __name__ == "__main__":
    test_schedule_builds_anchor_context_edges_without_gt()
    test_anchor_cycle_covers_every_foreground_class()
    test_targets_obey_additive_identity_for_present_and_absent_anchors()
    test_delta_loss_supervises_present_and_absent_queries()
    test_triplet_model_forward_backward()
    test_triplet_tiny_batch_overfit()
    print("PSEM-v3 triplet Loss3 tests passed")
