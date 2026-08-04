from types import SimpleNamespace

import torch
import torch.nn as nn

from OWT_models_psem_lossbalance_v3 import StateSeparatedLossMaskedAutoencoderViT


def _model():
    args = SimpleNamespace(
        arch_version="v11", dataset_type="2D", temp_stride=0, fix_frame=0,
        LA=True, organ_token_total=6, token_factor=2,
        num_classes_with_bg=3, loss_version=["L2"], text_encoding="None",
        roi_class_weights=[0.0, 1.0, 2.0], positive_roi_loss_weight=0.25,
    )
    return StateSeparatedLossMaskedAutoencoderViT(
        img_size=32, patch_size=16, in_chans=3, embed_dim=32, depth=2,
        num_heads=4, decoder_embed_dim=32, decoder_depth=1,
        decoder_num_heads=4, mlp_ratio=2, norm_layer=nn.LayerNorm,
        model_args=args,
    )


def test_forward_backward_uses_per_sample_queries_and_positive_state_only():
    torch.manual_seed(4)
    model = _model()
    image = torch.rand(2, 3, 32, 32)
    label = torch.zeros(2, 3, 32, 32, dtype=torch.long)
    label[0, :, :16] = 1
    label[1, :, 16:] = 2
    keep = torch.tensor([[True, True, False], [False, False, True]])
    target = image.clone()
    for index in range(2):
        for class_id in range(3):
            if not keep[index, class_id]:
                target[index][label[index] == class_id] = 0
    loss, pred, output = model(
        image,
        middle={"image_target": target, "label": label, "class_keep_mask": keep},
    )
    loss.backward()
    assert pred.shape == image.shape
    assert torch.isfinite(loss)
    assert output["valid_token_mask"].sum(dim=1).tolist() == [4, 2]
    assert output["positive_class_counts"].tolist() == [0, 1, 1]
    assert model.patch_embed.proj.weight.grad is not None


def test_tiny_batch_overfit():
    torch.manual_seed(5)
    model = _model()
    image = torch.rand(1, 3, 32, 32)
    label = torch.ones(1, 3, 32, 32, dtype=torch.long)
    keep = torch.tensor([[True, True, False]])
    middle = {
        "image_target": image,
        "label": label,
        "class_keep_mask": keep,
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0)
    losses = []
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = model(image, middle=middle)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.8, (losses[0], losses[-1])


if __name__ == "__main__":
    test_forward_backward_uses_per_sample_queries_and_positive_state_only()
    test_tiny_batch_overfit()
    print("PSEM-v2 + LossBalance-v3 test passed")
