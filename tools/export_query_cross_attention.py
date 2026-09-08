"""Export real checkpoint attention and fixed-feature interventions, not causal proof.

Default selection: first two GT-positive dataset rows for each requested organ.
Selection never uses predicted quality. Raw per-head attention is saved in NPZ.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def render(path, image, target, attention, probabilities, caption):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 6, figsize=(18, 3.4), constrained_layout=True)
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("CT (model input)")
    axes[1].imshow(image, cmap="gray")
    axes[1].imshow(np.ma.masked_where(~target, target), cmap="winter", alpha=0.6,
                   vmin=0, vmax=1)
    axes[1].set_title("Ground truth")
    # Uniform attention equals 1 after multiplying by the number of positions.
    lift = attention.mean(0) * attention.shape[-1] * attention.shape[-2]
    axes[2].imshow(image, cmap="gray")
    heat = axes[2].imshow(lift, cmap="magma", alpha=0.75,
                         extent=(-0.5, image.shape[1]-0.5, image.shape[0]-0.5, -0.5),
                         vmin=0, vmax=max(1.0, float(lift.max())))
    axes[2].set_title("Attention / uniform")
    fig.colorbar(heat, ax=axes[2], fraction=0.046)
    for ax, mode in zip(axes[3:], ("normal", "uniform", "bypass")):
        ax.imshow(image, cmap="gray")
        ax.imshow(probabilities[mode], cmap="viridis", alpha=0.7, vmin=0, vmax=1)
        ax.set_title(mode + " probability")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(caption, fontsize=10)
    fig.savefig(str(path) + ".png", dpi=300)
    fig.savefig(str(path) + ".pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--class-map", default="configs/orgslot/common8_offline.json")
    parser.add_argument("--class-ids", default="4,5,6")
    parser.add_argument("--samples-per-organ", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    from tools.eval_common8_orgslot_reconstruction_threshold import (
        build_model, build_dataset, parse_class_configuration, checkpoint_value, sha256,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint_value(checkpoint, "query_refinement", "none") != "cross_attn":
        raise ValueError("requires a trained cross_attn checkpoint, not original Arm E")
    size = int(checkpoint_value(checkpoint, "input_size", 448))
    specs, names = parse_class_configuration(args.class_map)
    model, report = build_model("orgslot", checkpoint, specs, size,
                               checkpoint_value(checkpoint, "fusion_mode", "linear_sqrt"))
    model.to(args.device).eval()
    dataset = build_dataset(args.data_csv, size, size)
    wanted = [int(v) for v in args.class_ids.split(",")]
    counts = {c: 0 for c in wanted}
    records = []
    decoder = model.pixel_query_decoder
    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]
            label = sample["full_label"].squeeze().numpy()
            chosen = [c for c in wanted if counts[c] < args.samples_per_organ
                      and bool((label == c).any())]
            if not chosen:
                continue
            image = sample["image"].unsqueeze(0).to(args.device)
            z, _ = model.forward_encoder(image)
            pixels = decoder.forward_pixels(image, z)
            for c in chosen:
                slot = model.slot_bank.get_slot(names[c])
                _, tokens, _, _ = slot.forward_canvas(z)
                query = decoder.forward_query(tokens, slot.head.embedding)
                probabilities = {}
                for mode in ("normal", "uniform", "bypass"):
                    updated, weights = decoder.query_cross_attention(query, pixels, mode)
                    logits = decoder.forward_mask_from_query(pixels, updated, image.shape[-2:])
                    calibrated = slot.calibration_scale * logits + slot.calibration_bias
                    probabilities[mode] = calibrated.sigmoid()[0, 0].cpu().numpy()
                    if mode == "normal":
                        attention = weights[0].cpu().numpy()
                target = label == c
                gt_low = F.adaptive_avg_pool2d(
                    torch.from_numpy(target.astype("float32"))[None, None],
                    attention.shape[-2:],
                )[0, 0].numpy()
                mass = float((attention.mean(0) * gt_low).sum())
                area = float(gt_low.mean())
                metrics = {"dataset_index": index, "class_id": c, "organ": names[c],
                           "gt_pixels": int(target.sum()), "attention_gt_mass": mass,
                           "gt_area_fraction": area, "attention_gt_lift": mass / max(area, 1e-12)}
                for mode, prob in probabilities.items():
                    pred = prob >= 0.5
                    metrics[mode + "_dice"] = float(2 * (pred & target).sum()
                                                       / max(1, pred.sum() + target.sum()))
                    metrics[mode + "_mean_abs_probability_change"] = float(
                        np.abs(prob - probabilities["normal"]).mean())
                stem = output / ("row{}_{}".format(index, names[c]))
                ct = image[0, 0].cpu().numpy()
                np.savez_compressed(str(stem) + ".npz", image=ct, gt=target,
                                    attention_per_head=attention, **probabilities)
                render(stem, ct, target, attention, probabilities,
                       "row {} | {} | checkpoint epoch {} | fixed-feature diagnostics, not causal proof".format(
                           index, names[c], checkpoint.get("epoch")))
                records.append(metrics)
                counts[c] += 1
            if all(n >= args.samples_per_organ for n in counts.values()):
                break
    manifest = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"), "load_report": report,
        "data_csv": str(Path(args.data_csv).resolve()), "data_csv_sha256": sha256(args.data_csv),
        "selection": "first GT-positive rows per organ; no prediction-based selection",
        "requested_per_organ": args.samples_per_organ, "actual_counts": counts,
        "records": records,
        "limitations": "Attention visualizes spatial routing of content, not absolute-position encoding or causal localization proof. Bypass includes FFN removal; uniform retains FFN. Diagnostic slice Dice is not full-case evaluation.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if not all(n >= args.samples_per_organ for n in counts.values()):
        raise RuntimeError("not enough GT-positive examples; inspect partial manifest")


if __name__ == "__main__":
    main()
