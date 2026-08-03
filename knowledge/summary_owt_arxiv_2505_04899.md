# OWT arXiv 2505.04899v2: paper-to-reproduction audit

## Paper

- Title: *OWT: A Foundational Organ-Wise Tokenization Framework for Medical Imaging*.
- Version audited: arXiv v2, revised 2025-11-19.
- OWT uses a ViT-base image encoder, Organ Collector, six-block Token Group
  Encoder, AHER, and an eight-block decoder. Each of five groups (background,
  liver, kidneys, spleen, pancreas) has 20 tokens.
- Token Group Reconstruction randomly retains semantic groups and supervises
  reconstruction of the corresponding image regions with L2 + LPIPS.

## Paper AutoPet protocol

- 700 training and 200 test volumes; each volume standardized to 112 slices.
- One-slice and four-slice variants are reported.
- 1200 epochs, 60 warm-up epochs; batch sizes 96 (2D) and 64 (4-slice).
- Reconstruction threshold: 0.02. Segmentation threshold: 0.15.
- Segmentation is reconstruction-derived thresholding, not a learned head.
- Table 13 AutoPet 2D Direct Dice: liver 96.14, kidney 92.33, spleen 92.79,
  pancreas 76.82, macro 89.52.
- Table 4 AutoPet 2D reconstruction: holistic L2 1.49e-4, LPIPS 0.0232,
  SSIM-3D 0.9881; one-token-group L2 3.32e-4, LPIPS 0.0071,
  SSIM-3D 0.9778.
- Four-slice OWT reports macro Dice 91.41 and holistic L2 1.13e-4.
- The VQGAN-backbone ablation reports AutoPet holistic L2 6.89e-5 and LPIPS
  0.0073.

## Local original-OWT reproduction

The local checkpoint is epoch 1199 and was evaluated on the same 200 cases
(22,400 slices). Recomputing the corrupted merged summary from all 1,600
per-case method/class records gives raw Direct Dice:

| Organ/label | Paper | Local raw Direct | Difference (local - paper) |
|---|---:|---:|---:|
| Liver / label 1 | 96.14 | 96.224 | +0.084 |
| Kidney / label 2 | 92.33 | 92.204 | -0.126 |
| Spleen / label 3 | 92.79 | 92.484 | -0.306 |
| Pancreas / label 4 | 76.82 | 77.089 | +0.269 |
| Macro | 89.52 | 89.501 | -0.019 |

The local postprocessed Direct macro Dice is 89.938, but it includes 3D binary
opening and removal of components smaller than 20 voxels, so it must not be
used as the paper-table comparison. The raw thresholded result is the matched
quantity.

Local holistic L2 is 1.566e-4 versus 1.49e-4 in the paper (absolute difference
7.64e-6; local is 5.1% higher). Averaging the four local class-only L2 values
gives about 3.67e-4 versus the paper's 3.32e-4 (about 10.6% higher).

## Comparison caveats

- Local SSIM is slice-wise 2D Gaussian-window SSIM (0.9196 holistic), whereas
  the paper reports SSIM-3D (0.9881); these are not directly comparable.
- LPIPS was skipped during the local test-set evaluation. The final training
  LPIPS (0.0199) is not a substitute for the paper's test LPIPS (0.0232).
- The local loader records `per_sample` min-max normalization; the paper
  describes fixed HU clipping followed by normalization. This is a pipeline
  discrepancy even though Dice and L2 reproduce closely.
- The paper table's segmentation value is the Direct, single-token-group,
  threshold-derived mask. Local Indirect and morphology-postprocessed scores
  are additional analyses, not the primary paper-table quantity.
- No completed local original-OWT AutoPet four-slice checkpoint/evaluation was
  found, so the paper's 91.41 four-slice Dice has not yet been reproduced.

## Verdict

The local AutoPet 2D original OWT reproduces the paper's main segmentation
result essentially exactly: 89.501 versus 89.52 macro Dice. Holistic L2 is also
close (1.566e-4 versus 1.49e-4), though slightly worse. SSIM and LPIPS cannot be
claimed as reproduced under the current evaluator because their protocols do
not match the paper.
