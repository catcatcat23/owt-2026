# 历史原文档案（按需查询）

来源提交：`300b53391e1a830c6ac0b0bd80b385f620c0c1b8`。
原文逐字保留；所有状态、指令和路径仅反映各自历史时点。
不要默认读取本文件；先读 ../README.md。

- [EXPERIMENT_SUMMARY.md](#source-1)

<a id="source-1"></a>

## EXPERIMENT_SUMMARY.md

SHA256: `29f272da3b35ad116971171441d654dc74edcd411a82581ff20d341a83fc8901`

````text
# OWT Experiment Summary

Last updated: 2026-07-19

This document is the primary experiment ledger for this repository. CSV files are
kept as machine-readable raw results, while experiment settings, conclusions, and
comparisons are maintained here.

## 1. Current Experiment Scope

The completed experiments are **joint full-class OWT reconstruction baselines**.
They are not class-incremental experiments.

Current training protocol:

```text
All class labels are available simultaneously
-> train OWT reconstruction model
-> inspect token-conditioned reconstruction (Step 2)
-> derive pseudo-segmentation from reconstruction signals (Step 3)
```

The following components have not yet been implemented or evaluated:

- old/new class splits during training;
- sequential 4-4, 4-2, or 2-2 stages;
- hiding old-class labels in later stages;
- frozen previous model;
- knowledge distillation;
- prototype-based incremental constraints;
- Old/New/All incremental metrics.

### Experiment objectives

1. Establish a non-incremental full-class reference for the existing OWT code.
2. Test whether organ tokens contain class-specific reconstruction information.
3. Evaluate reconstruction-derived pseudo-segmentation on AbdAutoPet, WORD, and BTCV.
4. Compare 2D OWT with Fixfr4 OWT.
5. Diagnose complete misses and instability on small organs.

## 2. Datasets

| Dataset | Classes | Train cases | Test cases | Train samples | Test slices | Notes |
|---|---:|---:|---:|---:|---:|---|
| AbdAutoPet | 4 | 700 | 200 | 78,400 slices | 22,400 | Original OWT-formatted JPG/PNG data; 112 slices per case |
| WORD Common8 | 8 | 96 | 24 | 19,078 slices | 4,665 | Paper-aligned 8:2 split, seed 42 |
| BTCV Common8 | 8 | 24 | 6 | 3,588 slices | 938 | Paper-aligned 8:2 split, seed 42 |

Common8 class order:

| ID | Class |
|---:|---|
| 1 | spleen |
| 2 | right kidney |
| 3 | left kidney |
| 4 | gallbladder |
| 5 | esophagus |
| 6 | pancreas |
| 7 | liver |
| 8 | stomach |

The AbdAutoPet files contain labels 1-4, but the repository does not contain a
reliable label-ID-to-organ-name mapping. AbdAutoPet results therefore retain the
names `label_1` through `label_4`.

## 3. Preprocessing

### WORD and BTCV Common8

```text
NIfTI image and mask
-> canonical RAS orientation
-> resample to 1 x 1 x 3 mm^3
-> image: linear interpolation
-> mask: nearest-neighbor interpolation
-> HU clipping [-175, 250]
-> fixed normalization to [0, 1]
-> center crop/pad to 512 x 512 without foreground loss
-> resize to 224 x 224
-> image: JPEG quality 95
-> mask: lossless uint8 PNG
```

Training uses `--intensity_norm fixed_255`, so the saved HU-windowed image is
divided by 255 without a second per-slice min-max normalization.

### Preprocessing integrity audit

The 512-to-224 resize did not remove any positive test slices and preserved the
expected area after correcting for the spatial scale factor.

| Dataset | Class | Positive slices before resize | Positive slices after resize | Lost slices |
|---|---|---:|---:|---:|
| WORD | gallbladder | 243 | 243 | 0 |
| WORD | esophagus | 570 | 570 | 0 |
| BTCV | gallbladder | 113 | 113 | 0 |
| BTCV | esophagus | 160 | 160 | 0 |

The labels are therefore not disappearing during preprocessing. However, the
full-field resize reduces the effective in-plane resolution to approximately
2.286 mm per pixel. At 224 x 224, the average foreground area per positive WORD
slice is only about 68 pixels for gallbladder and 39 pixels for esophagus. Both
are substantially smaller than one 16 x 16 ViT patch (256 pixels), which can
weaken their feature representation.

## 4. Model and Training Settings

### Architecture

Current `v11` models use the ViT branch in `OWT_models.py`:

```text
PatchEmbed / PatchEmbed3Dfix
-> six BlockLA image-encoding blocks
-> OrganCollector / OrganCollector3D
-> organ token groups
-> six shared BlockLA token-group blocks
-> AHER spatial recovery
-> eight BlockLA reconstruction-decoder blocks
-> patch prediction and image reconstruction
```

This is a custom MAE-ViT with Linear Attention. It is not the 3D hierarchical
Swin UNETR used by the incremental-segmentation reference paper.

### Shared hyperparameters

| Parameter | Value |
|---|---|
| Architecture version | `v11` |
| Model | ViT-Base, patch size 16, Linear Attention |
| Input size | 224 x 224 |
| Embedding dimension | 768 |
| Transformer depth | 6 image blocks + 6 token-group blocks |
| Decoder | AHER + 8 reconstruction blocks |
| Token factor | 20 tokens per class group |
| Base learning rate | `1e-4` |
| Weight decay | `0.05` |
| Epochs | 1,200 |
| Warm-up epochs | 60 |
| Mask ratio argument | `1.0` |
| Loss | full-image L2 + LPIPS |
| GPUs | 2 |

The 8-class Common8 model has 180 tokens: 20 background tokens plus 8 x 20
organ tokens. The 4-class AbdAutoPet model has 100 tokens.

### Experiment-specific settings

| Experiment | Input | Batch size | Normalization | Checkpoint |
|---|---|---:|---|---|
| AbdAutoPet 2D | 1 x 224 x 224, repeated to RGB | 96 | per-sample min-max | epoch 1199 |
| WORD 2D | 1 x 224 x 224, RGB | 32 | fixed `/255` | epoch 1199 |
| WORD 3D | 4 x 224 x 224, Fixfr4, TS1 | 32 | fixed `/255` | epoch 1199 |
| BTCV 2D | 1 x 224 x 224, RGB | 48 | fixed `/255` | epoch 1199 |
| BTCV 3D | 4 x 224 x 224, Fixfr4, TS1 | 32 | fixed `/255` | epoch 1199 |

Final epoch global training losses:

| Experiment | L2 (`train_loss`) | LPIPS (`train_p_loss`) | Approx. optimized sum |
|---|---:|---:|---:|
| WORD 2D | 0.000730 | 0.016920 | 0.017650 |
| WORD 3D | 0.000694 | 0.016783 | 0.017477 |
| BTCV 2D | 0.004199 | 0.061805 | 0.066003 |
| BTCV 3D | 0.003873 | 0.059084 | 0.062957 |

These are global full-image losses. The current training code does not log a
separate loss for each organ.

## 5. Evaluation Protocol

### Step 2: Token-conditioned reconstruction

Step 2 produces normalized CT reconstruction signals, not class probabilities.

| Mode | Tokens retained | Reconstruction target |
|---|---|---|
| `whole` | background and all organ tokens | complete CT image |
| `background_only` | background tokens | background CT with organs zeroed |
| `organs_only` | all organ tokens | organ CT with background zeroed |
| `class_c_only` | only class-c tokens | class-c CT region with all other pixels zeroed |

Reported metrics are L1, L2, LPIPS, PSNR, and SSIM. Thresholded variants first
set reconstruction values below 0.02 to zero.

Important limitation: class-only targets are extremely sparse, but current L1,
L2, PSNR, and SSIM are averaged over the complete volume. A nearly zero output
can therefore receive apparently excellent metrics while failing to reconstruct
the organ. Class-only full-volume metrics must not be interpreted as organ ROI
reconstruction quality.

### Step 3: Reconstruction-derived pseudo-segmentation

Two score maps are evaluated:

```text
Direct(c)   = reconstruction using only class-c tokens
Indirect(c) = max(input - reconstruction without class-c tokens, 0)
```

The score map is thresholded at 0.15. Common8 uses a minimum connected-component
size of 100 voxels; AbdAutoPet uses 20. Reported segmentation metrics are Dice,
NSD, and HD95 before and after morphology-based postprocessing.

This is not a learned softmax segmentation head.

## 6. Step 2 Reconstruction Results

### Whole-image reconstruction

| Dataset | Model | L1 | L2 | LPIPS | PSNR | SSIM |
|---|---|---:|---:|---:|---:|---:|
| WORD | 2D | 0.00691 | 0.000314 | 0.01358 | 35.31 | 0.9660 |
| WORD | 3D Fixfr4 | 0.00722 | 0.000321 | 0.01692 | 35.25 | 0.9535 |
| BTCV | 2D | 0.03318 | 0.006261 | 0.10914 | 22.32 | 0.7195 |
| BTCV | 3D Fixfr4 | 0.03363 | 0.006260 | 0.15078 | 22.26 | 0.6977 |
| AbdAutoPet | 2D | 0.00686 | 0.000157 | not evaluated | 38.42 | 0.9196 |

WORD whole-image reconstruction is substantially better than BTCV. Fixfr4 does
not improve whole-image reconstruction over 2D in the current checkpoints.

### Organs-only reconstruction

| Dataset | Model | L1 | L2 | LPIPS | PSNR | SSIM |
|---|---|---:|---:|---:|---:|---:|
| WORD | 2D | 0.00247 | 0.000679 | 0.01299 | 55.25 | 0.9848 |
| WORD | 3D Fixfr4 | 0.00224 | 0.000592 | 0.01364 | 59.44 | 0.9852 |
| BTCV | 2D | 0.00999 | 0.003361 | 0.03830 | 38.50 | 0.9037 |
| BTCV | 3D Fixfr4 | 0.00816 | 0.002694 | 0.04017 | 42.95 | 0.9271 |
| AbdAutoPet | 2D | 0.00600 | 0.001322 | not evaluated | 32.66 | 0.9115 |

Fixfr4 improves the aggregated organs-only L1/L2, PSNR, and SSIM for WORD and
BTCV, indicating that short through-plane context helps overall organ-region
reconstruction.

### Small-organ class-only reconstruction

| Dataset | Model | Class | L1 | L2 | LPIPS | PSNR | SSIM |
|---|---|---|---:|---:|---:|---:|---:|
| WORD | 3D | gallbladder | 0.000161 | 0.0000167 | 0.00217 | 75.40 | 0.9996 |
| WORD | 3D | esophagus | 0.000165 | 0.0000195 | 0.00459 | 72.84 | 0.9994 |
| BTCV | 3D | gallbladder | 0.001022 | 0.0000870 | 0.00576 | 57.97 | 0.9918 |
| BTCV | 3D | esophagus | 0.000932 | 0.0000277 | 0.00637 | 57.21 | 0.9916 |

These high PSNR/SSIM values are dominated by the zero background and are not
evidence of successful small-organ reconstruction. All four corresponding Direct
Dice values are zero in Step 3.

## 7. Step 3 Pseudo-segmentation Results

### Macro Dice

Postprocessed class-macro Dice over all target classes:

| Dataset | Model | Direct | Indirect | Better extraction mode |
|---|---|---:|---:|---|
| AbdAutoPet | 2D | 89.94 | 90.39 | Indirect |
| WORD | 2D | 55.07 | 58.31 | Indirect |
| WORD | 3D Fixfr4 | 59.93 | 60.94 | Indirect |
| BTCV | 2D | 10.39 | 10.43 | Neither is reliable |
| BTCV | 3D Fixfr4 | 34.16 | 12.24 | Direct |

### WORD per-class Dice

| Class | 2D Direct | 2D Indirect | 3D Direct | 3D Indirect |
|---|---:|---:|---:|---:|
| spleen | 86.00 | 88.31 | 89.24 | 89.92 |
| right kidney | 79.30 | 86.83 | 89.49 | 89.05 |
| left kidney | 79.98 | 85.55 | 88.25 | 87.87 |
| gallbladder | 0.00 | 0.00 | 0.00 | 0.00 |
| esophagus | 0.00 | 0.00 | 0.00 | 0.00 |
| pancreas | 38.43 | 43.73 | 47.61 | 54.12 |
| liver | 91.02 | 92.28 | 92.02 | 93.15 |
| stomach | 65.84 | 69.78 | 72.81 | 73.44 |

WORD learns the large organs well. Fixfr4 improves kidney, pancreas, and stomach
performance, but gallbladder and esophagus remain completely unresolved.

### BTCV per-class Dice

| Class | 2D Direct | 2D Indirect | 3D Direct | 3D Indirect |
|---|---:|---:|---:|---:|
| spleen | 0.00 | 8.27 | 60.07 | 14.05 |
| right kidney | 0.00 | 6.82 | 34.68 | 8.85 |
| left kidney | 0.00 | 6.68 | 37.24 | 8.67 |
| gallbladder | 0.00 | 0.25 | 0.00 | 0.27 |
| esophagus | 0.00 | 0.39 | 0.00 | 0.27 |
| pancreas | 0.00 | 0.90 | 8.59 | 0.77 |
| liver | 83.11 | 51.48 | 85.07 | 50.81 |
| stomach | 0.00 | 8.68 | 47.61 | 14.20 |

BTCV 2D collapses primarily to liver. Fixfr4 partially recovers spleen, kidneys,
pancreas, and stomach in Direct mode, but performance remains far below a usable
segmentation baseline.

### AbdAutoPet per-class Dice

| Class | Direct mean Dice | Indirect mean Dice | Direct median | Indirect median |
|---|---:|---:|---:|---:|
| label 1 | 96.45 | 96.51 | 96.70 | 96.76 |
| label 2 | 92.69 | 92.93 | 93.24 | 93.51 |
| label 3 | 92.94 | 93.04 | 94.67 | 94.72 |
| label 4 | 77.66 | 79.06 | 80.66 | 81.97 |

For the smallest AbdAutoPet class (`label_4`):

- no positive case produced an entirely empty prediction;
- Direct had 8/200 cases with Dice below 0.5;
- Indirect had 6/200 cases with Dice below 0.5;
- worst Dice was approximately 0.22;
- small-organ instability is present, but systematic complete misses are not.

## 8. Small-organ Failure Analysis

### Observed complete misses

| Dataset/model | Mode | Observation |
|---|---|---|
| WORD 2D | Direct | gallbladder and esophagus raw predictions empty in 24/24 cases |
| WORD 3D | Direct | gallbladder and esophagus raw predictions empty in 24/24 cases |
| WORD 2D | Indirect | postprocessing empties gallbladder in 23/24 and esophagus in 22/24 cases |
| WORD 3D | Indirect | postprocessing empties both classes in 23/24 cases |
| BTCV 2D | Direct | all classes except liver raw-empty in 6/6 cases |
| BTCV 3D | Direct | gallbladder and esophagus raw-empty in 6/6 cases |
| AbdAutoPet 2D | Direct/Indirect | no empty prediction in any GT-positive case |

WORD Indirect produces non-empty raw signals, but they are spatially incorrect:
raw Dice is approximately 0.0003 for gallbladder and 0.008 for esophagus. The
100-voxel component filter exposes rather than creates the underlying failure.

### Likely causes, ordered by current evidence

1. **Global reconstruction-loss imbalance.** Small organs contribute only a tiny
   fraction of the full-image L2 and LPIPS objectives.
2. **Low effective spatial resolution.** Full-field 224 x 224 input followed by
   16 x 16 patches makes small organs sub-patch structures.
3. **No small-organ foreground sampling.** Most slices do not contain the target
   organ, and the loader samples all slices/windows uniformly.
4. **Weak token identifiability.** OrganCollector has no direct mask-alignment
   supervision, so small-organ information may leak into background or other
   organ token groups.
5. **Shared TGEnc and AHER.** All classes share token processing and spatial
   recovery, allowing large classes to dominate optimization.
6. **Fixed reconstruction threshold.** The 0.15 score threshold is not a calibrated
   class probability and may suppress weak class-specific signals.
7. **Postprocessing.** The 100-voxel filter removes scattered Indirect artifacts,
   but it is not the root cause of Direct raw-empty predictions.

## 9. Relationship to the Incremental-Segmentation Paper

The reference paper uses a supervised 3D Swin UNETR segmentation network with
96 x 96 x 96 training crops, voxel-wise class logits, and explicit incremental
training stages. Current OWT uses a short Fixfr4 MAE-ViT reconstruction model and
derives segmentation from reconstruction amplitudes.

The current results must therefore not be reported as 4-4 or 2-2 incremental
results. Even comparison with the paper's Offline result is only an external
scale reference, not a controlled architecture comparison.

| Dataset | Current best 3D OWT macro Dice | Paper Offline Dice | Uncontrolled gap |
|---|---:|---:|---:|
| BTCV | 34.16 | 90.75 | -56.59 points |
| WORD | 60.94 | 85.47 | -24.53 points |

## 10. Current Conclusions

1. WORD OWT reconstructs complete CT images reasonably well and learns large
   organs, but does not form usable gallbladder or esophagus token signals.
2. BTCV is limited by only 24 training cases and shows broad class collapse,
   especially in 2D.
3. Fixfr4 improves aggregate organ reconstruction and several medium/large-organ
   Dice scores, but four slices are insufficient to solve the smallest classes.
4. AbdAutoPet demonstrates that reconstruction-based organ tokens can produce
   useful pseudo-segmentation when classes are larger, frequent, and supported by
   substantially more training data.
5. Current class-only PSNR/SSIM metrics are biased by zero background and should
   not be used as evidence of small-organ reconstruction quality.
6. No claim about incremental-learning performance is supported yet.

## 11. Required Next Evaluations

- Add foreground-only ROI-L1 and ROI-L2 for class-only reconstruction.
- Add positive-slice and organ-bounding-box PSNR/SSIM.
- Measure reconstruction energy recovered inside the organ and leakage outside it.
- Sweep class-specific reconstruction thresholds before changing the model.
- Inspect OrganCollector attention overlap with ground-truth masks.
- Run a small-organ-positive subset overfit test to separate optimization failure
  from dataset-scale imbalance.
- Establish a supervised segmentation reference under the same preprocessing.
- Only after validating the joint baseline, implement and evaluate matched 4-4
  and 2-2 incremental protocols.

### Small-organ and low-data intervention plan

Increasing the spatial grid from 14 x 14 (196 tokens) to 16 x 16 (256 tokens) is a valid minimal ablation, but it is not expected to solve the current complete misses by itself. With full-field preprocessing, this changes the physical patch width only from approximately 36.6 mm to 32 mm and increases small-organ pixels by about 1.31 times. The global reconstruction-loss imbalance remains unchanged.

Two implementations must be distinguished:

1. Resizing the existing 224 JPG files to 256 in the DataLoader creates no new anatomical information and is not a valid high-resolution experiment.
2. Regenerating 256 x 256 images directly from the 512 x 512 resampled NIfTI data, then training with input size 256 and patch size 16, is the valid 256-token ablation.

Recommended priority:

1. Replace full-field downscaling with native-resolution foreground-centered 224 x 224 crops. This retains the same 196-token cost while reducing physical patch width to approximately 16 mm. Use sliding-window aggregation at test time.
   Do not replace each volume with one fixed crop. Mix low-resolution global views with dynamically jittered native-resolution foreground and random crops, supervise only classes present in each local crop, and ensure whole-volume coverage across epochs. A 224 mm field of view retains local context but not the entire abdomen.
   Generate crops on the fly rather than materializing many duplicate files. For 2D, apply exactly the same crop and geometric transform to the image and mask. For Fixfr4/3D, use one shared XY crop across every slice in the window and crop image/mask with identical XYZ coordinates. These crops are correlated views of the same case, not additional independent patients, so dataset splits and cross-validation must remain case-wise.
2. Add class-balanced ROI reconstruction loss, averaged independently inside each present organ rather than over the complete image.
3. Use class-uniform positive-slice/window sampling, with explicit oversampling of gallbladder, esophagus, and pancreas.
4. Add an OrganCollector attention-alignment loss or an organ-level binary head so that each token group is explicitly tied to its class mask.
5. Run the regenerated 256 x 256, patch-16 experiment as a controlled resolution ablation; evaluate patch-8 or a high-resolution branch only after the loss and sampling problems are addressed.
6. For BTCV, initialize from the WORD Common8 checkpoint and fine-tune with a lower learning rate, or train on WORD and BTCV jointly with dataset-balanced sampling.
7. Report BTCV results with cross-validation because the current six-case test split has high variance.

The resolution ablation must retain the same split, model depth, token factor, loss, epochs, and evaluation protocol so that any improvement can be attributed to spatial resolution rather than a simultaneous training change.

## 12. Result Artifacts

- AbdAutoPet evaluation: `Results/AbdAutoPet_Eval/2D/OWT-Joint-C4-Recon_ckpt1199/`
- WORD 2D evaluation: `Results/Common8_Eval/WORD_2D/OWT-Joint-C8-Recon_ckpt1199/`
- WORD 3D evaluation: `Results/Common8_Eval/WORD_3D_Fixfr4/OWT-Joint-C8-Recon3D_ckpt1199/`
- BTCV 2D evaluation: `Results/Common8_Eval/BTCV_2D/OWT-Joint-C8-Recon_ckpt1199/`
- BTCV 3D evaluation: `Results/Common8_Eval/BTCV_3D_Fixfr4/OWT-Joint-C8-Recon3D_ckpt1199/`

The per-case CSV files remain raw evidence. Future experiment interpretation and
status updates should be added to this document.

## 13. CropMix-v1 Small-Organ Experiment

### Purpose

Test whether preserving native in-plane detail and increasing the frequency of
small-organ-positive views improves WORD gallbladder, esophagus, and pancreas
reconstruction/segmentation without changing the OWT architecture or test input.

### Training intervention

- Source data: WORD resampled to 1 x 1 x 3 mm, HU clipped to [-175, 250],
  mapped to [0, 1], and stored on a 512 x 512 canvas.
- Model input remains 224 x 224; patch size remains 16 and the grid remains
  14 x 14 = 196 patches.
- View mixture: 50% complete 512 x 512 images resized to 224 x 224 and 50%
  native-resolution 224 x 224 XY crops.
- Crop center: a present organ, with priority given to Common8 classes 4, 5,
  and 6 (gallbladder, esophagus, pancreas) and up to 32-pixel XY jitter.
- Focus sampling: 80% of local-crop requests are redirected to a slice/window
  containing at least one focus class.
- 3D geometry: all four Fixfr4 slices and their masks use one shared XY crop.
- A local crop's focus class is forced into that iteration's reconstructed
  organ-token set; at least one class always remains visible.
- Intensity handling: complete-view resizing is clipped to [0, 1] after cubic
  interpolation but is not per-sample min-max normalized.

This is a combined intervention (resolution-preserving crop, focused sampling,
and focus-token selection), not a single-factor crop ablation.

### Controlled settings

The Common8 mapping, case split, model architecture, token factor 20, learning
rate 1e-4, 1200 epochs, full 224 x 224 test input, thresholds, and Step2/Step3
evaluation code remain matched to the existing WORD baseline.

### Slurm chain submitted 2026-07-20

| Stage | 2D Job | 3D Job | Dependency |
|---|---:|---:|---|
| HR512 preprocessing | 1552981 | shared | none |
| 1-epoch smoke | 1552982 | 1552983 | preprocessing success |
| 1200-epoch full training | 1552984 | 1552985 | corresponding smoke success |
| Step2/Step3 evaluation | 1554939 | 1554940 | corresponding full training success |

At submission time, Slurm accepted five jobs and then reached
`QOSMaxSubmitJobPerUserLimit`. The evaluation jobs were submitted after the
smoke jobs released their slots.

Progress check on 2026-07-20 at approximately 21:31 CST:

- preprocessing and both 1-epoch smoke jobs completed successfully;
- 2D full training was in epoch 326/1200 with L2 0.0052, LPIPS 0.0851,
  crop fraction 0.492, and focus-crop fraction 0.432;
- 3D full training was in epoch 225/1200 with L2 0.0057, LPIPS 0.0956,
  crop fraction 0.499, and focus-crop fraction 0.437;
- both Slurm error logs were empty, with no observed OOM or NaN;
- estimated remaining time was about 39 hours for 2D and 49-55 hours for 3D,
  subject to epoch-time variation and checkpoint I/O.

## 14. LossBalance-v1 Reconstruction-Loss Variant

### Isolation boundary

LossBalance-v1 is implemented as a separate training version. It does not edit
or import-switch the legacy `main_pretrain.py`, `engine_pretrain.py`, or
`OWT_models.py` pipeline. The model subclass adds no parameters, so its state
dictionary remains architecture-compatible with legacy OWT checkpoints.

### Loss

```text
L_total = L_global_L2
        + lambda_roi * L_per_sample_per_organ_ROI_L2
        + lambda_lpips * L_LPIPS
```

For each sample, squared reconstruction error is averaged independently inside
each present foreground-organ mask. Those organ losses are averaged so organ
area does not determine gradient weight. Samples with no foreground organ skip
the ROI term. Background is excluded by default (`background_weight = 0`).
No Dice, BCE, CE, classification head, or segmentation head is added.

Default ablation settings are `lambda_roi = 1.0`, `lambda_lpips = 1.0`, and
`background_weight = 0.0`. The loss-only Slurm scripts use the original 224 x
224 WORD data and disable CropMix, isolating loss accounting from resolution
and sampling changes.

### Files

- `util/organ_balanced_loss.py`: vectorized 2D/3D organ-balanced ROI-L2.
- `OWT_models_lossbalance.py`: parameter-free subclass overriding loss/forward.
- `engine_pretrain_lossbalance.py`: passes masks and logs ROI accounting.
- `main_pretrain_lossbalance.py`: independent entry point and loss arguments.
- `tests/test_organ_balanced_loss.py`: area-invariance and Fixfr4 tests.
- `slurm/pretrain_lossbalance_word_2d.sbatch`: isolated WORD 2D run.
- `slurm/pretrain_lossbalance_word_3d.sbatch`: isolated WORD Fixfr4 run.

Synthetic verification produced global L2 0.4444, organ-balanced ROI-L2 2.5,
and total reconstruction loss 2.9444 for a hand-computable example. Both 2D
and 3D tests pass with finite gradients. Slurm training has not yet been
submitted because the four CropMix GPU jobs occupy the current QOS submission
limit.
````
