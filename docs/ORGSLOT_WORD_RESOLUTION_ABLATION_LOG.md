# OrganSlot WORD Resolution Ablation Log

Last updated: 2026-08-12 18:43 CST

## Objective

Test whether the large zero/black field in the 1.0×1.0×2.0 mm, 448×448
pipeline wastes model capacity. Keep the OrganSlot architecture, Loss3, optimizer,
effective batch, update budget, split, threshold, and evaluation modes fixed.

## Code provenance

- Branch: `experiment/orgslot-resolution-ablation-v0`
- Commit: `4fd6260`
- Base local Loss3/ROI line: `749cbec`
- Merged latest remote OrganSlot fixed-reference support through `fdca982`
- XEC account/worktree:
  `sifansong:/gpfs/work/aac/sifansong/worktrees/orgslot_resolution`
- XEC environment:
  `/gpfs/work/aac/sifansong/envs/abdpet`
- XEC OpenCV was changed from GUI `opencv-python 4.6.0.66` to
  `opencv-python-headless 4.6.0.66` because login/compute nodes do not provide
  `libGL.so.1`.

## Full-dataset geometry audit

The audit used all 35,576 2D slices from the fixed WORD training and test
manifests at 1.0×1.0×2.0 mm.

| Center crop | Median black pixels | Foreground-positive slices lost | Foreground pixel retention |
|---:|---:|---:|---:|
| 448 | 74.55% | 0 | approximately 100% |
| 384 | 66.28% | 0 | approximately 100% |
| 336 | 57.49% | 0 | approximately 100% |

The focus-organ maximum 2D boxes at 1 mm were:

| Class | Organ | Maximum box |
|---:|---|---:|
| 4 | gallbladder | 77×75 |
| 5 | esophagus | 44×46 |
| 6 | pancreas | 164×95 |

No focus-organ slice exceeded ROI224. Therefore 336 was selected over 384:
it removes more uninformative border without losing a single organ-positive
slice in this fixed dataset.

Machine-readable audits:

- `docs/WORD_CROP_GEOMETRY_ANALYSIS.json`
- `docs/WORD_TRAIN_CROP_GEOMETRY_ANALYSIS.json`

## Controlled arms

All arms use OrganSlot, `linear_sqrt` fusion, nine slots, ROI sampling
probability 0.2, Loss3 positive weight 0.25, alpha 0.5, maximum class-weight
ratio 4, segmentation head loss 0, effective batch 192, and 118,800 optimizer
updates.

| Arm | Spacing (mm) | Global/input | ROI | Purpose |
|---|---:|---:|---:|---|
| Current control | 1.0×1.0×2.0 | 448 | 384 | Existing running reference |
| A | 1.0×1.0×2.0 | 336 | 224 | Remove black border by runtime crop |
| B | 0.7×0.7×2.0 | 448 | 384 | Keep 448 while increasing in-plane sampling density |

Per user decision, Arm B retains ROI384 exactly like the current experiment.
This makes B a direct spacing ablation, although the physical ROI field becomes
268.8 mm instead of 384 mm.

Loss3 training positive-slice counts for the fixed split remain:

`4392 4875 5070 1631 3827 3753 7327 5556`

The 336 audit confirmed no positive slice is lost, so Arm A uses the same
counts. Arm B counts will be read from the completed 0.7 mm training audit and
must match the configuration before its smoke is submitted.

## Validation status

- 51 OrganSlot unit tests passed after merging the latest architecture.
- Three reconstruction-evaluator regression tests passed.
- 336 model forward/backward passed (21×21 ViT patch grid).
- 336 global + ROI224 paired transform passed.
- 448 global + ROI384 at 0.7 mm proxy geometry passed.
- Python compilation, Bash syntax, and `git diff --check` passed.
- XEC repeated all 51 OrganSlot tests successfully.
- Both launchers leave Slurm-provided `CUDA_VISIBLE_DEVICES` unchanged and
  fail immediately unless the exact allocated GPU count is visible.

## Jobs and acceptance gates

### Arm A: 1 mm / 336 / ROI224

| Stage | Account/cluster | Job | State at submission | Acceptance |
|---|---|---:|---|---|
| Smoke | sifansong/XEC | 115127 | PENDING (Priority) | 2 GPUs visible; 4 updates; finite losses; ROI and positive supervision exercised; checkpoint exists |
| Formal train | sifansong/XEC | 115128 | dependency | Smoke succeeds; 118,800 updates; no NaN/OOM |
| Evaluation | sifansong/XEC | 115129 | dependency | Exact checkpoint load; all 24 cases; all 8 classes and 4 modes; 100% complete |

### Arm B: 0.7 mm / 448 / ROI384

The 0.7×0.7×2.0 mm native-matrix dataset is being generated from the original
120 WORD volumes using the fixed seed-42 case split. Submit the same
smoke→train→evaluation chain only after preprocessing, ROI audit, crop audit,
checksums, and XEC transfer complete.

## Interpretation rule

Compare each completed arm to the existing 1 mm/448/ROI384 OrganSlot run using
the same primary metric (`direct_post`, fixed threshold 0.02), especially the
gallbladder, esophagus, and pancreas Dice. Do not treat training L2/LPIPS values
across different input resolutions as segmentation improvement by themselves.
