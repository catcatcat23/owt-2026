# OrganSlotBank WORD 1x1x2: 448 vs 448 + 20% Small-Organ Crop

## Locked experiment

Offline preprocessing performs only RAS reorientation, joint image/label
resampling to 1x1x2 mm, HU clipping to [-175,250], normalization, and native
matrix slice export. There is no intermediate 448 canvas and no offline
resize. All paths are rooted at `/mnt/DATA-4/anteng`.

WORD is used first. Its 120 cases have resampled XY sizes from 490x490 through
501x501, so every case supports the declared physical crop windows without
padding. BTCV is excluded from this first experiment because 23/30 cases are
smaller than 448 after resampling and require a separate protocol.

### Arm A: 448 control

```text
1x1x2 native matrix -> deterministic center crop 448x448 -> OrganSlot img_size=448
```

The output represents exactly 1 mm per pixel in XY.

### Arm B: 448 + 20% small-organ ROI

Each requested sample is selected as follows:

- 80%: identical global center crop 448x448;
- 20%: choose gallbladder (ID4), esophagus (ID5), or pancreas (ID6)
  uniformly; draw a positive slice from that class pool; crop a 384x384
  physical window around its bounding-box center with up to 10% bounded
  center jitter; resize image/label to 448x448.

The ROI branch enlarges linear size by 448/384=1.167 and area by about 1.36,
not the stronger 1.96 area factor of a 320 crop. The crop is guaranteed to
retain the entire focus-organ bounding box. It keeps all image context and all
other labels within the selected window; it never masks everything except the
focus organ.

The focus OrganSlot is forced to remain in the TGR keep mask for ROI samples.
Other slots follow the selected ordinary mask schedule. TensorBoard records
the realized ROI fraction, per-focus-class fraction, and retained-slot count.

Validation and test always use the deterministic center 448 crop. Labels are
never used to choose evaluation crops.

## Fairness invariants

Both arms share the exact case split, native 1x1x2 files, model variant,
initialization, loss weights, ordinary mask schedule, seed, per-GPU
micro-batch, accumulation, effective batch, learning rate, optimizer updates,
warmup updates, and evaluation. The intentional Arm-B changes are the 20%
positive-slice mixture, 384 ROI crop, and focus-slot retention.

At patch size16, 448 produces 28x28=784 spatial patch tokens versus196 at
224. Before formal runs, sweep per-GPU micro-batch 2,4,8. Select the largest
safe value with memory headroom and use accumulation to target effective
two-GPU batch192 (micro-batch8 implies accumulation12).

## Required gates

1. Current AutoPET jobs and queued inference finish; one architecture is
   selected and frozen for both arms.
2. Full WORD geometry/label validation and immutable manifests complete.
3. ROI index checksum matches the training manifest and contains nonempty
   pools for IDs4,5,6.
4. Image/label overlays are inspected before/after resampling and after both
   crop modes.
5. Unit tests prove paired 2D/Fixfr4 crop, nearest label resize, strict label
   isolation, positive focus selection, and focus-slot retention.
6. Audit per-class realized sampling, crop clipping, and label IDs.
7. Deterministic 448 forward/backward passes.
8. Four-to-eight real samples overfit with augmentation disabled.
9. Micro-batch memory sweep and short two-GPU Arm-B smoke pass without
   NaN/OOM and produce checkpoint, provenance, logs, and TensorBoard events.
10. Launch Arm A on GPUs0,1 and Arm B on GPUs2,3 only after all gates pass.
