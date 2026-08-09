# OrganSlot WORD 448 ROI20 + LossBalance-v3

## Objective

Test whether OrganSlotBank benefits from combining the 448 native-resolution
protocol and 20% small-organ ROI sampling with the positive-state-only
LossBalance-v3 objective. This is a new combination experiment, not a
loss-matched replacement for the existing OrganSlot 448 control.

## Locked configuration

- Dataset: WORD Common8 2D, 96 training and 24 test cases.
- Preprocessing: RAS, 1x1x2 mm, native resampled matrix, fixed-255 intensity.
- Input: global 448 center crop; 20% ROI branch uses 384 crop resized to 448.
- ROI focus classes: gallbladder 4, esophagus 5, pancreas 6.
- Architecture: OrganSlotBank, 20 tokens/class, slot TG depth 1.
- Fusion: `linear_sqrt`.
- Mask: `legacy_batch`; an ROI focus slot is always retained.
- Segmentation loss: `lambda_seg=1.0`, background weight 0.25.
- Reconstruction: global L2 + LPIPS.
- Loss-v3: present-and-kept foreground ROI-L2 only, weight 0.25.
- Removed foreground ROI-L2 is monitoring only.
- Frequency parameters: alpha 0.5, maximum weight ratio 4.0.
- Two A800 GPUs, micro-batch 8/GPU, accumulation 12, effective batch 192.
- Optimizer update budget: 118800; warmup updates: 5940.

Natural 448-center-crop positive-slice counts for raw IDs 1..8:

```text
4392 4875 5070 1631 3827 3753 7327 5556
```

These counts are frozen before ROI mixing so ROI20 remains an explicit data
intervention rather than silently redefining Loss-v3 weights.

## Isolation and implementation

Loss-v3 consumes only `visible_masks` returned by `StrictVisibilityDataset`;
the raw multi-class label is never exposed at the training boundary. Positive
and removed states use the exact OrganSlot `[B,9]` keep mask after focus-slot
retention. No trainable model parameter or checkpoint key is added. When the
Loss-v3 coefficient is zero, its per-mask computation is skipped.

## Validation

- Four manifests: 70,792 records checked, zero missing referenced files.
- 29 focused regression tests passed.
- Real WORD gallbladder ROI 448 forward/backward passed with finite backbone
  gradients and nonzero positive supervision.
- Frozen LPIPS state extracted from the completed WORD OWT checkpoint; SHA256
  `a99ef0ef436727d809148f5e9cbc09c1f0218079020918b0b1cc5167493c4287`.

## Jobs

Implementation commit: `a119677`. Both jobs use account
`angelosstefanidis`, QoS `8a800`, and two A800 GPUs.

| Job | Stage | Initial status |
|---:|---|---|
| 1651723 | ROI100 Loss-v3 smoke + validator | PENDING (Priority), estimated 2026-08-10 06:03 |
| 1651724 | ROI20 Loss-v3 formal training | PENDING, `afterok:1651723` |

The full job uses an `afterok` dependency on a two-GPU smoke whose validator
requires finite global, LPIPS, segmentation and positive ROI metrics, exercised
ROI samples, and a checkpoint.
