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
- Segmentation head supervision is disabled: `lambda_seg=0.0`.
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

Current retry implementation commit: `2231cd0`. The retry uses
`sifansong/4a800` and two typed A800 GPUs.

| Job | Stage | Status at submission |
|---:|---|---|
| 1651723 | Initial ROI100 Loss-v3 smoke | FAILED before Python: no devices |
| 1651724 | Initial ROI20 Loss-v3 formal | CANCELLED: DependencyNeverSatisfied |
| 1655555 | Corrected seg=0 ROI100 smoke + validator | PENDING (Priority), estimated 2026-08-11 09:33 |
| 1655556 | Corrected seg=0 ROI20 formal training | PENDING, `afterok:1655555` |

The full job uses an `afterok` dependency on a two-GPU smoke whose validator
requires visible CUDA devices, finite global/LPIPS/positive ROI metrics,
exercised ROI samples, and a checkpoint. The retry excludes `gpua800n2` and
`gpua800n6`.

## GPU visibility incident and retry

Jobs 1651723/1651724 are invalid: smoke 1651723 failed before Python with
`No devices were found` on `gpua800n2`; formal 1651724 became
`DependencyNeverSatisfied` and was cancelled. The initial scripts also used
`lambda_seg=1.0`, which did not isolate the planned pure ROI20 + Loss-v3
combination. Retry scripts use `lambda_seg=0.0`, typed A800 GRES, exclude the
two nodes implicated in CUDA invisibility, and require both PyTorch CUDA
discovery and a real CUDA tensor operation before training.

## 2026-08-11 second retry

Smoke `1655555` also started without an allocated GPU, this time on
`gpua800n1`, and failed at the CUDA preflight before any model forward.
Dependent formal job `1655556` was cancelled.

The replacement chain is:

| Stage | Job | Dependency | Initial state |
|---|---:|---|---|
| ROI100 smoke + validator | 1658405 | none | PENDING (Priority) |
| ROI20 formal training | 1658406 | afterok:1658405 | PENDING (Dependency) |

The `sbatch` command line explicitly requests `gpu:a800:2` and excludes
`gpua800n1`, `gpua800n2`, and `gpua800n6`. Post-submission inspection confirmed
`ReqTRES=...gres/gpu=2`, `TresPerNode=gres:gpu:a800:2`, and
`ExcNodeList=gpua800n[1-2,6]` for both jobs.
