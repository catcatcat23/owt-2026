# WORD448 OrganSlot retained multiscale Focal v2

## Objective

Test whether a stronger local segmentation decoder can improve small-organ
readout without breaking OrganSlot retained-query semantics.

## Locked configuration

- dataset: WORD 2D native-resample, 448 input;
- augmentation: ROI20 (384 crop resized to 448), focus classes 4/5/6;
- reconstruction: global L2 + LPIPS;
- segmentation: independent binary sigmoid focal loss;
- focal alpha/gamma: 0.75/2.0;
- segmentation weight: 0.1;
- background segmentation weight: 0;
- supervision: retained slots only;
- head: four-stage multiscale depthwise-separable convolution decoder;
- fusion: linear_sqrt;
- no fused-image Loss3.

## Execution semantics

The shared ViT encodes the complete image batch once. For each slot, only rows
whose slot is retained are gathered into Collector, slot TGEnc, AHER and the
binary head. Results are scattered back to batch layout for fusion and loss.
Dropped rows therefore consume no slot-path/head compute and receive no
segmentation gradient. Legacy all-slot supervision remains available explicitly
and computes the union of reconstruction-retained and head-supervised rows.

## Gates

1. Unit tests verify 2D shape, finite backward, sparse Collector/head batch
   sizes, zero dropped logits, and no-head reconstruction mode.
2. Real ROI100 smoke uses the formal 8 images/GPU microbatch for four optimizer
   updates and must produce finite reconstruction, LPIPS and focal metrics.
3. Formal ROI20 training starts only after the smoke succeeds.
4. Formal evaluation reports fixed-threshold reconstruction Direct/Indirect,
   fixed 0.5 binary-head Dice, and validation-only calibrated head thresholds.

## Job registry

Submitted from login account antengcai23 on XEC using Slurm project account
sifansong:

| Stage | Job ID | Dependency at submission |
|---|---:|---|
| real ROI100 smoke | 119640 | Priority |
| formal ROI20 training | 119641 | afterok:119640 |
| reconstruction Direct/Indirect | 119642 | afterok:119641 |
| head train-calibration | 119643 | afterok:119641 |
| head test | 119644 | afterok:119643 |

Submission audit:

- An initial account-name attempt was rejected before creating jobs.
- Jobs 119609-119613 were cancelled. The smoke file had accidentally been
  overwritten by the formal file because both had the same basename in a
  temporary patch-generation directory; 119609 ran 8 minutes of the formal
  configuration before cancellation and is not an experiment result.
- Commit ad98843 restores a distinct four-update ROI100 smoke and adds shell
  assertions for MAX_UPDATES=4, ACCUM_ITER=1, ROI probability 1.0 and a unique
  smoke tag before launching Python.


## Smoke acceptance and formal start

Corrected smoke 119640 completed on xgpua800n6 in 28 seconds with exit code 0.
It used two A800 GPUs, microbatch 8/GPU, accumulation 1, ROI probability 1.0
and exactly four optimizer updates. The real-data validator passed and saved
checkpoint-0.pth. Averaged finite metrics were:

- total loss: 1.0856;
- reconstruction L2: 0.2633;
- LPIPS: 0.8160;
- focal segmentation: 0.0632;
- weighted focal term: 0.0063;
- peak GPU memory: about 16.3 GiB on the first logged step.

All three focus classes were exercised and FP32 probability monitoring remained
finite. The afterok dependency released formal job 119641, which started on
xgpua800n6. Its first logged batches are finite, use ROI probability 0.2,
effective batch 192 and peak memory about 18.3 GiB. Jobs 119642-119644 remain
blocked behind the formal training/evaluation dependencies.
