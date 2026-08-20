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

Job IDs are added after remote submission.
