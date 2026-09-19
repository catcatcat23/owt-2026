# Controlled query/readout ablation

Existing E, `arm_f_attention`, and `arm_f_linear` implementations are retained.
New heads are `arm_f_query_dot` and `arm_f_reverse_dot`.

| Head | Token refinement | Reverse pixel update | Readout |
| --- | --- | --- | --- |
| arm_e_multiscale_query | Existing mean-pool query | None | Existing normalized dot |
| arm_f_query_dot | Existing F P16/P8/P4 blocks | None | Mean refined tokens, normalized dot with original P4 |
| arm_f_reverse_dot | Same F blocks | Existing reverse attention + residual + FFN | Same query, normalized dot with updated P4 |
| arm_f_attention | Unchanged | Unchanged | Existing LayerNorm + linear classifier |

New dot heads use `sqrt(channels) * cosine(pixel, mean(refined_tokens))`.
No learned pooling or added query projection. Slot identity is already injected
before refinement and is not added again. E's own query construction is unchanged.
The two new paths support 2D and 3D; the controlled E/F experiment is 2D because
the existing E baseline supports only 2D.

Planned common protocol: WORD native07072, 448/global448/ROI384, ROI probability
0.2, scratch, seed0, 20 tokens, 128 channels, small-organ loss, lambda_seg0.01,
LR7.5e-5, 118800 updates, warmup5940, 4 GPUs, microbatch8/accum6/effective192.
Use identical background reduction and optimizer settings across runs. No resume
or MAE initialization. Retain token self-attention, FFN and F attention PE.

Use the existing unified 2D evaluator with EXPECTED_SLOT_HEAD_TYPE set explicitly:
training first6000 calibration, test fixed0.5 and calibrated, raw/post, plus recon.
Do not compare calibrated results against an uncalibrated historical baseline.

Same seed alone does not ensure identical common-module initialization across
architectures (constructor RNG consumption differs). Before claiming an exact
initial-weight-matched ablation, audit or explicitly synchronize shared initial
weights in a separate experiment initialization step. Legacy initialization is
not changed here. E vs F also changes query aggregation/projection and PE, so it
tests the refinement design as a package, not cross-attention alone.

CPU tests cover new paths in 2D/3D, finite gradients and strict round-trip loading;
legacy decoder tests compare keys, initialization and outputs against fd9cf91.
No GPU training is implied by these tests.

## 2D Reverse-Dot segmentation-weight ablation (2026-09-20)

Add one independent scratch run against Reverse-Dot train2955011/eval2955012.
Only lambda_seg changes from 0.01 to 0.03. Set ARM_F_LAMBDA_SEG=0.03 for
training and EXPECTED_LAMBDA_SEG=0.03 for the unified evaluator; both defaults
remain 0.01 for existing experiments. The reconstruction/LPIPS weights, head,
seed0, WORD07072 split, ROI20, topk small-organ loss, LR7.5e-5, 118800 updates,
warmup5940, four GPUs and microbatch8/accum6/effective192 remain unchanged.
No MAE initialization or resume. Keep the original run and evaluation chain.
Evaluate fixed0.5 and training-calibrated head masks plus reconstruction with
the existing protocol. Increased loss weight alone is not evidence of improved
Dice; compare segmentation gains and possible reconstruction degradation.

## 3D controlled extension

The new `arm_e_multiscale_query_3d` baseline reuses F's slice-wise E spatial
stem/fusion without token refinement or reverse attention. Its query is exactly
E-style: token LayerNorm, mean, projection, slot identity; normalized dot with
P4 followed by trilinear upsampling. It does not modify the existing 2D E head.
The other two arms use `arm_f_query_dot` and `arm_f_reverse_dot` with dimension=3D.
F attention blocks operate across the slab with existing axial temporal/spatial PE;
the baseline adds no new PE or temporal convolution.

Use ARM_F_DIMENSION=3D, ARM_F_HEAD_TYPE explicitly selected, BACKGROUND_REDUCTION=topk,
scratch/seed0, microbatch6, four GPUs, accumulation2 (48 slabs), 118800 updates,
WORD07072 and ROI20. All three use the unified 3D calibration/head/reconstruction
evaluator, not the older post3d-only evaluator. No training jobs were submitted
as part of implementing this extension.
