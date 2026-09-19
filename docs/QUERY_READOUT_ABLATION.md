# Controlled query/readout ablation

## P2 + soft spatial attention prior (2026-09-20)

Independent 2D head: `arm_f_reverse_dot_p2_softmask`. Control:
`arm_f_reverse_dot_p2`. Both P2 arms use microbatch16/accum3 on four GPUs.
The original batch8 job chains 141125/141126 and 141128/141129 are superseded
before starting; preserve their historical runtime snapshots.
Only token-to-pixel cross-attention changes. Before each P16/P8/P4 block,
compute `M = sigmoid(sqrt(C) * normalize(memory) dot normalize(mean(tokens)))`
from the current projected/refined organ tokens and that scale's original
pixel features. Broadcast the same organ prior to all 20 tokens / 4 heads:
`attention = softmax(QK/sqrt(d) + log(0.2 + 0.8*M))`.
The prior is detached, alpha=1, epsilon=0.2; no GT masks, extra trainable
parameters, auxiliary losses, mask threshold, mask annealing or token matching.
The floor bounds the bias to [log(0.2), 0]; it does not guarantee recovery of
missed organs. P16 uses input-projected tokens; later scales use refined tokens.
This is same-scale self-guidance, not an exact Mask2Former/H-SAM reproduction
or a previous-layer supervised mask decoder. Coarse dot priors are not separately
supervised/calibrated and their usefulness remains an experimental hypothesis.

P4 reverse attention, S2/P2 fusion, dot readout, reconstruction and loss remain
identical. State-dict keys and initial values match P2 exactly. Checkpoint args
must retain the new head name: identical tensor keys alone cannot identify routing.
The existing FP32 attention region also computes this bias. 3D is rejected.

Use existing training entry point with
`ARM_F_HEAD_TYPE=arm_f_reverse_dot_p2_softmask`, `ARM_F_DIMENSION=2D`,
`ARM_F_MICRO_BATCH=16`, `ARM_F_ACCUM_ITER=3`, `ARM_F_LAMBDA_SEG=0.01`,
`BACKGROUND_REDUCTION=topk`, scratch seed0, no MAE init, and the same WORD07072
ROI20 dataset / 118800 updates / effective batch192 as P2. Evaluation must use
`EXPECTED_SLOT_HEAD_TYPE=arm_f_reverse_dot_p2_softmask` and the same unified
train-calibrated head + reconstruction protocol. Batch16 GPU memory is unverified;
do not silently reduce microbatch if it fails. Seven CPU tests passed on XEC.

Tests cover alpha=0 exact baseline equivalence, strict P2 weight loading,
identical initialization, finite non-hard biases, preferential synthetic routing,
BF16-input finite backward, full-model gradients and checkpoint roundtrip,
and unchanged legacy heads. GPU validation and scientific benefit remain pending.
Assess small-organ recall, empty-slice FP and predicted/GT volume as well as Dice;
more concentrated attention alone is not evidence of better organ localization.

## P2 shallow-skip readout (2026-09-20)

New 2D-only head `arm_f_reverse_dot_p2`. Keep P16/P8/P4 token refinement,
P4 reverse attention, 20-token mean query and normalized dot scale unchanged.
Return existing stem S2 (64 channels at standard width128); bilinearly upsample
organ-conditioned refined P4, concatenate with 1x1-projected S2, then apply
two 3x3 Conv/GroupNorm(1)/GELU layers at width128. Dot readout now occurs on P2
(224x224 for input448), followed by the original full-size bilinear resize.
Fusion is shared across organs, evaluated per organ. No mask attention, no
extra supervision, no new temporal mixing. Existing head defaults/weights stay
unchanged; P2 is explicitly rejected for 3D. Added modules preserve constructor
RNG for later slots. P2 increases memory/compute; GPU throughput is unverified.

Controlled training: scratch seed0, WORD07072 ROI20, lambda_seg0.01, topk
small-organ loss, LR7.5e-5, 118800 updates/warmup5940, four A800,
microbatch16/accum3/effective192. Control is original Reverse-Dot2955011, NOT
the lambda0.03 experiment. Different SIP/XEC software/hardware is a caveat.
Target antengcai23/XEC, account sifansong, QoS8gpus (maximum five days).
Use the existing unified2D calibration/head/recon evaluator with
EXPECTED_SLOT_HEAD_TYPE=arm_f_reverse_dot_p2 and EXPECTED_LAMBDA_SEG=0.01.
This tests the P2 fusion/readout package, not resolution alone.

Historical batch8 submission, superseded before starting: antengcai23/XEC train141125
(Priority), eval141126 (afterok:141125 verified), both QoS8gpus/5days.
Runtime /gpfs/work/aac/antengcai23/worktrees/reverse_dot_p2_3150ff3,
commit3150ff306ae47e57aa081b538c40de602a0c630a. Six CPU tests passed
locally and on XEC; dataset identity and revision gates passed.
GPU memory/throughput and training stability remain unverified until start.

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

Submitted 2026-09-20 00:04 CST on bolinren19/SIP, Slurm account
angelosstefanidis, QoS8a800: train2964744 (4A800/20CPU/192GB/7days),
eval2964745 (1A800/10CPU/128GB/7days), afterok:2964744 verified.
Initial states: Priority / Dependency. Pinned source
188555c2c18de95ebb76a21bb23ea3c50cba7db3 at
/gpfs/work/aac/bolinren19/OD_OWT/.worktrees/arm_f_reverse_dot_seg003_188555c.
CPU configuration/metadata checks passed; GPU execution remains pending.

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
