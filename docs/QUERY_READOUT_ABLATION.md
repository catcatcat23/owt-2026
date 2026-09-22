# Controlled query/readout ablation

## P4 Reverse-Dot batch16 correction (2026-09-20)

Replace baseline2955011/2955012 and lambda_seg0.03 chain2964744/2964745.
Both use arm_f_reverse_dot, four GPUs, microbatch16/accum3/effective192,
scratch seed0, WORD07072 ROI20, topk small-organ loss, LR7.5e-5,
118800 updates/warmup5940. Only lambda_seg differs: 0.01 vs 0.03.
Baseline had only just started (about7min), with no checkpoint at inspection;
both replacements start from epoch0, no resume or MAE. Retain old artifacts.
Keep bolinren19/SIP associations: baseline sifansong/8a800;
lambda0.03 angelosstefanidis/8a800. Train4A800/20CPU/192GB, seven days;
unified head calibration/test/recon eval1A800/10CPU/128GB, seven days,
afterok new training. Eval EXPECTED_LAMBDA_SEG must match each arm.
Use latest origin/feature/orgslot in a new pinned worktree, not either old runtime.
Microbatch16 GPU memory/throughput remain unverified. Do not call it a GPU-tested
configuration based on CPU compatibility tests from earlier work.

Submitted 2026-09-20 01:43 CST; both trainings Pending(Priority):

| lambda_seg | Train | Unified eval | Slurm account / QoS |
| --- | --- | --- | --- |
| 0.01 | 2965316 | 2965317, afterok:2965316 | sifansong / 8a800 |
| 0.03 | 2965318 | 2965319, afterok:2965318 | angelosstefanidis / 8a800 |

Runtime /gpfs/work/aac/bolinren19/OD_OWT/.worktrees/p4_b16_db9c788,
source db9c788152b87d755ef67dc46eb59c7d70526c5f. Revision gate and shell
configuration assertions (micro16/accum3/effective192/lambda per arm) passed.
Training/evaluation paths and afterok dependencies verified after submission.
Logs: artifacts/p4_b16_db9c788/logs. Old four jobs confirmed cancelled;
2955011 used8min43sec before cancellation, no saved progress used in replacement.
No GPU preflight run and no claim of batch16 GPU validation.

## SIP E / Query-Dot batch16 resume (2026-09-20)

User requested changing running 2D E2955007 and Query-Dot2955009 from
microbatch8/accum6 to microbatch16/accum3, retaining four GPUs/effective192.
Resume sources: arm_f_linear_2d/Results/OrganSlotBank/Common8/WORD_2D/
ArmF_2D_2955007/checkpoint-120.pth and ArmF_2D_2955009/checkpoint-10.pth.
E saved checkpoint120 during checks; validated and selected the newer file.
Restore model/optimizer/scaler, start at epoch121 / epoch11 respectively.
Both old/new microbatch settings yield148 optimizer updates per epoch;
retain LR7.5e-5, total118800 updates, WORD07072 ROI20, topk small-organ loss,
lambda_seg0.01, scratch lineage, no MAE. Resumed runs write separate job outputs.
Use latest origin/feature/orgslot in a new immutable worktree. Cancel old
training/evaluation chains only after checkpoint validation; rebuild afterok
evaluations against the new job IDs. Preserve all old results/checkpoints.
This is mixed-microbatch training, not a clean batch16-from-scratch experiment:
group-reduced losses can change gradients despite equal effective batch.
GPU peak memory/throughput at batch16 are not yet verified.

Submitted 2026-09-20 01:34 CST on bolinren19/SIP (Slurm account sifansong):
E train2965273 / eval2965274 (4a800); Query-Dot train2965275 / eval2965276
(8a800). Train4A800/20CPU/192GB; eval1A800/10CPU/128GB; seven days each.
Both training jobs initially Priority; both afterok evaluation dependencies
verified. Cancelled old2955007/2955008/2955009/2955010; old artifacts retained.
Runtime /gpfs/work/aac/bolinren19/OD_OWT/.worktrees/sip_2d_b16_d7dedbe,
source d7dedbe5a03d32224103fadf68f808d7f6266614; revision gate passed.
CPU strict model/optimizer loading and tensor finiteness passed on real
checkpoints, optimizer steps17908 / 1628. Scaler key present; training restores
it through existing resume logic (BF16 scaler disabled). GPU resume not yet tested.

## Active P2 submissions — batch16 correction, 2026-09-20 01:16 CST

| Experiment | Train | Unified eval | Status at submission |
| --- | --- | --- | --- |
| P2 baseline | 141130 | 141131, afterok:141130 | Priority / Dependency |
| P2 + soft mask | 141132 | 141133, afterok:141132 | Priority / Dependency |

Both: antengcai23/XEC, Slurm account sifansong, QoS8gpus, partition gpua8001t,
4A800/20CPU/192GB training; 1A800/10CPU/128GB evaluation; five-day limit.
Microbatch16 x 4 GPUs x accumulation3 = effective192. Scratch seed0,
lambda_seg0.01, topk background, WORD07072 ROI20, 118800 optimizer updates.
Pinned source efcef0739b581ec49c969c312d07c97574308cbb;
runtime /gpfs/work/aac/antengcai23/worktrees/p2_b16_efcef07.
Scripts: slurm/orgslot/train/arm_f.sbatch and
slurm/orgslot/eval/arm_e_mae_unified.sbatch. Both job paths/dependencies verified.
Seven CPU tests and dataset identity passed on XEC; revision gate passed.
No GPU preflight performed. Peak memory for microbatch16 remains unknown.
Cancelled the four still-pending batch8 jobs 141125/141126/141128/141129;
no training progress lost, historical source snapshots retained.

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
# Multiscale F + SAM-style tail (2026-09-22)

New opt-in head: `arm_f_sam_tail`. Implementation only; no GPU job submitted.
Existing E/F heads, losses, reconstruction and frozen job worktrees are unchanged.

Path: existing 20 organ tokens -> existing P16/P8/P4 TokenBlocks ->
prepend ONE shared learned mask token -> token self-attention ->
token-to-P4 attention -> token FFN -> P4-to-token attention ->
final token-to-updated-P4 attention -> mask token MLP ->
ordinary dynamic dot product with updated P4 -> existing output interpolation.

One terminal two-way block, four attention heads, usual width128. It retains
the F axial spatial PE (temporal + spatial for 3D); the input 21-token sequence
is a fixed sparse content reference within the tail, not a geometric prompt.
Post-residual LayerNorm follows each tail sublayer. The MLP is three Linear
layers with GELU between them. The readout has no cosine normalization or
sqrt(C) multiplier. There is no pixel FFN in the terminal reverse step.
Mask token and tail weights are shared across organs; organ conditioning comes
from existing slot-specific tokens/identity. No GT prompt, P2, soft mask,
IoU head, multimask output, new loss, or SAM/MAE weights are introduced.

This is a SAM-inspired tail, NOT original SAM: retained multiscale front end,
one rather than two blocks, existing axial PE and PyTorch MHA projections,
GELU, no SAM upscaler. Reference:
https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/transformer.py
and the adjacent mask_decoder.py.

CLI: `--slot_head_type arm_f_sam_tail --pixel_pe none --query_refinement none`.
Existing launcher accepts `ARM_F_HEAD_TYPE=arm_f_sam_tail`; for historical
small-organ comparisons explicitly set `BACKGROUND_REDUCTION=topk` (launcher
default is mean), `ARM_F_MICRO_BATCH=16 ARM_F_ACCUM_ITER=3` in 2D.
Set evaluation `EXPECTED_SLOT_HEAD_TYPE=arm_f_sam_tail`. New head requires its
own trained checkpoint; old-head checkpoints are not exact new-head resumes.
No submission until commit/push and immutable-runtime revision verification.

CPU tests cover 2D/3D shapes, operation order, BF16-input finite backward,
nonzero gradients through all tail parameters and organ inputs, token
conditioning, model/reconstruction integration, strict checkpoint roundtrip,
and exact pre-change outputs/state dictionaries of all six older F heads.
GPU/DDP, production-memory and accuracy remain unvalidated. 3D support here
uses the existing slice-wise spatial branch and flattened slab attention;
it is not evidence that this resolves previous 3D segmentation issues.
