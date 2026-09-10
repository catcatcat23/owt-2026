# Arm F: organ-conditioned two-stage attention

Branch: `experiment/orgslot-arm-f`, based on `247ded90e5d49603f68ace2153b9a6fad6ba2d8d`.
No training results yet. No Slurm jobs submitted by this implementation.

## Model configuration

Use the existing training entry `main_pretrain_orgslot_common8_a100.py`:

```
--slot_head_type arm_f_attention --slot_head_channels 128
--token_factor 20 --query_refinement none --pixel_pe none
--seg_loss_type small_organ --lambda_seg 0.01
--tversky_alpha_fp 0.3 --tversky_beta_fn 0.7
--balanced_focal_weight 0.5 --hard_negative_ratio 0.02
--negative_slice_weight 0.1 --focal_alpha 0.75 --focal_gamma 2
--organ_roi_aug --organ_roi_probability 0.2
--input_size 448 --global_crop_size 448 --roi_crop_size 384
--expected_spacing 0.7 0.7 2 --amp_dtype bf16 --clip_grad 1
```

2D: `--dimension 2D --segmentation_unit slice`.
3D: `--dimension 3D --fix_frame 4 --temp_stride 1 --segmentation_unit slice`.
Linear readout control: replace head type with `arm_f_linear`.
Arm F always uses fixed axial sin/cos attention PE: y/x in 2D, t/y/x in 3D.
`pixel_pe none` refers to the older decoder's optional PE, not disabling Arm F PE.
Arm F uses its own three refinement blocks; `query_refinement cross_attn` remains
an Arm E-only option and must not be used here.

Dataset CSV, ROI index, preprocessing summary, LPIPS weights, output paths and
initialization must be selected for the actual account/cluster. Do not copy
another account's paths. Match effective batch, LR, update budget and initial
encoder/reconstruction weights to the chosen comparison arm; no claim of fair
comparison is made merely from equal epochs across 2D/3D.

## Shape and implementation

Collector/TGEnc retains image-conditioned `[B,20,D]` tokens. Input projection
and the existing per-slot identity produce `[B,20,128]`. Three separately
parameterized blocks read P16, fused P8, fused P4. Each block is pre-norm cross
attention, organ-local self-attention, FFN. All decoder weights are shared across
organs. No Hungarian matching, instance loss or mask gating is introduced.

For 448 input: maps are `[B,128,28,28]`, `[B,128,56,56]`, `[B,128,112,112]`.
For 3D: maps are `[B,128,4,28,28]`, `[B,128,4,56,56]`, `[B,128,4,112,112]`.
The 3D stem applies the original Arm E 2D convolutions slice-wise. Cross-attention
can access all slab positions with fixed t/y/x PE, but there is no added temporal
Conv3D mixing. This new stem is a separate architectural change versus 3D Arm D;
use a matched Arm F linear readout to isolate reverse attention effects.
Temporal encoder stride >1 averages input slices to the encoder time grid before
the stem and interpolates final logits back; the intended baseline uses stride 1.

Stage B: P4 queries read updated tokens; output retains a P4 residual, then FFN,
shared LayerNorm/linear classifier, interpolation and original per-slot calibration.
Every organ has one output channel; raw/calibrated dictionaries and active-row
selection are unchanged. Reconstruction/AHER/fusion are retained.

Linear control: mask MLP -> 20 dot-product scores -> learned linear token fusion.
Only the fused mask is supervised. This is equivalent to a fused dynamic mask
embedding, not pixel-adaptive token selection. Token indices/count must be stable.

Existing 2D/3D head evaluators rebuild from saved `slot_head_type` and channels;
Arm F checkpoints require this branch's code and strict loading. No partial
loading of an Arm E checkpoint as a full Arm F resume is supported.

## Numerical and resource behavior

Standard `nn.MultiheadAttention(batch_first=True, need_weights=False)`, four heads,
FFN ratio 4, dropout 0. Attention/FFN/readout run locally in FP32 for compatibility
with the cluster's older PyTorch; spatial convolutions retain ambient AMP.
Interpolation uses the existing FP32-safe helper. Never call `model.bfloat16()`;
keep FP32 parameters and use autocast. No full attention heatmaps returned in the
training path. Attention weights are not foreground probabilities.

P4 features and FFN activations scale per active organ; loops do not free saved
autograd activations before backward. 3D has four times as many spatial positions
as 2D. Full-size GPU memory, speed and DDP preflight remain unverified; choose
microbatch from measurement, retaining effective batch via accumulation.

## Validation and next experiments

CPU tests cover 2D/3D attention and linear heads, finite backward, all new decoder
parameters receiving gradients, Collector gradients, active rows, unchanged
reconstruction shape, strict checkpoint roundtrip, 20-token sensitivity, temporal
PE identity and BF16 inputs to attention. CPU BF16 whole-stem execution is not
claimed: the installed CPU GroupNorm rejects mixed BF16/FP32 inputs.

Validated on 2026-09-10: 33 unittest cases (`test_arm_f`, `test_orgslot_model`,
`test_query_cross_attention`) passed, plus all 7 standalone slice-wise loss and
3D evaluator regression functions. `git diff --check` passed. No GPU/DDP run.

Compare Arm E, Arm F linear, Arm F attention under the same protocol. Additional
single-scale and depth-matched ablations are not implemented yet. Monitor eight
organ Dice, precision/recall, empty-slice FP, volume ratio, memory and throughput.
No SOTA or efficacy claim before full evaluation.

Future formal jobs use maximum allowed walltime, verified on submission: SIP
4a800/8a800 7 days; XEC 8gpus 5 days, 4gpus 7 days. Existing jobs unchanged.
Formal submission scripts: `slurm/orgslot/train/arm_f.sbatch` and
`slurm/orgslot/eval/arm_f.sbatch`, with explicit `ARM_F_TARGET`,
`ARM_F_DIMENSION`, `EXPERIMENT_WORKDIR`, `ARM_F_COMMIT` exports.
2D: microbatch 2 x 4 GPUs x accumulation 24 = 192 slices.
3D: microbatch 1 x 4 GPUs x accumulation 12 = 48 slabs (192 slices).
Both use scratch initialization, actual LR 7.5e-5, 118800 updates. Smaller
microbatches may change legacy_batch TGR behavior; do not claim exact equivalence
from matching effective batch alone. The first evaluation is fixed threshold 0.5.
The branch is fast-forwarded to origin/feature/orgslot before runtime snapshots
are created and checked; existing worktrees are not updated.
