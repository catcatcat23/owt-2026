# PSEM-v3 Triplet + LossBalance-v3 Experiment

## Status

- Branch: `experiment/psem-v3-triplet-loss3-word-v0`
- Base: `experiment/psem-v2-lossbalance-v3-word-v0` at `199f6d1`
- Dataset: WORD Common8 2D, 224 preprocessing
- Stage: implementation and CPU verification complete; GPU smoke queued
- Full training must not start until the smoke validator passes.

## Slurm chain

| Job | Stage | Dependency | Current submission state |
|---:|---|---|---|
| 1651604 | 2-GPU smoke | none | pending, Priority |
| 1651606 | 1200-epoch WORD 2D training | afterok:1651604 | pending, Dependency |
| 1651607 | common WORD 2D evaluation | afterok:1651606 | pending, Dependency |

The smoke validator is the final command in job 1651604, so an `afterok`
release requires the checkpoint and all required metrics to pass validation.

## Motivation

PSEM-v2 mixes seven manually weighted query modes in a 20-slot cycle. Loss3
optimizes present-and-kept organs, while Whole and Leave-one-out are trained in
unpaired samples. Low individual reconstruction errors therefore do not ensure
that `Whole - Without(c)` isolates organ `c`, which is required by Indirect
inference.

## Query construction

For each source sample, choose one foreground anchor `c` from only `epoch` and
`sample_index`; GT presence never selects the anchor. Construct three queries:

1. `Direct = {c}`
2. `Context = S`, where `c` is excluded
3. `Plus = S union {c}`

Half of sample/epoch pairs use `S = all classes except c`, exactly matching the
Whole/Without inference pair. The other half use a deterministic non-empty
subset of the complete class bank to retain arbitrary-composition training.
The targets satisfy exactly:

```text
Target(Plus) - Target(Context) = Target(Direct)
```

When `c` is absent, the Direct target and target difference are both zero.
When `c` is present, both equal the CT content inside the organ label.

## Objective

The three branches are concatenated along the batch dimension and passed
through the unchanged PSEM-v2 + Loss3 model. Existing Loss3 terms are averaged
over the resulting `3B` query batch:

```text
L_branch = average Loss3(Direct, Context, Plus)
```

The only new optimized term applies the Loss3 global-L2 plus positive-ROI
structure to the query increment:

```text
Delta = Prediction(Plus) - Prediction(Context)
L_delta = L2(Delta, Target(Direct))
          + 0.25 * PositiveROI(Delta, Target(Direct))
L_total = L_branch + LPIPS + 0.1 * L_delta
```

No extra LPIPS is applied to the difference image. No separate negative or
remove objective is added in v0: the Direct all-zero target and zero target
difference already provide negative-query and negative-addition supervision.

## Fairness and compute

- Architecture, trainable parameters, data, augmentation, LR rule, epochs,
  Loss3 class weights, and evaluator are unchanged.
- Each source image creates three query forwards. The v0 implementation repeats
  the full forward for correctness and isolation; it does not yet reuse the ViT
  token bank.
- Per-GPU source batch is 8 with accumulation 4 on two GPUs. The effective
  source-image batch is 64, equal to the PSEM-v2 + Loss3 baseline (`32 x 2`).
- Query compute is approximately three times larger and must be reported.

## Verification completed

- Deterministic schedule uses no GT labels.
- Anchors cover every foreground class.
- Direct contains exactly one class.
- Context is non-empty and excludes the anchor.
- Plus equals Context plus exactly the anchor.
- Present and absent target identities are exact.
- Difference loss is zero for exact predictions and backpropagates finite
  gradients after perturbation.
- Small 2D model forward/backward passes.
- One-sample triplet intentionally overfits by more than 20% in 20 steps.
- All 14 original PSEM tests and existing PSEM-v2 + Loss3 tests still pass.

## Files

- `util/triplet_query_schedule.py`
- `util/triplet_query_loss.py`
- `engine_pretrain_psem_triplet_loss3.py`
- `main_pretrain_psem_triplet_loss3.py`
- `tests/test_psem_triplet_loss3.py`
- `tools/validate_psem_v3_triplet_smoke.py`
- `slurm/psem_v3_triplet/smoke_word_2d.sbatch`
- `slurm/psem_v3_triplet/pretrain_word_2d.sbatch`
- `slurm/psem_v3_triplet/evaluate_word_2d.sbatch`

## Completion gate

The GPU smoke must produce finite branch, LPIPS, delta-global, delta-positive,
negative Direct-energy, and negative Delta-energy metrics; cover all eight
anchors plus present/absent cases; save `checkpoint-0.pth`; and pass
`tools/validate_psem_v3_triplet_smoke.py`. Only then may the 1200-epoch WORD 2D
training be submitted.
