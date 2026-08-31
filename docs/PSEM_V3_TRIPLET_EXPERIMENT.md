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

## Second retry (2026-08-11)

The first recovery smoke jobs `1655585` (WORD) and `1655588` (AbdAutoPET)
started on `gpua800n1` without a GPU allocation and failed at the strict CUDA
preflight. Their dependent chains became unusable and were cancelled. No model
forward or loss computation occurred, so these failures are infrastructure
incidents rather than method results.

The replacement submission explicitly passes typed A800 GRES on the `sbatch`
command line and excludes `gpua800n1`, `gpua800n2`, and `gpua800n6`:

| Dataset | Stage | Job | Dependency | Initial state |
|---|---|---:|---|---|
| WORD | smoke | 1658407 | none | PENDING (Priority) |
| WORD | train | 1658408 | afterok:1658407 | PENDING (Dependency) |
| WORD | evaluation | 1658409 | afterok:1658408 | PENDING (Dependency) |
| AbdAutoPET | smoke | 1658410 | none | PENDING (Priority) |
| AbdAutoPET | train | 1658411 | afterok:1658410 | PENDING (Dependency) |
| AbdAutoPET | evaluation | 1658412 | afterok:1658411 | PENDING (Dependency) |

Immediately after submission, every job showed the expected
`ReqTRES=...gres/gpu=N`, typed `TresPerNode=gres:gpu:a800:N`, and
`ExcNodeList=gpua800n[1-2,6]`. Future audits should use these six job IDs in
place of the superseded `1655585`--`1655590` chain.

## AutoPET NaN incident and numeric-stability repair (2026-08-20)

The resumed AutoPET run failed in job 116874 at epoch 500, step 1296. Rank 2
reported NaN for every loss component, while the other ranks were still finite
before the synchronized failure. The offending raw images and labels were
finite, the Direct/Context/Plus masks were valid, and checkpoints 400, 425, 450
and 475 contained finite model and optimizer tensors.

The GradScaler scale fell repeatedly (1024, 512, 1024, then 256 at those saved
checkpoints), which is evidence of recurring backward overflow before the final
forward became NaN. This does not prove one exact layer is solely responsible.
The concrete implementation defects were the unsafe combination of whole-model
FP16 autocast, no gradient clipping, no deterministic seeding, and no checks
between input loading and final scalar loss.

The repair on branch fix/psem-v3-numeric-stability is:

- A100 forward autocast uses BF16, whose exponent range is much larger than
  FP16; LPIPS remains explicitly FP32.
- GradScaler is disabled for BF16 and old scaler state can still be loaded.
- Global gradient norm is clipped to 1.0 with non-finite gradients rejected
  before optimizer.step.
- image, label, reconstruction targets, loss components and periodically all
  model parameters are checked for NaN/Inf with rank/epoch/step/sample context.
- Python, NumPy, Torch, CUDA, DistributedSampler, DataLoader and transform RNGs
  receive deterministic seeds for the diagnostic smoke.
- The diagnostic resumes from checkpoint 475 rather than restarting from
  checkpoint 400, so it tests close to the observed failure boundary.

The local test gate contains 23 PSEM tests, including nine dedicated numeric
tests, and all pass. The same nine numeric tests pass in the sifansong XEC
environment. Diagnostic smoke job 119639 failed during shell preflight before
Python started because its checkpoint path incorrectly pointed to the original
run directory. Checkpoint 475 actually belongs to the
`RESUME_CKPT400_FP32FIX` run. Both the smoke and formal-resume scripts now
use that directory and print an explicit error if the checkpoint is missing.
Formal training must not start until the corrected BF16 checkpoint-475 smoke
passes.

## Final formal evaluations (2026-08-31)

### AbdAutoPET 2D

The numerically repaired training completed on `sifansong/XEC` as Job
`121484`. Formal evaluation Job `124936` strictly loaded checkpoint 1199 and
processed all 200 cases (22,400 slices). The scientific configuration remained
PSEM-v3 Triplet+Loss3 with four foreground classes, 20 tokens per class,
per-sample normalization, effective source batch 192, positive-ROI weight 0.25,
and Delta weight 0.1. The execution policy used BF16, FP32 LPIPS, gradient
clipping, and component-level finite checks.

| Readout | label_1 | label_2 | label_3 | label_4 | Mean post Dice | Mean post NSD | Mean post HD95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct | 96.12% | 91.64% | 91.70% | 75.67% | 88.783% | 83.172% | 7.055 mm |
| Indirect | 96.36% | 92.56% | 92.69% | 78.15% | **89.938%** | **87.639%** | 8.074 mm |

The dataset repository does not provide a verified organ-name mapping for labels 1--4, so the table intentionally retains label IDs. Under the identical 200-case protocol, PSEM-v1 Indirect remains best at
90.512%. PSEM-v3 Indirect is statistically indistinguishable from PSEM-v2a
Indirect: the paired case-bootstrap difference is +0.049 percentage points
with 95% CI [-0.107, 0.224]. It is 0.574 points below PSEM-v1 Indirect with
95% CI [-0.718, -0.410]. PSEM-v3 Direct is 1.326 points below PSEM-v1 Direct
with 95% CI [-1.597, -0.988]. Therefore the AutoPET result shows that Indirect is preserved better than Direct, but it does not establish improved decomposition over v1 because both readouts are lower; it is not a new absolute SOTA.


### AbdAutoPET reconstruction evidence

| Variant | Whole L2 | Whole LPIPS | Whole PSNR | Whole SSIM | label_4-only L2 |
|---|---:|---:|---:|---:|---:|
| PSEM-v1 | **0.00015039** | **0.01907** | **38.624** | 0.91971 | 0.00019494 |
| PSEM-v3 Triplet+Loss3 | 0.00017368 | 0.02313 | 37.979 | 0.90798 | 0.00025977 |
| PSEM-v2a Query20 | 0.00017749 | 0.02356 | 37.908 | 0.92126 | **0.00019280** |
| PSEM+LossBalance-v2 | 0.00026088 | 0.03334 | 36.373 | **0.92377** | 0.00058690 |

PSEM-v3 Whole L2 is 15.49% higher than v1 and 2.15% lower than v2a. Its
`label_4-only` L2 is 33.25%/34.73% higher than v1/v2a, consistent with the
observed label-4 Direct Dice regression. SSIM is not monotonic with Dice, so
reconstruction metrics cannot replace class-decomposition evaluation.

### WORD 2D

Training Job `114767` and formal 24-case evaluation Job `114768` completed.

| Readout | Spleen | Right kidney | Left kidney | Gallbladder | Esophagus | Pancreas | Liver | Stomach | Mean post Dice |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct | 87.84% | 87.39% | 86.65% | 25.11% | 46.65% | 52.98% | 91.49% | 68.72% | **68.353%** |
| Indirect | 86.89% | 85.06% | 82.67% | 32.13% | 41.86% | 56.38% | 91.59% | 69.37% | **68.244%** |

Compared with the same Query20+Loss3 model without Triplet pairing, Direct
improves from 67.209% to 68.353%, whereas Indirect improves from 52.216% to
68.244%. This is the intended effect: Direct, Context, and Plus are paired on
the same source image so that `Prediction(Plus) - Prediction(Context)` is
explicitly trained to isolate the anchor organ. The gain is dataset-dependent;
it is strong on WORD but does not raise the AutoPET ceiling over PSEM-v1.

The branch-level ranked tables, protocol caveats, and complete experiment
registry are maintained in the repository `README.md`.
