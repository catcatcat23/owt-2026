# 历史原文档案（按需查询）

来源提交：`54d78761775af848a311fc0661c61aed5ec5de4b`。
原文逐字保留；所有状态、指令和路径仅反映各自历史时点。
不要默认读取本文件；先读 ../README.md。

- [docs/PSEM_V1_EXPERIMENT_LOG.md](#source-1)
- [docs/PSEM_V2_EXPERIMENT_LOG.md](#source-2)
- [docs/PSEM_V3_TRIPLET_EXPERIMENT.md](#source-3)
- [README.md](#source-4)

<a id="source-1"></a>

## docs/PSEM_V1_EXPERIMENT_LOG.md

SHA256: `4eb00aeb9b86e9c8f9dd188592cd2871873afca7f5d6e4e24916471175400635`

````text
# PSEM-v1 Experiment Log

## Experiment Identity

- Method: Per-Sample Exhaustive Present-Label Mask (PSEM-v1)
- Git branch: `experiment/psem-v1`
- Worktree: `/gpfs/work/aac/bolinren19/OD_OWT`
- Baseline worktree: `/gpfs/work/aac/bolinren19/2026-07/OD_OWT`
- Baseline branch: `main`
- Starting commit: `6cdf9b5`
- Status: implementation in progress

## Goal

Test whether organ-token learning improves when every sample receives its own
deterministic mask schedule over the organ labels that are actually present.
This experiment changes the masking strategy only. It does not include CropMix,
LossBalance, classification loss, or organ-slot architecture changes.

The first controlled comparison uses the original OWT AbdAutoPET data:

- 2D CSV: `/gpfs/work/aac/bolinren19/2026-07/Training_A100_Final112.csv`
- Fixfr4 3D CSV: `/gpfs/work/aac/bolinren19/2026-07/Training_Fixfr4_A100_Final112.csv`
- Foreground classes: 4
- Input size: 224
- Tokens per class: 20
- Reconstruction loss: L2 + LPIPS

## Isolation Rules

1. Existing jobs continue to read the baseline worktree on `main`.
2. PSEM code and Slurm jobs read only the independent PSEM worktree.
3. PSEM outputs use dedicated result and log directories.
4. Generated results are not committed to Git.

## Step 1: Baseline Code Audit

### Purpose

Identify the smallest correct change needed for per-sample masking and establish
which model components must remain unchanged for a fair comparison.

### Findings

- `engine_pretrain.py` creates one shuffled class list per batch.
- The resulting `random_selected_class` is shared by every sample in that batch.
- The target image is zeroed at pixels belonging to those deleted classes.
- `OWT_models.py::random_masking` physically removes each selected class's token
  group from every sample.
- The Token Group Encoder (`blocks2`) therefore receives a shorter token
  sequence, but all samples in the batch currently have the same length.
- AHER converts the variable organ-token sequence back to a fixed spatial patch
  sequence before the decoder.
- With `token_factor=20` and 8 foreground classes plus background, the full organ
  bank contains 180 tokens.

### Consequence

Per-sample class subsets produce different token lengths inside one batch. PSEM
must pack each sample's selected token groups, pad only to the current batch
maximum length, and carry a validity mask through the Token Group Encoder and
AHER. Invalid padded positions must be excluded from softmax and residual paths,
not merely initialized to zero.

### Controlled Variables

The following stay identical to the baseline:

- Patch Encoder and positional embeddings
- OrganCollector and number of tokens per class
- Token Group Encoder parameter shapes
- AHER parameter shapes
- Decoder
- Global L2 plus LPIPS reconstruction objective
- Optimizer, learning-rate scaling, warmup, and epoch count

### Acceptance Criteria

- A sample may delete only classes present in its own label.
- A sample must retain at least one present class.
- All valid deletion subsets are visited deterministically.
- Different samples in one batch may have different retained token lengths.
- Padded tokens have no effect on TGEnc or AHER outputs.
- Existing checkpoints remain loadable without parameter-shape changes.
- Both 2D and Fixfr4 3D complete finite forward/backward smoke tests.

## Step 2: AbdAutoPET Baseline Configuration Audit

### Purpose

Make the first PSEM run directly comparable with the completed original OWT
AbdAutoPET experiment.

### Findings

- The completed 2D baseline used 2 A800 GPUs, batch size 96 per GPU, 1200
  epochs, 60 warmup epochs, base learning rate `1e-4`, and weight decay `0.05`.
- The 2D model is `mae_vit_base_patch16-LA`.
- The 3D model is `mae_vit_basefix16_patch16-LA` with
  `v01-3D-Fixfr4-TS1`.
- AbdAutoPET has 4 foreground classes. Including background, OrganCollector
  creates `5 * 20 = 100` organ tokens.
- The completed 2D baseline checkpoint is stored under the baseline worktree's
  `Results/AbdAutoPet_2D` directory.

### Execution Decision

Run deterministic unit tests first, then short 2D and 3D GPU smoke tests.
Submit the 1200-epoch controlled comparison only after the smoke logs show
finite losses, correct per-sample mask statistics, and compatible checkpoints.
## Step 3: PSEM-v1 Implementation

### Data Layer

- datasets/dataset3D.py now returns the stable CSV row index as sample_index.
- The original per-sample min-max normalization remains the default for the
  original AbdAutoPET comparison.
- Optional fixed_255 support is retained for later pre-windowed datasets, but it
  is not enabled in this AbdAutoPET experiment.
- DataLoader uses drop_last=False so the final samples of each epoch are not
  systematically omitted.

### Per-Sample Schedule

For sample i, the transformed label determines its present class set P_i. If
m_i = |P_i|, PSEM uses a cycle of:

N_i = 2 ** m_i - 1

The cycle includes deleting no class and every partial deletion subset, but
excludes deleting all present classes. A stable offset derived from sample_index
prevents all samples from using the same combination in the same epoch.

Exact exhaustive coverage assumes a stable present set. Spatial augmentation
can very occasionally move a tiny class outside the field of view; in that
case, the schedule correctly follows the transformed mask seen by the model,
and the cycle is defined for that observed present set.

### Model Layer

- OrganCollector still creates the same fixed class-token bank.
- Each sample keeps only the token groups selected by its own class mask.
- Sequences are padded to the longest retained sequence in the current batch.
- TGEnc masks padded keys before its token-axis softmax and clears padded
  residual positions after attention and MLP.
- AHER masks padded token logits before softmax and masks padded values.
- No learned parameter was added, removed, or resized.

### Training Layer

- New entry point: main_pretrain_psem.py
- New engine: engine_pretrain_psem.py
- New model module: OWT_models_psem.py
- New schedule module: util/per_sample_mask_schedule.py
- Dedicated result root: Results/PSEM_v1
- Dedicated Slurm scripts: slurm/psem

## Step 4: Local Verification

### Automated Tests

Nine tests passed in the abdpet Conda environment:

1. Complete valid combination coverage.
2. Independent masks for different samples in one batch.
3. Per-sample target-image masking.
4. 3D present-label detection.
5. TGEnc padding invariance.
6. AHER padding invariance.
7. Original/PSEM checkpoint key and tensor-shape compatibility.
8. Finite 2D forward and backward pass.
9. Finite Fixfr4 3D forward and backward pass.

A PyTorch 1.13 compatibility issue in multi-axis any was found during the first
run and fixed by flattening spatial dimensions before reduction.

### Real AbdAutoPET Data Check

- 2D CSV rows: 78,400
- Fixfr4 3D CSV rows: 76,300
- 2D tensor shape: [B, 3, 224, 224]
- 3D tensor shape: [B, 3, 4, 224, 224]
- Image range: [0, 1]
- Observed labels were within the configured range 0 to 4.
- A background-only 2D slice correctly receives a one-state cycle and is never
  left with zero valid tokens.
## Step 5: First Slurm Submission Attempt

### Action

Submitted slurm/psem/smoke_psem_abdautopet_2d.sbatch after creating its
dedicated log directories.

### Result

Slurm rejected the submission before creating a job:

QOSMaxSubmitJobPerUserLimit

The 4a800 QOS currently permits four submitted jobs for this user, and all four
slots are occupied by jobs 1560192, 1560193, 1560195, and 1560196.

### Queue Audit

- 1560192: WORD 2D LossBalance training, pending by priority.
- 1560193: WORD 3D LossBalance training, pending by priority.
- 1560195: the same WORD 2D training script, dependent on 1560192.
- 1560196: the same WORD 3D training script, dependent on 1560193.

The dependent jobs use the same commands and output locations as their parent
jobs, so they appear to be duplicate reruns rather than evaluation jobs. No
existing job was cancelled automatically. No PSEM job ID exists yet.
## Step 6: Successful Smoke Submissions

At the next progress check, the two parent LossBalance jobs had completed and
their dependent full runs were active. This confirmed that the dependent jobs
were intentional smoke-to-full transitions, so preserving them was correct.

The freed QOS slots were used for the PSEM smoke tests:

- Job 1561497: AbdAutoPET 2D PSEM smoke.
- Job 1561498: AbdAutoPET Fixfr4 3D PSEM smoke.
- Initial state: PENDING (Priority).
- Requested resources per job: 2 A800 GPUs, 16 CPUs, 128 GB RAM.
- Time limit: 30 minutes.
- Scheduler estimate at submission: 2026-07-25 23:30:39.

No PSEM error or result files exist yet because neither job has started.

````

<a id="source-2"></a>

## docs/PSEM_V2_EXPERIMENT_LOG.md

SHA256: `d103553607b82321d038f4d2657daa3b97073fe743f0e1bbef0e9caac338c0e1`

````text
# PSEM-v2a Query-Matched Mask Experiment Log

## Identity

- Method: PSEM-v2a Query-Matched Mask
- Branch: `experiment/psem-v2-query-mask`
- Worktree: `/gpfs/work/aac/bolinren19/OD_OWT_psem_v2`
- Starting point: PSEM-v1 commit `8a7da1f`
- Implementation commit: `d7a0170`
- Model and optimized loss: unchanged PSEM model with global L2 plus LPIPS
- Datasets: WORD 2D and AbdAutoPET 2D
- Status: implementation validated locally; smoke and training chains submitted

## Motivation

PSEM-v1 allowed only GT-present token groups to enter the model. On WORD 2D,
this made token presence act like an existence label and created a train versus
inference mismatch. Direct queries hallucinated absent organs and leave-one-out,
whole, and organs-only inference used many more token groups than training.

PSEM-v2a keeps per-sample packing and padding isolation, but changes the mask
from a present-only deletion subset to a full-bank query. GT presence selects
positive versus negative supervision; it no longer restricts token eligibility.

## Twenty-Slot Schedule

For sample i:

`slot_i = (epoch + offset_i) mod 20`

The stable offset is `(17 * sample_index + 13) mod 20`. Since 17 and 20 are
coprime, every 20 consecutive sample indices cover all slots once at a fixed
epoch.

| Slots | Mode | Fraction | Query |
|---|---|---:|---|
| 0-3 | Direct-positive | 20% | Keep one present foreground class |
| 4-7 | Direct-negative | 20% | Keep one absent foreground class |
| 8-12 | Leave-one-out | 25% | Keep the full bank except one foreground class |
| 13-14 | Whole | 10% | Keep the full bank |
| 15 | Background-only | 5% | Keep background only |
| 16 | Organs-only | 5% | Keep all foreground classes |
| 17-19 | Random subset | 15% | Keep a nonempty full-bank subset |

Fallbacks are deterministic. A direct-positive slot on a background-only slice
becomes direct-negative. A direct-negative slot with every foreground present
becomes direct-positive. No query may keep zero token groups.

## Unchanged Components

- OrganCollector and token bank sizes
- Token Group Encoder and AHER parameters
- Physical per-sample token packing
- Batch Lmax padding and valid_token_mask isolation
- Decoder and checkpoint key or tensor shapes
- L2 plus LPIPS objective
- Optimizer, learning rate, warmup, epoch count, and dataset preprocessing

## New Diagnostics

Training logs include:

- Fraction and image-space MSE for every query mode
- Direct-positive and direct-negative fractions per class
- Present versus absent leave-one-out fractions
- Negative-query prediction energy
- Present, kept, dropped, and padded token statistics

## Local Verification

Fourteen tests pass in the abdpet environment:

1. PSEM-v1 exhaustive schedule regression tests
2. Independent masks and 2D or 3D target construction
3. TGEnc and AHER padding isolation
4. Original OWT state-dict key and shape compatibility
5. Finite 2D and 3D forward or backward
6. Exact 20-slot mode weights
7. All slots covered by 20 consecutive sample indices
8. Direct-positive target semantics
9. Direct-negative all-zero target semantics
10. Exact direct, leave-one-out, whole, background, and organs query masks
11. Positive and negative query tiny overfit with decreasing hallucination energy

Python compilation, Bash syntax checks, and git diff whitespace checks pass.

## Controlled Runs

### WORD 2D

- CSV: `WORD_Training_2D_224.csv`
- Classes: 8 foreground plus background
- Normalization: `fixed_255`
- GPUs: 2 A800
- Batch: 32 per GPU
- Epochs: 1200
- Evaluation: original common WORD 2D evaluator and thresholds

### AbdAutoPET 2D

- CSV: `Training_A100_Final112.csv`
- Classes: 4 foreground plus background
- Normalization: `per_sample`
- GPUs: 2 A800
- Batch: 96 per GPU
- Epochs: 1200
- Evaluation: original AbdAutoPET 2D evaluator and thresholds

Each dataset uses a one-epoch real-data smoke. The full training depends on a
successful smoke validator, and evaluation depends on successful full training.

## Slurm Jobs

| Dataset | Stage | Job ID | Account | State at submission | Dependency |
|---|---|---:|---|---|---|
| WORD 2D | Smoke | 1629558 | sifansong | PENDING (Priority) | None |
| WORD 2D | Full training | 1629560 | sifansong | PENDING (Dependency) | afterok:1629558 |
| WORD 2D | Evaluation | Not submitted | sifansong | QOSMaxSubmitJobPerUserLimit | afterok:1629560 planned |
| AbdAutoPET 2D | Smoke | 1629559 | angelosstefanidis | PENDING (Priority) | None |
| AbdAutoPET 2D | Full training | 1629561 | angelosstefanidis | PENDING (Dependency) | afterok:1629559 |
| AbdAutoPET 2D | Evaluation | 1629564 | angelosstefanidis | PENDING (Dependency) | afterok:1629561 |

The initial jobs `1629541`-`1629544` and `1629547` were cancelled before
execution because the smoke batch sizes did not match the full runs. The jobs
above use corrected per-GPU smoke batches of 32 (WORD) and 96 (AbdAutoPET).

The WORD evaluation script is ready, but its first submission was rejected by
the `sifansong` QOS submit-count limit. It must be submitted after that account
releases one job slot. This does not block the smoke or full training chain.

## Completion Criteria

- Both smokes create checkpoint-0 and finite logs
- Every query mode has nonzero coverage
- Direct-negative prediction energy is finite and logged
- No NaN, OOM, or traceback
- Full checkpoint-1199 is created
- WORD whole and organs-only reconstruction no longer catastrophically degrade
- WORD direct predictions no longer have approximately threefold organ volume
- AbdAutoPET performance does not regress materially
````

<a id="source-3"></a>

## docs/PSEM_V3_TRIPLET_EXPERIMENT.md

SHA256: `49981ae01ca3047b6a492127dd2e08cbf938e229f1ee3303fbad14086fe28320`

````text
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
````

<a id="source-4"></a>

## README.md

SHA256: `5206f4739cc5e0c99010af18c01b6e29a81ff9a6129cca1f6ec98980dde7b7ec`

````text
# PSEM 实验总览

本分支统一保存 Per-Sample Exhaustive/Structured Masking（PSEM）系列实现与实验记录。本文只对**相同数据集、相同测试病例和相同正式评估协议**下的结果排序；不同数据集之间不比较绝对数值。

## 结论摘要

- **AbdAutoPET 2D：** PSEM-v1 Indirect 仍是当前 PSEM 系列最优，4 类 post Dice 为 **90.512%**。PSEM-v3 Triplet+Loss3 的 Indirect 为 **89.938%**，与 PSEM-v2a Indirect 基本持平，但没有超过 v1。
- **WORD 2D：** PSEM-v3 同时取得当前 PSEM 系列最优 Direct（**68.353%**）和 Indirect（**68.244%**）。Triplet 的主要收益是修复 Indirect 查询分解，而不是显著提高 Direct。
- **AbdAutoPET 3D Fixfr4：** 目前只有 PSEM-v1 完成正式评估，尚不能判断 v2/v3 在 3D 上是否更好。
- **数值稳定性：** PSEM-v3 AutoPET 原 FP16 路线在约 epoch 500 附近出现 NaN。修复版本采用 BF16、LPIPS FP32、非有限值诊断和梯度裁剪，最终训练 Job `121484` 与 200 病例评估 Job `124936` 均正常完成。

## 统一指标口径

- 排名主指标：`all_cases`、类别等权宏平均 `post Dice`，数值越高越好。
- 辅助指标：`post NSD` 越高越好，`post HD95` 越低越好。
- Direct：只保留目标类别 token 的重建响应。
- Indirect：`input/full reconstruction - without-class reconstruction` 的差分响应。
- 分割结果来自重建响应阈值化及统一后处理，不是额外训练的 segmentation head。
- AutoPET 2D/3D 均为 200 个测试病例、4 个前景类；WORD 2D 为 24 个测试病例、8 个前景类。
- AutoPET 的可靠器官名称映射尚未确认，因此只使用 `label_1`–`label_4`；WORD 类别顺序为脾脏、右肾、左肾、胆囊、食管、胰腺、肝脏、胃。

## AbdAutoPET 2D 排名

相同 200 病例、224 输入、`per_sample` 归一化、固定阈值 0.15 和最小连通域 20。表内类别 Dice 顺序为 `label_1/label_2/label_3/label_4`。

| 排名 | 实验 | 读出 | post Dice | post NSD | post HD95 (mm) | 4 类 post Dice |
|---:|---|---|---:|---:|---:|---|
| 1 | PSEM-v1 | Indirect | **90.512%** | **89.981%** | 7.004 | 96.55 / 93.09 / 93.05 / 79.36 |
| 2 | 原始 OWT（对照） | Indirect | 90.386% | 89.616% | inf\* | 96.51 / 92.93 / 93.04 / 79.06 |
| 3 | PSEM-v1 | Direct | 90.110% | 89.118% | 10.277 | 96.53 / 92.95 / 91.89 / 79.06 |
| 4 | PSEM-v2a Query20 | Direct | 90.063% | 88.731% | **5.932** | 96.41 / 92.73 / 92.85 / 78.27 |
| 5 | PSEM-v3 Triplet+Loss3 | Indirect | 89.938% | 87.639% | 8.074 | 96.36 / 92.56 / 92.69 / 78.15 |
| 6 | 原始 OWT（对照） | Direct | 89.938% | 88.687% | inf\* | 96.45 / 92.69 / 92.94 / 77.66 |
| 7 | PSEM-v2a Query20 | Indirect | 89.888% | 88.421% | 6.904 | 96.38 / 92.70 / 92.70 / 77.77 |
| 8 | PSEM-v3 Triplet+Loss3 | Direct | 88.783% | 83.172% | 7.055 | 96.12 / 91.64 / 91.70 / 75.67 |
| 9 | PSEM+LossBalance-v2 | Indirect | 84.673% | 68.029% | 13.022 | 95.17 / 88.21 / 88.05 / 67.26 |
| 10 | LossBalance-v1（对照） | Direct | 84.186% | 66.665% | 9.672 | 94.82 / 87.68 / 88.01 / 66.23 |
| 11 | LossBalance-v1（对照） | Indirect | 83.299% | 64.387% | 14.753 | 94.64 / 87.17 / 86.98 / 64.40 |
| 12 | PSEM+LossBalance-v2 | Direct | 79.728% | 63.051% | 62.247 | 94.83 / 86.18 / 79.13 / 58.77 |

\* 原 OWT 合并汇总的每种读出各含 1 个无穷 HD95，因此正式宏平均记为 `inf`；Dice/NSD 从完整 200 病例逐病例文件恢复。

### PSEM-v3 的病例配对分析

以下为 10,000 次病例 bootstrap 的“PSEM-v3 减对照”宏平均 Dice 差值；置信区间只反映测试病例抽样，不包含不同训练 seed 的方差。

| 读出 | 对照 | 差值（百分点） | 95% CI | 结论 |
|---|---|---:|---|---|
| Direct | PSEM-v1 Direct | -1.326 | [-1.597, -0.988] | 明确下降 |
| Direct | 原始 OWT Direct | -1.154 | [-1.329, -0.972] | 明确下降 |
| Direct | PSEM-v2a Direct | -1.279 | [-1.436, -1.115] | 明确下降 |
| Direct | PSEM+LossBalance-v2 Direct | +9.056 | [8.675, 9.471] | 明确改善 |
| Indirect | PSEM-v1 Indirect | -0.574 | [-0.718, -0.410] | 明确下降 |
| Indirect | 原始 OWT Indirect | -0.448 | [-0.580, -0.307] | 明确下降 |
| Indirect | PSEM-v2a Indirect | +0.049 | [-0.107, 0.224] | 无法区分，视为持平 |
| Indirect | PSEM+LossBalance-v2 Indirect | +5.265 | [4.978, 5.610] | 明确改善 |

PSEM-v3 的 Indirect 比自身 Direct 高 1.155 个百分点，说明 Triplet/Delta 目标相对更偏向保留 Indirect；但由于 Direct 本身下降更大，这个差值不能单独证明分解质量改善。它没有提升 AutoPET 的总体上限，主要退化来自 `label_4`：Direct 75.67%，比 v1 Direct 低 3.39 个百分点。Direct HD95 从 v1 的 10.277 mm 降到 7.055 mm，但 Dice 和 NSD 同时下降，因此只能说极端边界误差改善，不能说总体分割更好。


## AbdAutoPET 2D 重建指标

按 Whole L2 从低到高排序。L2/LPIPS 越低越好，PSNR/SSIM 越高越好；`label_4-only L2` 用于检查本次最明显退化类别的重建质量。

| 排名 | 实验 | Whole L2 | Whole LPIPS | Whole PSNR | Whole SSIM | label_4-only L2 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | PSEM-v1 | **0.00015039** | **0.01907** | **38.624** | 0.91971 | 0.00019494 |
| 2 | PSEM-v3 Triplet+Loss3 | 0.00017368 | 0.02313 | 37.979 | 0.90798 | 0.00025977 |
| 3 | PSEM-v2a Query20 | 0.00017749 | 0.02356 | 37.908 | 0.92126 | **0.00019280** |
| 4 | PSEM+LossBalance-v2 | 0.00026088 | 0.03334 | 36.373 | **0.92377** | 0.00058690 |

PSEM-v3 的 Whole L2 比 v1 高 15.49%，但比 v2a 低 2.15%，说明总体像素误差与 v2a 接近、仍不如 v1。更关键的是 `label_4-only L2` 比 v1/v2a 分别高 33.25%/34.73%，与 `label_4` Direct Dice 的下降方向一致。另一方面，SSIM 与 Dice 并不单调一致：LossBalance-v2 的 Whole SSIM 最高但分割最差，因此不能用单个重建指标替代类别分解评估。

## AbdAutoPET 3D Fixfr4 排名

目前只有 PSEM-v1 完成同协议 200 病例评估。类别顺序为 `label_1/label_2/label_3/label_4`。

| 排名 | 实验 | 读出 | post Dice | post NSD | post HD95 (mm) | 4 类 post Dice |
|---:|---|---|---:|---:|---:|---|
| 1 | PSEM-v1 | Direct | **87.583%** | 82.450% | 16.677 | 94.54 / 91.71 / 90.33 / 73.75 |
| 2 | PSEM-v1 | Indirect | 87.519% | **83.100%** | **12.238** | 94.99 / 90.77 / 91.90 / 72.41 |

该表不支持“3D 不如 2D”的方法结论，因为 Fixfr4 的输入、训练和空间评估协议与 2D 不同；它只记录当前已完成结果。

## WORD 2D 排名

相同 24 病例、224 输入和 Common8 正式评估。类别顺序为脾脏/右肾/左肾/胆囊/食管/胰腺/肝脏/胃。

| 排名 | 实验 | 读出 | post Dice | post NSD | post HD95 (mm) | 8 类 post Dice |
|---:|---|---|---:|---:|---:|---|
| 1 | PSEM-v3 Triplet+Loss3 | Direct | **68.353%** | 57.516% | 15.133 | 87.84 / 87.39 / 86.65 / 25.11 / 46.65 / 52.98 / 91.49 / 68.72 |
| 2 | PSEM-v3 Triplet+Loss3 | Indirect | **68.244%** | **58.851%** | 26.001 | 86.89 / 85.06 / 82.67 / 32.13 / 41.86 / 56.38 / 91.59 / 69.37 |
| 3 | LossBalance-v3a（对照） | Direct | 68.216% | 55.495% | **15.079** | 87.25 / 86.55 / 85.15 / 24.94 / 46.42 / 53.18 / 91.29 / 70.94 |
| 4 | PSEM-v2 LossBalance-v3 | Direct | 67.209% | 54.177% | 15.080 | 86.79 / 85.38 / 84.09 / 24.11 / 47.05 / 51.20 / 90.58 / 68.46 |
| 5 | LossBalance-v1（对照） | Direct | 64.990% | 47.560% | 16.598 | 85.63 / 82.81 / 82.27 / 24.64 / 33.86 / 53.27 / 89.23 / 68.21 |
| 6 | 原始 OWT（对照） | Indirect | 58.309% | 49.866% | 80.962 | 88.31 / 86.83 / 85.55 / 0.00 / 0.00 / 43.73 / 92.28 / 69.78 |
| 7 | LossBalance-v3a（对照） | Indirect | 56.796% | 47.752% | 69.545 | 86.80 / 86.24 / 83.68 / 0.00 / 0.16 / 40.75 / 91.66 / 65.08 |
| 8 | 原始 OWT（对照） | Direct | 55.072% | 44.279% | inf | 86.00 / 79.30 / 79.98 / 0.00 / 0.00 / 38.43 / 91.02 / 65.84 |
| 9 | PSEM-v2a Query20 | Direct | 52.311% | 40.155% | inf | 83.56 / 66.20 / 79.83 / 0.00 / 2.72 / 31.95 / 90.36 / 63.86 |
| 10 | PSEM-v2 LossBalance-v3 | Indirect | 52.216% | 41.632% | 84.054 | 84.49 / 81.76 / 79.66 / 0.00 / 0.00 / 21.16 / 90.36 / 60.30 |
| 11 | PSEM-v2a Query20 | Indirect | 49.889% | 38.862% | 91.771 | 84.74 / 79.90 / 79.25 / 0.00 / 0.00 / 0.41 / 90.99 / 63.83 |
| 12 | LossBalance-v1（对照） | Indirect | 41.575% | 18.809% | 372.641 | 57.08 / 48.45 / 50.05 / 5.79 / 8.84 / 27.99 / 81.56 / 52.85 |
| 13 | PSEM-v1 | Direct | 36.266% | 29.127% | 319.094 | 37.75 / 43.57 / 45.11 / 10.83 / 9.89 / 45.87 / 46.97 / 50.13 |
| 14 | PSEM-v1 | Indirect | 34.866% | 17.367% | 355.002 | 48.66 / 40.81 / 42.65 / 0.78 / 0.74 / 18.28 / 79.73 / 47.28 |
| 15 | PSEM+LossBalance-v2 | Indirect | 21.129% | 7.103% | 375.216 | 28.79 / 20.57 / 21.84 / 1.49 / 0.18 / 1.17 / 63.60 / 31.39 |
| 16 | PSEM+LossBalance-v2 | Direct | 8.880% | 6.992% | 424.928 | 8.26 / 6.75 / 7.16 / 0.41 / 2.22 / 2.70 / 31.17 / 12.36 |

PSEM-v3 在 WORD 上相对 PSEM-v2+LossBalance-v3 的 Direct 增益为 1.14 个百分点，相对独立 LossBalance-v3a 的 Direct 增益仅 0.14 个百分点；关键变化是 Indirect 从 52.22% 提高到 68.24%（相对独立 LossBalance-v3a 也提高 11.45 点），并与 Direct 只差 0.11 个百分点。这直接支持 Triplet 的设计目标：让 Direct、Context、Plus 在同一源样本上成对训练，使 `Plus - Context` 与目标器官响应一致。Indirect 仍明显依赖后处理，因此报告时必须同时保留 raw/post，而不能把全部提升归因于网络。

## 已完成实验登记

| 版本 | 核心改动 | 数据集与正式评估 | 当前判断 |
|---|---|---|---|
| PSEM-v1 | 只允许 GT-present 类进入逐样本穷举删除日程；保持原 L2+LPIPS | AutoPET 2D、AutoPET 3D Fixfr4、WORD 2D | AutoPET 2D 当前最优；WORD 存在训练/推理查询状态错配 |
| PSEM-v2a Query20 | 20-slot 全 token-bank 查询日程，包含正/负 Direct、leave-one-out、whole、background、organs 和随机子集 | AutoPET 2D、WORD 2D | AutoPET 与 v1 接近；WORD 大器官恢复，但小器官仍严重失败 |
| PSEM+LossBalance-v2 | PSEM-v1 加按样本频率加权的 ROI 损失 | AutoPET 2D、WORD 2D | 两个数据集均明显退化，判定该组合无效 |
| PSEM-v2 LossBalance-v3 | Query20 加只优化 present-and-kept 正 ROI 的 Loss3 | WORD 2D | Direct 提升到 67.21%，但 Indirect 仍只有 52.22% |
| PSEM-v3 Triplet+Loss3 | 每个源样本构造 Direct/Context/Plus，优化三分支 Loss3 与 `0.1 × Delta loss` | WORD 2D；AutoPET 2D | WORD 修复 Indirect；AutoPET 与 v2a 持平、低于 v1，收益不具跨数据集一致性 |

### PSEM-v3 正式运行

| 数据集 | 训练 | 评估 | 资源/规模 | 状态 |
|---|---|---|---|---|
| WORD 2D | XEC Job `114767` | XEC Job `114768` | 2×A800；1200 epochs；24 病例评估 | 完成 |
| AbdAutoPET 2D | sifansong/XEC Job `121484` | sifansong/XEC Job `124936` | 4×A800；有效 source batch 192；1200 epochs；200 病例评估 | 完成；BF16/LPIPS-FP32 数值修复版本 |

AutoPET 原 FP16 训练链曾在约 epoch 500 附近出现非有限值。最终有效版本保留模型、数据、Triplet/Loss3 定义和优化预算，只修改数值执行策略：BF16 autocast、LPIPS FP32、梯度裁剪与逐分量有限值检查。该变化必须随 checkpoint 一起记录，不能把最终模型描述成未经修复的原 FP16 运行。

## 结果文件与详细记录

- PSEM-v1：`docs/PSEM_V1_EXPERIMENT_LOG.md`
- PSEM-v2a：`docs/PSEM_V2_EXPERIMENT_LOG.md`
- PSEM-v3：`docs/PSEM_V3_TRIPLET_EXPERIMENT.md`
- 正式结果以各实验目录的 `step3_segmentation_summary.csv`、`step3_segmentation_per_case.csv` 和 `protocol.json` 为准。

## 下一步

1. PSEM-v3 AutoPET 至少补 2 个 seed；当前病例 bootstrap 不包含训练随机性，不能据单 seed 宣称稳定差异。
2. 优先分析 AutoPET `label_4` 的 false negative、预测体积比和 Direct/Indirect 差分图，确认 Direct 回退来自类 token 读出还是 Triplet 的预算分配。
3. 若要判断 PSEM-v3 是否适用于 3D，需要在 AutoPET 3D Fixfr4 上补同协议 v3 对照；不能拿 2D 与 3D 的当前单次结果直接排名。
4. WORD 继续报告 raw/post 两套结果，并做固定后处理的消融，量化 Indirect 提升中模型与后处理各自的贡献。
````
