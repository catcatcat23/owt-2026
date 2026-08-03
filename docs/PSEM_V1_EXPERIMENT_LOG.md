# PSEM-v1 Experiment Log

## Experiment Identity

- Method: Per-Sample Exhaustive Present-Label Mask (PSEM-v1)
- Git branch: `experiment/psem-v1`
- Worktree: `/gpfs/work/aac/bolinren19/OD_OWT`
- Baseline worktree: `/gpfs/work/aac/bolinren19/2026-07/OD_OWT`
- Baseline branch: `main`
- Starting commit: `6cdf9b5`
- Status: AbdAutoPET 2D and WORD 2D formally evaluated; AbdAutoPET 3D running

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

## 2026-07-25 Full training and evaluation chain

Both smoke jobs completed successfully with finite losses and saved
checkpoints:

- 2D smoke `1561497`: max memory about 5.76 GB.
- 3D Fixfr4 smoke `1561498`: max memory about 7.15 GB.

The full runs were then submitted across both available account/QOS
associations:

- PSEM AbdAutoPET 2D full: job `1563805`,
  `account=sifansong`, `qos=4a800`, 2 GPUs.
- PSEM AbdAutoPET 3D Fixfr4 full: job `1563807`,
  `account=angelosstefanidis`, `qos=8a800`, 2 GPUs.

Evaluation jobs:

- PSEM 2D evaluation: job `1563831`, `afterok:1563805`.
- PSEM 3D Fixfr4 evaluation: job `1563842`, `afterok:1563807`.

The shared 3D evaluator defaults to fixed-255 intensity scaling, whereas
PSEM AbdAutoPET was trained with per-sample normalization. The small wrapper
`tools/evaluate_psem_abdautopet_3d.py` therefore reuses the shared evaluator
while forcing `per_sample` case loading and explicit `label_1` through
`label_4` names. Its import and command-line entry point were smoke-tested
before job `1563842` was submitted.

LossBalance evaluation jobs were also scheduled as part of the same queue
fill:

- completed WORD 2D checkpoint evaluation: job `1563822`;
- WORD 3D evaluation: job `1563824`, `afterok:1560196`.

At the final submission check, full PSEM training and immediate LossBalance
2D evaluation were pending for priority; the three dependent evaluations
were pending for their corresponding successful training completion. No
QOS submission limit was reached.


## 2026-08-02 Formal Result And Queue Refresh

### AbdAutoPET 2D

- Full training job 1563805 and evaluation job 1563831 completed.
- Direct post macro Dice is 90.11 versus OWT 89.94.
- Indirect post macro Dice is 90.51 versus OWT 90.39.
- Whole L2 is 0.00015039 versus OWT 0.00015664.
- The gains are only +0.17 and +0.13 points from one training seed. This is
  recorded as limited or marginal evidence, not a stable improvement claim.

### WORD Common8 2D

- Smoke 1605665, full training 1605670, and evaluation 1605673 completed.
- The evaluation used the same 24-case CSV, fixed-255 normalization, thresholds,
  component filtering, spacing, and common evaluator as the original OWT.
- Direct raw/post macro Dice is 34.48/36.27 versus OWT 55.30/55.07.
- Indirect raw/post macro Dice is 17.66/34.87 versus OWT 47.23/58.31.
- Whole L2 is 0.00171145, 5.45 times the OWT value.
- Organs-only L2 is 0.00667906, 9.83 times the OWT value.
- Direct predicted-volume/GT-volume ratio is 2.664 after class macro averaging.
- Gallbladder and esophagus Direct post Dice become 10.83 and 9.89 instead of
  zero, but large-organ performance collapses. The net result is formally
  classified as ineffective.

The WORD failure does not indicate a padding leak: TGEnc and AHER padding
isolation tests pass. The design problem is that PSEM-v1 uses GT presence both
as supervision and token eligibility. WORD training samples contain only
2.276 present classes on average and keep 1.511, whereas Whole, organs-only,
and leave-one-out inference combine most or all nine token groups. AbdAutoPET
has fewer and more frequent classes, so its train/inference state mismatch is
much smaller.

### AbdAutoPET 3D

- The original two-GPU job 1563807 was cancelled after 19:12:08.
- The run resumed from checkpoint-100 as four-GPU job 1567343.
- At this refresh it is running at epoch 1098 with L2 0.000926 and LPIPS
  0.018219.
- Evaluation job 1563842 waits on successful completion of 1567343.

### Follow-up

PSEM-v2a was created in a separate worktree to keep the padding-safe model but
train explicit Direct-positive, Direct-negative, leave-one-out, Whole,
background-only, organs-only, and random full-bank query states. Its evidence
must remain separate from PSEM-v1 until its WORD and AbdAutoPET 2D evaluations
finish.
