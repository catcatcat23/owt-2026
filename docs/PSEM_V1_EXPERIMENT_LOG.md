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

