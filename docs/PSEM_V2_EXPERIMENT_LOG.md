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
