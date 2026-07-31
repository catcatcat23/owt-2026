# LossBalance-v3a Experiment Log

## Controlled question

LossBalance-v2 improved WORD 2D Direct post Dice from 55.07 (OWT) to
65.44, but Whole L2 remained 260.75% above OWT and small-organ predictions
were oversized. V2 also applied the same frequency-balanced ROI term to both
kept classes (positive reconstruction) and removed classes (zero target).

V3a tests one isolated change:

> Does weak, frequency-balanced ROI supervision applied only to present and
> kept organ classes retain small-organ recall while improving precision,
> whole-image reconstruction, and token compositionality?

PSEM and marginal-consistency losses are intentionally excluded from this
first experiment.

## Objective

The unchanged OWT masked target is used by global L2 and LPIPS:

```text
L = L_global + L_LPIPS + lambda_positive * L_positive
```

For sample `b` and foreground class `c`:

```text
positive_active[b,c] = present[b,c] AND keep[b,c]
L_positive = (1 / B) * sum_b sum_c
             positive_active[b,c] * w[c] * organ_mean_l2[b,c]
```

The WORD frequency weights are identical to LossBalance-v2:

```text
alpha = 0.5
max_weight_ratio = 4.0
```

The only optimized ROI coefficient is:

```text
lambda_positive = 0.25
```

Removed foreground classes are separately measured against the standard
zeroed target, but `removed_monitor_loss` is detached from the optimized
objective. Global L2 and LPIPS still supervise removed regions.

## Isolation and compatibility

- Branch: `experiment/lossbalance-v3`
- Worktree: `/gpfs/work/aac/bolinren19/OD_OWT_lossbalance_v3`
- Base: `experiment/lossbalance-v2@350a867`
- Original OWT files remain unchanged.
- The original batch-shared random masking policy remains unchanged.
- `class_keep_mask` is derived directly from the classes physically removed
  from the token bank and has shape `[B, C+1]`.
- Background and absent classes are excluded from Positive/Removed ROI stats.
- Class weights are a non-persistent buffer; state-dict keys and shapes remain
  identical to original OWT.

## New training diagnostics

- `positive_roi_loss`
- `positive_valid_samples`
- `positive_weighted_mass`
- `positive_c{class}_sum/count/weight/weighted_sum`
- `removed_monitor_loss`
- `removed_valid_samples`
- `removed_mass`
- `removed_c{class}_sum/count`

The formal evaluation must additionally report per-class Precision, Recall,
and predicted-volume/GT-volume ratio, because v2's Dice improvement included
substantial over-segmentation.

## Controlled WORD 2D configuration

- Dataset: WORD Common8 2D
- Input: 224, `fixed_255`
- Architecture: original OWT v11, 20 tokens per class
- GPUs: 2 A800
- Batch: 32 per GPU, effective global batch 64
- Base LR: `1e-4`
- Epochs: 1200
- Global loss: L2 + LPIPS
- Positive ROI: frequency-balanced, weight 0.25
- Inference: unchanged OWT Common8 evaluator and thresholds

## Launch gate

1. Local loss/model tests, 2D/3D forward-backward, and tiny overfit.
2. One-epoch WORD 2D smoke.
3. Smoke validator checks checkpoint, finite required metrics, and nonzero
   Positive/Removed state coverage.
4. Full training is submitted with `afterok` dependency on the validated smoke.
5. Formal evaluation is submitted with `afterok` dependency on full training.

## Local validation

- `git diff --check`: passed.
- Python syntax compilation: passed.
- All three Slurm scripts pass `bash -n`.
- 13/13 unit tests pass.
- Positive/Removed routing and gradient isolation pass.
- Original OWT/v3 state-dict key and shape compatibility passes.
- 2D and Fixfr4-style 3D forward/backward pass with finite gradients.
- The 20-step one-sample tiny overfit reduces the loss by at least 20%.

## Jobs

| Job | Stage | Status |
|---:|---|---|
| 1627035 | WORD 2D smoke, 2 GPU, 4a800 | PENDING (Priority) |
| 1627036 | WORD 2D full, 2 GPU, 4a800 | PENDING (afterok:1627035) |
| 1627039 | WORD 2D formal evaluation, 1 GPU, 8a800 | PENDING (afterok:1627036) |

Submission time: 2026-07-31 15:47 CST. At submission, Slurm estimated
the smoke start at 2026-08-03 06:14 CST; dependency job start times remain
unknown until their prerequisites finish.
