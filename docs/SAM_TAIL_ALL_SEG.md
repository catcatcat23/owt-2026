# SAM-tail + MAE encoder: all-slot segmentation supervision

Independent controlled ablation against SAM-tail + MAE encoder (no auxiliary
mask supervision). Only intended change: seg_supervision retained -> all.

Entry: slurm/orgslot/train/sam_tail_mae_all_seg.sbatch.
Supply EXPERIMENT_WORKDIR, ARM_F_TARGET and MAE_INIT_CHECKPOINT as for the
existing SAM-tail encoder launcher. Use a new immutable runtime/output job.
Do not resume a retained-supervision run for the primary controlled comparison.

Unchanged: WORD07072, input448, ROI384/probability0.2, 20tokens, channels128,
SAM-tail architecture, MAE encoder initialization, microbatch16 x4GPU xaccum3
=192, lambda_seg0.01, existing small-organ loss, optimizer/LR/update budget.
Background weight remains0: all means all eight visible foreground organs,
including empty-slice negative supervision, not a newly supervised background.

Reconstruction still samples legacy_batch keep and forces the ROI focus slot
to remain. Its target and canvas fusion use this unchanged keep. Segmentation
uses all foreground head rows. Model computes slots using keep OR head_mask,
so reconstruction-dropped organs receive real token features, not zero masks.
Shared parameters still receive gradients from both branches; this is NOT
gradient isolation. Loss values/scales and cost may change when more rows are
supervised; lambda and loss reduction are deliberately unchanged for this test.

Default ARM_F_SEG_SUPERVISION remains retained. Only this new entry sets all.
Existing checkpoints and model implementation remain unchanged. Reuse the
SAM-tail encoder unified evaluation protocol; report full24 cases/6990 slices,
fixed and train-calibrated thresholds, recon, per-organ P/R and volume ratio.
Existing seg_*_supervised_samples/positive_samples/negative_samples logs allow
inspection of effective supervision; distinguish seg_* from unrelated zero
positive_* auxiliary-loss counters. Incremental training must still obey label
visibility; this launcher is for the fully labelled Common8 experiment only.

CPU test verifies exact reconstruction invariance for the same keep, real
dropped-slot computation and finite nonzero segmentation gradients. GPU memory,
throughput and full training have not been validated or submitted by this change.
