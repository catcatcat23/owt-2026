# E cross-attention + MAE encoder

Purpose: test existing E query_refinement=cross_attn with AutoPET MAE encoder initialization.
Mean-pooled organ query reads P4, then dot-product mask; no F multi-scale token refinement or reverse attention.

Recipe: slurm/orgslot/train/arm_e_crossattn_mae_encoder.sbatch.
WORD07072, 448/384 ROI20, seed0, 20 tokens, channels128, small-organ loss lambda0.01,
top-k background, 4 GPUs x microbatch16 x accumulation3 =192, lr7.5e-5,
118800 updates, 5940 warmup, BF16, clip1.0. Decoder is randomly initialized.
Same MAE checkpoint SHA as existing encoder-only experiments.

Evaluation: arm_e_mae_unified.sbatch; training6000 threshold calibration, test6990/24 cases,
fixed0.5 and calibrated head, reconstruction threshold0.02. Report eight classes separately.
Historical E encoder 84.99 is a reference, not a perfectly controlled comparison:
historical precision/microbatch/resume trajectory differ. Do not attribute all changes to cross-attention.
No claim of exceeding85 before complete evaluation.
