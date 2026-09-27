# SAM-tail: differentiable coarse-mask guidance ablation

Keep the existing MAE-encoder SAM-tail baseline. Add three runs:

| Variant | HSAM_SUPERVISION | Coarse loss |
|---|---|---|
| H-SAM-inspired low resolution | downsample_gt | P4 logits vs nearest-resized GT |
| Full-label supervision | upsample_logits | bilinear-upsampled P4 logits vs full GT |
| M2F-style routing | m2f_hard | M0-M3 upsampled logits vs full GT |

The third run uses a shared LayerNorm/3-layer mask MLP on mean organ tokens,
with shared P4 pixel features. Predict M0 before P16, M1 after P16, M2 after
P8, M3 after P4. Resize the preceding logits to each attention scale, threshold
sigmoid at .5, detach boolean routing, and reset fully blocked rows to global
attention. Keep SAM-tail unchanged without soft guidance. Use
lambda_seg*(final_loss + L_M0 + L_M1 + L_M2 + L_M3), i.e. equal weights rather
than H-SAM scheduling. This increases segmentation supervision strength versus
the baseline; do not attribute differences solely to routing. It adapts the
M2F mechanism, not instance matching or its original class/mask losses.

All use the same MAE encoder (encoder only), seed, WORD07072 ROI20,
microbatch16/accumulation3 on 4 GPUs (effective192), small-organ loss,
lambda_seg=.01 and unchanged training update budget. No aux-only run.

For the two H-SAM-inspired variants, after P16/P8/P4 refinement, mean-token cosine dot original P4 gives coarse
logits. Their sigmoid probabilities modulate both token-to-pixel reads in
SAM-tail. The gate multiplies projected V (equivalent to attention probability
multiplication), with no detach, floor or renormalization. Reverse reads are
unchanged. This is NOT a complete H-SAM reproduction (no CMAttn, SAM pretrained
decoder, or hierarchical high-resolution decoder).

For those two variants, final weight w(epoch)=0.6**(0.990**epoch), following the public H-SAM trainer,
not the different schedule described in its paper. Total segmentation term:
.01*((1-w)*coarse_loss+w*final_loss). Other losses unchanged. Coarse readout is
uncalibrated; final readout retains existing calibration. Monitor both losses,
final weight and lost positive slices after GT resizing.

Use slurm/orgslot/train/arm_f_sam_tail_hsam.sbatch with each mode. Pin source
and verify revision before submission. No GPU jobs have yet been submitted.
Evaluation restores hsam_supervision from checkpoint args; final-mask-only
evaluation is the main protocol (no coarse/final ensemble). Existing checkpoint
parameters remain compatible; omitting the mode defaults to the old behavior,
so evaluation must explicitly verify it instead of silently using the default.
