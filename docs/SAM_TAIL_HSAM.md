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
and verify revision before submission.
Evaluation restores hsam_supervision from checkpoint args; final-mask-only
evaluation is the main protocol (no coarse/final ensemble). Existing checkpoint
parameters remain compatible; omitting the mode defaults to the old behavior,
so evaluation must explicitly verify it instead of silently using the default.

## Submitted 2026-09-27 (Asia/Shanghai)

### Background-only coarse-loss fix (2026-09-28)

Job 2992278 failed when coarse predictions were empty. Legacy batch TGR can
retain background alone; with no ROI focus and lambda_bg_seg=0, no head runs.
This is legal: coarse loss is now zero and reconstruction backward continues.
Missing predictions for any supervised slot still raise an error (including
partially missing dictionaries). All three supervision modes share the fix.
Five CPU tests pass, including real-model background-only finite backward;
multi-rank GPU validation has not been run. Existing immutable runtime jobs
do not inherit this fix: replace their snapshots and evaluation dependencies
before claiming deployment. Keep retained supervision and original loss weights.

Pinned code: `569582d41b9baf481ca7712e9e7cbf0f27aab16b`.
Archive SHA256: `8e15cd5c166f9a79055fccca8a7e5e06621bfe03c016ecf539777047329a3499`.
MAE SHA256: `0b3571a3e79095686a0b12649735a298aaeb8a0b3bb93a47f3484439ac4a1ec6`.

| Mode | User/cluster | QoS | Training | afterok evaluation | Time limit |
|---|---|---|---|---|---|
| downsample_gt | bolinren19/SIP | 4a800 | 2992278 | 2992279 | 7 days |
| upsample_logits | sifansong/XEC | 8gpus | 148892 | 148895 | 5 days |
| m2f_hard | antengcai23/XEC | 4gpus | 148893 | 148896 | 7 days |

All Slurm accounts are `sifansong`; this is distinct from the login username.
Training: 4 A800, 20 CPU, 192 GB. Evaluation: 1 A800, 10 CPU, 128 GB.
No automatic requeue (avoid existing-output restart failure). Queued, GPU
validation not yet run. CPU: 9 existing-head regression tests and 3 new tests
passed locally; 3 new tests passed in each XEC environment.

Runtime paths (do not pull or edit under queued/running jobs):
- SIP: `/gpfs/work/aac/bolinren19/OD_OWT/.worktrees/sam_guided_569582d`
- sifansong/XEC: `/gpfs/work/aac/sifansong/worktrees/sam_guided_569582d`
- antengcai23/XEC: `/gpfs/work/aac/antengcai23/worktrees/sam_guided_569582d`

Train script: `slurm/orgslot/train/arm_f_sam_tail_hsam.sbatch`.
Evaluation: `slurm/orgslot/eval/sam_tail_guided.sbatch`, validates saved mode,
MAE scope and head, then runs 6000 training-calibration slices, full test head
evaluation and reconstruction threshold .02 using the existing protocol.
Training output: `Results/OrganSlotBank/Common8/WORD_2D/ArmF_2D_JOBID`.
Check start logs for encoder-only load, microbatch16, accumulation3, finite
coarse/final loss, lost-positive count and correct mode before trusting results.
