# Arm E MAE encoder+decoder recovery

133897 failed near epoch541. Ranks 0/2/3 report the same NCCL ALLREDUCE sequence
27417850 timeout; follow-on stacks are in DDP buffer synchronization. No initiating
rank1 stack is available. Transport failure versus collective desynchronization is
not proven. Do not describe the issue as a confirmed NaN or fixed root cause.

Mitigations: sorted packed metric reduction with fixed-size schema handshake
(all ranks fail together on schema mismatch), opt-in disabling of per-forward
buffer broadcasts only when buffers are exactly frozen LPIPS shift/scale. The
initial DDP state synchronization and parameter gradient reduction remain enabled.
Resume logs enable distributed DETAIL diagnostics to expose mismatched collectives.

Resume checkpoint500, including optimizer and FP16 GradScaler, at epoch501;
microbatch8 x 4 GPUs x accumulation6 =192, same loss, ROI20, spacing0.7, LR schedule,
118800 total updates. Clipping disabled to preserve historical no-clipping FP16
policy. GradScaler overflow skip remains allowed and logged as amp_overflow.
Save every25 epochs. RNG/worker state was not saved by the old checkpoint;
this is not a bitwise replay of the old trajectory.

Use a new snapshot from latest origin/feature/orgslot. Existing runtime remains
unchanged because encoder-only133/134494 shares it. New afterok evaluation uses
arm_e_mae_unified.sbatch: train calibration, test head, reconstruction.
CPU Gloo regression checks reverse dict insertion order and mismatched schemas;
GPU stability remains unproven until the resumed job crosses the old failure.
