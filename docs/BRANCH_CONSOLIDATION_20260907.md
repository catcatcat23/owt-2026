# Branch consolidation: 2026-09-07

Recommended development branches: main, feature/orgslot, experiment/psem, experiment/lossbalance.
feature/orgslot contains all OrganSlot histories/configurations. Development
worktree: .worktrees/orgslot_integrated. Historical experiment worktrees remain at
their original commits in detached HEAD state, preserving logs, checkpoints and uncommitted changes.
Existing Slurm tasks continue to use their original directories.

Consolidation and eight archive tags have been pushed to GitHub. After explicit
user authorization, all eight old local and six corresponding remote experiment
branch refs were deleted. Remote deletion used atomic push with exact expected
SHAs; local deletion used merged-branch checks. Archive tags remain on GitHub.
The old orgslot_head_ab worktree is detached at 1b104f4; its files are preserved.

Archive tags use prefix archive/20260907/ and the suffix below:

| Former experiment branch / tag suffix | Original commit |
|---|---|
| orgslot-3d-slicewise | 8984c9e |
| orgslot-armb-3d | 6bee802 |
| orgslot-autopet-mae-transfer | b43c7f0 |
| orgslot-pixel-pe | 4e7bf7c |
| orgslot-querymask | 4d7c5e0 |
| orgslot-querymask-3d | ab3bf1f |
| orgslot-querymask-multiquery | 4d7c5e0 |
| orgslot-querymask-multiscale-pixel | 81adba2 |

Recover with `git switch -c <new-name> <archive-tag>` in an unused checkout.
No experiment result or checkpoint is deleted.

Unified controls: dimension, slot_head_type, pixel_pe, segmentation_unit,
mae_init_checkpoint and mae_init_scope. segmentation_unit defaults to slab
for legacy behavior; slice-wise and 3D PE scripts explicitly select slice.
2D loss is unchanged. Arm E remains 2D-only.

Validation: 51 model/PE/Arm E/loss tests, 6 MAE tests, 3 slice-wise checks,
shell syntax and git diff checks passed. GPU experiment validation is separate.
