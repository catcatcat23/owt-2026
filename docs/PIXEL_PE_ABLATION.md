# Arm D pixel PE controlled ablation

## Restart after provenance failure — 2026-09-08

Supersedes the earlier job matrix. All jobs below: antengcai23/XEC,
account sifansong, QoS 8gpus; four A800/24 CPU/192 GB per training,
one A800/10 CPU/128 GB per evaluation. Training queues independently;
each evaluation has afterok on its own training. No smoke job submitted.

| Mode | Training | Evaluation |
|---|---|---|
| none, slice-wise baseline | 133848 | 133849 |
| spatial | 133850 | 133851 |
| temporal | 133852 | 133853 |
| spatiotemporal | 133854 | 133855 |

Frozen code: /gpfs/work/aac/antengcai23/worktrees/orgslot_3d_restart_01f1666.
Commit 01f1666; archive SHA256 1ae368a12e0495937a1c65d5f4729c8a414fddeab0241794fd0be9f75730893b.
Git metadata is optional with a 15-second timeout; actual source hashes
and dataset checksums remain recorded. Full real non-Git provenance writing,
three provenance tests, three slice-loss checks and target data identity passed.
Submission exports conda lib plus shared Mesa lib in LD_LIBRARY_PATH.
All eight new jobs verified pending (training Priority, evaluation Dependency).
Old 132071/132072/132097/132099/132101 cancelled; failed runs and files retained.

## Scheduling update — 2026-09-08 (supersedes original dependencies below)

Released baseline 132071 from user hold. Removed the baseline-evaluation
dependency from PE training 132096, 132098 and 132100: all four now queue
independently. Each evaluation retains afterok on its own training job.
Changed these eight antengcai23/XEC jobs to QoS 8gpus using scontrol:
132071/132072, 132096/132097, 132098/132099, 132100/132101.
Training still requests four A800 GPUs, not eight; no model/data/loss change.
The submitted script may still say 4gpus: scontrol's current job record is
authoritative for this scheduling override. New submissions must select QoS
explicitly after checking permissions, wall-time and live capacity.

At this audit SIP and XEC A800 nodes had no unallocated GPUs. bolinren19/SIP
retains Arm E and 2D PE; sifansong/XEC retains MAE encoder-only;
antengcai23/XEC retains MAE encoder-decoder plus these 3D jobs.
No cross-account data/code migration or duplicate training was submitted.
Eight-GPU per-user QoS permits up to two four-GPU jobs within that QoS,
subject to other limits and physical availability; it does not reserve GPUs.

2D: existing no-PE Arm D 82.12% (historical reported score; match its exact
threshold/postprocess protocol before comparing) versus spatial PE, same seed 0,
WORD 0.7/0.7/2, ROI20, original small-organ loss, batch192, 118800 updates.

3D: slice-wise-loss Arm D 132071 without PE versus spatial-only, temporal-only,
and spatial+temporal. All share slice-wise loss, seed0, batch48 slabs, 4 frames,
118800 updates, BF16, clipping1 and LR7.5e-5. Do NOT use old slab-loss 32.01%
as the isolated PE control. Three PE jobs depend on baseline completion.

CLI --pixel_pe defaults to none. Added after projection/reshape, before the
existing pixel decoder. Spatial PE [1,D,H,W] broadcasts over time, initialized
from OWT 2D sin/cos; temporal [1,D,T,1,1] broadcasts over H,W, normal std0.02.
Both learnable, as in OWT 3D decoder. This is local slab time, not absolute CT z.
OWT encoder already has PE; this tests reintroducing explicit position in the
pixel branch. It does not add a Transformer or change convolutions/collector.

Checkpoints store pixel_pe in args; 2D and 3D evaluators instantiate the matching
mode and strictly load state. Legacy checkpoints default to none. Temporal PE
initialization preserves the random stream used by existing parameters.

Compare fixed-threshold and original calibrated protocol separately. Report
eight organs, small-organ mean, precision/recall, volume ratio, and per-z counts.
Spatial-only vs none isolates space; temporal-only vs none isolates time;
both vs each single mode tests combination. An improvement does not by itself
prove PE absence was the sole cause of collapse.

## Submitted matrix (2026-09-06)

Code snapshot ab7742b. 23 unit/regression tests passed; full tiny 3D PE
forward/backward passed. Allocated-GPU execution remains pending.

| Arm | Account/cluster | Train | Eval | Dependency |
|---|---|---|---|---|
| 2D spatial | bolinren19/SIP | 2910849 | 2910855 | none |
| 3D none, slice loss | antengcai23/XEC | 132071 | 132072 | held for diagnostics review |
| 3D spatial | antengcai23/XEC | 132096 | 132097 | afterok:132072 |
| 3D temporal | antengcai23/XEC | 132098 | 132099 | afterok:132072 |
| 3D spatial+temporal | antengcai23/XEC | 132100 | 132101 | afterok:132072 |

All training requests are 4 A800 / 24 CPU / 192 GB, four-day limit.
No historical job was cancelled or modified. No new accuracy results yet.
