# Arm D pixel PE controlled ablation

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
