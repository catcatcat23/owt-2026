# 3D controlled diagnosis, 2026-09-06

Parent: ab3bf1f (Arm D, 128095). Existing C/D jobs are unchanged.

Confirmed: direct-head case Dice is B 24.32%, D 32.01%; full test coverage,
exact checkpoint-199 load. Strong false positives are measured. These figures
do not evaluate reconstruction. The original small-organ objective classifies
an entire [T,H,W] slab as positive or empty.

Not confirmed: whether z-smearing is the dominant error; whether overlap fusion,
missing temporal identity, temporal convolutions, or representation cause it.

Diagnostics: same checkpoint and deterministic crop; raw logits, checkpoint
calibrated logits (not test-fitted thresholds), and reconstruction via the
existing per-slot canvas -> fusion -> spatial/temporal PE -> decoder. Fixed
thresholds: head 0.5, reconstruction 0.02 (legacy). Report mean probability,
mean logit and center-only fusion, before/after identical 3D postprocessing.
per_slice.csv includes GT and predicted counts for every organ and z position.
These are cropped-grid case metrics, not native full-FOV claims.

Loss-only baseline: Arm D from the original random seed 0, no checkpoint resume.
Only 5D small-organ inputs are flattened B,T into supervision samples. Average
positive planes and empty planes separately, then add; empty weight remains 0.1.
2D and other loss types unchanged. Positive/negative diagnostic counts now count
slices for this loss. All optimizer, data, ROI20, update budget, BF16, head and
architecture settings match the parent. 118800 updates, LR 7.5e-5, batch 48 slabs.

Tests: mixed slab 1 positive + 3 empty, exact separate means, finite backward,
equivalence to the old 2D formula, and excluded-slab zero gradient.

No temporal PE, convolution, collector, or reconstruction architecture change.
Implementation snapshot: 678f97e. Local tests: 3 new loss checks and 20 existing
loss/TGR checks passed. Tiny 3D reconstruction via collected canvases is equal
to the existing selected-slot reconstruction forward, with finite output.

Submitted jobs (XEC): B head pilot 131959 -> full head/reconstruction 132069;
D head pilot 131960 -> full head/reconstruction 132070. Pilot cases=2, full=24.
Arm D loss-only training 132071 is USER HOLD: release only after diagnostics
are reviewed, preserving the requested no-retrain-first order. It is registered,
not training. No claim of loss correction improving Dice has been established.
Its same-protocol formal head evaluation is 132072, afterok:132071.
At 2026-09-06 21:30 CST, both pilots await QOSMaxGRESPerUser (MAE occupies
each account GPU quota). Full diagnostics await pilot success. No new
checkpoint diagnostic result is available yet.

Only consider temporal PE, then 1x3x3 conv as separate later experiments after
the loss-only result and evaluator controls are reviewed.
