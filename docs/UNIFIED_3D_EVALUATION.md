# 3D inference, shared 2D evaluation protocol

Entry: `tools/eval_common8_orgslot_unified_3d.py`; launcher:
`slurm/orgslot/eval/unified_3d_protocol.sbatch`.

The model still receives four-slice slabs. Overlapping scores are averaged at
each original slice **before** thresholding. Each original slice is scored once.
No retraining or architecture change is involved.

| Component | Shared with current 2D evaluation |
| --- | --- |
| Input | WORD 0.7×0.7×2.0, fixed-255 normalization, deterministic 448 crop |
| Calibration | First 6000 training manifest rows, disjoint test cases |
| Threshold selection | Per-class 0.1–0.9 sweep; maximize postprocessed case Dice on calibration only |
| Head test | Fixed 0.5 and training-calibrated thresholds, raw and post |
| Direct recon | Keep only selected slot; original fusion and reconstruction decoder; channel mean >0.02 |
| Indirect recon | Keep all except selected slot; clamp(input−reconstruction,0) before channel mean >0.02 |
| Postprocessing | Shared 2D function: remove components <20 pixels, radius-1 opening per slice |
| Aggregation | Shared 2D counters and summarize; per-case volume Dice and foreground macro mean |

Calibration reads complete slabs from the relevant training cases for context,
but only the selected first 6000 original slices contribute to calibration.
Threshold files bind the checkpoint and test-manifest SHA256. Test sweeps are
not performed. Reconstruction uses its fixed historical threshold, not head thresholds.

Submit calibration and reconstruction independently; head must depend on successful
calibration. Use immutable source and the repository revision gate before submission.
Outputs include resolved configuration, exact checkpoint-load report, scoring keys,
per-case records, metrics and completion status.

**Do not compare old 3D-post results as if they used this protocol:** the previous
3D evaluator removes 3D connected components and does not use the 2D opening.
Slab context/fusion remain intrinsically 3D; matching evaluation protocol does not
mean making 3D inference identical to single-slice inference.
