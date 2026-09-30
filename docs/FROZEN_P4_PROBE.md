# Frozen P4 probe diagnostic

## E badcase extension (2026-09-30)

Supports E fused P4 via a separate capture path; original F path unchanged.
E uses one linear probe (no reverse attention exists), compared with the SAME
checkpoint's original query head. Train organ4 and organ6 separately, fixed10
epochs, seed42, batch16, AdamW1e-3, existing BCE+positive Dice probe objective.
Original model fully frozen. Keep76/20 training-case split; evaluate selected
test cases0127/0114 ONLY after epoch9, no test selection/tuning. Test and training
IDs must be disjoint. Save all test slice probability maps and case/slice counts.
Fixed0.5 RAW results; do not compare against historical postprocessed/calibrated
numbers. These probes have extra supervised fitting and a different objective;
superiority supports readability, not proof that query is the sole bottleneck.
Script: slurm/orgslot/eval/e_badcase_probe.sbatch. Same E checkpoint as token
interventions; do NOT reuse existing F probe weights on E features.

Independent script; no changes to original model. Supports original 2D
arm_f_attention only. Each run selects one organ (6 pancreas, 5 esophagus,
4 gallbladder). Compare raw P4 immediately before reverse attention against
the organ-conditioned P4 after reverse residual and FFN, before classifier.
Hooks must fire exactly once. Both linear heads start with identical weights.
Backbone is eval/no_grad and receives no gradients. Original head predictions
are reported alongside probes, fixed0.5 raw without postprocessing.

Run from repository root:

```bash
PYTHONPATH=. python tools/frozen_p4_probe.py --checkpoint /absolute/checkpoint-802.pth --train-csv /absolute/WORD_Training_2D_native07072.csv --class-id 6 --output /absolute/new_probe_output --batch-size 16
```

Requires current project's torch environment. Defaults: single GPU, FP32,
10 epochs, AdamW lr1e-3, center448, no augmentation/ROI, BCE plus positive
Dice with equal positive/negative slice-group weighting. This is a diagnostic
probe objective, NOT the original small-organ training loss. Paired probes
receive the exact same batches and update count. Lower batch if 4090 OOM;
do not compare separate runs using different batches as controlled ablations.

Original96 training cases are deterministically split 76/20, seed42. Case IDs,
checkpoint and manifest hashes, load report, settings are saved. The backbone
previously saw ALL96; holdout is only unseen to the new probes, not independent
end-to-end validation. Never choose settings using the24 test cases. No early
stopping or best-checkpoint selection. Output includes per-case Dice/P/R/counts
and volume ratios for every epoch and final probe weights. Training metrics are
online across changing head weights, explicitly flagged; not a final fixed-head
training-set evaluation. Holdout metrics use fixed weights.

Interpretation: post worse than pre suggests reduced LINEAR readability, not
proof of information destruction. Original classifier includes LayerNorm,
unlike these linear probes; do not attribute all baseline differences to query.
Pre/post feature scales may affect optimization. This implementation has not
yet been GPU-preflighted or submitted. Full-checkpoint loading and peak memory
must be checked on actual4090 before claiming operational validation.
