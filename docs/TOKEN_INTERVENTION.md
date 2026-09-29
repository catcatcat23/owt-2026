# Frozen Arm E + MAE token intervention

Inference only, no optimizer or weight update. Existing 96 training cases supply
tokens after TGEnc, before the unchanged query LayerNorm/projection. Full training
manifest is traversed once. Cache per-organ positive/empty sums and reservoir128
tokens per stratum; record donor case/index. Means are slice-weighted (not
case-weighted); reservoir sampling can underrepresent rare donor cases.

Evaluate all24 test cases/6990 slices: original, all-training mean, presence-
matched mean, three independently sampled same-organ presence-matched donors.
Donors come only from train; select cached case uniformly then cached slice.
GT-presence matching is diagnostic oracle information, NOT deployable inference.
No volume or position matching; interpret replacements as interventions, not a
proof of absence of information. Mean raw tokens may be distribution-shifted;
retain the original LayerNorm and report mean versus donor controls separately.

Compute original spatial features once per batch, reuse exactly for each organ
readout; preserve slot identity and calibration. Assert original token re-read
reproduces original logits. Independent organ readouts are intervened separately;
there is no cross-class competition or joint argmax in this audit.

FP32, batch2, fixed0.5 raw/post(min_size20,opening1), center448, no augmentation,
no threshold tuning. Do not directly compare to historical calibrated Dice.
Full test set is exploratory, already examined previously; not a blind holdout.
No PCA/heatmap is necessary for the initial causal-use diagnostic.

Outputs: provenance/checkpoint+manifest hashes, token_bank.pt, donor assignments,
per-case Dice/P/R/volume/empty/zero-overlap/logitMAE, per-slice counts, progress,
eight-organ summary for each condition. Donor repeats are diagnostic variation,
not independent training seeds. No model architecture changes.

## Presence ablation (2026-09-30)

Use --reuse-bank PREVIOUS_OUTPUT --presence-ablation. Verify checkpoint and
train/test manifest hashes before reusing the bank. Compare original,
positive_only, negative_only, fixed_positive (no GT selection), mean_matched.
Only the selected presence stratum is replaced; the other retains original
tokens. Save maximum-GT-slice CT input, GT and probability maps as NPZ for
gallbladder 0041/0125/0127/0145 and pancreas 0114. No retraining or threshold fit.
Original and mean_matched must reproduce the first run before interpretation.

Initial postprocessed audit: matched means reduce positive-voxel recall for
gallbladder 74.15 to 62.53%, esophagus 82.06 to 78.34%, pancreas 83.71 to
79.98%, while reducing negative-slice FP counts. These are pooled voxel recall,
not case-mean recall. Gallbladder 0127 remains zero overlap. This does not
support claiming the higher overall Dice rescued missed organs.

Interpretation: true tokens outperforming donors suggests useful case-conditioned
readout. Little change suggests limited use by this readout, not necessarily no
information in tokens. Fixed identity conditioning may still support modularity.
