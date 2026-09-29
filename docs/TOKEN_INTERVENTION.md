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

Interpretation: true tokens outperforming donors suggests useful case-conditioned
readout. Little change suggests limited use by this readout, not necessarily no
information in tokens. Fixed identity conditioning may still support modularity.
