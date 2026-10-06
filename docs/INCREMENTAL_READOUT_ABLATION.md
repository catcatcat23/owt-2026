# Stage2: shared vs stage-private query readout

Controlled 4+4 comparison, initialized from the same completed Stage1 (2998659,
checkpoint-401). Neither run resumes a previous Stage2. Official-pool seed42
96 training / 26 validation / 24 test cases, unchanged manifests and ROI.

| Policy | Old4 readout | New4 readout | P4 spatial path |
|---|---|---|---|
| `query_shared` | Shared, trainable | Same shared weights | Frozen |
| `query_split` | Original, frozen | One copied readout shared by new4, trainable | Frozen |

Readout comprises query LayerNorm, mean pooling, projection and cross-attention
(Q/K/V/output projections, LayerNorms, residual FFN). Final normalized query dot
original P4 and upsampling remain unchanged. Split adds 232,704 parameters at
768 encoder / 128 mask channels, not four copies or a second pixel decoder.
New slot identities remain private. Old slots, ViT, image stem, pixel projections,
spatial blocks and shared reconstruction decoder freeze. Background separation
policy remains unchanged: background collector/TGEnc/token_norm/AHER train at
0.1x LR, background head/calibration freeze. New slots train; calibration freezes.

Both: seed0, AdamW LR7.5e-5, weight decay .05, batch16/GPU x4 xaccum3=192,
59400 updates, 2970 warmup, spacing .7/.7/2, input448, ROI384/probability.2,
retained-slot small-organ segmentation loss lambda .01, background loss .1.
No new distillation, loss, sampling, PE or reconstruction changes.

Training flag: `--stage2_shared_segmentation query_shared|query_split`;
launcher environment: `STAGE2_SHARED_SEGMENTATION`. Old default `frozen` and
previous full spatial/readout `train` are preserved. Evaluators reconstruct the
new readout from checkpoint args before strict loading. Missing metadata/weights
fails strict loading rather than silently evaluating the old shared route.

Verification: CPU initialization equivalence, nonzero finite query gradients,
frozen parameter invariance, split old raw logits invariant after an optimizer
step, shared old logits can change, strict checkpoint roundtrip. In-job two-update
four-GPU preflight precedes a fresh Stage1 restart for the full training budget.
CPU checks do not imply GPU preflight success.

Evaluation: existing validation-calibrated and fixed0.5 head protocols, per-case
old4/new4/all8 Dice, precision/recall and volumes; reconstruction direct/indirect
raw/post at .02. Binary old-logit invariance does not guarantee invariance of an
exclusive eight-class merged mask (new predictions can compete with old classes).
No claim of superiority before final evaluation or matched-protocol comparison.
