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
118800 updates per stage, 5940 warmup (5%), spacing .7/.7/2, input448, ROI384/probability.2,
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

## SAM-tail: private slot interaction, shared frozen tail

New opt-in recipe `INCREMENTAL_ARCHITECTURE=sam_soft_aux`, Stage2 policy
`slot_private`. This does NOT replace the submitted E query_shared/query_split
runs, and is not initialized from the eight-class Offline 85.55 checkpoint.

1. Train a fresh five-slot Stage1 with MAE encoder initialization, shared
   SAM-tail, soft prior and aux; only old-four labels visible. Shared spatial
   features, shared refinement and tail learn during Stage1.
2. Load that completed Stage1 exactly and retain a frozen teacher. Append new
   slots using the existing background initialization. Copy the trained Stage1
   `input_proj` and three `TokenBlock`s into EACH slot (old and new). Each block
   includes cross-attention, existing self-attention, LayerNorms and FFN; do not
   change the internal block architecture while testing parameter isolation.
3. Freeze ViT, pixel decoder, old slots and private interactions, the entire
   shared SAM-tail (mask token, both attention directions, FFN, final MLP), and
   reconstruction decoder. Train new slots and their own private interactions.
   Background collector/TGEnc/token_norm/AHER still follow the existing 0.1x-LR
   separation policy; background interaction/head/calibration remain frozen.

The spatial decoder and tail are NOT copied. Each organ uses the same spatial
maps and the same frozen tail with different private refined tokens. The old
shared projection/blocks remain as frozen checkpoint-compatible templates but
are not used for slot head inference after migration. Background also has an
unused frozen copy for uniform checkpoint routing. Stage2 migration is tested
to preserve all head and coarse logits before any update.

At encoder width768 / interaction width128, each private interaction is
892,928 parameters (three complete TokenBlocks, not E's single-query block).
New4 adds3,571,712 trainable interaction parameters. Migration materializes nine
copies including background:8,036,352 stored parameters, of which old/background
copies freeze; the legacy shared template also remains frozen. Do not confuse
these numbers with E's232,704-parameter query-only readout.

Soft prior is independently calculated per organ and per scale, not a learned
shared mask and not a new network. Preserve the Offline formula:

```
x = input_proj_c(tokens_c) + slot_identity_c
for feature, block_c in [(P16, block16_c), (P8, block8_c), (P4, block4_c)]:
    coarse = sqrt(C) * cosine(mean(x), feature)
    attention_bias = log(0.2 + 0.8 * sigmoid(coarse.detach()))
    x = block_c(x, feature, PE, attention_bias)
mask = frozen_shared_SAM_tail(x, original_P4, PE)
```

The prior formula is identical across slots; its values come from each slot's
own tokens and interaction weights. Do NOT pass this prior to the tail or add
M0-M3 hard routing. Three coarse logits are upsampled to full-resolution GT;
each uses the existing small-organ loss. Stage2 supervises ONLY new labels:

`L = Lrecon + .01 * (Lfinal_new + .25 * mean(L16_new,L8_new,L4_new)) + .1 * Lbgsep`.

Frozen tail must remain differentiable with respect to tokens (no no_grad or
detach around its forward). Only the prior routing detaches. Tests separately
check final-mask gradients reach private projection through the frozen tail,
in addition to auxiliary-loss gradients. Metrics keep a fixed DDP schema even
if one rank retains no new foreground slots.

### Running the recipe

Reuse the existing pinned Stage1/Stage2 Slurm scripts and account-specific data
paths. Before submission commit/push, create immutable runtime, and pass the
revision guard. Export `INCREMENTAL_ARCHITECTURE=sam_soft_aux` for BOTH stages;
for Stage2 export `STAGE2_SHARED_SEGMENTATION=slot_private` and point STAGE1_RUN
to THIS recipe's completed Stage1, never to the E Stage1 or an Offline run.
The launcher defaults remain E cross-attention unless explicitly selected.
Keep 16/GPU x4 xaccum3=192,118800 updates/stage,ROI20 and existing split/seed.

## Full per-stage budget revision (2026-10-06)

Both stages now start fresh with 118800 updates and 5940 warmup updates;
the existing update-based learning-rate schedule spans the new full budget.
Peak LR, model, split, losses and sampling remain unchanged. Total training is
237600 updates, twice the Offline budget; this is NOT an equal-total-compute comparison.
Stage2 requires a completed 118800-update Stage1, including its two-update preflight.
Historical 59400-update checkpoints remain evaluable, but cannot initialize new Stage2 runs.
Replace queued/running chains with new immutable snapshots; retain all old artifacts.

Evaluation loaders create private slot modules BEFORE strict checkpoint load.
Use the existing Stage1 old-four and Stage2 all-eight head evaluation scripts;
enable `EVALUATE_RECONSTRUCTION=1` for Stage2. Compare new4 learning, old4 binary
retention, final merged-mask competition, and reconstruction separately.
This run tests whether private interaction can work through a frozen tail;
failure alone would not prove the tail is the bottleneck (frozen P4 and slot
representation also remain possible constraints).

Verification (2026-10-06):30 CPU tests passed across incremental44, private SAM
interaction, soft prior, H-SAM supervision, ArmF and all-seg regression suites.
Includes final-mask gradients through frozen tail, old-logit invariance, strict
evaluator reload, fixed empty-rank metric schema and tiny-batch loss descent
(not a full overfit or GPU-validation claim). CLI and shell syntax checks pass.
GPU preflight and full training are NOT yet submitted for this new recipe.
