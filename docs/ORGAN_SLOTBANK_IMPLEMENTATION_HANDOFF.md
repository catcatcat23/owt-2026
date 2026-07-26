# E-OWT-Seg / OrganSlotBank Implementation Handoff

Updated: 2026-07-24

This document is an implementation specification for a new coding session. It
combines the authoritative planning files with a code-aware audit of the
current OWT repository. It does not mean that OrganSlotBank has already been
implemented.

## 1. Status And Source Of Truth

Current repository:

```text
/gpfs/work/aac/bolinren19/OD_OWT
```

Current branch at handoff time:

```text
experiment/psem-v1
```

Important:

- PSEM and LossBalance are auxiliary OWT pretraining/masking experiments.
- They are not the E-OWT-Seg main method.
- Do not implement OrganSlotBank on top of the PSEM branch by default.
- Create a separate worktree and branch from the clean `main` baseline so
  pending/running PSEM jobs and their source tree are not changed.

Recommended setup:

```bash
git -C /gpfs/work/aac/bolinren19/OD_OWT worktree add \
  -b feature/orgslotbank-v0 \
  /gpfs/work/aac/bolinren19/OD_OWT_orgslot \
  main
```

Before running this command, verify that `main` is the intended clean OWT
baseline and that it contains any required environment compatibility fixes.

Read the plan in this order:

1. `00_README.md`
2. `CONTEXT.md`
3. `07_codeaware_grill_log.md`
4. `08_project3_implementation_handoff.md`
5. `10_integrated_final_plan.md`
6. `11_project3_experiment_protocol_handoff.md`

Plan directory:

```text
/gpfs/work/aac/bolinren19/2026-07/plan/2026-06-29_expandable_owt_incremental_seg
```

Authority rule:

- `10_integrated_final_plan.md` defines the final method and architecture.
- `11_project3_experiment_protocol_handoff.md` defines the newest experiment
  protocol, debug split, initialization comparisons, and execution order.
- `08_project3_implementation_handoff.md` defines code boundaries.
- `03`, `04`, and `05` are historical drafts and must not override `10` or
  `11`.
- `CONTEXT.md` contains useful terminology, but any stale side-path wording is
  superseded by the Minimal OrganSlotBank Refactor in `10`.

## 2. Research Question And Claim Boundary

Task:

```text
partial-label organ-incremental medical image segmentation
```

It is not online learning. At each stage, images may come from the same
dataset, but only a subset of labels is visible.

Base stage:

```text
visible labels = old organs only
future organs = hidden and included in provisional background
```

Incremental stage:

```text
visible labels = current new organ only
old organ GT = forbidden for training
future organ GT = forbidden for training
old organ GT = evaluation/reporting only
```

Main claim to test:

> Organ-wise token/restoration paths can be appended for new organs while old
> paths remain frozen, reducing old-organ drift under new-organ-only labels.

Do not claim:

- generic online learning;
- future-organ discovery;
- complete solution to class-incremental segmentation;
- complete solution to background shift;
- direct implementation-equivalent superiority over PCDD unless its official
  code, split, and runtime protocol are actually reproduced.

## 3. Current OWT Code Audit

The original path must remain runnable as a reference baseline.

### Current model

Relevant files:

```text
OWT_models.py
OrganEmbed.py
engine_pretrain.py
main_pretrain.py
datasets/dataset3D.py
```

Current architecture:

```text
image
-> ViT encoder blocks1
-> patch features [B, N, 768]
-> one joint OrganCollector
-> all class token groups [B, S*K, 768]
-> one shared blocks2 TGEnc
-> one shared AHER
-> one spatial canvas [B, N, 768]
-> shared decoder
-> reconstructed image
```

For the current base model:

```text
input size = 224
patch size = 16
K = token_factor = 20
2D N = 14*14 = 196
3D Fixfr4-TS1 N = 4*14*14 = 784
```

Current joint components:

- `OWT_models.py:190`: shared `blocks2`;
- `OWT_models.py:201`: one `organ_embed`;
- `OWT_models.py:212`: one `decoder_embed` AHER;
- `OWT_models.py:428`: all organ tokens are collected together;
- `OWT_models.py:438`: all retained tokens enter shared TGEnc;
- `OWT_models.py:453`: all retained tokens enter one AHER.

Current training limitations:

- `engine_pretrain.py` samples one dropped-class list for the entire batch.
- `random_selected_class` means dropped classes, despite its ambiguous name.
- Dropped class regions are zeroed in the reconstruction target.
- There is no supervised segmentation head or validation loop.
- Current segmentation scripts infer masks from reconstruction differences and
  hand-selected thresholds. This remains an auxiliary diagnostic only.
- `main_pretrain.py` silently loads checkpoints with `strict=False`.
- Random seeds are currently commented out.
- Train/validation/test manifests are not first-class stage configurations.

Current data details:

- CSV columns are `image_pth` and `mask_pth`.
- 2D images and labels are loaded as three-channel arrays.
- 3D samples are four adjacent numbered slices for `Fixfr4`.
- Runtime `RandomGenerator` may resize, but for the OWT-native v0 path the CSV
  should already point to offline-preprocessed 224x224 slices.
- The data loader currently returns the complete label map to the training
  engine. That is too easy to misuse for strict incremental experiments.

## 4. Target Architecture

### 4.1 Shared encoder

Input:

```text
2D: x [B, 3, 224, 224]
3D: x [B, 3, 4, 224, 224]
```

Output:

```text
z [B, N, C]
C = 768
N = 196 for 2D
N = 784 for Fixfr4-TS1
```

The encoder remains shared. It is trainable during base fitting and frozen
during the main incremental method.

### 4.2 One explicit slot per class

Slot bank:

```text
background
old organ 1
old organ 2
...
new organ slots appended at later stages
```

Each slot owns:

```text
OrganCollector_s
OrganWiseTGEnc_s
AHER_s
BinaryHead_s
calibration scale a_s
calibration bias b_s
```

Per-slot data flow:

```text
z [B,N,C]
-> OrganCollector_s
tokens_s [B,K,C], collector_attention [B,K,N]
-> OrganWiseTGEnc_s
encoded_tokens_s [B,K,C]
-> AHER_s
canvas_s [B,N,D], aher_attention [B,N,K]
-> BinaryHead_s
patch/pixel logit_s
```

Defaults inherited from OWT:

```text
C = 768
D = 768
K = 20
```

### 4.3 Normalized additive canvas

For a per-sample slot keep mask `keep [B,S]`:

```text
weighted_s = canvas_s * keep[:,s,None,None]
canvas_sum = sum(weighted_s)
count = keep.sum(dim=1).clamp_min(1)
canvas_all = canvas_sum / sqrt(count[:,None,None])
canvas_all = LayerNorm(canvas_all)
```

Then:

```text
canvas_all -> existing shared decoder -> reconstruction
```

The keep mask must be per sample. Do not take the union of slot choices across
the batch.

Memory rule:

- Do not stack every `[B,S,N,D]` canvas during normal training.
- Aggregate canvases in a loop and retain only logits/attention requested for
  losses or diagnostics.
- Return all canvases only under `return_diagnostics=True`.
- This matters especially for 3D where `N=784`.

### 4.4 Binary segmentation head

The plan locks the head location to the AHER canvas but does not lock its exact
layers.

Recommended minimal v0:

```text
LayerNorm(D)
-> Linear(D,1)
-> reshape patch logits
-> bilinear/trilinear interpolation to input resolution
```

Equivalent implementation:

```text
1x1 Conv2d/Conv3d after reshaping the canvas
```

Why this default:

- same head works for every slot;
- parameter cost is small;
- it isolates the value of OC/TGEnc/AHER;
- it works for both 2D and 3D;
- a larger convolutional head can be a later ablation.

Expected output:

```text
2D logit_s [B,1,H,W]
3D logit_s [B,1,T,H,W]
```

### 4.5 Organ-wise TGEnc depth

The plan locks organ-wise ownership but does not lock the exact slot depth.
Copying all six original `blocks2` layers for every class may grow parameters
too quickly.

Recommended v0 engineering default:

```text
slot_tg_depth = 1
```

Required ablation if the method becomes a paper result:

```text
depth 0 / 1 / 2
or shared frozen TGEnc + slot adapter
```

Whichever default is selected must be recorded with:

- total parameters;
- added parameters per new organ;
- GPU memory;
- inference time.

Do not silently describe a one-block implementation as a copied six-layer
TGEnc.

## 5. New File Layout

Keep the original files unchanged and runnable.

Recommended additions:

```text
OrganSlotEmbed.py
OWT_models_orgslot.py
losses_orgslot.py
engine_pretrain_orgslot.py
main_pretrain_orgslot.py
eval_orgslot.py
util/label_visibility.py
util/slot_tgr.py
util/checkpoint_orgslot.py
configs/orgslot/
tests/test_orgslot_model.py
tests/test_orgslot_visibility.py
tests/test_orgslot_losses.py
tests/test_orgslot_checkpoint.py
slurm/orgslot/
docs/ORGAN_SLOTBANK_EXPERIMENT_LOG.md
```

### `OrganSlotEmbed.py`

Implement:

```text
PatchBinaryHead
OrganSlot
OrganSlotBank
```

`OrganSlot` should contain `collector`, `tg_encoder`, `aher`, `head`, and
calibration parameters.

Use `nn.ModuleDict` with stable sanitized slot names. Store the semantic
name/raw class ID separately in checkpoint metadata.

Required methods:

```python
forward_tokens(z)
forward_canvas(z)
forward(z, output_size, return_attention=False)
set_trainable(enabled)
copy_from(other_slot, reset_head_bias=False)
```

`add_slot()` must be called before wrapping the model in DDP and before
constructing the optimizer.

### `OWT_models_orgslot.py`

Implement a pure model forward API rather than calculating all losses inside
the model.

Recommended output:

```python
{
    "reconstruction": reconstruction,
    "slot_logits": {name: logits},
    "slot_tokens": optional_dict,
    "slot_canvases": optional_dict,
    "collector_attention": optional_dict,
    "aher_attention": optional_dict,
    "slot_keep_mask": keep_mask,
}
```

Recommended methods:

```text
forward_encoder(x)
forward_slots(z, slot_keep_mask, return_diagnostics)
fuse_canvases(...)
forward_decoder(canvas)
forward(...)
append_slot(name, init_from)
freeze_for_incremental(old_slots, new_slots, background_policy)
parameter_report()
```

Separate encoder normalization and slot normalization. The current OWT model
reuses `self.norm` after both `blocks1` and `blocks2`; an explicit new model
should not accidentally couple those semantics.

### `losses_orgslot.py`

Implement independently testable functions:

```text
soft_dice_loss
dice_bce_loss
masked_mean
base_reconstruction_loss
base_segmentation_loss
new_region_reconstruction_loss
old_confidence_suppression_loss
background_new_complementarity_loss
```

Every masked loss must divide by the number of valid pixels/samples plus an
epsilon, not by the full image size.

### `util/label_visibility.py`

Implement:

```text
ClassSpec
StageSpec
build_visible_binary_masks
build_provisional_background
validate_raw_label_ids
```

Use a config file to map raw dataset IDs to semantic names. Do not assume the
OWT legacy raw IDs or BTCV IDs without checking the preprocessed masks.

Training batches should return only stage-visible masks:

```python
{
    "image": ...,
    "visible_masks": {slot_name: binary_mask},
    "case_id": ...,
    "slice_index": ...,
    "sample_index": ...,
}
```

The full label map may be returned only by an evaluation dataset/loader.
This makes old/future GT leakage mechanically harder.

### `util/slot_tgr.py`

Implement explicit names:

```text
sample_base_slot_keep_mask
sample_retain_new_keep_mask
build_base_reconstruction_target
build_incremental_pseudo_target
```

Never reuse the ambiguous name `random_selected_class`.

### `util/checkpoint_orgslot.py`

Implement:

```text
load_original_owt_initialization
load_orgslot_base_checkpoint
append_and_initialize_new_slots
visible_slot_transfer
hash_frozen_parameters
compare_parameter_hashes
```

All load functions must print and save:

- loaded keys;
- missing keys;
- unexpected keys;
- shape mismatches;
- slots loaded;
- slots intentionally skipped.

Do not use unexplained `strict=False`.

### `engine_pretrain_orgslot.py`

Support explicit stages:

```text
base
incremental_min
incremental_suppress
incremental_formal
offline
sequential_ft
head_only
```

Log each loss component separately. The displayed total must be the exact loss
used for backpropagation.

### `main_pretrain_orgslot.py`

Add configuration for:

```text
stage
slot names and raw IDs
visible old slots
new slots
initialization source
base checkpoint
slot_tg_depth
lambda values
background policy
old confidence threshold
evaluation interval
seed
```

Restore deterministic seed setup for Python, NumPy, PyTorch, distributed
samplers, and DataLoader workers. Save the fully resolved configuration to the
run directory.

### `eval_orgslot.py`

This is the main segmentation evaluator. Do not use reconstruction threshold
as the primary result.

It must support:

- binary per-slot probabilities;
- calibrated multiclass fusion;
- slice-to-case aggregation;
- 2D and 3D input paths;
- per-class and aggregate metrics;
- raw old-path and final fused outputs;
- visual overlays and slot attention diagnostics.

## 6. Label Visibility Implementation

### 6.1 Base stage

For visible old organ masks `M_1...M_n`:

```text
M_bg = NOT(M_1 OR ... OR M_n)
```

Future organs are not removed from `M_bg`, because doing that would use future
labels and invalidate the protocol.

Example OWT legacy debug:

```text
visible old:
  Kidney-combined
  Spleen
  Pancreas

hidden future:
  Liver

base background:
  all pixels outside the three visible old masks, including Liver
```

### 6.2 Incremental stage

The training loader exposes only:

```text
M_new
```

It must not expose:

```text
M_old
full multiclass label map
future masks
```

The evaluator may load complete GT separately after training.

### 6.3 Dataset split rules

- Split at patient/case level before expanding into 2D slices or 3D windows.
- Never let slices/windows from one case appear in more than one split.
- Keep the final test split untouched.
- Use validation for hyperparameters and checkpoint choice.
- Save case lists, CSV checksums, preprocessing version, and class map.
- For OWT debug, preserve the existing official train/test division where
  available and create validation from training cases only.
- For BTCV/WORD, recover and document the PCDD split where feasible. If it
  cannot be recovered, use a fixed case-level split and report that the result
  is protocol-inspired rather than split-identical.

## 7. Slot-Aware TGR

### 7.1 Base sampler

For `S` base slots including provisional background, produce:

```text
keep [B,S] boolean
```

Recommended random v0:

1. For each sample independently, sample a retained count `k` uniformly from
   `1...S`.
2. Generate a deterministic sample/epoch permutation of slots.
3. Keep the first `k`.
4. Ensure at least one slot is retained.

This gives different samples different retained combinations without variable
token lengths, because every slot always has exactly `K` internal tokens.

For each slot:

```text
compute canvas_s [B,N,D]
multiply by keep[:,s,None,None]
```

No token padding is required across samples.

Base target:

```text
target = input.clone()
for every dropped visible old slot:
    zero its GT region in target
if provisional background is dropped:
    zero provisional-background region in target
```

The zeroing is per sample.

Segmentation loss:

- retained slot: supervise its binary mask;
- dropped slot: ignore it;
- do not relabel a dropped organ as background.

### 7.2 Incremental minimal run

For the first mechanism test:

- new slot is always retained;
- keep all frozen old/background slots;
- compute reconstruction loss only in the new-organ GT region;
- background remains fully frozen;
- no suppress loss.

This isolates whether the appended slot can learn.

### 7.3 Retain-New formal sampler

For each sample:

```text
existing = [background, old_1, ..., old_n]
shuffle(existing)
k ~ Uniform(1, len(existing)+1)
retained = [new] + existing[:k-1]
```

New is always retained.

When old/background slots are dropped, reconstruction target masks must come
from detached frozen predictions:

```text
M_old_high = sigmoid(old_logit_frozen) > tau_old
```

Default:

```text
tau_old = 0.7
```

Always remove the current new GT from old/background pseudo masks before using
them:

```text
M_old_high = M_old_high AND NOT(M_new)
```

Old GT must never be used for incremental target construction.

## 8. Losses

### 8.1 Base loss

```text
L_base = L_rec_base + lambda_seg * L_seg_base
```

Reconstruction:

```text
L_rec_base =
    mean((reconstruction - TGR_target)^2)
    + lambda_lpips * LPIPS(reconstruction, TGR_target)
```

For 3D, preserve the existing slice-wise LPIPS behavior unless a validated 3D
perceptual loss is introduced.

Segmentation:

```text
L_seg_base =
    mean retained old-slot DiceBCE
    + lambda_bg_seg * retained background DiceBCE
```

Binary loss:

```text
DiceBCE(logit, mask) =
    soft_dice_loss(sigmoid(logit), mask)
    + BCEWithLogits(logit, mask)
```

Use `lambda_bg_seg < 1` because provisional background contains future organs.

### 8.2 Incremental minimal loss

```text
L_inc_min =
    L_seg_new
    + lambda_rec * L_rec_new_region
```

```text
L_seg_new = DiceBCE(logit_new, M_new)
```

```text
L_rec_new_region =
    sum((reconstruction - input)^2 * M_new)
    / (sum(M_new) * channels + eps)
```

The new-region loss checks that the new slot produces a canvas compatible with
the frozen decoder. It must not become a full-image new-organ decoder.

### 8.3 Suppress ablation

```text
L_inc_sup =
    L_inc_min
    + lambda_sup * L_sup_old_conf
```

Build:

```text
M_old_union = union of frozen old high-confidence masks
M_sup = M_old_union AND NOT(M_new)
```

Then:

```text
L_sup_old_conf =
    masked BCEWithLogits(logit_new, target=0, mask=M_sup)
```

This reduces new-on-old false positives without using old GT.

### 8.4 Background release

Later formal run:

```text
L_bg_new =
    BCEWithLogits(logit_bg[M_new], 0)
```

Only local complementarity on the new GT region is used in P0. Do not add
full-image background/new complementarity.

### 8.5 Formal P0 loss

```text
L_inc =
    L_rec_inc
    + lambda_seg * L_seg_new
    + lambda_bg_new * L_bg_new
    + lambda_sup * L_sup_old_conf
```

P1/P2 losses such as old fused KL, background teacher stability, token
orthogonality, contrastive purity, or register regularization must not be added
before P0 is validated.

### 8.6 Provisional starting values

The plan does not lock most numerical weights. Use the following only as
debugging defaults, then tune on validation:

```text
lambda_seg = 1.0
lambda_lpips = 1.0, matching the current OWT addition
lambda_bg_seg = 0.25
lambda_rec = 0.1
lambda_sup = 0.0 for minimal run, then 1.0 for suppress ablation
lambda_bg_new = 1.0
tau_old = 0.7
```

Required small validation grids:

```text
lambda_bg_seg: 0.1, 0.25, 0.5
lambda_rec: 0.01, 0.1, 1.0
lambda_sup: 0.1, 0.5, 1.0
tau_old: 0.6, 0.7, 0.8
```

Do not tune on the test set.

## 9. Freeze And Optimizer Policies

### 9.1 Base stage

Train:

- shared encoder;
- visible old slots;
- provisional background slot;
- shared decoder;
- segmentation heads;
- identity calibration parameters if included.

If initialized from an original OWT checkpoint, the code-aware plan suggests:

```text
inherited OWT path: lr_scale = 0.1
new slot heads/modules: lr_scale = 1.0
```

The current scheduler supports per-group `lr_scale`.

### 9.2 Incremental minimal Ours

Freeze:

- shared encoder;
- all old collectors;
- all old TGEnc modules;
- all old AHER modules;
- all old heads;
- full background path;
- shared decoder.

Train:

- new collector;
- new TGEnc;
- new AHER;
- new binary head.

### 9.3 Incremental formal background policy

Low-LR trainable:

- background collector;
- background AHER;
- background head.

Frozen:

- background TGEnc.

Recommended LR scales:

```text
new slot = 1.0
background plastic modules = 0.1
calibration = 0.1 or 1.0, but report it
```

Fallback only:

- open background TGEnc at low LR;
- tiny decoder adapter;
- low-LR shared decoder tuning.

### 9.4 Freeze verification

Before and after incremental training:

1. Save SHA256 hashes of every frozen parameter tensor.
2. Assert frozen parameters have `requires_grad=False`.
3. After backward, assert frozen gradients are `None`.
4. After optimizer step, compare hashes.
5. Run a fixed probe batch through raw old paths before and after training and
   assert logits/canvases are numerically unchanged in `eval()` mode.

Do not rely only on optimizer parameter-group printouts.

## 10. New Slot Initialization

Default:

```text
new.collector <- deepcopy(background.collector)
new.tg_encoder <- deepcopy(background.tg_encoder)
new.aher <- deepcopy(background.aher)
new.head <- deepcopy(background.head)
```

Reset/keep head bias consistently and record the choice.

Required ablation:

```text
random new slot initialization
```

The rationale is that future organs were included in provisional background
during base training, so new-slot learning is a background release/refinement
process.

## 11. Original OWT Checkpoint Migration

An original joint OWT checkpoint is an initialization source, not the final
old-knowledge checkpoint. After migration, a new OrganSlotBank base stage must
still be trained under protocol-visible labels.

Recommended mapping:

### Shared encoder

Copy compatible:

```text
patch_embed
cls_token
positional embeddings
blocks1
encoder normalization
```

### Shared decoder

Copy compatible:

```text
decoder positional embeddings
decoder_blocks
decoder_norm
decoder_pred
```

### Joint OrganCollector to slots

The current joint collector emits `S*K` channels.

For slot index `s`:

```text
slot.conv1 <- joint.conv1
slot.conv3 <- joint.conv3
slot.conv2.weight <- joint.conv2.weight[s*K:(s+1)*K]
```

This preserves the class-token output-channel grouping at initialization.

### Shared TGEnc to organ-wise TGEnc

Copy the selected first `slot_tg_depth` blocks from original `blocks2` into
each visible slot and background slot.

If using adapters instead, document the exact mapping and initialization.

### Shared AHER to per-slot AHER

The current AHER linears are token-shared, so copy the complete compatible
AHER state into every visible slot/background AHER.

### Segmentation heads

Initialize new unless loading a previous OrganSlotBank checkpoint.

### Migration tests

- Verify collector output slices match the corresponding joint collector
  outputs before TGEnc.
- Verify all loaded tensor shapes.
- Save a migration report JSON.
- Never silently load future slot weights into a strict Base4/Base7 path.

## 12. Calibration And Inference

Per-slot calibration:

```text
calibrated_logit_s = a_s * logit_s + b_s
```

Initialize:

```text
a_s = 1
b_s = 0
```

Final prediction:

```text
stack [background, old organs, new organs]
argmax over calibrated logits
```

Important unresolved detail:

The plan says old/new/background calibration parameters may be trained during
incremental learning, but old calibration has no valid supervised gradient if
only new GT is used. Therefore:

- minimal debug: keep all calibration identity/frozen;
- first calibrated run: train new/background calibration with new GT;
- train old calibration only if a pseudo-label consistency objective using
  frozen old predictions is explicitly implemented;
- never tune old calibration with old GT during incremental training.

Report:

```text
raw old binary-head DSC
final calibrated fused DSC
```

Raw old path proves representation/output invariance. Final fused DSC measures
the usable multiclass result.

## 13. Validation And Metrics

### 13.1 Primary segmentation metrics

Report:

- per-organ DSC;
- `Old DSC` mean;
- `New DSC` mean;
- `All DSC` mean;
- old DSC before incremental;
- old DSC after incremental;
- forgetting;
- new-on-old false positives.

Forgetting:

```text
F = Old_DSC_before - Old_DSC_after
```

Also report per-old-organ forgetting.

New-on-old FP:

```text
sum(pred_new AND GT_old_union) / (sum(GT_old_union) + eps)
```

Old GT is allowed here because this is evaluation, not training.

### 13.2 OWT diagnostics

Report:

- new-region reconstruction MSE/PSNR;
- full reconstruction MSE/LPIPS where relevant;
- collector attention maps;
- SDTG/token maps;
- AHER maps;
- prediction overlays;
- added parameters per organ;
- peak GPU memory;
- inference time.

Reconstruction-threshold direct/indirect Dice remains auxiliary and cannot
replace head-based segmentation results.

### 13.3 2D case-level evaluation

1. Predict every slice.
2. Recover `case_id` and numeric `slice_index`.
3. Sort slices by numeric index.
4. Stack predictions into a case volume.
5. Compute case-level 3D DSC in the declared preprocessed space.
6. If preprocessing metadata supports inverse mapping, also report
   original-volume-space DSC.

Never call cropped/resized-space Dice original-volume Dice.

For empty GT classes:

- record whether GT and prediction are empty;
- do not silently assign Dice 1 and average it with present-organ cases;
- report presence-conditioned DSC and the chosen empty-case policy.

### 13.4 Checkpoint selection

Base:

- select by validation old-organ mean DSC;
- use reconstruction as a secondary diagnostic.

Incremental:

- select by visible new-organ validation DSC or visible validation loss;
- do not select by old test DSC;
- old validation GT may be reported for analysis only if the protocol allows
  it, but must not drive optimization or checkpoint selection.

Test is run once for the selected checkpoint.

## 14. Required Baselines

### Initial mechanism-debug table

1. Ours minimal.
2. Sequential full fine-tuning.
3. Head-only adapter.

Use the same:

- base checkpoint;
- data split;
- preprocessing;
- incremental labels;
- new-organ loss;
- iteration budget;
- head family where applicable.

### Ours

OrganSlotBank, append new slot, freeze old path and decoder.

### Sequential FT

Start from the same OrganSlotBank base checkpoint, append the same new slot,
then train the full model with new-organ-only labels:

- encoder;
- old slots;
- background;
- decoder;
- all heads.

This is the naive forgetting lower bound.

### Head-only adapter

Recommended precise definition:

- copy/fix the background-derived new slot path;
- freeze its collector, TGEnc, and AHER;
- train only the new binary head on that frozen canvas.

This isolates whether a small readout alone explains the result.

### Full controlled table

Add after the initial debug passes:

- Offline upper bound: all selected labels jointly visible.
- OWT-Seg joint baseline: original joint OC/shared TGEnc/shared AHER plus the
  same supervised head family and visibility protocol.
- OrganSlotBank naive/full FT.
- Ours freeze+append.
- Optional dense adapter from frozen encoder features.

Potential naming ambiguity:

`Sequential FT` and `OrganSlotBank naive` can become identical if both mean
full fine-tuning of the same OrganSlotBank checkpoint. Do not report duplicate
runs as separate methods. For the full table, distinguish:

```text
OWT-Seg Joint FT:
  joint original architecture, full fine-tuning

OrganSlotBank FT:
  organ-wise architecture, full fine-tuning

Ours:
  organ-wise architecture, strict freeze+append
```

## 15. PCDD Role And Formal Protocols

PCDD is:

- a benchmark/protocol reference;
- a reported-number reference;
- not the engineering code base;
- not a direct implementation dependency.

Match where feasible:

- class order;
- train/validation/test split;
- label visibility;
- Old/New/All DSC;
- forgetting.

Do not force-match unless necessary:

- optimizer;
- OWT preprocessing;
- loss recipe;
- backbone internals.

PCDD eight-organ order:

```text
1 spleen
2 right kidney
3 left kidney
4 gallbladder
5 esophagus
6 pancreas
7 liver
8 stomach
```

Formal order:

### BTCV 4-1

```text
Base: spleen, right kidney, left kidney, gallbladder
Step 1: esophagus
Step 2: pancreas
Step 3: liver
Step 4: stomach
```

At each incremental step, only the current new label is visible.

### BTCV 7-1

```text
Base: classes 1-7
Increment: stomach
```

### BTCV 4-4

```text
Base: classes 1-4
Increment: classes 5-8
```

Implement 4-4 only after single-new-organ append works. Define whether four
new slots are trained jointly in one incremental stage, matching the protocol.

### WORD

Run after BTCV is credible. Use the same eight-class semantic mapping and
strict visibility rules.

PCDD comparison table must label rows honestly:

```text
PCDD reported result
our reproduction, only if code/split reproduced
E-OWT-Seg under matched protocol
```

Do not call a protocol-level comparison an implementation reproduction.

## 16. Initialization Experiments

For BTCV/WORD Base4/Base7 compare:

1. Random initialization.
2. External MAE encoder initialization.
3. External all-8 OWT visible-slot transfer initialization.

### External MAE

Load:

- encoder by default;
- decoder only if structurally compatible.

Do not load:

- organ slots;
- segmentation heads.

### External all-8 OWT

External manual/pseudo masks are allowed only to produce the external
initialization checkpoint.

For strict Base4/Base7:

- load shared encoder/decoder;
- load only protocol-visible old slots;
- do not load future slots;
- fit Base4/Base7 again on BTCV/WORD with visible labels only.

The external dataset is not part of the strict BTCV/WORD training data.

Use the same fixed base and incremental budgets across initialization routes.
Longer best-route runs must be reported separately.

## 17. Experiment Execution Order

### Stage A: CPU/unit tests

Pass all model, visibility, loss, and checkpoint tests before Slurm.

### Stage B: one-batch forward/backward

Check:

- 2D and 3D shapes;
- finite losses;
- expected trainable gradients;
- no frozen gradients;
- TGR keep masks;
- checkpoint save/load.

### Stage C: tiny overfit

Use 2-8 cases/slices with augmentation disabled.

Success:

- visible segmentation loss drops strongly;
- new-organ Dice approaches overfit behavior;
- reconstruction loss decreases;
- no NaN/Inf.

If tiny data cannot overfit, do not submit full training.

### Stage D: 2D OWT-legacy smoke

Split:

```text
Base old: Kidney-combined, Spleen, Pancreas
New: Liver
```

Run:

1. Base training smoke.
2. Incremental minimal Ours.
3. Evaluation.

Use one seed and a small subset.

### Stage E: early 3D Fixfr4-TS1 smoke

Run immediately after minimal 2D works. Verify that no 2D-only reshape,
upsampling, or metric assumptions remain.

### Stage F: 2D mechanism table

Run:

- Ours minimal;
- Sequential FT;
- Head-only adapter.

Then:

- suppress-loss ablation;
- background-plasticity ablation;
- random versus background new-slot initialization.

Use at least three seeds for reported conclusions.

### Stage G: full controlled OWT table

Add:

- Offline upper bound;
- OWT-Seg joint;
- OrganSlotBank full FT;
- optional dense adapter.

### Stage H: BTCV

Order:

```text
4-1 -> 7-1 -> 4-4
```

### Stage I: full 3D

Run after 2D mechanism and early 3D smoke are stable.

### Stage J: WORD

Run after BTCV gives a credible mechanism result.

## 18. Suggested Run Budgets

The plan locks fairness, not exact epoch counts.

Recommended development budgets:

```text
unit test: seconds
one-batch smoke: 2-5 iterations
tiny overfit: 200-500 optimizer steps
subset smoke: 2 epochs
mechanism pilot: 20-100 epochs depending on convergence
final: chosen from validation convergence, identical across compared methods
```

The original OWT scripts use:

```text
blr = 1e-4
weight_decay = 0.05
warmup = 60
epochs = 1200
token_factor = 20
input = 224
LA = enabled
2D batch per GPU = 96 in the old two-GPU script
3D batch per GPU = 64 in the old two-GPU script
```

Do not immediately spend 1200 epochs on an unverified new pipeline. Determine
the final fixed budget after smoke and convergence pilots. Keep effective
batch size, optimizer, schedule, augmentation, and iteration count matched
within every comparison table.

Final reporting:

- seeds: at least 3;
- mean and standard deviation;
- exact base checkpoint shared across incremental baselines;
- exact run command and resolved config saved.

## 19. Slurm And Output Isolation

Environment:

```text
/gpfs/work/aac/bolinren19/.conda/envs/abdpet
```

Recommended layout:

```text
slurm/orgslot/smoke/
slurm/orgslot/debug/
slurm/orgslot/btcv/
slurm/orgslot/word/
slurm/orgslot/logs/

Results/OrganSlotBank/
  OWTLegacy/
  BTCV/
  WORD/
```

Run ID should include:

```text
dataset
protocol
stage
method
dimension
initialization
seed
config hash
```

Every result directory should contain:

```text
resolved_config.yaml
command.txt
git_commit.txt
git_diff.patch or dirty-status warning
dataset_manifest_checksums.json
class_map.yaml
train.log
metrics.jsonl
best_checkpoint.pth
last_checkpoint.pth
eval/
visualizations/
parameter_report.json
checkpoint_load_report.json
```

Do not write OrganSlotBank results into PSEM, LossBalance, or original OWT
result directories.

## 20. Required Tests

### Model tests

- 2D forward output shapes.
- 3D Fixfr4 output shapes.
- Every slot produces exactly `K` tokens.
- AHER produces exactly `N` canvas positions.
- normalized additive fusion is correct for 1, 2, and S slots.
- per-sample keep masks do not affect other samples.
- all-dropped input is rejected or repaired to one retained slot.
- appending a slot preserves old state keys and tensors.

### Visibility tests

- base loader exposes only old masks and provisional background.
- hidden Liver is included in base provisional background.
- incremental train loader exposes only Liver mask.
- old/full GT is unavailable from the incremental training batch.
- train/val/test case IDs are disjoint.

### Loss tests

- dropped slots contribute zero segmentation loss.
- retained slots receive gradients.
- new-region reconstruction ignores pixels outside new GT.
- empty masks do not produce NaN.
- suppress loss uses detached pseudo masks.
- old GT never enters incremental loss functions.

### Freeze tests

- frozen parameter hashes are unchanged.
- frozen gradients are `None`.
- raw old logits match before/after incremental training.
- new slot and allowed background modules do update.

### Migration tests

- joint collector `conv2` slices map to the correct slots.
- original checkpoint key report is complete.
- visible-slot transfer skips future slots.
- base and incremental checkpoint resume work.

### Metric tests

- synthetic masks have known Dice.
- numeric slice sorting is correct (`_2` before `_10`).
- case aggregation is correct.
- empty-case policy is explicit.
- Old/New/All means and forgetting are correct.

## 21. Decision Gates

### Gate A: wiring

Pass only if:

- 2D smoke runs;
- all losses are finite;
- freeze/train groups are verified;
- base reconstruction and segmentation are nontrivial;
- checkpoint save/load is exact.

### Gate B: mechanism

Pass only if:

- new Liver learns;
- raw old paths do not drift;
- final old DSC drop is lower than Sequential FT/OrganSlotBank FT;
- new-on-old FP is controlled;
- result is better than head-only adapter in a meaningful way.

### Gate C: paper evidence

Pass only if:

- forgetting improves over OWT-Seg joint FT and OrganSlotBank FT;
- new-class DSC remains acceptable;
- added parameters are defensible;
- slot/reconstruction diagnostics support the mechanism;
- results hold across seeds and formal BTCV protocols.

If Gate C is weak, simplify the claim or improve the P0 mechanism. Do not hide
failure by adding many P1/P2 modules at once.

## 22. P0, P1, And P2 Boundary

P0:

- base reconstruction + segmentation;
- organ-wise slots;
- strict freeze+append;
- new-region reconstruction;
- suppress ablation;
- weak background update;
- calibrated argmax;
- Retain-New TGR.

P1:

- incremental LPIPS crop;
- old fused-logit KL;
- background teacher;
- low-LR background TGEnc;
- threshold sensitivity.

P2:

- register tokens;
- learnable canvas fusion;
- CrossSlotMixer;
- token orthogonality/contrastive objectives;
- decoder adapter.

Implement P0 first. Add one component at a time with a controlled ablation.

## 23. Relationship To PSEM And LossBalance

PSEM:

- tests/fixes per-sample class-token masking in original OWT;
- may inform the deterministic slot TGR sampler;
- does not implement expandable organ slots.

LossBalance:

- tests organ-aware reconstruction weighting for small organs;
- may later be combined with OrganSlotBank after the main mechanism works;
- is not the incremental freeze+append method.

Do not start by combining:

```text
PSEM + LossBalance + OrganSlotBank + background plasticity + suppression
```

That would make failure attribution impossible.

## 24. First Implementation Checklist

1. Create a separate worktree/branch from clean `main`.
2. Copy this handoff and record the starting commit.
3. Add class/stage config and strict visibility dataset wrapper.
4. Add `OrganSlot`, `OrganSlotBank`, and minimal binary head.
5. Add the new model with encoder, per-slot path, normalized fusion, decoder.
6. Add model/loss/visibility/migration unit tests.
7. Implement base loss and per-sample Slot-Aware TGR.
8. Implement base training and validation.
9. Implement OrganSlotBank checkpoint save/load and OWT migration report.
10. Train/overfit a tiny OWT-legacy base sample.
11. Implement append/copy-from-background and freeze verification.
12. Implement incremental minimal Liver loss.
13. Implement head-based evaluator and case aggregation.
14. Run 2D minimal smoke.
15. Run early 3D Fixfr4-TS1 smoke.
16. Run Ours, Sequential FT, and Head-only with one shared base checkpoint.
17. Add suppression as a separate ablation.
18. Add weak background plasticity as a separate ablation.
19. Add full controlled baselines.
20. Only then create strict BTCV 4-1/Base4 manifests and run formal tests.

## 25. Completion Definition For The Next Coding Session

The first coding session is complete only when it delivers:

- new files without breaking original OWT/PSEM paths;
- passing CPU unit tests;
- passing 2D forward/backward;
- passing 3D Fixfr4 forward/backward;
- a tiny-batch overfit result;
- a saved/loaded base checkpoint;
- an appended Liver slot initialized from background;
- verified frozen old parameters;
- one incremental forward/backward using only Liver GT;
- head-based validation metrics;
- a Slurm smoke job that exits successfully;
- an experiment log recording commands, configs, and observed results.

It is not complete merely because the model imports or a long training job has
been submitted.
