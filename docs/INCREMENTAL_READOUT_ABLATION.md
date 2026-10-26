## 2026-10-26 当前结果更新

当前完成的SAM新四类独立pixel decoder＋共享新交互：167611/167612，
固定0.5后处理Old4/New4/All8为86.776017/80.677357/83.726687；
验证集校准后为87.116398/81.067890/84.092144。
相对156538/156539：New4提升4.50、All8提升2.25个百分点。
源码365dc78d2185a8e36161b99a3d49977919db7768；2×4090，每卡16、累积6、有效batch192，
118800 updates，checkpoint802；冻结审计match=true。
注意同时改变新空间分支、交互共享方式及microbatch组合，不能解释为纯pixel decoder单因素收益。

完整冻结矩阵、配置、消融表、PCDD/Offline对照及产物路径统一维护在
[主README增量章节](../README.md)。
下文按历史时间保留；旧“未提交/等待”状态以本节最新结果为准，其他实验状态不可类推。

# Stage2: shared vs stage-private query readout

## Spatial adapter32 controlled ablation (2026-10-25)

Opt-in `STAGE2_PIXEL_DECODER=adapter32`, SAM `slot_private`, readout adapter `none`.
Unlike `new4`, all new-organ interaction stacks remain independent. Frozen
P16/P8/P4 are preserved for old/background heads. New4 share three residual
adapters using fused maps, a shared 768->32 Z projection, and pre-fusion S4/S8.
Each scale uses a 128->32 projection, depthwise3x3, GroupNorm(8), GELU and
zero-initialized 32->128 output. Exactly 58,400 added parameters at C=128,D=768.
Local FP32 computation; output maps retain original dtype. ViT, original pixels,
old slots, shared SAM-tail and recon decoder freeze; frozen tail preserves input
gradients. Background separation and existing segmentation/aux losses unchanged.
Resume policy is checked; evaluator reconstructs adapter before strict loading.
Initialize from the same SAM Stage1, not a Stage2 checkpoint; 118800 updates,
batch16/GPU, effective192, original official-pool split and validation calibration.
This is a proposed spatial adaptation test, not an established performance gain.

## New4 shared spatial + interaction ablation (2026-10-16)

Implemented opt-in `STAGE2_PIXEL_DECODER=new4` with
`INCREMENTAL_ARCHITECTURE=sam_soft_aux`, `STAGE2_SHARED_SEGMENTATION=slot_private`,
`STAGE2_READOUT_ADAPTER=none`. New4 share ONE copied pixel decoder (including
spatial stem, excluding unused query projections) and ONE copied token projection
+ P16/P8/P4 interaction stack. Organ Collectors/TGEnc remain independent.
Both copies initialize from Stage1; old/background use original frozen spatial
features and frozen interactions. ViT, shared SAM-tail and reconstruction decoder
remain frozen; existing background separation policy remains unchanged.

This differs from the per-slot private baseline in TWO respects: trainable new4
spatial features and stage-shared rather than per-organ new4 interaction weights.
It must not be presented as an isolated pixel-only ablation against that baseline.
Train from the same Stage1 with 118800 Stage2 updates, not from a trained Stage2.
Old checkpoints default to shared frozen pixels. New checkpoints reconstruct the
new4 branches before strict loading; resume rejects a changed pixel policy.
No GPU run or performance result is established by this implementation record.

## 阅读顺序与当前冻结矩阵（2026-10-13整理）

先读本节与下方Formal SAM结果；Historical design以下是有日期的设计原文，不是当前队列。
当前全预算为118800更新/阶段、总237600；早期E59400/阶段必须单列。
两阶段重复访问同96病例，仅开放不同标签，不声称阶段间无旧图像访问。
Stage1为背景+旧4类，Stage2背景+旧4+新4；Stage1新类指标为NA，不填0。

| Stage2策略 | ViT/空间pixel decoder | 旧slot | 新slot | 交互/读出 | 共享recon decoder |
|---|---|---|---|---|---|
| E frozen | 冻结/冻结 | 冻结 | 训练 | 全冻结 | 冻结 |
| E train，旧59400对照 | 冻结/训练 | 冻结 | 训练 | 共享query与读出训练 | 冻结 |
| E query_shared | 冻结/冻结 | 冻结 | 训练 | 共享query投影、cross-attn、FFN训练 | 冻结 |
| E query_split | 冻结/冻结 | 冻结 | 训练 | 旧组冻结，新4共用一套复制的读出训练 | 冻结 |
| SAM slot_private | 冻结/冻结 | 冻结 | 训练 | 每slot投影/多尺度交互私有；新4训练，旧冻结；tail冻结 | 冻结 |
| SAM slot_private+residual16 | 同上 | 冻结 | 训练 | 同上，额外新4共享4240参数读出adapter | 冻结 |

背景Collector/TGEnc/token_norm/AHER按separation策略0.1倍LR训练，背景head/calibration冻结。
冻结tail/decoder仍保留输入梯度，不是no_grad切断新slot反传。旧阈值锁定，新阈值用验证集。
标签隔离、teacher 0.1/0.9、ignore和背景损失公式见[配置](CONFIGURATION.md#incremental44当前契约)。

### 早期共享空间解冻的已读原始对照

本轮读取SIP仓库根 `artifacts/inc44_spatial_readout_20261004/evaluation/stage_comparison.json`。
训练3004004，59400 Stage2更新；其Stage1为2998659，不是下方SAM的Stage1。

| 阶段 | 固定post旧/新/全 | 校准post旧/新/全 |
|---|---|---|
| E Stage1 | 84.89 / NA / NA | 85.48 / NA / NA |
| E共享空间+读出解冻Stage2 | 0.04 / 81.20 / 40.62 | 0.09 / 82.23 / 41.16 |

这是“旧slot参数冻结不保证旧输出保持”的直接反例。不能将其与SAM81.84的差距全部
归因于私有交互：架构、Stage1、训练预算也不同。尚未完成的query_shared/query_split
正是为了进一步隔离空间分支与交互读出的影响；实时状态另查[STATUS](STATUS.md)。

诊断目标仍需区分旧参数hash、旧binary logits/指标、增加新类后的互斥mask；
本页没有宣称跨增量顺序、跨seed的普遍零遗忘。

## Formal SAM Stage2 results (2026-10-13)

This section supersedes the historical pending-submission statements below for
the full-budget SAM `slot_private` run. Account/cluster: sifansong/XEC.
Stage1 train/eval: 156536/156537; Stage2 train/eval: 156538/156539.
Stage2 completed 2026-10-13 03:00:43 CST; evaluation completed 04:24:30.
Both stages use 118800 optimizer updates; Stage2 final epoch802. This is
237600 total updates, not compute-matched to 118800-update Offline training.
Head `complete_split=true`, reconstruction `complete_test_set=true`; both
exactly loaded checkpoint802 and evaluated 7315 slices / 24 cases.
Official-pool seed42: 96 train / 26 validation / 24 test; thresholds selected
on validation, not test. Background excluded from foreground averages.

| Postprocessed case-mean Dice (%) | Old4 | New4 | All8 |
|---|---:|---:|---:|
| Stage1 fixed0.5 | 86.78 | N/A | N/A |
| Stage2 fixed0.5 | 86.78 | 76.39 | 81.58 |
| Stage1 calibrated | 87.12 | N/A | N/A |
| Stage2 calibrated | 87.12 | 76.57 | 81.84 |
| PCDD reported 4–4 | 84.09 | 82.75 | 83.42 |

PCDD is an external reported-number reference, not a matched-protocol
reproduction: https://ojs.aaai.org/index.php/AAAI/article/download/38406/42368
(Table 2). Calibrated differences are +3.03 / -6.18 / -1.58 percentage points.
Do not compare our four-class Stage1 mean directly with an eight-class mean.

| Organ | Stage1 calibrated | Stage2 fixed0.5 | Stage2 calibrated | Offline E seed42 calibrated | Stage2 recon direct-post |
|---|---:|---:|---:|---:|---:|
| Spleen (old) | 92.83 | 92.62 | 92.83 | 94.07 | 90.72 |
| Right kidney (old) | 92.94 | 92.18 | 92.94 | 93.72 | 90.01 |
| Left kidney (old) | 93.14 | 92.48 | 93.14 | 93.93 | 90.05 |
| Gallbladder (old) | 69.56 | 69.83 | 69.56 | 69.63 | 61.35 |
| Esophagus (new) | N/A | 63.23 | 63.79 | 71.72 | 47.16 |
| Pancreas (new) | N/A | 66.75 | 66.75 | 78.91 | 60.00 |
| Liver (new) | N/A | 94.51 | 94.51 | 95.72 | 90.84 |
| Stomach (new) | N/A | 81.09 | 81.23 | 87.43 | 72.13 |
| All8 | N/A | 81.58 | 81.84 | 85.64 | 75.28 |

Offline E uses the same split but a different architecture and full eight-class
supervision; it is a reference, not a controlled ablation of private interaction.
The unrounded Stage2 head averages are 81.5847223890469 fixed-post and
81.84336561343312 selected-post. Old4 fixed-post forgetting is exactly 0 in the
report; per-organ binary scores also match Stage1. Final freeze audit reports
`match=true`, with no missing, added, or changed audited frozen parameters.
This establishes retention for the evaluated binary outputs, not invariance of
exclusive merged masks or a statistical claim of general zero forgetting.

Adaptation: Stage1 shares multiscale interaction weights across organ calls.
Stage2 copies its projection and P16/P8/P4 TokenBlocks into each organ slot;
new4 slots and their private interactions train, old paths freeze. ViT, pixel
decoder, shared SAM-tail and shared reconstruction decoder freeze. Background
collector/TGEnc/token_norm/AHER train under the existing 0.1x-LR separation
policy. The frozen tail remains differentiable with respect to its inputs.

Head calibration thresholds (organ IDs1–8): .6/.8/.8/.3/.7/.5/.5/.6.
Calibrated global predicted/GT volume ratios are 1.1608 for esophagus and
1.1109 for pancreas: low Dice is not explained by underprediction alone.

Reconstruction threshold .02: direct-post All8=75.28328597392482%;
indirect-post All8=6.103221352517983%. Indirect-post predicted/GT volume ratios
are 545.95 (esophagus) and 88.96 (pancreas). Severe indirect false positives
are confirmed; the cause (difference definition, fusion/normalization,
background residual, threshold scale) remains unconfirmed. A low reconstruction
training loss does not establish good segmentation or incremental disentanglement.

Next controlled comparison: adapter run157119 started 2026-10-13 04:24:30 CST,
still training at the last check, no final result. It retains slot-private
interaction and adds one new4-shared 4240-parameter residual readout adapter:
`w = frozen_mask_mlp(m) + Linear16_GELU_Linear128(m)` with a zero-initialized
last layer. Old4/background bypass it. It is neither stage-shared attention
nor Q/V LoRA. New-class gains and old-class retention must both be evaluated.

Evidence (remote sifansong/XEC, base
`/gpfs/work/aac/sifansong/artifacts/inc44_full_20261006/SAM/`):

- `stage2_evaluation/stage_comparison.json`
- `stage2_evaluation/test/results.json` and `resolved_config.json`
- `stage2_evaluation/reconstruction/results.json` and `resolved_config.json`
- `stage1_evaluation/test/results.json`
- `stage2_separation_156538/freeze_audit.json` and `log.txt`
- Training test-manifest SHA256: `1a1a6fc6a5e82dd37276d5b5be0ee6496b32cac4147a3b18a3003f3ec5eba703`.

Offline E reference (local repository root):
`artifacts/offline_eval_recovery_20261005/evaluation/ArmE_CrossAttn_MAEenc_Official96_2998655_heads_test/results.json`.

## Historical design and implementation notes

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
