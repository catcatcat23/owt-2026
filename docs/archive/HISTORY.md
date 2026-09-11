# 历史原文档案（按需查询）

来源提交：`0ba5304c71c961c6cf5dfd5b94f5b03fe52b3111`。
原文逐字保留；所有状态、指令和路径仅反映各自历史时点。
不要默认读取本文件；先读 ../README.md。

- [docs/ARM_E_MAE_AND_RECON_20260908_CN.md](#source-1)
- [docs/ARM_F_2D_3D.md](#source-2)
- [docs/BRANCH_CONSOLIDATION_20260907.md](#source-3)
- [docs/ORGAN_SLOTBANK_ARCHITECTURE_CN.md](#source-4)
- [docs/ORGAN_SLOTBANK_EXPERIMENT_LOG.md](#source-5)
- [docs/ORGAN_SLOTBANK_EXPERIMENT_SUMMARY_CN.md](#source-6)
- [docs/ORGAN_SLOTBANK_IMPLEMENTATION_HANDOFF.md](#source-7)
- [docs/ORGSLOT_112_NATIVE_CROP448_EXPERIMENT_PLAN.md](#source-8)
- [docs/ORGSLOT_ARMB_3D_WORD070_ROI20_CN.md](#source-9)
- [docs/ORGSLOT_ARM_E_CROSS_ATTENTION_CN.md](#source-10)
- [docs/ORGSLOT_AUTOPET_MAE_TRANSFER_CN.md](#source-11)
- [docs/ORGSLOT_CONFIGURATION_CONTRACT_CN.md](#source-12)
- [docs/ORGSLOT_WORD070_ARM_E_MULTISCALE_PIXEL.md](#source-13)
- [docs/ORGSLOT_WORD070_MULTIQUERY_EXPERIMENT.md](#source-14)
- [docs/ORGSLOT_WORD070_QUERYMASK_3D_EXPERIMENT_CN.md](#source-15)
- [docs/ORGSLOT_WORD070_QUERYMASK_EXPERIMENT.md](#source-16)
- [docs/ORGSLOT_WORD070_SMALL_ORGAN_LOSS_CN.md](#source-17)
- [docs/ORGSLOT_WORD448_JOINT_SLOT_SEG_V1.md](#source-18)
- [docs/ORGSLOT_WORD448_RETAINED_MULTISCALE_FOCAL_V2.md](#source-19)
- [docs/ORGSLOT_WORD448_ROI20_LOSS3_EXPERIMENT.md](#source-20)
- [docs/ORGSLOT_WORD_COMMON8_SOTA_CN.md](#source-21)
- [docs/ORGSLOT_WORD_RESOLUTION_ABLATION_LOG.md](#source-22)
- [docs/PIXEL_PE_ABLATION.md](#source-23)
- [docs/PIXEL_PE_NUMERICS_20260908.md](#source-24)
- [docs/THREED_SLICEWISE_DIAGNOSTIC.md](#source-25)
- [README.md](#source-26)
- [README_EXPERIMENTS_CN.md](#source-27)

<a id="source-1"></a>

## docs/ARM_E_MAE_AND_RECON_20260908_CN.md

SHA256: `7c2d369982cfe2fbc16b8d97c1e9838dddf68f6caa1a1bb907a30be0aca9c5f0`

````text
# Arm B MAE recon 补齐与 Arm E MAE 三臂实验（2026-09-08）

## 已完成的 Arm B MAE：统一八类结果
指标：24 病例 / 6990 切片，checkpoint-802 exact=true，case Dice presence mean ×100。
Reconstruction 为 Direct-post，固定阈值 0.02，min_size=20、opening_radius=1。
两组 MAE 原始 JSON 已归档到 artifacts/mae_recon_audit_20260908/。
原始 Arm B 数字来自既有 ORGSLOT_WORD_COMMON8_SOTA_CN.md 的已验证记录。

| 初始化 / recon | 脾脏 | 右肾 | 左肾 | 胆囊 | 食管 | 胰腺 | 肝脏 | 胃 | 八类均值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Encoder+decoder | 93.19 | 92.12 | 92.21 | 57.05 | 69.83 | 71.33 | 94.70 | 80.95 | **81.42** |
| Encoder-only | 92.56 | 92.61 | 91.78 | 50.98 | 71.00 | 71.10 | 94.25 | 80.25 | **80.57** |
| Scratch Arm B | 92.74 | 91.48 | 91.92 | 53.74 | 69.48 | 70.35 | 94.58 | 79.28 | **80.45** |

| 初始化 | Head fixed-0.5 post | Head train-calibrated post | Recon Direct-post |
|---|---:|---:|---:|
| Scratch Arm B | 79.89 | 80.36 | 80.45 |
| Encoder-only | 80.60 | 80.97 | 80.57 |
| Encoder+decoder | 81.09 | 81.69 | 81.42 |

Head 数字沿用此前已核验汇报；本次新核验的是 recon JSON。
Encoder+decoder 相对 encoder-only 的 recon 提高 0.8552 个百分点。
相对 scratch recon 约 +0.97；胆囊提升最明显，但 encoder-only 胆囊低于 scratch。
不能声称所有器官同步获益，或凭单 seed 证明收益机制。
以上是 Arm B MAE 内部排名，不是全项目新 SOTA；Arm C/D 的既有约 82% 结果仍需同列比较。
Head/recon 各自单独排名，不在测试集逐器官挑输出拼接。

## 数据来源
- Encoder-only：sifansong / XEC，训练 128402，评估 128404。
  worktree: /gpfs/work/aac/sifansong/worktrees/orgslot_autopet_mae
  result: Results/OrganSlotBank/evaluation/Common8/WORD_2D/OrgSlot_WORD07072_ROI20_RetainedMultiConv_SmallOrgan_L001_AutoPETMAE_EncoderOnly_128402_ckpt802_reconstruction_fixedthr002/results.json
- Encoder+decoder：antengcai23 / XEC，训练 128401，评估 128403。
  worktree: /gpfs/work/aac/antengcai23/worktrees/orgslot_autopet_mae
  result: Results/OrganSlotBank/evaluation/Common8/WORD_2D/OrgSlot_WORD07072_ROI20_RetainedMultiConv_SmallOrgan_L001_AutoPETMAE_EncDec_128401_ckpt802_reconstruction_fixedthr002/results.json
- 两文件 complete_test_set=true，samples=6990，八类 case_count 均为24。

## Arm E MAE 三臂
用户明确要求在原始 Arm E 上也比较 scratch / encoder / encoder+decoder。
不加 cross-attention，不加 PE，不加载 pixel decoder、Collector/TGEnc/AHER 权重。

| 组 | 目的 | 账号/集群 | 训练 | 评估 | 状态 |
|---|---|---|---|---|---|
| E0 scratch | 多尺度空间分支无预训练对照 | bolinren19 / SIP | 2892114 | 2921054 | 训练 epoch498，评估 afterok 等待 |
| E1 encoder | 隔离 MAE encoder 收益 | sifansong / XEC | 133896 | 133898 | 训练 PENDING (Priority)，评估 afterok |
| E2 encoder+decoder | 检验共享重建 decoder 额外收益 | sifansong / XEC | 133897 | 133899 | 训练 PENDING (Priority)，评估 afterok |

E0 正式评估已提交：1×A800、10 CPU、128GB、angelosstefanidis/8a800；
afterok:2892114，依次 train calibration / full head / full recon。
E1/E2 每条4×A800、24CPU、192GB、sifansong/8gpus。
两条正式训练于2026-09-08直接提交，没有申请 smoke。

固定：WORD 0.7×0.7×2、input/ROI 448/384、ROI probability0.2、small-organ loss、
lambda_seg0.01、retained、有效 batch192=4×8×6、118800更新、
warmup5940、blr1e-4（实际lr7.5e-5）、seed0、FP16 GradScaler、无新增梯度裁剪。
MAE checkpoint SHA256:
0b3571a3e79095686a0b12649735a298aaeb8a0b3bb93a47f3484439ac4a1ec6

## 兼容快照与复现
本地运行快照：.worktrees/orgslot_arm_e_mae_20260908。
远端运行快照：
/gpfs/work/aac/sifansong/worktrees/orgslot_arm_e_mae_20260908
；上传归档为
/gpfs/work/aac/sifansong/arm_e_mae_20260908.tar
，SHA256 为
d57d709a1822f398450607555c40ba357cd0cbec427f377b240055e49f1d4a1d。
基于原 Arm E 提交81adba2，而非直接使用改变了数值策略的最新版训练引擎。
原始训练引擎和 OrganSlotEmbed.py 与 E0 运行目录 SHA256 完全一致。
仅补 MAE 参数/严格加载/加载记录、Git provenance 容错和源文件哈希。
加载器 util/mae_transfer.py 及数据核验器取自统一分支01f1666。
入口：
- slurm/orgslot/train/arm_e_mae_sifan_xec.sbatch
- slurm/orgslot/eval/arm_e_mae_unified.sbatch
- scripts/orgslot/arm_e_mae_81adba2.patch：旧 main/launcher 的完整兼容补丁。
在81adba2归档上应用补丁，再加入上述加载器和入口即可复现。
不要把训练入口指向任意后续 worktree，否则会失去 E0 可比性。

已完成 Python compile、bash -n 和旧新引擎哈希核验；没有新增 GPU smoke。
最终以三组完整checkpoint802同协议结果判断 Arm E MAE 是否超过既有项目内最好结果；
不能用当前epoch498训练loss推断最终SOTA。
````

<a id="source-2"></a>

## docs/ARM_F_2D_3D.md

SHA256: `0b5949136bca918878aadb20bea44c1fca01bd63f1096527995ff8c6027cbf9a`

````text
# Arm F: organ-conditioned two-stage attention

Branch: `experiment/orgslot-arm-f`, based on `247ded90e5d49603f68ace2153b9a6fad6ba2d8d`.
No training results yet. No Slurm jobs submitted by this implementation.

## Model configuration

Use the existing training entry `main_pretrain_orgslot_common8_a100.py`:

```
--slot_head_type arm_f_attention --slot_head_channels 128
--token_factor 20 --query_refinement none --pixel_pe none
--seg_loss_type small_organ --lambda_seg 0.01
--tversky_alpha_fp 0.3 --tversky_beta_fn 0.7
--balanced_focal_weight 0.5 --hard_negative_ratio 0.02
--negative_slice_weight 0.1 --focal_alpha 0.75 --focal_gamma 2
--organ_roi_aug --organ_roi_probability 0.2
--input_size 448 --global_crop_size 448 --roi_crop_size 384
--expected_spacing 0.7 0.7 2 --amp_dtype bf16 --clip_grad 1
```

2D: `--dimension 2D --segmentation_unit slice`.
3D: `--dimension 3D --fix_frame 4 --temp_stride 1 --segmentation_unit slice`.
Linear readout control: replace head type with `arm_f_linear`.
Arm F always uses fixed axial sin/cos attention PE: y/x in 2D, t/y/x in 3D.
`pixel_pe none` refers to the older decoder's optional PE, not disabling Arm F PE.
Arm F uses its own three refinement blocks; `query_refinement cross_attn` remains
an Arm E-only option and must not be used here.

Dataset CSV, ROI index, preprocessing summary, LPIPS weights, output paths and
initialization must be selected for the actual account/cluster. Do not copy
another account's paths. Match effective batch, LR, update budget and initial
encoder/reconstruction weights to the chosen comparison arm; no claim of fair
comparison is made merely from equal epochs across 2D/3D.

## Shape and implementation

Collector/TGEnc retains image-conditioned `[B,20,D]` tokens. Input projection
and the existing per-slot identity produce `[B,20,128]`. Three separately
parameterized blocks read P16, fused P8, fused P4. Each block is pre-norm cross
attention, organ-local self-attention, FFN. All decoder weights are shared across
organs. No Hungarian matching, instance loss or mask gating is introduced.

For 448 input: maps are `[B,128,28,28]`, `[B,128,56,56]`, `[B,128,112,112]`.
For 3D: maps are `[B,128,4,28,28]`, `[B,128,4,56,56]`, `[B,128,4,112,112]`.
The 3D stem applies the original Arm E 2D convolutions slice-wise. Cross-attention
can access all slab positions with fixed t/y/x PE, but there is no added temporal
Conv3D mixing. This new stem is a separate architectural change versus 3D Arm D;
use a matched Arm F linear readout to isolate reverse attention effects.
Temporal encoder stride >1 averages input slices to the encoder time grid before
the stem and interpolates final logits back; the intended baseline uses stride 1.

Stage B: P4 queries read updated tokens; output retains a P4 residual, then FFN,
shared LayerNorm/linear classifier, interpolation and original per-slot calibration.
Every organ has one output channel; raw/calibrated dictionaries and active-row
selection are unchanged. Reconstruction/AHER/fusion are retained.

Linear control: mask MLP -> 20 dot-product scores -> learned linear token fusion.
Only the fused mask is supervised. This is equivalent to a fused dynamic mask
embedding, not pixel-adaptive token selection. Token indices/count must be stable.

Existing 2D/3D head evaluators rebuild from saved `slot_head_type` and channels;
Arm F checkpoints require this branch's code and strict loading. No partial
loading of an Arm E checkpoint as a full Arm F resume is supported.

## Numerical and resource behavior

Standard `nn.MultiheadAttention(batch_first=True, need_weights=False)`, four heads,
FFN ratio 4, dropout 0. Attention/FFN/readout run locally in FP32 for compatibility
with the cluster's older PyTorch; spatial convolutions retain ambient AMP.
Interpolation uses the existing FP32-safe helper. Never call `model.bfloat16()`;
keep FP32 parameters and use autocast. No full attention heatmaps returned in the
training path. Attention weights are not foreground probabilities.

P4 features and FFN activations scale per active organ; loops do not free saved
autograd activations before backward. 3D has four times as many spatial positions
as 2D. Full-size GPU memory, speed and DDP preflight remain unverified; choose
microbatch from measurement, retaining effective batch via accumulation.

## Validation and next experiments

CPU tests cover 2D/3D attention and linear heads, finite backward, all new decoder
parameters receiving gradients, Collector gradients, active rows, unchanged
reconstruction shape, strict checkpoint roundtrip, 20-token sensitivity, temporal
PE identity and BF16 inputs to attention. CPU BF16 whole-stem execution is not
claimed: the installed CPU GroupNorm rejects mixed BF16/FP32 inputs.

Validated on 2026-09-10: 33 unittest cases (`test_arm_f`, `test_orgslot_model`,
`test_query_cross_attention`) passed, plus all 7 standalone slice-wise loss and
3D evaluator regression functions. `git diff --check` passed. No GPU/DDP run.

Compare Arm E, Arm F linear, Arm F attention under the same protocol. Additional
single-scale and depth-matched ablations are not implemented yet. Monitor eight
organ Dice, precision/recall, empty-slice FP, volume ratio, memory and throughput.
No SOTA or efficacy claim before full evaluation.

Future formal jobs use maximum allowed walltime, verified on submission: SIP
4a800/8a800 7 days; XEC 8gpus 5 days, 4gpus 7 days. Existing jobs unchanged.
Formal submission scripts: `slurm/orgslot/train/arm_f.sbatch` and
`slurm/orgslot/eval/arm_f.sbatch`, with explicit `ARM_F_TARGET`,
`ARM_F_DIMENSION`, `EXPERIMENT_WORKDIR`, `ARM_F_COMMIT` exports.
2D: microbatch 2 x 4 GPUs x accumulation 24 = 192 slices.
3D: microbatch 1 x 4 GPUs x accumulation 12 = 48 slabs (192 slices).
Both use scratch initialization, actual LR 7.5e-5, 118800 updates. Smaller
microbatches may change legacy_batch TGR behavior; do not claim exact equivalence
from matching effective batch alone. The first evaluation is fixed threshold 0.5.
The branch is fast-forwarded to origin/feature/orgslot before runtime snapshots
are created and checked; existing worktrees are not updated.
````

<a id="source-3"></a>

## docs/BRANCH_CONSOLIDATION_20260907.md

SHA256: `7f129cd42c9be627ea3173f8bf53808d480a5d9dc9e49c5b1611260d90ab5608`

````text
# Branch consolidation: 2026-09-07

Recommended development branches: main, feature/orgslot, experiment/psem, experiment/lossbalance.
feature/orgslot contains all OrganSlot histories/configurations. Development
worktree: .worktrees/orgslot_integrated. Historical experiment worktrees remain at
their original commits in detached HEAD state, preserving logs, checkpoints and uncommitted changes.
Existing Slurm tasks continue to use their original directories.

Consolidation and eight archive tags have been pushed to GitHub. After explicit
user authorization, all eight old local and six corresponding remote experiment
branch refs were deleted. Remote deletion used atomic push with exact expected
SHAs; local deletion used merged-branch checks. Archive tags remain on GitHub.
The old orgslot_head_ab worktree is detached at 1b104f4; its files are preserved.

Archive tags use prefix archive/20260907/ and the suffix below:

| Former experiment branch / tag suffix | Original commit |
|---|---|
| orgslot-3d-slicewise | 8984c9e |
| orgslot-armb-3d | 6bee802 |
| orgslot-autopet-mae-transfer | b43c7f0 |
| orgslot-pixel-pe | 4e7bf7c |
| orgslot-querymask | 4d7c5e0 |
| orgslot-querymask-3d | ab3bf1f |
| orgslot-querymask-multiquery | 4d7c5e0 |
| orgslot-querymask-multiscale-pixel | 81adba2 |

Recover with `git switch -c <new-name> <archive-tag>` in an unused checkout.
No experiment result or checkpoint is deleted.

Unified controls: dimension, slot_head_type, pixel_pe, segmentation_unit,
mae_init_checkpoint and mae_init_scope. segmentation_unit defaults to slab
for legacy behavior; slice-wise and 3D PE scripts explicitly select slice.
2D loss is unchanged. Arm E remains 2D-only.

Validation: 51 model/PE/Arm E/loss tests, 6 MAE tests, 3 slice-wise checks,
shell syntax and git diff checks passed. GPU experiment validation is separate.
````

<a id="source-4"></a>

## docs/ORGAN_SLOTBANK_ARCHITECTURE_CN.md

SHA256: `b318439569364da96ff37aecf3795ec5383cfc68b3459d8ff2f7ea8e79130092`

````text
# E-OWT-Seg / OrganSlotBank 架构与实现说明

## 1. 一句话说明

OrganSlotBank 把原始 OWT 中“所有器官共用的一条联合 token 路径”，改成了
“每个器官拥有一个可以独立追加、独立训练、独立冻结和独立输出分割结果的
slot”。共享 ViT 图像编码器和共享重建解码器仍然保留，因此新增一个器官时不需要
重建整个网络，也不需要在增量阶段读取旧器官标注。

当前 v0 的目标不是一次完成最终论文实验，而是先证明四件事：

1. 增量训练的数据接口拿不到旧类和未来类 GT；
2. 新器官可以通过 append slot 加入，并且旧参数不会被暗中修改；
3. 每个 slot 既能参与 OWT 重建，也能直接输出器官二值分割；
4. 2D、Fixfr4 3D、checkpoint 和 tiny overfit 的工程链路全部可运行。

## 2. 原始 OWT 的数据流和限制

原始路径为：

```text
image
  -> shared ViT patch encoder (blocks1)
  -> one joint OrganCollector
  -> [B, S*K, C] joint organ token bank
  -> one shared TGEnc (blocks2)
  -> one shared AHER
  -> one spatial patch canvas
  -> shared Transformer decoder
  -> reconstructed image
```

其中 `S` 是类别数，`K` 是每类 token 数。不同类别只是在联合 token 序列中占据
不同的连续区间，后面的 TGEnc 和 AHER 仍由所有类别共享。原模型没有器官专属的
监督分割头，主要依靠 token 删除后的图像重建进行学习。

这会给类别增量学习带来三个结构问题：

- 类别数写进联合 collector 的输出通道和总 token 数，追加类别会改变已有张量形状；
- 新类训练会经过共享 TGEnc、AHER 和 decoder，容易改坏旧类表示；
- 没有每类独立 head，无法直接区分“旧表示是否保持不变”和“最终融合是否发生竞争”。

## 3. 新架构总览

新的数据流为：

```text
image
  -> shared ViT patch encoder
  -> patch feature z [B,N,C]
       |-> OrganSlot_background -> canvas_bg, logit_bg
       |-> OrganSlot_old_1      -> canvas_1,  logit_1
       |-> OrganSlot_old_2      -> canvas_2,  logit_2
       `-> OrganSlot_new        -> canvas_new,logit_new

selected slot canvases
  -> normalized additive fusion
  -> shared reconstruction decoder
  -> reconstructed image

calibrated per-slot logits
  -> stack in a fixed semantic order
  -> argmax
  -> final multiclass segmentation
```

共享部分是 patch encoder 和 reconstruction decoder；器官专属部分被封装为
`OrganSlot`，所有 slot 由 `OrganSlotBank` 管理。

## 4. 一个 OrganSlot 内部有什么

每个器官 slot 拥有以下完整路径：

```text
OrganCollector_s
-> K organ tokens
-> OrganWiseTGEnc_s
-> AHER_s
-> patch canvas_s
-> PatchBinaryHead_s
-> raw binary logit_s
-> scale_s * logit_s + bias_s
-> calibrated logit_s
```

### 4.1 器官专属 OrganCollector

每个 slot 从同一个 patch feature `z` 中独立收集 `K` 个该器官 token。这样新增器官
只会新增一个 collector，不改变任何旧 collector 的形状或权重。

### 4.2 器官专属 TGEnc

原始 OWT 有一组共享 `blocks2`。v0 为每个 slot 放置一组局部 Linear-Attention
block，并把默认深度明确设为 `slot_tg_depth=1`。其残差和归一化语义保持原 OWT：

```text
encoded = tokens + SlotNorm(TGEnc(tokens))
```

这里没有声称复制了原始六层 TGEnc。深度 0/1/2 是后续必须报告的消融项。

### 4.3 器官专属 AHER 和 canvas

每个 slot 的 AHER 把自己的 `K` 个 token 恢复成 `N` 个空间 patch feature：

```text
canvas_s: [B,N,D]
```

旧架构只有一个联合 canvas；新架构能明确观察每个器官对空间特征的独立贡献。

### 4.4 轻量二值分割头

`PatchBinaryHead` 使用 `LayerNorm + Linear(D,1)` 在 patch canvas 上产生一个通道的
logit，再用双线性或三线性插值恢复到输入分辨率：

```text
2D: [B,1,H,W]
3D: [B,1,T,H,W]
```

这个 head 很小，能用 Head-only baseline 检验性能究竟来自完整新 slot，还是只需要
在冻结特征上训练一个读出层。

### 4.5 Calibration

每类都有两个标量：

```text
calibrated_logit_s = scale_s * raw_logit_s + bias_s
```

初始值严格为 scale=1、bias=0。Base 阶段可以从可见 GT 学习 calibration；增量
minimal 默认冻结 calibration，因为只用新类 GT 时没有合法的旧类监督去调整旧类
calibration。只有显式启用时才训练新类 calibration。

## 5. 多个 slot 如何共同重建

设样本 `i` 保留的 slot 集合为 `R_i`。代码先对被保留的 canvas 求和，再除以保留
数量的平方根，最后通过共享 `LayerNorm`：

```text
fused_i = LayerNorm(sum(canvas_s, s in R_i) / sqrt(|R_i|))
```

而不是简单求和，是为了避免 slot 数越多，特征幅值线性变大。不能把全部 slot 都
删除；代码和测试都强制每个样本至少保留一个 slot。融合后的 patch canvas 进入原有
共享 decoder，输出 CT/PETCT 图像重建。

## 6. Slot-aware TGR

Base 阶段仍保留 OWT 的 token-group reconstruction 思想，但 mask 改成逐样本的
slot keep mask：

- 由 `sample_index + epoch + seed` 确定性地产生；
- 同一 batch 的样本可以保留不同 slot 子集；
- 保留数量在 1 到 S 之间，不会出现全删；
- 被删除 slot 对应的可见器官区域在重建 target 中置零；
- 被删除 slot 的分割 logit 不进入 segmentation loss，也没有梯度。

它与 PSEM 的“穷举 present-label 二进制组合”不是同一个算法。当前 v0 使用的是
确定性、逐样本、非空的随机外观调度，目的是先隔离 OrganSlotBank 自身的贡献。

增量 minimal 阶段暂时保留全部 slot，确保新器官 canvas 与冻结 decoder 的兼容性；
后续再单独加入 retain-new TGR，不能在第一轮把多项创新混在一起。

## 7. 严格标签隔离

这是当前实现最重要的实验约束。

原始 manifest reader 只负责读取图像和 raw label。训练 DataLoader 之前必须套上
`StrictVisibilityDataset`：

- Base 阶段只返回 Base 可见器官的二值 mask；
- provisional background 只由 Base 可见前景的补集构造；
- incremental Liver 阶段只返回 `visible_masks={liver}`；
- 包装后的 sample 不包含 `label` 或 `full_label`；
- 只有 evaluation wrapper 被允许返回完整 raw label。

因此增量 loss API 在机械上拿不到旧器官 GT。如果 batch 同时出现 old mask 或 full
mask，objective 会直接抛错，而不是静默使用。

病例拆分工具只解析路径中的 case ID，不读取像素或标签内容。训练和验证病例存在任何
交集时，runner 会在建 DataLoader 前终止。

## 8. 损失函数

### 8.1 Base 阶段

```text
L_base = L_reconstruction
       + lambda_lpips * L_LPIPS
       + lambda_seg * mean_retained_slot(Dice + BCE)
```

background 分割项默认权重为 0.25，避免面积巨大的背景主导所有 slot。分割损失只对
当前样本保留的 slot 计算。

### 8.2 增量 minimal

增量阶段只使用当前新器官 mask：

```text
L_inc_min = DiceBCE(logit_new, M_new)
          + lambda_rec * masked_MSE(reconstruction, image, M_new)
```

第二项只在新器官区域计算，用来检查新 slot 产生的 canvas 是否能被冻结 decoder
理解；它不是让新 slot 负责重建整张图。

### 8.3 Old-confidence suppression 消融

这项不读取旧 GT，而是读取冻结旧 head 的高置信预测：

```text
M_old = union(sigmoid(detach(old_logits)) > 0.7)
M_sup = M_old AND NOT M_new
L_sup = masked_BCEWithLogits(logit_new, 0, M_sup)
```

它抑制“新类侵占旧类高置信区域”的 false positive。若没有满足阈值的旧区域，loss
安全地返回有限的 0。

## 9. 类别增量的实际操作

### 9.1 追加新器官

```text
load Base checkpoint
-> append_slot("liver", raw_id)
-> copy background collector/TGEnc/AHER/head weights
-> reset new-slot calibration to identity (1,0)
-> restore semantic name/raw ID metadata
-> verify every pre-existing tensor byte级不变
```

从 background 初始化的原因是背景路径已经学会覆盖广泛解剖区域，比完全随机初始化更
接近可用的空间 collector/canvas。追加操作会在内部克隆 append 前的整个 state dict，
如果任何旧 tensor 改变就立即报错。

### 9.2 Ours

冻结 shared encoder、shared decoder、background 和所有 old slots；只训练新 Liver
slot。minimal 默认也冻结新 slot 的两个 calibration 标量。

### 9.3 Sequential FT

从同一个 Base checkpoint、同一个 background 初始化的新 slot 开始，但开放全部
参数，只用新类 GT 训练。它是用于观察灾难性遗忘的下界，不是另一个新方法。

### 9.4 Head-only

冻结新 slot 的 collector、TGEnc 和 AHER，只训练新 slot 的 `LayerNorm+Linear`
binary head。它用于排除“仅增加一个小 head 就足够”的解释。

### 9.5 冻结验证

runner 在训练前为每个 `requires_grad=False` tensor 计算 SHA-256；每个 epoch 后再次
计算并逐名称比较。missing、added 或 changed 任一非空都会终止训练。

## 10. Checkpoint 与原 OWT 迁移

OrganSlot checkpoint 保存：

- 完整 model state dict；
- optimizer state；
- epoch；
- 有序 slot names；
- semantic name、sanitized key 和 raw class ID metadata；
- stage/method 等额外信息。

同构 OrganSlot checkpoint 使用 `strict=True` 精确回载。原 OWT 初始化则生成显式迁移
报告：共享 patch encoder/blocks1/decoder 能按形状继承；联合 collector 的类别输出
通道按 `K` 切片给对应 slot；原 shared blocks2 的前 `slot_tg_depth` 层复制到各 slot；
原 shared AHER 复制到各 slot。所有未载入、形状不匹配和故意跳过的 slot 都记录在
JSON 中，不能静默吞掉。

## 11. 2D/3D 数据和评估

`OrganSlotManifestDataset` 支持：

- 2D `[3,H,W]` 图像和 `[1,H,W]` raw label；
- Fixfr4 3D `[3,4,H,W]` 图像和 `[1,4,H,W]` raw label；
- 数字 slice index，而不是字符串排序；
- image/mask case 和 slice 一致性检查；
- 固定空间尺寸检查；
- per-sample min-max 或 fixed-255 归一化。

评估侧已实现 binary Dice、calibrated multiclass argmax、按 case 数字 slice 聚合、
Old/New/All DSC 和 forgetting。完整命令行评估器及正式 before/after 表仍待 Gate-D/F。

## 12. 本轮实际完成的代码

核心架构和训练文件：

- `OrganSlotEmbed.py`：slot、bank、binary head；
- `OWT_models_orgslot.py`：共享编码/解码、slot forward、canvas fusion、append/freeze；
- `losses_orgslot.py`：Base、incremental、suppression 和 background loss primitive；
- `engine_pretrain_orgslot.py`：stage-explicit objective、有限步 smoke、验证；
- `main_pretrain_orgslot.py`：synthetic/real、2D/3D、Ours/Sequential FT/Head-only、
  manifest checksum、病例隔离、best/last checkpoint、冻结哈希审计；
- `datasets/orgslot_manifest.py`：真实 CSV adapter；
- `util/label_visibility.py`：严格标签可见性边界；
- `util/slot_tgr.py`：逐样本 slot schedule 和 target；
- `util/checkpoint_orgslot.py`：保存、精确回载、迁移、append、hash；
- `eval_orgslot.py`：head-based metric primitive；
- `tools/split_orgslot_manifest.py`：确定性病例拆分；
- `tools/run_orgslot_tiny_overfit.py`：tiny memorization 与增量冻结审计。

所有实现均在独立 `feature/orgslotbank-v0` worktree 中。原始
`OWT_models.py`、`OrganEmbed.py`、`engine_pretrain.py` 和 `main_pretrain.py` 未修改，
仍作为受控基线。

## 13. 当前验证结果

- CPU 单元测试：32/32 通过；
- 2D 和 Fixfr4 3D synthetic forward/backward：通过；
- 真实 2D `[B,3,224,224]` 单 batch debug-width forward/backward：通过；
- 真实 Fixfr4 `[B,3,4,224,224]` 单 batch debug-width forward/backward：通过；
- Base -> append Liver -> Ours/Sequential FT/Head-only 完整编排 smoke：通过；
- Ours 冻结张量 138 个、Head-only 冻结张量 157 个，训练后 hash 完全一致；
- tiny overfit：loss `1.7362165 -> 0.1302057`，ratio `0.0749939`，organ Dice
  `0.9354208`；
- tiny checkpoint 精确回载：通过；
- incremental batch 仅有 Liver mask，旧参数 hash 不变：通过；
- 700 个病例按 seed=0 拆为 train 630 / validation 70，2D 和 3D 均无病例交集。

## 14. 尚未完成和不能过度宣称的部分

1. 真实单 batch 当前使用 debug-width 模型；正式 ViT-Base GPU 2D/3D smoke 尚未运行。
2. 尚未运行真实 Base 训练、Liver 增量训练和三方法机制表。
3. 尚未运行 suppression/background-plasticity 的正式消融。
4. 尚未形成完整的 case-level before/after 命令行评估和论文表格。
5. 尚未提交任何 OrganSlot 完整 1200-epoch 任务。
6. 配置中的 raw ID 语义来自计划假设：0=background、1=kidney_combined、
   2=spleen、3=pancreas、4=liver。当前仓库未找到独立数据说明验证这份映射；在确认
   前只能把真实数据 smoke 解释为 ID 级工程验证，不能作为器官语义结果。

因此当前准确结论是：OrganSlotBank v0 的结构、隔离约束、训练编排和 debug 数据链路
已通过门控；模型效果和增量抗遗忘优势仍必须由后续受控真实实验回答。
````

<a id="source-5"></a>

## docs/ORGAN_SLOTBANK_EXPERIMENT_LOG.md

SHA256: `b71b74534d190b55145cd256b2dad298aaa67e150c3c59d8b8d050ef025adf90`

````text
# OrganSlotBank v0 Experiment Log

- Branch: `feature/orgslotbank-v0`
- Starting commit: `6cdf9b575edd6845b7518c18d7ef9be9b89508f7`
- Scope: strict visibility, model/loss/checkpoint tests, 2D/3D forward-backward,
  and tiny overfit only. No full training submission.

## 2026-07-24 Gate-A implementation

Worktree: `/gpfs/work/aac/bolinren19/OD_OWT_orgslot`

Implemented:

- stage/class config and a training wrapper that never returns raw/full labels;
- provisional background built only from base-visible old masks;
- explicit fixed-K `OrganSlot` paths, appendable `OrganSlotBank`, binary heads,
  calibration, normalized additive per-sample canvas fusion, and shared decoder;
- per-sample Slot-Aware TGR, base reconstruction/segmentation loss, minimal
  Liver-only incremental loss, detached suppression ablation primitive;
- exact checkpoint save/load, original OWT migration, visible-slot transfer,
  append-from-background, freeze/hash audits, and head-based metrics;
- isolated synthetic smoke entry point. Formal datasets and long training were
  deliberately not started.

Validation commands:

```text
/gpfs/work/aac/bolinren19/.conda/envs/abdpet/bin/python -m unittest discover -s tests -p 'test_orgslot_*.py' -v
OMP_NUM_THREADS=4 /gpfs/work/aac/bolinren19/.conda/envs/abdpet/bin/python -m tools.run_orgslot_tiny_overfit --steps 250 --output-dir Results/OrganSlotBank/_tiny_overfit/gate_a_cpu --device cpu
sbatch slurm/orgslot/smoke/orgslot_gate_a_cpu.sbatch
```

Observed results before final checkpoint-resume test addition:

- 24/24 CPU tests passed, including 2D and Fixfr4-TS1 3D forward/backward and
  an original OWT forward regression. The final count is recorded below.
- tiny overfit: loss `1.7362165 -> 0.1302058` (ratio `0.0749940`), organ Dice
  `0.9354208`, exact checkpoint reload `true`;
- the same audit appended Liver from background, ran incremental backward with
  `visible_masks={liver}` only, and kept all frozen parameter hashes unchanged;
- final current-code Slurm smoke job `1561641`: `COMPLETED`, exit `0:0`, elapsed
  `00:00:27`; base total loss `1.6614242 -> 1.4783453`, reconstruction MSE
  `0.1312498 -> 0.1153881`, with finite head metrics and saved checkpoint.

Boundary/status:

- Gate A wiring is exercised on synthetic data only.
- Gate B/C, real-data manifests, Sequential-FT/Head-only comparisons,
  suppression/background-plasticity ablations, and full training are not run.
- Original `OWT_models.py`, `OrganEmbed.py`, `engine_pretrain.py`, and
  `main_pretrain.py` are unchanged; the regression test confirms legacy OWT
  forward remains runnable.

Final test result after adding base/incremental resume coverage: **25/25 passed**.

Resolved debug events:

- The first tiny-overfit target used non-patch-aligned, sample-varying regions.
  On the intentionally tiny 2x2 canvas it reached loss ratio `0.7845` and Dice
  `0`, so it failed the memorization gate. The test was corrected to the
  canonical repeated, patch-aligned single-pattern target; the success
  threshold was not relaxed, and the corrected run reached ratio `0.0749940`.
- The first Slurm submission was rejected before job creation because the
  account default QOS was invalid. The script was updated to the existing
  project association `account=sifansong`, `partition/qos=cpudebug`; current
  code then completed as job `1561641` with exit `0:0`.

## 2026-07-26 real-data wiring and controlled-baseline update

Added without modifying the legacy OWT files:

- deterministic 2D/Fixfr4 manifest adapter and strict case/slice checks;
- deterministic case-level train/validation split tool that never reads label
  pixels;
- real/synthetic stage-aware runner with manifest checksums, case-overlap
  rejection, 2D/3D model selection, finite-step smoke mode, best/last
  checkpoints, and per-epoch frozen-parameter SHA-256 audits;
- explicit Ours, Sequential-FT, and Head-only trainable scopes;
- automatic detached frozen-old prediction path for the suppression ablation;
- identity/frozen incremental calibration by default, matching the handoff;
- two new behavior tests for baseline scopes and detached old-confidence
  supervision, plus two case-split tests.

Current validation:

- **32/32** OrganSlot CPU tests pass;
- complete synthetic Base checkpoint -> append Liver -> Ours/Sequential-FT/
  Head-only orchestration passes;
- Ours trains 11,969/127,279 debug parameters and preserves 138 frozen tensor
  hashes; Head-only trains 97/127,279 and preserves 157 frozen hashes;
- current-code tiny overfit again reaches loss `1.7362165 -> 0.1302057`, ratio
  `0.0749939`, Dice `0.9354208`, exact reload true, frozen hashes unchanged;
- real 2D `[B,3,224,224]` debug-width one-batch forward/backward passes with
  finite total loss `1.9208560`;
- real Fixfr4 `[B,3,4,224,224]` debug-width one-batch forward/backward passes
  with finite total loss `1.8933843`;
- seed-0 case split contains 630 train and 70 validation cases. 2D records are
  70,560/7,840 and Fixfr4 records are 68,670/7,630.

The real-data checks above are engineering smokes, not accuracy results. They
use debug-width models and only one training batch. The assumed raw-ID semantic
mapping in `owt_legacy_debug.json` still needs independent dataset evidence
before formal organ-named claims or long training.

## 2026-08-15 WORD448 JointSeg Focal v2

目的：在 ROI20 reconstruction baseline 上加入每个 slot 独立的二分类 Focal
监督，替代出现前景全背景塌缩风险的 Dice+BCE tiny 目标；不启用融合后 Loss3。

固定配置：

- 分支 `experiment/orgslot-jointseg-focal-v0`，提交 `7d8343e`；
- XEC账号 `antengcai23`，Slurm account `sifansong`；
- WORD 2D，448输入，ROI20使用384 crop resize到448；
- `L = global L2 + LPIPS + 0.1 * slot focal`；
- `focal_alpha=0.75`，`focal_gamma=2.0`；
- `seg_supervision=all`，`lambda_bg_seg=0.25`；
- 真实keep/drop继续控制reconstruction target和fusion；
- 118800 optimizer updates，有效batch 192，2张A800。

验证：本地53项OrganSlot回归测试和15项Focal/TGR目标测试通过；XEC同环境15项
目标测试通过。逐slot日志新增预测体积、GT体积和阳性像素概率，防止仅凭总loss
误判全背景输出为成功。

任务链：

- smoke `117005`；
- 正式训练 `117006`，依赖smoke成功；
- reconstruction Direct/Indirect评估 `117007`；
- head训练集阈值校准 `117008`；
- 固定0.5及冻结校准阈值head测试 `117009`。
````

<a id="source-6"></a>

## docs/ORGAN_SLOTBANK_EXPERIMENT_SUMMARY_CN.md

SHA256: `2d33f7c552f054bc1017acc2e7443296f3f73f301cf6602d909f3f057aed0f54`

````text
# OrganSlotBank 实验总览

最后更新：2026-08-15 15:17 CST
分支：`feature/orgslotbank-v0`
工作区：`/home/Anteng/OD_OWT_orgslot`
环境：`/home/Anteng/miniconda3/envs/abdpet`
数据根目录：`/mnt/DATA-4/anteng`

本文档维护迁移到本地 GPU 服务器后完成的正式实验、工程验证和当前运行任务。
数值来自各实验的 `resolved_config.json`、`log.txt` 和评估 `results.json`。
`Results/` 中的 checkpoint、TensorBoard event 和可视化不提交到 Git。

## 1. 评价口径

- AutoPET：完整测试集 22,400 张切片、200 个病例。
- WORD：完整测试集 6,990 张切片、24 个病例、8 个前景器官。
- 表中的 case Dice 是“GT 存在病例的 Dice 均值”，不是 slice Dice 或 global Dice。
- Direct：直接保留/调用目标器官响应得到分割。
- Indirect：通过全器官重建与删除目标 slot 后重建的差分得到分割。
- `post` 表示应用既定形态学后处理；不同推理规则不能混在一起比较。

## 2. AutoPET 224 架构消融

三个正式训练使用相同 AutoPET 2D 训练/测试 CSV、224 输入、legacy batch mask、
有效 batch 96 和 1200 epochs。

| 实验 | 主要变量 | 最终训练 loss | Direct | Indirect |
|---|---|---:|---:|---:|
| OWT-compatible legacy-mask | 受控原始架构基线 | 0.02561 | **88.30%** | **84.89%** |
| OrganSlot linear-sqrt | slot 独立路径；画布除以 `sqrt(保留slot数)` | 0.02404 | 87.98% | 81.33% |
| OrganSlot + DiceBCE head | 联合重建和分割头监督 | 0.59723（seg 0.28084） | 使用 head 评价 | 不作为 head 指标 |

结论：

- OrganSlot 的 reconstruction Direct 只比 OWT 低 0.31 个百分点，但 Indirect 低
  3.55 个百分点，说明组合差分能力比直接重建更容易受架构变化影响。
- 联合监督分割头必须使用 logits 评价。完整测试集上：epoch 600 的
  binary-post/argmax-post 分别为 78.64%/78.42%；epoch 1199 为
  79.06%/77.95%。分割头确实学到语义，但没有超过 reconstruction Direct。
- epoch 500、600、1199 均已推理；binary threshold 和互斥 argmax 是两种不同
  推理规则，后续必须分别报告。

结果位置：

- `Results/OrganSlotBank/evaluation/AbdAutoPet_2D/reconstruction_threshold_v1/`
- `Results/OrganSlotBank/evaluation/AbdAutoPet_2D/head_v1/`

## 3. WORD 1x1x2 mm / 448 消融

三组实验共享 native 1x1x2 mm 预处理、center-448 数据、训练/测试 CSV、有效
batch 192 和 118,800 optimizer updates；均在 epoch 802 达到更新预算。

| 实验 | 变量 | Direct-post | Indirect-post |
|---|---|---:|---:|
| control linear-sqrt | 分母 `sqrt(当前保留slot数)` | 64.44% | 6.92% |
| fixed-sqrt K=9 | 分母固定为 `sqrt(9)` | 64.65% | 6.82% |
| ROI20 linear-sqrt | 20% 小器官 ROI crop | **75.20%** | **7.39%** |

统一使用固定 response threshold 0.02 和完整测试集。

ROI20 epoch 802 的 Direct-post 器官 Dice：

| 脾 | 右肾 | 左肾 | 胆囊 | 食管 | 胰腺 | 肝 | 胃 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 90.22% | 89.96% | 89.47% | 40.60% | 58.67% | 63.92% | 93.26% | 75.48% |

结论：

- 固定分母几乎没有改善 Direct，也没有挽救 Indirect，因此 LayerNorm/动态缩放
  不是 Indirect 崩溃的唯一原因。
- ROI20 将 Direct 提高 10.76 个百分点，并改善部分小器官形状，但 Indirect
  仍很低；定位改善与可组合差分是两个问题。
- ROI crop 仅用于训练。测试图像和 GT 使用同一 deterministic center-448
  变换，不存在“输入被 crop、GT 未 crop”造成的评价错位。
- 当前响应常能学习到相似形状但发生空间偏移；形态学后处理无法修复系统性偏移。

结果位置：

- `Results/OrganSlotBank/evaluation/Common8/WORD_2D/control_ckpt802_fixedthr002/results.json`
- `Results/OrganSlotBank/evaluation/Common8/WORD_2D/fixedk9_ckpt802_fixedthr002/results.json`
- `Results/OrganSlotBank/evaluation/Common8/WORD_2D/roi20_ckpt802_fixedthr002/results.json`
- `Results/OrganSlotBank/visualizations/WORD_2D/raw_small_organs_control_vs_roi20/`

## 4. 第三组：严格 head-only WORD 实验

### 实验问题

固定 ROI20 表征后，只训练专用分割读出，能否得到比 reconstruction threshold
更稳定的定位和小器官分割？该设计同时验证联合分割损失是否曾破坏重建表征。

### 训练设计

- 初始化：WORD ROI20 `checkpoint-802.pth`。
- 数据：同一 WORD native-112/center-448 manifests。
- 增强：同一 20% ROI crop，focus raw IDs 为 4/5/6。
- 输入时 9 个 slot 全部可见。
- 冻结：ViT、collector、TGEnc、AHER、slot token、fusion、reconstruction decoder。
- 训练：9 个 `PatchBinaryHead` 和各 slot 的 calibration scale/bias，共 20,763 参数。
- 损失：Dice+BCE；`lambda_seg=1`，背景权重 0.25。
- reconstruction 和 LPIPS 均关闭，forward 跳过 reconstruction decoder。
- AdamW，lr `1e-3`，weight decay 0。
- GPU 0/1；micro-batch 8/GPU；accumulation 12；有效 batch 192。
- 14,800 updates；warmup 740 updates；约 100 epochs。
- tmux：`word448-headprobe`。

### 启动前验证

- OrganSlot 测试：45/45 passed。
- 真实 WORD 12-slice tiny set 覆盖 raw class 0--8；胆囊/食管/胰腺阳性切片
  分别为 6/4/8。
- 100-update tiny overfit：Dice+BCE `2.0015 -> 0.6458`。
- checkpoint audit：54 个张量发生变化，全部属于 head/calibration；其余冻结
  参数逐张量完全相同。

### 当前状态（2026-08-15 15:17 CST）

- 状态：RUNNING，GPU 0/1。
- epoch 0：segmentation loss 1.42607，148 optimizer updates。
- 实测 ROI fraction 0.20196，与 20% 配置一致。
- epoch 1 进行中，阶段平均 segmentation loss 约 0.915。
- reconstruction loss 和 LPIPS 恒为 0，符合实验定义。
- 未发现 NaN、OOM、traceback 或 checkpoint 不兼容。
- 尚无测试 Dice；训练 loss 下降不能当作最终效果提升。

启动脚本：`scripts/orgslot/run_word_112_448_head_probe.sh`

有效运行目录：

`Results/OrganSlotBank/Common8/WORD_2D/OrgSlot_WORD112_input448_roi20_headonly_DiceBCE_from_roi20ckpt802_eb192_u14800`

异常启动记录：第一次启动的 ROI 增强正确，但 head-only 分支把 ROI 日志错误记为
0；在未完成一个 epoch 前停止并修复。第二次因 25743 端口未释放而没有开始训练。
当前有效任务从 epoch 0 在 25744 端口重新开始；这两次无效启动不计入结果。

### 完成标准

训练完成后，在不变的完整 WORD 测试集上同时评价 calibrated argmax 和 independent
binary threshold，分别给出 raw/post、每器官 Dice 和 8 类均值，重点检查胆囊、
食管、胰腺。正式对照为 ROI20 reconstruction Direct-post 75.20%；在完整推理前
不声明第三组优于现有结果。

## 5. 工程边界和后续维护

- 原始 `OWT_models.py`、`OrganEmbed.py`、`engine_pretrain.py`、`main_pretrain.py`
  保持可运行，作为受控基线。
- PSEM 未叠加进以上 OrganSlot 实验，避免一次改变多个变量。
- `Results/` 是本机实验工件，不进入 Git；复现依赖 launcher、resolved config、
  数据 manifest checksum 和源 checkpoint 路径。
- 每次正式评估后，应更新本文件中的状态、checkpoint、完整测试集规模、评价模式、
  per-organ 指标和结论，并保留失败/无效实验的原因。
- 早期 Gate-A/Gate-B、严格标签隔离和增量学习工程记录见
  `docs/ORGAN_SLOTBANK_EXPERIMENT_LOG.md`。
````

<a id="source-7"></a>

## docs/ORGAN_SLOTBANK_IMPLEMENTATION_HANDOFF.md

SHA256: `6fcb0bed9560aee9359726ab6051c297e770e61167169dcc36522cf409cf4959`

````text
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
````

<a id="source-8"></a>

## docs/ORGSLOT_112_NATIVE_CROP448_EXPERIMENT_PLAN.md

SHA256: `debad717f08cfb1e7a300a0abf4deadaeda2773ec3e994eb5785a4e02a4db081`

````text
# OrganSlotBank WORD 1x1x2: 448 vs 448 + 20% Small-Organ Crop

## Locked experiment

Offline preprocessing performs only RAS reorientation, joint image/label
resampling to 1x1x2 mm, HU clipping to [-175,250], normalization, and native
matrix slice export. There is no intermediate 448 canvas and no offline
resize. All paths are rooted at `/mnt/DATA-4/anteng`.

WORD is used first. Its 120 cases have resampled XY sizes from 490x490 through
501x501, so every case supports the declared physical crop windows without
padding. BTCV is excluded from this first experiment because 23/30 cases are
smaller than 448 after resampling and require a separate protocol.

### Arm A: 448 control

```text
1x1x2 native matrix -> deterministic center crop 448x448 -> OrganSlot img_size=448
```

The output represents exactly 1 mm per pixel in XY.

### Arm B: 448 + 20% small-organ ROI

Each requested sample is selected as follows:

- 80%: identical global center crop 448x448;
- 20%: choose gallbladder (ID4), esophagus (ID5), or pancreas (ID6)
  uniformly; draw a positive slice from that class pool; crop a 384x384
  physical window around its bounding-box center with up to 10% bounded
  center jitter; resize image/label to 448x448.

The ROI branch enlarges linear size by 448/384=1.167 and area by about 1.36,
not the stronger 1.96 area factor of a 320 crop. The crop is guaranteed to
retain the entire focus-organ bounding box. It keeps all image context and all
other labels within the selected window; it never masks everything except the
focus organ.

The focus OrganSlot is forced to remain in the TGR keep mask for ROI samples.
Other slots follow the selected ordinary mask schedule. TensorBoard records
the realized ROI fraction, per-focus-class fraction, and retained-slot count.

Validation and test always use the deterministic center 448 crop. Labels are
never used to choose evaluation crops.

## Fairness invariants

Both arms share the exact case split, native 1x1x2 files, model variant,
initialization, loss weights, ordinary mask schedule, seed, per-GPU
micro-batch, accumulation, effective batch, learning rate, optimizer updates,
warmup updates, and evaluation. The intentional Arm-B changes are the 20%
positive-slice mixture, 384 ROI crop, and focus-slot retention.

At patch size16, 448 produces 28x28=784 spatial patch tokens versus196 at
224. Before formal runs, sweep per-GPU micro-batch 2,4,8. Select the largest
safe value with memory headroom and use accumulation to target effective
two-GPU batch192 (micro-batch8 implies accumulation12).

## Required gates

1. Current AutoPET jobs and queued inference finish; one architecture is
   selected and frozen for both arms.
2. Full WORD geometry/label validation and immutable manifests complete.
3. ROI index checksum matches the training manifest and contains nonempty
   pools for IDs4,5,6.
4. Image/label overlays are inspected before/after resampling and after both
   crop modes.
5. Unit tests prove paired 2D/Fixfr4 crop, nearest label resize, strict label
   isolation, positive focus selection, and focus-slot retention.
6. Audit per-class realized sampling, crop clipping, and label IDs.
7. Deterministic 448 forward/backward passes.
8. Four-to-eight real samples overfit with augmentation disabled.
9. Micro-batch memory sweep and short two-GPU Arm-B smoke pass without
   NaN/OOM and produce checkpoint, provenance, logs, and TensorBoard events.
10. Launch Arm A on GPUs0,1 and Arm B on GPUs2,3 only after all gates pass.
````

<a id="source-9"></a>

## docs/ORGSLOT_ARMB_3D_WORD070_ROI20_CN.md

SHA256: `b1e34db7905c60f5af14b1a554407e971029dfbe7ac006373090429923575787`

````text
# OrganSlot Arm B 3D：WORD 0.7 spacing + ROI20

## 目的

把已完成的 2D Arm B 改成短体块 3D 训练，验证相邻轴位层面的上下文是否能改善
小器官连续性和边界，同时保持数据、slot 数、重建目标、分割损失和 ROI20 策略不变。

## 与 2D Arm B 的关系

| 项目 | 2D Arm B | 3D Arm B |
|---|---:|---:|
| spacing | 0.7 x 0.7 x 2.0 mm | 相同 |
| xy 输入 | 448 x 448 | 相同 |
| ROI | 20% 概率，384 裁剪后缩放到 448 | 相同 xy 策略 |
| z 输入 | 单层 | 连续 4 层（约 8 mm） |
| patch embedding | Conv2d，16 x 16 | Conv3d，1 x 16 x 16 |
| head | multiscale Conv2d | multiscale Conv3d |
| loss | small-organ，lambda_seg=0.01 | 相同，按 3D slab 计算 |
| batch | 192 slices/update | 48 slabs/update，约 192 slices/update |
| actual LR | 7.5e-5 | 7.5e-5（显式固定） |
| optimizer updates | 118800 | 118800 |

## 3D ROI20

ROI 索引仍来自阳性 2D anchor 切片。抽中 ROI 样本后，以 anchor 为中心读取连续
4 层；在这 4 层的目标器官 mask 上取 z 向并集，再确定共同的 384 x 384 xy
裁剪框。相同裁剪作用于所有层，最后统一缩放到 448 x 448。病例边界处平移窗口，
不做跨病例填充，也不越界。

这不是完整病例 3D crop，而是可由 A800 训练、并尽量接近原 Arm B 协议的
short-slab volumetric 3D 实验。

## 关键配置

- `dimension=3D`
- `fix_frame=4`
- `temp_stride=1`
- `slot_head_type=multiscale_conv`
- `slot_head_channels=128`
- `seg_loss_type=small_organ`
- Tversky FP/FN = 0.3/0.7
- Balanced Focal weight = 0.5
- hard negative = top 2%
- negative slab weight = 0.1
- `lambda_seg=0.01`
- 4 x A800，micro batch 2/GPU，accumulation 6
- actual LR = 7.5e-5，与 2D Arm B 一致

## 正式评估

训练完成后必须按病例重组完整 3D 预测，报告八类 case-level 3D Dice、IoU、
precision、recall、NSD 和 HD95（mm）。2D slice 指标只作为错误分析，不作为主结果。
````

<a id="source-10"></a>

## docs/ORGSLOT_ARM_E_CROSS_ATTENTION_CN.md

SHA256: `da21d51ee8b6a30f408de0b7169a07de00ad9a6ef916bf0282c3fbdc509e6a39`

````text
# Arm E + Cross-Attention（2026-09-08）

## 实验目的与范围
在Arm E上增加一个共享的query-to-image cross-attention block，测试query在
最终mask匹配前再次读取空间特征是否有帮助。不是替换AHER、不是新增P2、
不是SAM two-way decoder；不增加query数量，不新增绝对位置PE。

统一开关：`--slot_head_type arm_e_multiscale_query --query_refinement cross_attn`。
环境变量为QUERY_REFINEMENT。默认none完全保留旧E路径与state dict；
仅2D Arm E支持本开关。checkpoint args用于统一head/reconstruction evaluator
恢复结构并严格加载，不允许把原E checkpoint伪装成已训练attention checkpoint。

## 实际结构
CT的P4/P8与ViT P16融合 → P4 pixel feature（448输入时112×112）。
原20个TGEnc tokens经LN、均值、投影和器官身份向量产生一个query。
P4平均池化2倍 → 56×56 memory（注意：不是直接使用P8 lateral feature）。
4-head cross-attention，通道128、无dropout：
q + Attention(LN(q), LN(memory)) → 残差FFN（128→256→128）→ refined q。
refined q与原P4 feature做归一化点积 → 112 logits → 插值回448。
attention计算局部FP32，整体训练BF16。attention模块初始化使用fork_rng，
避免改变后续原有organ slot参数初始化；新模块本身不是零初始化。

长宽112/448=1/4，面积1/16；P4每位置仍有128通道，不能等同“信息只剩1/16”。
边界可能受P4上生成logit再插值的限制。真实P2 skip（224）应作为独立实验，
不能与本次attention一起加。224同通道特征元素数是112的4倍。

## 锁定配置及可比性
WORD spacing0.7/0.7/2，448/384，ROI0.2，small-organ loss，
retained，lambda_seg0.01，lambda_bg_seg0，原Tversky/Focal/top2%/negative0.1，
batch192，seed0，118800更新、warmup5940、blr1e-4、AdamW weight decay0.05。
本组BF16、clip1、finite每次更新、每25epoch保存。先同分配2次真实DDP更新验证，
再从seed0正式训练，不从预检查checkpoint续跑。

重要：历史E任务2892114使用旧FP16/scaler流程，因此与本组并非严格单变量比较。
严格attention结论需同一代码、BF16/clip等设置的query_refinement=none对照；
目前只新增用户要求的一组，不擅自多提交一组baseline。

## 热图与交互检查
工具：`python -m tools.export_query_cross_attention`。
严格加载训练后checkpoint，默认固定选择测试CSV中每个目标器官最先出现的
两张阳性切片（4胆囊、5食管、6胰腺），不按预测质量挑选。
每例导出PNG/PDF、NPZ和manifest：
- CT、GT、head均值attention / uniform（1代表均匀水平）；
- normal / uniform-attention / bypass整个refinement的概率图；
- NPZ保存4个head的原始56×56归一化权重，供逐head审查；
- GT attention mass、GT面积占比、二者之比；
- 三种预测的固定0.5 slice Dice和相对normal的平均概率变化；
- checkpoint及CSV哈希、epoch、dataset index与严格加载报告。

uniform保留value、输出投影、FFN，只取消位置选择；bypass连同FFN一起跳过，
两者含义不同。所有干预保持pixel features和原query不变。
注意力图不是分割概率，也不是因果证明；关注器官外上下文不必然错误。
本模块没有新增绝对坐标PE。同步重排keys/values可以不改变聚合结果，因此
空间热图不能证明“理解了解剖绝对坐标”。干预结果也可能是分布外响应，
要和完整病例级指标一起判断，不能靠几个样例宣称提升。

## 运行与评估
训练脚本：slurm/orgslot/train/orgslot_word07072_arm_e_crossattn_bolin_sip.sbatch。
评估脚本：slurm/orgslot/eval/orgslot_word07072_arm_e_crossattn_bolin_sip.sbatch。
运行使用独立detached快照，通过EXPERIMENT_WORKDIR/EVAL_WORKDIR传入，
固定代码后不再原地改动。评估以afterok跟随训练，先导出固定样例热图，
再运行原24病例统一评估。最终无真实训练checkpoint前，不提供“已证明有效”图。

## 验证
已提交：bolinren19/SIP，account angelosstefanidis，QoS 8a800。
训练2920724（4 A800、24 CPU、192 GB）；热图与统一评估2920726
（1 A800、10 CPU、128 GB），afterok:2920724。
运行快照：`.worktrees/orgslot_arm_e_crossattn`，detached commit `4cbd096`。
此次未推送GitHub。5项cross-attention测试、26项整模型回归、2项Arm E回归通过。
PNG/PDF合成渲染测试通过，但图片查看受沙箱限制，未完成视觉排版验收。
真实GPU预检查和训练后热图仍等待调度，不能记为已通过/已产生。

已增加：共享参数初始化一致性、严格state dict roundtrip、
attention逐head空间归一化、normal/uniform/bypass干预差异、
query对memory依赖、部分器官head参与的整模型finite backward。
渲染测试仅用合成未训练输入，并明确标注非实验证据。
GPU预检查通过与否必须读取任务日志，CPU测试不能代替长程稳定性审核。
````

<a id="source-11"></a>

## docs/ORGSLOT_AUTOPET_MAE_TRANSFER_CN.md

SHA256: `355667781dc9ad0e14a2b81b2c7f0917a5028885261eda18484cb8cb6bad1b14`

````text
# AutoPET MAE → WORD OrganSlot Arm B 迁移实验

## 研究问题

在当前最优的 WORD 2D Arm B 训练协议不变时，使用 AutoPET CT 进行纯图像 MAE 预训练，是否能通过共享 image encoder 和共享 reconstruction decoder 改善 WORD Common8，特别是胆囊、食管和胰腺？

## 固定的 WORD 配置

- spacing：0.7×0.7×2.0 mm；
- input/global crop：448；ROI crop：384；ROI probability：0.2；
- head：`multiscale_conv`，128 channels；
- retained-slot supervision；
- small-organ loss：Tversky FP/FN=0.3/0.7，Balanced Focal权重0.5，hard negatives top 2%，negative slice weight 0.1；
- `lambda_seg=0.01`，`lambda_bg_seg=0`；
- effective batch 192，118800 optimizer updates，seed 0。

现有 scratch Arm B 是直接对照。完整对比固定为三臂，新增实验只改变初始化范围：

| Arm | 初始化 | 状态/用途 |
|---|---|---|
| B0 scratch | 全部随机初始化 | 已完成的 Arm B checkpoint-802，作为零预训练基线 |
| B1 encoder-only | AutoPET MAE `patch_embed + cls + 6层LA encoder + encoder_norm` | 隔离 encoder 预训练收益 |
| B2 encoder+decoder | B1全部内容 + 共享 `decoder_blocks/norm/pred` | 判断共享重建 decoder 是否提供额外收益 |

三臂中的 Collector、TGEnc、AHER 和 `multiscale_conv` segmentation head 均从相同随机种子初始化。B0不重复训练，但必须和B1/B2一起使用同一套3D病例级 evaluator 重评。

B0来自提交`b748e59`，本分支起点为`5226496`。两提交间针对该训练主线的差异仅为可选Loss-v3、resume和附加日志；模型、数据与既有small-organ loss代码未改。三臂固定`positive_roi_loss_weight=0`，该可选项不进入总损失或梯度，因此可以复用B0。

跨账号数据一致性由`tools/validate_word070_dataset_identity.py`在启动前检查：train 28586张、test 6990张；规范化清单SHA256分别为`ffbe5e...1fc8c`和`871a31...a9f1`，ROI核心SHA256为`3a2596...fb8b`。规范化会移除`/WORD/`之前的账号根路径。

## AutoPET MAE

AutoPET XEC 训练清单包含78400张224×224 RGB JPEG，三个通道是相同灰度复制。第一版保持原224图像，不放大到448；WORD位置编码按448重新生成，因此不迁移位置编码。

MAE配置：patch16、LA encoder 6层/768维、decoder 8层/768维、mask ratio 0.75、masked-patch pixel MSE、BF16、clip-grad 1.0、effective batch 192、100000 updates、5000 warmup updates。按患者从AutoPET training划分90%/10% train/validation，不使用AutoPET test和器官标签。

## 权重映射

加载：`patch_embed`、`cls_token`、6个encoder blocks、`encoder_norm`、共享`decoder_blocks`、`decoder_norm`和`decoder_pred`。

不加载：MAE位置编码、`mask_token`、`decoder_embed`、optimizer；WORD所有slot的Collector、TGEnc、AHER、segmentation head和校准参数保持随机初始化。加载器对每个映射张量做名称和shape验证，并保存`mae_initial_checkpoint_load.json`。

## 结果判定

主要比较B0/B1/B2的同协议3D病例级结果：八类平均Dice、小器官平均Dice、逐器官Dice、precision/recall、NSD、HD95、预测/GT体积比以及固定/验证集校准阈值。`B1-B0`量化encoder预训练收益，`B2-B1`量化共享decoder的额外贡献；不能只汇报B2相对B0。

单seed只能作为探索性证据。若B1或B2平均Dice提高至少0.5点且小器官平均提高至少1点、同时大器官下降不超过0.5点，再补两个seed。

## 执行链

1. `autopet_mae_transfer_smoke_sifan_xec.sbatch`：真实数据、完整模型做2次更新并验证checkpoint；
2. `autopet_mae_transfer_pretrain_bolin_sip.sbatch`：正式MAE当前选择bolinren19/SIP的`sifansong/4a800`，其调度预估早于XEC；`autopet_mae_transfer_pretrain_sifan_xec.sbatch`保留为备用；
3. `orgslot_word07072_armb_autopet_mae_encoder_sifan_xec.sbatch`：B1 encoder-only；
4. `orgslot_word07072_armb_autopet_mae_sifan_xec.sbatch`：B2 encoder+decoder 的 sifansong/XEC 版本；
5. `orgslot_word07072_armb_autopet_mae_encoder_decoder_anteng_xec.sbatch`：B2 的 antengcai23/XEC 并行版本，必须先复制并校验同一 MAE checkpoint；
6. 两个新 checkpoint-802 完成后，将B0/B1/B2使用相同3D病例级重建/head评估脚本重评。

MAE预训练当前运行在`bolinren19/SIP`，因为该账号已有完整AutoPET，且4卡调度预估比XEC早；它不承担后续WORD训练。MAE完成后仅传输模型checkpoint并记录SHA256，再由`sifansong/XEC`运行B1、`antengcai23/XEC`运行B2。不得把三个账号或SIP/XEC的绝对路径互换。
````

<a id="source-12"></a>

## docs/ORGSLOT_CONFIGURATION_CONTRACT_CN.md

SHA256: `9ea64e2bfa65dc21b7b766f01655280140be460f79acb3ac66512362818affa5`

````text
# OrganSlot 架构、数值精度与提交配置统一规范

更新时间：2026-09-08。适用分支：`feature/orgslot`。
本文件是新实验的配置与审核规范，不表示所有历史脚本已经自动整改。

## 1. 唯一开发入口与运行快照

- 统一开发目录：`OD_OWT/.worktrees/orgslot_integrated`。
- Arm A/B/C/D/E、2D/3D、PE、MAE迁移通过参数选择，不再为每种模型复制实现。
- 历史目录可能处于 detached HEAD；归档tag用于追溯，不能视作实时同步副本。
- 新任务应使用经过核验的独立代码快照，并记录绝对路径、commit与未提交diff。
- 若修复必须应用到旧实验目录，必须分别核验统一入口和实际运行目录；只合并Git分支不算部署完成。
- 正在运行的任务不要原地替换代码。需要变更时建立新快照、新输出目录及新任务。

当前例外：重提的2D PE任务 `2919868` 仍明确使用
`.worktrees/orgslot_pixel_pe`，相关修复也已同步到统一开发目录。
这不意味着其他旧目录已经同步。未提交diff也属于实际实验版本，不能只记录commit。

## 2. 四层配置必须同时核验

| 层级 | 负责内容 | 实际证据 |
|---|---|---|
| 模型 | head、query、PE、维度、token数量 | 模型构建参数、checkpoint args及state dict |
| 优化与数值 | loss、AMP、裁剪、LR、batch、更新预算、初始化/续训 | resolved_config.json、训练日志 |
| 启动与部署 | worktree、环境、数据路径、脚本环境变量、最终CLI | Slurm脚本、command.txt、启动日志 |
| 调度与评估 | 登录用户、Slurm account、QoS、资源、依赖、评估协议 | scontrol、评估结果与完整性标记 |

**模型支持BF16 ≠ 启动任务选择了BF16；分支包含修复 ≠ 运行目录包含修复。**

参数不是简单的“命令行永远覆盖一切”，实际顺序为：

`sbatch导出环境 → 作业脚本export → shell启动器构造CLI → Python解析 → 模型/优化器`

- 作业脚本 `export AMP_DTYPE=bf16` 会覆盖继承环境中的同名变量。
- 启动器 `AMP_DTYPE=${AMP_DTYPE:-fp16}` 在变量未设置或为空时使用FP16。
- 启动器显式传入 `--amp_dtype` 后，Python默认值不再决定该次运行精度。
- 当前通用启动器和Python入口的AMP默认值**仍是FP16**，没有全局改成BF16。
- 当前 `PIXEL_PE` 默认none、`SEGMENTATION_UNIT` 默认slab，是历史兼容行为，不代表新实验推荐值。
- Shell变量只有被启动器转换成CLI参数才会生效；不能凭变量名称推断生效。

审查入口：
[通用启动器](../scripts/orgslot/run_word_112_448_a100.sh)、
[Python入口](../main_pretrain_orgslot_common8_a100.py)、
[训练循环](../engine_pretrain_orgslot_common8_a100.py)。

## 3. 架构参数统一表

| 控制项 | 环境变量 / CLI | 含义与边界 |
|---|---|---|
| 维度 | DIMENSION / --dimension | 2D或3D；3D还需核验FIX_FRAME、TEMP_STRIDE |
| Arm A | SLOT_HEAD_TYPE=linear | AHER canvas后线性分割头 |
| Arm B | SLOT_HEAD_TYPE=multiscale_conv | AHER canvas后多尺度卷积分割头 |
| Arm C | SLOT_HEAD_TYPE=query_dot | 共享pixel decoder；organ tokens汇总为一个query |
| Arm D | SLOT_HEAD_TYPE=multi_query_dot | 共享pixel decoder；保留多个query，log-mean-exp汇总 |
| Arm E | SLOT_HEAD_TYPE=arm_e_multiscale_query | 输入图像P4/P8与ViT P16融合；单query；目前仅2D |
| Pixel PE | PIXEL_PE / --pixel_pe | none、spatial、temporal、spatiotemporal；当前仅C/D |
| 监督单位 | SEGMENTATION_UNIT / --segmentation_unit | slab保留历史3D语义；slice按切片判断阳性/空切片；2D不受此开关影响 |
| MAE初始化 | MAE_INIT_CHECKPOINT、MAE_INIT_SCOPE | encoder或encoder_decoder；必须记录来源checkpoint |
| 训练续跑 | RESUME_CHECKPOINT / --resume | 与MAE初始化互斥；核验epoch、optimizer和更新计数连续性 |

2D只允许none/spatial，不存在独立的切片时间轴。3D可以分别比较四种PE。
PE添加在pixel projection之后、pixel decoder之前；不改OrganCollector的时间压缩，
也不等同于给direct head接入完整OWT reconstruction Transformer decoder。

`token_factor=20` 与 `ROI probability=0.2` 是两个不同参数。
“ROI20”指20%的ROI采样概率，不表示20个token。

## 4. 可比实验必须锁定的项目

2D WORD head对照使用：spacing 0.7×0.7×2.0、输入448、ROI384、
ROI probability 0.2、token factor20、seed0、有效batch192、
118800次optimizer更新。3D不能照搬2D有效batch，应另行记录slab数量、
每slab切片数及更新预算。

分割损失：`small_organ`；retained监督；lambda_seg=0.01；
lambda_bg_seg=0；Tversky FP/FN=0.3/0.7、eps=1e-6；
Balanced Focal权重0.5、alpha=0.75、gamma=2；
hard negatives=top2%、negative slice weight=0.1。
不能把“背景像素负样本”误认为独立background slot监督，两者不同。

总训练目标也不是只有Dice+Focal：当前路线仍有L2 reconstruction及LPIPS，
并加权叠加segmentation loss。核验 `loss_version=L2-LPIPS`、
`lambda_lpips=1`、`fusion_mode=linear_sqrt`、reference count=9，
以及旧Loss3相关权重是否为0。

比较表中必须增加以下列：
**AMP dtype、clip_grad、LR、有效batch、更新次数、监督单位、初始化来源**。
只保持head和loss相同不足以保证公平对照。
本次BF16 PE结果与历史FP16无PE结果只能作带精度差异的比较；
严格PE消融还需同精度无PE对照，不能把差异全部解释成PE收益。

## 5. 新A800任务的数值规范

以下是新任务必须显式配置的要求，不是所有旧任务的当前状态：

```bash
export AMP_DTYPE=bf16
export CLIP_GRAD=1.0
export FINITE_CHECK_INTERVAL=1
```

还需显式设置DIMENSION、SLOT_HEAD_TYPE、PIXEL_PE、SEGMENTATION_UNIT、
loss和采样参数。不得从其他实验的shell环境隐式继承MAE/RESUME配置。

| 保护措施 | 能做什么 | 不能保证什么 |
|---|---|---|
| BF16 autocast | 比FP16更宽的指数范围；当前入口不启用FP16 GradScaler | 不能防止除零、非法运算或所有反向不稳定 |
| 梯度裁剪1.0 | 在梯度有限时限制全局范数 | 不能将已产生的NaN/Inf修复为有效梯度 |
| strict finite checks | 异常时停止，避免污染optimizer/参数 | 不等于训练能够自动恢复 |
| FP32局部fallback | 处理特定算子兼容性或精度问题 | 仅对已覆盖的算子/路径有效，不代表全模型FP32 |
| checkpoint | 提供可追溯恢复点 | 文件存在不等于参数有限、配置正确或可成功续训 |

当前scaler流程：backward → unscale → strict clip → step → scaler update。
FP16下如果strict clip先抛错，就到不了GradScaler正常的skip/backoff。
不要通过关闭 `error_if_nonfinite`、静默跳过异常或nan_to_num掩盖问题。

## 6. 2026-09-08事故记录与修复范围

任务2910849在epoch64 step233出现非有限梯度。启动日志和完整CLI均显示
`amp_dtype=fp16`；原因是PE作业脚本没有显式设置AMP_DTYPE。
首个异常算子未被旧日志记录，**FP16溢出仍是机制假设，不是已复现的算子级根因**。

已实施：
- 2D PE脚本固定BF16、clip1.0、finite interval1，并检查GPU支持BF16。
- backward/update错误报告包含异常梯度参数名称和scaler状态。
- checkpoint从每100改为每25个epoch。
- 同一分配内先进行两次更新DDP预检查，单独输出，通过后从头正式训练。
- CPU侧5项回归测试通过；实际GPU预检查和跨epoch64验证仍需看新任务日志。
- 新训练2919868；评估2919869依赖afterok:2919868；旧失效评估2910855已取消。

详见[数值修复记录](PIXEL_PE_NUMERICS_20260908.md)。
两次更新只能验证启动链路，不能证明长期稳定，不能提前标记“NaN彻底解决”。

## 7. 三账号、两集群：路径与额度分离

每个任务必须记录：
登录用户、集群、Slurm account、QoS、partition、GPU型号/数量、
环境Python绝对路径、worktree、训练/评估CSV、ROI索引、LPIPS权重、
预训练/续训checkpoint、输出目录。

- bolinren19/SIP已用数据根为 `/gpfs/work/aac/bolinren19/2026-07/DATA`。
- sifansong与antengcai23的XEC工作目录分别在各自用户根下；数据路径须现场读取其配置，
  不能通过替换用户名猜测存在性。
- 登录用户名不等于Slurm account；QoS名也不是GPU类型。
- SIP与XEC Job ID属于不同命名空间；不能用SIP sacct核验XEC任务。
- 训练与评估必须各自核验环境和路径，不能只验证训练目录。
- README不保存密码、私钥或token；访问配置使用本地HPC skill。

## 8. 提交前及运行后审核清单

### 提交前

1. 确认统一分支修复与实际运行快照一致，记录commit和dirty diff。
2. 阅读作业脚本及其调用的启动器，检查显式参数与默认回退。
3. 检查真实CSV、ROI索引、权重、Python环境及独立输出目录；不得覆盖旧结果。
4. 核对每卡batch×GPU数×accum_iter；3D同时记录slab单位。
5. 运行配置/PE回归测试及bash语法检查：
   `python -m unittest discover -s tests -p 'test_pixel_pe*.py' -v`。
6. 先分配内验证再正式训练；不要截断ROI池制造无效smoke。
7. sbatch返回Job ID后，通过scontrol确认account、QoS、资源、WorkDir及依赖。

### 启动后

核验同一输出目录中的 `resolved_config.json`、`command.txt`、启动日志：
AMP、head、PE、监督单位、spacing、loss、batch、初始化必须与实验表一致。
不以README、脚本名称或环境变量的“预期值”替代真实解析值。
检查首个epoch有限性、ROI路径、weighted_seg≈0.01×seg_loss；
checkpoint保存后检查args、epoch、可加载性及参数有限性。
本次PE额外要求稳定跨过epoch64，之后检查checkpoint75/100。

### 评估及依赖

- sbatch保存提交时的batch脚本；修改本地脚本不会改写已提交副本。
  但脚本调用的外部Python/shell通常在运行时读取，所以代码快照也必须冻结。
- 旧afterok上游失败后不能自动改为依赖新任务。重新建立训练—评估链并记录新ID。
- evaluator从checkpoint args重建head/PE，严格加载权重；核验八类、病例数和完整性。
- 固定0.5与校准阈值结果分别列出；历史训练集校准协议必须明确标注，
  新阈值选择优先用验证集，不能在测试集挑最佳阈值。
- 3D必须报告病例体积级指标，注明slab重叠融合、后处理及head/reconstruction路径。

## 9. 尚未自动化的部分

当前并没有全仓库唯一的强制配置schema，也没有检查所有sbatch的通用验证器。
现有新测试只防止2D PE精度入口回退；其他实验仍必须逐任务审核。
本规范不修改通用FP16默认、不改变正在运行的实验、不自动重提任务。
后续若实现全局强制校验，应另行测试历史兼容性，不能仅凭文档声称已经完成。

````

<a id="source-13"></a>

## docs/ORGSLOT_WORD070_ARM_E_MULTISCALE_PIXEL.md

SHA256: `71d8fc335d66ffd65f13f5adaa376d277e3297fb9dd0b12551eb2c07d6fd8601`

````text
# OrganSlot Arm E: high-resolution multiscale pixel decoder

## Research question

Arm C and Arm D obtain similar WORD 2D results even though Arm D keeps 20
independent token queries. Arm E therefore tests a different bottleneck:
whether the final 28x28 ViT feature lacks the spatial detail required by small
organs.

The controlled change is:

    Arm C: ViT P16 -> upsampled pixel feature -> single organ query dot product
    Arm E: ViT P16 + image P8 + image P4 -> fused pixel feature
                                        -> same single organ query dot product

The convolutional path never predicts an organ mask by itself. The raw mask is
always the normalized dot product between the shared pixel feature and an
OrganSlot query.

## Exact architecture

For a 448x448 input:

    image -> lightweight Conv/GN/GELU stem -> P4  [B,128,112,112]
                                        -> P8  [B,128, 56, 56]
    ViT final patch tokens                 -> P16 [B,128, 28, 28]

    P16 upsample + projected P8 -> depthwise/pointwise refinement
         upsample + projected P4 -> depthwise/pointwise refinement
                                -> pixel feature [B,128,112,112]

    20 OrganSlot tokens -> Arm C mean aggregation/projection
                        -> one query [B,128]

    FP32 normalized dot product -> [B,1,112,112] -> bilinear 448x448

Arm E adds 256,320 parameters. The formal model has 186,799,826 total
parameters; the Arm E shared pixel decoder has 491,840 parameters.

## Fixed experimental controls

- WORD 2D, 8 foreground organs and one background slot
- spacing 0.7 x 0.7 x 2.0 mm
- input 448, ROI crop 384, ROI probability 0.2
- token factor 20, one aggregated organ query
- retained segmentation supervision, background segmentation weight 0
- MSE + LPIPS + 0.01 x small-organ segmentation loss
- Tversky FP/FN 0.3/0.7
- balanced focal weight 0.5, focal alpha/gamma 0.75/2
- top-2% hard negatives, negative-slice weight 0.1
- effective batch 192, 118800 optimizer updates, seed 0

No query-count, loss, sampling, ViT patch-size, threshold, or reconstruction
change is part of Arm E v1.

## Verification

Local tests:

- Arm C/D/E model regression: 20/20 passed
- retained/small-organ loss regression: 20/20 passed
- Arm E checkpoint compatibility: passed
- B=2 at 448x448, P4/P8/P16 and K=1/3/8 shapes: passed
- stem, P16 projection, OrganCollector, and query gradients: non-zero
- zero query: raw mask is exactly zero
- shuffled query: raw mask changes
- reconstruction with shared weights: Arm C and Arm E are bitwise equal

Single-GPU real-data smoke:

- SIP job 2814754, 1x A800, completed
- 4 optimizer updates completed without NaN/Inf
- peak allocated GPU memory: 12.76 GiB
- mean segmentation loss: 1.0118
- mean weighted segmentation loss: 0.0101
- observed ROI fraction: 0.1875
- checkpoint and formal smoke validator passed

Two-GPU DDP:

- job 2814794 exposed a launcher import-path error before model execution
- launcher fixed in commit 75c3b45
- replacement SIP job 2864374 requests 2x A800 under 8a800
- the synthetic DDP preflight deliberately gives rank 0 kidney and rank 1
  spleen, then performs three optimizer steps with fixed collective schema

## Slurm entry points

- single GPU smoke:
  slurm/orgslot/smoke/orgslot_word07072_arm_e_single_bolin_sip.sbatch
- two GPU DDP smoke:
  slurm/orgslot/smoke/orgslot_word07072_arm_e_ddp_bolin_sip.sbatch
- formal four GPU training:
  slurm/orgslot/train/orgslot_word07072_arm_e_bolin_sip.sbatch

Formal training must not be submitted until job 2864374 passes.

## Success criteria

Compare checkpoint-802 to Arm C with the same 24 test cases, fixed threshold
0.5, and identical postprocessing. Report all eight organs, overall foreground
Dice, and the mean of gallbladder, esophagus, and pancreas. Arm E supports the
hypothesis only if the small organs improve coherently rather than trading one
organ against another.
````

<a id="source-14"></a>

## docs/ORGSLOT_WORD070_MULTIQUERY_EXPERIMENT.md

SHA256: `7e6974940ee9cc442b9bcd2137b74d4598e62a0d2fda3ac91e920bcc035fe429`

````text
# WORD 0.7 ROI20 Multi-Query Pixel-Mask Experiment

## Question

Does preserving all 20 TGEnc token queries improve organ-mask readout over the
single pooled-query Arm C while preserving the same pixel and reconstruction paths?

## Architecture

The reconstruction path is unchanged:

```text
z -> OrganCollector_s -> TGEnc_s -> AHER_s -> canvas_s -> fusion -> reconstruction
```

The new segmentation path is:

```text
z -> shared pixel decoder -> P(x)
TGEnc_s token k -> shared query projection + slot identity -> q_{s,k}
part_logit_{s,k}(x) = sqrt(D) * cosine(P(x), q_{s,k})
mask_logit_s(x) = logmeanexp_k(part_logit_{s,k}(x))
```

The shared pixel decoder upsamples the final 28x28 ViT patch grid to a 112x112
128-channel feature map. All 20 token queries produce part-response maps. Log-mean-exp is used as a
smooth maximum with no token-count logit bias; identical token queries exactly
recover the single-query score. Per-slot logits are bilinearly resized to
448x448.
Segmentation gradients update the shared ViT, shared pixel/query decoder,
OrganCollector, TGEnc, and slot identity, but do not pass through AHER.

## Locked comparison

This Arm D changes only the token-to-query aggregation relative to Arm C.

| Setting | Arm A | Arm B | Arm C | Arm D |
|---|---|---|---|---|
| head | linear | multiscale_conv | pooled query-dot | multi-query-dot |
| head input | AHER canvas | AHER canvas | mean of 20 TGEnc tokens + final ViT pixels | 20 independent TGEnc queries + same pixels |
| query aggregation | n/a | n/a | mean before projection | log-mean-exp after 20 part maps |
| spacing | 0.7x0.7x2.0 | same | same | same |
| input / ROI crop | 448 / 384 | same | same | same |
| ROI probability | 0.2 | same | same | same |
| supervision | retained | same | same | same |
| loss | small_organ | same | same | same |
| lambda_seg | 0.01 | same | same | same |
| effective batch | 192 | same | same | same |
| updates / warmup | 118800 / 5940 | same | same | same |
| optimizer / base LR / WD | AdamW / 1e-4 / 0.05 | same | same | same |
| seed | 0 | same | same | same |

No diversity loss is added in the first controlled run. Token specialization or
collapse must be measured after training instead of changing head and loss at
the same time.

## Run registry

- Branch: `experiment/orgslot-querymask-multiquery`
- Implementation commit: `606c7ed`
- Account/cluster: `bolinren19 / SIP`
- Smoke: Job `2548701`, 2xA800, 16 CPU, 128 GB
- Formal train: Job `2548702`, 4xA800, 24 CPU, 192 GB, `afterok:2548701`
- Formal output: checkpoint-802 after 118,800 optimizer updates

## Acceptance gates

1. Unit tests: finite forward/backward, exact mask shapes, dropped-row zeros,
   shared pixel and slot-query gradients, no AHER segmentation gradient.
2. Real 0.7 ROI20 smoke: eight optimizer updates, finite reconstruction,
   LPIPS, small-organ metrics, positive focus-organ coverage, checkpoint keys.
3. Formal training is submitted with `afterok` on the smoke job.
4. Final evaluation must use the same validation/test inference protocol as
   Arm A/B, including fixed 0.5 head threshold as the primary result.
````

<a id="source-15"></a>

## docs/ORGSLOT_WORD070_QUERYMASK_3D_EXPERIMENT_CN.md

SHA256: `15c965a457b094a32c7afaf4063a84867d3e52531115dc460caef017346b83c8`

````text
# OrganSlot 3D Arm C/D：WORD 0.7 spacing + ROI20

## 研究问题

在4层short-slab 3D协议下，直接保留ViT空间patch特征的query-mask读出是否仍优于
AHER canvas上的multiscale head；保留20个独立token queries是否进一步优于把20个
tokens汇总成一个器官query？

## 严格对照

| Arm | 唯一主要变量 | 分割空间来源 | 器官query |
|---|---|---|---|
| 3D B | `multiscale_conv` | AHER canvas | 无动态query |
| 3D C | `query_dot` | 共享3D ViT pixel decoder | 20 tokens均值后形成1个query |
| 3D D | `multi_query_dot` | 与3D C相同 | 20个独立queries，以无token数偏置的log-mean-exp聚合 |

其余配置固定：0.7×0.7×2.0 mm、448输入、384 ROI crop、ROI概率0.2、连续4层、
stride1、20 tokens/organ、small-organ loss、`lambda_seg=0.01`、BF16、clip-grad1.0、
有效batch 48 slabs（约192 slices）、actual LR 7.5e-5、118800 optimizer updates和seed0。

3D C/D均从随机初始化开始，不加载2D C/D checkpoint，避免把架构收益和2D预训练收益
混在一起。训练集和测试集由账号无关的规范化SHA256门禁验证。

## 3D实现

共享pixel decoder把ViT输出reshape为`[B,C,T,H,W]`，只在xy方向逐级上采样；
每一级3×3×3 depthwise convolution交换相邻层信息。所有3D trilinear resize在FP32
执行后转回原dtype，以兼容XEC PyTorch 1.13的BF16算子集合。

3D C：

[
q_s=W_q\operatorname{Mean}_k(\operatorname{LN}(T_{s,k}))+e_s,
\qquad
l_s(t,x,y)=\sqrt{D}\,\hat P(t,x,y)^\top\hat q_s.
]

3D D：

[
l_{s,k}(t,x,y)=\sqrt{D}\,\hat P(t,x,y)^\top\hat q_{s,k},
]

[
l_s=\operatorname{logsumexp}_k(l_{s,k})-\log K.
]

减去`log K`消除token数量导致的固定logit偏置；当20个queries完全相同时，3D D与
3D C输出严格一致。

## 执行账号

- 3D C：`sifansong/XEC`，账号根目录
  `/gpfs/work/aac/sifansong`；
- 3D D：`antengcai23/XEC`，账号根目录
  `/gpfs/work/aac/antengcai23`。

两个账号分别使用自己的Python、数据、LPIPS和结果目录，不交换绝对路径。分开提交是为了
利用各自的4-GPU QOS并行排队，不代表数据协议不同。

## 验收与判定

训练验收：

1. 4张A800可见；
2. 数据SHA256门禁通过；
3. 3D query/pixel decoder均有非零有限梯度；
4. 无NaN/Inf，`weighted_segmentation_loss≈0.01×segmentation_loss`；
5. 完成118800 updates并生成checkpoint-199。

正式结果只使用真正病例级3D评估：24病例、八类、固定0.5阈值，报告Dice、IoU、
precision、recall、NSD、HD95和预测/GT体积比。主要比较为3D C-B和3D D-C；在三臂
全部完成前不做最终架构结论。
````

<a id="source-16"></a>

## docs/ORGSLOT_WORD070_QUERYMASK_EXPERIMENT.md

SHA256: `e7219d7a5bd9b813f438251c377ee6c2944b5d2508c63e9263bbc072577be135`

````text
# WORD 0.7 ROI20 Shared Pixel Query-Mask Experiment

## Question

Does bypassing the AHER canvas for segmentation improve organ-mask readout while
preserving the existing OrganSlot reconstruction path?

## Architecture

The reconstruction path is unchanged:

```text
z -> OrganCollector_s -> TGEnc_s -> AHER_s -> canvas_s -> fusion -> reconstruction
```

The new segmentation path is:

```text
z -> shared pixel decoder -> P(x)
TGEnc_s tokens -> shared query projection + slot identity -> q_s
mask_logit_s(x) = sqrt(D) * cosine(P(x), q_s)
```

The shared pixel decoder upsamples the final 28x28 ViT patch grid to a 112x112
128-channel feature map. Per-slot logits are bilinearly resized to 448x448.
Segmentation gradients update the shared ViT, shared pixel/query decoder,
OrganCollector, TGEnc, and slot identity, but do not pass through AHER.

## Locked comparison

Arm C changes only the segmentation readout relative to the existing 0.7
spacing Arm A/B runs.

| Setting | Arm A | Arm B | Arm C |
|---|---|---|---|
| head | linear | multiscale_conv on AHER canvas | shared pixel query-dot |
| spacing | 0.7x0.7x2.0 | same | same |
| input / ROI crop | 448 / 384 | same | same |
| ROI probability | 0.2 | same | same |
| supervision | retained | same | same |
| loss | small_organ | same | same |
| lambda_seg | 0.01 | same | same |
| Tversky FP/FN | 0.3/0.7 | same | same |
| Balanced Focal weight | 0.5 | same | same |
| hard-negative ratio | 0.02 | same | same |
| negative-slice weight | 0.1 | same | same |
| effective batch | 192 | same | same |
| updates / warmup | 118800 / 5940 | same | same |
| optimizer / base LR / WD | AdamW / 1e-4 / 0.05 | same | same |
| seed | 0 | same | same |

## Acceptance gates

1. Unit tests: finite forward/backward, exact mask shapes, dropped-row zeros,
   shared pixel and slot-query gradients, no AHER segmentation gradient.
2. Real 0.7 ROI20 smoke: eight optimizer updates, finite reconstruction,
   LPIPS, small-organ metrics, positive focus-organ coverage, checkpoint keys.
3. Formal training is submitted with `afterok` on the smoke job.
4. Final evaluation must use the same validation/test inference protocol as
   Arm A/B, including fixed 0.5 head threshold as the primary result.
````

<a id="source-17"></a>

## docs/ORGSLOT_WORD070_SMALL_ORGAN_LOSS_CN.md

SHA256: `4a8453d84eb2966341558d00519f96b8e917f9e82a4242e25799caf7dc5d1ebe`

````text
# WORD 0.7-space 当前训练损失说明

## 1. 适用实验

本文描述当前 WORD 2D OrganSlotBank 小器官实验使用的训练损失。对应配置为：

- 数据 spacing：`0.7 × 0.7 × 2 mm`；
- 输入尺寸：`448 × 448`；
- segmentation head：`multiscale_conv`；
- segmentation supervision：仅监督 TGR 当前保留的 organ slots（`retained`）；
- ROI sampling probability：`0.2`，与对照实验一致；
- 不使用额外 organ weighting；
- 不使用 PCGrad；
- background slot 的 segmentation weight 为 `0`。

正式训练入口为：

```text
slurm/orgslot/train/orgslot_word07072_roi20_retained_multiconv_smallorgan_xec.sbatch
```

## 2. 总损失

当前仍然是单阶段联合训练，同时优化重建任务和分割任务：

\[
\mathcal L_{total}
=\mathcal L_{rec}
+\lambda_{lpips}\mathcal L_{lpips}
+\lambda_{seg}\mathcal L_{seg}.
\]

当前权重为：

\[
\lambda_{lpips}=1,\qquad \lambda_{seg}=0.01.
\]

因此实际配置是：

\[
\boxed{
\mathcal L_{total}
=\mathcal L_{MSE}
+\mathcal L_{LPIPS}
+0.01\mathcal L_{small\text{-}organ}
}
\]

其中：

- `MSE` 约束像素级重建；
- `LPIPS` 约束感知结构和纹理；
- `small-organ` loss 监督 retained organ slots 的 binary masks。

重建 target 不是无条件复制原始图像，而是由当前 TGR keep mask 和可见器官 mask 构造的重建目标。分割 loss 不改变该重建 target。

## 3. 单个器官、单张切片的分割损失

每个 organ slot 都是一个独立二分类问题。设：

- \(z(x)\)：像素 \(x\) 的预测 logit；
- \(p(x)=\sigma(z(x))\)：前景概率；
- \(y(x)\in\{0,1\}\)：该器官的 GT mask。

根据当前切片是否包含该器官，采用不同的损失。

### 3.1 阳性切片

如果：

\[
\sum_x y(x)>0,
\]

则使用：

\[
\boxed{
\mathcal L_{+}
=\mathcal L_{Tversky}
+0.5\left(
\mathcal L_{Focal}^{fg}
+\mathcal L_{Focal}^{hard\text{-}bg}
\right)
}
\]

#### Tversky loss

\[
TP=\sum_x p(x)y(x),
\]

\[
FP=\sum_x p(x)(1-y(x)),
\]

\[
FN=\sum_x (1-p(x))y(x).
\]

当前定义：

\[
\mathcal L_{Tversky}
=1-
\frac{TP+\epsilon}
{TP+0.3FP+0.7FN+\epsilon},
\qquad \epsilon=10^{-6}.
\]

这里 FN 的系数 `0.7` 大于 FP 的 `0.3`，表示训练初期更重视减少漏分。对于胆囊、食管和胰腺等小器官，漏掉少量前景像素就可能破坏主体结构，因此当前设置偏向提高召回率。

#### 前景 Focal

前景像素单独平均：

\[
\mathcal L_{Focal}^{fg}
=\frac{1}{|\Omega_{fg}|}
\sum_{x\in\Omega_{fg}}
0.75(1-p(x))^2\operatorname{BCE}(z(x),1).
\]

这样前景贡献不再被大量背景像素稀释。

#### 困难背景 Focal

对每个背景像素计算：

\[
h(x)=0.25p(x)^2\operatorname{BCE}(z(x),0).
\]

只选择损失最大的前 `2%` 背景像素：

\[
\mathcal L_{Focal}^{hard\text{-}bg}
=\operatorname{mean}\left(\operatorname{TopK}_{2\%}\{h(x)\}\right).
\]

这部分主要处理器官边界附近、相邻器官区域和模型高置信度误报位置，而不是让大量容易背景主导优化。

### 3.2 阴性切片

如果当前切片完全不含该器官：

\[
\sum_x y(x)=0,
\]

则不计算 Tversky，也不使用全图 BCE 平均。当前损失为：

\[
\boxed{
\mathcal L_{-}
=0.1\cdot
\operatorname{mean}\left(
\operatorname{TopK}_{2\%}
\{\operatorname{BCE}(z(x),0)\}
\right)
}
\]

它只惩罚模型最容易误报的 `2%` 像素，并将整个阴性切片分支乘以 `0.1`。这可以保留对假阳性的约束，同时防止数量众多的阴性切片压倒稀少的阳性切片。

Top-K 数量使用向上取整，并保证至少选择一个像素。

## 4. 一个 batch 内如何聚合

对器官 \(c\)，阳性切片和阴性切片分别求平均：

\[
\mathcal L_c
=\operatorname{mean}_{i\in P_c}(\mathcal L_{+,i})
+\operatorname{mean}_{j\in N_c}(\mathcal L_{-,j}).
\]

如果当前 batch 没有某一类切片，对应项记为 0。关键点是先分别平均，再相加，而不是把所有切片直接混在一起平均。因此增加阴性切片数量不会自动稀释阳性切片的梯度。

随后，对当前 batch 中实际受到监督的非背景 organ slots 等权平均，得到：

\[
\mathcal L_{seg}=\operatorname{mean}_{c\in\mathcal C_{supervised}}\mathcal L_c.
\]

当前没有额外的器官级权重。background slot 虽可记录诊断指标，但其 segmentation weight 为 0，不进入 \(\mathcal L_{seg}\)。

## 5. 为什么总权重使用 0.01

旧 focal loss 对全图像素直接求均值，数值通常较小；当前损失包含一个接近 \([0,1]\) 范围的 Tversky 项，并把前景和困难背景分别平均，因此原始数值自然约为 1，不能沿用旧 loss 的数值尺度直接比较。

已通过的两卡 smoke 中：

- 原始 segmentation loss：约 `1.0407`；
- 加权 segmentation term：约 `0.0104`；
- reconstruction MSE：约 `0.2457`；
- LPIPS：约 `0.8087`。

所以 `lambda_seg=0.01` 的作用是让分割梯度产生明确影响，同时避免在实验开始时突然压过重建目标。这里比较的是乘权重后的 loss/梯度影响，而不是仅比较原始 loss 数值。

## 6. 数值稳定性与多卡语义

- small-organ loss 内部将 logits 转为 FP32 后计算；
- 每一步检查 total loss 是否为有限值，出现 NaN/Inf 会立即终止；
- 所有 DDP ranks 使用固定的 per-organ 诊断字段，避免集合通信次数不一致；
- 没有监督样本的器官指标记录为 0，并配合 `supervised_samples=0` 解读；
- dropped slots 不计算 segmentation gradient；
- 当前仍是普通加权求和，不使用 PCGrad 或两阶段训练。

## 7. 需要重点观察的训练指标

总量指标：

- `reconstruction_loss`；
- `p_loss`；
- `segmentation_loss`；
- `weighted_segmentation_loss`；
- `loss`。

每个器官分别记录：

- `seg_<organ>_positive_samples`；
- `seg_<organ>_negative_samples`；
- `seg_<organ>_tversky_loss`；
- `seg_<organ>_positive_focal_loss`；
- `seg_<organ>_hard_negative_focal_loss`；
- `seg_<organ>_empty_negative_loss`；
- `seg_<organ>_positive_slice_predicted_fraction`；
- `seg_<organ>_empty_slice_predicted_fraction`；
- `seg_<organ>_target_fraction`。

判断是否改善小器官时，不能只看总 loss 是否下降，还要检查：

1. 阳性切片预测比例是否从接近全图前景逐渐收缩到合理区域；
2. 阴性切片预测比例是否下降，说明假阳性受到抑制；
3. Tversky loss 是否下降，说明召回和区域重叠在改善；
4. 前景 Focal 是否下降，说明小器官前景置信度在提高；
5. 困难背景和空切片 loss 是否下降，说明边界误报和无器官误报在减少；
6. 最终仍须用验证集 Dice、HD95/NSD 和逐器官结果判断效果，训练 loss 不能替代分割评估。

## 8. 代码对应关系

- 总损失组合与训练监控：`engine_pretrain_orgslot_common8_a100.py`；
- Tversky、balanced focal、hard-negative mining 和 batch 聚合：`losses_orgslot.py`；
- 参数定义和合法性检查：`main_pretrain_orgslot_common8_a100.py`；
- 0.7-space 正式实验参数：`slurm/orgslot/train/orgslot_word07072_roi20_retained_multiconv_smallorgan_xec.sbatch`；
- smoke 验证器：`tools/validate_orgslot_jointseg_smoke.py`。
````

<a id="source-18"></a>

## docs/ORGSLOT_WORD448_JOINT_SLOT_SEG_V1.md

SHA256: `44fb283eea3bcbdad2c403cbd68abe13748afdb65d708f12f2ed077a5b1b0c00`

````text
# OrganSlot WORD448 Joint Slot Segmentation v1

## 1. 目标与假设

ROI20-only 的 Direct post Dice 为 75.20%，ROI20+融合后 Loss3 为 65.18%。
v1 不再对融合重建结果增加器官 ROI loss，而是用每个 slot 自己的 binary head
约束该 slot canvas 必须能独立解码对应器官。

主目标为：

\[
L=L_{global\ L2}+L_{LPIPS}+0.01L_{slot\ Dice+BCE}.
\]

该损失增强“对应器官可从对应 slot 解码”，但不预先宣称其他 slot 完全不含
该器官。是否真正解决职责混合，要联合观察 head Dice 与 reconstruction Direct。

## 2. 主实验配置

| 参数 | 值 |
|---|---|
| 数据 | WORD 2D native-resampled，448 deterministic eval crop |
| 训练增强 | ROI20，384 crop resize 到 448 |
| ROI probability | 0.20 |
| fusion | `linear_sqrt`，reference count 9 |
| reconstruction | Global L2 + LPIPS，`lambda_lpips=1` |
| fused Loss3 | 不存在；最新版分支已删除该实现 |
| slot segmentation | Dice+BCE，`lambda_seg=0.01` |
| background segmentation | `lambda_bg_seg=0.25` |
| segmentation supervision | `all` |
| reconstruction/fusion mask | 原始真实随机 keep/drop |
| effective batch | 192 |
| optimizer updates | 118800 |

`all` 只改变 segmentation loss 的监督选择：被 drop 的 slot 不参与融合，但它的
binary head、AHER、TGEnc 和 Collector 仍接受自己器官的监督。重建 target 不变，
不增加第二次 decoder forward，Direct/Indirect 协议不变。

## 3. 分阶段实验

### Gate A：单元测试

必须证明：

1. `all` 时 dropped slot 的 head、slot 路径和共享 backbone 获得非零梯度；
2. `retained` 保持旧行为，dropped head 梯度为零；
3. segmentation supervision mode 不改变 reconstruction target；
4. segmentation-only backward 不直接更新 reconstruction decoder；
5. optimizer 能完成一步并改变 dropped slot head 参数；
6. 旧 checkpoint、fusion、head-only probe 测试仍通过。

当前结果：59 项全量测试通过。

### Gate B：deterministic tiny overfit

- 8 个真实训练样本；
- deterministic center crop，关闭随机图像增强；
- 每个样本固定 keep mask；
- 1 GPU，有效 batch 4，400 updates；
- 同时优化 L2、LPIPS 和 `0.01 * segmentation`。

验收：total、reconstruction、segmentation 三项末窗口均低于初始窗口的 90%，
所有 9 个 slot 均有监督且产出 checkpoint。

### Gate C：真实 ROI forward/backward smoke

- 真实 ROI index；
- ROI probability 暂设为 1.0，仅为强制覆盖 ROI 路径；
- 2 GPU，4 updates；
- 检查 loss 有限、每 slot loss/监督数存在、加权项精确为
  `0.01 * segmentation_loss`、无旧 Loss3 指标、产出 checkpoint。

### Gate D：正式训练

只有 Gate B、C 均通过后才提交。正式 ROI probability 恢复为 0.20，其他训练
预算与 ROI20-only 完全一致。

## 4. 正式对照与后续消融

| ROI20 | slot segmentation | supervision | 用途 | 状态 |
|---:|---:|---|---|---|
| 0 | 0 | - | control，Direct 64.44% | 已完成 |
| 1 | 0 | - | ROI20 baseline，Direct 75.20% | 已完成 |
| 1 | fused Loss3 | - | 失败组合，Direct 65.18% | 已完成 |
| 1 | 0.01 | all | v1 主实验 | Gate B 前，不提交正式 |
| 1 | 0.01 | retained | 判断 dropped-slot supervision 的贡献 | 主实验后按需 |
| 0 | 0.01 | all | 判断 ROI 与 seg 的交互 | 只在主实验结果含糊时 |

不把 segmentation-only 从头训练作为 reconstruction 正式对照，因为它不训练
shared reconstruction decoder。已有 frozen head-only probe 仅作为表示诊断。

若主实验明显提升，再以短预算比较 `0.01/0.025/0.05`，不同时提交三组完整训练。

## 5. 评估

### Reconstruction

- 固定 threshold 0.02；
- Direct raw/post；
- Indirect raw/post；
- 每器官 case Dice/global Dice；
- 每类预测体积/GT体积比。

### Binary heads

- 固定 sigmoid threshold 0.5 是正式主结果；
- raw/post Dice；
- 每器官 case/global Dice；
- 每类预测体积/GT体积比。

当前数据只有 Train/Test，没有独立 validation。禁止在 Test 上选阈值。为诊断纯
校准漂移，可以在最多 6000 个训练样本上选择每类 threshold，再冻结到 Test；
该结果明确标注为 `train_calibration`，不能冒充独立 validation 结果，也不能替代
固定 0.5 主结果。

## 6. 判读

1. head Dice 与 Direct 都提高：slot identity 辅助监督成功。
2. head Dice 高、Direct 不提高：瓶颈位于 slot canvas→fusion→shared decoder。
3. Direct 提高、Indirect 下降：共享 encoder/其他 slot 可能重新读取 dropped organ，
   优先比较 `all` 与 `retained`，不要增加 fused ROI loss。
4. head Dice 低且预测体积接近零：大量空切片负监督可能压制稀有器官，先检查
   阳性/空标签统计，再决定是否调整 sampling 或 loss。
5. 固定 0.5 差、train-calibrated threshold 好：主要是 head calibration drift，
   不是 reconstruction Direct 已解决。

## 7. 文件与任务

核心代码：

- `engine_pretrain_orgslot_common8_a100.py`
- `main_pretrain_orgslot_common8_a100.py`
- `losses_orgslot.py`
- `scripts/orgslot/run_word_112_448_a100.sh`

验证/评估：

- `tools/validate_orgslot_jointseg_tiny.py`
- `tools/validate_orgslot_jointseg_smoke.py`
- `tools/eval_common8_orgslot_heads.py`
- `tools/eval_common8_orgslot_reconstruction_threshold.py`

Job IDs：等待 Gate B 提交后填写。

## 8. Focal v2 正式实验

Dice+BCE tiny 的数值门禁虽然通过，但最终 background loss 约为 0.0036，
八个前景 slot loss 均接近 1.0，存在全背景塌缩风险。因此 v2 保留联合重建、
ROI20 和 all-slot supervision，只替换独立 binary head 的损失：

\[
L=L_{global\ L2}+L_{LPIPS}+0.1L_{slot\ focal}.
\]

固定配置：

- `seg_loss_type=focal`；
- `focal_alpha=0.75`，其中 alpha 是前景正类权重；
- `focal_gamma=2.0`；
- `lambda_seg=0.1`；
- `seg_supervision=all`；
- `organ_roi_probability=0.20`；
- fused Loss3 关闭。

Focal 的原始数值约为 Dice+BCE 的十分之一，因此使用 0.1 而不是直接沿用
0.01。该设置预期使加权辅助项约为 0.004，仍小于旧 Dice+BCE 联合实验的
0.01--0.02，避免分割目标主导重建。

训练日志额外记录每个 slot 的 `predicted_fraction`、`target_fraction` 和
`positive_probability`。smoke 必须验证这些字段有限，正式判断还必须依赖固定
0.5 threshold 的逐 head Dice/预测体积，以及相同 0.02 threshold 的
reconstruction Direct/Indirect Dice。

XEC `antengcai23` 任务链（2026-08-15）：

| 阶段 | Job ID | 依赖/状态 |
|---|---:|---|
| ROI100 Focal smoke | 117005 | 已提交，等待调度 |
| ROI20正式训练 | 117006 | `afterok:117005` |
| reconstruction评估 | 117007 | `afterok:117006` |
| head train calibration | 117008 | `afterok:117006` |
| head test | 117009 | `afterok:117008` |

代码分支：`experiment/orgslot-jointseg-focal-v0`；提交：`7d8343e`。
````

<a id="source-19"></a>

## docs/ORGSLOT_WORD448_RETAINED_MULTISCALE_FOCAL_V2.md

SHA256: `3be6490f8629bf3a2e4a00b1bc592e266efe069dba7f89935883add0572a4e41`

````text
# WORD448 OrganSlot retained multiscale Focal v2

## Objective

Test whether a stronger local segmentation decoder can improve small-organ
readout without breaking OrganSlot retained-query semantics.

## Locked configuration

- dataset: WORD 2D native-resample, 448 input;
- augmentation: ROI20 (384 crop resized to 448), focus classes 4/5/6;
- reconstruction: global L2 + LPIPS;
- segmentation: independent binary sigmoid focal loss;
- focal alpha/gamma: 0.75/2.0;
- segmentation weight: 0.1;
- background segmentation weight: 0;
- supervision: retained slots only;
- head: four-stage multiscale depthwise-separable convolution decoder;
- fusion: linear_sqrt;
- no fused-image Loss3.

## Execution semantics

The shared ViT encodes the complete image batch once. For each slot, only rows
whose slot is retained are gathered into Collector, slot TGEnc, AHER and the
binary head. Results are scattered back to batch layout for fusion and loss.
Dropped rows therefore consume no slot-path/head compute and receive no
segmentation gradient. Legacy all-slot supervision remains available explicitly
and computes the union of reconstruction-retained and head-supervised rows.

## Gates

1. Unit tests verify 2D shape, finite backward, sparse Collector/head batch
   sizes, zero dropped logits, and no-head reconstruction mode.
2. Real ROI100 smoke uses the formal 8 images/GPU microbatch for four optimizer
   updates and must produce finite reconstruction, LPIPS and focal metrics.
3. Formal ROI20 training starts only after the smoke succeeds.
4. Formal evaluation reports fixed-threshold reconstruction Direct/Indirect,
   fixed 0.5 binary-head Dice, and validation-only calibrated head thresholds.

## Job registry

Submitted from login account antengcai23 on XEC using Slurm project account
sifansong:

| Stage | Job ID | Dependency at submission |
|---|---:|---|
| real ROI100 smoke | 119640 | Priority |
| formal ROI20 training | 119641 | afterok:119640 |
| reconstruction Direct/Indirect | 119642 | afterok:119641 |
| head train-calibration | 119643 | afterok:119641 |
| head test | 119644 | afterok:119643 |

Submission audit:

- An initial account-name attempt was rejected before creating jobs.
- Jobs 119609-119613 were cancelled. The smoke file had accidentally been
  overwritten by the formal file because both had the same basename in a
  temporary patch-generation directory; 119609 ran 8 minutes of the formal
  configuration before cancellation and is not an experiment result.
- Commit ad98843 restores a distinct four-update ROI100 smoke and adds shell
  assertions for MAX_UPDATES=4, ACCUM_ITER=1, ROI probability 1.0 and a unique
  smoke tag before launching Python.


## Smoke acceptance and formal start

Corrected smoke 119640 completed on xgpua800n6 in 28 seconds with exit code 0.
It used two A800 GPUs, microbatch 8/GPU, accumulation 1, ROI probability 1.0
and exactly four optimizer updates. The real-data validator passed and saved
checkpoint-0.pth. Averaged finite metrics were:

- total loss: 1.0856;
- reconstruction L2: 0.2633;
- LPIPS: 0.8160;
- focal segmentation: 0.0632;
- weighted focal term: 0.0063;
- peak GPU memory: about 16.3 GiB on the first logged step.

All three focus classes were exercised and FP32 probability monitoring remained
finite. The afterok dependency released formal job 119641, which started on
xgpua800n6. Its first logged batches are finite, use ROI probability 0.2,
effective batch 192 and peak memory about 18.3 GiB. Jobs 119642-119644 remain
blocked behind the formal training/evaluation dependencies.
````

<a id="source-20"></a>

## docs/ORGSLOT_WORD448_ROI20_LOSS3_EXPERIMENT.md

SHA256: `63aefe557217b9aab5a481a2487cc9b0ed0ff1926c11a2cfe89c39a61507bc12`

````text
# OrganSlot WORD 448 ROI20 + LossBalance-v3

## Objective

Test whether OrganSlotBank benefits from combining the 448 native-resolution
protocol and 20% small-organ ROI sampling with the positive-state-only
LossBalance-v3 objective. This is a new combination experiment, not a
loss-matched replacement for the existing OrganSlot 448 control.

## Locked configuration

- Dataset: WORD Common8 2D, 96 training and 24 test cases.
- Preprocessing: RAS, 1x1x2 mm, native resampled matrix, fixed-255 intensity.
- Input: global 448 center crop; 20% ROI branch uses 384 crop resized to 448.
- ROI focus classes: gallbladder 4, esophagus 5, pancreas 6.
- Architecture: OrganSlotBank, 20 tokens/class, slot TG depth 1.
- Fusion: `linear_sqrt`.
- Mask: `legacy_batch`; an ROI focus slot is always retained.
- Segmentation head supervision is disabled: `lambda_seg=0.0`.
- Reconstruction: global L2 + LPIPS.
- Loss-v3: present-and-kept foreground ROI-L2 only, weight 0.25.
- Removed foreground ROI-L2 is monitoring only.
- Frequency parameters: alpha 0.5, maximum weight ratio 4.0.
- Two A800 GPUs, micro-batch 8/GPU, accumulation 12, effective batch 192.
- Optimizer update budget: 118800; warmup updates: 5940.

Natural 448-center-crop positive-slice counts for raw IDs 1..8:

```text
4392 4875 5070 1631 3827 3753 7327 5556
```

These counts are frozen before ROI mixing so ROI20 remains an explicit data
intervention rather than silently redefining Loss-v3 weights.

## Isolation and implementation

Loss-v3 consumes only `visible_masks` returned by `StrictVisibilityDataset`;
the raw multi-class label is never exposed at the training boundary. Positive
and removed states use the exact OrganSlot `[B,9]` keep mask after focus-slot
retention. No trainable model parameter or checkpoint key is added. When the
Loss-v3 coefficient is zero, its per-mask computation is skipped.

## Validation

- Four manifests: 70,792 records checked, zero missing referenced files.
- 29 focused regression tests passed.
- Real WORD gallbladder ROI 448 forward/backward passed with finite backbone
  gradients and nonzero positive supervision.
- Frozen LPIPS state extracted from the completed WORD OWT checkpoint; SHA256
  `a99ef0ef436727d809148f5e9cbc09c1f0218079020918b0b1cc5167493c4287`.

## Jobs

Current retry implementation commit: `2231cd0`. The retry uses
`sifansong/4a800` and two typed A800 GPUs.

| Job | Stage | Status at submission |
|---:|---|---|
| 1651723 | Initial ROI100 Loss-v3 smoke | FAILED before Python: no devices |
| 1651724 | Initial ROI20 Loss-v3 formal | CANCELLED: DependencyNeverSatisfied |
| 1655555 | Corrected seg=0 ROI100 smoke + validator | PENDING (Priority), estimated 2026-08-11 09:33 |
| 1655556 | Corrected seg=0 ROI20 formal training | PENDING, `afterok:1655555` |

The full job uses an `afterok` dependency on a two-GPU smoke whose validator
requires visible CUDA devices, finite global/LPIPS/positive ROI metrics,
exercised ROI samples, and a checkpoint. The retry excludes `gpua800n2` and
`gpua800n6`.

## GPU visibility incident and retry

Jobs 1651723/1651724 are invalid: smoke 1651723 failed before Python with
`No devices were found` on `gpua800n2`; formal 1651724 became
`DependencyNeverSatisfied` and was cancelled. The initial scripts also used
`lambda_seg=1.0`, which did not isolate the planned pure ROI20 + Loss-v3
combination. Retry scripts use `lambda_seg=0.0`, typed A800 GRES, exclude the
two nodes implicated in CUDA invisibility, and require both PyTorch CUDA
discovery and a real CUDA tensor operation before training.

## 2026-08-11 second retry

Smoke `1655555` also started without an allocated GPU, this time on
`gpua800n1`, and failed at the CUDA preflight before any model forward.
Dependent formal job `1655556` was cancelled.

The replacement chain is:

| Stage | Job | Dependency | Initial state |
|---|---:|---|---|
| ROI100 smoke + validator | 1658405 | none | PENDING (Priority) |
| ROI20 formal training | 1658406 | afterok:1658405 | PENDING (Dependency) |

The `sbatch` command line explicitly requests `gpu:a800:2` and excludes
`gpua800n1`, `gpua800n2`, and `gpua800n6`. Post-submission inspection confirmed
`ReqTRES=...gres/gpu=2`, `TresPerNode=gres:gpu:a800:2`, and
`ExcNodeList=gpua800n[1-2,6]` for both jobs.
````

<a id="source-21"></a>

## docs/ORGSLOT_WORD_COMMON8_SOTA_CN.md

SHA256: `32257a4ac336a7ca5cb2293c0367d3ca6c525e0ba0d2248a59b49d3d562a4458`

````text
# OrganSlot WORD Common8 八类结果与项目内 SOTA

更新时间：2026-08-28

## 排名口径

本表只比较 WORD 同一完整测试集上的 OrganSlot 实验：24 个病例、6990 张切片，
八类顺序固定为脾脏、右肾、左肾、胆囊、食管、胰腺、肝脏、胃。指标为
case-level Dice presence mean，表中乘以 100。

主排名要求同一个 checkpoint、同一种输出和一套预先固定的后处理，不允许按测试集
为不同器官临时挑选不同方法。小器官均值固定为胆囊、食管、胰腺三类的算术平均。
这里的“SOTA”仅指本项目内已完成且协议匹配的实验，不代表公开 WORD benchmark
或 PCDD 的外部 SOTA。

## 已完成结果：统一输出排名

| 排名 | 实验 / 输出 | 脾脏 | 右肾 | 左肾 | 胆囊 | 食管 | 胰腺 | 肝脏 | 胃 | 小器官均值 | 八类均值 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1** | **Arm B joint small-organ / reconstruction Direct-post** | 92.74 | 91.48 | 91.92 | **53.74** | **69.48** | **70.35** | **94.58** | 79.28 | **64.53** | **80.45** |
| 2 | Arm B head / train-calibrated selected-post | **92.98** | **91.73** | **92.38** | 51.45 | 68.75 | 68.71 | 94.41 | 82.47 | 62.97 | 80.36 |
| 3 | Arm B head / fixed-0.5 binary-post | 92.62 | 91.04 | 91.80 | 49.72 | 67.86 | 69.03 | 94.41 | **82.65** | 62.20 | 79.89 |
| 4 | 0.7 spacing ROI20 / no segmentation head / reconstruction Direct-post | 92.11 | 91.03 | 91.51 | 40.54 | 67.32 | 69.53 | 93.84 | 79.40 | 59.13 | 78.16 |
| 5 | 1.0 spacing ROI20 / no segmentation head / reconstruction Direct-post | 90.22 | 89.96 | 89.47 | 40.60 | 58.67 | 63.92 | 93.26 | 75.48 | 54.40 | 75.20 |

Arm B 的 train-calibrated 阈值为
`[0.7, 0.8, 0.8, 0.9, 0.9, 0.8, 0.5, 0.6]`，来自训练集 calibration，
并在测试前固定；因此可报告，但固定 0.5 仍是 head 的主结果。reconstruction 使用
固定阈值 0.02。post-processing 统一使用 minimum component size 20 和 opening
radius 1。

## 相对 0.7 ROI20 无 head 基线的增量

| 实验 / 输出 | 脾脏 | 右肾 | 左肾 | 胆囊 | 食管 | 胰腺 | 肝脏 | 胃 | 小器官均值 | 八类均值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Arm B reconstruction | +0.63 | +0.45 | +0.40 | **+13.19** | +2.16 | +0.83 | +0.74 | -0.12 | **+5.39** | **+2.29** |
| Arm B head fixed 0.5 | +0.52 | +0.02 | +0.28 | +9.18 | +0.54 | -0.50 | +0.57 | +3.25 | +3.07 | +1.73 |
| Arm B head train-calibrated | +0.87 | +0.70 | +0.87 | +10.91 | +1.43 | -0.81 | +0.57 | +3.07 | +3.84 | +2.20 |
| 0.7 spacing 相对 1.0 spacing | +1.89 | +1.07 | +2.04 | -0.06 | **+8.65** | **+5.61** | +0.58 | +3.92 | +4.73 | +2.96 |

主要结论：

1. 0.7 spacing 的主要收益在食管、胰腺和胃，证明更高平面分辨率对细长/低对比结构有效；
   胆囊并未仅靠 spacing 改善。
2. Arm B joint small-organ training 让 reconstruction 平均提升 2.29，胆囊提升
   13.19，说明分割监督确实改善了共享表示与 AHER canvas。
3. Arm B 的 multiscale head 固定阈值结果比同 checkpoint reconstruction 低 0.56；
   小器官均值低 2.32。因此目前证据支持“联合监督有效”，尚不支持
   “AHER-canvas multiscale head 优于 reconstruction”。
4. train calibration 改善了 head 总均值，但未改变上述结论，且胰腺仍低于固定阈值结果。

## 八类逐器官观测最优

| 器官 | 当前最高 Dice | 来源 |
|---|---:|---|
| 脾脏 | **92.98** | Arm B head，train-calibrated |
| 右肾 | **91.73** | Arm B head，train-calibrated |
| 左肾 | **92.38** | Arm B head，train-calibrated |
| 胆囊 | **53.74** | Arm B reconstruction Direct-post |
| 食管 | **69.48** | Arm B reconstruction Direct-post |
| 胰腺 | **70.35** | Arm B reconstruction Direct-post |
| 肝脏 | **94.58** | Arm B reconstruction Direct-post |
| 胃 | **82.65** | Arm B head，fixed 0.5 |

这张表用于定位每类上限，不是一个可直接部署的“拼接模型”。若要把逐器官选择变成
正式方法，必须先在验证集确定每类输出源，再对测试集做一次冻结评估。

## 未完成的同协议 head 对照

| Arm | 唯一主要变量 | 账号 / 集群 | Job | 2026-08-28 状态 | 八类结果 |
|---|---|---|---:|---|---|
| A | AHER canvas + linear head | bolinren19 / SIP | 2413564 | PENDING (Resources) | 尚无 |
| B | AHER canvas + multiscale head | antengcai23 / XEC | 121398 | COMPLETED | 已列入排名 |
| C | final ViT pixels + 20 tokens 先汇总成 1 query | bolinren19 / SIP | 2415372 | RUNNING | 尚无 |
| D | final ViT pixels + 20 个独立 queries + log-mean-exp | bolinren19 / SIP | 2548701 / 2548702 | smoke 等待；formal dependency | 尚无 |

A/C/D 完成前不得写入估计值。最终 A/B/C/D 的主要 head 对比必须统一使用
checkpoint-802、fixed-0.5 head、相同测试 CSV 和相同后处理。

## 可追溯来源

- 0.7 spacing baseline：
  `bolinren19 / SIP` 的
  `/gpfs/work/aac/bolinren19/OD_OWT_orgslot_resolution/Results/OrganSlotBank/Common8/WORD_2D/OrgSlot_WORD07072_C448_R384_BaseLoss_ROI20_1664156_eval/results.json`。
- Arm B reconstruction：
  `antengcai23 / XEC` 的
  `.../OrgSlot_WORD07072_ROI20_RetainedMultiConv_SmallOrgan_L0.01_121398_reconstruction_fixedthr002/results.json`。
- Arm B head：
  `antengcai23 / XEC` 的
  `.../OrgSlot_WORD07072_ROI20_RetainedMultiConv_SmallOrgan_L0.01_121398_heads_test/results.json`
  和同目录 `resolved_config.json`。
- 1.0 spacing ROI20 baseline：
  [ORGAN_SLOTBANK_EXPERIMENT_SUMMARY_CN.md](ORGAN_SLOTBANK_EXPERIMENT_SUMMARY_CN.md)。

Arm B 两个正式 JSON 均加载 epoch 802 checkpoint，`exact=true`；reconstruction
结果标记完整测试集，head 结果标记 `complete_split=true`，样本数均为 6990。

## 更新规则

每次新结果只在同时满足以下条件时进入主排名：

1. checkpoint 精确加载且 epoch、head 类型与配置匹配；
2. 24 病例 / 6990 切片完整评估；
3. 阈值和后处理在测试前固定；
4. 记录账号、集群、Job ID 和真实结果路径；
5. 重新计算八类均值、小器官均值和排名，不手工挑选有利器官。
````

<a id="source-22"></a>

## docs/ORGSLOT_WORD_RESOLUTION_ABLATION_LOG.md

SHA256: `48a917d087a8489287780e97f8b901f03d0f614af5599c9c9ff39c132b0d8d9e`

````text
# OrganSlot WORD Resolution Ablation Log

Last updated: 2026-08-12 18:43 CST

## Objective

Test whether the large zero/black field in the 1.0×1.0×2.0 mm, 448×448
pipeline wastes model capacity. Keep the OrganSlot architecture, Loss3, optimizer,
effective batch, update budget, split, threshold, and evaluation modes fixed.

## Code provenance

- Branch: `experiment/orgslot-resolution-ablation-v0`
- Commit: `4fd6260`
- Base local Loss3/ROI line: `749cbec`
- Merged latest remote OrganSlot fixed-reference support through `fdca982`
- XEC account/worktree:
  `sifansong:/gpfs/work/aac/sifansong/worktrees/orgslot_resolution`
- XEC environment:
  `/gpfs/work/aac/sifansong/envs/abdpet`
- XEC OpenCV was changed from GUI `opencv-python 4.6.0.66` to
  `opencv-python-headless 4.6.0.66` because login/compute nodes do not provide
  `libGL.so.1`.

## Full-dataset geometry audit

The audit used all 35,576 2D slices from the fixed WORD training and test
manifests at 1.0×1.0×2.0 mm.

| Center crop | Median black pixels | Foreground-positive slices lost | Foreground pixel retention |
|---:|---:|---:|---:|
| 448 | 74.55% | 0 | approximately 100% |
| 384 | 66.28% | 0 | approximately 100% |
| 336 | 57.49% | 0 | approximately 100% |

The focus-organ maximum 2D boxes at 1 mm were:

| Class | Organ | Maximum box |
|---:|---|---:|
| 4 | gallbladder | 77×75 |
| 5 | esophagus | 44×46 |
| 6 | pancreas | 164×95 |

No focus-organ slice exceeded ROI224. Therefore 336 was selected over 384:
it removes more uninformative border without losing a single organ-positive
slice in this fixed dataset.

Machine-readable audits:

- `docs/WORD_CROP_GEOMETRY_ANALYSIS.json`
- `docs/WORD_TRAIN_CROP_GEOMETRY_ANALYSIS.json`

## Controlled arms

All arms use OrganSlot, `linear_sqrt` fusion, nine slots, ROI sampling
probability 0.2, Loss3 positive weight 0.25, alpha 0.5, maximum class-weight
ratio 4, segmentation head loss 0, effective batch 192, and 118,800 optimizer
updates.

| Arm | Spacing (mm) | Global/input | ROI | Purpose |
|---|---:|---:|---:|---|
| Current control | 1.0×1.0×2.0 | 448 | 384 | Existing running reference |
| A | 1.0×1.0×2.0 | 336 | 224 | Remove black border by runtime crop |
| B | 0.7×0.7×2.0 | 448 | 384 | Keep 448 while increasing in-plane sampling density |

Per user decision, Arm B retains ROI384 exactly like the current experiment.
This makes B a direct spacing ablation, although the physical ROI field becomes
268.8 mm instead of 384 mm.

Loss3 training positive-slice counts for the fixed split remain:

`4392 4875 5070 1631 3827 3753 7327 5556`

The 336 audit confirmed no positive slice is lost, so Arm A uses the same
counts. Arm B counts will be read from the completed 0.7 mm training audit and
must match the configuration before its smoke is submitted.

## Validation status

- 51 OrganSlot unit tests passed after merging the latest architecture.
- Three reconstruction-evaluator regression tests passed.
- 336 model forward/backward passed (21×21 ViT patch grid).
- 336 global + ROI224 paired transform passed.
- 448 global + ROI384 at 0.7 mm proxy geometry passed.
- Python compilation, Bash syntax, and `git diff --check` passed.
- XEC repeated all 51 OrganSlot tests successfully.
- Both launchers leave Slurm-provided `CUDA_VISIBLE_DEVICES` unchanged and
  fail immediately unless the exact allocated GPU count is visible.

## Jobs and acceptance gates

### Arm A: 1 mm / 336 / ROI224

| Stage | Account/cluster | Job | State at submission | Acceptance |
|---|---|---:|---|---|
| Smoke | sifansong/XEC | 115127 | PENDING (Priority) | 2 GPUs visible; 4 updates; finite losses; ROI and positive supervision exercised; checkpoint exists |
| Formal train | sifansong/XEC | 115128 | dependency | Smoke succeeds; 118,800 updates; no NaN/OOM |
| Evaluation | sifansong/XEC | 115129 | dependency | Exact checkpoint load; all 24 cases; all 8 classes and 4 modes; 100% complete |

### Arm B: 0.7 mm / 448 / ROI384

The 0.7×0.7×2.0 mm native-matrix dataset is being generated from the original
120 WORD volumes using the fixed seed-42 case split. Submit the same
smoke→train→evaluation chain only after preprocessing, ROI audit, crop audit,
checksums, and XEC transfer complete.

## Interpretation rule

Compare each completed arm to the existing 1 mm/448/ROI384 OrganSlot run using
the same primary metric (`direct_post`, fixed threshold 0.02), especially the
gallbladder, esophagus, and pancreas Dice. Do not treat training L2/LPIPS values
across different input resolutions as segmentation improvement by themselves.
````

<a id="source-23"></a>

## docs/PIXEL_PE_ABLATION.md

SHA256: `378ad9e6ce8b8a87ac32d3adb1d3e80423de1e5e7402c0f85e7a7997d0fe05bc`

````text
# Arm D pixel PE controlled ablation

## Restart after provenance failure — 2026-09-08

Supersedes the earlier job matrix. All jobs below: antengcai23/XEC,
account sifansong, QoS 8gpus; four A800/24 CPU/192 GB per training,
one A800/10 CPU/128 GB per evaluation. Training queues independently;
each evaluation has afterok on its own training. No smoke job submitted.

| Mode | Training | Evaluation |
|---|---|---|
| none, slice-wise baseline | 133848 | 133849 |
| spatial | 133850 | 133851 |
| temporal | 133852 | 133853 |
| spatiotemporal | 133854 | 133855 |

Frozen code: /gpfs/work/aac/antengcai23/worktrees/orgslot_3d_restart_01f1666.
Commit 01f1666; archive SHA256 1ae368a12e0495937a1c65d5f4729c8a414fddeab0241794fd0be9f75730893b.
Git metadata is optional with a 15-second timeout; actual source hashes
and dataset checksums remain recorded. Full real non-Git provenance writing,
three provenance tests, three slice-loss checks and target data identity passed.
Submission exports conda lib plus shared Mesa lib in LD_LIBRARY_PATH.
All eight new jobs verified pending (training Priority, evaluation Dependency).
Old 132071/132072/132097/132099/132101 cancelled; failed runs and files retained.

## Scheduling update — 2026-09-08 (supersedes original dependencies below)

Released baseline 132071 from user hold. Removed the baseline-evaluation
dependency from PE training 132096, 132098 and 132100: all four now queue
independently. Each evaluation retains afterok on its own training job.
Changed these eight antengcai23/XEC jobs to QoS 8gpus using scontrol:
132071/132072, 132096/132097, 132098/132099, 132100/132101.
Training still requests four A800 GPUs, not eight; no model/data/loss change.
The submitted script may still say 4gpus: scontrol's current job record is
authoritative for this scheduling override. New submissions must select QoS
explicitly after checking permissions, wall-time and live capacity.

At this audit SIP and XEC A800 nodes had no unallocated GPUs. bolinren19/SIP
retains Arm E and 2D PE; sifansong/XEC retains MAE encoder-only;
antengcai23/XEC retains MAE encoder-decoder plus these 3D jobs.
No cross-account data/code migration or duplicate training was submitted.
Eight-GPU per-user QoS permits up to two four-GPU jobs within that QoS,
subject to other limits and physical availability; it does not reserve GPUs.

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
````

<a id="source-24"></a>

## docs/PIXEL_PE_NUMERICS_20260908.md

SHA256: `df4f2f855d9c7aaa2d9463916b6ea0797eb232494cba619caf8cb236befdbdd2`

````text
# 2D spatial PE restart: 2026-09-08

Confirmed: SIP job 2910849 used FP16 (both launch banner and argv), not BF16.
The PE sbatch omitted AMP_DTYPE and inherited the common launcher's fp16 default.
At epoch 64 step 233, gradient norm was nonfinite; strict clipping aborted
before optimizer.step. This also prevents GradScaler from reaching its usual
skip/backoff path. Loss clipping cannot repair an already nonfinite gradient.
The first offending operator is NOT known: old logs did not retain per-parameter
gradient diagnostics or the failing state. FP16 overflow is a hypothesis, not
a proven operator-level diagnosis.

Fix: explicitly select BF16 (no FP16 scaler), retain clipping 1.0 and strict
failure, check parameter finiteness every update, and report bad gradient
parameter names plus scaler state on any future backward/update failure.
Save every 25 epochs. Use a two-update, full-data, same-batch DDP preflight
inside the training allocation; only start formal training if validation passes.
Restart seed 0 from scratch in a new output directory (only checkpoint-0 survived).
Do not alter PE, loss, sampling, LR, or optimizer update budget.

This restart changes numeric precision relative to the historical FP16 no-PE
control; a strictly isolated PE claim ultimately needs a matched BF16 no-PE
control. A passed preflight is NOT proof of stability past epoch 64.
Keep failed output intact; replace only its permanently blocked evaluation job.

Submitted: SIP bolinren19 / account angelosstefanidis / QoS 8a800.
Training 2919868 (4 A800, 24 CPU, 192 GB); unified evaluation 2919869
(1 A800, 10 CPU, 128 GB), afterok:2919868. Old dead evaluation 2910855
cancelled; old training logs/checkpoint preserved. Five unittest checks passed
(PE shape/gradients/initialization plus launch-precision guards), bash syntax
and git diff checks passed. GPU preflight remains pending with training.
````

<a id="source-25"></a>

## docs/THREED_SLICEWISE_DIAGNOSTIC.md

SHA256: `cc0746c609fb8d51ac766488706ca95591b8adee65013503a9ffd8f4e1b03a18`

````text
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
````

<a id="source-26"></a>

## README.md

SHA256: `d879604e2dfd440dddebd68baa5188f5cd13ac057ba7cc80dd5ae49e2bf46c6f`

````text
# OrganSlot：统一开发入口

本地与GitHub统一使用 `feature/orgslot`。已整合2D/3D Arm A/B/C/D/E、
AutoPET MAE迁移、3D诊断、逐slice损失和Pixel PE。旧实验使用归档tag追溯。

新增实验：[Arm E + Cross-Attention与热图诊断](docs/ORGSLOT_ARM_E_CROSS_ATTENTION_CN.md)。
使用`--query_refinement cross_attn`；默认none保留原E。只改query refinement，
输出仍为P4（112×112）；新增实验BF16与历史E的FP16存在数值配置差异，
严格结论需同精度对照。热图显示读取位置，不自动构成因果或定位正确的证明。

## 配置与部署必读（2026-09-08）

**统一代码不等于所有任务自动使用同一配置。** 新实验必须遵循
[架构、数值精度与提交配置统一规范](docs/ORGSLOT_CONFIGURATION_CONTRACT_CN.md)。
该文档列出A–E架构参数、loss/采样配置、覆盖顺序、账号路径及训练—评估审核清单。

| 必须统一的层级 | 核验重点 |
|---|---|
| 模型选择 | dimension、slot_head_type、pixel_pe、token数量 |
| 训练语义 | segmentation_unit、loss、采样、MAE初始化或resume |
| 数值与优化 | AMP dtype、梯度裁剪、finite检查、LR、有效batch、更新预算 |
| 实际部署 | 运行worktree/commit/diff、Python环境、数据及权重绝对路径 |
| 评估 | checkpoint配置、严格加载、病例完整性、阈值/后处理、新afterok依赖 |

**已确认的风险：** 通用shell/Python入口仍默认FP16。新A800任务必须显式设置
`AMP_DTYPE=bf16 CLIP_GRAD=1.0 FINITE_CHECK_INTERVAL=1`，不能仅凭模型支持BF16
就认为启用了它。最终以运行目录的`resolved_config.json`、`command.txt`和日志为准。
这是新任务规范，不代表全部历史脚本已自动整改。

2D PE旧任务2910849遗漏精度设置，在epoch64出现非有限梯度。
[修复记录](docs/PIXEL_PE_NUMERICS_20260908.md)记录了配置证据、验证范围及
新训练2919868→评估2919869；修复已同步实际PE目录和统一入口，
尚需GPU预检查及跨epoch64验证。其他旧worktree不会自动获得修复。

新开发以`.worktrees/orgslot_integrated`为入口；运行使用可追溯快照。
本次配置文档更新不修改运行任务，不替代历史实验的原始配置。

| 配置 | 参数 |
|---|---|
| 2D / 3D | `--dimension 2D/3D` |
| Arm A/B/C/D/E | `--slot_head_type linear/multiscale_conv/query_dot/multi_query_dot/arm_e_multiscale_query` |
| 无PE / Spatial / Temporal / 两者 | `--pixel_pe none/spatial/temporal/spatiotemporal`，仅C/D；2D不支持Temporal |
| 旧3D监督 / 新逐slice监督 | `--segmentation_unit slab/slice`；默认slab保持旧实验语义，2D不受影响 |
| MAE迁移 | `--mae_init_checkpoint`、`--mae_init_scope` |

Arm E目前仅支持2D。各实验现有Slurm脚本记录原账号路径；整合不会自动迁移
正在运行或排队的任务。新实验应从统一分支建立代码快照并显式设置上述参数。

详见 [PE排表](docs/PIXEL_PE_ABLATION.md)、
[3D诊断](docs/THREED_SLICEWISE_DIAGNOSTIC.md)。以下保留Arm E历史说明。

## Arm E历史说明（2026-09-03快照，非实时任务状态）

本分支 `experiment/orgslot-querymask-multiscale-pixel` 建立在统一的 Arm C/D
参数化实现之上，并加入 Arm E。C、D、E 共用同一模型入口，不复制 OrganSlot
语义与 reconstruction 路径。Arm E 在 Arm C 单 query 路径上增加来自输入图像的
P4/P8 空间特征：

```text
image -> Conv stem -> P4/P8 --+
final ViT z -> P16 -----------+-> top-down pixel decoder -> P(x)
20 TGEnc tokens -> mean pool -> organ query q_s
mask_s(x) = sqrt(D) * cosine(P(x), q_s)
```

reconstruction 路径保持 `OrganCollector -> TGEnc -> AHER -> canvas -> fusion` 不变。
空间支路不能独立输出 mask，最终 mask 始终由 organ query 点积控制。详细设计、
验证证据和任务记录见
[docs/ORGSLOT_WORD070_ARM_E_MULTISCALE_PIXEL.md](docs/ORGSLOT_WORD070_ARM_E_MULTISCALE_PIXEL.md)。

## 统一模型配置

| Arm | `--slot_head_type` | 空间特征 | query 聚合 |
|---|---|---|---|
| C | `query_dot` | ViT P16，28 x 28 | 20 tokens 均值为 1 query |
| D | `multi_query_dot` | ViT P16，28 x 28 | 保留 20 queries，log-mean-exp 汇总 |
| E | `arm_e_multiscale_query` | Conv P4/P8 + ViT P16，输出 112 x 112 | 与 C 相同的单 query |

## 锁定的可比配置

spacing 0.7 x 0.7 x 2.0、input/ROI 448/384、ROI probability 0.2、token
factor 20、retained supervision、small-organ loss、segmentation weight 0.01、
effective batch 192、118800 updates、AdamW、seed 0 均与 Arm A/B 相同。

## MAE 迁移与 recon 最新补充（2026-09-08）

Arm B MAE encoder / encoder+decoder 的 recon 已完整核验，分别80.57% / 81.42%。
Arm E MAE 三臂配置、八类表及任务编号见 [专项记录](docs/ARM_E_MAE_AND_RECON_20260908_CN.md)。
下面部分状态是历史记录，最新三臂提交状态以专项记录为准。

## 状态

| Arm | 账号 / 集群 | Job | 状态 | 正式结果 |
|---|---|---:|---|---:|
| C query-dot | bolinren19 / SIP | 2415372 | COMPLETED | mean Dice 82.01% |
| D multi-query dot | bolinren19 / SIP | 2548702 | COMPLETED | mean Dice 82.12% |
| E multiscale query-dot 单卡 smoke | bolinren19 / SIP | 2814754 | COMPLETED | 通过 |
| E multiscale query-dot 双卡 smoke | bolinren19 / SIP | 2864374 | PENDING | 尚无 |

Arm E 第一版固定沿用 Arm C 的 20 token 平均单 query；Arm D 已说明增加 query
数量不是主要瓶颈，因此本实验只检验高分辨率空间信息。

完成后必须使用和 Arm A/B 相同的 24 病例、6990 切片、post-processing 协议；
head 主结果使用固定 0.5 阈值，训练集校准阈值只能作为次要结果。

当前已完成基线、Arm B 八类结果及项目内 SOTA 见
[docs/ORGSLOT_WORD_COMMON8_SOTA_CN.md](docs/ORGSLOT_WORD_COMMON8_SOTA_CN.md)。
更详细的结构说明见
[docs/ORGSLOT_WORD070_QUERYMASK_EXPERIMENT.md](docs/ORGSLOT_WORD070_QUERYMASK_EXPERIMENT.md)。

## 结果路径

C/D 的历史结果与 E 的新结果必须分别写入运行账号对应的 worktree
`Results/OrganSlotBank/...`。不得混用 `bolinren19 / SIP`、`sifansong / XEC`
或其他账号的数据绝对路径。运行日志和 checkpoint 不纳入 Git。
# Arm F experimental branch

2D/3D organ-conditioned two-stage attention is available through
`--slot_head_type arm_f_attention`; `arm_f_linear` is the matched linear-readout
control. See [configuration, architecture and validation](docs/ARM_F_2D_3D.md).
No formal Arm F training or benchmark results are available yet.
````

<a id="source-27"></a>

## README_EXPERIMENTS_CN.md

SHA256: `89e73c06dd6cdc598e491267d377e393baf96f4f337550bc152a6ca7cf691f64`

````text
# OrganSlot / PSEM 实验状态与结果归档

> 基础状态快照：2026-09-01 16:30；最新迁移更新：2026-09-01 22:00（Asia/Shanghai）
> 规则：运行状态以 Slurm 为准；结果仅收录已经由完整 checkpoint、评估 JSON/CSV 或训练日志核验的数值。`PENDING`、计划和 exploratory 结果不写成正式结论。

## 0. 最新迁移与活动任务（22:00增量）

本节覆盖下方16:30基础快照中已经变化的任务；未变化的完成结果仍以下文为准。

| 状态 | 实验目的 | 账号/集群/QOS | Job ID | 资源/进度 | 下一审核 |
|---|---|---|---:|---|---|
| RUNNING | 2D Arm A：linear head严格基线 | bolinren19 / SIP / angelosstefanidis-8a800 | 2413564 | 4×A800；已完成epoch439；checkpoint-400；loss有限 | checkpoint-500与最终802 |
| RUNNING | 2D Arm D：20个独立query + shared pixel decoder | bolinren19 / SIP / angelosstefanidis-8a800 | 2548702 | 4×A800；已完成epoch732；checkpoint-700；loss有限 | checkpoint-800/802与统一推理 |
| PENDING | AutoPET MAE 100k预训练 | bolinren19 / SIP / sifansong-4a800 | 2713895 | 4×A800；Priority | 启动后确认预训练loss与checkpoint |
| VALIDATED | 3D Arm C迁移真实数据门禁 | bolinren19 / SIP / sifansong-4a800 | 2755712 | 1×A800；2个optimizer update与checkpoint-0均通过；Slurm状态FAILED仅因旧validator要求2步覆盖全部slot | 已离线用修正规则复验通过，不再重复smoke |
| PENDING | 3D Arm C正式训练：20 tokens汇总为1 query，与共享3D pixel feature点积 | bolinren19 / SIP / sifansong-4a800 | 2755754 | 4×A800、24 CPU、192 GB；Priority；预计启动时间仅供参考 | 启动后首2步、checkpoint-10/50/100/199、NaN与吞吐 |
| DEPENDENCY | 3D Arm C病例级真正3D评估 | bolinren19 / SIP / sifansong-4a800 | 2755765 | 1×A800、10 CPU、128 GB；afterok:2755754 | 24病例×8器官，Dice/IoU/precision/recall/NSD/HD95 |
| PENDING | 3D Arm B从checkpoint-50续训 | sifansong / XEC / 4gpus | 127706 | 4×A800、24 CPU、192 GB；Priority | epoch51连续性、BF16与loss有限性 |
| FAILED | 3D Arm D真实数据门禁 | antengcai23 / XEC / 4gpus | 127723 | 1×A800；因MAX_TRAIN_SAMPLES=8清空ROI池而失败 | 使用本次修复代码重新提交smoke |
| BLOCKED | 3D Arm D正式训练/评估 | antengcai23 / XEC / 4gpus | 127726 / 127727 | DependencyNeverSatisfied / Dependency | 先替换失败smoke链 |

### 本次Arm C迁移修复

- SIP路径已逐项独立配置：Python、WORD070数据、ROI index、LPIPS权重、输出和日志均使用bolinren19本地路径，没有混用XEC账号目录。
- 数据身份验证器支持显式ROI路径；本地ROI文件名不同，但核心签名、28,586训练条目和6,990测试条目与固定0.7协议完全一致。
- 3D smoke不再把训练dataset截成前8条；完整ROI池为胆囊1,631、食管3,827、胰腺3,753。
- SIP PyTorch缺少BF16 depthwise Conv3d kernel；仅将两个3D depthwise层置于局部FP32/autograd路径，其余AMP保持BF16，并通过输入/权重梯度回归测试。
- 两步smoke允许未随机抽到的slot，但仍强制至少一个阳性小器官、有限重建/感知/分割loss、ROI生效、权重关系正确和checkpoint存在。
- XEC旧Arm C链127722/127724/127725已失败或取消，不会与SIP正式训练重复运行。

结论：Arm C迁移已经完成到“正式训练已排队、3D评估依赖已建立”；当前尚无3D Arm C八类指标，不能提前判断是否优于Arm B/D。

## 1. 账号、集群与 QOS

这里的“账号”必须区分 Linux 用户、Slurm account 和 QOS。特别是 `bolinren19/SIP` 同时拥有两套独立关联，不能合并计算：

| Linux用户 | 集群 | Slurm account / QOS | GPU上限 | 当前占用 | 说明 |
|---|---|---|---:|---:|---|
| `bolinren19` | SIP | `angelosstefanidis / 8a800` | 8×A800 | 8×A800 | 2D Arm A和Arm D各占4卡，已满 |
| `bolinren19` | SIP | `sifansong / 4a800` | 4×A800 | 0 | 独立额度，适合1卡推理和汇总任务 |
| `sifansong` | SIP | `sifansong / 4a800` | 4×A800 | 0 | 无活动任务；SIP数据/环境路径尚未完成启动前核验 |
| `antengcai23` | SIP | `sifansong / 4a800` | 4×A800 | 0 | 无活动任务；SIP数据/环境路径尚未完成启动前核验 |
| `sifansong` | XEC | `sifansong / 4gpus` | 4×GPU | 0 running | 3D Arm B、C入口任务等待；同一用户最多只能运行一组4卡训练 |
| `antengcai23` | XEC | `sifansong / 4gpus` | 4×GPU | 0 running | 3D Arm D入口任务等待；与`sifansong/XEC`是独立用户额度 |

快照时的物理资源：

- SIP `gpua800`：常规A800节点没有连续4张空闲卡；`gpua800n6`最多只有约2张零散标准A800可见。因此有空闲QOS不等于4卡训练能够立即启动。
- XEC `gpua800,gpua8001t`：A800类GPU在快照时全部分配，没有即时空闲卡。
- Slurm预测：XEC Arm C/D smoke最早约为2026-09-02 22:44；3D Arm B约为2026-09-03 19:07。预测会随其他用户任务变化。

## 2. 正在训练或等待训练

| 状态 | 实验目的 | 账号/集群/QOS | Job ID | 资源 | 当前进度或依赖 | 下一检查点 |
|---|---|---|---:|---|---|---|
| RUNNING | 2D Arm A：验证简单linear head，在0.7 spacing、ROI20和small-organ loss下作为严格head基线 | `bolinren19/SIP/8a800` | `2413564` | 4×A800、40 CPU、256 GB | epoch约377/802；最新checkpoint-300；loss有限 | checkpoint-400及NaN/ROI/seg loss稳定性 |
| RUNNING | 2D Arm D：20个organ token保留为20个独立query，与共享pixel decoder点积并LogMeanExp融合 | `bolinren19/SIP/8a800` | `2548702` | 4×A800、24 CPU、192 GB | epoch约657/802；最新checkpoint-600；loss有限 | checkpoint-700、最终802；预计比Arm A先释放GPU |
| PENDING | AutoPET MAE预训练：为后续WORD的encoder-only与encoder+decoder初始化对照提供预训练权重 | `bolinren19/SIP`（提交脚本QOS需再次核对） | `2713895` | 4×A800、24 CPU、192 GB | `Priority` | 启动配置、100k更新、预训练checkpoint完整性 |
| PENDING | 3D Arm B续训：原multiscale AHER-canvas head，修复NaN/BF16后从checkpoint-50继续 | `sifansong/XEC/4gpus` | `127706` | 4×A800、24 CPU、192 GB | `Priority` | 必须确认从epoch51连续恢复，BF16/梯度/optimizer均有限 |
| PENDING | 3D Arm C真实数据门控：20 tokens先汇总成1个query，再与3D pixel feature点积 | `sifansong/XEC/4gpus` | `127722` | 1×A800、8 CPU、64 GB | `Priority` | 两次真实optimizer step、ROI和small-organ诊断通过 |
| DEPENDENCY | 3D Arm C正式训练 | `sifansong/XEC/4gpus` | `127724` | 4×A800、24 CPU、192 GB | `afterok:127722` | checkpoint-10/50/100/199和数值稳定性 |
| PENDING | 3D Arm D真实数据门控：20个独立query的3D版本 | `antengcai23/XEC/4gpus` | `127723` | 1×A800、8 CPU、64 GB | `Priority` | 两次真实optimizer step、BF16和反向传播通过 |
| DEPENDENCY | 3D Arm D正式训练 | `antengcai23/XEC/4gpus` | `127726` | 4×A800、24 CPU、192 GB | `afterok:127723` | checkpoint-10/50/100/199和数值稳定性 |

### 当前训练分布判断

- 正确部分：3D Arm D放在`antengcai23/XEC`，能与`sifansong/XEC`独立占用4卡QOS。
- 主要瓶颈：3D Arm B和Arm C都位于`sifansong/XEC`，二者正式训练不能并行。
- 推荐但尚未执行：优先调查将3D Arm C迁到一个已核验数据和环境的SIP 4卡通道。迁移前必须核验SIP账号自己的Python、WORD070数据、ROI index、LPIPS权重和输出路径；不能直接套用XEC路径。
- 不建议中断已经运行的2D Arm A/D。Arm D离终点更近，完成后自然释放`8a800`中的4卡。

## 3. 等待推理与结果汇总

| 推理/汇总目的 | 账号/集群 | Job ID | 前置条件 | 当前问题 | 完整性标准 |
|---|---|---:|---|---|---|
| Focal checkpoint-802统一推理，和checkpoint-500同协议比较续训收益 | `bolinren19/SIP` | `2715065` | checkpoint已存在 | 当前受`QOSMaxGRESPerUser`限制；应迁到空闲的`4a800` 1卡通道 | 6990 slices、24 cases、八类指标齐全 |
| 2D Arm A统一head评估 | `bolinren19/SIP` | `2715066` | `afterok:2413564` | 等训练完成 | 固定0.5阈值；与B/C/D相同数据、脚本和后处理 |
| 2D Arm D统一head评估 | `bolinren19/SIP` | `2715067` | `afterok:2548702` | 等训练完成 | 同上 |
| 2D A/B/C/D总表 | `bolinren19/SIP` | `2715236` | 依赖head评估完成 | 等待依赖 | 八类Dice、mean、precision/recall、速度和资源完整 |
| 3D Arm B病例级评估 | `sifansong/XEC` | `127707` | `afterok:127706` | 等训练 | 24个病例×8器官；Dice/IoU/precision/recall/NSD/HD95 |
| 3D Arm C病例级评估 | `sifansong/XEC` | `127725` | `afterok:127724` | 等训练 | 同上 |
| 3D Arm D病例级评估 | `antengcai23/XEC` | `127727` | `afterok:127726` | 等训练 | 同上 |

推理任务通常只需1张GPU。合理的资源策略是把SIP本地推理从已满的`8a800`关联迁到`bolinren19 --account=sifansong --qos=4a800`，优先产出已训练模型的结果；不要用4张训练卡长期等待一个1卡评估。

## 4. 已完成且数值已核验的实验

### 4.1 WORD：0.7 spacing + ROI20 + 2D Arm B multiscale head

checkpoint-802完整加载，覆盖6990张测试切片和24个病例。该结果是2D slice级统一head协议，不应冒充病例级3D评估。

| 器官 | 固定0.5 Dice | 训练集校准阈值 Dice |
|---|---:|---:|
| 脾脏 | 92.62% | 92.98% |
| 右肾 | 91.04% | 91.73% |
| 左肾 | 91.80% | 92.38% |
| 胆囊 | 49.72% | 51.45% |
| 食管 | 67.86% | 68.75% |
| 胰腺 | 69.03% | 68.71% |
| 肝脏 | 94.41% | 94.41% |
| 胃 | 82.65% | 82.47% |
| 前景平均 | 79.89% | 80.36% |

分析：

- multiscale head能稳定读出肾、脾、肝等大器官，但胆囊仍是明显短板。
- 校准只带来`+0.47`个百分点mean Dice，说明主要瓶颈不是单一全局阈值。
- 胰腺固定0.5反而略优于训练校准，提示不能在测试集上逐器官调阈值并把它当作方法增益。
- 该head依赖AHER canvas；如果OrganCollector/AHER已经压缩空间细节，增加卷积容量不能完全恢复细长、小体积和低对比边界。这正是Arm C/D绕开canvas、引入共享pixel path的动机。

### 4.2 AutoPET：PSEM-v3正式下游评估

已完整处理200个病例：

| 方法 | 4类平均post Dice |
|---|---:|
| Indirect input-minus-without-class | **89.94%** |
| Direct class token | 88.78% |

分析：Indirect比Direct高`1.16`个百分点。该结论支持PSEM-v3表征有效，但还需要与相同AutoPET拆分、相同后处理和相同checkpoint选择规则下的历史PSEM版本比较；不能把不同协议的数字直接排序。

### 4.3 WORD分辨率消融

- 后续实验已统一采用`0.7×0.7×2.0` spacing、输入448、ROI384、ROI probability 0.2。
- 分辨率消融与定性可视化已经完成，当前不再重复训练。
- 该消融曾使用24个测试病例进行探索性比较，因此应标注为`exploratory ablation`；正式方法选择应由验证集完成，测试集只作最终报告。

## 5. 尚不能写成最终结论的项目

- 2D Arm A、Arm D尚未到checkpoint-802，也未完成统一推理。
- 3D Arm B/C/D尚未完成训练和病例级3D评估，当前没有可比较的3D八类结果。
- Focal checkpoint-802统一推理尚未启动，不能判断从checkpoint-500续训是否提高最终指标。
- MAE预训练尚未启动，encoder-only和encoder+decoder初始化对照也未产生WORD结果。
- Arm C历史结果和可视化若要进入SOTA表，必须从其正式评估JSON重新导入精确八类指标和证据路径；本文件不凭记忆补数值。

## 6. 推荐资源分布与执行顺序

1. 保持2D Arm A/D在`bolinren19/SIP/8a800`继续运行，不中断。
2. 将Focal-802及后续2D A/D推理放到`bolinren19/SIP/4a800`，每项1卡；优先产出已完成checkpoint的结果。
3. `sifansong/XEC`优先3D Arm B，因为它是已经从checkpoint-50恢复的基准；Arm C与它共享4卡上限，应避免同时保留多个无优先级说明的4卡正式任务。
4. `antengcai23/XEC`继续3D Arm D，形成与B独立并行的训练通道。
5. 若要让B/C/D真正三路并行，先为Arm C核验并准备SIP 4卡环境，再迁移；确认新任务成功启动后才取消XEC副本，避免重复训练或两边都失败。
6. Arm D 2D预计最先完成；完成后先跑统一推理，再让MAE占用释放的4张`8a800`训练卡。

## 7. 下次固定审核清单

1. 2D Arm D是否达到checkpoint-700/802，依赖评估`2715067`是否释放。
2. 2D Arm A是否达到checkpoint-400，loss、ROI比例和小器官监督是否仍有限。
3. Focal推理是否已经迁到4a800并完成6990 slices/24 cases。
4. 3D Arm B是否从checkpoint-50连续恢复，且未再次出现NaN。
5. 3D Arm C/D smoke是否完成两次真实optimizer step；正式任务是否自动释放。
6. 所有3D评估是否为病例级重建，而不是把2D slice Dice重新命名为3D。
7. 每次报告同时记录：目的、账号/集群/QOS、Job ID、资源、epoch、最新checkpoint、loss有限性、推理完整性、八类指标、是否达到预期和下一步。

## 8. 证据与相关文档

- 3D Arm C/D方法及公平协议：`docs/ORGSLOT_WORD070_QUERYMASK_3D_EXPERIMENT_CN.md`
- small-organ loss：`docs/ORGSLOT_WORD070_SMALL_ORGAN_LOSS_CN.md`
- 完整实验总结：`docs/COMPLETE_EXPERIMENT_SUMMARY.md`
- 分辨率定性结果：`OD_OWT_orgslot_resolution/artifacts/resolution_ablation_qualitative/`
````
