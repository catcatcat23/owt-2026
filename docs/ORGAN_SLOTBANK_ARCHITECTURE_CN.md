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
