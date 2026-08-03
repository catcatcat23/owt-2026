# OWT 项目完整实验总结

最后核对：2026-08-03 CST
维护 worktree：/gpfs/work/aac/bolinren19/OD_OWT
用途：统一说明已经做过什么、哪些结果可信、哪些实验正在运行，以及下一步应优先做什么。

## 1. 阅读和结论口径

本文件是项目级总览。详细正式指标以 FORMAL_EXPERIMENT_REGISTRY.md
为准，单个方法的实现细节以对应分支中的专项实验日志为准，最终原始证据是
checkpoint、protocol.json、Step2/Step3 CSV 和 Slurm sacct 记录。

状态含义：

| 状态 | 含义 |
|---|---|
| 正式完成 | 1200 epochs、checkpoint-1199、完整 Step2/Step3 都存在 |
| 训练完成待评估 | checkpoint-1199 存在，但不能判断效果 |
| 运行中 | 正式训练尚未结束，epoch 指标只能判断稳定性 |
| 工程验证 | 单元测试、forward/backward、tiny overfit 或 smoke 通过，不代表性能 |
| 排除 | 存在已知语义错误或协议不可比，不进入方法排名 |
| 取消/暂缓 | 未运行或主动停止，保留代码和记录 |

所有正式方法目前都只有一个训练随机种子。本文的变化量是描述性效应，
不是跨随机种子的统计显著性结论。病例级 bootstrap 区间只反映当前测试病例
的差异，不能覆盖重新训练模型产生的方差。

当前 OWT 正式结果是全类别联合训练，不是类别增量学习结果。OrganSlotBank
已经实现增量训练所需的结构与标签隔离，但尚无正式真实数据增量指标。

## 2. 数据集、架构与统一评估

| 数据集 | 前景类 | 训练病例 | 测试病例 | 训练样本 | 备注 |
|---|---:|---:|---:|---:|---|
| AbdAutoPET | 4 | 700 | 200 | 78,400 slices | 类别只可靠命名为 label_1 至 label_4 |
| WORD Common8 | 8 | 96 | 24 | 19,078 slices | spleen、双肾、gallbladder、esophagus、pancreas、liver、stomach |
| BTCV Common8 | 8 | 24 | 6 | 3,588 slices | 测试集很小，结论方差高 |

原始 OWT v11 主路径：

图像 -> ViT Patch Encoder -> OrganCollector -> 固定类别 Token Bank
-> TGEnc -> AHER -> Transformer Decoder -> 条件重建。

WORD/BTCV 使用 224×224、patch 16、固定 HU 窗口后除以 255；AbdAutoPET
使用原始 per-sample min-max。每类 20 tokens，Common8 含背景共 180 tokens，
AbdAutoPET 含背景共 100 tokens。

统一推理设置：

- Direct：只保留目标类别 tokens。
- Indirect：输入减去“不含目标类别 tokens”的重建。
- 重建阈值 0.02，分割阈值 0.15，NSD 容差 3 mm。
- Common8 最小连通域 100 voxels，AbdAutoPET 为 20。
- 主要指标是同数据集、同维度下的 Direct post macro Dice；同时必须检查
  Direct raw、Indirect、逐类 Dice、NSD、预测体积和 Step2 重建。

## 3. 已实现方法总览

| 方法 | 改动 | 当前证据边界 |
|---|---|---|
| OWT | 原始全图 L2 + LPIPS，batch 共享随机类别 mask | 三个数据集的正式基线 |
| LossBalance-v1 | 增加逐样本、逐器官等权 ROI-L2，背景权重 0 | WORD 2D/3D、AbdAutoPET 2D 正式完成 |
| PSEM-v1 | 每样本对 GT-present 类穷举 mask；变长 token 打包和 padding 隔离 | AbdAutoPET 2D、WORD 2D 正式完成；3D 运行中 |
| LossBalance-v2 | ROI 再乘样本频率权重，权重约为 N_c^-0.5，最大比 4 | WORD 2D 训练完成，待统一评估 |
| PSEM+LossBalance-v2 | PSEM-v1 与 LossBalance-v2 的受控组合 | WORD 2D 训练完成；AbdAutoPET 2D 运行中 |
| LossBalance-v3a | 只优化 present 且 kept 的正向 ROI，系数 0.25；removed ROI 只监控 | WORD 2D 运行中 |
| VQGAN-CNN | 用连续 CNN 编码器/解码器替换 ViT 图像端 | WORD 2D 运行中；没有 codebook，不得称为 VQ-VAE |
| PSEM-v2a | 20-slot 全 Token Bank 推理状态调度，显式正/负查询 | WORD/AbdAutoPET 2D 等待 smoke |
| OrganSlotBank v0 | 每类独立 slot/TGEnc/AHER/binary head，append/freeze/严格标签隔离 | 工程 Gate A 通过，无正式性能结论 |

## 4. 正式完成实验与核心指标

Dice 和 NSD 均以百分数表示。Whole L2 越低越好。E01-E09 沿用正式台账
编号；E10 是本次补录的 PSEM-v1 WORD 2D。

| ID | 方法 / 数据 / 输入 | 训练 Job | 评估 Job | Direct post | Indirect post | Whole L2 | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| E01 | OWT / AbdAutoPET / 2D | 1512521 | 1549968+1549972 | 89.94 | 90.39 | 0.00015664 | 基线 |
| E02 | OWT / WORD / 2D | 1527637 | 1539085 | 55.07 | 58.31 | 0.00031396 | 基线，小器官完全漏检 |
| E03 | OWT / WORD / 3D Fixfr4 | 1539406 | 1539427 | 59.93 | 60.94 | 0.00032112 | 3D 改善中大器官，未解决最小器官 |
| E04 | OWT / BTCV / 2D | 1527638 | 1539086 | 10.39 | 10.43 | 0.00626063 | 多类别塌缩 |
| E05 | OWT / BTCV / 3D Fixfr4 | 1539407 | 1539428 | 34.16 | 12.24 | 0.00626027 | 优于 2D，但绝对性能仍低 |
| E06 | LossBalance-v1 / WORD / 2D | 1560195 | 1563822 | 64.99 | 41.57 | 0.00140731 | Direct 有效，Indirect 与重建退化 |
| E07 | LossBalance-v1 / WORD / 3D | 1560196 | 1563824 | 67.57 | 62.39 | 0.00076073 | 当前最明确的有效损失实验 |
| E08 | LossBalance-v1 / AbdAutoPET / 2D | 1564397 | 1564399 | 84.19 | 83.30 | 0.00027698 | 无效 |
| E09 | PSEM-v1 / AbdAutoPET / 2D | 1563805 | 1563831 | 90.11 | 90.51 | 0.00015039 | 边际改善，需重复种子 |
| E10 | PSEM-v1 / WORD / 2D | 1605670 | 1605673 | 36.27 | 34.87 | 0.00171145 | 无效，训练/推理查询状态失配 |

### 4.1 WORD 2D 横向比较

| 方法 | Direct raw | Direct post | 相对 OWT | Indirect raw | Indirect post | 相对 OWT |
|---|---:|---:|---:|---:|---:|---:|
| OWT | 55.30 | 55.07 | 0.00 | 47.23 | 58.31 | 0.00 |
| LossBalance-v1 | 63.85 | 64.99 | +9.92 | 19.69 | 41.57 | -16.74 |
| PSEM-v1 | 34.48 | 36.27 | -18.80 | 17.66 | 34.87 | -23.44 |

### 4.2 WORD 2D 小器官与逐类 post Dice

每格为 Direct / Indirect。

| 器官 | OWT | LossBalance-v1 | PSEM-v1 |
|---|---:|---:|---:|
| spleen | 86.00 / 88.31 | 85.63 / 57.08 | 37.75 / 48.66 |
| right kidney | 79.30 / 86.83 | 82.81 / 48.45 | 43.57 / 40.81 |
| left kidney | 79.98 / 85.55 | 82.27 / 50.05 | 45.11 / 42.65 |
| gallbladder | 0.00 / 0.00 | 24.64 / 5.79 | 10.83 / 0.78 |
| esophagus | 0.00 / 0.00 | 33.86 / 8.84 | 9.89 / 0.74 |
| pancreas | 38.43 / 43.73 | 53.27 / 27.99 | 45.87 / 18.28 |
| liver | 91.02 / 92.28 | 89.23 / 81.56 | 46.97 / 79.73 |
| stomach | 65.84 / 69.78 | 68.21 / 52.85 | 50.13 / 47.28 |

LossBalance-v1 的确解决了部分 Direct 小器官“完全为零”的问题，但预测偏大：
Direct 预测/GT 体积比的类别宏平均由 OWT 的 0.647 上升到 1.930，
Indirect 更达到 7.416。PSEM-v1 WORD 的 Direct 体积比为 2.664，
说明其较低 Dice 同时伴随广泛过分割，而不是单纯阈值过高。

### 4.3 WORD 3D

LossBalance-v1 相对 OWT：

- Direct post：59.93 -> 67.57，+7.64 点。
- Indirect post：60.94 -> 62.39，+1.45 点。
- gallbladder：0/0 -> 26.90/23.11。
- esophagus：0/0 -> 39.00/28.62。
- pancreas Direct：47.61 -> 55.51。

代价是 spleen、双肾、liver、stomach 小幅下降，NSD 和 Indirect raw
也没有同步全面改善。结论是“小器官目标明确受益，但不是所有质量指标都提高”。

### 4.4 AbdAutoPET 2D

- PSEM-v1 相对 OWT：Direct post +0.17，Indirect post +0.13；
  主要收益来自 label_4 Direct +1.40，但 label_3 Direct -1.05。
- LossBalance-v1 相对 OWT：Direct post -5.75，Indirect post -7.09，
  whole L2 上升 76.82%，明确无效。
- PSEM 的增益小于 0.2 点且只有单次训练，不能写成稳定提升。

### 4.5 BTCV

BTCV 2D 几乎只学会 liver。3D Direct post 从 10.39 提升到 34.16，
恢复了部分 spleen、kidney、pancreas 和 stomach，但 gallbladder/esophagus
仍接近 0。与论文 Offline 结果不是受控比较；仅 6 个测试病例也不足以给出稳定排名。

## 5. 描述性病例级差异区间

方法：同一测试病例内先对类别做宏平均，再对方法差值进行 10,000 次病例级
配对 bootstrap。区间不包含训练随机种子不确定性。

| 比较 | 模式 | 病例数 | 平均变化点 | 95% 病例 bootstrap 区间 |
|---|---|---:|---:|---:|
| PSEM-v1 - OWT，AbdAutoPET 2D | Direct post | 200 | +0.17 | [-0.15, +0.40] |
| PSEM-v1 - OWT，AbdAutoPET 2D | Indirect post | 200 | +0.13 | [+0.02, +0.23] |
| LossBalance-v1 - OWT，AbdAutoPET 2D | Direct post | 200 | -5.75 | [-6.12, -5.44] |
| LossBalance-v1 - OWT，AbdAutoPET 2D | Indirect post | 200 | -7.09 | [-7.47, -6.75] |
| LossBalance-v1 - OWT，WORD 2D | Direct post | 24 | +9.92 | [+8.57, +11.30] |
| LossBalance-v1 - OWT，WORD 2D | Indirect post | 24 | -16.73 | [-18.26, -15.24] |
| PSEM-v1 - OWT，WORD 2D | Direct post | 24 | -18.81 | [-20.66, -16.92] |
| PSEM-v1 - OWT，WORD 2D | Indirect post | 24 | -23.44 | [-25.32, -21.65] |
| LossBalance-v1 - OWT，WORD 3D | Direct post | 24 | +7.64 | [+6.56, +8.72] |
| LossBalance-v1 - OWT，WORD 3D | Indirect post | 24 | +1.45 | [+0.27, +2.77] |

实际解释：WORD 的大幅正负变化在当前 24 个病例上方向一致；AbdAutoPET
PSEM 的 Direct 区间跨 0，说明边际提升很容易被病例构成或重新训练波动抵消。

## 6. 关键机制分析

### 6.1 为什么 LossBalance-v1 在 WORD 有效、在 AbdAutoPET 无效

WORD 的 gallbladder、esophagus 和 pancreas 面积小、阳性切片少，全图 L2
几乎不给它们梯度。逐器官 ROI 平均显著增加这些类别的优化权重，因此 Direct
信号从 0 恢复。但 v1 同时在被删除类别的 GT 区域学习零目标抑制，而且强 ROI
项改变了灰度重建与 token 分解，导致 over-segmentation、Indirect 退化和 whole
重建变差。

AbdAutoPET 原始 OWT 已经达到约 90 Dice，类别更常见、训练样本更多，原损失
没有同样严重的小器官完全漏检。额外 ROI 监督反而扰乱已经良好的重建分解，
所以三个主要类别和整体重建都下降。

### 6.2 为什么 PSEM-v1 在 AbdAutoPET 尚可、在 WORD 失败

PSEM-v1 只允许当前 GT 中实际存在的类别 token 进入 TGEnc。AbdAutoPET 有 4 个
前景类且在切片中更常见，训练时的 token 组合与 Direct/Whole/Leave-one-out
推理状态相对接近。

WORD 有 8 个前景类，而训练日志显示每个样本平均只有 2.276 个存在类、保留
1.511 类，padding 比例 0.663。推理时 Whole、organs-only、leave-one-out 会把
大量训练时从未和该样本共同出现的 token 放在一起。结果是：

- Direct post -18.80 点，Indirect post -23.44 点；
- whole L2 是 OWT 的 5.45 倍，organs-only L2 是 9.83 倍；
- Direct 预测体积平均是 GT 的 2.664 倍。

因此问题不是 padding 补零本身；padding 已被严格 mask。根因是“GT presence
被错误地同时当成 token eligibility”，造成训练与推理查询状态不一致。

### 6.3 PSEM-v2a 对应修正

PSEM-v2a 保留逐样本打包和 padding 隔离，但训练直接覆盖推理查询状态：
20% Direct-positive、20% Direct-negative、25% leave-one-out、10% whole、
5% background-only、5% organs-only、15%随机非空全-bank子集。GT 只决定
查询是正样本还是负样本，不再阻止不存在类别 token 进入模型。

这能回答 PSEM-v1 暴露的机制问题，但在正式 WORD/AbdAutoPET 2D 评估完成前，
不能宣称有效。

### 6.4 LossBalance-v2 和 v3a 的迭代逻辑

LossBalance-v2 在逐器官平均之外引入样本频率权重，目的是让阳性样本少的类别
在整个训练期间也获得更高累计梯度。WORD 2D 已训练完成，但训练 loss 的变化
不能代替统一推理。

LossBalance-v3a 进一步只优化 present 且 kept 的正向器官 ROI，系数降到 0.25；
removed 类仍由 global L2/LPIPS 学零目标，额外 ROI 只作为 monitor。它试图保留
小器官 recall，同时减少 v1/v2 的过分割与 token compositionality 破坏。

## 7. 新完成与待完成评估

| 方法 | Job | 产物 | 最终训练指标 | 缺少 |
|---|---:|---|---|---|
| LossBalance-v2 WORD 2D | train 1586123，eval 1623976 | checkpoint-1199 + 完整评估 | global 0.001536，ROI 0.005534，LPIPS 0.030113 | 已完成，待正式排名 |
| PSEM+LossBalance-v2 WORD 2D | train 1605963，eval 1634605 | checkpoint-1199 存在，评估等待 Priority | global 0.002448，ROI 0.004574，LPIPS 0.044490 | 等统一 Step2/Step3 |

两项训练都稳定完成，但损失定义不同，不能用最终训练总和横向排名。LossBalance-v2
已经用统一 WORD 2D evaluator 完成评估；组合实验也已提交同协议评估。

## 8. 当前运行和排队快照

快照时间：2026-08-03。状态来自 `sacct`/`squeue`，只用于进度检查。

| Job | 方法 / 数据 | 状态 | 最新进度 | 后续 |
|---:|---|---|---|---|
| 1567343 / 1563842 | PSEM-v1 AbdAutoPET 3D，4 GPU续训 / 评估 | COMPLETED / COMPLETED | 完整 200 病例结果已生成 | 待正式排名 |
| 1586125 / 1634607 | VQGAN-CNN WORD 2D / 评估 | COMPLETED / PENDING Priority | checkpoint-1199；VQ 模型严格加载通过 | 等统一评估 |
| 1605961 / 1634606 | PSEM+LossBalance-v2 AbdAutoPET 2D / 评估 | COMPLETED / PENDING Priority | checkpoint-1199 | 等统一评估 |
| 1605963 / 1634605 | PSEM+LossBalance-v2 WORD 2D / 评估 | COMPLETED / PENDING Priority | checkpoint-1199 | 等统一评估 |
| 1627036 / 1627039 | LossBalance-v3a WORD 2D / 评估 | COMPLETED / COMPLETED | 完整 24 病例结果已生成 | 待正式排名 |
| 1629560 / 1634604 | PSEM-v2a WORD 2D / 评估 | COMPLETED / PENDING Priority | checkpoint-1199 | 等统一评估 |
| 1629561 / 1629564 | PSEM-v2a AbdAutoPET 2D / 依赖评估 | RUNNING / PENDING Dependency | 训练运行中 | 结束后自动评估 |

2026-08-03 已补交四个缺失推理：PSEM-v2a WORD `1634604`、组合 WORD
`1634605`、组合 AbdAutoPET `1634606`、VQGAN-CNN WORD `1634607`。
LossBalance-v2 WORD 的评估 `1623976` 已完成，不重复提交；PSEM-v2a
AbdAutoPET 已有依赖评估 `1629564`，也不重复提交。

## 9. 排除、取消与工程验证

| 项目 | Job / 状态 | 处理 |
|---|---|---|
| CropMix-v1 WORD 2D/3D | train 1552984/1552985，eval 1554939/1554940，均完成 | 排除；focus 类被加入删除集合且 batch 并集应用到所有样本，监督语义错误 |
| 早期 WORD/BTCV legacy | 1524575、1524576 等 | 排除；不是统一 Common8 协议 |
| LossBalance-v2 WORD 3D full | 1586124，CANCELLED，00:00:00 | 暂缓；先完成 2D 评估 |
| PSEM-v1 AbdAutoPET 3D 原2卡任务 | 1563807，运行19:12:08后取消 | 从 checkpoint-100 改为4卡任务1567343续训 |
| OrganSlotBank | CPU smoke 1561641 完成；32/32测试通过 | 仅工程 Gate A，不进入性能排名 |
| 各类 smoke/tiny overfit | 多个通过 | 证明代码可运行，不证明泛化性能 |

PSEM-v2a 第一组 1629541-1629544、1629547 在启动前取消，因为 smoke batch
没有匹配正式 batch；修正后的任务是 1629558-1629561、1629564。

## 10. OrganSlotBank 当前完成边界

已完成：

- 每类显式 OrganSlot、独立 TGEnc/AHER、binary segmentation head；
- normalized additive canvas 和共享 decoder；
- Base/Incremental 标签严格可见性；
- append new slot、从 background 初始化、冻结范围与 SHA-256 hash 审计；
- Ours、Sequential-FT、Head-only 三种 trainable scope；
- 2D/Fixfr4 synthetic forward/backward、真实数据 debug-width 单 batch；
- tiny overfit loss 1.7362 -> 0.1302，Dice 0.9354；
- 32/32 CPU 测试和 CPU Slurm smoke 1561641 通过。

尚未完成：

- 正式 ViT-Base GPU 2D smoke；
- 真实 Base 与增量完整训练；
- Old/New/All、forgetting 的 case-level 正式评估；
- suppression/background plasticity 消融；
- 数据 raw label ID 到器官名称的独立证据。

所以当前只能说结构、隔离和训练编排通过，不能声称优于 PCDD 或具有抗遗忘优势。

## 11. 资源与实验决策

从 2026-08-02 起，新 loss、mask 或架构遵循：

2D 单元测试/tiny overfit -> 真实 2D smoke -> 2D 正式训练
-> 统一 2D Step2/Step3 -> 证明有意义且无灾难性退化 -> 才提交 3D。

“有意义”至少要求：

1. Direct/Indirect 不只看一个模式；
2. gallbladder、esophagus、pancreas 的 Dice/Recall/体积比改善；
3. whole/organs reconstruction 不出现数量级恶化；
4. 使用同一 CSV、归一化、阈值、后处理和病例数；
5. 小于 1 个百分点的单次变化默认标为边际，优先补随机种子。

## 12. 下一步优先级

1. 立即评估 LossBalance-v2 WORD 2D checkpoint-1199。
2. 立即评估 PSEM+LossBalance-v2 WORD 2D checkpoint-1199。
3. 等 VQGAN-CNN 完成后补统一 WORD 2D 评估。
4. 等 LossBalance-v3a 完成并自动评估，重点看 precision/recall 与体积比。
5. 完成 PSEM+LossBalance-v2 AbdAutoPET 2D 并提交评估。
6. 监控 PSEM-v1 AbdAutoPET 3D 完成及依赖评估。
7. 运行 PSEM-v2a 两个 full-batch smoke；通过后再启动 2D 正式训练。
8. 在上述 2D 结果明确前，不新增 3D full run。

## 13. 分支与关键记录

| 分支 | Worktree | 当前 HEAD / 关键提交 |
|---|---|---|
| main | /gpfs/work/aac/bolinren19/2026-07/OD_OWT | 6cdf9b5 |
| experiment/psem-v1 | /gpfs/work/aac/bolinren19/OD_OWT | 8a7da1f |
| experiment/lossbalance-v2 | /gpfs/work/aac/bolinren19/OD_OWT_lossbalance_v2 | 406a683 |
| experiment/lossbalance-v3 | /gpfs/work/aac/bolinren19/OD_OWT_lossbalance_v3 | 15413e2 |
| experiment/psem-lossbalance-v2 | /gpfs/work/aac/bolinren19/OD_OWT_psem_lossbalance_v2 | e1987dd |
| experiment/psem-v2-query-mask | /gpfs/work/aac/bolinren19/OD_OWT_psem_v2 | 0fcc3a4 |
| experiment/vqcnn-word-v0 | /gpfs/work/aac/bolinren19/OD_OWT_vqcnn | 918dd6e |
| feature/orgslotbank-v0 | /gpfs/work/aac/bolinren19/OD_OWT_orgslot | dc545a3 |

主要文档：

- FORMAL_EXPERIMENT_REGISTRY.md：正式指标和排名。
- PSEM_V1_EXPERIMENT_LOG.md：PSEM-v1 实现与早期任务记录。
- 各独立 worktree 的 docs/*EXPERIMENT_LOG.md：方法实现、验证和 Job。
- OrganSlotBank 架构说明与 handoff：位于 orgslot worktree 的 docs/。

每次新正式评估完成后，应先保留原始 per-case CSV，再按统一口径更新正式台账
和本总文档。不能用训练 loss、smoke 或 class-only 全背景指标替代正式 Dice。
