# PSEM 实验总览

本分支统一保存 Per-Sample Exhaustive/Structured Masking（PSEM）系列实现与实验记录。本文只对**相同数据集、相同测试病例和相同正式评估协议**下的结果排序；不同数据集之间不比较绝对数值。

## 结论摘要

- **AbdAutoPET 2D：** PSEM-v1 Indirect 仍是当前 PSEM 系列最优，4 类 post Dice 为 **90.512%**。PSEM-v3 Triplet+Loss3 的 Indirect 为 **89.938%**，与 PSEM-v2a Indirect 基本持平，但没有超过 v1。
- **WORD 2D：** PSEM-v3 同时取得当前 PSEM 系列最优 Direct（**68.353%**）和 Indirect（**68.244%**）。Triplet 的主要收益是修复 Indirect 查询分解，而不是显著提高 Direct。
- **AbdAutoPET 3D Fixfr4：** 目前只有 PSEM-v1 完成正式评估，尚不能判断 v2/v3 在 3D 上是否更好。
- **数值稳定性：** PSEM-v3 AutoPET 原 FP16 路线在约 epoch 500 附近出现 NaN。修复版本采用 BF16、LPIPS FP32、非有限值诊断和梯度裁剪，最终训练 Job `121484` 与 200 病例评估 Job `124936` 均正常完成。

## 统一指标口径

- 排名主指标：`all_cases`、类别等权宏平均 `post Dice`，数值越高越好。
- 辅助指标：`post NSD` 越高越好，`post HD95` 越低越好。
- Direct：只保留目标类别 token 的重建响应。
- Indirect：`input/full reconstruction - without-class reconstruction` 的差分响应。
- 分割结果来自重建响应阈值化及统一后处理，不是额外训练的 segmentation head。
- AutoPET 2D/3D 均为 200 个测试病例、4 个前景类；WORD 2D 为 24 个测试病例、8 个前景类。
- AutoPET 的可靠器官名称映射尚未确认，因此只使用 `label_1`–`label_4`；WORD 类别顺序为脾脏、右肾、左肾、胆囊、食管、胰腺、肝脏、胃。

## AbdAutoPET 2D 排名

相同 200 病例、224 输入、`per_sample` 归一化、固定阈值 0.15 和最小连通域 20。表内类别 Dice 顺序为 `label_1/label_2/label_3/label_4`。

| 排名 | 实验 | 读出 | post Dice | post NSD | post HD95 (mm) | 4 类 post Dice |
|---:|---|---|---:|---:|---:|---|
| 1 | PSEM-v1 | Indirect | **90.512%** | **89.981%** | 7.004 | 96.55 / 93.09 / 93.05 / 79.36 |
| 2 | 原始 OWT（对照） | Indirect | 90.386% | 89.616% | inf\* | 96.51 / 92.93 / 93.04 / 79.06 |
| 3 | PSEM-v1 | Direct | 90.110% | 89.118% | 10.277 | 96.53 / 92.95 / 91.89 / 79.06 |
| 4 | PSEM-v2a Query20 | Direct | 90.063% | 88.731% | **5.932** | 96.41 / 92.73 / 92.85 / 78.27 |
| 5 | PSEM-v3 Triplet+Loss3 | Indirect | 89.938% | 87.639% | 8.074 | 96.36 / 92.56 / 92.69 / 78.15 |
| 6 | 原始 OWT（对照） | Direct | 89.938% | 88.687% | inf\* | 96.45 / 92.69 / 92.94 / 77.66 |
| 7 | PSEM-v2a Query20 | Indirect | 89.888% | 88.421% | 6.904 | 96.38 / 92.70 / 92.70 / 77.77 |
| 8 | PSEM-v3 Triplet+Loss3 | Direct | 88.783% | 83.172% | 7.055 | 96.12 / 91.64 / 91.70 / 75.67 |
| 9 | PSEM+LossBalance-v2 | Indirect | 84.673% | 68.029% | 13.022 | 95.17 / 88.21 / 88.05 / 67.26 |
| 10 | LossBalance-v1（对照） | Direct | 84.186% | 66.665% | 9.672 | 94.82 / 87.68 / 88.01 / 66.23 |
| 11 | LossBalance-v1（对照） | Indirect | 83.299% | 64.387% | 14.753 | 94.64 / 87.17 / 86.98 / 64.40 |
| 12 | PSEM+LossBalance-v2 | Direct | 79.728% | 63.051% | 62.247 | 94.83 / 86.18 / 79.13 / 58.77 |

\* 原 OWT 合并汇总的每种读出各含 1 个无穷 HD95，因此正式宏平均记为 `inf`；Dice/NSD 从完整 200 病例逐病例文件恢复。

### PSEM-v3 的病例配对分析

以下为 10,000 次病例 bootstrap 的“PSEM-v3 减对照”宏平均 Dice 差值；置信区间只反映测试病例抽样，不包含不同训练 seed 的方差。

| 读出 | 对照 | 差值（百分点） | 95% CI | 结论 |
|---|---|---:|---|---|
| Direct | PSEM-v1 Direct | -1.326 | [-1.597, -0.988] | 明确下降 |
| Direct | 原始 OWT Direct | -1.154 | [-1.329, -0.972] | 明确下降 |
| Direct | PSEM-v2a Direct | -1.279 | [-1.436, -1.115] | 明确下降 |
| Direct | PSEM+LossBalance-v2 Direct | +9.056 | [8.675, 9.471] | 明确改善 |
| Indirect | PSEM-v1 Indirect | -0.574 | [-0.718, -0.410] | 明确下降 |
| Indirect | 原始 OWT Indirect | -0.448 | [-0.580, -0.307] | 明确下降 |
| Indirect | PSEM-v2a Indirect | +0.049 | [-0.107, 0.224] | 无法区分，视为持平 |
| Indirect | PSEM+LossBalance-v2 Indirect | +5.265 | [4.978, 5.610] | 明确改善 |

PSEM-v3 的 Indirect 比自身 Direct 高 1.155 个百分点，说明 Triplet/Delta 目标相对更偏向保留 Indirect；但由于 Direct 本身下降更大，这个差值不能单独证明分解质量改善。它没有提升 AutoPET 的总体上限，主要退化来自 `label_4`：Direct 75.67%，比 v1 Direct 低 3.39 个百分点。Direct HD95 从 v1 的 10.277 mm 降到 7.055 mm，但 Dice 和 NSD 同时下降，因此只能说极端边界误差改善，不能说总体分割更好。


## AbdAutoPET 2D 重建指标

按 Whole L2 从低到高排序。L2/LPIPS 越低越好，PSNR/SSIM 越高越好；`label_4-only L2` 用于检查本次最明显退化类别的重建质量。

| 排名 | 实验 | Whole L2 | Whole LPIPS | Whole PSNR | Whole SSIM | label_4-only L2 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | PSEM-v1 | **0.00015039** | **0.01907** | **38.624** | 0.91971 | 0.00019494 |
| 2 | PSEM-v3 Triplet+Loss3 | 0.00017368 | 0.02313 | 37.979 | 0.90798 | 0.00025977 |
| 3 | PSEM-v2a Query20 | 0.00017749 | 0.02356 | 37.908 | 0.92126 | **0.00019280** |
| 4 | PSEM+LossBalance-v2 | 0.00026088 | 0.03334 | 36.373 | **0.92377** | 0.00058690 |

PSEM-v3 的 Whole L2 比 v1 高 15.49%，但比 v2a 低 2.15%，说明总体像素误差与 v2a 接近、仍不如 v1。更关键的是 `label_4-only L2` 比 v1/v2a 分别高 33.25%/34.73%，与 `label_4` Direct Dice 的下降方向一致。另一方面，SSIM 与 Dice 并不单调一致：LossBalance-v2 的 Whole SSIM 最高但分割最差，因此不能用单个重建指标替代类别分解评估。

## AbdAutoPET 3D Fixfr4 排名

目前只有 PSEM-v1 完成同协议 200 病例评估。类别顺序为 `label_1/label_2/label_3/label_4`。

| 排名 | 实验 | 读出 | post Dice | post NSD | post HD95 (mm) | 4 类 post Dice |
|---:|---|---|---:|---:|---:|---|
| 1 | PSEM-v1 | Direct | **87.583%** | 82.450% | 16.677 | 94.54 / 91.71 / 90.33 / 73.75 |
| 2 | PSEM-v1 | Indirect | 87.519% | **83.100%** | **12.238** | 94.99 / 90.77 / 91.90 / 72.41 |

该表不支持“3D 不如 2D”的方法结论，因为 Fixfr4 的输入、训练和空间评估协议与 2D 不同；它只记录当前已完成结果。

## WORD 2D 排名

相同 24 病例、224 输入和 Common8 正式评估。类别顺序为脾脏/右肾/左肾/胆囊/食管/胰腺/肝脏/胃。

| 排名 | 实验 | 读出 | post Dice | post NSD | post HD95 (mm) | 8 类 post Dice |
|---:|---|---|---:|---:|---:|---|
| 1 | PSEM-v3 Triplet+Loss3 | Direct | **68.353%** | 57.516% | 15.133 | 87.84 / 87.39 / 86.65 / 25.11 / 46.65 / 52.98 / 91.49 / 68.72 |
| 2 | PSEM-v3 Triplet+Loss3 | Indirect | **68.244%** | **58.851%** | 26.001 | 86.89 / 85.06 / 82.67 / 32.13 / 41.86 / 56.38 / 91.59 / 69.37 |
| 3 | LossBalance-v3a（对照） | Direct | 68.216% | 55.495% | **15.079** | 87.25 / 86.55 / 85.15 / 24.94 / 46.42 / 53.18 / 91.29 / 70.94 |
| 4 | PSEM-v2 LossBalance-v3 | Direct | 67.209% | 54.177% | 15.080 | 86.79 / 85.38 / 84.09 / 24.11 / 47.05 / 51.20 / 90.58 / 68.46 |
| 5 | LossBalance-v1（对照） | Direct | 64.990% | 47.560% | 16.598 | 85.63 / 82.81 / 82.27 / 24.64 / 33.86 / 53.27 / 89.23 / 68.21 |
| 6 | 原始 OWT（对照） | Indirect | 58.309% | 49.866% | 80.962 | 88.31 / 86.83 / 85.55 / 0.00 / 0.00 / 43.73 / 92.28 / 69.78 |
| 7 | LossBalance-v3a（对照） | Indirect | 56.796% | 47.752% | 69.545 | 86.80 / 86.24 / 83.68 / 0.00 / 0.16 / 40.75 / 91.66 / 65.08 |
| 8 | 原始 OWT（对照） | Direct | 55.072% | 44.279% | inf | 86.00 / 79.30 / 79.98 / 0.00 / 0.00 / 38.43 / 91.02 / 65.84 |
| 9 | PSEM-v2a Query20 | Direct | 52.311% | 40.155% | inf | 83.56 / 66.20 / 79.83 / 0.00 / 2.72 / 31.95 / 90.36 / 63.86 |
| 10 | PSEM-v2 LossBalance-v3 | Indirect | 52.216% | 41.632% | 84.054 | 84.49 / 81.76 / 79.66 / 0.00 / 0.00 / 21.16 / 90.36 / 60.30 |
| 11 | PSEM-v2a Query20 | Indirect | 49.889% | 38.862% | 91.771 | 84.74 / 79.90 / 79.25 / 0.00 / 0.00 / 0.41 / 90.99 / 63.83 |
| 12 | LossBalance-v1（对照） | Indirect | 41.575% | 18.809% | 372.641 | 57.08 / 48.45 / 50.05 / 5.79 / 8.84 / 27.99 / 81.56 / 52.85 |
| 13 | PSEM-v1 | Direct | 36.266% | 29.127% | 319.094 | 37.75 / 43.57 / 45.11 / 10.83 / 9.89 / 45.87 / 46.97 / 50.13 |
| 14 | PSEM-v1 | Indirect | 34.866% | 17.367% | 355.002 | 48.66 / 40.81 / 42.65 / 0.78 / 0.74 / 18.28 / 79.73 / 47.28 |
| 15 | PSEM+LossBalance-v2 | Indirect | 21.129% | 7.103% | 375.216 | 28.79 / 20.57 / 21.84 / 1.49 / 0.18 / 1.17 / 63.60 / 31.39 |
| 16 | PSEM+LossBalance-v2 | Direct | 8.880% | 6.992% | 424.928 | 8.26 / 6.75 / 7.16 / 0.41 / 2.22 / 2.70 / 31.17 / 12.36 |

PSEM-v3 在 WORD 上相对 PSEM-v2+LossBalance-v3 的 Direct 增益为 1.14 个百分点，相对独立 LossBalance-v3a 的 Direct 增益仅 0.14 个百分点；关键变化是 Indirect 从 52.22% 提高到 68.24%（相对独立 LossBalance-v3a 也提高 11.45 点），并与 Direct 只差 0.11 个百分点。这直接支持 Triplet 的设计目标：让 Direct、Context、Plus 在同一源样本上成对训练，使 `Plus - Context` 与目标器官响应一致。Indirect 仍明显依赖后处理，因此报告时必须同时保留 raw/post，而不能把全部提升归因于网络。

## 已完成实验登记

| 版本 | 核心改动 | 数据集与正式评估 | 当前判断 |
|---|---|---|---|
| PSEM-v1 | 只允许 GT-present 类进入逐样本穷举删除日程；保持原 L2+LPIPS | AutoPET 2D、AutoPET 3D Fixfr4、WORD 2D | AutoPET 2D 当前最优；WORD 存在训练/推理查询状态错配 |
| PSEM-v2a Query20 | 20-slot 全 token-bank 查询日程，包含正/负 Direct、leave-one-out、whole、background、organs 和随机子集 | AutoPET 2D、WORD 2D | AutoPET 与 v1 接近；WORD 大器官恢复，但小器官仍严重失败 |
| PSEM+LossBalance-v2 | PSEM-v1 加按样本频率加权的 ROI 损失 | AutoPET 2D、WORD 2D | 两个数据集均明显退化，判定该组合无效 |
| PSEM-v2 LossBalance-v3 | Query20 加只优化 present-and-kept 正 ROI 的 Loss3 | WORD 2D | Direct 提升到 67.21%，但 Indirect 仍只有 52.22% |
| PSEM-v3 Triplet+Loss3 | 每个源样本构造 Direct/Context/Plus，优化三分支 Loss3 与 `0.1 × Delta loss` | WORD 2D；AutoPET 2D | WORD 修复 Indirect；AutoPET 与 v2a 持平、低于 v1，收益不具跨数据集一致性 |

### PSEM-v3 正式运行

| 数据集 | 训练 | 评估 | 资源/规模 | 状态 |
|---|---|---|---|---|
| WORD 2D | XEC Job `114767` | XEC Job `114768` | 2×A800；1200 epochs；24 病例评估 | 完成 |
| AbdAutoPET 2D | sifansong/XEC Job `121484` | sifansong/XEC Job `124936` | 4×A800；有效 source batch 192；1200 epochs；200 病例评估 | 完成；BF16/LPIPS-FP32 数值修复版本 |

AutoPET 原 FP16 训练链曾在约 epoch 500 附近出现非有限值。最终有效版本保留模型、数据、Triplet/Loss3 定义和优化预算，只修改数值执行策略：BF16 autocast、LPIPS FP32、梯度裁剪与逐分量有限值检查。该变化必须随 checkpoint 一起记录，不能把最终模型描述成未经修复的原 FP16 运行。

## 结果文件与详细记录

- PSEM-v1：`docs/PSEM_V1_EXPERIMENT_LOG.md`
- PSEM-v2a：`docs/PSEM_V2_EXPERIMENT_LOG.md`
- PSEM-v3：`docs/PSEM_V3_TRIPLET_EXPERIMENT.md`
- 正式结果以各实验目录的 `step3_segmentation_summary.csv`、`step3_segmentation_per_case.csv` 和 `protocol.json` 为准。

## 下一步

1. PSEM-v3 AutoPET 至少补 2 个 seed；当前病例 bootstrap 不包含训练随机性，不能据单 seed 宣称稳定差异。
2. 优先分析 AutoPET `label_4` 的 false negative、预测体积比和 Direct/Indirect 差分图，确认 Direct 回退来自类 token 读出还是 Triplet 的预算分配。
3. 若要判断 PSEM-v3 是否适用于 3D，需要在 AutoPET 3D Fixfr4 上补同协议 v3 对照；不能拿 2D 与 3D 的当前单次结果直接排名。
4. WORD 继续报告 raw/post 两套结果，并做固定后处理的消融，量化 Indirect 提升中模型与后处理各自的贡献。
