# OWT 正式实验台账与指标排名

最后核对：2026-08-03 CST
维护位置：`experiment/psem-v1` worktree
用途：集中记录已经完成训练和完整推理的 OWT 实验，作为后续横向比较和实验有效性判断的唯一入口。

## 1. 纳入标准和指标口径

只有同时满足以下条件的实验才进入“正式实验”表：

1. 完成计划中的 1,200 epochs，而不是 smoke 或 tiny overfit；
2. `checkpoint-1199.pth` 存在；
3. 完成完整测试集 Step 2 重建评估和 Step 3 伪分割评估；
4. 没有已知的数据、标签或 mask 语义错误；
5. 能定位训练 Job、评估 Job、训练日志、checkpoint 和逐病例结果。

本文所有分割宏平均均从 `step3_segmentation_per_case.csv` 重新计算：先对每个类别的所有测试病例求平均，再对类别做等权宏平均。表中的 Dice 和 NSD 以百分数表示，例如 `90.11` 表示 `0.9011`。

- `Direct`：仅保留目标类别 token 得到的重建信号。
- `Indirect`：输入减去“不含目标类别 token”的重建信号。
- `raw`：阈值化后的原始结果。
- `post`：连通域后处理后的结果。
- 主要排名指标：`Direct post macro Dice`。
- 次要判断：Direct raw、Indirect、逐类别 Dice、NSD 和重建误差。
- 不跨数据集直接排名；2D 和 3D 默认分组排名。
- 所有结果都是单次训练，没有多随机种子均值和显著性检验，因此“有效”表示当前受控实验中有效，不等同于统计显著。

正式推理的共同设置：重建阈值 `0.02`、分割阈值 `0.15`、NSD 容差 `3 mm`。AbdAutoPET 的最小连通域为 20 voxels，WORD/BTCV Common8 为 100 voxels。3D Fixfr4 使用 stride-1 滑窗并对重叠预测求平均。

### 状态标记

| 标记 | 含义 |
|---|---|
| 基线 | 用于同数据集、同维度变体的参照，不评价自身是否“创新有效” |
| 有效 | 主要指标和目标类别明显改善，且没有决定性的协议问题 |
| 部分有效 | 只在部分提取模式、后处理结果或部分类别上改善 |
| 无效 | 相比匹配基线主要指标明显下降 |
| 无法判断 | 缺少同组对照、重复实验或正式评估 |
| 排除 | 已知实现/语义错误，不允许进入正式排名 |

## 2. 当前结论总览

| 方法与数据 | 当前判断 | 核心证据 |
|---|---|---|
| LossBalance-v1，WORD 2D | **有效于 Direct，整体属于部分有效** | Direct post `55.07 -> 64.99`，胆囊 `0 -> 24.64`、食管 `0 -> 33.86`、胰腺 `38.43 -> 53.27`；但 Indirect post `58.31 -> 41.57` |
| LossBalance-v1，WORD 3D Fixfr4 | **有效** | Direct post `59.93 -> 67.57`，Indirect post `60.94 -> 62.39`；胆囊、食管从 0 提升到 `26.90/39.00` |
| PSEM-v1，AbdAutoPET 2D | **有限/边际有效** | Direct post `89.94 -> 90.11`、Indirect post `90.39 -> 90.51`、whole L2 降低 3.99%；但 Direct raw 下降 0.40 点，增益小于 0.2 点且只有单次训练 |
| PSEM-v1，WORD 2D | **无效** | Direct post 55.07 -> 36.27、Indirect post 58.31 -> 34.87；whole L2 是基线 5.45 倍，Direct 预测/GT 体积比宏平均为 2.664 |
| LossBalance-v1，AbdAutoPET 2D | **无效** | Direct post `89.94 -> 84.19`，Indirect post `90.39 -> 83.30`，whole L2 上升 76.82% |
| OWT 3D Fixfr4，WORD | **3D 上下文有效但未解决最小器官** | 相对 OWT 2D，Direct post `+4.86`、Indirect post `+2.63`；胆囊和食管仍为 0 |
| OWT 3D Fixfr4，BTCV | **相对 2D 有改善，但绝对结果仍不可用** | Direct post `10.39 -> 34.16`，但胆囊/食管仍为 0，且测试集只有 6 例 |

## 3. 已完成正式训练任务

| ID | 训练 Job | 评估 Job | 方法 | 数据集 | 输入 | GPUs | batch 参数 | 训练耗时 | 最终产物 |
|---|---:|---:|---|---|---|---:|---:|---:|---|
| E01 | 1512521 | 1549968 + 1549972 | 原始 OWT | AbdAutoPET | 2D | 2 | 96 | 2d 10:29:14 | ckpt1199 + 200例分块合并评估 |
| E02 | 1527637 | 1539085 | 原始 OWT | WORD Common8 | 2D | 2 | 32 | 19:22:16 | ckpt1199 + 24例评估 |
| E03 | 1539406 | 1539427 | 原始 OWT | WORD Common8 | 3D Fixfr4 | 2 | 32 | 2d 04:52:09 | ckpt1199 + 24例评估 |
| E04 | 1527638 | 1539086 | 原始 OWT | BTCV Common8 | 2D | 2 | 48 | 03:37:44 | ckpt1199 + 6例评估 |
| E05 | 1539407 | 1539428 | 原始 OWT | BTCV Common8 | 3D Fixfr4 | 2 | 32 | 11:04:11 | ckpt1199 + 6例评估 |
| E06 | 1560195 | 1563822 | LossBalance-v1 | WORD Common8 | 2D | 2 | 32 | 19:31:57 | ckpt1199 + 24例评估 |
| E07 | 1560196 | 1563824 | LossBalance-v1 | WORD Common8 | 3D Fixfr4 | 2 | 32 | 2d 07:22:58 | ckpt1199 + 24例评估 |
| E08 | 1564397 | 1564399 | LossBalance-v1 | AbdAutoPET | 2D | 2 | 96 | 2d 11:37:09 | ckpt1199 + 200例评估 |
| E09 | 1563805 | 1563831 | PSEM-v1 | AbdAutoPET | 2D | 2 | 96 | 2d 11:23:54 | ckpt1199 + 200例评估 |
| E10 | 1605670 | 1605673 | PSEM-v1 | WORD Common8 | 2D | 2 | 32 | 19:22:43 | ckpt1199 + 24例评估 |

E01-E09 的九个 `checkpoint-1199.pth` 已于 2026-07-29 再次验证存在；E10 的 checkpoint、protocol 和完整逐病例 CSV 已于 2026-08-02 验证存在。

### 方法差异

| 方法 | 相对原始 OWT 的唯一主要改动 | 训练目标 |
|---|---|---|
| 原始 OWT | 无 | 全图 L2 + LPIPS；batch 共用随机类别 mask |
| LossBalance-v1 | 增加逐样本、逐前景器官等权 ROI-L2；背景权重 0 | 全图 L2 + ROI-L2 + LPIPS |
| PSEM-v1 | 每个样本仅对实际存在类别执行确定性的穷举 mask；变长 token 物理打包并严格屏蔽 padding | 与原始 OWT 相同的全图 L2 + LPIPS |

LossBalance-v1 没有新增模型参数；PSEM-v1 也没有新增、删除或改变参数形状。

## 4. 分组指标排名

`D/I 排名` 分别表示 Direct post 和 Indirect post 在同数据集、同维度组内的排名。`ΔD/ΔI` 是相对同组原始 OWT 的绝对百分点变化。

### 4.1 AbdAutoPET 2D

| D/I 排名 | ID | 方法 | Direct raw | Direct post | ΔD | Direct post NSD | Indirect raw | Indirect post | ΔI | Indirect post NSD | 判断 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 / 1 | E09 | PSEM-v1 | 89.10 | **90.11** | +0.17 | **89.12** | **88.61** | **90.51** | +0.13 | **89.98** | 有限/边际有效 |
| 2 / 2 | E01 | 原始 OWT | **89.50** | 89.94 | 0.00 | 88.69 | 88.14 | 90.39 | 0.00 | 89.62 | 基线 |
| 3 / 3 | E08 | LossBalance-v1 | 83.34 | 84.19 | -5.75 | 66.66 | 78.97 | 83.30 | -7.09 | 64.39 | 无效 |

这里 PSEM 的 post Dice 排名第一，但 Direct raw 仍由原始 OWT 第一。因此不能仅凭 `+0.17` 点宣称稳定提升，至少还需要重复种子验证。

### 4.2 WORD Common8 2D

| D/I 排名 | ID | 方法 | Direct raw | Direct post | ΔD | Direct post NSD | Indirect raw | Indirect post | ΔI | Indirect post NSD | 判断 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 / 2 | E06 | LossBalance-v1 | **63.85** | **64.99** | +9.92 | **47.56** | 19.69 | 41.57 | -16.74 | 18.81 | Direct 有效，Indirect 退化 |
| 2 / 1 | E02 | 原始 OWT | 55.30 | 55.07 | 0.00 | 44.28 | **47.23** | **58.31** | 0.00 | **49.87** | 基线 |
| 3 / 3 | E10 | PSEM-v1 | 34.48 | 36.27 | -18.80 | 29.13 | 17.66 | 34.87 | -23.44 | 17.37 | 无效，查询状态失配 |

### 4.3 WORD Common8 3D Fixfr4

| D/I 排名 | ID | 方法 | Direct raw | Direct post | ΔD | Direct post NSD | Indirect raw | Indirect post | ΔI | Indirect post NSD | 判断 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 / 1 | E07 | LossBalance-v1 | **67.01** | **67.57** | +7.64 | 52.71 | 29.45 | **62.39** | +1.45 | 43.25 | 有效，Direct 增益最明确 |
| 2 / 2 | E03 | 原始 OWT | 60.00 | 59.93 | 0.00 | **52.89** | **48.26** | 60.94 | 0.00 | **53.58** | 基线 |

虽然 E07 的 Dice 更高，但 NSD 和 Indirect raw 下降，说明它改善了阈值/连通域后的区域重叠，却没有全面改善边界质量和原始 Indirect 信号。

### 4.4 BTCV Common8

BTCV 当前只有原始 OWT，因此下表是 2D/3D 输入消融，不是不同创新方法的排名。

| ID | 输入 | Direct raw | Direct post | Direct post NSD | Indirect raw | Indirect post | Indirect post NSD | 判断 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| E05 | 3D Fixfr4 | **35.42** | **34.16** | **21.72** | **8.98** | **12.24** | **4.58** | 优于 2D，但绝对性能低 |
| E04 | 2D | 10.39 | 10.39 | 5.07 | 7.99 | 10.43 | 4.21 | 多类别塌缩 |

## 5. 逐类别 post Dice

每格为 `Direct / Indirect`。这部分用于判断宏平均提升究竟来自目标小器官，还是来自大器官和后处理。

### 5.1 AbdAutoPET 2D

AbdAutoPET 仓库中没有可靠的 label ID 到器官名称映射，因此保留 `label_1` 至 `label_4`。

| ID / 方法 | label_1 | label_2 | label_3 | label_4 | Macro |
|---|---:|---:|---:|---:|---:|
| E09 PSEM-v1 | **96.53 / 96.55** | **92.95 / 93.09** | 91.89 / **93.05** | **79.06 / 79.36** | **90.11 / 90.51** |
| E01 原始 OWT | 96.45 / 96.51 | 92.69 / 92.93 | **92.94 / 93.04** | 77.66 / 79.06 | 89.94 / 90.39 |
| E08 LossBalance-v1 | 94.82 / 94.64 | 87.68 / 87.17 | 88.01 / 86.98 | 66.23 / 64.40 | 84.19 / 83.30 |

PSEM 的主要收益来自 `label_4`：Direct `+1.40`、Indirect `+0.30`；但 `label_3` Direct 下降 `1.05` 点。

### 5.2 WORD Common8 2D

| 器官 | E02 OWT D/I | E06 LossBalance-v1 D/I | Direct 变化 | Indirect 变化 |
|---|---:|---:|---:|---:|
| spleen | 86.00 / 88.31 | 85.63 / 57.08 | -0.37 | -31.23 |
| right kidney | 79.30 / 86.83 | 82.81 / 48.45 | +3.51 | -38.38 |
| left kidney | 79.98 / 85.55 | 82.27 / 50.05 | +2.29 | -35.50 |
| gallbladder | 0.00 / 0.00 | **24.64 / 5.79** | **+24.64** | +5.79 |
| esophagus | 0.00 / 0.00 | **33.86 / 8.84** | **+33.86** | +8.84 |
| pancreas | 38.43 / 43.73 | **53.27 / 27.99** | **+14.84** | -15.74 |
| liver | 91.02 / 92.28 | 89.23 / 81.56 | -1.79 | -10.72 |
| stomach | 65.84 / 69.78 | 68.21 / 52.85 | +2.37 | -16.93 |

结论：LossBalance-v1 2D 确实让原本完全漏检的胆囊和食管产生了可用 Direct 信号，但同时破坏了 Indirect 分解，因此不能简单称为“所有模式全面提升”。

### 5.3 WORD Common8 3D Fixfr4

| 器官 | E03 OWT D/I | E07 LossBalance-v1 D/I | Direct 变化 | Indirect 变化 |
|---|---:|---:|---:|---:|
| spleen | 89.24 / 89.92 | 86.75 / 83.28 | -2.49 | -6.64 |
| right kidney | 89.49 / 89.05 | 85.53 / 78.23 | -3.96 | -10.82 |
| left kidney | 88.25 / 87.87 | 85.69 / 78.65 | -2.56 | -9.22 |
| gallbladder | 0.00 / 0.00 | **26.90 / 23.11** | **+26.90** | **+23.11** |
| esophagus | 0.00 / 0.00 | **39.00 / 28.62** | **+39.00** | **+28.62** |
| pancreas | 47.61 / 54.12 | **55.51 / 47.87** | **+7.90** | -6.25 |
| liver | 92.02 / 93.15 | 90.25 / 89.78 | -1.77 | -3.37 |
| stomach | 72.81 / 73.44 | 70.90 / 69.56 | -1.91 | -3.88 |

结论：3D LossBalance-v1 用少量大器官退化换来了很大的小器官提升，Direct 宏平均净增 7.64 点；它是当前最明确的有效损失实验。

### 5.4 BTCV 原始 OWT

| 器官 | E04 2D D/I | E05 3D Fixfr4 D/I |
|---|---:|---:|
| spleen | 0.00 / 8.27 | 60.07 / 14.05 |
| right kidney | 0.00 / 6.82 | 34.68 / 8.85 |
| left kidney | 0.00 / 6.68 | 37.24 / 8.67 |
| gallbladder | 0.00 / 0.25 | 0.00 / 0.27 |
| esophagus | 0.00 / 0.39 | 0.00 / 0.27 |
| pancreas | 0.00 / 0.90 | 8.59 / 0.77 |
| liver | 83.11 / 51.48 | 85.07 / 50.81 |
| stomach | 0.00 / 8.68 | 47.61 / 14.20 |

## 6. Step 2 whole reconstruction 指标

L1、L2、LPIPS 越低越好；PSNR、SSIM 越高越好。`Organs L2` 是 organs-only 重建在全体素上的 L2，仍会受到大量零背景影响，不能替代器官 ROI 指标。

| ID | 方法 / 数据 / 输入 | L1 | L2 | LPIPS | PSNR | SSIM | Organs L2 |
|---|---|---:|---:|---:|---:|---:|---:|
| E09 | PSEM / AbdAutoPET / 2D | **0.006772** | **0.00015039** | 0.01907 | **38.62** | **0.9197** | 0.00137765 |
| E01 | OWT / AbdAutoPET / 2D | 0.006856 | 0.00015664 | 未计算 | 38.42 | 0.9196 | **0.00132167** |
| E08 | LossBalance / AbdAutoPET / 2D | 0.009831 | 0.00027698 | 0.03479 | 35.98 | 0.8063 | 0.00216790 |
| E02 | OWT / WORD / 2D | **0.006909** | **0.00031396** | **0.01358** | **35.31** | **0.9660** | **0.00067945** |
| E06 | LossBalance / WORD / 2D | 0.012416 | 0.00140731 | 0.03961 | 28.88 | 0.9240 | 0.00119951 |
| E10 | PSEM / WORD / 2D | 0.013248 | 0.00171145 | 0.04080 | 28.06 | 0.9218 | 0.00667906 |
| E03 | OWT / WORD / 3D | **0.007221** | **0.00032112** | **0.01692** | **35.25** | **0.9535** | **0.00059233** |
| E07 | LossBalance / WORD / 3D | 0.009645 | 0.00076073 | 0.03270 | 31.57 | 0.9470 | 0.00087244 |
| E05 | OWT / BTCV / 3D | 0.033626 | **0.00626027** | 0.15078 | 22.26 | 0.6977 | **0.00269394** |
| E04 | OWT / BTCV / 2D | **0.033183** | 0.00626063 | **0.10914** | **22.32** | **0.7195** | 0.00336075 |

重要现象：LossBalance-v1 在 WORD 上提高分割 Dice 的同时显著恶化 whole reconstruction。WORD 2D whole L2 相对基线上升 348.24%，3D 上升 136.90%。这说明 ROI loss 改变了 token 的类别可分性，但牺牲了全图灰度重建；评价该方法不能只看训练 loss 或 Step 2 whole L2。

## 7. 最终 epoch 训练指标

| ID | Global L2 | ROI-L2 | LPIPS | 实际优化总和约值 |
|---|---:|---:|---:|---:|
| E01 | 0.000872 | — | 0.019887 | 0.020759 |
| E02 | 0.000730 | — | 0.016920 | 0.017650 |
| E03 | 0.000694 | — | 0.016783 | 0.017477 |
| E04 | 0.004199 | — | 0.061805 | 0.066003 |
| E05 | 0.003873 | — | 0.059084 | 0.062957 |
| E06 | 0.001614 | 0.005467 | 0.031965 | 0.039046 |
| E07 | 0.001161 | 0.003731 | 0.025496 | 0.030387 |
| E08 | 0.001609 | 0.002883 | 0.031539 | 0.036031 |
| E09 | 0.001084 | — | 0.020205 | 0.021289 |
| E10 | 0.001878 | — | 0.037961 | 0.039839 |

不同损失定义下的训练总和不能直接排名。LossBalance 的总和包含 ROI-L2，数值天然高于 OWT/PSEM；真正可比较的是统一 Step 2/Step 3 推理结果。

## 8. 结果与 checkpoint 索引

| ID | Checkpoint | 完整评估目录 |
|---|---|---|
| E01 | [checkpoint-1199.pth](/gpfs/work/aac/bolinren19/2026-07/OD_OWT/Results/AbdAutoPet_2D/Token_mae_vit_base_patch16-LA_1e-4_4_224_20_v11_v01_L2-LPIPS_GPU2_96_1200/checkpoint-1199.pth) | [OWT AbdAutoPET 2D Eval](/gpfs/work/aac/bolinren19/2026-07/OD_OWT/Results/AbdAutoPet_Eval/2D/OWT-Joint-C4-Recon_ckpt1199) |
| E02 | [checkpoint-1199.pth](/gpfs/work/aac/bolinren19/2026-07/OD_OWT/Results/Common8_Full/WORD_2D/Common8Full_HU-175_250_Token_mae_vit_base_patch16-LA_1e-4_8_224_20_v11_v01_L2-LPIPS_GPU2_32_1200/checkpoint-1199.pth) | [OWT WORD 2D Eval](/gpfs/work/aac/bolinren19/2026-07/OD_OWT/Results/Common8_Eval/WORD_2D/OWT-Joint-C8-Recon_ckpt1199) |
| E03 | [checkpoint-1199.pth](/gpfs/work/aac/bolinren19/2026-07/OD_OWT/Results/Common8_Full/WORD_3D_Fixfr4/Common8Full_HU-175_250_Token_mae_vit_basefix16_patch16-LA_1e-4_8_224_20_v11_v01-3D-Fixfr4-TS1_L2-LPIPS_GPU2_32_1200/checkpoint-1199.pth) | [OWT WORD 3D Eval](/gpfs/work/aac/bolinren19/2026-07/OD_OWT/Results/Common8_Eval/WORD_3D_Fixfr4/OWT-Joint-C8-Recon3D_ckpt1199) |
| E04 | [checkpoint-1199.pth](/gpfs/work/aac/bolinren19/2026-07/OD_OWT/Results/Common8_Full/BTCV_2D/Common8Full_HU-175_250_Token_mae_vit_base_patch16-LA_1e-4_8_224_20_v11_v01_L2-LPIPS_GPU2_48_1200/checkpoint-1199.pth) | [OWT BTCV 2D Eval](/gpfs/work/aac/bolinren19/2026-07/OD_OWT/Results/Common8_Eval/BTCV_2D/OWT-Joint-C8-Recon_ckpt1199) |
| E05 | [checkpoint-1199.pth](/gpfs/work/aac/bolinren19/2026-07/OD_OWT/Results/Common8_Full/BTCV_3D_Fixfr4/Common8Full_HU-175_250_Token_mae_vit_basefix16_patch16-LA_1e-4_8_224_20_v11_v01-3D-Fixfr4-TS1_L2-LPIPS_GPU2_32_1200/checkpoint-1199.pth) | [OWT BTCV 3D Eval](/gpfs/work/aac/bolinren19/2026-07/OD_OWT/Results/Common8_Eval/BTCV_3D_Fixfr4/OWT-Joint-C8-Recon3D_ckpt1199) |
| E06 | [checkpoint-1199.pth](/gpfs/work/aac/bolinren19/2026-07/OD_OWT/Results/LossBalance_v1/WORD_2D/LossBalanceV1_ROI1.0_BG0.0_Token_mae_vit_base_patch16-LA_1e-4_8_224_20_v11_v01_L2-LPIPS_GPU2_32_1200/checkpoint-1199.pth) | [LossBalance WORD 2D Eval](/gpfs/work/aac/bolinren19/2026-07/OD_OWT/Results/LossBalance_v1/Eval/WORD_2D/ckpt1199) |
| E07 | [checkpoint-1199.pth](/gpfs/work/aac/bolinren19/2026-07/OD_OWT/Results/LossBalance_v1/WORD_3D_Fixfr4/LossBalanceV1_ROI1.0_BG0.0_Token_mae_vit_basefix16_patch16-LA_1e-4_8_224_20_v11_v01-3D-Fixfr4-TS1_L2-LPIPS_GPU2_32_1200/checkpoint-1199.pth) | [LossBalance WORD 3D Eval](/gpfs/work/aac/bolinren19/2026-07/OD_OWT/Results/LossBalance_v1/Eval/WORD_3D_Fixfr4/ckpt1199) |
| E08 | [checkpoint-1199.pth](/gpfs/work/aac/bolinren19/2026-07/OD_OWT/Results/LossBalance_v1/AbdAutoPet_2D/LossBalanceV1_ROI1.0_BG0.0_Token_mae_vit_base_patch16-LA_1e-4_C4_224_T20_v11_v01_L2-LPIPS_GPU2_B96_E1200/checkpoint-1199.pth) | [LossBalance AbdAutoPET 2D Eval](/gpfs/work/aac/bolinren19/2026-07/OD_OWT/Results/LossBalance_v1/Eval/AbdAutoPet_2D/ckpt1199) |
| E09 | [checkpoint-1199.pth](/gpfs/work/aac/bolinren19/OD_OWT/Results/PSEM_v1/AbdAutoPet_2D/PSEM_v1_Token_mae_vit_base_patch16-LA_1e-4_C4_224_T20_v11_v01_L2-LPIPS_GPU2_B96_E1200/checkpoint-1199.pth) | [PSEM AbdAutoPET 2D Eval](/gpfs/work/aac/bolinren19/OD_OWT/Results/PSEM_v1/Eval/AbdAutoPet_2D/ckpt1199) |
| E10 | [checkpoint-1199.pth](/gpfs/work/aac/bolinren19/OD_OWT/Results/PSEM_v1/WORD_2D/PSEM_v1_Common8_HU-175_250_Token_mae_vit_base_patch16-LA_1e-4_C8_224_T20_v11_v01_L2-LPIPS_GPU2_B32_E1200/checkpoint-1199.pth) | [PSEM WORD 2D Eval](/gpfs/work/aac/bolinren19/OD_OWT/Results/PSEM_v1/Eval/WORD_2D/ckpt1199) |

代码版本：原始 OWT 基线为 `main@6cdf9b5`；PSEM-v1 正式训练对应 `experiment/psem-v1` 的 PSEM 实现提交链 `7c1217a`、`71a346b`、`8a7da1f`。LossBalance-v1 使用 main worktree 中隔离的 `*_lossbalance.py` 与 `util/organ_balanced_loss.py` 文件，但这些文件目前仍未形成独立 Git 提交；迁移或复现前必须将它们提交或打包保存。

## 9. 已运行但排除正式排名的任务

| 任务 | Job | 状态 | 排除原因 |
|---|---|---|---|
| CropMix-v1 WORD 2D | train 1552984，eval 1554939 | 训练和评估完成 | focus 类被加入“删除 token”集合，且 batch 内 focus 类并集被应用到所有样本；监督语义错误 |
| CropMix-v1 WORD 3D | train 1552985，eval 1554940 | 训练和评估完成 | 与 2D 相同的语义错误 |
| 早期 WORD/BTCV legacy 配置 | 1524575、1524576 等 | 完成 | 不是后来统一的 Common8 配置，不能与 E02-E05 公平比较 |
| 各类 smoke/tiny overfit | 多个 | 通过或完成 | 只验证代码可运行，不代表模型性能 |
| OrganSlotBank Gate A/真实数据 smoke | 1561634、1561641、1567673 | 完成 | 尚未进行 1,200 epoch 正式训练和完整评估 |

CropMix 的数值不得用于论文表格、方法排名或“有效/无效”结论；它只能作为实现错误案例保留。

## 10. 尚未进入正式排名的候选任务

以下是 2026-08-03 的队列与结果快照。状态会变化，只有训练和评估都完成后才能分配新的 E 编号。

| 方法 / 数据 | 训练链 | 当时状态 | 进入正式表还缺什么 |
|---|---|---|---|
| PSEM-v1 AbdAutoPET 3D | 原2卡 1563807 -> 4卡续训 1567343 -> eval 1563842 | 训练和评估均完成 | 已具备完整结果，待纳入正式排名 |
| LossBalance-v2 WORD 2D | train 1586123 -> eval 1623976 | 训练和评估均完成 | 已具备完整结果，待纳入正式排名 |
| LossBalance-v2 WORD 3D | 1586124 | 已取消，未启动（00:00:00） | 按“先完成2D统一评估，证明有效后再开3D”策略暂缓 |
| VQGAN-CNN（无 codebook）WORD 2D | train 1586125 -> eval 1634607 | 训练完成，评估等待 Priority | 完整统一推理 |
| PSEM+LossBalance-v2 AbdAutoPET 2D | smoke 1605960 -> train 1605961 -> eval 1634606 | 训练完成，评估等待 Priority | 完整统一推理 |
| PSEM+LossBalance-v2 WORD 2D | smoke 1605962 -> train 1605963 -> eval 1634605 | 训练完成，评估等待 Priority | 完整统一推理 |
| LossBalance-v3a WORD 2D | smoke 1627035 -> train 1627036 -> eval 1627039 | 训练和评估均完成 | 已具备完整结果，待纳入正式排名 |
| PSEM-v2a WORD 2D | smoke 1629558 -> train 1629560 -> eval 1634604 | 训练完成，评估等待 Priority | 完整统一推理 |
| PSEM-v2a AbdAutoPET 2D | smoke 1629559 -> train 1629561 -> eval 1629564 | 训练运行中，依赖评估等待 | 训练结束后自动启动统一推理 |

### 2026-08-03 补齐的评估任务

| 评估 Job | 方法 / 数据 | 账户 / QOS | GPU | 提交后状态 |
|---:|---|---|---:|---|
| 1634604 | PSEM-v2a WORD 2D | sifansong / 4a800 | 1 | PENDING Priority |
| 1634605 | PSEM+LossBalance-v2 WORD 2D | angelosstefanidis / 8a800 | 1 | PENDING Priority |
| 1634606 | PSEM+LossBalance-v2 AbdAutoPET 2D | angelosstefanidis / 8a800 | 1 | PENDING Priority |
| 1634607 | VQGAN-CNN（无 codebook）WORD 2D | sifansong / 4a800 | 1 | PENDING Priority |

LossBalance-v2 WORD 2D 的统一评估 Job `1623976` 已于 2026-07-31 完成，因而没有重复提交。PSEM-v2a AbdAutoPET 2D 已有依赖评估 Job `1629564`，也没有重复提交。

### 两项新完成评估的初步结论

- LossBalance-v3a WORD 2D：Direct post Dice 宏平均 `68.22`，高于 OWT `55.07`、v1 `64.99` 和 v2 `65.44`；Indirect post 为 `56.80`，较 v1 `41.57`、v2 `49.67` 明显恢复，但仍低于 OWT `58.31`。Direct 食管达到 `46.42`，胆囊 `24.94`，胰腺 `53.18`。v3a 是目前 LossBalance 系列最均衡的版本，但 Indirect 胆囊仍为 `0`、食管仅 `0.16`，不能说已经解决全部小器官问题。
- PSEM-v1 AbdAutoPET 3D Fixfr4：Direct/Indirect post Dice 宏平均分别为 `87.58/87.52`。与同一 PSEM 的 2D `90.11/90.51` 相比下降 `2.53/2.99` 点，主要下降来自 label 4（Direct `79.06 -> 73.75`，Indirect `79.36 -> 72.41`）。当前没有原始 OWT AbdAutoPET 3D 的同协议正式基线，所以该结果只能说明 PSEM 3D 的绝对表现和相对 PSEM 2D 的差异，不能单独证明 PSEM 在 3D 有效或无效。

后续实验资源策略（2026-08-03）：新损失、mask 或架构先完成 2D smoke、2D 正式训练和统一推理；只有 2D 的 Direct/Indirect、关键小器官 Dice 与重建指标证明有意义且无灾难性退化后，才提交对应 3D 正式训练。3D smoke 仅作为代码可运行性验证。

## 11. 已知数据质量问题

原始 OWT AbdAutoPET 合并目录中的 `step3_segmentation_summary.csv` 被错误写成 `cases=0` 和 `inf`，但同目录的 `step3_segmentation_per_case.csv` 完整包含 200 个唯一病例、1,600 行结果，`Result.txt` 也记录了正确的合并结果。本文 E01 的所有 Step 3 数值均由逐病例 CSV 按统一口径重算，而不是读取这个损坏的 summary 文件。

E01 的 LPIPS 在原始正式推理中通过 `skip_lpips=true` 跳过，因此表中标为“未计算”，不能把文件中的 `inf` 当成模型性能。

HD95 在预测或 GT 为空时可能为 `inf`，宏平均会失去解释性，因此本文不据此排序。逐病例 HD95 仍保留在各评估目录的原始 CSV 中。

## 12. 后续维护规则

每个新正式任务完成后按以下顺序更新：

1. 用 `sacct` 核对训练和评估 Job 均为 `COMPLETED`，记录 GPU 数和耗时；
2. 验证最终 checkpoint 存在，记录代码分支和 commit；
3. 核对 `protocol.json` 的测试 CSV、归一化、阈值、后处理和病例数；
4. 从逐病例 CSV 重新计算 class mean 和 class-macro，而不是盲信 summary；
5. 先按数据集和维度加入分组排名，再记录相对同组 OWT 的绝对百分点变化；
6. 同时检查 Direct raw/post、Indirect raw/post、逐类别 Dice、NSD 和 Step 2，不以单一数字下结论；
7. 有语义错误、错误类别配置或不完整测试集的任务移入“排除表”，不得删除证据；
8. 单次小于 1 个百分点的变化默认标记为“有限/待复现”，至少补充重复种子后再升级为“有效”。
