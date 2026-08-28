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
