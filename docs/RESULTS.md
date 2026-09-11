# 已记录结果与证据

本页合并既有已提交文档，不是本次重新推理或重新审核全部 JSON。
只称“项目内已记录结果”；没有外部 PCDD 同协议复核，不能声称公开 SOTA。
八类顺序：脾、右肾、左肾、胆囊、食管、胰腺、肝、胃。数值单位为百分比。

## WORD head 已记录均值（来源摘要，不作严格统一排名）

| 方法 | 八类 mean Dice | 证据与限制 |
|---|---:|---|
| D multi-query | 82.12 | 原分支 README；阈值协议和八类向量需从正式结果重新核验 |
| C single-query | 82.01 | 原分支 README；阈值协议和八类向量需从正式结果重新核验 |
| B MAE encoder+decoder | 81.09 | MAE 专项已核验记录 |
| B MAE encoder-only | 80.60 | MAE 专项已核验记录 |
| B scratch | 79.89 | 下方完整八类表 |

这是已找到来源的摘要，不补写 Arm A/E/F 或新3D未核验数字，不代表全量最新排行。
Arm B 三行使用 fixed0.5；不将 C/D 摘要默认为相同阈值协议。
下方历史表中的“排名/最优”限定于该表候选，不覆盖后来 C/D。

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
  [ORGAN_SLOTBANK_EXPERIMENT_SUMMARY_CN.md](archive/HISTORY.md#source-6)。

Arm B 两个正式 JSON 均加载 epoch 802 checkpoint，`exact=true`；reconstruction
结果标记完整测试集，head 结果标记 `complete_split=true`，样本数均为 6990。


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


## 跨路线边界

PSEM/AutoPET 四类、WORD224 与 WORD070 不混排；见 experiment/psem 的 RESULTS。
早期3D异常与后续slice/PE实验需在正式结果中分开。当前摘要不将“诊断运行完成”
写成“3D问题已解决”。全量增量学习 Old/New/All 表仍需独立验证，不由联合head分数推导。
