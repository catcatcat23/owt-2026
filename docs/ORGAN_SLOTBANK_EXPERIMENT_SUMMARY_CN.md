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
