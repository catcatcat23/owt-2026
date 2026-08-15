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
