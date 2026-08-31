# OrganSlot WORD Common8：AutoPET MAE → Arm B

更新时间：2026-08-31

当前分支 `experiment/orgslot-autopet-mae-transfer` 在既有最优 Arm B 训练协议上，
只新增 AutoPET image-only MAE 初始化。主实验加载共享 encoder 与共享 reconstruction
decoder；所有 organ slot（Collector、TGEnc、AHER）和 segmentation head 均保持随机初始化。
完整设计、严格权重映射及执行链见
[docs/ORGSLOT_AUTOPET_MAE_TRANSFER_CN.md](docs/ORGSLOT_AUTOPET_MAE_TRANSFER_CN.md)。

主对照是相同 0.7 mm spacing、ROI20、multiscale head、small-organ loss 和训练更新数下：

| 实验 | 初始化 | 其余训练配置 |
|---|---|---|
| scratch Arm B | 随机 | 固定 |
| AutoPET-MAE Arm B | AutoPET MAE encoder+shared decoder | 固定 |

下面保留分支起点的 Arm A/B 记录，作为 scratch 基线说明。

基线分支 `feature/orgslot` 保存 0.7 mm spacing、ROI20、small-organ loss 下的
AHER-canvas 分割头对照：

- Arm A：`linear` head；
- Arm B：`multiscale_conv` head。

## 公平对照配置

| 配置 | Arm A | Arm B |
|---|---:|---:|
| spacing | 0.7 x 0.7 x 2.0 | 相同 |
| input / ROI crop | 448 / 384 | 相同 |
| ROI probability | 0.2 | 相同 |
| token factor | 20 | 相同 |
| segmentation supervision | retained | 相同 |
| segmentation loss | small-organ | 相同 |
| segmentation weight | 0.01 | 相同 |
| effective batch / optimizer updates | 192 / 118800 | 相同 |
| head | linear | multiscale_conv, 128 channels |

small-organ loss 在阳性切片使用 Tversky（FP/FN=0.3/0.7）和 0.5 倍
Balanced Focal，在阴性切片使用 top-2% hard-negative BCE；positive alpha=0.75，
gamma=2，negative-slice weight=0.1。

## 状态与结果

| Arm | 账号 / 集群 | Job | 状态 | checkpoint | 正式八类均值 |
|---|---|---:|---|---|---:|
| A linear | bolinren19 / SIP | 2413564 | PENDING (Resources) | 尚无 | 尚无 |
| B multiscale | antengcai23 / XEC | 121398 | COMPLETED | 802，精确加载 | reconstruction **80.45%**；head 固定 0.5 为 **79.89%** |

Arm B 完整测试为 24 个病例、6990 张切片。其 reconstruction Direct-post 为当前
项目内统一输出 SOTA：胆囊 53.74%、食管 69.48%、胰腺 70.35%、八类均值
80.45%。这说明 joint segmentation supervision 改善了共享表示和 AHER canvas；
但当前 multiscale head 的固定阈值输出仍比同一 checkpoint 的 reconstruction 低
0.56 个百分点，不能表述成“新 head 已经胜出”。

完整八类结果、增量、来源和排名见
[docs/ORGSLOT_WORD_COMMON8_SOTA_CN.md](docs/ORGSLOT_WORD_COMMON8_SOTA_CN.md)。

## 结果路径

不同账号的数据和结果目录不可互换：

- Arm A：`bolinren19 / SIP`，本 worktree 下的 `Results/OrganSlotBank/...`；
- Arm B：`antengcai23 / XEC`，`/gpfs/work/aac/antengcai23/worktrees/orgslot_smallorgan_loss/Results/OrganSlotBank/...`。

训练入口在 `slurm/orgslot/train/`，评估入口在 `slurm/orgslot/eval/`。运行日志不纳入 Git。
