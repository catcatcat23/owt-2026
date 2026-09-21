# 当前实验交接

## 2026-09-21 16:53 Collector attention alignment 已提交

目的：以2D Arm E scratch为基准，仅新增Collector器官外注意力软惩罚，验证器官采集定位是否改善；不增加网络模块，不使用MAE、不续训，默认lambda=0保持旧实验。

| 实验 | 账号/集群 | Job | 资源与时限 | 提交后核验 |
|---|---|---|---|---|
| Arm E scratch + collector loss 0.01 | sifansong / XEC，account=sifansong，QoS=4gpus | 143780 | 4×A800、20CPU、192GB、7天 | PENDING (Priority)；无GPU验证 |
| 同协议阈值校准 + head测试 + reconstruction | 同上 | 143781 | 1×A800、10CPU、128GB、7天 | PENDING (Dependency)，afterok:143780 |

- 固定代码：`aff9f696d50fb2dcd0edfb47bc6312e8f9caae3e`，已在提交前推送origin/feature/orgslot并通过revision guard。
- 本地固定worktree：`/gpfs/work/aac/bolinren19/OD_OWT/.worktrees/collector_align_aff9f69`。
- XEC固定源码快照：`/gpfs/work/aac/sifansong/worktrees/collector_align_aff9f69`。此目录不pull、不覆盖；tar无.git时训练provenance允许unavailable，真实来源以本条full commit、源码checksum、Slurm SOURCE_COMMIT和压缩包校验为准。
- 压缩包：`/gpfs/work/aac/sifansong/collector_align_aff9f69.tar`，SHA256=`d197f72fb8f53367e4824580c46978d3e1ba3ac3685e8fbff93db45f3237475d`，本地/远端一致。
- train/eval入口：`slurm/orgslot/train/arm_e_collector_align.sbatch`、`slurm/orgslot/eval/arm_e_collector_align.sbatch`。
- 训练输出：运行目录下`Results/OrganSlotBank/Common8/WORD_2D/ArmE_CollectorAlign001_scratch_143780`；Slurm日志：`/gpfs/work/aac/sifansong/logs/collector_align_aff9f69/`。
- 2D每卡16×4卡×累积3=192；118800 updates，预计最终checkpoint-802；WORD07072、ROI20、seed0、small-organ lambda_seg0.01不变。
- 对照：E scratch训练系列2965273 / 评估2965274。对照历史中途换过microbatch，新实验从头16；不是逐更新完全相同随机轨迹。其余已有任务未修改。
- 本地与XEC各6项新增CPU测试通过，包括两进程Gloo；数据身份28586训练切片/6990测试切片、ROI core SHA256均匹配基线。GPU正式运行尚未开始，不能宣称GPU验证通过。
- 下一次先查143780是否启动、resolved_config是否scratch/batch16/lambda_collect0.01、首epoch各器官collector count/foreground_mass及有限梯度、weighted_collect=0.01×raw；再检查checkpoint与143781依赖。最终比较八类head fixed/calibrated和recon，不只看attention集中度。


## 2026-09-18 13:53 已核验快照（当前入口）

新六组：1组运行、5组排队；未发现失败。此次文档整理未再次刷新队列。
每组训练4×A800/20CPU/192GB；scratch、seed0、top-k small-organ loss、ROI20、
118800 updates。2D每卡8/累积6/有效192；3D每卡6/累积2/有效48 slabs。

| 目的 | 账号/集群 | 训练 | 校准/Head/recon评估 | QoS/时限 | 状态 |
|---|---|---|---|---|---|
| 2D E单query点积基线 | bolinren19/SIP | 2955007 | 2955008（统一三阶段） | 4a800/7天 | Priority |
| 2D F Query-Dot：细化tokens、不反向读取 | bolinren19/SIP | 2955009 | 2955010（统一三阶段） | 8a800/7天 | Priority |
| 2D F Reverse-Dot：反向读取后点积 | bolinren19/SIP | 2955011 | 2955012（统一三阶段） | 8a800/7天 | Priority |
| 3D E式点积基线 | sifansong/XEC | 140585 | 140586 / 140587 / 140588 | 4gpus/7天 | RUNNING epoch4，checkpoint-0存在 |
| 3D F Query-Dot | sifansong/XEC | 140591 | 140592 / 140593 / 140594 | 8gpus/5天 | Priority |
| 3D F Reverse-Dot | antengcai23/XEC | 140595 | 140596 / 140597 / 140598 | 4gpus/7天 | Resources |

3D基线最新抽查：loss≈0.2511，seg≈0.7667，weighted_seg≈0.0077，ROI≈20.3%，
日志torch峰值allocated33.57GiB，overflow0、错误日志空。非完整稳定性证明；
checkpoint-0仅确认文件存在，未重新加载。其余五组尚无GPU运行验证。

固定运行源（不要在排队/运行目录pull）：

- 2D：bolinren19/SIP `/gpfs/work/aac/bolinren19/OD_OWT/.worktrees/arm_f_linear_2d`，
  `5f2d1cfb7fb20d499c79cb8e901d7279718cd8bc`。目录旧名linear不代表当前提交了arm_f_linear。
- 3D：各账号XEC `/gpfs/work/aac/<账号>/worktrees/dot3d_ac8cf99`，
  `ac8cf992cdac30c9d7bed33cc5e2f8b8ad4d8ecd`。
- 训练入口 `slurm/orgslot/train/arm_f.sbatch`；2D评估 `arm_e_mae_unified.sbatch`；
  3D评估 `unified_3d_protocol.sbatch`。3D校准/recon依赖训练afterok，Head依赖校准afterok。
- 同seed不保证共有模块初始权重逐张量相同，本轮未配平；跨架构因果解释需保留此限制。

已完成：F scratch 2929821/2929822；F encoder 137867/137868；F encdec137880/137881；
3D topk137410及140443–140445统一评估；3D mean137412/137413（旧post3d协议）；
3D MAE2951384完成100000updates，日志报告checkpoint验证通过，尚无其下游迁移任务。
最新指标统一见RESULTS.md，不从训练校准分数推断测试成绩。

下次：优先检查140585的checkpoint10；五组排队任务启动后核验head、batch、topk、
梯度与首epoch；140591五天时限余量偏紧；3D mean若需校准比较须补统一协议。
F2D scratch的校准/recon仍为缺口。Indirect异常需独立诊断，不能概括为recon全正常。

## 历史快照：2026-09-14（以下状态已过期，仅保留背景）

日志/结果核验时间：2026-09-14 14:05 Asia/Shanghai。之后队列复查仍为4组训练运行、2组训练排队。
epoch为快照，不是实时页面；估时不含评估和排队。结果统一见[RESULTS.md](RESULTS.md)。

| 实验目的 | 账号/集群 | 训练 → 评估 | 资源/QoS | epoch / checkpoint | 状态 |
|---|---|---|---|---|---|
| E cross-attention：显式query与空间交互 | bolinren19/SIP | 2923562 → 2923565 | 4A800/24CPU，8a800 | 441/802，425 | RUNNING |
| F 2D scratch：双向交互 | bolinren19/SIP | 2929821 → 2929822 | 4A800/20CPU，4a800 | 222/802，220 | RUNNING |
| F 3D topk：从30续训 | sifansong/XEC | 137410 → 137411 | 4A800/20CPU/192GB，4gpus，7天 | 61/199，60 | RUNNING |
| F 3D mean：背景全平均对照 | antengcai23/XEC | 137412 → 137413 | 4A800/20CPU/192GB，8gpus，5天 | 14/199，10 | RUNNING |
| F 2D MAE encoder-only | sifansong/XEC | 137867 → 137868 | 4A800/24CPU/192GB，8gpus，5天 | — | PENDING Priority |
| F 2D MAE encoder+重建decoder | antengcai23/XEC | 137880 → 137881 | 4A800/24CPU/192GB，4gpus，7天 | — | PENDING Priority |

评估均等待对应训练；未新增任务或修改依赖。sifansong/SIP、antengcai23/SIP无活动任务。
E scratch 2922770/2922772、E MAE encoder 135283/135284、encdec 134915/134916均已完成。
E两条续训跨过旧541故障点并完成802；不能据此声称所有NCCL底层问题已消除。

## 配置、运行源与公平性

未来四卡2D每卡16×累积3=有效192；3D每卡6×累积2=有效48 slabs。
运行中F2D仍为8×累积6，新F MAE为16×累积3；microbatch不同必须披露。
small-organ loss分组归约意味着有效batch相同不保证梯度完全相同。

- F3D两账号runtime：各自 /gpfs/work/aac/<用户名>/worktrees/arm_f_batch6_20260913；
  固定代码2697b05642d527bfd137debc0f9ed175e9ae345f。
- F MAE encoder：/gpfs/work/aac/sifansong/worktrees/arm_f_mae_encoder_df7ed39；
  df7ed39cb8529afebb8f1dd982eb8daf382c8c6d。
- F MAE encdec：/gpfs/work/aac/antengcai23/worktrees/arm_f_mae_encdec_20260913；
  5cee1e95de7e9c758df0e3cb3fc569cdc0fa8d57。旧137869/137870已取消。
- F MAE入口 slurm/orgslot/train/arm_f_mae.sbatch；评估 arm_f_mae_unified.sbatch。
  7项CPU加载测试和数据/hash门禁通过，正式GPU加载及batch16显存仍待启动。
- E encoder runtime：/gpfs/work/aac/sifansong/worktrees/arme_mae_encoder_runtime_f70b831。
- E encdec runtime：/gpfs/work/aac/sifansong/worktrees/arme_mae_runtime_43cad30。
- 仅开发目录同步Git；运行与排队快照保持固定。各账号数据路径不得互换。

## 稳定性与估时

最新epoch平均总loss/seg：E CA 0.0809/0.1266；F2D 0.1029/0.1611；
F3D topk 0.1135/0.1920；mean 0.1744/0.3603，均有限，但只是日志抽查。
日志打印进程的torch峰值allocated显存分别16.84、23.98、57.88、57.90 GiB；
不是实时nvidia-smi，不是全部rank峰值，也不含全部缓存/通信开销。
实时显存查询未成功（SIP节点SSH拒绝；XEC工具审批服务失败）。

按14:05吞吐外推：E CA剩32小时；F2D约81小时；F3D topk约73–76小时；
mean约93小时，需预留10–20%波动。mean的120小时时限余量较紧。
F MAE尚排队，batch16没有实测，不承诺结束日期。

## 下次审核

1. E CA checkpoint450/500、F2D checkpoint230后稳定性。
2. F3D topk checkpoint70、mean checkpoint20；最终看Dice/P/R/volume ratio/per-z，不用低loss证明3D已修复。
3. F MAE启动后核验初始化scope、每卡16、有效192、GPU峰值与真实加载报告。
4. E encoder-only recon Direct-post81.17已收齐；Indirect-post7.53异常，需独立诊断。
5. 3D D slice-wise与spatial既有均值21.13/30.25，仍过分割；temporal/ST旧链已取消，不再列作待跑。
6. PCDD Offline85.47固定列入对比；未匹配协议，不能宣称增量或外部SOTA。
