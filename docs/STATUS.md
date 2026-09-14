# 当前实验交接

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
