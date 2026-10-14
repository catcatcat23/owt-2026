# 实验状态快照

## 2026-10-14：SAM组合aux失败后恢复链

- 原157059于13:07 FAILED：epoch480附近ALLREDUCE SeqNum7003207超时1800秒。
- checkpoint470：CPU模型/优化器有限；69708更新；最新代码strict exact加载通过。
- 新训练163200，sifansong/XEC，4A800/20CPU/192GB、4gpus/7天，PENDING(Priority)。
- 新评估163201，1RTX4090/10CPU/128GB、4gpus/1天，afterok:163200，PENDING(Dependency)。
- 从epoch471恢复optimizer/scaler，总预算118800；仅启用已有常量LPIPS buffer广播关闭保护。
- 固定runtime `/gpfs/work/aac/sifansong/worktrees/sam_combo_resume_1f2b0e1`，
  SHA `1f2b0e169b7d454d355ba331a8a308db65ab92ed`，提交前revision guard通过。
- 旧失效评估157061已取消；原模型/日志保留。不运行额外GPU smoke；正式GPU尚未启动。
- 其它更新：157058约epoch769、157119约434；157599今天13:07启动、约epoch6。

以下保留10月13日检查快照，157059状态以本节为准；估计启动时间须重新查Slurm。

时间口径：2026-10-13 Asia/Shanghai。
队列/runtime本次整理约22:19前后只读核查；epoch来自21:04日志快照，**不是实时epoch**。
本次只整理文档，未提交、取消、迁移或修改任何任务。

| 实验 | 账号/集群 | 训练 → 评估 | 队列状态 | 最近已核验训练快照 |
|---|---|---|---|---|
| SAM两阶段+soft prior，无额外prior aux | antengcai23/XEC | 157058 → 157060 | RUNNING / Dependency | epoch618/803（21:03） |
| SAM两阶段+soft prior+aux | sifansong/XEC | 157059 → 157061 | RUNNING / Dependency | epoch399；21:03仍推进但此前有23–42分钟/20step停顿 |
| SAM增量私有交互+adapter | sifansong/XEC | 157119 → 157120 | RUNNING / Dependency | epoch194（21:03） |
| E Stage2共享query恢复 | bolinren19/SIP | 3044537 → 3044538 | PENDING / Dependency | Resources；无新结果 |
| E Stage2分组query恢复 | bolinren19/SIP | 3044539 → 3044540 | PENDING / Dependency | 无新结果 |
| 3D SAM恢复 | bolinren19/SIP | 3044541 → 3044542 | PENDING / Dependency | 不能据任务名声明已恢复 |
| 3D SAM缩小GT | antengcai23/XEC | 157599 → 157600 | PENDING / Dependency | 无最终结果 |
| ResEnc AutoPET MIM | bolinren19/SIP | 3048464 | PENDING | 尚无GPU预训练结果 |
| ResEnc迁移soft/缩小GT | bolinren19/SIP | 3048465/66 → 3048467/68 | Dependency | 等预训练成功 |
| MedNeXt AutoPET MIM | sifansong/XEC | 162141 | PENDING | 尚无GPU预训练结果 |
| MedNeXt迁移soft/缩小GT | sifansong/XEC | 162142/44 → 162143/45 | Dependency | 等预训练成功 |
| 旧Stage1五slot评估 | bolinren19/SIP | 3001901 | JobHeldUser | 不与已完成替代评估混同 |

SAM私有交互156538/156539的81.84结果已归档到 [增量专项](INCREMENTAL_READOUT_ABLATION.md)，
不是每次查询COMPLETED都出现一个新实验。其它旧结果见 [台账](RESULTS.md)。
同账号另有EEG任务，不属于OWT/OrganSlot清单，未修改。

## 固定runtime与安全边界

- SIP恢复任务：`.worktrees/quota_resume_381171c`；CNN：`.worktrees/cnn_mim_7ae0689`。
- sifansong/XEC：`/gpfs/work/aac/sifansong/worktrees/sam_combo_7837635`、
  `inc44_adapter_348fd46`、`cnn_mim_ab5c0aa`。
- antengcai23/XEC：`/gpfs/work/aac/antengcai23/worktrees/sam_combo_7837635`、`sam3d_2558bac`。
- 本次仅修改开发目录 `.worktrees/orgslot_dev_current`，上述runtime保持不动。
- 最新commit不能自动修复旧快照；后续重提必须按AGENTS核验SHA与afterok依赖。

## 下次优先检查

1. 157059间歇性变慢：先查日志、GPU/进程/I/O；未确认NaN、NCCL或GPFS根因，不擅自重启。
2. 两组SAM组合最终head/recon，与原downsample_gt基线对齐；不要用loss预测Dice。
3. adapter新类收益与旧类binary/互斥mask分别检查；checkpoint需要加载核验。
4. CNN预训练启动后检查数据身份、100k更新预算、权重迁移报告；CPU测试不冒充GPU验证。
5. 已知Indirect异常仍需独立诊断；不概括为所有recon正常。
6. 需要新状态时重新查Slurm，不沿用历史预计启动日期。

历史完整状态、CPU/GPU边界、源SHA及提交记录已保存在
[旧STATUS](archive/HISTORY.md#merge-status)；禁止把该归档当今日队列。
