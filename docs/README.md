# PSEM：唯一阅读入口

研究问题：token子集查询训练是否支持Direct与组合差分Indirect重建。
v1为present类调度；v2拓展全token-bank查询；v3同源Direct/Context/Plus配对并加Delta。
不要把PSEM查询日程与OrganSlot独立权重/增量协议混为一谈。

结果、按数据集排名、证据路径与下一步：[RESULTS](RESULTS.md)。
实现公式、旧提交和测试全文：[历史原文](archive/HISTORY.md)，只按需读。
已记录：AutoPET2D v1 Indirect90.512优于v3 89.938；WORD v3 Direct68.353、Indirect68.244。
这些是重建派生分割，不是额外head；AutoPET label1–4映射未核实时不补器官名。
旧日志中的pending不覆盖后来README的正式完成记录。本次不重新提交任何PSEM任务。

| 分支 | 职责 |
|---|---|
| `main` | 原始 OWT / 全类别基线；不是最新 OrganSlot |
| `feature/orgslot` | OrganSlot 统一架构、配置和正式结果入口 |
| `experiment/orgslot-arm-f` | Arm F 2D/3D 双向交互与匹配读出对照 |
| `fix/arme-mae-ddp` | Arm E MAE 续训的 DDP 修复记录；不是另一套方法 |
| `experiment/psem` | PSEM v1/v2/v3 查询与组合重建实验 |
| `experiment/lossbalance` | 重建 ROI loss v2/v3；不要混同 small-organ segmentation loss |

## 维护规则

- 新会话先读本页，再按任务读取一份主文档；不要默认遍历历史档案。
- 架构、固定配置、结果、运行状态分开维护；不再新增按日期命名的交接文档。
- 结果必须标注数据集、协议、checkpoint、读出、阈值、证据路径；缺失值写“未核验”。
- Slurm 状态必须带查询日期；本次文档整理不是完整训练/评估审计。
- 历史原文集中在 `archive/HISTORY.md`，可能含过期路径、状态和计划，不作为当前指令。
- 不将 joint training 的 Dice 当作增量抗遗忘或模块可组合性的证明。
- 不改正在运行/排队的源快照；提交任务仍须遵循根目录 AGENTS.md。
