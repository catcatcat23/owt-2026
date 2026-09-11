# 原始 OWT 基线入口

main保留原始联合全类别基线，不是最新OrganSlot开发入口。
[RESULTS](RESULTS.md)保留原始数据划分、结果与限制；不要混用WORD224与WORD070协议。

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
