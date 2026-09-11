# OrganSlot 文档入口

研究主线：可插拔器官权重的增量学习，以及解耦/重接 tokens 的下游价值。
统一分支含 A–F；F实验分支保留其独立代码版本，不能假定不同分支修复已同步。

| 要做什么 | 只需读取 |
|---|---|
| 理解研究目标和 A–F 差异 | [ARCHITECTURE](ARCHITECTURE.md) |
| 改代码、续训或提交评估 | [CONFIGURATION](CONFIGURATION.md)及根目录AGENTS |
| 汇报效果与已知结论 | [RESULTS](RESULTS.md) |
| 检查实验和安排下一步 | [STATUS](STATUS.md)，再读真实Slurm/log |
| 查历史公式、任务和原始结果路径 | [历史原文](archive/HISTORY.md)，按原文件名定位 |

| 分支 | 职责 |
|---|---|
| `main` | 原始 OWT / 全类别基线；不是最新 OrganSlot |
| `feature/orgslot` | OrganSlot 统一架构、配置和正式结果入口 |
| `experiment/orgslot-arm-f` | Arm F 2D/3D 双向交互与匹配读出对照 |
| `fix/arme-mae-ddp` | Arm E MAE 续训的 DDP 修复记录；不是另一套方法 |
| `experiment/psem` | PSEM v1/v2/v3 查询与组合重建实验 |
| `experiment/lossbalance` | 重建 ROI loss v2/v3；不要混同 small-organ segmentation loss |

本次整理来源为43cad30；fix/arme-mae-ddp与此提交拥有相同代码基础与文档。

## 维护规则

- 新会话先读本页，再按任务读取一份主文档；不要默认遍历历史档案。
- 架构、固定配置、结果、运行状态分开维护；不再新增按日期命名的交接文档。
- 结果必须标注数据集、协议、checkpoint、读出、阈值、证据路径；缺失值写“未核验”。
- Slurm 状态必须带查询日期；本次文档整理不是完整训练/评估审计。
- 历史原文集中在 `archive/HISTORY.md`，可能含过期路径、状态和计划，不作为当前指令。
- 不将 joint training 的 Dice 当作增量抗遗忘或模块可组合性的证明。
- 不改正在运行/排队的源快照；提交任务仍须遵循根目录 AGENTS.md。
