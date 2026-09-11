# PSEM

请从 [docs/README.md](docs/README.md) 开始。该入口按任务指向架构、配置、结果或交接，避免重复加载历史记录。

| 分支 | 职责 |
|---|---|
| `main` | 原始 OWT / 全类别基线；不是最新 OrganSlot |
| `feature/orgslot` | OrganSlot 统一架构、配置和正式结果入口 |
| `experiment/orgslot-arm-f` | Arm F 2D/3D 双向交互与匹配读出对照 |
| `fix/arme-mae-ddp` | Arm E MAE 续训的 DDP 修复记录；不是另一套方法 |
| `experiment/psem` | PSEM v1/v2/v3 查询与组合重建实验 |
| `experiment/lossbalance` | 重建 ROI loss v2/v3；不要混同 small-organ segmentation loss |
