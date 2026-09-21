# OrganSlot

新增受控实验：**2D Arm E scratch + Collector attention alignment**。不增加网络模块，只在阳性保留器官上加入 `lambda_collector_attention=0.01` 的器官外注意力惩罚；默认0保持历史行为。每卡16、4卡累积3次、有效batch192，118800次更新。设计和可比性限制见 [CONFIGURATION](docs/CONFIGURATION.md#collector-attention-alignment)，任务状态见 [STATUS](docs/STATUS.md)。

最新实验汇总（2026-09-18）：F 2D MAE encoder-only固定84.21%、校准84.93%；encoder+重建decoder固定83.88%、校准84.59%。F 3D top-k统一协议固定83.29%、校准84.46%。既有E MAE encoder-only校准84.99%仍略高，单seed不能判定稳定优势。PCDD Offline参考85.47%尚非同协议复现。八类与协议边界见[RESULTS](docs/RESULTS.md)。

已提交2D/3D各三组受控读出对照：E式基线、F Query-Dot、F Reverse-Dot；保留旧F分类器路径，不覆盖旧实验。配置与机制见[读出消融](docs/QUERY_READOUT_ABLATION.md)，任务编号和核验时间见[STATUS](docs/STATUS.md)，3D统一校准/recon协议见[评估说明](docs/UNIFIED_3D_EVALUATION.md)。状态文档是带时间戳的快照，不是实时队列。

请从 [docs/README.md](docs/README.md) 开始。该入口按任务指向架构、配置、结果或交接，避免重复加载历史记录。

| 分支 | 职责 |
|---|---|
| `main` | 原始 OWT / 全类别基线；不是最新 OrganSlot |
| `feature/orgslot` | OrganSlot 统一架构、配置和正式结果入口 |
| `experiment/orgslot-arm-f` | Arm F 2D/3D 双向交互与匹配读出对照 |
| `fix/arme-mae-ddp` | Arm E MAE 续训的 DDP 修复记录；不是另一套方法 |
| `experiment/psem` | PSEM v1/v2/v3 查询与组合重建实验 |
| `experiment/lossbalance` | 重建 ROI loss v2/v3；不要混同 small-organ segmentation loss |
