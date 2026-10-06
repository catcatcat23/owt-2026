# OrganSlot

2026-10-06：新增 Stage2 **固定空间分支的共享 query 读出微调 / 新旧双读出**对照。
两组均从同一 Stage1 初始化；配置、冻结边界和验证要求见
[增量读出对照](docs/INCREMENTAL_READOUT_ABLATION.md)。旧实验默认行为不变。

新增可选 SAM-tail 增量配置：`INCREMENTAL_ARCHITECTURE=sam_soft_aux`，Stage2
`slot_private`。每slot独立投影和三尺度token交互，soft prior＋aux不变，空间分支和
共享tail冻结。需独立训练旧四类Stage1，不能使用八类Offline权重；详见同一[设计文档](docs/INCREMENTAL_READOUT_ABLATION.md#sam-tail-private-slot-interaction-shared-frozen-tail)。尚未提交该组GPU任务。

4–4评估统一报告Offline、Stage1、Stage2的Old/New/All及逐器官Dice；
Stage1未见新类记NA，背景单列。汇总工具`python -m tools.report_incremental44`，
结果、遗忘量与PCDD协议差异见[统一报告](docs/RESULTS.md#44统一报告offline--stage1--stage2)。

新增独立 **4+4 类增量训练入口**：基于 E cross-attention＋MAE encoder，Stage1五个slot，Stage2九个slot；背景冻结/组合更新/显式分离三个对照。仅实现，未提交GPU训练；协议和运行方式见 [4+4 配置](docs/CONFIGURATION.md#incremental44)。旧全八类入口不变。

2026-09-30 新协议：官方100训练病例中固定抽96训练，官方20验证＋30测试池中固定抽24测试，余26验证/阈值校准（seed42）。使用 E cross-attention＋MAE encoder 重新训练，不能复用旧 WORD checkpoint，也不能与旧划分85.36%作严格横向比较。清单见 [word_official96_seed42.json](configs/orgslot/word_official96_seed42.json)，执行细节见 [CONFIGURATION](docs/CONFIGURATION.md#word-official96-protocol)。旧数据与任务不变。

新增可选 head：`arm_f_sam_tail`（2026-09-22，仅实现，未提交GPU任务）。
保留P16→P8→P4 token refinement，追加共享mask token、一次双向交互、
最终token回读和MLP动态点积。旧E/F不变；14项CPU测试通过，GPU/DDP待验证。
配置和SAM差异见 [Query readout ablation](docs/QUERY_READOUT_ABLATION.md)。

新增受控实验：**2D Arm E scratch + Collector attention alignment**。不增加网络模块，只在阳性保留器官上加入 `lambda_collector_attention=0.01` 的器官外注意力惩罚；默认0保持历史行为。每卡16、4卡累积3次、有效batch192，118800次更新。设计和可比性限制见 [CONFIGURATION](docs/CONFIGURATION.md#collector-attention-alignment)，任务状态见 [STATUS](docs/STATUS.md)。

最新实验汇总（2026-09-18）：F 2D MAE encoder-only固定84.21%、校准84.93%；encoder+重建decoder固定83.88%、校准84.59%。F 3D top-k统一协议固定83.29%、校准84.46%。既有E MAE encoder-only校准84.99%仍略高，单seed不能判定稳定优势。PCDD Offline参考85.47%尚非同协议复现。八类与协议边界见[RESULTS](docs/RESULTS.md)。

已提交2D/3D各三组受控读出对照：E式基线、F Query-Dot、F Reverse-Dot；保留旧F分类器路径，不覆盖旧实验。配置与机制见[读出消融](docs/QUERY_READOUT_ABLATION.md)，任务编号和核验时间见[STATUS](docs/STATUS.md)，3D统一校准/recon协议见[评估说明](docs/UNIFIED_3D_EVALUATION.md)。状态文档是带时间戳的快照，不是实时队列。

请从 [docs/README.md](docs/README.md) 开始。该入口按任务指向架构、配置、结果或交接，避免重复加载历史记录。

| 分支 | 职责 |
|---|---|
| `main` | 原始 OWT / 全类别基线；不是最新 OrganSlot |
| `feature/orgslot` | OrganSlot 统一架构、配置和正式结果入口 |
| `experiment/psem` | PSEM v1/v2/v3 查询与组合重建实验 |
| `experiment/lossbalance` | 重建 ROI loss v2/v3；不要混同 small-organ segmentation loss |
