# 文档地图与交接入口

整理日期：2026-10-13。docs顶层只保留6份MD，另有archive/HISTORY.md；不删除实验产物。

| 问题 | 入口 |
|---|---|
| 最终留下什么、指标/架构图/怎么启动 | [根README](../README.md) |
| OWT→LossBalance/PSEM→OrganSlot→E/F/SAM如何演进 | [ARCHITECTURE](ARCHITECTURE.md) |
| 哪些消融有收益、有代价或证据不足；逐类及原始证据 | [RESULTS](RESULTS.md) |
| loss/先验公式、数据协议、MAE/CNN、诊断/评估如何复现 | [CONFIGURATION](CONFIGURATION.md) |
| 增量Stage1/2、冻结矩阵、背景、adapter、Old/New/All | [INCREMENTAL_READOUT_ABLATION](INCREMENTAL_READOUT_ABLATION.md) |
| 最近任务状态和下一步 | [STATUS](STATUS.md)，注意时间戳 |
| 原文、历史参数、提交SHA/Job/结果路径 | [archive/HISTORY](archive/HISTORY.md) |

## 远端分支与固定来源

仓库：[catcatcat23/owt-2026](https://github.com/catcatcat23/owt-2026)。
当前主线看 [feature/orgslot](https://github.com/catcatcat23/owt-2026/tree/feature/orgslot)，
不是main的原OWT。本次文档整理通过feature/orgslot发布，不更新其它分支或训练快照。

| 来源 | 整理时origin版本 | 用途 |
|---|---|---|
| main | db349d9b66de953ed21dc088b0c2ca723644fb43 | OWT联合重建、WORD/BTCV/AutoPET；docs/RESULTS.md |
| experiment/lossbalance | 144b427c79737257406e86ea3e94d2a33e8fae7e | ROI重建v2/v3；docs/EXPERIMENTS.md与archive |
| experiment/psem | e8ecb1a1fc25f45c5ff93682cf69db4cb00d2fe3 | v1/v2/v3及组合正式结果；docs/RESULTS.md与archive |
| feature/orgslot | f198a191fe9460410c0e8ed32c3392564b165ae1 | 本次整理基底，当前开发worktree是detached，不移动旧本地分支 |

原OWT的早期COMPLETE_EXPERIMENT_SUMMARY、FORMAL_EXPERIMENT_REGISTRY和PSEM_V1日志
属于历史旧树文件名，不保证当前main顶层仍存在。当前远端入口用上表，
固定来源已归档到 [OWT](archive/HISTORY.md#owt-results)、
[PSEM](archive/HISTORY.md#psem-results)、[LossBalance](archive/HISTORY.md#lossbalance-experiments)。
本地根目录已修改的FORMAL_EXPERIMENT_REGISTRY及未跟踪画图提示不覆盖。

## 原文档 → 合并目标 → 保留内容

| 原文档 | 新目标 | 保留内容/处理 |
|---|---|---|
| 各分支结果表 | RESULTS + archive | 所有版本正负结果、逐类表、协议/路径；固定版本原文无损保留 |
| ARCHITECTURE | ARCHITECTURE | 重写演进与三个当前配置，旧原文归档 |
| RESULTS、STATUS旧流水账 | RESULTS、STATUS + archive | 正式结果分类；旧状态完整归档，不作实时状态 |
| E_CROSSATTN_MAE | ARCHITECTURE + CONFIGURATION | query机制、初始化、脚本和公平性；删除重复MD |
| SAM_SOFT_PRIOR、SAM_TAIL_HSAM | ARCHITECTURE + CONFIGURATION | 先验公式、detach、aux、H-SAM调度、M0–M3及SHA；删除重复MD |
| SAM_TAIL_ALL_SEG | CONFIGURATION + RESULTS | keep/head_mask语义、监督与结果；删除重复MD |
| QUERY_READOUT_ABLATION | ARCHITECTURE + RESULTS + archive | F dot/P2/权重消融、恢复和提交记录；删除重复MD |
| FROZEN_P4_PROBE、TOKEN_INTERVENTION | CONFIGURATION + RESULTS | 诊断设计、限制、脚本、原始结果索引；删除专项MD |
| UNIFIED_3D_EVALUATION | CONFIGURATION | 校准/head/recon/fusion协议；删除专项MD |
| CNN_BACKBONES | CONFIGURATION + ARCHITECTURE | 精确stage/depth/通道、MIM/迁移、任务链；删除专项MD |
| ABCD_RETIREMENT、CLEANUP_20260927 | archive + 本页 | 恢复标签与维护历史；删除专项MD，不删除结果 |
| INCREMENTAL_READOUT_ABLATION | 原位保留 | 独立复杂冻结/标签/适配契约和正式八类结果 |
| docs/figures及两份几何分析JSON | 原位保留 | README三张架构图、生成来源及几何证据，不是重复MD |

所有移除MD原文先归档，再删除独立副本；可从HISTORY或基底commit恢复。
不删除任何checkpoint、数据、任务或历史runtime，不清理远端分支。

## 结论修正与未核验项

- PSEM不是全无效：WORD v3的主要收益是Indirect；AutoPET未跨数据集一致改善。
- 重建ROI loss不是OrganSlot small-organ segmentation loss。
- E cross固定阈值与校准后方向不同；SAM aux收益不能全归于soft routing。
- 历史3D过分割在更强F分支后显著改善，loss不是已证明的唯一根因。
- 早期59400/阶段与全预算118800/阶段分开；不把NA当0，不把旧binary保留等同互斥mask不变。
- RESULTS仍标待核验的dot/P2/PE/Collector/探针等需定向读取最终产物，不能根据旧pending判失败。
- SOURCE旧文档可能缺账号/SHA或路径只写省略号；保留缺口，不编造绝对路径。
- 同协议随机划分seed0/1/42不是k折CV；替换badcase及GT匹配诊断不纳入正式均值。

## 整理核验

- 11份移除MD逐一与基底commit原文比对，均完整包含在HISTORY中，可从档案或Git恢复。
- 当前README/docs中本地Markdown链接、锚点及图片路径检查为0错误；外部站点未重新联网核验。
- 当前文档中显式tools/scripts/slurm入口均存在；归档代码块路径保留原时点语义，不保证仍存在。
- 三主线、三次划分均值、最新SAM增量关键数值在README/RESULTS/增量专项一致。
- git diff --check通过；改动仅Markdown。未改模型、运行快照、数据、checkpoint或Slurm。
- 发布仅提交文档及重复MD删除；远端是否同步以对应commit和分支HEAD核验为准。

## 微信交接短文（可复制）

这个仓库从OWT类别重建出发，先做了LossBalance器官均衡和PSEM查询组合消融，
再发展成器官独立Collector/TGEnc/AHER的OrganSlot。随后对比canvas、query-dot、
多尺度/反向交互、MAE、SAM-tail及mask监督；当前保留E cross、SAM soft prior+aux、
SAM两阶段缩小GT三条主线。增量实验继续比较共享/私有交互和adapter。
各数据集有效与无效实验都保留，不跨协议刷排名。远端看feature/orgslot的README，
完整结果看docs/RESULTS.md，演进看ARCHITECTURE.md，复现看CONFIGURATION.md，
增量看INCREMENTAL_READOUT_ABLATION.md；main/psem/lossbalance是早期路线。
以上路径均相对于feature/orgslot分支根目录。
