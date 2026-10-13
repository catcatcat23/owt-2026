# OWT → OrganSlot：架构路线与消融逻辑

核心研究目标：器官专属权重的可插拔增量学习，以及器官tokens的解耦、重接与下游可用性。
参数隔离、语义解耦、独立模块可组合是不同主张，不能由一次联合分割Dice同时证明。

## 1. 从重建到模块化

| 阶段 | 问题 | 实际改动 | 实验证据在哪里 |
|---|---|---|---|
| 原OWT | 类别条件重建能否分割 | ViT→Collector→共享TGEnc→AHER→重建Transformer | RESULTS的OWT224表 |
| LossBalance | 小器官是否被全图损失淹没 | v1器官ROI等权；v2频率加权；v3只优化present/kept正ROI | WORD收益与AutoPET退化均保留 |
| PSEM | 查询组合与推理状态是否匹配 | v1逐样本present组合；v2Query20；v3Direct/Context/Plus triplet+Delta | WORD v3主要修复Indirect，AutoPET未一致受益 |
| 分辨率/ROI | 预处理是否限制细节与有效监督 | 原生0.7 spacing、448输入、ROI384概率.2 | 与224协议分开，不把多项改动归因单模块 |
| OrganSlot | 能否独立追加器官参数 | 每器官Collector/TGEnc/AHER，append/freeze、背景slot、标签隔离 | 工程基础≠正式增量结果 |
| A–F/SAM | tokens如何读取空间信息 | canvas→query-dot→多尺度交互→动态mask读出 | 同协议消融与负结果见RESULTS |
| 增量适配 | 冻结共享读出时新slot能否合作 | shared/stage-split/slot-private/adapter | 冻结矩阵见INCREMENTAL_READOUT_ABLATION |

原OWT的AHER、位置编码和重建decoder是继承模块，不是OrganSlot新增贡献。
原始3D Collector已有T→1；不能将其当新增bug。

## 2. 当前共同路径

```text
CT → shared encoder → z
                      ├→ Collector_s → TGEnc_s → token_norm → 20 organ tokens
                      │                                      ├→ AHER_s → canvas_s
                      │                                      │           ↓
                      │                                      │    fusion → shared recon decoder
                      │                                      └→ segmentation query/readout
CT → shallow CNN skips + z → shared pixel decoder → P16/P8/P4 ─────────────┘
```

八前景加背景共9个slot，每个20 tokens。ROI20表示采样概率20%，不是token数量。
448输入：P16=28、P8=56、P4=112；P4边长1/4、面积1/16。
浅层CNN提供空间细节；不是把ViT低分辨率特征简单放大就产生新信息。
重建canvas路径与direct segmentation空间路径不同；重建decoder不是分割head必经模块。

## 3. A–F：改了什么，不能混称什么

| 配置 | tokens与空间交互 | 最终读出 |
|---|---|---|
| A linear | AHER canvas | 线性mask |
| B multiscale_conv | AHER canvas | 多尺度卷积mask |
| C query_dot | K tokens汇总为1 query；ViT空间分支 | query·pixel |
| D multi_query_dot | K独立queries | 多mask，经log-mean-exp汇聚 |
| E no-cross | mean后的1 query | query·原P4 |
| E cross | 1 query读取P4、FFN | refined query·原P4 |
| F Query-Dot | 20 tokens依次读取P16/P8/P4 | mean refined tokens·原P4 |
| F Reverse-Dot | 同上；P4反向读取tokens | mean refined tokens·updated P4 |
| 原F attention | 同上反向读取 | LayerNorm+共享Linear分类器 |
| SAM-tail | 多尺度refinement后加mask token、双向交互、最终回读 | mask-token MLP权重·updated pixels |

F不是E的cross开关；F使用query_refinement=none。
历史arm_f_linear也不是新arm_f_query_dot的同义词，应以checkpoint head类型为准。
20 tokens是图像条件化的器官表示，不是20个独立器官实例；没有Hungarian matching。
Attention权重不等于前景概率，热图集中不证明表示解耦。

## 4. 最终保留三个配置

| 主线 | 多尺度token读取 | 引导/监督 | tail |
|---|---|---|---|
| E cross + MAE encoder | 无；mean query只读P4 | 无粗mask辅助 | 无 |
| SAM soft_prior_aux + MAE encoder | P16→P8→P4 | 每次读取前同尺度软偏置，三尺度aux | mask-token双向tail，无额外coarse prior |
| SAM downsample_gt + MAE encoder | P16→P8→P4 | refinement后P4粗mask，nearest GT监督 | 粗mask门控tail的两次V读取 |

架构图和启动入口在 [README](../README.md#三张架构图与差异)。
soft_prior_aux与downsample_gt不是同一机制；后来新增组合枚举
downsample_gt_soft_prior / downsample_gt_soft_prior_aux，属于独立消融，不覆盖三条主线。
精确detach、门控位置、loss权重和调度见 [CONFIGURATION](CONFIGURATION.md)。

## 5. 支线实验的定位

- **MAE**：迁移encoder与可选共享重建decoder；不直接迁移Collector、TGEnc、AHER或pixel decoder。
- **P2**：P4 reverse后上采样并融合S2，两层卷积得到224×224的P2再点积；
  测试的是高分辨率+浅层融合这一整体，不只是放大特征图。
- **P2 softmask**：token→pixel加detach软偏置、floor.2，无独立aux；与SAM+aux不能混称同一实验。
- **Collector soft loss**：用GT patch支持区惩罚器官外注意力，lambda .01；不增加参数、不强制推理GT。
- **all-slot监督**：分割监督所有可见前景，重建仍随机keep；共享梯度仍耦合。
- **3D**：F/SAM逐slice空间CNN+跨slab attention及t/y/x PE；不是新增3×3×3卷积decoder。
  slice-wise、PE、temporal conv、overlap fusion必须拆开；不能用后续改善倒推唯一根因。
- **CNN**：nnU-Net ResEnc风格或MedNeXt-B 2D输出S16，经512→768投影接原接口。
  保留现有浅层CNN分支，不额外使用backbone的S2/S4/S8跳接；不是完整官方分割系统。

所有支线结果（包括无收益和待核验）见 [RESULTS](RESULTS.md)；
完整旧设计、参数、提交/取消链见 [历史读出消融](archive/HISTORY.md#merge-query-readout-ablation)。

## 6. 增量交互与adapter

Stage1共享器官交互；Stage2私有化投影和P16/P8/P4交互，旧路径冻结，新路径训练。
ViT/pixel decoder/共享SAM-tail/共享重建decoder冻结；冻结tail仍对输入可导。
背景单独按separation策略训练，不把未来器官永远锁在旧背景中。
新增residual16 adapter位于tail最终mask token到动态权重的读出处，4240参数，
**四个新器官共享一套**，旧器官与背景绕过，不是每个slot重训一个完整模型。
完整参数策略、标签访问、预算和结果见 [增量专项](INCREMENTAL_READOUT_ABLATION.md)。

## 7. 实现入口

- `OWT_models_orgslot.py`、`OrganSlotEmbed.py`：主模型、slot和E空间读出。
- `ArmFDecoder.py`：F/SAM多尺度交互、prior和mask读出。
- `CNNBackbone.py`、`models_cnn_mim.py`：CNN编码器及masked reconstruction。
- `main_pretrain_orgslot_common8_a100.py`、`engine_pretrain_orgslot_common8_a100.py`：离线训练/loss。
- `main_pretrain_orgslot_incremental44.py`：增量独立入口。
- 删除的A–D启动器在标签 `archive/20260927/abcd-before-retirement`，不代表科学结果无效。
