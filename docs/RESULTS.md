# 实验结果与关键消融台账

整理日期：2026-10-13（Asia/Shanghai）。本页是结果入口，不是实时队列。
以固定版本文档及指定结果文件为依据；本次未重跑评估。
历史数值来源为 [原OWT账本](archive/HISTORY.md#owt-results)、
[PSEM账本](archive/HISTORY.md#psem-results)、[旧OrganSlot账本](archive/HISTORY.md#merge-results)。
原始八类表、P/R、NSD/HD95、路径、Job和恢复细节未删，全文归档可查。
缺失的账号、SHA、初始化报告、逐类指标均记“待核验”，不得从相邻实验推断。

## 阅读口径与证据等级

- **已归档结果**：已有完整结果记录；不表示本次重新加载checkpoint或重算。
- **本次读到原始汇总**：仅用于明确指向的JSON/CSV，不能外推其他实验也已复核。
- **待核验**：只有设计、旧提交或聊天提及，不填写猜测指标，也不判方法无效。
- fixed / calibrated、raw / post、head / reconstruction分别记录；不可逐器官选最好输出拼接。
- 旧WORD224、WORD070旧24例、官方池96/26/24、替换5例探索性评估互不混排。
- PCDD是外部论文参考，划分/预处理/模型/阶段病例访问未对齐，不宣称同协议复现。
- 单次训练不证明稳定优势；表中的“收益”仅指对应已记录对照，不是统计显著性结论。

## 协议索引

| ID | 数据/输入/划分 | 初始化与预算 | 评估 |
|---|---|---|---|
| OWT224 | WORD 96/24、BTCV24/6；1×1×3mm，224；AutoPET700/200，224 | 原OWT，20tokens/class，1200epochs，ckpt1199；具体batch见原账本 | Direct/Indirect重建；分割阈值0.15；Common8 min-component100、AutoPET20 |
| WORD070 | 旧96/24，测试6990切片，0.7×0.7×2mm，448/ROI384，ROI0.2 | 常用118800updates、seed0、20tokens、128channels；历史microbatch/AMP/续训轨迹不同 | head fixed0.5与训练6000切片校准；recon阈值0.02；min_size20/opening1；通常ckpt802 |
| Official96 | 官方Tr100选96，Val20+Ts30中选24测试，余26验证；seed0/1/42 | 每次独立训练，模型seed0，同AutoPET MAE encoder；118800updates | 验证26例校准；测试24例；最终epoch依训练清单，不固定802 |
| Inc44 | Official96 seed42；旧：脾/右肾/左肾/胆囊；新：食管/胰腺/肝/胃 | 早期59400/阶段与新118800/阶段分开；同96病例分阶段标签可见 | Old/New/All；旧阈值锁定；背景不计；binary与互斥合并结果分开 |

最新通用2D为16/卡×4卡×累积3=192；3D为6/卡×4卡×累积2=48 slabs。
不能把最新配置反填给历史运行。完整设置见 [配置契约](CONFIGURATION.md)。

## 1. OWT / LossBalance / PSEM：重建路线

问题：重建误差低是否等于小器官可分？器官均衡损失和查询日程是否有效？
原始OWT是联合全类别训练，不是增量实验。

### WORD224：相同早期24病例，post Dice（%）

| 方法/改动 | Direct | Indirect | 对照结论 |
|---|---:|---:|---|
| OWT，全图L2+LPIPS | 55.07 | 58.31 | 胆囊/食管Direct接近0；低全图误差不能证明器官重建成功 |
| LossBalance-v1，逐器官ROI等权 | 64.99 | 41.58 | Direct改善、Indirect退化；不是全面收益 |
| LossBalance-v2，频率权重 | 待核验 | 待核验 | 训练1586123完成的历史记录，不借组合结果代填 |
| LossBalance-v3a，只优化present且kept正ROI | 68.216 | 56.796 | Direct改善；removed区域只监控 |
| PSEM-v1，present类逐样本组合 | 36.266 | 34.866 | 当前实验明显退化，记录指出查询状态失配 |
| PSEM-v2a Query20，显式查询状态 | 52.311 | 49.889 | 较v1恢复，但仍不及原OWT |
| PSEM-v1 + LossBalance-v2 | 8.880 | 21.129 | 组合明显退化，保留负结果 |
| PSEM-v2 + LossBalance-v3 | 67.209 | 52.216 | Direct改善，差分读出仍弱 |
| PSEM-v3 Triplet+Loss3 | 68.353 | 68.244 | 相对Loss3 Direct仅+0.137点；Indirect+11.448点是主要收益 |

**修正早期概括：不能写“PSEM全系列无效”。** v3对WORD的Indirect有价值；
但raw/post须一起看，不能把后处理收益全部归因于模型。
完整八类、raw/post、NSD/HD95、置信区间及原始CSV出处：
[PSEM原表](archive/HISTORY.md#psem-results)。
WORD v3训练/评估114767/114768（XEC），1200epochs。
LossBalance公式、频率权重与原运行记录：
[分支公式](archive/HISTORY.md#lossbalance-experiments)及
[固定版本LossBalance历史](https://github.com/catcatcat23/owt-2026/blob/144b427c79737257406e86ea3e94d2a33e8fae7e/docs/archive/HISTORY.md)。
v1 WORD训练/评估1560195/1563822、3D1560196/1563824来自早期总览；SHA/账号待原产物核验。

### AutoPET：200病例，四类只能可靠命名label_1–4

| 2D方法 | Direct post | Indirect post | 结论 |
|---|---:|---:|---|
| OWT | 89.938 | 90.386 | 基线 |
| LossBalance-v1 | 84.186 | 83.299 | 没有跨数据集一致收益 |
| PSEM-v1 | 90.110 | 90.512 | 单次边际改善，不证明稳定提升 |
| PSEM-v2a | 90.063 | 89.888 | 与v1接近但Indirect更低 |
| PSEM+LossBalance-v2 | 79.728 | 84.673 | 明显退化 |
| PSEM-v3 Triplet+Loss3 | 88.783 | 89.938 | 未超过v1；Direct较OWT下降 |

v3训练121484、评估124936，sifansong/XEC，4A800、有效source batch192、
1200epochs；BF16/LPIPS-FP32/裁剪是有效运行的数值策略，不能写成原FP16未修复运行。
PSEM-v1 AutoPET3D Fixfr4：Direct87.583、Indirect87.519；不是v2/v3的3D对照。
数据四类向量、Whole L2/LPIPS、PSEM运行原始记录：
[PSEM归档](archive/HISTORY.md#psem-results)、
[固定版本历史](https://github.com/catcatcat23/owt-2026/blob/e8ecb1a1fc25f45c5ff93682cf69db4cb00d2fe3/docs/archive/HISTORY.md)。

### 早期WORD3D与BTCV

| 方法 | WORD3D Direct/Indirect | BTCV2D Direct/Indirect | BTCV3D Direct/Indirect |
|---|---|---|---|
| OWT | 59.93 / 60.94 | 10.39 / 10.43 | 34.16 / 12.24 |
| LossBalance-v1 | 67.57 / 62.39 | 待核验 | 待核验 |

原OWT训练/评估：AutoPET1512521/1549968+1549972；
WORD2D1527637/1539085、3D1539406/1539427；
BTCV2D1527638/1539086、3D1539407/1539428。
原表保留逐器官Dice、空预测病例数、训练batch/归一化和Step2/Step3定义：
[OWT归档](archive/HISTORY.md#owt-results)。
BTCV仅6测试病例，不能据此推断通用架构排名。

## 2. WORD070：分辨率、ROI、A–D和MAE

| 实验/问题 | 结果 | 判断 |
|---|---|---|
| 无head，1.0 spacing ROI20 → 0.7 spacing ROI20 | recon Direct75.20 → 78.16 | 更细预处理有收益；与最初224同时改变多项，不能跨表单因素归因 |
| A canvas线性 vs B canvas多尺度卷积 | A正式数值待核验；B head79.89/80.36（固定/校准），recon80.45 | 增加head不保证胜过recon |
| C单query vs D20-query | 历史摘要82.01 / 82.12 | 阈值和八类向量待核验，不据0.11点定胜负 |
| B scratch / MAE encoder / enc+recon-decoder | head固定79.89 / 80.60 / 81.09；校准80.36 / 80.97 / 81.69 | B中迁移有收益；不能推广为所有head都应加载decoder |
| B三臂recon | 80.45 / 80.57 / 81.42 | encoder-only胆囊未同步受益 |
| 旧Focal及续训 | checkpoint500/802比较待核验 | 不把排队/失败或缺结果视为方法无效 |

0.7基线训练1664156（bolinren19/SIP）；B scratch121398（antengcai23/XEC）；
B encoder128402/128404（sifansong/XEC）、encdec128401/128403（antengcai23/XEC）。
各JSON/八类向量及阈值见本页后面的“保留八类明细”及 [旧结果原文](archive/HISTORY.md#merge-results)。
A–D恢复标签为 `archive/20260927/abcd-before-retirement`；归档不等于结果无效。

## 3. WORD070：E / F / SAM已完成消融

共用主线评估协议WORD070，以下均为post Dice（%）；除标注外MAE encoder-only。
记录保留训练Job，评估Job/账号未在该行原始结果登记的，须按证据索引回查，不猜测。
历史microbatch、AMP和恢复轨迹不完全一致，不宣称整表严格单变量。

| 配置 | 固定0.5 | 训练校准后 | 训练Job | 判断/对应问题 |
|---|---:|---:|---|---|
| E scratch | 83.53 | 83.72 | 2922770 | 初始化基线 |
| E MAE encoder | 84.58 | 84.99 | 135283 | 比scratch改善 |
| E MAE encoder+recon decoder | 84.48 | 84.76 | 134915 | 未进一步提升E |
| E cross-attention | 84.30 | 85.36 | 148833 | 校准后提升，固定阈值下降，不是全面改善 |
| F scratch，reverse+Linear | 82.91 | 待核验 | 2929821 | 旧缺口不当成当前未运行结论 |
| F MAE encoder，reverse+Linear | 84.21 | 84.93 | 137867 | 相比scratch固定值改善；不高于E encoder校准 |
| F MAE encdec | 83.88 | 84.59 | 137880 | 未进一步提升F |
| SAM-tail无引导 | 83.99 | 84.94 | 2984833 | 单纯加tail未明显超过E |
| SAM全部八类分割监督 | 84.27 | 84.81 | 149295 | 固定升、校准降；不是两分支梯度隔离 |
| SAM soft prior无aux | 84.28 | 84.68 | 150615 | 固定升、校准降，不能声称全面有效/无效 |
| SAM soft prior+aux | **84.86** | **85.55** | 150971 | 相比无aux固定+0.58、校准+0.87；当前旧划分主线最好 |
| SAM两阶段缩小GT | 84.41 | 85.23 | 2998657 | 比无引导改善，保留主线 |
| SAM两阶段放大logits | 84.26 | 85.30 | 150173 | 校准略高于缩小GT，差异不证明稳定优势 |
| SAM M0–M3硬mask | 84.68 | 84.84 | 150175 | 固定有收益、校准未胜无引导；还改变总监督强度 |

主线八类向量及结果绝对路径见 [README正式结果](../README.md#旧划分同协议比较)。
三条主线保留E cross、SAM soft_prior_aux、SAM downsample_gt；其它作为消融，不删除证据。
重建不是head：E scratch/encoder/encdec recon79.54/81.17/80.29；
F encoder/encdec recon79.81/79.36。部分Indirect约7%，不能写“recon全部正常”。

### 已实现/提交但本轮证据不齐的对照

| 实验 | 改动与对照 | 记录/结果状态 |
|---|---|---|
| F Query-Dot / Reverse-Dot / 原F Linear | 保留多尺度读取，分别改变reverse和readout | 2D2965275、2965316等；历史恢复轨迹见归档；完整最终表待核验 |
| Reverse-Dot lambda_seg .01 vs .03 | 其它配方保持；恢复后的链2965316/17 vs2965318/19 | 结果待核验，不能宣称loss权重无效 |
| P2 vs P4 | P2增加S2浅层融合和更大读出，不只是插值 | 141130/131，antengcai23/XEC；结果待核验 |
| P2 softmask vs P2 | 仅token→pixel软偏置，detach、floor.2、无aux | 141132/133；结果待核验，不等于Mask2Former |
| Collector outside-attention loss | E scratch + lambda .01，无新模块 | 143780/81，sifansong/XEC；完整成绩待核验 |
| downsample_gt + soft prior，无/有aux | 固定粗mask监督，额外有无prior loss | 157058/60、157059/61；2026-10-13在训练，无最终成绩 |

精确配置、提交SHA、路径已完整保留于 [读出消融原文](archive/HISTORY.md#merge-query-readout-ablation)、
[旧状态原文](archive/HISTORY.md#merge-status)、[配置契约](CONFIGURATION.md)。

## 4. 3D：观察已更新，但不要倒推单一根因

早期B/D direct head出现低precision、高recall和z-smearing；Collector的T→1原本存在，
不能写为新增bug。历史slice-wise/PE实验单独不构成修复证明。
**后续更强F分割分支已经显著改善；不能继续把slab loss当作已证实根因。**
F还改变空间路径、交互、PE等，不能把改善只归因于增加某一层。

| 实验/协议 | 固定post | 校准post | Direct recon | 边界 |
|---|---:|---:|---:|---|
| F3D top-k，统一协议 | 83.29 | 84.46 | 78.64 | 140443–140445评估，ckpt199，24例6990切片 |
| F3D top-k，旧post3d | 83.25 | — | — | 仅与下一行比较 |
| F3D mean背景，旧post3d | 78.77 | — | — | 137413；P71.01/R92.72，胆囊总体积比2.72 |
| D slice-wise / spatial，历史均值 | 21.13 / 30.25 | 待核验 | 待核验 | 协议/最终JSON待核验；不能证明loss或PE根因 |
| 3D E式 / Query-Dot / Reverse-Dot | 待核验 | 待核验 | 待核验 | 140585、140591、140595及各评估链 |
| 3D SAM soft / 两阶段 + MAE | 待核验 | 待核验 | 待核验 | 当前状态见STATUS，不把CPU测试当GPU结果 |

F3D Indirect6.95异常；head恢复不等于差分重建已修好。
AutoPET3D MAE2951384的100000updates完成见旧状态来源，下游迁移必须核对兼容加载报告。
2D y/x、3D t/y/x PE及评估slab融合细节见 [配置](CONFIGURATION.md)；
完整统一3D evaluator设计在 [原文](archive/HISTORY.md#merge-unified-3d-evaluation)。

## 5. 诊断实验：信息存在与可读出必须分开

| 实验 | 保持/改变 | 已观察结果与边界 |
|---|---|---|
| 冻结P4探针 | 冻结全模型，交互前/后相同轻量head独立监督训练 | 设计/脚本完整；最终数值本轮待核验。额外训练和不同loss，不证明query是唯一瓶颈 |
| E0127/0114同病例探针 | 同checkpoint原query vs新P4 probe，raw .5 | 不与post/calibrated混排；结果待核验 |
| Token替换 | 保持空间特征与读出，换训练均值/跨病例tokens | GT存在性匹配是oracle；不是正式SOTA或训练seed |
| 存在性干预 | positive_only / negative_only / mean_matched | 原始84.58、mean_matched85.21；部分小器官召回反而下降，0127仍零重叠 |
| 全24例共同badcase审计 | E encoder / 原F encoder / SAM-tail配对 | 胆囊0041/0125/0127/0145、胰腺0114共同低分；SAM胰腺17/24病例低于E |
| crop/ROI覆盖 | 全6990测试mask及512训练切片抽样ROI | 测试center448保留100%八类GT；抽样focus ROI保留100%；不覆盖所有随机增强 |

本轮直接读取 `artifacts/token_presence_review/results.json`（complete=true）及
`artifacts/coverage_case_audit_20260928/README.md`。
这些路径基于SIP仓库根 `/gpfs/work/aac/bolinren19/OD_OWT`，不保证GitHub包含大产物。
在测试112分辨率GT没有阳性切片完全消失，但不能推断边界没有损失、空间配准完全正确。
P/R、体积比和完整2304行配对记录在coverage目录，不只展示挑选的切片。
探针76/20划分来自encoder见过的96训练病例，不是独立端到端测试。
完整设计与训练/推理边界：[P4原文](archive/HISTORY.md#merge-frozen-p4-probe)、
[Token原文](archive/HISTORY.md#merge-token-intervention)。

## 6. Official96：三次随机划分，不是三折或五次训练seed

| split seed（模型seed均0） | 训练Job | 测试切片 | fixed | 验证校准 |
|---|---|---:|---:|---:|
| 42 | 2998655 | 7315 | 84.95 | 85.64 |
| 0 | 3006796 | 6854 | 85.20 | 85.57 |
| 1 | 3006797 | 7079 | 86.71 | 87.44 |
| 等权均值 | 三次独立训练 | 不合并病例 | **85.62** | **86.22** |

八类及来源已在 [README三次划分](../README.md#三个配置的正式指标)。
测试集有交叠；旧24例和替换5个badcase的探索性评估不得纳入均值。
替换病例实验沿用checkpoint与既定阈值，属于测试集敏感性分析，不是新训练seed。
其完整数值待原结果核验，不因本次整理补造。
Official seed0/1 Indirect recon7.19/7.10异常，head结果与其分开。

## 7. 增量：保留、适配与预算

精确冻结矩阵、八类和证据见 [增量专项](INCREMENTAL_READOUT_ABLATION.md)。
两阶段均访问同96图像、开放不同标签，不是阶段间禁止旧图像的no-replay设置。

| 实验/预算 | fixed Old/New/All | 校准Old/New/All | 结论 |
|---|---|---|---|
| E早期Stage1，59400更新 | 84.89 / NA / NA | 85.48 / NA / NA | 2998659；不可冒充八类Offline |
| E共享空间+读出解冻Stage2，59400 | 0.04 / 81.20 / 40.62 | 0.09 / 82.23 / 41.16 | 3004004；旧slot参数冻结仍发生旧输出崩溃 |
| SAM私有交互Stage1，118800 | 86.78 / NA / NA | 87.12 / NA / NA | 156536/37 |
| SAM私有交互Stage2，118800 | 86.78 / 76.39 / 81.58 | 87.12 / 76.57 / 81.84 | 156538/39；旧binary保持，新类仍有差距 |
| E只训slot / query_shared / query_split | 待核验 | 待核验 | 不混不同Stage1/预算；运行状态见STATUS |
| SAM私有交互+4240参数adapter | 待核验 | 待核验 | 157119/20；新四类共用，不是每slot独立tail |
| PCDD论文4–4 | 不同协议 | 84.09 / 82.75 / 83.42（论文值，非本项目校准） | 外部参考，不作同协议排名 |

本轮读取 `artifacts/inc44_spatial_readout_20261004/evaluation/stage_comparison.json`，
保留八类：固定0.086/0/0.030/0.037/68.427/76.064/95.060/85.236；
校准0.227/0.003/0.047/0.103/71.882/76.064/95.060/85.901。
SAM最新与E早期同时改变架构/预算，不能把40→81全部归因于私有化。
全预算两阶段总237600更新，不与118800更新Offline声称算力匹配。
旧参数hash一致、旧binary分数不变、互斥mask不变是三个不同主张。
SAM direct recon75.28、indirect6.10；不能用低重建loss证明解耦成功。

## 8. CNN最新路线与证据缺口

ResEnc风格/MedNeXt-B 2D输出S16替换ViT，保留原Collector和SAM解码；
不是完整nnU-Net/MedNeXt；2026-10-14将跳接来源由独立浅层CNN替换为编码器S8/S4，
预训练结构/权重接口保持，旧S16-only实现记录保留在历史档案。本配置尚无Dice。
AutoPET masked reconstruction预训练 → encoder-only迁移 → SAM soft_aux/downsample两组。
CPU实现核验不等于GPU性能，当前无Dice。精确接口、参数量、100k预训练预算、
迁移key和提交链见 [配置](CONFIGURATION.md)；当前任务见 [状态](STATUS.md)。

仍需核验而非擅自定论：A与C/D阈值、F dot/P2/PE/Collector完整结果、
LossBalance-v2独立最终表、探针最终数值、替换病例探索结果、旧E只slot完整指标、
部分运行SHA和评估Job、3D各条件的最终统一指标。
文档存在的所有历史字段均已归档；“待核验”表示本轮未拿到最终证据，不代表实验没做过。

## 保留八类明细与原结果路径

下面保留旧账本中的向量和证据段落；其中“本次/当前”限定原记录日期。
上面的协议与状态勘误优先。以下为历史结果，不追加推断。
| 输出 | 脾 | 右肾 | 左肾 | 胆囊 | 食管 | 胰腺 | 肝 | 胃 | 均值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F2D scratch fixed | 94.15 | 93.23 | 93.29 | 55.34 | 74.87 | 73.82 | 95.16 | 83.39 | 82.91 |
| F2D encoder calibrated | 94.83 | 94.77 | 94.31 | 60.62 | 77.29 | 75.92 | 95.32 | 86.39 | 84.93 |
| F2D encdec fixed | 94.50 | 93.88 | 93.70 | 57.46 | 75.23 | 74.76 | 95.35 | 86.16 | 83.88 |
| F2D encdec calibrated | 94.88 | 94.80 | 94.37 | 59.10 | 77.45 | 74.73 | 95.35 | 86.05 | 84.59 |
| F3D topk fixed | 94.34 | 93.50 | 92.93 | 56.26 | 73.57 | 74.32 | 95.32 | 86.11 | 83.29 |
| F3D topk calibrated | 94.83 | 94.52 | 93.96 | 59.40 | 77.30 | 74.53 | 95.32 | 85.84 | 84.46 |

3D背景归约消融：旧post3d固定协议下topk83.25、mean78.77（137413），差4.48。
mean Precision71.01%、Recall92.72%，胆囊预测/GT总体积比2.72。此结果支持topk更有效
抑制假阳性，但不得把旧post3d均值与统一2D后处理/校准值当作同协议比较。
Indirect仍异常：F3D topk6.95%、F2D encdec7.91%；Direct正常不意味着差分重建正常。

结果证据（各账号路径独立，不互换）：

- F scratch：bolinren19/SIP `.worktrees/arm_f_2d_batch8/Results/OrganSlotBank/evaluation/ArmF_2D_2929821/results.json`。
- F encoder：sifansong/XEC `worktrees/arm_f_mae_encoder_df7ed39/Results/OrganSlotBank/evaluation/Common8/WORD_2D/ArmF_2D_137867_ckpt802_*`。
- F encdec：antengcai23/XEC `worktrees/arm_f_mae_encdec_20260913/Results/OrganSlotBank/evaluation/Common8/WORD_2D/ArmF_2D_137880_ckpt802_*`。
- 3D topk统一：sifansong/XEC `evaluations/arm_f_3d_137410_unified_d53bb66/{head,reconstruction}/results.json`。
- 3D mean旧协议：antengcai23/XEC `worktrees/arm_f_batch6_20260913/Results/OrganSlotBank/evaluation/ArmF_3D_137412/results.json`。

## 历史八类明细：Arm E MAE三臂（2026-09-14记录）

24病例/6990切片，checkpoint802精确加载。主指标为case Dice presence mean；
fixed0.5 binary_post与训练集校准selected_post分列，禁止测试集逐器官拼接最优输出。

| 初始化 | 固定0.5 post | 训练集校准 post | Recon Direct-post |
|---|---:|---:|---:|
| Encoder-only | **84.58** | **84.99** | **81.17** |
| Encoder+reconstruction decoder | 84.48 | 84.76 | 80.29 |
| Scratch | 83.53 | 83.72 | 79.54 |
| PCDD论文Offline全量监督参考 | **85.47** | — | — |

PCDD来源：[AAAI2026论文Table2](https://ojs.aaai.org/index.php/AAAI/article/download/38406/42368)。
85.47是Offline，不是增量方法成绩。WORD增量4-4/4-2/2-2/7-1的All分别83.42/83.50/72.02/72.89。
该行是跨协议参考，尚未同划分复现；不能用全量监督胜过增量行来宣称抗遗忘优势。
encoder-only距Offline固定阈值0.89、校准0.48个百分点。单seed的小差异不能认定稳定优势。

| 初始化/输出 | 脾 | 右肾 | 左肾 | 胆囊 | 食管 | 胰腺 | 肝 | 胃 | 均值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Encoder fixed | 94.24 | 93.55 | 93.68 | 60.72 | 75.82 | 76.61 | 95.34 | 86.72 | 84.58 |
| Encdec fixed | 94.34 | 93.84 | 93.95 | 59.56 | 75.72 | 76.47 | 95.36 | 86.58 | 84.48 |
| Scratch fixed | 94.11 | 93.80 | 93.35 | 57.34 | 74.53 | 75.34 | 95.05 | 84.71 | 83.53 |
| Encoder calibrated | 94.66 | 94.56 | 94.38 | 60.76 | 76.81 | 76.86 | 95.34 | 86.55 | 84.99 |
| Encdec calibrated | 94.74 | 94.79 | 94.50 | 59.04 | 77.28 | 76.14 | 95.36 | 86.21 | 84.76 |
| Scratch calibrated | 94.61 | 94.45 | 93.86 | 57.19 | 76.09 | 74.54 | 95.05 | 83.96 | 83.72 |
| Encoder recon Direct-post | 93.13 | 92.01 | 92.37 | 53.14 | 72.17 | 71.53 | 94.65 | 80.37 | 81.17 |
| Encdec recon Direct-post | 92.64 | 92.37 | 91.56 | 50.57 | 69.85 | 70.06 | 94.56 | 80.74 | 80.29 |
| Scratch recon Direct-post | 92.76 | 91.50 | 91.02 | 48.46 | 69.37 | 70.09 | 94.38 | 78.72 | 79.54 |

Encoder-only head/recon都略高，暂不支持“decoder提升重建但牺牲分割”解释。
迁移decoder输入分布不匹配和训练波动仍是假设。此处decoder是重建Transformer，
不是pixel decoder、AHER或分割head。Encoder recon Indirect-post仅7.53，scratch为7.21；
不能把Direct正常概括为所有重建分割输出正常。

证据入口（各目录内heads_test、reconstruction_fixedthr002的results.json）：

- sifansong/XEC，runtime worktrees/arme_mae_encoder_runtime_f70b831，
  Results/OrganSlotBank/evaluation/Common8/WORD_2D/ArmE_MAE_encoder_resume500_135283_ckpt802_*；
  评估135284 COMPLETED exit0，head完整6990，recon6990，均exact。
- sifansong/XEC，runtime worktrees/arme_mae_runtime_43cad30，
  同上评估父目录/ArmE_MAE_encdec_resume500_134915_ckpt802_*；134916 COMPLETED exit0。
- bolinren19/SIP，.worktrees/orgslot_querymask_multiscale_pixel，
  同上评估父目录/OrgSlot_WORD07072_ROI20_ArmE_Resume500_2922770_ckpt802_*。

以下保留历史结果；旧局部排名不代表最新全项目排名。



本页合并既有已提交文档，不是本次重新推理或重新审核全部 JSON。
只称“项目内已记录结果”；没有外部 PCDD 同协议复核，不能声称公开 SOTA。
八类顺序：脾、右肾、左肾、胆囊、食管、胰腺、肝、胃。数值单位为百分比。


## 已完成结果：统一输出排名

| 排名 | 实验 / 输出 | 脾脏 | 右肾 | 左肾 | 胆囊 | 食管 | 胰腺 | 肝脏 | 胃 | 小器官均值 | 八类均值 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1** | **Arm B joint small-organ / reconstruction Direct-post** | 92.74 | 91.48 | 91.92 | **53.74** | **69.48** | **70.35** | **94.58** | 79.28 | **64.53** | **80.45** |
| 2 | Arm B head / train-calibrated selected-post | **92.98** | **91.73** | **92.38** | 51.45 | 68.75 | 68.71 | 94.41 | 82.47 | 62.97 | 80.36 |
| 3 | Arm B head / fixed-0.5 binary-post | 92.62 | 91.04 | 91.80 | 49.72 | 67.86 | 69.03 | 94.41 | **82.65** | 62.20 | 79.89 |
| 4 | 0.7 spacing ROI20 / no segmentation head / reconstruction Direct-post | 92.11 | 91.03 | 91.51 | 40.54 | 67.32 | 69.53 | 93.84 | 79.40 | 59.13 | 78.16 |
| 5 | 1.0 spacing ROI20 / no segmentation head / reconstruction Direct-post | 90.22 | 89.96 | 89.47 | 40.60 | 58.67 | 63.92 | 93.26 | 75.48 | 54.40 | 75.20 |

Arm B 的 train-calibrated 阈值为
`[0.7, 0.8, 0.8, 0.9, 0.9, 0.8, 0.5, 0.6]`，来自训练集 calibration，
并在测试前固定；因此可报告，但固定 0.5 仍是 head 的主结果。reconstruction 使用
固定阈值 0.02。post-processing 统一使用 minimum component size 20 和 opening
radius 1。


## 可追溯来源

- 0.7 spacing baseline：
  `bolinren19 / SIP` 的
  `/gpfs/work/aac/bolinren19/OD_OWT_orgslot_resolution/Results/OrganSlotBank/Common8/WORD_2D/OrgSlot_WORD07072_C448_R384_BaseLoss_ROI20_1664156_eval/results.json`。
- Arm B reconstruction：
  `antengcai23 / XEC` 的
  `.../OrgSlot_WORD07072_ROI20_RetainedMultiConv_SmallOrgan_L0.01_121398_reconstruction_fixedthr002/results.json`。
- Arm B head：
  `antengcai23 / XEC` 的
  `.../OrgSlot_WORD07072_ROI20_RetainedMultiConv_SmallOrgan_L0.01_121398_heads_test/results.json`
  和同目录 `resolved_config.json`。
- 1.0 spacing ROI20 baseline：
  [ORGAN_SLOTBANK_EXPERIMENT_SUMMARY_CN.md](archive/HISTORY.md#source-7)。

Arm B 两个正式 JSON 均加载 epoch 802 checkpoint，`exact=true`；reconstruction
结果标记完整测试集，head 结果标记 `complete_split=true`，样本数均为 6990。


## 已完成的 Arm B MAE：统一八类结果
指标：24 病例 / 6990 切片，checkpoint-802 exact=true，case Dice presence mean ×100。
Reconstruction 为 Direct-post，固定阈值 0.02，min_size=20、opening_radius=1。
两组 MAE 原始 JSON 已归档到 artifacts/mae_recon_audit_20260908/。
原始 Arm B 数字来自既有 ORGSLOT_WORD_COMMON8_SOTA_CN.md 的已验证记录。

| 初始化 / recon | 脾脏 | 右肾 | 左肾 | 胆囊 | 食管 | 胰腺 | 肝脏 | 胃 | 八类均值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Encoder+decoder | 93.19 | 92.12 | 92.21 | 57.05 | 69.83 | 71.33 | 94.70 | 80.95 | **81.42** |
| Encoder-only | 92.56 | 92.61 | 91.78 | 50.98 | 71.00 | 71.10 | 94.25 | 80.25 | **80.57** |
| Scratch Arm B | 92.74 | 91.48 | 91.92 | 53.74 | 69.48 | 70.35 | 94.58 | 79.28 | **80.45** |

| 初始化 | Head fixed-0.5 post | Head train-calibrated post | Recon Direct-post |
|---|---:|---:|---:|
| Scratch Arm B | 79.89 | 80.36 | 80.45 |
| Encoder-only | 80.60 | 80.97 | 80.57 |
| Encoder+decoder | 81.09 | 81.69 | 81.42 |

Head/recon数字沿用原审计记录；本次文档整理没有重新计算recon JSON。
Encoder+decoder 相对 encoder-only 的 recon 提高 0.8552 个百分点。
相对 scratch recon 约 +0.97；胆囊提升最明显，但 encoder-only 胆囊低于 scratch。
不能声称所有器官同步获益，或凭单 seed 证明收益机制。
以上是 Arm B MAE 内部排名，不是全项目新 SOTA；Arm C/D 的既有约 82% 结果仍需同列比较。
Head/recon 各自单独排名，不在测试集逐器官挑输出拼接。

## 数据来源
- Encoder-only：sifansong / XEC，训练 128402，评估 128404。
  worktree: /gpfs/work/aac/sifansong/worktrees/orgslot_autopet_mae
  result: Results/OrganSlotBank/evaluation/Common8/WORD_2D/OrgSlot_WORD07072_ROI20_RetainedMultiConv_SmallOrgan_L001_AutoPETMAE_EncoderOnly_128402_ckpt802_reconstruction_fixedthr002/results.json
- Encoder+decoder：antengcai23 / XEC，训练 128401，评估 128403。
  worktree: /gpfs/work/aac/antengcai23/worktrees/orgslot_autopet_mae
  result: Results/OrganSlotBank/evaluation/Common8/WORD_2D/OrgSlot_WORD07072_ROI20_RetainedMultiConv_SmallOrgan_L001_AutoPETMAE_EncDec_128401_ckpt802_reconstruction_fixedthr002/results.json
- 两文件 complete_test_set=true，samples=6990，八类 case_count 均为24。
