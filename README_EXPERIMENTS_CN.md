# OrganSlot / PSEM 实验状态与结果归档

> 基础状态快照：2026-09-01 16:30；最新迁移更新：2026-09-01 22:00（Asia/Shanghai）
> 规则：运行状态以 Slurm 为准；结果仅收录已经由完整 checkpoint、评估 JSON/CSV 或训练日志核验的数值。`PENDING`、计划和 exploratory 结果不写成正式结论。

## 0. 最新迁移与活动任务（22:00增量）

本节覆盖下方16:30基础快照中已经变化的任务；未变化的完成结果仍以下文为准。

| 状态 | 实验目的 | 账号/集群/QOS | Job ID | 资源/进度 | 下一审核 |
|---|---|---|---:|---|---|
| RUNNING | 2D Arm A：linear head严格基线 | bolinren19 / SIP / angelosstefanidis-8a800 | 2413564 | 4×A800；已完成epoch439；checkpoint-400；loss有限 | checkpoint-500与最终802 |
| RUNNING | 2D Arm D：20个独立query + shared pixel decoder | bolinren19 / SIP / angelosstefanidis-8a800 | 2548702 | 4×A800；已完成epoch732；checkpoint-700；loss有限 | checkpoint-800/802与统一推理 |
| PENDING | AutoPET MAE 100k预训练 | bolinren19 / SIP / sifansong-4a800 | 2713895 | 4×A800；Priority | 启动后确认预训练loss与checkpoint |
| VALIDATED | 3D Arm C迁移真实数据门禁 | bolinren19 / SIP / sifansong-4a800 | 2755712 | 1×A800；2个optimizer update与checkpoint-0均通过；Slurm状态FAILED仅因旧validator要求2步覆盖全部slot | 已离线用修正规则复验通过，不再重复smoke |
| PENDING | 3D Arm C正式训练：20 tokens汇总为1 query，与共享3D pixel feature点积 | bolinren19 / SIP / sifansong-4a800 | 2755754 | 4×A800、24 CPU、192 GB；Priority；预计启动时间仅供参考 | 启动后首2步、checkpoint-10/50/100/199、NaN与吞吐 |
| DEPENDENCY | 3D Arm C病例级真正3D评估 | bolinren19 / SIP / sifansong-4a800 | 2755765 | 1×A800、10 CPU、128 GB；afterok:2755754 | 24病例×8器官，Dice/IoU/precision/recall/NSD/HD95 |
| PENDING | 3D Arm B从checkpoint-50续训 | sifansong / XEC / 4gpus | 127706 | 4×A800、24 CPU、192 GB；Priority | epoch51连续性、BF16与loss有限性 |
| FAILED | 3D Arm D真实数据门禁 | antengcai23 / XEC / 4gpus | 127723 | 1×A800；因MAX_TRAIN_SAMPLES=8清空ROI池而失败 | 使用本次修复代码重新提交smoke |
| BLOCKED | 3D Arm D正式训练/评估 | antengcai23 / XEC / 4gpus | 127726 / 127727 | DependencyNeverSatisfied / Dependency | 先替换失败smoke链 |

### 本次Arm C迁移修复

- SIP路径已逐项独立配置：Python、WORD070数据、ROI index、LPIPS权重、输出和日志均使用bolinren19本地路径，没有混用XEC账号目录。
- 数据身份验证器支持显式ROI路径；本地ROI文件名不同，但核心签名、28,586训练条目和6,990测试条目与固定0.7协议完全一致。
- 3D smoke不再把训练dataset截成前8条；完整ROI池为胆囊1,631、食管3,827、胰腺3,753。
- SIP PyTorch缺少BF16 depthwise Conv3d kernel；仅将两个3D depthwise层置于局部FP32/autograd路径，其余AMP保持BF16，并通过输入/权重梯度回归测试。
- 两步smoke允许未随机抽到的slot，但仍强制至少一个阳性小器官、有限重建/感知/分割loss、ROI生效、权重关系正确和checkpoint存在。
- XEC旧Arm C链127722/127724/127725已失败或取消，不会与SIP正式训练重复运行。

结论：Arm C迁移已经完成到“正式训练已排队、3D评估依赖已建立”；当前尚无3D Arm C八类指标，不能提前判断是否优于Arm B/D。

## 1. 账号、集群与 QOS

这里的“账号”必须区分 Linux 用户、Slurm account 和 QOS。特别是 `bolinren19/SIP` 同时拥有两套独立关联，不能合并计算：

| Linux用户 | 集群 | Slurm account / QOS | GPU上限 | 当前占用 | 说明 |
|---|---|---|---:|---:|---|
| `bolinren19` | SIP | `angelosstefanidis / 8a800` | 8×A800 | 8×A800 | 2D Arm A和Arm D各占4卡，已满 |
| `bolinren19` | SIP | `sifansong / 4a800` | 4×A800 | 0 | 独立额度，适合1卡推理和汇总任务 |
| `sifansong` | SIP | `sifansong / 4a800` | 4×A800 | 0 | 无活动任务；SIP数据/环境路径尚未完成启动前核验 |
| `antengcai23` | SIP | `sifansong / 4a800` | 4×A800 | 0 | 无活动任务；SIP数据/环境路径尚未完成启动前核验 |
| `sifansong` | XEC | `sifansong / 4gpus` | 4×GPU | 0 running | 3D Arm B、C入口任务等待；同一用户最多只能运行一组4卡训练 |
| `antengcai23` | XEC | `sifansong / 4gpus` | 4×GPU | 0 running | 3D Arm D入口任务等待；与`sifansong/XEC`是独立用户额度 |

快照时的物理资源：

- SIP `gpua800`：常规A800节点没有连续4张空闲卡；`gpua800n6`最多只有约2张零散标准A800可见。因此有空闲QOS不等于4卡训练能够立即启动。
- XEC `gpua800,gpua8001t`：A800类GPU在快照时全部分配，没有即时空闲卡。
- Slurm预测：XEC Arm C/D smoke最早约为2026-09-02 22:44；3D Arm B约为2026-09-03 19:07。预测会随其他用户任务变化。

## 2. 正在训练或等待训练

| 状态 | 实验目的 | 账号/集群/QOS | Job ID | 资源 | 当前进度或依赖 | 下一检查点 |
|---|---|---|---:|---|---|---|
| RUNNING | 2D Arm A：验证简单linear head，在0.7 spacing、ROI20和small-organ loss下作为严格head基线 | `bolinren19/SIP/8a800` | `2413564` | 4×A800、40 CPU、256 GB | epoch约377/802；最新checkpoint-300；loss有限 | checkpoint-400及NaN/ROI/seg loss稳定性 |
| RUNNING | 2D Arm D：20个organ token保留为20个独立query，与共享pixel decoder点积并LogMeanExp融合 | `bolinren19/SIP/8a800` | `2548702` | 4×A800、24 CPU、192 GB | epoch约657/802；最新checkpoint-600；loss有限 | checkpoint-700、最终802；预计比Arm A先释放GPU |
| PENDING | AutoPET MAE预训练：为后续WORD的encoder-only与encoder+decoder初始化对照提供预训练权重 | `bolinren19/SIP`（提交脚本QOS需再次核对） | `2713895` | 4×A800、24 CPU、192 GB | `Priority` | 启动配置、100k更新、预训练checkpoint完整性 |
| PENDING | 3D Arm B续训：原multiscale AHER-canvas head，修复NaN/BF16后从checkpoint-50继续 | `sifansong/XEC/4gpus` | `127706` | 4×A800、24 CPU、192 GB | `Priority` | 必须确认从epoch51连续恢复，BF16/梯度/optimizer均有限 |
| PENDING | 3D Arm C真实数据门控：20 tokens先汇总成1个query，再与3D pixel feature点积 | `sifansong/XEC/4gpus` | `127722` | 1×A800、8 CPU、64 GB | `Priority` | 两次真实optimizer step、ROI和small-organ诊断通过 |
| DEPENDENCY | 3D Arm C正式训练 | `sifansong/XEC/4gpus` | `127724` | 4×A800、24 CPU、192 GB | `afterok:127722` | checkpoint-10/50/100/199和数值稳定性 |
| PENDING | 3D Arm D真实数据门控：20个独立query的3D版本 | `antengcai23/XEC/4gpus` | `127723` | 1×A800、8 CPU、64 GB | `Priority` | 两次真实optimizer step、BF16和反向传播通过 |
| DEPENDENCY | 3D Arm D正式训练 | `antengcai23/XEC/4gpus` | `127726` | 4×A800、24 CPU、192 GB | `afterok:127723` | checkpoint-10/50/100/199和数值稳定性 |

### 当前训练分布判断

- 正确部分：3D Arm D放在`antengcai23/XEC`，能与`sifansong/XEC`独立占用4卡QOS。
- 主要瓶颈：3D Arm B和Arm C都位于`sifansong/XEC`，二者正式训练不能并行。
- 推荐但尚未执行：优先调查将3D Arm C迁到一个已核验数据和环境的SIP 4卡通道。迁移前必须核验SIP账号自己的Python、WORD070数据、ROI index、LPIPS权重和输出路径；不能直接套用XEC路径。
- 不建议中断已经运行的2D Arm A/D。Arm D离终点更近，完成后自然释放`8a800`中的4卡。

## 3. 等待推理与结果汇总

| 推理/汇总目的 | 账号/集群 | Job ID | 前置条件 | 当前问题 | 完整性标准 |
|---|---|---:|---|---|---|
| Focal checkpoint-802统一推理，和checkpoint-500同协议比较续训收益 | `bolinren19/SIP` | `2715065` | checkpoint已存在 | 当前受`QOSMaxGRESPerUser`限制；应迁到空闲的`4a800` 1卡通道 | 6990 slices、24 cases、八类指标齐全 |
| 2D Arm A统一head评估 | `bolinren19/SIP` | `2715066` | `afterok:2413564` | 等训练完成 | 固定0.5阈值；与B/C/D相同数据、脚本和后处理 |
| 2D Arm D统一head评估 | `bolinren19/SIP` | `2715067` | `afterok:2548702` | 等训练完成 | 同上 |
| 2D A/B/C/D总表 | `bolinren19/SIP` | `2715236` | 依赖head评估完成 | 等待依赖 | 八类Dice、mean、precision/recall、速度和资源完整 |
| 3D Arm B病例级评估 | `sifansong/XEC` | `127707` | `afterok:127706` | 等训练 | 24个病例×8器官；Dice/IoU/precision/recall/NSD/HD95 |
| 3D Arm C病例级评估 | `sifansong/XEC` | `127725` | `afterok:127724` | 等训练 | 同上 |
| 3D Arm D病例级评估 | `antengcai23/XEC` | `127727` | `afterok:127726` | 等训练 | 同上 |

推理任务通常只需1张GPU。合理的资源策略是把SIP本地推理从已满的`8a800`关联迁到`bolinren19 --account=sifansong --qos=4a800`，优先产出已训练模型的结果；不要用4张训练卡长期等待一个1卡评估。

## 4. 已完成且数值已核验的实验

### 4.1 WORD：0.7 spacing + ROI20 + 2D Arm B multiscale head

checkpoint-802完整加载，覆盖6990张测试切片和24个病例。该结果是2D slice级统一head协议，不应冒充病例级3D评估。

| 器官 | 固定0.5 Dice | 训练集校准阈值 Dice |
|---|---:|---:|
| 脾脏 | 92.62% | 92.98% |
| 右肾 | 91.04% | 91.73% |
| 左肾 | 91.80% | 92.38% |
| 胆囊 | 49.72% | 51.45% |
| 食管 | 67.86% | 68.75% |
| 胰腺 | 69.03% | 68.71% |
| 肝脏 | 94.41% | 94.41% |
| 胃 | 82.65% | 82.47% |
| 前景平均 | 79.89% | 80.36% |

分析：

- multiscale head能稳定读出肾、脾、肝等大器官，但胆囊仍是明显短板。
- 校准只带来`+0.47`个百分点mean Dice，说明主要瓶颈不是单一全局阈值。
- 胰腺固定0.5反而略优于训练校准，提示不能在测试集上逐器官调阈值并把它当作方法增益。
- 该head依赖AHER canvas；如果OrganCollector/AHER已经压缩空间细节，增加卷积容量不能完全恢复细长、小体积和低对比边界。这正是Arm C/D绕开canvas、引入共享pixel path的动机。

### 4.2 AutoPET：PSEM-v3正式下游评估

已完整处理200个病例：

| 方法 | 4类平均post Dice |
|---|---:|
| Indirect input-minus-without-class | **89.94%** |
| Direct class token | 88.78% |

分析：Indirect比Direct高`1.16`个百分点。该结论支持PSEM-v3表征有效，但还需要与相同AutoPET拆分、相同后处理和相同checkpoint选择规则下的历史PSEM版本比较；不能把不同协议的数字直接排序。

### 4.3 WORD分辨率消融

- 后续实验已统一采用`0.7×0.7×2.0` spacing、输入448、ROI384、ROI probability 0.2。
- 分辨率消融与定性可视化已经完成，当前不再重复训练。
- 该消融曾使用24个测试病例进行探索性比较，因此应标注为`exploratory ablation`；正式方法选择应由验证集完成，测试集只作最终报告。

## 5. 尚不能写成最终结论的项目

- 2D Arm A、Arm D尚未到checkpoint-802，也未完成统一推理。
- 3D Arm B/C/D尚未完成训练和病例级3D评估，当前没有可比较的3D八类结果。
- Focal checkpoint-802统一推理尚未启动，不能判断从checkpoint-500续训是否提高最终指标。
- MAE预训练尚未启动，encoder-only和encoder+decoder初始化对照也未产生WORD结果。
- Arm C历史结果和可视化若要进入SOTA表，必须从其正式评估JSON重新导入精确八类指标和证据路径；本文件不凭记忆补数值。

## 6. 推荐资源分布与执行顺序

1. 保持2D Arm A/D在`bolinren19/SIP/8a800`继续运行，不中断。
2. 将Focal-802及后续2D A/D推理放到`bolinren19/SIP/4a800`，每项1卡；优先产出已完成checkpoint的结果。
3. `sifansong/XEC`优先3D Arm B，因为它是已经从checkpoint-50恢复的基准；Arm C与它共享4卡上限，应避免同时保留多个无优先级说明的4卡正式任务。
4. `antengcai23/XEC`继续3D Arm D，形成与B独立并行的训练通道。
5. 若要让B/C/D真正三路并行，先为Arm C核验并准备SIP 4卡环境，再迁移；确认新任务成功启动后才取消XEC副本，避免重复训练或两边都失败。
6. Arm D 2D预计最先完成；完成后先跑统一推理，再让MAE占用释放的4张`8a800`训练卡。

## 7. 下次固定审核清单

1. 2D Arm D是否达到checkpoint-700/802，依赖评估`2715067`是否释放。
2. 2D Arm A是否达到checkpoint-400，loss、ROI比例和小器官监督是否仍有限。
3. Focal推理是否已经迁到4a800并完成6990 slices/24 cases。
4. 3D Arm B是否从checkpoint-50连续恢复，且未再次出现NaN。
5. 3D Arm C/D smoke是否完成两次真实optimizer step；正式任务是否自动释放。
6. 所有3D评估是否为病例级重建，而不是把2D slice Dice重新命名为3D。
7. 每次报告同时记录：目的、账号/集群/QOS、Job ID、资源、epoch、最新checkpoint、loss有限性、推理完整性、八类指标、是否达到预期和下一步。

## 8. 证据与相关文档

- 3D Arm C/D方法及公平协议：`docs/ORGSLOT_WORD070_QUERYMASK_3D_EXPERIMENT_CN.md`
- small-organ loss：`docs/ORGSLOT_WORD070_SMALL_ORGAN_LOSS_CN.md`
- 完整实验总结：`docs/COMPLETE_EXPERIMENT_SUMMARY.md`
- 分辨率定性结果：`OD_OWT_orgslot_resolution/artifacts/resolution_ablation_qualitative/`
