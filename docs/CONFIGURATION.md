# 统一配置与部署契约

## Soft-prior aux 的DDP指标契约

2026-10-01：`150613`失败并非已确认的NaN。TGR在某rank只保留背景时，coarse预测为空是合法情况；旧实现提前返回0辅助loss，却没有生成`soft_prior_p16_loss/p8_loss/p4_loss`，其他rank有这三个字段，`reduce_metrics`的schema检查因此终止。修复在提前返回前初始化三个0值字段，非空分支仍写实际loss；不改loss权重、mask机制或空监督语义，也不删除schema保护。回归覆盖双rank不同保留情况、有限梯度、真实字段不一致仍报错。

恢复脚本`slurm/orgslot/train/sam_tail_soft_prior_aux_resume.sbatch`只恢复已核验的checkpoint-0，从epoch1续训并恢复optimizer/scaler。启动后先运行两GPU NCCL指标回归；失败则不进入正式训练。新训练需新`afterok`评估；旧`150614`依赖失败不会自行恢复。

## incremental44

独立入口 `main_pretrain_orgslot_incremental44.py`，启动器 `scripts/orgslot/run_incremental44.sh`。
基于 **2D E cross-attention + AutoPET MAE encoder-only**，不修改原离线训练入口。
这是**同图像、分阶段标签开放**的4+4机制实验：两阶段都用固定96训练病例；不是禁止重访旧图像的 data-incremental/no-replay 协议。
使用上节官方池新划分的96训练/26验证/24测试；不可使用“替换五个badcase”的探索性测试清单。

| 内容 | Stage1 | Stage2 |
|---|---|---|
| 可见器官GT | 脾、右肾、左肾、胆囊 | 食管、胰腺、肝、胃 |
| slot数 | 背景+4旧类=5 | 背景+4旧类+4新类=9 |
| 初始化 | 仅AutoPET MAE encoder；其余随机 | 精确加载Stage1；四个新slot分别复制同一Stage1背景slot |
| 可训练 | 原E共享模块和5个slot | 仅4新slot；背景按下述策略；共享模块/旧slot全部冻结 |
| ROI定向池 | 仅胆囊4 | 仅食管5、胰腺6 |
| 重建 | 既有随机保留组合、MSE+LPIPS | 新GT+旧类teacher伪区域的masked MSE；不计算LPIPS |
| 分割 | 旧4类small-organ loss | 新4类相同loss；默认仅保留slot监督，ROI目标强制保留 |

Stage1背景=非旧4类，因此包括未来新类；**Stage2不能再把所有非新类GT位置当背景**。
冻结Stage1 teacher的旧4类calibrated head sigmoid（与Stage1分割损失输入一致）：新GT优先；恰好一个旧类≥0.9的位置为旧伪区域；全部旧类<0.1且不在新GT的位置为候选背景；其他位置ignore。teacher背景head从未监督，不参与伪标签。
这些阈值和背景权重是待验证的实验超参，不是论文默认值。teacher有误差，候选背景并非真实背景。

Stage2三种独立运行参数 `--background_policy`：

- `frozen`：背景完全冻结；只更新新slot。
- `composition`：背景Collector、TGEnc、token_norm、AHER以新slot的0.1倍LR更新；只接受组合重建梯度。
- `separation`（默认）：同上，额外单独解码背景canvas。在新GT和高置信旧区域目标为0，在候选背景目标为输入图像，分别求区域均值，后者乘0.1。空区域贡献0，ignore不监督。

Stage2总损失=`L_composition + 0.01 L_seg_new + 0.1 L_background`；非separation模式最后一项为0。
冻结重建decoder的参数，但不阻断其对新/背景canvas的反传。背景head及所有校准参数冻结，旧类和共享模块保持eval模式。
现有`linear_sqrt`按保留slot数归一化，5→9会改变重建组合尺度；保留原语义并记录，不宣称重建输出也严格不变。

固定：spacing0.7/0.7/2、448/384、ROI0.2、20 tokens、128 channels、seed0、small-organ原超参、AdamW7.5e-5/wd0.05、BF16+clip1。
启动器4卡×每卡16×累积3=192。预算每阶段59400更新、2970 warmup，共118800；Stage2重置优化器。
这是**总更新预算匹配**，不等价于每类监督次数或算力完全匹配；不得静默将每阶段都训练118800。

运行前先在对应账号设置其本地绝对路径（脚本不猜跨账号路径）：

```bash
export WORD_ROOT=/该账号/数据/WORD
export PYTHON_BIN=/该账号/环境/bin/python
export MAE_CHECKPOINT=/该账号/AutoPET_MAE/checkpoint-final.pth
export LPIPS_STATE=/该账号/owt_lpips_vgg16.pth
export OUTPUT_DIR=/该账号/结果/Inc44_stage1
bash scripts/orgslot/run_incremental44.sh stage1
# Stage1完成后，手动确认最终59400-update checkpoint，不按测试成绩选取。
export STAGE1_CHECKPOINT=/该账号/结果/Inc44_stage1/checkpoint-最终epoch.pth
export OUTPUT_DIR=/该账号/结果/Inc44_stage2_separation
export BACKGROUND_POLICY=separation
bash scripts/orgslot/run_incremental44.sh stage2
```

提交前仍须commit/push、revision guard、固定运行快照、核验账号路径；脚本本身不提交Slurm，不更新运行中的源目录。
保存resolved_config、manifest哈希、初始化报告、slot元数据、可训练参数清单、Stage1 checkpoint哈希；Stage2保存checkpoint时校验冻结参数哈希。
续训必须保留同一Stage1 teacher及协议，使用`--resume`，不得从全8类WORD checkpoint启动Stage1。

评估要求：同24测试病例，Stage1旧4类、Stage2旧4/新4/全部8类；原始独立binary head与互斥类别合并指标必须分开。旧类阈值在Stage1验证集确定并锁定；新类只在Stage2验证集校准。病例级Dice/P/R/体积比，另外报告旧类raw logits漂移和背景在新器官GT内的残留。**现有Common8 evaluator不能直接用来加载5-slot Stage1模型；增量评估适配与GPU预检尚未完成，不自动挂接旧评估脚本。**
当前仅完成CPU机制测试：标签隐藏/ROI隔离、伪区域冲突与ignore、空区域finite backward、新/背景梯度、旧logits不变、背景策略、随机冻结decoder下的优化下降；这不等于真实数据过拟合或GPU/DDP通过。

SIP提交脚本：`slurm/orgslot/train/incremental44_stage1_sip.sbatch`，4×A800、20CPU、192GB、7天。启动后先以同样每卡16/累积3跑独立的2-update预检，只有正常退出、更新数正确且日志数值有限才启动全新Stage1（每10epoch保存）。预检不是小样本过拟合验证，权重不用于正式训练；排队时不得宣称GPU通过。

Stage2脚本 `slurm/orgslot/train/incremental44_stage2_sip.sbatch` 使用 `afterok:Stage1_JOB`，显式设置`STAGE1_RUN`。启动时解析最后一行训练日志和对应checkpoint，要求59400更新、Stage1身份、optimizer最大step同为59400、参数全有限，再严格加载。不能只看文件存在、猜checkpoint-802或把中间checkpoint当最终模型。Stage2也先独立2-update预检，随后重新加载Stage1正式训练。当前只提交默认背景separation组；frozen/composition保持可选，不自动扩大三倍预算。五slot评估适配仍待完成，没有虚挂评估依赖。

审计边界：背景slot没有独立分割监督，其背景分离约束作用在重建支路；它不等价于9类softmax背景分类。旧raw/calibrated输出不变不代表互斥分类图零遗忘；必须另外评估新增类别冲突。Stage2新slot校准保持初始化scale1/bias0，旧slot校准冻结在Stage1值；新slot使用raw logits与其calibrated logits等价，但两阶段校准参数的可训练性不同，这属于明确的模块冻结方案。Stage2只有masked MSE而无LPIPS，不是完全相同的训练目标；不在未知区域上使用LPIPS以免其感受野引入错误监督。候选背景含teacher漏检风险，日志记录各旧伪区域、四新类监督数量和ignore比例。辅助权重0.1及置信度0.1/0.9尚非验证最优。

## 模型与数据

以运行目录 `resolved_config.json`、`command.txt`、checkpoint args 为准，不以脚本名推断配置。
2D WORD 对照：spacing 0.7×0.7×2.0、input/ROI 448/384、ROI probability 0.2、
token_factor=20、seed=0、有效 batch192、118800 optimizer updates。
ROI20 是采样概率，不是 token 数量。3D 单独记录 slab batch、T、stride 和更新次数。
F 预定 T=4/stride=1；2D microbatch2×4卡×accum24=192 slices，
3D microbatch1×4卡×accum12=48 slabs。相同有效 batch 不保证 legacy_batch TGR 完全相同。

`dimension`、`slot_head_type` 按 ARCHITECTURE.md 选择。
C/D 的 `pixel_pe` 为 none/spatial/temporal/spatiotemporal；2D 不允许 temporal。
F 内部固定 axial sin/cos PE：2D y/x，3D t/y/x；`pixel_pe=none` 不会关闭 F 内部 PE。
F 使用 `query_refinement=none`，不能套用 E 的 cross_attn 开关。
旧 3D 默认 `segmentation_unit=slab`；受控修正版显式 `slice`，2D 行为不变。

## Loss：不能误写成 Dice 重建 + Focal

当前 small-organ 联合路线：L2 reconstruction + LPIPS + 0.01×segmentation loss。
retained slot supervision，lambda_bg_seg=0；不要混同原 v0 DiceBCE 和旧 Focal 实验。
阳性切片：Tversky（FP/FN=0.3/0.7，eps=1e-6）+0.5×Balanced Focal。
Focal 前景和 top2% 背景分别平均，alpha_pos/neg=0.75/0.25、gamma=2。
空切片：0.1×top2% hard-negative BCE，不算 Tversky。
slice-wise 3D 对阳性/空切片分别统计平均后组合，不把整个 slab 当一个阳性样本。
核验 loss_version=L2-LPIPS、lambda_lpips=1、fusion_mode/reference count 与对应基线一致；
不可从其他 Loss3/PSEM 实验继承额外权重。

## 初始化与数值：新训练和历史续训分开

MAE encoder-only 只载共享 encoder；encoder_decoder 还载共享重建 decoder，
不加载 pixel decoder、Collector/TGEnc/AHER。必须保存迁移 key 报告及来源权重 hash。
MAE 初始化不等于 resume；resume 要恢复 epoch、optimizer/scaler 和更新预算。

新 A800 实验显式配置 AMP_DTYPE=bf16、CLIP_GRAD=1.0、FINITE_CHECK_INTERVAL=1。
默认入口仍可能回退 FP16，不能只凭模型支持 BF16 判断启用；以解析配置和日志为准。
历史 Arm E MAE 三臂/续训保留原 FP16 GradScaler、不新增裁剪，以保持原协议；
不得在修复 NCCL 时默默切换 loss/AMP/采样。
BF16 不能防止非法运算，梯度裁剪不能修复 NaN；老 PyTorch 的插值/depthwise3D
需要局部 FP32 fallback。F attention/FFN/readout 局部 FP32，不要 model.bfloat16()。

NCCL ALLREDUCE 超时不是 NaN 的同义词。E encdec 恢复采用固定 schema/顺序的指标归约，
并仅对确认为冻结 LPIPS 常量的 buffers 允许关闭逐 forward broadcast；梯度同步仍启用。
这属于缓解措施，不能宣称已定位原始失联 rank 的根因。旧 checkpoint 未保存 RNG/worker
状态，恢复 optimizer 不等于逐位重演原训练轨迹。

## 三账号、两集群与版本

- Linux 用户、Slurm account、QoS 是不同字段；不得替换用户名猜数据绝对路径。
- SIP 与 XEC 分别查 Slurm。Python、CSV、ROI index、LPIPS、ckpt、输出目录逐项核验。
- 新提交前 fetch origin，核验开发分支与实际运行树；执行
  `bash tools/check_experiment_revision.sh RUNTIME_WORKTREE`。
- 修复先 commit/push；源快照不可变，记录完整 commit/hash、账号、脚本、资源、输出。
- 不更新运行中/排队任务引用的源目录。旧任务失败后须重建评估 afterok，不会自动转依赖。
- 用户要求按允许最高时限申请，但上限必须提交时查实际 partition/QoS，不把旧天数当常量。
- CPU 测试与 GPU 验证分别报告；当前用户要求的提交流程优先，不额外擅自排 smoke。

## 评估

WORD070：24病例/6990切片；保存 evaluator 的 presence/all-cases 等聚合定义，
2D 模型也可堆叠成病例体积评估，不能仅凭训练维度命名指标。
head fixed0.5 为主结果；train-calibrated 单列并明确校准来源；新阈值优先用验证集选。
reconstruction Direct-post 与 head 分开排行；不得测试集逐器官选输出拼最优。
旧 recon 固定阈值0.02、min component20/opening1，仍需核验具体协议。
3D 记录 slab overlap fusion、volume ratio、precision/recall、per-z，以及 head/recon 路径。
checkpoint 必须 strict/exact 加载；评估完成标记和病例完整性核验后才入结果表。
# 后续提交 batch 约定（2026-09-13）

Arm F 2D MAE两臂：`slurm/orgslot/train/arm_f_mae.sbatch`，MAE_INIT_SCOPE为
encoder或encoder_decoder；复用AutoPET checkpoint-final，初始化不加载optimizer。
保留topk背景loss、BF16、clip1、ROI20、0.7/0.7/2、118800更新、seed0。
每卡16/累积3/有效192；运行中无MAE F每卡8，需报告micro-batch差异，不称严格单变量。
只迁移ViT以及可选重建decoder，不迁移F pixel/attention/slot模块。
统一评估`slurm/orgslot/eval/arm_f_mae_unified.sbatch`：ckpt802，head固定0.5及
训练集6000样本阈值校准、完整测试集head与reconstruction固定0.02。

用户指定：后续四卡2D训练每卡batch16、累积3、有效batch192；
四卡3D训练每卡batch6、累积2、有效batch48 slabs。Arm F共享环境入口已实现。
其他实验专用脚本新提交前也须按此约定核对，不能沿用旧脚本的batch覆盖值。
已运行或已归档任务保持原快照；资源不足时先报告，不静默改变有效batch。
改变micro-batch可能改变阳性/阴性分组loss归约权重，不能称为严格梯度等价。
## Collector attention alignment

2026-09-21：以 **2D Arm E scratch** 为对照，新增轻量训练辅助损失，不新增参数、不改变推理与旧checkpoint结构。`--lambda_collector_attention` 默认0；本次探索值0.01（不是已优化超参）。只实现outside loss，不加coverage、diversity、gate或其他attention模块。

- `A:[B,K,N]` 为现有Collector空间softmax；`M` 是增强/ROI后当前GT按真实patch格子max-pool得到的器官相交支持区。loss=`mean_valid_pairs(mean_tokens(sum_positions(A*(1-M))))`。
- 只使用visible_masks中有标注、当前裁剪有前景、segmentation_keep和slot_compute_mask均为真的器官。排除background、空切片、缺标注和未计算slot。当前只支持2D joint训练，3D/head-only显式拒绝。
- DDP每microstep对有效样本—器官数做全局归一化，局部可导分子乘world_size抵消DDP梯度平均。固定形状统计collective和固定日志键；无阳性rank保持零梯度路径。梯度累积平均的是microstep的全局均值，未改原采样/累积策略。
- FP32计算辅助loss；记录raw/weighted loss、每器官count/foreground_mass/support_fraction/occupancy_mass。max-pool支持区不等于纯器官patch；注意力集中不等于解耦证明，也不保证token分工。
- 模型保持arm_e_multiscale_query、20 tokens、TGEnc depth1、channels128；WORD07072、448/384、ROI概率0.2、seed0；scratch，无resume/MAE；small-organ参数全部不变、lambda_seg0.01；BF16、clip1、实际LR7.5e-5、warmup5940、118800 updates；4卡×batch16×accum3=192。
- 对照为当前E scratch系列（SIP续训2965273；统一评估2965274）。历史对照中途由较小microbatch切到16，且发生过恢复；新实验从头batch16，不能宣称随机轨迹逐步一致。不要与E MAE初始化结果混为严格对照。后续若需严格归因，需匹配从头batch16的基线。
- 评估复用`slurm/orgslot/eval/arm_e_mae_unified.sbatch`：训练集子集阈值校准、24病例6990切片head fixed/calibrated、原有recon阈值0.02及后处理；不在测试集选阈值。此脚本名称含MAE，但不要求MAE权重。
- 启动脚本：`slurm/orgslot/train/arm_e_collector_align.sbatch`。继承按账号显式映射的数据/环境路径，不复用另一账号的目录；冻结最新origin/feature/orgslot源快照。
- CPU核验：6项新增测试全部通过，包括实际Arm E前向不变/strict checkpoint加载、Collector梯度、小型定位拟合、两进程Gloo不均衡与空rank。另有23项回归通过；旧`test_small_organ_empty_slice_uses_topk_bce_and_weight`在未修改origin源码上同样失败（测试未显式传topk，而当前API默认mean）。本实验显式topk，不修改历史loss语义。GPU正式运行尚待调度验证。
# WORD official96 protocol

2026-09-30，独立于历史随机96/24协议，命名 `official_pool_96_26_24_seed42`。
清单固定在 `configs/orgslot/word_official96_seed42.json`，按排序后的官方病例ID，分别以独立的 `random.Random(42).sample` 抽取96训练与24测试。训练池官方100例，剩4例不用；验证/测试候选池为官方20验证＋30测试，剩26例用于验证和完整阈值校准。只抽一次，不根据badcase或最终分数重抽。这不是官方100/20/30评估，也未复现PCDD划分。

以旧划分85.358%的 E cross-attention＋AutoPET MAE encoder-only为配置来源。只加载原AutoPET MAE encoder，不加载任何旧WORD权重。2D、spacing0.7/0.7/2、448输入/384ROI、ROI概率0.2、20tokens、128channels、small-organ loss、lambda_seg0.01、retained监督、top-k background、legacy_batch TGR、seed0、AdamW lr7.5e-5/wd0.05、BF16、clip1均保持。4卡×每卡16×累积3＝192，118800 optimizer updates、5940 warmup updates；最终epoch由新切片数推导，不再硬编码802。

准备脚本 `slurm/orgslot/train/arm_e_official96_prepare.sbatch` 仅对原未处理的官方30测试病例按相同流程预处理。训练/验证原病例重用无学习的既有逐病例JPEG/PNG（新目录软链接），生成新的完整CSV、summary和ROI索引，绝不覆盖旧数据。严格检查146病例互斥、来源、完整切片、ROI manifest hash。`NEW_DATA_ROOT`必须是不存在的新目录。准备CPU任务成功后才释放训练。

训练 `slurm/orgslot/train/arm_e_crossattn_official96.sbatch`；评估 `slurm/orgslot/eval/arm_e_official96.sbatch`。使用 `EXPERIMENT_WORKDIR`固定源码、`NEW_DATA_ROOT`固定数据，评估还需 `TRAIN_JOB_ID`。训练完成后按日志中118800 updates取最终checkpoint，非测试集选优。26例验证完整阈值扫描0.1至0.9，测试24例同时报告固定0.5与验证校准结果；min_size20/opening1不变。reconstruction仍用0.02固定阈值与现有OWT路径。保存新的训练—评估afterok链；新旧结果分别汇报，不把新划分成绩变动归因为架构改进。
# Stage2 shared spatial/readout ablation (2026-10-04)

`--stage2_shared_segmentation train` (launcher environment
`STAGE2_SHARED_SEGMENTATION=train`) unfreezes the existing entire
`pixel_query_decoder`: image spatial stem, pixel projection/fusion blocks,
query normalization/projection and P4 query cross-attention/FFN. No parameters
are added. Default `frozen` retains the previous experiment behavior.

Start fresh from the same completed Stage1, NOT the completed Stage2. ViT,
old slots and reconstruction decoder remain frozen. New slots and the existing
background separation policy remain trainable. Use unchanged 59400 updates,
4 GPUs x batch16 x accumulation3 = 192, LR7.5e-5, lambda_seg0.01 and seed0.
The old slot weights remain invariant but old segmentation outputs need not:
shared spatial/readout updates can cause forgetting. No new distillation loss
is added in this controlled ablation. Resume rejects a changed shared policy.

Set `EVALUATE_RECONSTRUCTION=1` for the Stage2 evaluation job: validation-only
head threshold selection, fixed/calibrated test heads, plus original direct and
indirect reconstruction Dice at threshold0.02, min_size20/opening_radius1.
Use the same official-pool 24 test cases; never select thresholds on test.
