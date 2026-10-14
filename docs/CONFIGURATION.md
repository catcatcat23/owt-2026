# 统一配置与部署契约

## SAM组合aux续训：2026-10-14

157059在epoch480/step320后因NCCL ALLREDUCE超时1800秒失败，序号7003207；
rank1/2/3均等待同一操作，Python堆栈位于DDP前向buffer广播。没有更早异常证实
NaN、指标schema失配或具体I/O/硬件根因；前一天间歇性停顿属于相关观察而非因果证明。
checkpoint470模型/优化器CPU有限值核验通过，optimizer step69708，恢复epoch471。
入口 `slurm/orgslot/train/sam_combo_aux_resume.sbatch` 保持batch16×4×accum3、
118800总更新、MAE encoder来源、组合loss、seed及采样不变，恢复optimizer/scaler。
仅启用已有 `no_ddp_buffer_broadcast`：训练器只允许LPIPS常量shift/scale buffers，
发现其它buffer即拒绝。梯度同步保持；这是针对报错路径的缓解，不宣称底层根因已修复。
采用最新origin/feature/orgslot的新固定runtime，旧157061依赖必须替换。
本轮不另跑GPU smoke；CPU加载/语法验证与正式GPU运行状态分别报告。

新链163200→163201（sifansong/XEC）：训练4A800/4gpus/7天，评估1RTX4090/1天。
运行树sam_combo_resume_1f2b0e1，SHA1f2b0e169b7d454d355ba331a8a308db65ab92ed。
提交后训练Priority、评估afterok:163200；旧失效157061取消，原checkpoint保留。

## 2026-10-07：缩小GT与soft prior的组合对照

基线是旧24病例协议的SAM-tail＋MAE encoder＋`downsample_gt`（不是增量实验）。
两组保持WORD07072、448/ROI384、ROI概率0.2、seed0、每卡16×4卡×累积3=192、
118800 updates、warmup5940、small-organ loss和lambda_seg0.01不变。

| 配置 | 多尺度读取前soft prior | tail前coarse监督 | prior辅助监督 |
|---|---|---|---|
| `downsample_gt_soft_prior` | 有，detach的log(0.2+0.8sigmoid)偏置 | nearest缩小GT | 无 |
| `downsample_gt_soft_prior_aux` | 同左 | 同左 | 三层logits上采样后对全分辨率GT监督，平均后权重0.25 |

两组均为 `Lrec + LPIPS + .01*(w*Lfinal + (1-w)*Lcoarse + .25*Lprior)`，
第一组Lprior=0；w=0.6^(0.990^epoch)，沿用缩小GT基线调度。
注意最后一层读取前P4 prior与读取后tail coarse是不同张量。
tail coarse不detach，经sigmoid作为tail的V gate；三层prior路由detach，
第二组prior预测自身仍接受辅助loss梯度。没有新增参数，不改变旧head行为。

启动入口：`slurm/orgslot/train/arm_f_sam_tail_hsam.sbatch`，设置上述HSAM_SUPERVISION；
仍自动加载AutoPET MAE encoder；按用户要求直接正式训练，不运行GPU smoke/preflight。
CPU回归与数据/权重身份核验已完成，不代表GPU验证已通过。
评估入口：`slurm/orgslot/eval/sam_tail_guided.sbatch`，checkpoint802，
训练前6000切片校准＋原24测试病例固定/校准head＋recon；与历史基线口径一致，
校准不是独立验证集，须披露。不要混入official96的新划分排名。

CPU测试：`PYTHONPATH=.:tests python -m unittest test_hsam_combined test_hsam_supervision test_sam_soft_prior test_distributed_metrics`。
覆盖旧参数strict加载与旧模式输出不变、组合前向相同、prior路由detach/aux梯度、
background-only rank指标schema、有限backward和双rank Gloo；GPU通过与否另记。

## Soft-prior aux 的DDP指标契约

2026-10-01：`150613`失败并非已确认的NaN。TGR在某rank只保留背景时，coarse预测为空是合法情况；旧实现提前返回0辅助loss，却没有生成`soft_prior_p16_loss/p8_loss/p4_loss`，其他rank有这三个字段，`reduce_metrics`的schema检查因此终止。修复在提前返回前初始化三个0值字段，非空分支仍写实际loss；不改loss权重、mask机制或空监督语义，也不删除schema保护。回归覆盖双rank不同保留情况、有限梯度、真实字段不一致仍报错。

恢复脚本`slurm/orgslot/train/sam_tail_soft_prior_aux_resume.sbatch`只恢复已核验的checkpoint-0，从epoch1续训并恢复optimizer/scaler。启动后先运行两GPU NCCL指标回归；失败则不进入正式训练。新训练需新`afterok`评估；旧`150614`依赖失败不会自行恢复。

## incremental44：当前契约

完整冻结矩阵、标签可见性、预算沿革及结果见 [增量专项](INCREMENTAL_READOUT_ABLATION.md)。
当前新实验每阶段118800更新、5940 warmup，共237600；旧59400/阶段记录保留于
[整理前配置](archive/HISTORY.md#merge-configuration)，不能覆盖旧运行快照或混算收益。
Stage1最终checkpoint按更新数和日志核验，不只凭文件名或存在性；
Stage2从同一个已完成Stage1初始化，重置优化器，不能从已有Stage2开始新的消融。
入口：`main_pretrain_orgslot_incremental44.py`、`scripts/orgslot/run_incremental44.sh`。
Stage1/Stage2及五slot评估已实际运行，不再沿用“评估尚未实现”的旧通知。

背景separation：新GT优先；Stage1 teacher恰好一个旧类≥0.9为旧伪区域，
全部旧类<0.1且非新GT为候选背景，其他ignore。背景head未监督，不作为teacher。
背景Collector/TGEnc/token_norm/AHER以0.1倍LR更新；新/旧前景区域目标0，
候选背景目标CT（该区域项权重0.1）。背景分离总权重0.1。
Stage2基础目标为masked composition MSE + .01新类分割 + .1背景分离；
无Stage1的LPIPS，SAM辅助项按已保存配置叠加。不要改写为两阶段同一loss。
共享重建decoder冻结但可对输入反传；背景head/calibration冻结。
同96训练病例可重复访问，只开放当前阶段标签，不声称禁止旧图像访问。
旧binary分数、旧raw logits、互斥argmax分别审计；teacher误差与背景残留仍需诊断。

## 模型与数据

以运行目录 `resolved_config.json`、`command.txt`、checkpoint args 为准，不以脚本名推断配置。
2D WORD 对照：spacing 0.7×0.7×2.0、input/ROI 448/384、ROI probability 0.2、
token_factor=20、seed=0、有效 batch192、118800 optimizer updates。
ROI20 是采样概率，不是 token 数量。3D 单独记录 slab batch、T、stride 和更新次数。
新四卡配方：2D16/卡×累积3=192 slices；3D6/卡×累积2=48 slabs，T=4/stride=1。
早期2D2/卡、3D1/卡等属于历史快照；相同有效batch不保证legacy_batch TGR完全相同。

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
background separation policy remain trainable. Historical experiment used 59400 updates (not the newer full-budget Stage2),
4 GPUs x batch16 x accumulation3 = 192, LR7.5e-5, lambda_seg0.01 and seed0.
The old slot weights remain invariant but old segmentation outputs need not:
shared spatial/readout updates can cause forgetting. No new distillation loss
is added in this controlled ablation. Resume rejects a changed shared policy.

Set `EVALUATE_RECONSTRUCTION=1` for the Stage2 evaluation job: validation-only
head threshold selection, fixed/calibrated test heads, plus original direct and
indirect reconstruction Dice at threshold0.02, min_size20/opening_radius1.
Use the same official-pool 24 test cases; never select thresholds on test.
# Offline split robustness: seeds0/1/42 (2026-10-06)

Only dataset split seed changes. Each samples96 official training cases and24
test cases from official Val+Ts; remaining26 holdout cases are validation.
Plans are configs/orgslot/word_official96_seed{0,1,42}.json, generated before
training without filtering bad cases. Model seed remains0. No WORD checkpoint
transfer between splits: same AutoPET MAE encoder initialization,118800 updates,
batch16 x4GPU xaccum3, ROI20, losses/LR unchanged.

`tools.relink_word_official_split` reuses existing preprocessed pixels, rebuilds
manifests and training ROI index, and validates disjoint cases and all files.
Set SPLIT_PLAN and NEW_DATA_ROOT for both existing official96 train/eval scripts.
SPLIT_GPU_PREFLIGHT=1 runs2 fresh DDP updates separately, then discards those
weights and starts formal training from MAE. Evaluation:26-case validation-only
threshold calibration,24-case test head fixed/calibrated and recon direct/indirect.
Report all three splits, not just the best. Test sets overlap, so these are
split-sensitivity runs, not three independent test cohorts or CV folds.
# Stage2 SAM 私有交互叠加读出 adapter（可选，尚无训练结果）

保持 `sam_soft_aux` / `slot_private` 不变，新增
`--stage2_readout_adapter residual16`，默认 `none` 完全保留旧行为。
在共享 SAM-tail 的最后 mask token 上计算：
`w = frozen_mask_mlp(m) + stage_adapter(m)`，再与 tail 更新后的 pixels 点积。
adapter 为 `Linear(128,16) -> GELU -> Linear(16,128)`，4240参数，
四个新器官共享一套，只有食管、胰腺、肝、胃启用；旧四类及背景绕过。
最后一层权重/偏置零初始化；不改变空间分支、soft prior、aux、重建和采样。
共享tail参数冻结但保留输入梯度；原新slots、背景策略及私有交互照常训练。

已配置本账号数据/环境和同一Stage1 checkpoint后，可使用：

```bash
INCREMENTAL_ARCHITECTURE=sam_soft_aux \
STAGE2_SHARED_SEGMENTATION=slot_private \
STAGE2_READOUT_ADAPTER=residual16 \
bash scripts/orgslot/run_incremental44.sh stage2
```

基线将 `STAGE2_READOUT_ADAPTER` 设为 `none`，其余保持相同。
仅支持Stage2、2D SAM `soft_prior_aux`、`slot_private`；E/Stage1拒绝开启。
resume核验开关一致，不能将旧Stage2续训改成新实验；正式对照从同一Stage1开始。
统一head/reconstruction evaluator依据checkpoint参数安装adapter并strict加载。
旧checkpoint缺少此配置时按`none`处理。提交前仍须commit/push和不可变runtime检查；
不得修改当前排队/运行的 `inc44_full_4656f1e` 源码。

## 3D SAM-tail 两条主线

复用现有3D Arm F空间路径：3D ViT输出按时间网格切片，浅层CNN及
P16/P8/P4融合逐slice计算；随后展平T/H/W，使用t/y/x位置编码进行跨slab
token交互及SAM-tail双向读取。不是新增3D卷积decoder，不改变Collector。
输入4-slice、temp_stride=1；448输入时P4为[ B,128,4,112,112 ]。

| 配置 | 先验位置 | 辅助监督 | 分割部分损失（lambda_seg=0.01） |
|---|---|---|---|
| `soft_prior_aux` | 每次P16/P8/P4读取前，log(0.2+0.8 sigmoid(coarse.detach()))加到attention logits；tail不加粗mask门控 | 三个prior logits FP32上采样至原GT，逐slice small-organ loss | .01*(Lfinal+.25*mean(L16,L8,L4)) |
| `downsample_gt` | 多尺度读取后生成P4粗mask，sigmoid概率门控tail两次token→pixel的V；保留梯度 | GT最近邻仅缩小H/W，T保持4；粗/最终均逐slice small-organ loss | .01*(w*Lfinal+(1-w)*Lcoarse)，w=.6**(.990**epoch) |

两阶段指同一次forward内的粗预测→tail细预测，不是分两次训练。
重建及LPIPS保持不变。lost_positive_slices在3D按每个slice计数，统计缩小GT
丢掉的阳性切片。拒绝temporal stride不为1及粗预测T不匹配；不混入3D
M0-M3硬mask、P2或soft+两阶段组合实验。

入口 `slurm/orgslot/train/arm_f_sam_tail_3d.sbatch`，分别指定：

```bash
export HSAM_SUPERVISION=soft_prior_aux  # 另一组为 downsample_gt
# 在已分配4卡的环境中，并配置好EXPERIMENT_WORKDIR、ARM_F_TARGET、ARM_F_COMMIT：
bash slurm/orgslot/train/arm_f_sam_tail_3d.sbatch
```

复用3D配方：每卡6×4卡×累积2=48 slabs（192 slices），118800 updates、
ROI20、spacing .7/.7/2、BF16、clip1、small-organ loss不变。无MAE路径时
为scratch；要做MAE对照必须显式给两组同一兼容3D `MAE_INIT_CHECKPOINT`，
默认仅加载encoder，不隐式使用历史2D权重。

3D checkpoint loader恢复hsam_supervision后严格加载；统一3D head/calibration/
recon评估共用这个loader，不允许权重能加载但推理静默遗漏先验。
CPU21项测试通过：全模型FP32有限反传、两种模式BF16 decoder反传、
mixed slab/empty监督、lost-positive计数、checkpoint严格重载及2D回归。
旧PyTorch的CPU BF16 GroupNorm不支持混合dtype，全模型CPU BF16未通过，
没有为此更改现有空间分支。这是实现时CPU限制；后续任务已提交，见STATUS。
该CPU记录不能替代当前GPU验证或最终评估。

## 合并专项：E cross / SAM读出与监督

### E cross与MAE

`slot_head_type=arm_e_multiscale_query`，`query_refinement=cross_attn`：
20 tokens → LayerNorm/mean/projection/identity → query读取P4+FFN → query·原P4。
无F多尺度20-token refinement，无reverse。仅加载MAE encoder，decoder随机。
启动 `slurm/orgslot/train/arm_e_crossattn_mae_encoder.sbatch`；
统一评估 `slurm/orgslot/eval/arm_e_mae_unified.sbatch`。
历史E no-cross的AMP/microbatch/恢复轨迹有差异，不是严格共享随机轨迹对照。

### SAM soft_prior与soft_prior_aux

每次读取P16/P8/P4前，以当前tokens均值与该尺度pixels归一化点积乘sqrt(C)，
形成粗logits；`log(.2+.8*sigmoid(logits.detach()))`加到token→pixel attention logits。
20 tokens与各heads共享器官先验；只改变读取偏置，不是二值hard mask。
soft_prior模式无独立粗监督，aux模式三尺度logits上采样到GT后用原small-organ loss，
`L=Lrec+LPIPS+.01*(Lfinal+.25*mean(L16,L8,L4))`。
路由detach，但aux logits不detach。tail不接收此模式的粗mask。
入口 `slurm/orgslot/train/sam_tail_soft_prior.sbatch`；
`HSAM_SUPERVISION=soft_prior|soft_prior_aux`。
.25是本项目的适配值，不是Mask2Former或H-SAM的官方默认。
历史150613失败是DDP日志schema不一致，不是已确认NaN；修复不能被写成架构性能改进。

### SAM两阶段mask与M0–M3

`downsample_gt`：多尺度refinement后，用tokens与原P4生成coarse logits；
sigmoid以不detach、无floor、不归一化的V gate引导tail两次token→pixel读取。
P4粗logits对nearest缩小GT监督；`upsample_logits`则放大粗logits对全GT监督。
`w=.6**(.990**epoch)`，
`Lseg=.01*((1-w)*Lcoarse+w*Lfinal)`。
该调度取自H-SAM公开训练器；本实现不是完整H-SAM（无CMAttn或预训练SAM decoder）。
评估只用final mask，不把coarse/final集成。监控lost_positive_slices。

`m2f_hard`：共享LayerNorm/3层mask MLP，均值tokens对共享P4预测M0；
读取P16/P8/P4后产生M1/M2/M3。前一mask缩放到对应尺度，sigmoid>.5形成detach路由，
全屏蔽行退回全局attention。`Lseg=.01*(Lfinal+LM0+LM1+LM2+LM3)`。
tail无soft prior；同时改变总监督强度，不能把收益仅归于hard routing。
没有Hungarian matching，不是Mask2Former完整复刻。
入口 `slurm/orgslot/train/arm_f_sam_tail_hsam.sbatch`，
评估 `slurm/orgslot/eval/sam_tail_guided.sbatch`，必须恢复checkpoint保存的mode。
原公式、来源链接、CPU测试和提交SHA见 [原文](archive/HISTORY.md#merge-sam-tail-hsam)。

### 全八类监督

`slurm/orgslot/train/sam_tail_mae_all_seg.sbatch`将seg_supervision从retained改all，
背景权重仍0；包括八前景的empty-slice监督。
重建仍按随机keep及ROI focus保留，slot实际计算用keep OR head_mask；
被重建丢弃的器官仍提供真实tokens。共享权重同时接收两分支梯度，不是梯度隔离。
只用于全标注Offline，不可越过增量标签可见性。对应成绩见RESULTS。

## 诊断复现：P4探针与token干预


## E badcase extension (2026-09-30)

Supports E fused P4 via a separate capture path; original F path unchanged.
E uses one linear probe (no reverse attention exists), compared with the SAME
checkpoint's original query head. Train organ4 and organ6 separately, fixed10
epochs, seed42, batch16, AdamW1e-3, existing BCE+positive Dice probe objective.
Original model fully frozen. Keep76/20 training-case split; evaluate selected
test cases0127/0114 ONLY after epoch9, no test selection/tuning. Test and training
IDs must be disjoint. Save all test slice probability maps and case/slice counts.
Fixed0.5 RAW results; do not compare against historical postprocessed/calibrated
numbers. These probes have extra supervised fitting and a different objective;
superiority supports readability, not proof that query is the sole bottleneck.
Script: slurm/orgslot/eval/e_badcase_probe.sbatch. Same E checkpoint as token
interventions; do NOT reuse existing F probe weights on E features.

Independent script; no changes to original model. Supports original 2D
arm_f_attention only. Each run selects one organ (6 pancreas, 5 esophagus,
4 gallbladder). Compare raw P4 immediately before reverse attention against
the organ-conditioned P4 after reverse residual and FFN, before classifier.
Hooks must fire exactly once. Both linear heads start with identical weights.
Backbone is eval/no_grad and receives no gradients. Original head predictions
are reported alongside probes, fixed0.5 raw without postprocessing.

Run from repository root:

```bash
PYTHONPATH=. python tools/frozen_p4_probe.py --checkpoint /absolute/checkpoint-802.pth --train-csv /absolute/WORD_Training_2D_native07072.csv --class-id 6 --output /absolute/new_probe_output --batch-size 16
```

Requires current project's torch environment. Defaults: single GPU, FP32,
10 epochs, AdamW lr1e-3, center448, no augmentation/ROI, BCE plus positive
Dice with equal positive/negative slice-group weighting. This is a diagnostic
probe objective, NOT the original small-organ training loss. Paired probes
receive the exact same batches and update count. Lower batch if 4090 OOM;
do not compare separate runs using different batches as controlled ablations.

Original96 training cases are deterministically split 76/20, seed42. Case IDs,
checkpoint and manifest hashes, load report, settings are saved. The backbone
previously saw ALL96; holdout is only unseen to the new probes, not independent
end-to-end validation. Never choose settings using the24 test cases. No early
stopping or best-checkpoint selection. Output includes per-case Dice/P/R/counts
and volume ratios for every epoch and final probe weights. Training metrics are
online across changing head weights, explicitly flagged; not a final fixed-head
training-set evaluation. Holdout metrics use fixed weights.

Interpretation: post worse than pre suggests reduced LINEAR readability, not
proof of information destruction. Original classifier includes LayerNorm,
unlike these linear probes; do not attribute all baseline differences to query.
Pre/post feature scales may affect optimization. This is the original implementation boundary, not current queue state. Final numerical probe results remain to be verified in RESULTS.

### Token干预


Inference only, no optimizer or weight update. Existing 96 training cases supply
tokens after TGEnc, before the unchanged query LayerNorm/projection. Full training
manifest is traversed once. Cache per-organ positive/empty sums and reservoir128
tokens per stratum; record donor case/index. Means are slice-weighted (not
case-weighted); reservoir sampling can underrepresent rare donor cases.

Evaluate all24 test cases/6990 slices: original, all-training mean, presence-
matched mean, three independently sampled same-organ presence-matched donors.
Donors come only from train; select cached case uniformly then cached slice.
GT-presence matching is diagnostic oracle information, NOT deployable inference.
No volume or position matching; interpret replacements as interventions, not a
proof of absence of information. Mean raw tokens may be distribution-shifted;
retain the original LayerNorm and report mean versus donor controls separately.

Compute original spatial features once per batch, reuse exactly for each organ
readout; preserve slot identity and calibration. Assert original token re-read
reproduces original logits. Independent organ readouts are intervened separately;
there is no cross-class competition or joint argmax in this audit.

FP32, batch2, fixed0.5 raw/post(min_size20,opening1), center448, no augmentation,
no threshold tuning. Do not directly compare to historical calibrated Dice.
Full test set is exploratory, already examined previously; not a blind holdout.
No PCA/heatmap is necessary for the initial causal-use diagnostic.

Outputs: provenance/checkpoint+manifest hashes, token_bank.pt, donor assignments,
per-case Dice/P/R/volume/empty/zero-overlap/logitMAE, per-slice counts, progress,
eight-organ summary for each condition. Donor repeats are diagnostic variation,
not independent training seeds. No model architecture changes.

## Presence ablation (2026-09-30)

Use --reuse-bank PREVIOUS_OUTPUT --presence-ablation. Verify checkpoint and
train/test manifest hashes before reusing the bank. Compare original,
positive_only, negative_only, fixed_positive (no GT selection), mean_matched.
Only the selected presence stratum is replaced; the other retains original
tokens. Save maximum-GT-slice CT input, GT and probability maps as NPZ for
gallbladder 0041/0125/0127/0145 and pancreas 0114. No retraining or threshold fit.
Original and mean_matched must reproduce the first run before interpretation.

Initial postprocessed audit: matched means reduce positive-voxel recall for
gallbladder 74.15 to 62.53%, esophagus 82.06 to 78.34%, pancreas 83.71 to
79.98%, while reducing negative-slice FP counts. These are pooled voxel recall,
not case-mean recall. Gallbladder 0127 remains zero overlap. This does not
support claiming the higher overall Dice rescued missed organs.

Interpretation: true tokens outperforming donors suggests useful case-conditioned
readout. Little change suggests limited use by this readout, not necessarily no
information in tokens. Fixed identity conditioning may still support modularity.

## 统一3D评估


Entry: `tools/eval_common8_orgslot_unified_3d.py`; launcher:
`slurm/orgslot/eval/unified_3d_protocol.sbatch`.

The model still receives four-slice slabs. Overlapping scores are averaged at
each original slice **before** thresholding. Each original slice is scored once.
No retraining or architecture change is involved.

| Component | Shared with current 2D evaluation |
| --- | --- |
| Input | WORD 0.7×0.7×2.0, fixed-255 normalization, deterministic 448 crop |
| Calibration | First 6000 training manifest rows, disjoint test cases |
| Threshold selection | Per-class 0.1–0.9 sweep; maximize postprocessed case Dice on calibration only |
| Head test | Fixed 0.5 and training-calibrated thresholds, raw and post |
| Direct recon | Keep only selected slot; original fusion and reconstruction decoder; channel mean >0.02 |
| Indirect recon | Keep all except selected slot; clamp(input−reconstruction,0) before channel mean >0.02 |
| Postprocessing | Shared 2D function: remove components <20 pixels, radius-1 opening per slice |
| Aggregation | Shared 2D counters and summarize; per-case volume Dice and foreground macro mean |

Calibration reads complete slabs from the relevant training cases for context,
but only the selected first 6000 original slices contribute to calibration.
Threshold files bind the checkpoint and test-manifest SHA256. Test sweeps are
not performed. Reconstruction uses its fixed historical threshold, not head thresholds.

Submit calibration and reconstruction independently; head must depend on successful
calibration. Use immutable source and the repository revision gate before submission.
Outputs include resolved configuration, exact checkpoint-load report, scoring keys,
per-case records, metrics and completion status.

**Do not compare old 3D-post results as if they used this protocol:** the previous
3D evaluator removes 3D connected components and does not use the 2D opening.
Slab context/fusion remain intrinsically 3D; matching evaluation protocol does not
mean making 3D inference identical to single-slice inference.

## CNN编码器与AutoPET迁移


Status: implementation validated on CPU; submission details recorded separately;
no accuracy measured for these CNN configurations yet.
Validation: 5 CNN CPU unit tests (including both CNNs with both SAM-tail
supervision modes) and 5 original E cross tests pass. Both
encoders return finite [1,784,768] tokens on a 448x448 CPU forward. Encoder
parameters including projection: ResEnc 35,844,992; MedNeXt-B 7,215,008.
These counts exclude Collector, pixel decoder, and reconstruction modules.
2D E cross and SAM-tail are supported. Existing ViT is the default and its state keys
are unchanged. The abandoned ResNet-18 prototype is not supported.

## Exact structure

| Stage | Resolution (448 input) | Channels | nnU-Net ResEnc blocks | MedNeXt-B blocks / expansion |
|---|---|---|---|---|
| S1 | 448 x 448 | 32 | 1 | 2 / 2 |
| S2 | 224 x 224 | 64 | 3 | 2 / 3 |
| S4 | 112 x 112 | 128 | 4 | 2 / 4 |
| S8 | 56 x 56 | 256 | 6 | 2 / 4 |
| S16 | 28 x 28 | 512 | 6 | 2 / 4 |

Both export S16 through a 1x1 projection (512 -> 768 for the base model),
flatten to [B,784,768], add the existing spatial positional embedding, and use
the existing encoder LayerNorm. This tensor feeds the original Collector and
the E pixel decoder. Since 2026-10-14, CNN S4 (128 channels) and S8 (256 channels)
directly supply lateral skips, replacing the separate image spatial stem.
1x1 projections map both to mask channels (128 in production). Query cross-attention,
AHER, reconstruction decoder, token count and losses are unchanged. ViT blocks and its
unused CLS parameter are removed only in CNN models.

### nnunet_resenc_s16

nnU-Net ResEnc-style 2D encoder using the first five planner stage depths
(1,3,4,6,6), channels (32,64,128,256,512), strides (1,2,2,2,2).
Stem: 3x3 convolution + affine InstanceNorm + LeakyReLU(0.01).
Residual main path: 3x3 conv -> InstanceNorm -> LeakyReLU -> 3x3 conv ->
InstanceNorm; add skip and apply LeakyReLU. Downsampling skip uses average
pooling then channel projection plus InstanceNorm (BasicBlockD topology).
Convolutions use He initialization; final residual normalization starts at zero.

This is a self-contained encoder adaptation, not a full nnU-Net pipeline,
not an automatically planned WORD network, and not a claim to reproduce
ResEnc-L scores. Depth is fixed at output stride16 for interface compatibility.

### mednext_b_s16

MedNeXt-v1 B encoder and bottleneck in 2D: base channels32, kernel3,
block counts (2,2,2,2,2), expansion (2,3,4,4,4), residual blocks and residual
downsampling enabled, GRN disabled. Stem is 1x1 convolution.
Block: depthwise 3x3 -> per-channel GroupNorm -> 1x1 expansion -> GELU ->
1x1 compression -> residual addition. Between stages, a separate stride2
depthwise block changes resolution/channels, with a stride2 1x1 residual.
Stage block counts exclude these four downsampling blocks.

This is an encoder-only implementation of the official v1 B topology using
its supported 2D operations, not MedNeXt-L, MedNeXt-v2 or the full U-shaped model.
No official checkpoint key compatibility is claimed.

## Configuration and fair comparison

Use the existing E cross training recipe with exactly one of:

```bash
export BACKBONE=nnunet_resenc_s16  # or mednext_b_s16; default vit
export SLOT_HEAD_TYPE=arm_e_multiscale_query QUERY_REFINEMENT=cross_attn
export DIMENSION=2D MAE_INIT_CHECKPOINT=
```

The shared launcher forwards BACKBONE to --backbone. Checkpoint args retain
the selection; unified 2D head/reconstruction evaluation rebuilds it, defaulting
to vit for old checkpoints. Resume uses strict state loading. CNN partial
initialization from --init_checkpoint or --mae_init_checkpoint is rejected;
resume is for a matching CNN run, not for transferring ViT weights.
CNN self-supervised transfer is now available separately through
`--cnn_init_checkpoint` / `CNN_INIT_CHECKPOINT`; it must reference a completed
matching-backbone checkpoint, not a ViT MAE checkpoint.

Compare scratch against scratch with identical manifests, seed, ROI selection,
loss, LR, batch/accumulation, update budget and evaluation. These encoders are
not parameter/FLOP matched. Do not change batch silently: high-resolution
encoder activations may be expensive, especially at per-GPU batch16.
CPU validation does not establish GPU BF16 compatibility or peak memory.
Follow the repository revision policy before any submission.

## SAM-tail integration

Use either CNN with the existing SAM-tail recipe, setting:

```bash
export BACKBONE=nnunet_resenc_s16  # or mednext_b_s16
export SLOT_HEAD_TYPE=arm_f_sam_tail QUERY_REFINEMENT=none PIXEL_PE=none
export DIMENSION=2D MAE_INIT_CHECKPOINT=
export HSAM_SUPERVISION=soft_prior_aux  # or downsample_gt
```

`QUERY_REFINEMENT=none` disables the E-only option, NOT SAM-tail refinement.
The original P16 -> P8 -> P4 token refinement, reverse attention, mask token,
final token-to-pixel attention and mask readout remain in ArmFDecoder unchanged.
CNN now contributes S16 tokens and S8/S4 skips computed in the same encoder pass.
No detached features or second encoder pass: segmentation gradients reach S8/S4.
`--cnn_skip_source encoder` is the new default. `stem` preserves the old CNN
implementation for checkpoint compatibility; old checkpoints lacking this argument
are reconstructed as `stem` by evaluation. Existing ViT models retain their stem.
Both original soft-prior auxiliary loss and downsampled-GT two-stage loss are
retained; reuse their existing loss weights/schedules. No MAE encoder transfer
from ViT is implemented for these CNNs. Compare against scratch SAM-tail to isolate
the backbone change, not only against the existing MAE-initialized results.

## AutoPET CNN masked pretraining (2026-10-13)

`main_pretrain_autopet_mae.py --backbone nnunet_resenc_s16|mednext_b_s16`
uses the same image-only AutoPET loader and seed0 patient-level 90/10 split.
Input224, mask75% of 16x16 patches before the encoder, masked-only raw pixel
MSE. Dense convolution processes the masked image; this is not sparse ViT MAE.
A temporary linear reconstruction decoder predicts 16x16x3 pixels per token.
100000 updates, 5000 warmup, 4 GPUs x batch16 x accumulation3 =192,
AdamW, base LR1e-4 (actual7.5e-5), weight decay.05, BF16, gradient clip1.

Only stem/stages transfer (strict keys/shapes/finiteness/backbone validation).
The pretraining reconstruction decoder is discarded; downstream 512->768
projection, token modules, spatial decoder and SAM-tail remain randomly
initialized and all are trained. A SHA256 load report is saved in each run.

Downstream: original WORD split (28586 train slices /6990 test slices),
0.7/0.7/2 spacing, ROI20, 118800 updates, same effective batch192,
SAM-tail soft_prior_aux or downsample_gt. Historical training-slice calibration
and test fixed/calibrated postprocessing are retained for comparability; this
is not the newer official-pool split. Use afterok dependencies on the completed
pretraining job; evaluation depends on its downstream training job.
Training scripts: `slurm/orgslot/train/cnn_mim_sip.sbatch` and
`slurm/orgslot/train/cnn_sam_transfer_sip.sbatch`.

### Submitted job chains

以下10月13日提交已于10月14日取消替换（全部未启动）。新编码器原生S8/S4链：
ResEnc3052380 → 3052381/3052383 → 3052382/3052384；
MedNeXt163397 → 163398/163400 → 163399/163401。
新固定runtime为各账号cnn_skips_6f9a854，SHA6f9a8541ba2b99cef06d0b5b01da553471c669a1。
soft_prior_aux/downsample_gt各一组，原预训练预算/参数保留，只有下游跳接来源改变。

| Encoder | Cluster / user | Pretrain | soft prior+aux train / eval | downsample GT train / eval |
|---|---|---|---|---|
| nnU-Net ResEnc | SIP / bolinren19 | 3048464 | 3048465 / 3048467 | 3048466 / 3048468 |
| MedNeXt-B | XEC / sifansong | 162141 | 162142 / 162143 | 162144 / 162145 |

Checked at submission: both pretrains PENDING(Priority); all downstream tasks
PENDING(Dependency) with afterok links verified. No GPU validation yet.
SIP pretrain/soft train use account sifansong QoS4a800; other ResEnc tasks use
account angelosstefanidis QoS8a800 because the 4a800 submission-count limit is4.
XEC uses account sifansong QoS8gpus. All training jobs request4 A800/24CPU/192G;
SIP limit7days, XEC8gpus limit5days. Evaluations request1 A800/10CPU/128G/1day.

Pinned runtime commits (do not update underneath jobs):
- SIP: `7ae0689250489383a9a4611b1dd1560c513a2acf`,
  `/gpfs/work/aac/bolinren19/OD_OWT/.worktrees/cnn_mim_7ae0689`
- XEC: `ab5c0aa9e3ef47cf785effdfd58a0bd29a115cf1`,
  `/gpfs/work/aac/sifansong/worktrees/cnn_mim_ab5c0aa`

Both passed check_experiment_revision.sh before submission and match GitHub
at their submission time. The second commit only adds explicit account path
overrides; the model/training algorithm is unchanged. CNN_PYTHON,
CNN_PRETRAIN_CSV, CNN_WORD_ROOT, CNN_ROI_INDEX and CNN_LPIPS_STATE are explicitly
set to sifansong paths on XEC, never the SIP defaults.

CPU verification:6 CNN tests passed including transfer rejection, unchanged
non-encoder weights and joint SAM losses. A 25-step fixed-mask two-image
AutoPET CPU fit reduced loss0.19832->0.01306 (ResEnc),0.14443->0.00167 (MedNeXt).
This is a pipeline check, not a GPU memory test or a generalization result.

## Source links

- [nnU-Net ResEnc planner](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/experiment_planning/experiment_planners/residual_unets/residual_encoder_unet_planners.py)
- [BasicBlockD](https://github.com/MIC-DKFZ/dynamic-network-architectures/blob/main/dynamic_network_architectures/building_blocks/residual.py)
- [MedNeXt blocks](https://github.com/MIC-DKFZ/MedNeXt/blob/main/nnunet_mednext/network_architecture/mednextv1/blocks.py)
- [MedNeXt preset configurations](https://github.com/MIC-DKFZ/MedNeXt/blob/main/nnunet_mednext/network_architecture/mednextv1/create_mednext_v1.py)

Architecture references checked 2026-10-13. Implemented locally without importing
or installing upstream packages; benchmark claims require separate evaluation.
