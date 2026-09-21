# 统一配置与部署契约

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
