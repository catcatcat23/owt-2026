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
