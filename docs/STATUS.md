## 2026-11-02：MedNeXt历史续训结果补录

已核验sifansong/XEC 174304/174305均COMPLETED（10月28日），并非新完成任务。
MedNeXt缩小GT：head固定85.2983%、校准86.2684%；recon direct_post82.1326%、
indirect_post7.4181%（异常）。README与RESULTS已补齐ViT/ResEnc/MedNeXt对照。
MedNeXt soft prior＋aux原163398超时、166492取消，本次未找到最终评估证据。

## 2026-11-02 20:13 CST：CNN结果更新

ResEnc缩小GT 3052383/3052384均COMPLETED；head固定85.8013%、校准86.4217%，
recon direct_post81.1808%、indirect_post4.8190%。完整八类和证据见[结果台账](RESULTS.md#8-cnn编码器与原生跳接对照)。
ResEnc soft prior＋aux 3052381仍RUNNING（epoch390），3052382等待依赖。
以下较早状态为历史快照，不能覆盖本节已完成状态。

## 2026-11-01 16:32 CST — SAM new-tail continuation

- Original sifansong/XEC `174375` reached TIMEOUT after5days, epoch765 partial. CPU-loaded `stage2_separation_174375/checkpoint-760.pth`: epoch760,112628 optimizer updates, model tensors finite, optimizer/scaler present;6172 updates remain to118800. Resume from epoch761; no GPU smoke.
- Submitted **185801** (2RTX4090,12CPU,192GiB,5days,QoS8gpus; initially PENDING Priority). Same batch16×2×accum6=192, same new4 pixel decoder and new4 tail/slot-private interaction, same data/loss/freeze policy.
- Replacement evaluation **185802**,1RTX4090/10CPU/128GiB/1day, `afterok:185801`; points to `stage2_separation_185801`, includes validation calibration/test/head/reconstruction. Cancelled obsolete `174376` only after both replacement jobs existed.
- Existing experiment retains immutable runtime `/gpfs/work/aac/sifansong/worktrees/inc44_newtail_ea8780a`, source `ea8780adb46e2f6aa682dd0169a1277c4e3b4406`; verified SOURCE_COMMIT and nine critical source/script SHA256 values against Git. No model code changes. Submission wrapper committed/pushed as `cfb063cf` before sbatch: `scripts/orgslot/resume_newtail_20261101.sh`.
- Evidence: `/gpfs/work/aac/sifansong/artifacts/inc44_newtail_20261027/{resume_20261101.sh,resume_source_hashes.json,resume_20261101_jobs.txt,185801.out,185802.out}`. Remaining training estimated6–7hours after allocation from prior throughput, excludes queue and evaluation.

## 2026-10-31 00:17 CST — BTCV body-center512 submitted

- Approved replacement: image-only body center → fixed512×512 crop/pad, no global resize; spacing0.7×0.7×2mm unchanged; physical ROI384 retained (ROI-only384→512 augmentation). 2D SAM-tail soft prior+aux, AutoPET MAE encoder, full Common8 Offline24/6, seed42 split/model seed0.
- sifansong/XEC/account sifansong/QoS8gpus: **182668** training (PENDING Priority at submission), **182669** continuation afterany:182668, **182670** head+reconstruction evaluation afterok:182669. Train/continuation each2A800,16CPU,128GiB,5days; eval1RTX4090,8CPU,96GiB,1day. Per-GPU batch16, accumulation6, effective192,118800updates. Continuation exits if budget already complete; evaluation requires a verified completed checkpoint.
- Pinned source `b8779a78d61d6c3b1e9b478ac88d92100d4bfd13`; local immutable worktree passed `tools/check_experiment_revision.sh` against origin/feature/orgslot before submission. Remote immutable runtime `/gpfs/work/aac/sifansong/worktrees/btcv_body512_b8779a78`; archive SHA256 `5dc761011c093cbf498bd7c643651ba16b88771879e9a3babb689ec510cf5c69` verified remotely.
- Run root `/gpfs/work/aac/sifansong/artifacts/btcv_body512_20261031`: `launch.sh`, `jobs.txt`, `body_center_audit.json`, `train_JOB`, `evaluation`. Reuses native data `/gpfs/work/aac/sifansong/artifacts/btcv_offline_20261030/data/BTCV`, never resumes withdrawn182515.
- CPU checks: local AND remote production geometry audit30cases/6775slices, zero clipped GT; MAE encoder strict load77 tensors;512 target position embeddings regenerated; label-independent crop and pixel-preserving padding checks passed. No GPU smoke; no training metric or GPU-memory claim yet. Existing random rotation/ROI augmentation can change coverage; the audit covers deterministic global evaluation only.
- Complete protocol and launch parameters: [CONFIGURATION.md](CONFIGURATION.md#btcv-body-center512-replacement-2026-10-31).

## 2026-10-30 — BTCV full-field run withdrawn

User requires exact WORD preprocessing. Cancelled sifansong/XEC182515 (train),182516 (continuation),182517 (evaluation). Removed the newly added full-field scaling path; restored original WORD geometry. No replacement submitted. Audit: img0004 is435x435 (size error);8other cases lose GT under center448, including test img0033 (390spleen voxels). `tools/audit_btcv_fixed_crop.py` writes the evidence and blocks submission. Records below describe the superseded run, not an active or valid baseline.

## 2026-10-30 23:07 CST — BTCV Offline submitted

- SAM-tail soft prior + aux, AutoPET MAE encoder only, eight foreground organs + background; all modules train.
- sifansong / XEC / QoS8gpus: train **182515** RUNNING on xgpua800n5, start23:07:13; continuation **182516** afterany:182515; head + reconstruction evaluation **182517** afterok:182516 on one4090.
- Source `60c289eb60232b0ec098d0497b65af6f827baf65`, immutable runtime `/gpfs/work/aac/sifansong/worktrees/btcv_offline_60c289e`; source archive SHA256 `40ac0b93b59e48a7b36a2d974fc9d122519009200dc2cd8627b9dff2d517dc96`.
- Artifacts `/gpfs/work/aac/sifansong/artifacts/btcv_offline_20261030`; runs/jobs.txt, data/BTCV/metadata/full_field_audit.json and small_organ_roi_index.json are evidence.
- Fixed seed42 split24/6, 5372 training / 1403 test slices. Spacing **0.7x0.7x2 mm**, cubic image / nearest label interpolation matches current WORD. Superseded1x1x3mm data not used.
- Full-field square-pad/resize448 instead of legacy WORD center crop. Audit: eight classes have zero positive slices lost during deterministic native-to448 label resize in both train and test. This does not prove boundary/voxel preservation or absence of losses during source resampling/augmentation. Focus4/5/6 bounding boxes fit384 before augmentation; ROI may clip non-focus organs.
- Two A800, batch16/GPU, accumulation6, effective192;118800updates; training5days + continuation5days; evaluation1day. Explicit no-validation-set mode; threshold tuning uses training cases only and is labelled accordingly, fixed0.5 raw/post retained.
- CPU geometry regression4/4 + syntax checks passed; no GPU smoke. At submission verification, Slurm allocation/start confirmed, not yet a completed training epoch or performance result.

# 实验状态快照

## 2026-10-14：CNN编码器原生S8/S4跳接替换

新下游使用cnn_skip_source=encoder，移除独立浅层stem；S8/S4来自同一次编码器前向。
12项CPU回归通过，旧ViT路径及旧CNN stem checkpoint兼容保留；GPU尚未启动。
固定SHA6f9a8541ba2b99cef06d0b5b01da553471c669a1，SIP/XEC运行树cnn_skips_6f9a854，
提交前两端revision guard通过。因为原预训练/下游均未启动，统一重建明确输出路径的新链。

| 编码器/账号集群 | 预训练 | soft_prior_aux训练/评估 | downsample_gt训练/评估 |
|---|---|---|---|
| ResEnc / bolinren19 SIP | 3052380 | 3052381 / 3052382 | 3052383 / 3052384 |
| MedNeXt / sifansong XEC | 163397 | 163398 / 163399 | 163400 / 163401 |

提交状态：预训练Priority；下游afterok预训练；评估afterok下游，尚无性能结果。
旧SIP3048464–3048468、XEC162141–162145取消；取消前均PENDING，旧snapshot保留。
MIM仍100000更新，WORD118800更新、每卡16×4×累积3=192，loss/数据/seed不变。
SIP训练7天、XEC8gpus训练5天；评估1GPU/1天。其它任务不变。

## 2026-10-14：SAM组合aux失败后恢复链

- 原157059于13:07 FAILED：epoch480附近ALLREDUCE SeqNum7003207超时1800秒。
- checkpoint470：CPU模型/优化器有限；69708更新；最新代码strict exact加载通过。
- 新训练163200，sifansong/XEC，4A800/20CPU/192GB、4gpus/7天，PENDING(Priority)。
- 新评估163201，1RTX4090/10CPU/128GB、4gpus/1天，afterok:163200，PENDING(Dependency)。
- 从epoch471恢复optimizer/scaler，总预算118800；仅启用已有常量LPIPS buffer广播关闭保护。
- 固定runtime `/gpfs/work/aac/sifansong/worktrees/sam_combo_resume_1f2b0e1`，
  SHA `1f2b0e169b7d454d355ba331a8a308db65ab92ed`，提交前revision guard通过。
- 旧失效评估157061已取消；原模型/日志保留。不运行额外GPU smoke；正式GPU尚未启动。
- 其它更新：157058约epoch769、157119约434；157599今天13:07启动、约epoch6。

以下保留10月13日检查快照，157059状态以本节为准；估计启动时间须重新查Slurm。

时间口径：2026-10-13 Asia/Shanghai。
队列/runtime本次整理约22:19前后只读核查；epoch来自21:04日志快照，**不是实时epoch**。
本次只整理文档，未提交、取消、迁移或修改任何任务。

| 实验 | 账号/集群 | 训练 → 评估 | 队列状态 | 最近已核验训练快照 |
|---|---|---|---|---|
| SAM两阶段+soft prior，无额外prior aux | antengcai23/XEC | 157058 → 157060 | RUNNING / Dependency | epoch618/803（21:03） |
| SAM两阶段+soft prior+aux | sifansong/XEC | 157059 → 157061 | RUNNING / Dependency | epoch399；21:03仍推进但此前有23–42分钟/20step停顿 |
| SAM增量私有交互+adapter | sifansong/XEC | 157119 → 157120 | RUNNING / Dependency | epoch194（21:03） |
| E Stage2共享query恢复 | bolinren19/SIP | 3044537 → 3044538 | PENDING / Dependency | Resources；无新结果 |
| E Stage2分组query恢复 | bolinren19/SIP | 3044539 → 3044540 | PENDING / Dependency | 无新结果 |
| 3D SAM恢复 | bolinren19/SIP | 3044541 → 3044542 | PENDING / Dependency | 不能据任务名声明已恢复 |
| 3D SAM缩小GT | antengcai23/XEC | 157599 → 157600 | PENDING / Dependency | 无最终结果 |
| ResEnc AutoPET MIM | bolinren19/SIP | 3048464 | PENDING | 尚无GPU预训练结果 |
| ResEnc迁移soft/缩小GT | bolinren19/SIP | 3048465/66 → 3048467/68 | Dependency | 等预训练成功 |
| MedNeXt AutoPET MIM | sifansong/XEC | 162141 | PENDING | 尚无GPU预训练结果 |
| MedNeXt迁移soft/缩小GT | sifansong/XEC | 162142/44 → 162143/45 | Dependency | 等预训练成功 |
| 旧Stage1五slot评估 | bolinren19/SIP | 3001901 | JobHeldUser | 不与已完成替代评估混同 |

SAM私有交互156538/156539的81.84结果已归档到 [增量专项](INCREMENTAL_READOUT_ABLATION.md)，
不是每次查询COMPLETED都出现一个新实验。其它旧结果见 [台账](RESULTS.md)。
同账号另有EEG任务，不属于OWT/OrganSlot清单，未修改。

## 固定runtime与安全边界

- SIP恢复任务：`.worktrees/quota_resume_381171c`；CNN：`.worktrees/cnn_mim_7ae0689`。
- sifansong/XEC：`/gpfs/work/aac/sifansong/worktrees/sam_combo_7837635`、
  `inc44_adapter_348fd46`、`cnn_mim_ab5c0aa`。
- antengcai23/XEC：`/gpfs/work/aac/antengcai23/worktrees/sam_combo_7837635`、`sam3d_2558bac`。
- 本次仅修改开发目录 `.worktrees/orgslot_dev_current`，上述runtime保持不动。
- 最新commit不能自动修复旧快照；后续重提必须按AGENTS核验SHA与afterok依赖。

## 下次优先检查

1. 157059间歇性变慢：先查日志、GPU/进程/I/O；未确认NaN、NCCL或GPFS根因，不擅自重启。
2. 两组SAM组合最终head/recon，与原downsample_gt基线对齐；不要用loss预测Dice。
3. adapter新类收益与旧类binary/互斥mask分别检查；checkpoint需要加载核验。
4. CNN预训练启动后检查数据身份、100k更新预算、权重迁移报告；CPU测试不冒充GPU验证。
5. 已知Indirect异常仍需独立诊断；不概括为所有recon正常。
6. 需要新状态时重新查Slurm，不沿用历史预计启动日期。

历史完整状态、CPU/GPU边界、源SHA及提交记录已保存在
[旧STATUS](archive/HISTORY.md#merge-status)；禁止把该归档当今日队列。

## 2026-11-02: 2+2+2+2 ROI-controlled rerun

At user request, stopped anteng/XEC Stage2 180499 (checkpoint-200 retained),
held 180500–180507. Existing runtime and outputs remain unchanged.
New opt-in `roi_focus_policy=current` includes every newly supervised organ
in that stage's ROI pools; historical default `small` remains unchanged.
First controlled rerun: Stage1 spleen/right kidney, same MAE encoder, official96
split, seed0, 118800 updates, 4 GPUs × batch16 × accumulation3, 448 input,
384 ROI, 20% ROI probability (uniform target choice), retained-slot loss.
This does not guarantee the ROI target is retained for segmentation supervision.
Only Stage1, continuation, and evaluation are submitted before deciding later stages.
Evaluation protocol is unchanged. New root: anteng/XEC
`/gpfs/work/aac/antengcai23/artifacts/inc2222_roi_20261102`.
Job IDs and pinned source are recorded in its `runs/jobs.txt`.

Submitted 2026-11-02 18:38 CST, antengcai23/XEC, Slurm account sifansong,
QoS 8gpus: train **190004** (Priority), continuation **190005**
(afterany:190004), evaluation **190006** (afterok:190005).
Pinned source `744355de85fb359f3000f1b66f7d340ab6621663`, runtime
`/gpfs/work/aac/antengcai23/worktrees/inc2222_roi_744355de`.
Source tar SHA256 `349a53c2465c451c652747676b776a5abbd187f2a974246872c970039764af36` verified remotely.
Training/continuation: 4 A800, 24 CPU, 192 GiB, 5 days each;
evaluation: 1 RTX4090, 10 CPU, 128 GiB, 1 day.
CPU checks: Python parse, shell syntax, revision guard, data manifest validation,
and Stage1 ROI geometry passed. Positive pools: spleen 4455, right kidney 4887;
maximum bounding-box diagonals 311.23 and 197.71 pixels (both below 384).
GPU smoke was not run; formal training was pending at submission.

## 2026-11-03：补齐任务准备

PCDD suite已实现可变分组；拟排WORD Offline/4+2+2/7+1，BTCV4+4/4+2+2/4+1+1+1+1/7+1。
保留原WORD4+4已完成结果和BTCV Offline在训链，不重复启动。
当前ROI Stage1的后续全ROI阶段单独设置决策hold，不擅自释放此前暂停的旧任务链。
提交回执由各账号`artifacts/pcdd_suite_20261103/{WORD,BTCV}/PROTOCOL/jobs.txt`保存。
CPU回归：旧2+2+2+2两项通过；新分组可见标签、单新类梯度/旧logits保持及strict reload两项通过。
未运行GPU smoke，正式训练状态以Slurm为准。

批量提交触发QoS `MaxSubmitPU=16`，改为`queue.sh`有界自动接续。
本次初始等待任务191465–191482将由新队列替换；不改变运行任务或历史hold任务。
后续阶段已配置但尚无Job ID，必须区分“自动接续计划”与“已进入Slurm”。
CPU模拟调度测试覆盖三任务依赖、hold/release和重复提交拒绝；无GPU smoke。

### 2026-11-03 17:18 CST：新队列提交回执

| 账号/集群 | 已提交训练→续训→评估 | 前置评估 | 后续自动接续（尚无Job ID） |
|---|---|---|---|
| antengcai23/XEC | WORD Offline `191497→191498→191499` | `190006` | WORD 4+2+2三阶段→7+1两阶段 |
| sifansong/XEC | BTCV 4+4 Stage1 `191500→191501→191502` | `182670` | 4+4 Stage2→4+2+2三阶段→4+1+1+1+1五阶段→7+1两阶段 |

首批6个任务均PENDING依赖等待，训练4×A800/24 CPU/192 GiB/5天，
评估1×4090/10 CPU/128 GiB/1天；每卡batch16、累积3、有效batch192。
所有阶段118800 updates；现有BTCV Offline的2卡任务保持原资源。
原始失败批量提交的191465–191482均在PENDING时取消，未启动训练。
历史180500–180507的hold链及现有运行任务保持不变。

固定源提交 `9b9815136efa48a3532ae730c807aff7e70a67a5`，两个账号运行路径
`/gpfs/work/aac/{antengcai23,sifansong}/worktrees/pcdd_queue_9b981513`；
归档SHA256 `48feefde5277bacd16a2ce5e2b180a9263f57694165e9732c4e85603819070a4`，两端均校验通过。
提交入口 `scripts/orgslot/pcdd_suite/queue.sh`；完整计划、launch、日志、逐阶段回执保存在
两账号的 `artifacts/pcdd_queue_20261103/{plan.txt,launch.sh,submission.log,receipts/}`。
每个阶段评估输出 `stageN_evaluation/comparison.json`，包含逐器官Offline差值及最终阶段PCDD论文分组参考。
论文没有已核验逐器官值的格子保持null，不能用组均值代替；病例协议未完全对齐，PCDD仅为外部参考。

全当前器官ROI的2+2+2+2仍是单独消融：Stage1在训，后续ROI索引已生成；
后续几何覆盖和Stage1效果待核验，不混入本次small-organ ROI正式对照队列。

## 2026-11-04 16:27 CST：BTCV续训改为4×A800

按用户要求将等待中的2卡续训182669替换为 **194188**（sifansong/XEC，account=sifansong，QoS8gpus）。
配置4×A800、24 CPU、192 GiB、5天，每卡batch16、accum3、有效batch192。
当前182668保持原快照运行；新续训依赖afterany:182668，动态选择最新有效checkpoint，恢复模型、Adam和scaler。
评估182670已改为afterok:194188；后续191500仍依赖afterok:182670。旧182669已取消。
固定源 `fed23c5791ac2ed61cd059edda11de334e48543b`，runtime
`/gpfs/work/aac/sifansong/worktrees/btcv_resume4_fed23c57`；源tar SHA256
`d2fa856b949acc77df0b10343ff705a2e10d951ae80f4f553d3b940e8da4b3c4`两端一致。
回执 `artifacts/btcv_body512_20261031/resume4_jobs.txt`，提交入口 `resume4_launch.sh`。
同时修正改变world size后按epoch推算更新数的问题：恢复后从Adam实际step计算偏移和剩余epoch，
保持118800更新及按更新数的学习率计划。每epoch取整、数据排列和随机器官组合会随并行布局变化，
因此有效batch相同不代表逐步轨迹完全相同。
CPU：Python语法、shell语法、2→4卡更新预算算术与revision guard通过；未运行GPU smoke。
新续训尚PENDING，GPU恢复与加速效果待启动后核验。
