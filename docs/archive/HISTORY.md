# 历史原文档案（按需查询）

## 专项文档合并原文（2026-10-13）

<a id="merge-cnn-backbones"></a>

### docs/CNN_BACKBONES.md

来源：`feature/orgslot @ f198a191fe9460410c0e8ed32c3392564b165ae1`。历史状态不作为当前状态。

`````text
# E cross: encoder-only CNN ablation

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

Both export only S16 through a 1x1 projection (512 -> 768 for the base model),
flatten to [B,784,768], add the existing spatial positional embedding, and use
the existing encoder LayerNorm. This tensor feeds the original Collector and
the original E pixel decoder. CNN S1/S2/S4/S8 are NOT new decoder skips.
The existing image spatial stem, query cross-attention, AHER, reconstruction
decoder, token count and losses are unchanged. ViT encoder blocks and its
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
CNN contributes only its S16 grid; its shallow features are not new skips.
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
`````

<a id="merge-frozen-p4-probe"></a>

### docs/FROZEN_P4_PROBE.md

来源：`feature/orgslot @ f198a191fe9460410c0e8ed32c3392564b165ae1`。历史状态不作为当前状态。

`````text
# Frozen P4 probe diagnostic

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
Pre/post feature scales may affect optimization. This implementation has not
yet been GPU-preflighted or submitted. Full-checkpoint loading and peak memory
must be checked on actual4090 before claiming operational validation.
`````

<a id="merge-token-intervention"></a>

### docs/TOKEN_INTERVENTION.md

来源：`feature/orgslot @ f198a191fe9460410c0e8ed32c3392564b165ae1`。历史状态不作为当前状态。

`````text
# Frozen Arm E + MAE token intervention

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
`````

<a id="merge-unified-3d-evaluation"></a>

### docs/UNIFIED_3D_EVALUATION.md

来源：`feature/orgslot @ f198a191fe9460410c0e8ed32c3392564b165ae1`。历史状态不作为当前状态。

`````text
# 3D inference, shared 2D evaluation protocol

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
`````

<a id="merge-query-readout-ablation"></a>

### docs/QUERY_READOUT_ABLATION.md

来源：`feature/orgslot @ f198a191fe9460410c0e8ed32c3392564b165ae1`。历史状态不作为当前状态。

`````text
# Controlled query/readout ablation

## 当前架构导航（2026-10-06）

本节为后续工作入口；以下历史实验记录保留，不代表仍需继续全部路线。

用户确认仅保留三个主线配置：E cross_attn、SAM-tail soft_prior_aux、
SAM-tail downsample_gt。其余配置的指标、实验目的、训练预算及比较限制保留为消融记录；
下文提及none、soft_prior、upsample_logits等仅为解释历史对照，不是新增主线。
当前本地/远端仅main、feature/orgslot、experiment/psem、experiment/lossbalance四个分支，
无需再删除OrganSlot实验分支。历史detached worktree不是分支，不在分支清理中删除。

共同路径：CT → ViT z；每器官 Collector → TGEnc → token_norm → 20 tokens。
重建路径保持 tokens → AHER → canvas融合 → 共享重建decoder。
空间路径为 CT浅层CNN跳接 + z → 共享多尺度pixel decoder → P16/P8/P4。
448输入时P4为112×112（边长1/4、面积1/16），不是边长1/16。

### E：简单读出与单query交互

`slot_head_type=arm_e_multiscale_query`。
20 tokens → query_norm → mean pool → query_proj + slot identity → query。
`query_refinement=none`直接query·P4；`cross_attn`先query读取P4，经交互FFN再点积。
最终logits上采样到输入尺寸。E没有F的20-token P16→P8→P4逐尺度refinement。

### SAM-tail：保留两条明确区分的配置

共同head为`arm_f_sam_tail`，20 tokens经投影+identity，依次读取P16/P8/P4；
之后SAM-tail使用专用mask token、双向交互及最终回读，mask-token MLP生成
动态权重，与tail更新后的像素特征点积，最后上采样。

1. **soft prior**：`hsam_supervision=soft_prior`或`soft_prior_aux`。
   每次读取前，以当前tokens的归一化均值与该尺度归一化pixels点积乘sqrt(C)，
   得到粗logits。`gate=0.2+0.8*sigmoid(detach(logits))`，将`log(gate)`加入
   token→pixel attention logits。aux版本对三张粗预测加监督；路由先验detach，
   粗预测本身仍通过aux获得梯度。此模式不会把粗mask传入tail的prior参数。
2. **两阶段粗到精mask**：`hsam_supervision=downsample_gt`或`upsample_logits`。
   完成多尺度refinement后，用tokens与P4点积得到coarse mask；其sigmoid传给
   SAM-tail作为prior，再输出final mask。两个配置分别采用缩小GT或上采样粗logits
   的辅助监督。这里“两阶段”指一次forward内的两级解码，不是增量Stage1/Stage2，
   也不是分两次训练。

两条路线使用互斥枚举；合并soft prior与tail粗mask引导需要新的受控实验，现有
`soft_prior_aux`不能冒充二者组合。`none`保留为SAM-tail无引导对照。

### 增量与历史兼容边界

当前E Stage2保留query_shared/query_split；SAM私有交互增量只支持soft_prior_aux，
尚不支持两阶段粗mask配置的slot_private版本。冻结范围及118800/阶段预算见
[增量设计](INCREMENTAL_READOUT_ABLATION.md)。不能仅切换枚举就声称完成对应增量实验。

其它F读出、P2、hard-mask等仅保留历史加载/复现，不删除原始checkpoint及指标。
本次仅收敛文档入口，不改模型forward、state_dict、训练配置或运行快照。

## P4 Reverse-Dot batch16 correction (2026-09-20)

Replace baseline2955011/2955012 and lambda_seg0.03 chain2964744/2964745.
Both use arm_f_reverse_dot, four GPUs, microbatch16/accum3/effective192,
scratch seed0, WORD07072 ROI20, topk small-organ loss, LR7.5e-5,
118800 updates/warmup5940. Only lambda_seg differs: 0.01 vs 0.03.
Baseline had only just started (about7min), with no checkpoint at inspection;
both replacements start from epoch0, no resume or MAE. Retain old artifacts.
Keep bolinren19/SIP associations: baseline sifansong/8a800;
lambda0.03 angelosstefanidis/8a800. Train4A800/20CPU/192GB, seven days;
unified head calibration/test/recon eval1A800/10CPU/128GB, seven days,
afterok new training. Eval EXPECTED_LAMBDA_SEG must match each arm.
Use latest origin/feature/orgslot in a new pinned worktree, not either old runtime.
Microbatch16 GPU memory/throughput remain unverified. Do not call it a GPU-tested
configuration based on CPU compatibility tests from earlier work.

Submitted 2026-09-20 01:43 CST; both trainings Pending(Priority):

| lambda_seg | Train | Unified eval | Slurm account / QoS |
| --- | --- | --- | --- |
| 0.01 | 2965316 | 2965317, afterok:2965316 | sifansong / 8a800 |
| 0.03 | 2965318 | 2965319, afterok:2965318 | angelosstefanidis / 8a800 |

Runtime /gpfs/work/aac/bolinren19/OD_OWT/.worktrees/p4_b16_db9c788,
source db9c788152b87d755ef67dc46eb59c7d70526c5f. Revision gate and shell
configuration assertions (micro16/accum3/effective192/lambda per arm) passed.
Training/evaluation paths and afterok dependencies verified after submission.
Logs: artifacts/p4_b16_db9c788/logs. Old four jobs confirmed cancelled;
2955011 used8min43sec before cancellation, no saved progress used in replacement.
No GPU preflight run and no claim of batch16 GPU validation.

## SIP E / Query-Dot batch16 resume (2026-09-20)

User requested changing running 2D E2955007 and Query-Dot2955009 from
microbatch8/accum6 to microbatch16/accum3, retaining four GPUs/effective192.
Resume sources: arm_f_linear_2d/Results/OrganSlotBank/Common8/WORD_2D/
ArmF_2D_2955007/checkpoint-120.pth and ArmF_2D_2955009/checkpoint-10.pth.
E saved checkpoint120 during checks; validated and selected the newer file.
Restore model/optimizer/scaler, start at epoch121 / epoch11 respectively.
Both old/new microbatch settings yield148 optimizer updates per epoch;
retain LR7.5e-5, total118800 updates, WORD07072 ROI20, topk small-organ loss,
lambda_seg0.01, scratch lineage, no MAE. Resumed runs write separate job outputs.
Use latest origin/feature/orgslot in a new immutable worktree. Cancel old
training/evaluation chains only after checkpoint validation; rebuild afterok
evaluations against the new job IDs. Preserve all old results/checkpoints.
This is mixed-microbatch training, not a clean batch16-from-scratch experiment:
group-reduced losses can change gradients despite equal effective batch.
GPU peak memory/throughput at batch16 are not yet verified.

Submitted 2026-09-20 01:34 CST on bolinren19/SIP (Slurm account sifansong):
E train2965273 / eval2965274 (4a800); Query-Dot train2965275 / eval2965276
(8a800). Train4A800/20CPU/192GB; eval1A800/10CPU/128GB; seven days each.
Both training jobs initially Priority; both afterok evaluation dependencies
verified. Cancelled old2955007/2955008/2955009/2955010; old artifacts retained.
Runtime /gpfs/work/aac/bolinren19/OD_OWT/.worktrees/sip_2d_b16_d7dedbe,
source d7dedbe5a03d32224103fadf68f808d7f6266614; revision gate passed.
CPU strict model/optimizer loading and tensor finiteness passed on real
checkpoints, optimizer steps17908 / 1628. Scaler key present; training restores
it through existing resume logic (BF16 scaler disabled). GPU resume not yet tested.

## Active P2 submissions — batch16 correction, 2026-09-20 01:16 CST

| Experiment | Train | Unified eval | Status at submission |
| --- | --- | --- | --- |
| P2 baseline | 141130 | 141131, afterok:141130 | Priority / Dependency |
| P2 + soft mask | 141132 | 141133, afterok:141132 | Priority / Dependency |

Both: antengcai23/XEC, Slurm account sifansong, QoS8gpus, partition gpua8001t,
4A800/20CPU/192GB training; 1A800/10CPU/128GB evaluation; five-day limit.
Microbatch16 x 4 GPUs x accumulation3 = effective192. Scratch seed0,
lambda_seg0.01, topk background, WORD07072 ROI20, 118800 optimizer updates.
Pinned source efcef0739b581ec49c969c312d07c97574308cbb;
runtime /gpfs/work/aac/antengcai23/worktrees/p2_b16_efcef07.
Scripts: slurm/orgslot/train/arm_f.sbatch and
slurm/orgslot/eval/arm_e_mae_unified.sbatch. Both job paths/dependencies verified.
Seven CPU tests and dataset identity passed on XEC; revision gate passed.
No GPU preflight performed. Peak memory for microbatch16 remains unknown.
Cancelled the four still-pending batch8 jobs 141125/141126/141128/141129;
no training progress lost, historical source snapshots retained.

## P2 + soft spatial attention prior (2026-09-20)

Independent 2D head: `arm_f_reverse_dot_p2_softmask`. Control:
`arm_f_reverse_dot_p2`. Both P2 arms use microbatch16/accum3 on four GPUs.
The original batch8 job chains 141125/141126 and 141128/141129 are superseded
before starting; preserve their historical runtime snapshots.
Only token-to-pixel cross-attention changes. Before each P16/P8/P4 block,
compute `M = sigmoid(sqrt(C) * normalize(memory) dot normalize(mean(tokens)))`
from the current projected/refined organ tokens and that scale's original
pixel features. Broadcast the same organ prior to all 20 tokens / 4 heads:
`attention = softmax(QK/sqrt(d) + log(0.2 + 0.8*M))`.
The prior is detached, alpha=1, epsilon=0.2; no GT masks, extra trainable
parameters, auxiliary losses, mask threshold, mask annealing or token matching.
The floor bounds the bias to [log(0.2), 0]; it does not guarantee recovery of
missed organs. P16 uses input-projected tokens; later scales use refined tokens.
This is same-scale self-guidance, not an exact Mask2Former/H-SAM reproduction
or a previous-layer supervised mask decoder. Coarse dot priors are not separately
supervised/calibrated and their usefulness remains an experimental hypothesis.

P4 reverse attention, S2/P2 fusion, dot readout, reconstruction and loss remain
identical. State-dict keys and initial values match P2 exactly. Checkpoint args
must retain the new head name: identical tensor keys alone cannot identify routing.
The existing FP32 attention region also computes this bias. 3D is rejected.

Use existing training entry point with
`ARM_F_HEAD_TYPE=arm_f_reverse_dot_p2_softmask`, `ARM_F_DIMENSION=2D`,
`ARM_F_MICRO_BATCH=16`, `ARM_F_ACCUM_ITER=3`, `ARM_F_LAMBDA_SEG=0.01`,
`BACKGROUND_REDUCTION=topk`, scratch seed0, no MAE init, and the same WORD07072
ROI20 dataset / 118800 updates / effective batch192 as P2. Evaluation must use
`EXPECTED_SLOT_HEAD_TYPE=arm_f_reverse_dot_p2_softmask` and the same unified
train-calibrated head + reconstruction protocol. Batch16 GPU memory is unverified;
do not silently reduce microbatch if it fails. Seven CPU tests passed on XEC.

Tests cover alpha=0 exact baseline equivalence, strict P2 weight loading,
identical initialization, finite non-hard biases, preferential synthetic routing,
BF16-input finite backward, full-model gradients and checkpoint roundtrip,
and unchanged legacy heads. GPU validation and scientific benefit remain pending.
Assess small-organ recall, empty-slice FP and predicted/GT volume as well as Dice;
more concentrated attention alone is not evidence of better organ localization.

## P2 shallow-skip readout (2026-09-20)

New 2D-only head `arm_f_reverse_dot_p2`. Keep P16/P8/P4 token refinement,
P4 reverse attention, 20-token mean query and normalized dot scale unchanged.
Return existing stem S2 (64 channels at standard width128); bilinearly upsample
organ-conditioned refined P4, concatenate with 1x1-projected S2, then apply
two 3x3 Conv/GroupNorm(1)/GELU layers at width128. Dot readout now occurs on P2
(224x224 for input448), followed by the original full-size bilinear resize.
Fusion is shared across organs, evaluated per organ. No mask attention, no
extra supervision, no new temporal mixing. Existing head defaults/weights stay
unchanged; P2 is explicitly rejected for 3D. Added modules preserve constructor
RNG for later slots. P2 increases memory/compute; GPU throughput is unverified.

Controlled training: scratch seed0, WORD07072 ROI20, lambda_seg0.01, topk
small-organ loss, LR7.5e-5, 118800 updates/warmup5940, four A800,
microbatch16/accum3/effective192. Control is original Reverse-Dot2955011, NOT
the lambda0.03 experiment. Different SIP/XEC software/hardware is a caveat.
Target antengcai23/XEC, account sifansong, QoS8gpus (maximum five days).
Use the existing unified2D calibration/head/recon evaluator with
EXPECTED_SLOT_HEAD_TYPE=arm_f_reverse_dot_p2 and EXPECTED_LAMBDA_SEG=0.01.
This tests the P2 fusion/readout package, not resolution alone.

Historical batch8 submission, superseded before starting: antengcai23/XEC train141125
(Priority), eval141126 (afterok:141125 verified), both QoS8gpus/5days.
Runtime /gpfs/work/aac/antengcai23/worktrees/reverse_dot_p2_3150ff3,
commit3150ff306ae47e57aa081b538c40de602a0c630a. Six CPU tests passed
locally and on XEC; dataset identity and revision gates passed.
GPU memory/throughput and training stability remain unverified until start.

Existing E, `arm_f_attention`, and `arm_f_linear` implementations are retained.
New heads are `arm_f_query_dot` and `arm_f_reverse_dot`.

| Head | Token refinement | Reverse pixel update | Readout |
| --- | --- | --- | --- |
| arm_e_multiscale_query | Existing mean-pool query | None | Existing normalized dot |
| arm_f_query_dot | Existing F P16/P8/P4 blocks | None | Mean refined tokens, normalized dot with original P4 |
| arm_f_reverse_dot | Same F blocks | Existing reverse attention + residual + FFN | Same query, normalized dot with updated P4 |
| arm_f_attention | Unchanged | Unchanged | Existing LayerNorm + linear classifier |

New dot heads use `sqrt(channels) * cosine(pixel, mean(refined_tokens))`.
No learned pooling or added query projection. Slot identity is already injected
before refinement and is not added again. E's own query construction is unchanged.
The two new paths support 2D and 3D; the controlled E/F experiment is 2D because
the existing E baseline supports only 2D.

Planned common protocol: WORD native07072, 448/global448/ROI384, ROI probability
0.2, scratch, seed0, 20 tokens, 128 channels, small-organ loss, lambda_seg0.01,
LR7.5e-5, 118800 updates, warmup5940, 4 GPUs, microbatch8/accum6/effective192.
Use identical background reduction and optimizer settings across runs. No resume
or MAE initialization. Retain token self-attention, FFN and F attention PE.

Use the existing unified 2D evaluator with EXPECTED_SLOT_HEAD_TYPE set explicitly:
training first6000 calibration, test fixed0.5 and calibrated, raw/post, plus recon.
Do not compare calibrated results against an uncalibrated historical baseline.

Same seed alone does not ensure identical common-module initialization across
architectures (constructor RNG consumption differs). Before claiming an exact
initial-weight-matched ablation, audit or explicitly synchronize shared initial
weights in a separate experiment initialization step. Legacy initialization is
not changed here. E vs F also changes query aggregation/projection and PE, so it
tests the refinement design as a package, not cross-attention alone.

CPU tests cover new paths in 2D/3D, finite gradients and strict round-trip loading;
legacy decoder tests compare keys, initialization and outputs against fd9cf91.
No GPU training is implied by these tests.

## 2D Reverse-Dot segmentation-weight ablation (2026-09-20)

Add one independent scratch run against Reverse-Dot train2955011/eval2955012.
Only lambda_seg changes from 0.01 to 0.03. Set ARM_F_LAMBDA_SEG=0.03 for
training and EXPECTED_LAMBDA_SEG=0.03 for the unified evaluator; both defaults
remain 0.01 for existing experiments. The reconstruction/LPIPS weights, head,
seed0, WORD07072 split, ROI20, topk small-organ loss, LR7.5e-5, 118800 updates,
warmup5940, four GPUs and microbatch8/accum6/effective192 remain unchanged.
No MAE initialization or resume. Keep the original run and evaluation chain.
Evaluate fixed0.5 and training-calibrated head masks plus reconstruction with
the existing protocol. Increased loss weight alone is not evidence of improved
Dice; compare segmentation gains and possible reconstruction degradation.

Submitted 2026-09-20 00:04 CST on bolinren19/SIP, Slurm account
angelosstefanidis, QoS8a800: train2964744 (4A800/20CPU/192GB/7days),
eval2964745 (1A800/10CPU/128GB/7days), afterok:2964744 verified.
Initial states: Priority / Dependency. Pinned source
188555c2c18de95ebb76a21bb23ea3c50cba7db3 at
/gpfs/work/aac/bolinren19/OD_OWT/.worktrees/arm_f_reverse_dot_seg003_188555c.
CPU configuration/metadata checks passed; GPU execution remains pending.

## 3D controlled extension

The new `arm_e_multiscale_query_3d` baseline reuses F's slice-wise E spatial
stem/fusion without token refinement or reverse attention. Its query is exactly
E-style: token LayerNorm, mean, projection, slot identity; normalized dot with
P4 followed by trilinear upsampling. It does not modify the existing 2D E head.
The other two arms use `arm_f_query_dot` and `arm_f_reverse_dot` with dimension=3D.
F attention blocks operate across the slab with existing axial temporal/spatial PE;
the baseline adds no new PE or temporal convolution.

Use ARM_F_DIMENSION=3D, ARM_F_HEAD_TYPE explicitly selected, BACKGROUND_REDUCTION=topk,
scratch/seed0, microbatch6, four GPUs, accumulation2 (48 slabs), 118800 updates,
WORD07072 and ROI20. All three use the unified 3D calibration/head/reconstruction
evaluator, not the older post3d-only evaluator. No training jobs were submitted
as part of implementing this extension.
# Multiscale F + SAM-style tail (2026-09-22)

Requested first training is **2D MAE encoder-only**, not scratch. Launcher:
`slurm/orgslot/train/arm_f_sam_tail_encoder.sbatch`; AutoPET 2D MAE final
checkpoint SHA256 `0b3571a3e79095686a0b12649735a298aaeb8a0b3bb93a47f3484439ac4a1ec6`.
Only the existing encoder transfer mapping is loaded; organ modules, spatial
decoder, reconstruction decoder and SAM tail retain random initialization.
No optimizer resume. 4 GPUs x 16 x accumulation3 = 192, 118800 updates,
lambda_seg0.01, historical top-k small-organ loss. Compared with scratch
Query/Reverse-Dot this changes BOTH initialization and decoder; use the MAE
encoder-only counterparts for architectural comparisons.

New opt-in head: `arm_f_sam_tail`. Implementation only; no GPU job submitted.
Existing E/F heads, losses, reconstruction and frozen job worktrees are unchanged.

Path: existing 20 organ tokens -> existing P16/P8/P4 TokenBlocks ->
prepend ONE shared learned mask token -> token self-attention ->
token-to-P4 attention -> token FFN -> P4-to-token attention ->
final token-to-updated-P4 attention -> mask token MLP ->
ordinary dynamic dot product with updated P4 -> existing output interpolation.

One terminal two-way block, four attention heads, usual width128. It retains
the F axial spatial PE (temporal + spatial for 3D); the input 21-token sequence
is a fixed sparse content reference within the tail, not a geometric prompt.
Post-residual LayerNorm follows each tail sublayer. The MLP is three Linear
layers with GELU between them. The readout has no cosine normalization or
sqrt(C) multiplier. There is no pixel FFN in the terminal reverse step.
Mask token and tail weights are shared across organs; organ conditioning comes
from existing slot-specific tokens/identity. No GT prompt, P2, soft mask,
IoU head, multimask output, new loss, or SAM/MAE weights are introduced.

This is a SAM-inspired tail, NOT original SAM: retained multiscale front end,
one rather than two blocks, existing axial PE and PyTorch MHA projections,
GELU, no SAM upscaler. Reference:
https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/transformer.py
and the adjacent mask_decoder.py.

CLI: `--slot_head_type arm_f_sam_tail --pixel_pe none --query_refinement none`.
Existing launcher accepts `ARM_F_HEAD_TYPE=arm_f_sam_tail`; for historical
small-organ comparisons explicitly set `BACKGROUND_REDUCTION=topk` (launcher
default is mean), `ARM_F_MICRO_BATCH=16 ARM_F_ACCUM_ITER=3` in 2D.
Set evaluation `EXPECTED_SLOT_HEAD_TYPE=arm_f_sam_tail`. New head requires its
own trained checkpoint; old-head checkpoints are not exact new-head resumes.
No submission until commit/push and immutable-runtime revision verification.

CPU tests cover 2D/3D shapes, operation order, BF16-input finite backward,
nonzero gradients through all tail parameters and organ inputs, token
conditioning, model/reconstruction integration, strict checkpoint roundtrip,
and exact pre-change outputs/state dictionaries of all six older F heads.
GPU/DDP, production-memory and accuracy remain unvalidated. 3D support here
uses the existing slice-wise spatial branch and flattened slab attention;
it is not evidence that this resolves previous 3D segmentation issues.
`````

<a id="merge-architecture"></a>

### docs/ARCHITECTURE.md

来源：`feature/orgslot @ f198a191fe9460410c0e8ed32c3392564b165ae1`。历史状态不作为当前状态。

`````text
# OrganSlot：研究主线与架构

## 研究主线

主要 idea 是器官专属权重的可插拔增量学习，以及解耦后重接的器官 tokens 如何辅助下游任务。
分割头 A–F 是检验下游读出与空间信息利用的实验，不是替代增量主线的六个独立贡献。
参数隔离、表示解耦、独立模块可组合是三个不同主张；联合 WORD 指标不能自动证明后两者。

## 共享与专属路径

共享 ViT -> z；每器官 Collector_s -> K tokens -> TGEnc_s -> AHER_s -> canvas_s。
选定 canvases 经 fusion -> 共享 reconstruction Transformer decoder -> 图像重建。
分割路径按 head 配置选择 canvas 或共享 pixel features，输出每器官 logit 与 calibration。
原始 OWT 的 joint token 路径与新的器官专属模块应明确区分，不能将继承组件称为新增贡献。
AHER 后重建路径的 spatial/temporal PE 和 Transformer 不等同于 direct head 的解码路径。

| Arm | head 配置 | 空间路径 / token 使用 |
|---|---|---|
| A | `linear` | AHER canvas -> 线性读出 |
| B | `multiscale_conv` | AHER canvas -> 多尺度卷积读出 |
| C | `query_dot` | ViT 空间分支；K tokens 汇总为单 query |
| D | `multi_query_dot` | 保留 K queries；点积 masks 经 log-mean-exp 汇总 |
| E | `arm_e_multiscale_query` | 输入 P4/P8 + ViT P16；单 query；当前 E 为 2D |
| E+CA | E + `query_refinement=cross_attn` | query 读取空间特征，再点积；不等于 F |
| F | `arm_f_attention` | K tokens 顺序读取 P16/P8/P4，再由 P4 反向读取 tokens |
| F 对照 | `arm_f_linear` | 相同多尺度 token refinement，改用线性点积读出 |

448 输入对应 P16=28、P8=56、P4=112；112 是边长的 1/4、面积的 1/16。
F 保留 20 个图像条件化 tokens；三层 cross-attention/self-attention/FFN 后反向读取，
P4 residual + FFN + 共享分类器生成 mask。解码器跨器官共享，不做 Hungarian matching。
F 3D stem 为逐 slice 2D 卷积，attention 读取整个 slab，不新增 temporal Conv3D。
因此 F 与旧 3D D 不只差一个反向 attention；应使用 F linear 匹配对照隔离该因素。
Attention 权重不是前景概率；热图不能单独证明语义解耦或因果作用。

## 增量学习的边界与待验证主张

原始 v0 包含 append_slot、可见标签隔离、冻结/hash 审计与 tiny 工程验证。
这些不替代正式多阶段增量实验。新增器官时必须列明共享 ViT、pixel decoder、
reconstruction decoder、旧 slots、calibration 的冻结/更新状态和旧数据访问权限。
仅冻结旧 slots 而更新共享路径，仍可能改变旧器官输出；独立 binary 输出不变也不保证
新增类别后 multiclass argmax 不受竞争影响。

论文最小验证：不同增量顺序的 Old/New/All 指标；独立模块直接重接 vs 微调；
普通分割头/固定类别 query vs 图像 organ tokens；有无重建监督；参数增长与旧数据需求。
这些是待验证问题，不在本次文档整理中标记为完成。

## 实现入口

- `OWT_models_orgslot.py`、`OrganSlotEmbed.py`：主模型与器官模块。
- `ArmFDecoder.py`：F 的多尺度双向解码与 linear 对照。
- `main_pretrain_orgslot_common8_a100.py`：训练配置入口。
- `engine_pretrain_orgslot_common8_a100.py`：训练、数值执行与 loss 集成。
- 各历史设计的公式、测试和路径在历史档案中按原文件名查询。
`````


## 文档整理原文备份（2026-10-13）

以下为整理前原文，包含过期状态；当前结论以 [结果台账](../RESULTS.md) 为准。
代码块中的相对路径以原分支/原文件为基准，不表示当前可执行命令。

- [README.md](#merge-readme)
- [docs/RESULTS.md](#merge-results)
- [docs/STATUS.md](#merge-status)
- [docs/CONFIGURATION.md](#merge-configuration)
- [docs/E_CROSSATTN_MAE.md](#merge-e-crossattn-mae)
- [docs/SAM_SOFT_PRIOR.md](#merge-sam-soft-prior)
- [docs/SAM_TAIL_HSAM.md](#merge-sam-tail-hsam)
- [docs/SAM_TAIL_ALL_SEG.md](#merge-sam-tail-all-seg)
- [docs/ABCD_RETIREMENT.md](#merge-abcd-retirement)
- [docs/CLEANUP_20260927.md](#merge-cleanup-20260927)
- [main:docs/RESULTS.md](#owt-results)
- [experiment/psem:docs/RESULTS.md](#psem-results)
- [experiment/lossbalance:docs/EXPERIMENTS.md](#lossbalance-experiments)

<a id="merge-readme"></a>

### README.md

来源：`feature/orgslot @ f198a191fe9460410c0e8ed32c3392564b165ae1`。原文完整保留。

`````text
# OrganSlot

开发中的 E cross / SAM-tail 编码器对照：[nnU-Net ResEnc / MedNeXt-B 2D](docs/CNN_BACKBONES.md)。
仅替换 backbone，CPU验证已通过；2026-10-13已提交两条AutoPET预训练及四条SAM-tail迁移训练依赖链，尚无分割成绩，不计入下方正式指标。

共享ViT＋器官独立tokens＋共享空间分支的分割/重建框架。后续仅保留以下三个2D主线配置。
架构图为按代码绘制的AI生成示意图，PNG供GitHub预览，PDF供下载；不是SAM/H-SAM官方复刻。
使用内置image_gen生成（工具未提供image2.5版本选择）；[重绘提示词记录](docs/figures/redraw_prompts.json)。
采用纯白、柔和配色、英文短标签的扁平矢量风格；文件本身仍是栅格图，PDF非可编辑矢量。

## 三个配置的正式指标

### 最新增量结果：SAM Stage2 私有交互（2026-10-13）

训练 `156538`、评估 `156539`（sifansong/XEC）均完成；checkpoint-802精确加载，
118800次Stage2更新，24病例／7315切片完整评估。官方池seed42，验证集选择阈值。

| 结果（Dice %，后处理） | 旧四类 | 新四类 | 八类平均 |
|---|---:|---:|---:|
| Stage1，固定0.5 | 86.78 | — | — |
| Stage2，固定0.5 | 86.78 | 76.39 | 81.58 |
| Stage1，校准 | 87.12 | — | — |
| **Stage2，校准** | **87.12** | **76.57** | **81.84** |
| PCDD论文4–4参考 | 84.09 | 82.75 | 83.42 |

旧四类逐器官二值指标保持不变，不等于互斥八类合并mask也零遗忘。
与PCDD报告值相比旧类+3.03、新类−6.18、全部−1.58个百分点；划分、预处理及评估
未完全对齐，不能当作严格同协议胜负。重建direct-post为75.28%，indirect-post仅6.10%，
后者存在大面积假阳性，尚待诊断。此结果不混入Offline排序。
完整八类指标、适配方式和证据路径见[增量实验记录](docs/INCREMENTAL_READOUT_ABLATION.md#formal-sam-stage2-results-2026-10-13)。

### 最新汇总：E cross＋MAE三次同协议随机划分（2026-10-08）

**主指标：固定0.5平均Dice 85.62%，验证集校准平均Dice 86.22%。**
使用官方池划分seed0/1/42，分别独立训练；每次96例训练、26例验证、24例测试，
模型seed均为0。以下为三次八类病例Dice均值的等权算术平均，不合并重复病例。
这是重复随机划分评估，不是三折交叉验证：seed0/1、0/42、1/42分别有13、10、11例
共同测试病例。旧划分及替换5病例的探索性评估不纳入主均值。

| E cross＋MAE encoder配置 | 训练Job | 测试病例／切片 | 阈值校准数据 | 固定0.5 | 校准后 |
|---|---|---|---|---:|---:|
| 官方池seed42 | 2998655，bolinren19/SIP | 24／7315 | 独立26例验证集 | 84.95 | 85.64 |
| 官方池seed0 | 3006796，bolinren19/SIP | 24／6854 | 独立26例验证集 | 85.20 | 85.57 |
| 官方池seed1 | 3006797，bolinren19/SIP | 24／7079 | 独立26例验证集 | 86.71 | 87.44 |
| **官方池三次划分平均（同协议）** | seed42/0/1等权 | 非互斥测试集 | 验证集校准 | **85.62** | **86.22** |

八类校准Dice：

| 器官 | 旧划分 | seed42 | seed0 | seed1 |
|---|---:|---:|---:|---:|
| 脾脏 | 94.74 | 94.07 | 94.71 | 94.92 |
| 右肾 | 94.62 | 93.72 | 94.22 | 95.21 |
| 左肾 | 94.46 | 93.93 | 94.52 | 94.69 |
| 胆囊 | 62.34 | 69.63 | 69.92 | 74.64 |
| 食管 | 78.00 | 71.72 | 70.80 | 75.40 |
| 胰腺 | 77.29 | 78.91 | 77.39 | 80.91 |
| 肝脏 | 95.29 | 95.72 | 95.67 | 96.17 |
| 胃 | 86.13 | 87.43 | 87.34 | 87.61 |

### SOTA数值汇总（按展示值降序，协议不同，不是公平胜负排名）

| 方法 | 校准Dice展示值 | 汇总口径 |
|---|---:|---|
| E cross＋MAE encoder | **86.22** | 官方池seed0/1/42三次同协议随机划分平均 |
| SAM soft prior＋aux | 85.55 | 旧划分单次，尚未完成相同三次划分 |
| SAM两阶段mask、缩小GT监督 | 85.23 | 旧划分单次，尚未完成相同三次划分 |

不能凭这张混合协议汇总表断言E优于SAM。严格同旧划分的E仍为85.36，
下方消融排序保留原单次值。PCDD Offline85.47是外部论文参考，未核实相同
病例划分，不能据86.22宣称同协议超过。正式方法比较应使用所有方法相同
划分集合的均值。

单列记录（不参与上述均值）：旧划分训练148833固定84.30%、校准85.36%，
采用训练集前6000切片校准；替换5个badcase的评估复用该checkpoint，
属于测试病例组成的探索性分析，不是独立重训，也不纳入主排序。

结果来源：旧划分路径见下方；seed42为仓库根
`artifacts/offline_eval_recovery_20261005/evaluation/ArmE_CrossAttn_MAEenc_Official96_2998655_heads_test/results.json`；
seed0/1为`.worktrees/offline_splits_d553bc7/Results/OrganSlotBank/evaluation/Common8/WORD_2D/ArmE_CrossAttn_MAEenc_Official96_{3006796,3006797}_heads_test/results.json`。
均读取`binary_post/selected_post.foreground_mean_case_dice_presence_mean`的未舍入值求平均。

### 旧划分同协议比较

**旧24例测试划分、6990张切片、checkpoint-802、MAE encoder-only**。Dice单位%，
按病例计算再按八个前景器官平均，固定/校准结果均后处理。校准使用训练集前6000切片，
不是测试GT调参，也不是独立验证集校准。单seed，不能据微小差距声称稳定优势。

| 主线 | 固定0.5 | 校准后 | 训练Job / 账号集群 |
|---|---:|---:|---|
| E cross-attention | 84.30 | 85.36 | 148833 / sifansong-XEC |
| SAM-tail＋soft prior＋aux | **84.86** | **85.55** | 150971 / sifansong-XEC |
| SAM-tail＋两阶段mask、缩小GT监督 | 84.41 | 85.23 | 2998657 / bolinren19-SIP |

| 器官（校准后） | E cross | SAM soft prior＋aux | SAM 缩小GT |
|---|---:|---:|---:|
| 脾脏 | 94.74 | 94.74 | **94.83** |
| 右肾 | 94.62 | 94.43 | **94.66** |
| 左肾 | 94.46 | 94.45 | **94.47** |
| 胆囊 | 62.34 | **64.15** | 61.59 |
| 食管 | 78.00 | 78.47 | **78.50** |
| 胰腺 | **77.29** | 76.46 | 76.73 |
| 肝脏 | 95.29 | 95.30 | **95.32** |
| 胃 | 86.13 | **86.43** | 85.77 |
| 平均 | 85.36 | **85.55** | 85.23 |

soft prior＋aux相对E主要改善胆囊，胰腺并未全面改善。核验来源为以下目录的
`results.json`（`complete_split=true`，`binary_post/selected_post`）：

- E：`sifansong/XEC:/gpfs/work/aac/sifansong/worktrees/e_cross_mae_9791e87/Results/OrganSlotBank/evaluation/Common8/WORD_2D/ArmE_CrossAttn_MAEenc_148833_ckpt802_heads_test/`
- soft prior＋aux：`sifansong/XEC:/gpfs/work/aac/sifansong/worktrees/sam_soft_aux_2b5aa2f/Results/OrganSlotBank/evaluation/Common8/WORD_2D/ArmF_2D_150971_ckpt802_heads_test/`
- 缩小GT：`bolinren19/SIP:.worktrees/quota_recovery_b5a4792/Results/OrganSlotBank/evaluation/Common8/WORD_2D/ArmF_2D_2998657_ckpt802_heads_test/`

**新划分单列**：官方池96训练/26验证/24测试（seed42），E cross＋MAE固定84.95%、
验证集校准85.64%，7315测试切片，训练2998655。不混入上表。
PCDD Offline85.47%、4–4增量83.42%只是不同协议的外部参考，不是同病例、同预处理的复现比较。
完整历史见[RESULTS](docs/RESULTS.md)。

## 三张架构图与差异

以下为Offline联合训练结构。每个器官独立Collector/TGEnc/AHER；共享ViT、浅层CNN、
pixel decoder、读出和重建decoder。448输入下P16/P8/P4边长28/56/112；P4边长1/4、面积1/16。
SAM图重点展开分割decoder，共用的ViT、Collector和重建路径省略处与E一致，未删除模块。

### 1. E cross-attention

![E cross-attention架构](docs/figures/architecture_e_cross.png)

[下载PDF](docs/figures/architecture_e_cross.pdf)

`slot_head_type=arm_e_multiscale_query`，`query_refinement=cross_attn`，`hsam_supervision=none`。
20 tokens经query_norm、mean pool、query_proj及slot identity变成1个query；读取P4，经FFN后
与原P4点积生成mask。没有20-token逐尺度交互，也没有SAM-tail。
代码：[OrganSlotEmbed.py](OrganSlotEmbed.py)的`MultiScalePixelQueryDecoder2D`。

启动配置（先执行下方公共环境准备，再用公共`sbatch`命令提交）：

```bash
export HSAM_SUPERVISION=none
TRAIN_SCRIPT=slurm/orgslot/train/arm_e_crossattn_mae_encoder.sbatch
```

### 2. SAM-tail＋soft prior＋aux

![SAM soft prior与辅助监督架构](docs/figures/architecture_sam_soft_prior_aux.png)

[下载PDF](docs/figures/architecture_sam_soft_prior_aux.pdf)

`slot_head_type=arm_f_sam_tail`，`query_refinement=none`，`hsam_supervision=soft_prior_aux`。
20 tokens依次读取P16→P8→P4。每次读取前，用当前tokens归一化均值与该尺度pixels
余弦读出粗logits，`log(0.2+0.8*sigmoid(logits.detach()))`作为attention-score bias。
三张粗logits上采样至完整GT大小，辅助loss取均值；路由detach，aux保留梯度。
之后SAM-tail用mask token、双向交互、最终回读及MLP动态点积生成最终mask。
**此模式不会将粗mask传入tail内部作为prior**。
分割目标：`0.01 × (L_final + 0.25 × mean(L_P16,L_P8,L_P4))`。

启动配置：

```bash
export HSAM_SUPERVISION=soft_prior_aux
TRAIN_SCRIPT=slurm/orgslot/train/sam_tail_soft_prior.sbatch
```

### 3. SAM-tail＋两阶段mask、缩小GT监督

![SAM两阶段mask与缩小GT监督架构](docs/figures/architecture_sam_downsample_gt.png)

[下载PDF](docs/figures/architecture_sam_downsample_gt.pdf) · [三图合并PDF](docs/figures/organslot_three_architectures.pdf)

`slot_head_type=arm_f_sam_tail`，`query_refinement=none`，`hsam_supervision=downsample_gt`。
20 tokens完成不带mask引导的P16→P8→P4交互后，先与P4生成112×112粗mask。
GT最近邻缩小后监督粗mask；其sigmoid概率不detach，门控SAM-tail两次token读取中的
投影V，不重新归一化attention。最终mask上采样并使用完整GT。
分割目标：`0.01 × [w(e)L_final + (1-w(e))L_coarse]`，`w(e)=0.6^(0.990^e)`。
监控`hsam_lost_positive_slices`，检查缩小GT是否丢失小阳性区域。
这里“两阶段”是**一次forward中的粗到精解码**，不是增量训练Stage1/Stage2。
实现：[ArmFDecoder.py](ArmFDecoder.py)、[hsam_supervision.py](hsam_supervision.py)。

启动配置：

```bash
export HSAM_SUPERVISION=downsample_gt
TRAIN_SCRIPT=slurm/orgslot/train/arm_f_sam_tail_hsam.sbatch
```

| 差异 | E cross | SAM soft prior＋aux | SAM 缩小GT |
|---|---|---|---|
| 交互tokens | pool后的1个query | 20个tokens | 20个tokens |
| 交互尺度 | P4 | P16→P8→P4 | P16→P8→P4 |
| mask引导位置 | 无 | 三尺度attention logits | tail两次token读取的V |
| 粗监督 | 无 | 三尺度logits放大后监督 | P4大小的缩小GT监督 |
| 最终读出 | query · 原P4 | mask-token MLP · 更新后P4 | mask-token MLP · 更新后P4 |

## 公共配置与启动

| 配置项 | 2D Offline主线 |
|---|---|
| 数据 / spacing / ROI | WORD八类＋背景slot；0.7×0.7×2.0；448输入、384 ROI、概率0.2 |
| tokens / channels | 每slot20 tokens；pixel channels128 |
| 初始化 / seed | AutoPET MAE encoder-only，其它随机初始化；seed0 |
| 更新 / warmup | 118800 / 5940 |
| batch | 16/卡 × 4GPU × 累积3 = 192 |
| 优化 | AdamW，LR7.5e-5，weight decay0.05，clip_grad1，BF16及局部FP32 |
| 分割监督 | retained slots；small-organ loss；lambda_seg0.01、lambda_bg_seg0 |
| 阳性切片loss | Tversky(FP0.3/FN0.7)＋0.5 BalancedFocal；alpha0.75、gamma2 |
| 阴性切片loss | top2% hard-negative BCE，权重0.1 |
| 重建 | 保留原有重建与感知目标，三种head均不删除重建分支 |

以下为 **sifansong/XEC旧划分复现示例**。E脚本固定此账号路径，不能原样跨账号使用；
每个账号必须核对Python、WORD、ROI、LPIPS和MAE文件。先准备最新干净的固定runtime。

```bash
export EXPERIMENT_WORKDIR=/absolute/path/to/pinned/runtime
bash "$EXPERIMENT_WORKDIR/tools/check_experiment_revision.sh" "$EXPERIMENT_WORKDIR"
export ARM_F_TARGET=sifan-xec
export ARM_F_COMMIT=$(git -C "$EXPERIMENT_WORKDIR" rev-parse HEAD)
export MAE_INIT_CHECKPOINT=/gpfs/work/aac/sifansong/Results/AutoPET_MAE/MAE_LA6D8_224_M75_U100K_seed0/checkpoint-final.pth
mkdir -p "$EXPERIMENT_WORKDIR/logs"

# 三选一：E cross
export HSAM_SUPERVISION=none
TRAIN_SCRIPT=slurm/orgslot/train/arm_e_crossattn_mae_encoder.sbatch

# 或：SAM soft prior + aux
# export HSAM_SUPERVISION=soft_prior_aux
# TRAIN_SCRIPT=slurm/orgslot/train/sam_tail_soft_prior.sbatch

# 或：SAM两阶段缩小GT
# export HSAM_SUPERVISION=downsample_gt
# TRAIN_SCRIPT=slurm/orgslot/train/arm_f_sam_tail_hsam.sbatch

sbatch --parsable --account=sifansong --partition=gpua8001t --qos=8gpus \
  --nodes=1 --ntasks=1 --gres=gpu:a800:4 --cpus-per-task=20 --mem=192G \
  --time=5-00:00:00 --export=ALL --chdir="$EXPERIMENT_WORKDIR" \
  --output="$EXPERIMENT_WORKDIR/logs/%j.out" --error="$EXPERIMENT_WORKDIR/logs/%j.err" \
  "$EXPERIMENT_WORKDIR/$TRAIN_SCRIPT"
```

提交前记录SHA、runtime、账号和manifest；远端archive须核验校验和。不得更新运行快照。
新96/26/24划分使用[独立入口](docs/CONFIGURATION.md#word-official96-protocol)，不混用上述旧划分。

### 推理

[统一head＋recon脚本](slurm/orgslot/eval/arm_e_mae_unified.sbatch)需要设置
`EVAL_WORKDIR`、`RUN_NAME`、`ACCOUNT_ROOT`、`DATA`、`PYTHON`、`EXPECTED_SLOT_HEAD_TYPE`。
E运行名`ArmE_CrossAttn_MAEenc_JOB`；SAM运行名`ArmF_2D_JOB`。
SAM可用[sam_tail_guided.sbatch](slurm/orgslot/eval/sam_tail_guided.sbatch)核验HSAM配置。
使用1GPU并显式指定评估时限及`afterok:训练JOB`。旧脚本固定checkpoint802、训练6000切片校准，
输出固定/校准head和recon direct/indirect；recon阈值0.02。新划分使用验证26例校准入口。

## 消融实验记录

旧24例协议的主要已核验消融，按校准后Dice降序；非全部历史3D/P2的穷尽列表。
除注明scratch或encoder+decoder外，均MAE encoder-only。

| 消融配置 | 固定0.5 | 校准后 | 训练Job |
|---|---:|---:|---|
| **SAM soft prior＋aux** | 84.86 | **85.55** | 150971 |
| **E cross** | 84.30 | **85.36** | 148833 |
| SAM粗logits放大监督 | 84.26 | 85.30 | 150173 |
| **SAM缩小GT监督** | 84.41 | **85.23** | 2998657 |
| E无cross | 84.58 | 84.99 | 135283 |
| SAM-tail无引导 | 83.99 | 84.94 | 2984833 |
| 原F reverse＋Linear | 84.21 | 84.93 | 137867 |
| SAM M0～M3硬mask引导 | 84.68 | 84.84 | 150175 |
| SAM全部8类分割监督 | 84.27 | 84.81 | 149295 |
| E无cross＋MAE encoder+decoder | 84.48 | 84.76 | 134915 |
| SAM soft prior无aux | 84.28 | 84.68 | 150615 |
| 原F＋MAE encoder+decoder | 83.88 | 84.59 | 137880 |
| E无cross scratch | 83.53 | 83.72 | 2922770 |

- soft prior＋aux优先对比无aux、SAM无引导，不能将增益全部归于引导机制。
- 缩小GT与放大logits比较监督分辨率；E cross对比E无cross比较query读取。
- SAM与E同时改变多个模块，不是单因素消融。历史microbatch/TGR等差异须查原记录。
- 3D、P2、早期A/B/C/D详见[历史记录](docs/QUERY_READOUT_ABLATION.md)和[RESULTS](docs/RESULTS.md)。

## 增量实验边界

新4→4任务每阶段118800更新，总237600；旧四类为脾/右肾/左肾/胆囊，新四类为食管/胰腺/肝/胃。
E支持共享query微调和新旧分组query；SAM soft_prior_aux支持slot私有交互、共享tail冻结。
**缩小GT配置尚无对应slot_private增量实现**。见[增量设计](docs/INCREMENTAL_READOUT_ABLATION.md)。
以下旧通知是历史快照，不代表当前实时队列。

<details>
<summary>历史更新与仓库导航（保留记录）</summary>

## 当前保留的架构入口（2026-10-06）

后续主线收敛为 **E** 与 **SAM-tail** 两个系列；统一配置映射与路径见
[架构导航](docs/QUERY_READOUT_ABLATION.md#当前架构导航2026-10-06)。

| 系列 | 配置 | 用途 |
|---|---|---|
| E cross-attention | `arm_e_multiscale_query`，`query_refinement=cross_attn` | E主线 |
| SAM-tail soft prior＋aux | `arm_f_sam_tail`，`hsam_supervision=soft_prior_aux` | 多尺度软引导和辅助监督 |
| SAM-tail 两阶段mask预测 | `arm_f_sam_tail`，`hsam_supervision=downsample_gt` | 粗mask引导tail、最终精mask；缩小GT监督 |

仅上述三个配置作为后续主线。E无cross-attention、SAM-tail无引导/无aux/上采样监督，
以及历史F Linear/Query-Dot/Reverse-Dot/P2、硬mask等
保留加载与复现兼容，但不作为后续主线入口。此整理不删除权重、结果或代码，
不修改任何在运行/排队的快照。soft prior与两阶段tail引导目前是不同模式，不能同时选择。

2026-10-06 预算更新：新4–4实验 Stage1、Stage2各118800 updates，warmup各5940，
有效batch192不变；总237600，不再与Offline等总预算。Stage2要求完整预算Stage1。
历史59400/阶段结果和评估兼容保留，旧运行快照不修改；详见[预算说明](docs/INCREMENTAL_READOUT_ABLATION.md#full-per-stage-budget-revision-2026-10-06)。

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

</details>
`````

<a id="merge-results"></a>

### docs/RESULTS.md

来源：`feature/orgslot @ f198a191fe9460410c0e8ed32c3392564b165ae1`。原文完整保留。

`````text
# 已记录结果与证据

2026-10-08：E cross＋MAE主指标及八类明细已统一记录到
[README最新汇总](../README.md#三个配置的正式指标)。官方池seed0/1/42三次
同协议随机划分平均固定85.62、校准86.22，模型seed均为0。
这是重复随机划分，不是三折交叉验证；旧划分与替换5病例探索性评估
单列，不纳入主均值。历史同划分消融保留。seed0/1的indirect reconstruction异常
（7.19/7.10）仍待诊断，不作为有效重建SOTA；head完整结果单独记录。

## 4–4统一报告：Offline / Stage1 / Stage2

旧四类为脾、右肾、左肾、胆囊；新四类为食管、胰腺、肝、胃。
每个阶段单列旧四类、新四类、八类病例Dice均值及八个器官明细，背景不计入均值。
Stage1仅有旧四类输出，新四类和八类整体记为NA（尚未学习、无输出头），不能把背景当新类，不能填造一个实测0分。
如另做“未见类别强制空预测”的诊断，必须单列，不能冒充PCDD主表口径。

| 方法/阶段 | 旧四类 | 新四类 | 八类整体 |
|---|---:|---:|---:|
| PCDD论文Offline参考 | 86.18 | 84.77 | 85.47 |
| PCDD论文4–4最终阶段参考 | 84.09 | 82.75 | 83.42 |
| 本项目历史Offline E cross+MAE，旧划分fixed-post | 85.22 | 83.37 | 84.30 |
| 本项目历史Offline E cross+MAE，旧划分calibrated-post | 86.54 | 84.18 | 85.36 |
| 本项目同划分Offline | 待完成 | 待完成 | 待完成 |
| Stage1固定0.5 post | 84.89 | NA | NA |
| Stage1验证集校准post | 85.48 | NA | NA |
| Stage2 | 待完成 | 待完成 | 待完成 |

Stage1证据：sifansong/XEC152957，完整24病例7315切片，checkpoint401精确加载；
`/gpfs/work/aac/sifansong/artifacts/inc44_stage1_eval_20261003/evaluation_90min/test/results.json`。
Offline为官方池96/26/24划分E cross-attention+MAE训练2998655及评估2998656；不能用历史旧划分模型代替。
Stage2训练2998660，最终评估尚未提交（2026-10-03核验）。

2026-10-04：Stage2完成59400更新、checkpoint401，最终freeze_audit通过。
新增`slurm/orgslot/eval/incremental44_stage2.sbatch`，验证集26例校准后测试24例7315切片，
报告fixed/calibrated、raw/post及Old/New/All；与Stage1固定0.5比较遗忘量。
提交目标为sifansong/XEC单RTX4090，沿用已校验Stage1数据副本，运行输出独立保存。
checkpoint选择器检查最终日志、优化器步数、有限权重、配置与冻结审计；GPU状态以提交记录为准。

历史Offline是八类联合训练的Arm E cross-attention+MAE encoder，训练148833、checkpoint802。
2026-10-03核验完整24病例/6990切片、exact加载；校准使用训练集6000切片，阈值固定后测试。
结果位于sifansong/XEC：
`/gpfs/work/aac/sifansong/worktrees/e_cross_mae_9791e87/Results/OrganSlotBank/evaluation/Common8/WORD_2D/ArmE_CrossAttn_MAEenc_148833_ckpt802_heads_test/results.json`。
此行是旧划分参考，不能与当前Stage1/Stage2新24例计算遗忘量或严格的Offline差距。

| 历史Offline输出 | 脾 | 右肾 | 左肾 | 胆囊 | 食管 | 胰腺 | 肝 | 胃 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed0.5 post | 94.22 | 93.60 | 93.70 | 59.39 | 75.05 | 76.93 | 95.29 | 86.21 |
| calibrated post | 94.74 | 94.62 | 94.46 | 62.34 | 78.00 | 77.29 | 95.29 | 86.13 |

统一汇总入口：
```bash
python -m tools.report_incremental44 --offline OFFLINE_TEST/results.json \
  --stage1 STAGE1_TEST/results.json --stage2 STAGE2_TEST/results.json
```
各参数可在结果未齐时省略。工具核验相同24病例、每病例切片数、评估设置和精确加载；
分别输出raw/post和fixed/calibrated。遗忘量=Stage1旧类fixed-post减Stage2旧类fixed-post（百分点）；
逐阶段各自校准的差值不作为主遗忘量。结果均使用完整病例体积统计，而非切片Dice平均。

与PCDD一致的目标是4–4类别顺序、分阶段标签可见性以及Old/New/All统计。
当前使用同一96病例池分阶段暴露不同标签，不能据此宣称复现了PCDD的阶段间病例分配；
其论文问题定义提到旧阶段数据不可访问，而公开实验细节未给出可核对的阶段病例清单。
PCDD附录使用120例随机96/24、1×1×3mm、3D Swin UNETR；本项目使用官方池96/26/24、0.7×0.7×2mm、2D模型。
这些差异必须保留，不能写“完全同协议复现”。来源：
[论文Table2](https://ojs.aaai.org/index.php/AAAI/article/download/38406/42368)、
[附录C](https://arxiv.org/html/2511.07749v1#A3)、
[官方仓库](https://github.com/shengqianzhu/PCDD)（本次检查仍为Coming soon）。

## 2026-10-03 增量4–4 Stage1评估记录

Stage1训练2998659已完成59400更新，最终checkpoint-401；Stage2训练2998660已自动加载该模型。
Stage1独立评估此前缺失，现补slurm/orgslot/eval/incremental44_stage1.sbatch。
模型含5个slot：背景、脾、右肾、左肾、胆囊；主指标报告旧4类病例Dice均值。
背景head未被直接监督，背景诊断使用旧4类预测并集的补集；GT背景为原始标签0、5、6、7、8。
固定官方池96/26/24划分，完整验证集选阈值；测试同时报告固定0.5与验证集校准、raw/post。
背景Dice单列，禁止纳入旧4类前景均值。逐病例CSV另含precision、recall、体积比。
遗忘量须比较同一测试集Stage1/Stage2旧4类固定0.5结果；stage-specific校准另列。
输出：artifacts/incremental44_20261001/stage1_evaluation/{validation,test}。
提交前8项CPU测试通过，最终五slot模型CPU精确加载检查完成后提交；GPU结果以正式任务为准。

## 2026-09-18：F预训练与3D评估更新

测试集均24病例/6990切片；下表主指标为病例级前景Dice均值（%）。
校准只用训练集前6000切片；不使用测试集选阈值。2D checkpoint802、3D checkpoint199。

| 实验 | 固定0.5 post | 校准post | Direct recon post | 备注 |
|---|---:|---:|---:|---|
| F 2D scratch | 82.91 | 未补 | 未补 | 2929821/2929822 |
| F 2D MAE encoder | 84.21 | 84.93 | 79.81 | 137867/137868，完整结束 |
| F 2D MAE encoder+recon decoder | 83.88 | 84.59 | 79.36 | 137880/137881，完整结束 |
| F 3D top-k | 83.29 | 84.46 | 78.64 | 140443–140445统一协议 |

F encoder比scratch固定阈值提高约1.31个百分点，但scratch microbatch8、MAE16，
有效batch均192不消除small-organ分组归约差异。F encdec相比encoder固定/校准均低约0.34，
主要落在胆囊、胰腺；单seed不证明decoder迁移普遍有害。既有E encoder校准84.99，
F encoder84.93仅差0.06，不作稳定胜负或外部SOTA宣称。

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

## 2026-09-14 最新：Arm E MAE三臂

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

## WORD head 已记录均值（来源摘要，不作严格统一排名）

| 方法 | 八类 mean Dice | 证据与限制 |
|---|---:|---|
| D multi-query | 82.12 | 原分支 README；阈值协议和八类向量需从正式结果重新核验 |
| C single-query | 82.01 | 原分支 README；阈值协议和八类向量需从正式结果重新核验 |
| B MAE encoder+decoder | 81.09 | MAE 专项已核验记录 |
| B MAE encoder-only | 80.60 | MAE 专项已核验记录 |
| B scratch | 79.89 | 下方完整八类表 |

这是已找到来源的摘要，不补写 Arm A/E/F 或新3D未核验数字，不代表全量最新排行。
Arm B 三行使用 fixed0.5；不将 C/D 摘要默认为相同阈值协议。
下方历史表中的“排名/最优”限定于该表候选，不覆盖后来 C/D。

## 排名口径

本表只比较 WORD 同一完整测试集上的 OrganSlot 实验：24 个病例、6990 张切片，
八类顺序固定为脾脏、右肾、左肾、胆囊、食管、胰腺、肝脏、胃。指标为
case-level Dice presence mean，表中乘以 100。

主排名要求同一个 checkpoint、同一种输出和一套预先固定的后处理，不允许按测试集
为不同器官临时挑选不同方法。小器官均值固定为胆囊、食管、胰腺三类的算术平均。
这里的“SOTA”仅指本项目内已完成且协议匹配的实验，不代表公开 WORD benchmark
或 PCDD 的外部 SOTA。

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

Head 数字沿用此前已核验汇报；本次新核验的是 recon JSON。
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


## 跨路线边界

PSEM/AutoPET 四类、WORD224 与 WORD070 不混排；见 experiment/psem 的 RESULTS。
早期3D异常与后续slice/PE实验需在正式结果中分开。当前摘要不将“诊断运行完成”
写成“3D问题已解决”。全量增量学习 Old/New/All 表仍需独立验证，不由联合head分数推导。
`````

<a id="merge-status"></a>

### docs/STATUS.md

来源：`feature/orgslot @ f198a191fe9460410c0e8ed32c3392564b165ae1`。原文完整保留。

`````text
# 当前实验交接

## 2026-09-21 16:53 Collector attention alignment 已提交

目的：以2D Arm E scratch为基准，仅新增Collector器官外注意力软惩罚，验证器官采集定位是否改善；不增加网络模块，不使用MAE、不续训，默认lambda=0保持旧实验。

| 实验 | 账号/集群 | Job | 资源与时限 | 提交后核验 |
|---|---|---|---|---|
| Arm E scratch + collector loss 0.01 | sifansong / XEC，account=sifansong，QoS=4gpus | 143780 | 4×A800、20CPU、192GB、7天 | PENDING (Priority)；无GPU验证 |
| 同协议阈值校准 + head测试 + reconstruction | 同上 | 143781 | 1×A800、10CPU、128GB、7天 | PENDING (Dependency)，afterok:143780 |

- 固定代码：`aff9f696d50fb2dcd0edfb47bc6312e8f9caae3e`，已在提交前推送origin/feature/orgslot并通过revision guard。
- 本地固定worktree：`/gpfs/work/aac/bolinren19/OD_OWT/.worktrees/collector_align_aff9f69`。
- XEC固定源码快照：`/gpfs/work/aac/sifansong/worktrees/collector_align_aff9f69`。此目录不pull、不覆盖；tar无.git时训练provenance允许unavailable，真实来源以本条full commit、源码checksum、Slurm SOURCE_COMMIT和压缩包校验为准。
- 压缩包：`/gpfs/work/aac/sifansong/collector_align_aff9f69.tar`，SHA256=`d197f72fb8f53367e4824580c46978d3e1ba3ac3685e8fbff93db45f3237475d`，本地/远端一致。
- train/eval入口：`slurm/orgslot/train/arm_e_collector_align.sbatch`、`slurm/orgslot/eval/arm_e_collector_align.sbatch`。
- 训练输出：运行目录下`Results/OrganSlotBank/Common8/WORD_2D/ArmE_CollectorAlign001_scratch_143780`；Slurm日志：`/gpfs/work/aac/sifansong/logs/collector_align_aff9f69/`。
- 2D每卡16×4卡×累积3=192；118800 updates，预计最终checkpoint-802；WORD07072、ROI20、seed0、small-organ lambda_seg0.01不变。
- 对照：E scratch训练系列2965273 / 评估2965274。对照历史中途换过microbatch，新实验从头16；不是逐更新完全相同随机轨迹。其余已有任务未修改。
- 本地与XEC各6项新增CPU测试通过，包括两进程Gloo；数据身份28586训练切片/6990测试切片、ROI core SHA256均匹配基线。GPU正式运行尚未开始，不能宣称GPU验证通过。
- 下一次先查143780是否启动、resolved_config是否scratch/batch16/lambda_collect0.01、首epoch各器官collector count/foreground_mass及有限梯度、weighted_collect=0.01×raw；再检查checkpoint与143781依赖。最终比较八类head fixed/calibrated和recon，不只看attention集中度。


## 2026-09-18 13:53 已核验快照（当前入口）

新六组：1组运行、5组排队；未发现失败。此次文档整理未再次刷新队列。
每组训练4×A800/20CPU/192GB；scratch、seed0、top-k small-organ loss、ROI20、
118800 updates。2D每卡8/累积6/有效192；3D每卡6/累积2/有效48 slabs。

| 目的 | 账号/集群 | 训练 | 校准/Head/recon评估 | QoS/时限 | 状态 |
|---|---|---|---|---|---|
| 2D E单query点积基线 | bolinren19/SIP | 2955007 | 2955008（统一三阶段） | 4a800/7天 | Priority |
| 2D F Query-Dot：细化tokens、不反向读取 | bolinren19/SIP | 2955009 | 2955010（统一三阶段） | 8a800/7天 | Priority |
| 2D F Reverse-Dot：反向读取后点积 | bolinren19/SIP | 2955011 | 2955012（统一三阶段） | 8a800/7天 | Priority |
| 3D E式点积基线 | sifansong/XEC | 140585 | 140586 / 140587 / 140588 | 4gpus/7天 | RUNNING epoch4，checkpoint-0存在 |
| 3D F Query-Dot | sifansong/XEC | 140591 | 140592 / 140593 / 140594 | 8gpus/5天 | Priority |
| 3D F Reverse-Dot | antengcai23/XEC | 140595 | 140596 / 140597 / 140598 | 4gpus/7天 | Resources |

3D基线最新抽查：loss≈0.2511，seg≈0.7667，weighted_seg≈0.0077，ROI≈20.3%，
日志torch峰值allocated33.57GiB，overflow0、错误日志空。非完整稳定性证明；
checkpoint-0仅确认文件存在，未重新加载。其余五组尚无GPU运行验证。

固定运行源（不要在排队/运行目录pull）：

- 2D：bolinren19/SIP `/gpfs/work/aac/bolinren19/OD_OWT/.worktrees/arm_f_linear_2d`，
  `5f2d1cfb7fb20d499c79cb8e901d7279718cd8bc`。目录旧名linear不代表当前提交了arm_f_linear。
- 3D：各账号XEC `/gpfs/work/aac/<账号>/worktrees/dot3d_ac8cf99`，
  `ac8cf992cdac30c9d7bed33cc5e2f8b8ad4d8ecd`。
- 训练入口 `slurm/orgslot/train/arm_f.sbatch`；2D评估 `arm_e_mae_unified.sbatch`；
  3D评估 `unified_3d_protocol.sbatch`。3D校准/recon依赖训练afterok，Head依赖校准afterok。
- 同seed不保证共有模块初始权重逐张量相同，本轮未配平；跨架构因果解释需保留此限制。

已完成：F scratch 2929821/2929822；F encoder 137867/137868；F encdec137880/137881；
3D topk137410及140443–140445统一评估；3D mean137412/137413（旧post3d协议）；
3D MAE2951384完成100000updates，日志报告checkpoint验证通过，尚无其下游迁移任务。
最新指标统一见RESULTS.md，不从训练校准分数推断测试成绩。

下次：优先检查140585的checkpoint10；五组排队任务启动后核验head、batch、topk、
梯度与首epoch；140591五天时限余量偏紧；3D mean若需校准比较须补统一协议。
F2D scratch的校准/recon仍为缺口。Indirect异常需独立诊断，不能概括为recon全正常。

## 历史快照：2026-09-14（以下状态已过期，仅保留背景）

日志/结果核验时间：2026-09-14 14:05 Asia/Shanghai。之后队列复查仍为4组训练运行、2组训练排队。
epoch为快照，不是实时页面；估时不含评估和排队。结果统一见[RESULTS.md](RESULTS.md)。

| 实验目的 | 账号/集群 | 训练 → 评估 | 资源/QoS | epoch / checkpoint | 状态 |
|---|---|---|---|---|---|
| E cross-attention：显式query与空间交互 | bolinren19/SIP | 2923562 → 2923565 | 4A800/24CPU，8a800 | 441/802，425 | RUNNING |
| F 2D scratch：双向交互 | bolinren19/SIP | 2929821 → 2929822 | 4A800/20CPU，4a800 | 222/802，220 | RUNNING |
| F 3D topk：从30续训 | sifansong/XEC | 137410 → 137411 | 4A800/20CPU/192GB，4gpus，7天 | 61/199，60 | RUNNING |
| F 3D mean：背景全平均对照 | antengcai23/XEC | 137412 → 137413 | 4A800/20CPU/192GB，8gpus，5天 | 14/199，10 | RUNNING |
| F 2D MAE encoder-only | sifansong/XEC | 137867 → 137868 | 4A800/24CPU/192GB，8gpus，5天 | — | PENDING Priority |
| F 2D MAE encoder+重建decoder | antengcai23/XEC | 137880 → 137881 | 4A800/24CPU/192GB，4gpus，7天 | — | PENDING Priority |

评估均等待对应训练；未新增任务或修改依赖。sifansong/SIP、antengcai23/SIP无活动任务。
E scratch 2922770/2922772、E MAE encoder 135283/135284、encdec 134915/134916均已完成。
E两条续训跨过旧541故障点并完成802；不能据此声称所有NCCL底层问题已消除。

## 配置、运行源与公平性

未来四卡2D每卡16×累积3=有效192；3D每卡6×累积2=有效48 slabs。
运行中F2D仍为8×累积6，新F MAE为16×累积3；microbatch不同必须披露。
small-organ loss分组归约意味着有效batch相同不保证梯度完全相同。

- F3D两账号runtime：各自 /gpfs/work/aac/<用户名>/worktrees/arm_f_batch6_20260913；
  固定代码2697b05642d527bfd137debc0f9ed175e9ae345f。
- F MAE encoder：/gpfs/work/aac/sifansong/worktrees/arm_f_mae_encoder_df7ed39；
  df7ed39cb8529afebb8f1dd982eb8daf382c8c6d。
- F MAE encdec：/gpfs/work/aac/antengcai23/worktrees/arm_f_mae_encdec_20260913；
  5cee1e95de7e9c758df0e3cb3fc569cdc0fa8d57。旧137869/137870已取消。
- F MAE入口 slurm/orgslot/train/arm_f_mae.sbatch；评估 arm_f_mae_unified.sbatch。
  7项CPU加载测试和数据/hash门禁通过，正式GPU加载及batch16显存仍待启动。
- E encoder runtime：/gpfs/work/aac/sifansong/worktrees/arme_mae_encoder_runtime_f70b831。
- E encdec runtime：/gpfs/work/aac/sifansong/worktrees/arme_mae_runtime_43cad30。
- 仅开发目录同步Git；运行与排队快照保持固定。各账号数据路径不得互换。

## 稳定性与估时

最新epoch平均总loss/seg：E CA 0.0809/0.1266；F2D 0.1029/0.1611；
F3D topk 0.1135/0.1920；mean 0.1744/0.3603，均有限，但只是日志抽查。
日志打印进程的torch峰值allocated显存分别16.84、23.98、57.88、57.90 GiB；
不是实时nvidia-smi，不是全部rank峰值，也不含全部缓存/通信开销。
实时显存查询未成功（SIP节点SSH拒绝；XEC工具审批服务失败）。

按14:05吞吐外推：E CA剩32小时；F2D约81小时；F3D topk约73–76小时；
mean约93小时，需预留10–20%波动。mean的120小时时限余量较紧。
F MAE尚排队，batch16没有实测，不承诺结束日期。

## 下次审核

1. E CA checkpoint450/500、F2D checkpoint230后稳定性。
2. F3D topk checkpoint70、mean checkpoint20；最终看Dice/P/R/volume ratio/per-z，不用低loss证明3D已修复。
3. F MAE启动后核验初始化scope、每卡16、有效192、GPU峰值与真实加载报告。
4. E encoder-only recon Direct-post81.17已收齐；Indirect-post7.53异常，需独立诊断。
5. 3D D slice-wise与spatial既有均值21.13/30.25，仍过分割；temporal/ST旧链已取消，不再列作待跑。
6. PCDD Offline85.47固定列入对比；未匹配协议，不能宣称增量或外部SOTA。
# SAM-tail + MAE encoder-only (2026-09-22)

- Purpose: retain P16/P8/P4 refinement and test mask-token two-way tail.
- bolinren19 / SIP, account sifansong, QoS4a800.
- Training 2984833: 4 A800,20CPU,192GB,7days; PENDING Priority at submission.
- Unified evaluation 2984834: afterok:2984833,1 A800,10CPU,128GB,7days;
  train6000 threshold calibration + test6990 fixed/calibrated head + recon.
- Runtime: /gpfs/work/aac/bolinren19/OD_OWT/.worktrees/sam_tail_encoder_b3794a6
- Pinned code: b3794a6d9fca789fc47044f1929cadb27a0abe2b (revision guard passed).
- Train script: slurm/orgslot/train/arm_f_sam_tail_encoder.sbatch.
- Eval script: slurm/orgslot/eval/arm_e_mae_unified.sbatch, expected arm_f_sam_tail.
- AutoPET MAE encoder-only, NOT scratch or encoder+decoder; checkpoint SHA
  0b3571a3e79095686a0b12649735a298aaeb8a0b3bb93a47f3484439ac4a1ec6.
- 2D batch16 x4 xaccum3=192; updates118800, ROI20, lambda_seg0.01, topk loss.
- CPU: previous14 tests + SAM-tail MAE transfer test passed; data identity passed.
  GPU/DDP/pretrained production model load report pending job start.
- Next: verify mae_initial_checkpoint_load.json, resolved configuration, first
  finite epoch and checkpoint. Initialization differs from scratch ablations.
## 2026-09-30：WORD official-pool 96/26/24 重跑

配置：Arm E cross-attention＋AutoPET MAE encoder-only（旧划分校准85.358%的配置），不是续训旧WORD模型。固定split seed42：官方Tr100取96，官方Val20＋Ts30取24测试（8Val＋16Ts），剩26验证/完整阈值校准，4Tr不用。此前5个badcase全部归入新训练集，不能继续视为独立测试。协议清单见 `configs/orgslot/word_official96_seed42.json`。

| 阶段 | 账号/集群/Slurm account | Job | 资源/QoS | 提交后核验状态 |
|---|---|---|---|---|
| 准备数据 | bolinren19/SIP/sifansong | 2997385 | 4CPU/64GB/cpudebug/1h | RUNNING，CPU cv2/nibabel导入通过，开始30例官方Ts几何核验与预处理 |
| E cross-attn＋MAE encoder | bolinren19/SIP/sifansong | 2997386 | 4A800/20CPU/192GB/8a800/7d | PENDING afterok:2997385 |
| 固定与校准head＋recon | bolinren19/SIP/sifansong | 2997387 | 1A800/10CPU/128GB/8a800/7d | PENDING afterok:2997386 |

固定运行提交 `95296e1ca9e1061896fd8f777d3be73fb68a1a11`；运行worktree `/gpfs/work/aac/bolinren19/OD_OWT/.worktrees/e_official96_95296e1`；数据根 `/gpfs/work/aac/bolinren19/OD_OWT/artifacts/word_official96_seed42_20260930_v2`；提交回执 `artifacts/official96_submission_20260930/jobs_v2.json`。每卡16×4卡×累积3=192，118800updates；不硬编码最终802epoch。划分CPU测试3项、语法/编译、版本guard通过；GPU尚未运行。

首次准备2997374因CPU节点缺libGL而在重采样前失败。已增加隔离动态库路径并在CPU节点验证导入，失效训练/评估2997375/2997376已取消重建；旧记录保留，无其他历史任务变动。下一次检查2997385是否完成、READY.json计数及2997386启动配置；新旧测试集不同，分数不得直接归因架构提升。
`````

<a id="merge-configuration"></a>

### docs/CONFIGURATION.md

来源：`feature/orgslot @ f198a191fe9460410c0e8ed32c3392564b165ae1`。原文完整保留。

`````text
# 统一配置与部署契约

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

## 3D SAM-tail 两条主线（2026-10-08，尚未训练）

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
没有为此更改现有空间分支；GPU未验证、未提交任务，不能宣称GPU通过。
`````

<a id="merge-e-crossattn-mae"></a>

### docs/E_CROSSATTN_MAE.md

来源：`feature/orgslot @ f198a191fe9460410c0e8ed32c3392564b165ae1`。原文完整保留。

`````text
# E cross-attention + MAE encoder

Purpose: test existing E query_refinement=cross_attn with AutoPET MAE encoder initialization.
Mean-pooled organ query reads P4, then dot-product mask; no F multi-scale token refinement or reverse attention.

Recipe: slurm/orgslot/train/arm_e_crossattn_mae_encoder.sbatch.
WORD07072, 448/384 ROI20, seed0, 20 tokens, channels128, small-organ loss lambda0.01,
top-k background, 4 GPUs x microbatch16 x accumulation3 =192, lr7.5e-5,
118800 updates, 5940 warmup, BF16, clip1.0. Decoder is randomly initialized.
Same MAE checkpoint SHA as existing encoder-only experiments.

Evaluation: arm_e_mae_unified.sbatch; training6000 threshold calibration, test6990/24 cases,
fixed0.5 and calibrated head, reconstruction threshold0.02. Report eight classes separately.
Historical E encoder 84.99 is a reference, not a perfectly controlled comparison:
historical precision/microbatch/resume trajectory differ. Do not attribute all changes to cross-attention.
No claim of exceeding85 before complete evaluation.
`````

<a id="merge-sam-soft-prior"></a>

### docs/SAM_SOFT_PRIOR.md

来源：`feature/orgslot @ f198a191fe9460410c0e8ed32c3392564b165ae1`。原文完整保留。

`````text
# SAM-tail + MAE encoder: soft prior ablation

Modes: soft_prior (routing only), soft_prior_aux (same routing plus supervision).
No P2, no tail modification, no added parameters. Before each P16/P8/P4 token
read, mean current tokens, L2 normalize query/pixels, dot times sqrt(C).
Use detached log(0.2+0.8*sigmoid(logits)) as additive attention bias shared by
20 tokens and four heads. Tail itself receives no mask prior in these modes.

Auxiliary logits retain gradients, are upsampled directly to GT resolution,
and use existing small-organ loss. Average three scale losses, eta=0.25:
L = L_reconstruction_related + 0.01*(L_final + 0.25*mean(L16,L8,L4)).
Log each scale, average and weighted auxiliary loss. Retained supervision;
background-only batches use zero auxiliary loss without skipping backward.
Same seed0, WORD07072 ROI20, microbatch16 x4GPU xaccum3=192, 118800 updates,
MAE encoder initialization; unified head calibration/test and recon evaluation.

## Literature and scope

Mask2Former official maskformer_model.py copies final loss weights to every
auxiliary layer and sums terms. Its Cityscapes panoptic config uses class2,
mask BCE5, Dice5. These are not relative auxiliary weights for our loss.
https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/maskformer_model.py
https://github.com/facebookresearch/Mask2Former/blob/main/configs/cityscapes/panoptic-segmentation/maskformer2_R50_bs16_90k.yaml

H-SAM official trainer.py uses w=0.6**(0.990**epoch), loss=(1-w)*coarse+w*final.
Thus coarse/final starts0.4/0.6, then coarse decays. It does not use fixed0.25.
https://github.com/Cccccczh404/H-SAM/blob/main/trainer.py

Our fixed0.25 on the mean is an explicit adaptation to retain the established
final segmentation/reconstruction balance, NOT an official paper parameter.
Do not change it based on test results. Only one seed; no significance claim.
CPU regression validation is separate from GPU/DDP validation.
`````

<a id="merge-sam-tail-hsam"></a>

### docs/SAM_TAIL_HSAM.md

来源：`feature/orgslot @ f198a191fe9460410c0e8ed32c3392564b165ae1`。原文完整保留。

`````text
# SAM-tail: differentiable coarse-mask guidance ablation

Keep the existing MAE-encoder SAM-tail baseline. Add three runs:

| Variant | HSAM_SUPERVISION | Coarse loss |
|---|---|---|
| H-SAM-inspired low resolution | downsample_gt | P4 logits vs nearest-resized GT |
| Full-label supervision | upsample_logits | bilinear-upsampled P4 logits vs full GT |
| M2F-style routing | m2f_hard | M0-M3 upsampled logits vs full GT |

The third run uses a shared LayerNorm/3-layer mask MLP on mean organ tokens,
with shared P4 pixel features. Predict M0 before P16, M1 after P16, M2 after
P8, M3 after P4. Resize the preceding logits to each attention scale, threshold
sigmoid at .5, detach boolean routing, and reset fully blocked rows to global
attention. Keep SAM-tail unchanged without soft guidance. Use
lambda_seg*(final_loss + L_M0 + L_M1 + L_M2 + L_M3), i.e. equal weights rather
than H-SAM scheduling. This increases segmentation supervision strength versus
the baseline; do not attribute differences solely to routing. It adapts the
M2F mechanism, not instance matching or its original class/mask losses.

All use the same MAE encoder (encoder only), seed, WORD07072 ROI20,
microbatch16/accumulation3 on 4 GPUs (effective192), small-organ loss,
lambda_seg=.01 and unchanged training update budget. No aux-only run.

For the two H-SAM-inspired variants, after P16/P8/P4 refinement, mean-token cosine dot original P4 gives coarse
logits. Their sigmoid probabilities modulate both token-to-pixel reads in
SAM-tail. The gate multiplies projected V (equivalent to attention probability
multiplication), with no detach, floor or renormalization. Reverse reads are
unchanged. This is NOT a complete H-SAM reproduction (no CMAttn, SAM pretrained
decoder, or hierarchical high-resolution decoder).

For those two variants, final weight w(epoch)=0.6**(0.990**epoch), following the public H-SAM trainer,
not the different schedule described in its paper. Total segmentation term:
.01*((1-w)*coarse_loss+w*final_loss). Other losses unchanged. Coarse readout is
uncalibrated; final readout retains existing calibration. Monitor both losses,
final weight and lost positive slices after GT resizing.

Use slurm/orgslot/train/arm_f_sam_tail_hsam.sbatch with each mode. Pin source
and verify revision before submission.
Evaluation restores hsam_supervision from checkpoint args; final-mask-only
evaluation is the main protocol (no coarse/final ensemble). Existing checkpoint
parameters remain compatible; omitting the mode defaults to the old behavior,
so evaluation must explicitly verify it instead of silently using the default.

## Submitted 2026-09-27 (Asia/Shanghai)

### Background-only coarse-loss fix (2026-09-28)

Job 2992278 failed when coarse predictions were empty. Legacy batch TGR can
retain background alone; with no ROI focus and lambda_bg_seg=0, no head runs.
This is legal: coarse loss is now zero and reconstruction backward continues.
Missing predictions for any supervised slot still raise an error (including
partially missing dictionaries). All three supervision modes share the fix.
Five CPU tests pass, including real-model background-only finite backward;
multi-rank GPU validation has not been run. Existing immutable runtime jobs
do not inherit this fix: replace their snapshots and evaluation dependencies
before claiming deployment. Keep retained supervision and original loss weights.

Pinned code: `569582d41b9baf481ca7712e9e7cbf0f27aab16b`.
Archive SHA256: `8e15cd5c166f9a79055fccca8a7e5e06621bfe03c016ecf539777047329a3499`.
MAE SHA256: `0b3571a3e79095686a0b12649735a298aaeb8a0b3bb93a47f3484439ac4a1ec6`.

| Mode | User/cluster | QoS | Training | afterok evaluation | Time limit |
|---|---|---|---|---|---|
| downsample_gt | bolinren19/SIP | 4a800 | 2992278 | 2992279 | 7 days |
| upsample_logits | sifansong/XEC | 8gpus | 148892 | 148895 | 5 days |
| m2f_hard | antengcai23/XEC | 4gpus | 148893 | 148896 | 7 days |

All Slurm accounts are `sifansong`; this is distinct from the login username.
Training: 4 A800, 20 CPU, 192 GB. Evaluation: 1 A800, 10 CPU, 128 GB.
No automatic requeue (avoid existing-output restart failure). Queued, GPU
validation not yet run. CPU: 9 existing-head regression tests and 3 new tests
passed locally; 3 new tests passed in each XEC environment.

Runtime paths (do not pull or edit under queued/running jobs):
- SIP: `/gpfs/work/aac/bolinren19/OD_OWT/.worktrees/sam_guided_569582d`
- sifansong/XEC: `/gpfs/work/aac/sifansong/worktrees/sam_guided_569582d`
- antengcai23/XEC: `/gpfs/work/aac/antengcai23/worktrees/sam_guided_569582d`

Train script: `slurm/orgslot/train/arm_f_sam_tail_hsam.sbatch`.
Evaluation: `slurm/orgslot/eval/sam_tail_guided.sbatch`, validates saved mode,
MAE scope and head, then runs 6000 training-calibration slices, full test head
evaluation and reconstruction threshold .02 using the existing protocol.
Training output: `Results/OrganSlotBank/Common8/WORD_2D/ArmF_2D_JOBID`.
Check start logs for encoder-only load, microbatch16, accumulation3, finite
coarse/final loss, lost-positive count and correct mode before trusting results.
`````

<a id="merge-sam-tail-all-seg"></a>

### docs/SAM_TAIL_ALL_SEG.md

来源：`feature/orgslot @ f198a191fe9460410c0e8ed32c3392564b165ae1`。原文完整保留。

`````text
# SAM-tail + MAE encoder: all-slot segmentation supervision

Independent controlled ablation against SAM-tail + MAE encoder (no auxiliary
mask supervision). Only intended change: seg_supervision retained -> all.

Entry: slurm/orgslot/train/sam_tail_mae_all_seg.sbatch.
Supply EXPERIMENT_WORKDIR, ARM_F_TARGET and MAE_INIT_CHECKPOINT as for the
existing SAM-tail encoder launcher. Use a new immutable runtime/output job.
Do not resume a retained-supervision run for the primary controlled comparison.

Unchanged: WORD07072, input448, ROI384/probability0.2, 20tokens, channels128,
SAM-tail architecture, MAE encoder initialization, microbatch16 x4GPU xaccum3
=192, lambda_seg0.01, existing small-organ loss, optimizer/LR/update budget.
Background weight remains0: all means all eight visible foreground organs,
including empty-slice negative supervision, not a newly supervised background.

Reconstruction still samples legacy_batch keep and forces the ROI focus slot
to remain. Its target and canvas fusion use this unchanged keep. Segmentation
uses all foreground head rows. Model computes slots using keep OR head_mask,
so reconstruction-dropped organs receive real token features, not zero masks.
Shared parameters still receive gradients from both branches; this is NOT
gradient isolation. Loss values/scales and cost may change when more rows are
supervised; lambda and loss reduction are deliberately unchanged for this test.

Default ARM_F_SEG_SUPERVISION remains retained. Only this new entry sets all.
Existing checkpoints and model implementation remain unchanged. Reuse the
SAM-tail encoder unified evaluation protocol; report full24 cases/6990 slices,
fixed and train-calibrated thresholds, recon, per-organ P/R and volume ratio.
Existing seg_*_supervised_samples/positive_samples/negative_samples logs allow
inspection of effective supervision; distinguish seg_* from unrelated zero
positive_* auxiliary-loss counters. Incremental training must still obey label
visibility; this launcher is for the fully labelled Common8 experiment only.

CPU test verifies exact reconstruction invariance for the same keep, real
dropped-slot computation and finite nonzero segmentation gradients. GPU memory,
throughput and full training have not been validated or submitted by this change.
`````

<a id="merge-abcd-retirement"></a>

### docs/ABCD_RETIREMENT.md

来源：`feature/orgslot @ f198a191fe9460410c0e8ed32c3392564b165ae1`。原文完整保留。

`````text
# A/B/C/D retirement (2026-09-27)

User authorized retiring A/B/C/D code, with results preserved.
Removed 18 dedicated WORD07072 A-D training/smoke entrypoints from the current
development tree. Recover any removed file from the remote tag
`archive/20260927/abcd-before-retirement` (7b250c9).

Checkpoints, experiment metrics, inference outputs, historical runtime worktrees,
and all queued/running jobs are unchanged. No experiment data was deleted.

Partial cleanup only: core head removal/default changes/test migration were
blocked by approval review due to shared E/F dependencies. Those operations
were NOT executed. A-D model implementations and evaluation compatibility
remain pending explicit approval for that shared-path refactor. E/F still
inherit SharedPixelQueryDecoder; never delete that class without preserving
its initialization/state_dict behavior. Existing defaults are unchanged.

Historical docs may name the deleted launchers; use the archive tag to reproduce
those commands. This retirement is a maintenance choice, not a declaration
that the scientific results were invalid.
`````

<a id="merge-cleanup-20260927"></a>

### docs/CLEANUP_20260927.md

来源：`feature/orgslot @ f198a191fe9460410c0e8ed32c3392564b165ae1`。原文完整保留。

`````text
# Repository cleanup — 2026-09-27

Keep four development branches: main (OWT), feature/orgslot (all OrganSlot
variants), experiment/psem and experiment/lossbalance (independent research).

Retired branch names, locally and on origin:
- experiment/orgslot-arm-f: code already integrated; unique tip only reorganizes
  historical documentation. Recover from archive/20260927/orgslot-arm-f.
- fix/arme-mae-ddp: ancestor of unified branch. Recover from
  archive/20260927/arme-mae-ddp.

Removed obsolete fixed-batch smoke tools and their two launchers:
tools/smoke_arm_f_2d_batch8.py and tools/smoke_arm_f_batch4.py, plus the
corresponding slurm/orgslot/smoke/arm_f_2d_batch8.sbatch and arm_f_batch4.sbatch.
These had no other executable callers. Current recipes use 2D16 and 3D6.
The removed tools are recoverable from archive/20260927/orgslot-before-cleanup.
This does not imply that batch8 scientific results are invalid.

Do NOT delete low-scoring scientific controls just for being below SOTA.
Keep A-F model paths for checkpoint loading, negative results and ablation.
Keep all checkpoints, results, data and original worktrees. Detached old
worktrees are historical runtime snapshots, not extra development branches.
Unrelated dirty root docs/FORMAL_EXPERIMENT_REGISTRY.md was untouched.

Protected active/queued jobs at cleanup: SIP 2965316/2965317 and
2992278/2992279; XEC 148830/148831, 148833/148834, 148892/148895,
148893/148896. Their runtime snapshots are unchanged.
No scientific experiment was classified as invalid without supporting evidence.
`````

<a id="owt-results"></a>

### main:docs/RESULTS.md

来源：`main @ db349d9b66de953ed21dc088b0c2ca723644fb43`。原文完整保留。

`````text
> 原始OWT实验账本，历史快照；不是最新OrganSlot状态。

# OWT Experiment Summary

Last updated: 2026-07-19

This document is the primary experiment ledger for this repository. CSV files are
kept as machine-readable raw results, while experiment settings, conclusions, and
comparisons are maintained here.

## 1. Current Experiment Scope

The completed experiments are **joint full-class OWT reconstruction baselines**.
They are not class-incremental experiments.

Current training protocol:

```text
All class labels are available simultaneously
-> train OWT reconstruction model
-> inspect token-conditioned reconstruction (Step 2)
-> derive pseudo-segmentation from reconstruction signals (Step 3)
```

The following components have not yet been implemented or evaluated:

- old/new class splits during training;
- sequential 4-4, 4-2, or 2-2 stages;
- hiding old-class labels in later stages;
- frozen previous model;
- knowledge distillation;
- prototype-based incremental constraints;
- Old/New/All incremental metrics.

### Experiment objectives

1. Establish a non-incremental full-class reference for the existing OWT code.
2. Test whether organ tokens contain class-specific reconstruction information.
3. Evaluate reconstruction-derived pseudo-segmentation on AbdAutoPet, WORD, and BTCV.
4. Compare 2D OWT with Fixfr4 OWT.
5. Diagnose complete misses and instability on small organs.

## 2. Datasets

| Dataset | Classes | Train cases | Test cases | Train samples | Test slices | Notes |
|---|---:|---:|---:|---:|---:|---|
| AbdAutoPet | 4 | 700 | 200 | 78,400 slices | 22,400 | Original OWT-formatted JPG/PNG data; 112 slices per case |
| WORD Common8 | 8 | 96 | 24 | 19,078 slices | 4,665 | Paper-aligned 8:2 split, seed 42 |
| BTCV Common8 | 8 | 24 | 6 | 3,588 slices | 938 | Paper-aligned 8:2 split, seed 42 |

Common8 class order:

| ID | Class |
|---:|---|
| 1 | spleen |
| 2 | right kidney |
| 3 | left kidney |
| 4 | gallbladder |
| 5 | esophagus |
| 6 | pancreas |
| 7 | liver |
| 8 | stomach |

The AbdAutoPet files contain labels 1-4, but the repository does not contain a
reliable label-ID-to-organ-name mapping. AbdAutoPet results therefore retain the
names `label_1` through `label_4`.

## 3. Preprocessing

### WORD and BTCV Common8

```text
NIfTI image and mask
-> canonical RAS orientation
-> resample to 1 x 1 x 3 mm^3
-> image: linear interpolation
-> mask: nearest-neighbor interpolation
-> HU clipping [-175, 250]
-> fixed normalization to [0, 1]
-> center crop/pad to 512 x 512 without foreground loss
-> resize to 224 x 224
-> image: JPEG quality 95
-> mask: lossless uint8 PNG
```

Training uses `--intensity_norm fixed_255`, so the saved HU-windowed image is
divided by 255 without a second per-slice min-max normalization.

### Preprocessing integrity audit

The 512-to-224 resize did not remove any positive test slices and preserved the
expected area after correcting for the spatial scale factor.

| Dataset | Class | Positive slices before resize | Positive slices after resize | Lost slices |
|---|---|---:|---:|---:|
| WORD | gallbladder | 243 | 243 | 0 |
| WORD | esophagus | 570 | 570 | 0 |
| BTCV | gallbladder | 113 | 113 | 0 |
| BTCV | esophagus | 160 | 160 | 0 |

The labels are therefore not disappearing during preprocessing. However, the
full-field resize reduces the effective in-plane resolution to approximately
2.286 mm per pixel. At 224 x 224, the average foreground area per positive WORD
slice is only about 68 pixels for gallbladder and 39 pixels for esophagus. Both
are substantially smaller than one 16 x 16 ViT patch (256 pixels), which can
weaken their feature representation.

## 4. Model and Training Settings

### Architecture

Current `v11` models use the ViT branch in `OWT_models.py`:

```text
PatchEmbed / PatchEmbed3Dfix
-> six BlockLA image-encoding blocks
-> OrganCollector / OrganCollector3D
-> organ token groups
-> six shared BlockLA token-group blocks
-> AHER spatial recovery
-> eight BlockLA reconstruction-decoder blocks
-> patch prediction and image reconstruction
```

This is a custom MAE-ViT with Linear Attention. It is not the 3D hierarchical
Swin UNETR used by the incremental-segmentation reference paper.

### Shared hyperparameters

| Parameter | Value |
|---|---|
| Architecture version | `v11` |
| Model | ViT-Base, patch size 16, Linear Attention |
| Input size | 224 x 224 |
| Embedding dimension | 768 |
| Transformer depth | 6 image blocks + 6 token-group blocks |
| Decoder | AHER + 8 reconstruction blocks |
| Token factor | 20 tokens per class group |
| Base learning rate | `1e-4` |
| Weight decay | `0.05` |
| Epochs | 1,200 |
| Warm-up epochs | 60 |
| Mask ratio argument | `1.0` |
| Loss | full-image L2 + LPIPS |
| GPUs | 2 |

The 8-class Common8 model has 180 tokens: 20 background tokens plus 8 x 20
organ tokens. The 4-class AbdAutoPet model has 100 tokens.

### Experiment-specific settings

| Experiment | Input | Batch size | Normalization | Checkpoint |
|---|---|---:|---|---|
| AbdAutoPet 2D | 1 x 224 x 224, repeated to RGB | 96 | per-sample min-max | epoch 1199 |
| WORD 2D | 1 x 224 x 224, RGB | 32 | fixed `/255` | epoch 1199 |
| WORD 3D | 4 x 224 x 224, Fixfr4, TS1 | 32 | fixed `/255` | epoch 1199 |
| BTCV 2D | 1 x 224 x 224, RGB | 48 | fixed `/255` | epoch 1199 |
| BTCV 3D | 4 x 224 x 224, Fixfr4, TS1 | 32 | fixed `/255` | epoch 1199 |

Final epoch global training losses:

| Experiment | L2 (`train_loss`) | LPIPS (`train_p_loss`) | Approx. optimized sum |
|---|---:|---:|---:|
| WORD 2D | 0.000730 | 0.016920 | 0.017650 |
| WORD 3D | 0.000694 | 0.016783 | 0.017477 |
| BTCV 2D | 0.004199 | 0.061805 | 0.066003 |
| BTCV 3D | 0.003873 | 0.059084 | 0.062957 |

These are global full-image losses. The current training code does not log a
separate loss for each organ.

## 5. Evaluation Protocol

### Step 2: Token-conditioned reconstruction

Step 2 produces normalized CT reconstruction signals, not class probabilities.

| Mode | Tokens retained | Reconstruction target |
|---|---|---|
| `whole` | background and all organ tokens | complete CT image |
| `background_only` | background tokens | background CT with organs zeroed |
| `organs_only` | all organ tokens | organ CT with background zeroed |
| `class_c_only` | only class-c tokens | class-c CT region with all other pixels zeroed |

Reported metrics are L1, L2, LPIPS, PSNR, and SSIM. Thresholded variants first
set reconstruction values below 0.02 to zero.

Important limitation: class-only targets are extremely sparse, but current L1,
L2, PSNR, and SSIM are averaged over the complete volume. A nearly zero output
can therefore receive apparently excellent metrics while failing to reconstruct
the organ. Class-only full-volume metrics must not be interpreted as organ ROI
reconstruction quality.

### Step 3: Reconstruction-derived pseudo-segmentation

Two score maps are evaluated:

```text
Direct(c)   = reconstruction using only class-c tokens
Indirect(c) = max(input - reconstruction without class-c tokens, 0)
```

The score map is thresholded at 0.15. Common8 uses a minimum connected-component
size of 100 voxels; AbdAutoPet uses 20. Reported segmentation metrics are Dice,
NSD, and HD95 before and after morphology-based postprocessing.

This is not a learned softmax segmentation head.

## 6. Step 2 Reconstruction Results

### Whole-image reconstruction

| Dataset | Model | L1 | L2 | LPIPS | PSNR | SSIM |
|---|---|---:|---:|---:|---:|---:|
| WORD | 2D | 0.00691 | 0.000314 | 0.01358 | 35.31 | 0.9660 |
| WORD | 3D Fixfr4 | 0.00722 | 0.000321 | 0.01692 | 35.25 | 0.9535 |
| BTCV | 2D | 0.03318 | 0.006261 | 0.10914 | 22.32 | 0.7195 |
| BTCV | 3D Fixfr4 | 0.03363 | 0.006260 | 0.15078 | 22.26 | 0.6977 |
| AbdAutoPet | 2D | 0.00686 | 0.000157 | not evaluated | 38.42 | 0.9196 |

WORD whole-image reconstruction is substantially better than BTCV. Fixfr4 does
not improve whole-image reconstruction over 2D in the current checkpoints.

### Organs-only reconstruction

| Dataset | Model | L1 | L2 | LPIPS | PSNR | SSIM |
|---|---|---:|---:|---:|---:|---:|
| WORD | 2D | 0.00247 | 0.000679 | 0.01299 | 55.25 | 0.9848 |
| WORD | 3D Fixfr4 | 0.00224 | 0.000592 | 0.01364 | 59.44 | 0.9852 |
| BTCV | 2D | 0.00999 | 0.003361 | 0.03830 | 38.50 | 0.9037 |
| BTCV | 3D Fixfr4 | 0.00816 | 0.002694 | 0.04017 | 42.95 | 0.9271 |
| AbdAutoPet | 2D | 0.00600 | 0.001322 | not evaluated | 32.66 | 0.9115 |

Fixfr4 improves the aggregated organs-only L1/L2, PSNR, and SSIM for WORD and
BTCV, indicating that short through-plane context helps overall organ-region
reconstruction.

### Small-organ class-only reconstruction

| Dataset | Model | Class | L1 | L2 | LPIPS | PSNR | SSIM |
|---|---|---|---:|---:|---:|---:|---:|
| WORD | 3D | gallbladder | 0.000161 | 0.0000167 | 0.00217 | 75.40 | 0.9996 |
| WORD | 3D | esophagus | 0.000165 | 0.0000195 | 0.00459 | 72.84 | 0.9994 |
| BTCV | 3D | gallbladder | 0.001022 | 0.0000870 | 0.00576 | 57.97 | 0.9918 |
| BTCV | 3D | esophagus | 0.000932 | 0.0000277 | 0.00637 | 57.21 | 0.9916 |

These high PSNR/SSIM values are dominated by the zero background and are not
evidence of successful small-organ reconstruction. All four corresponding Direct
Dice values are zero in Step 3.

## 7. Step 3 Pseudo-segmentation Results

### Macro Dice

Postprocessed class-macro Dice over all target classes:

| Dataset | Model | Direct | Indirect | Better extraction mode |
|---|---|---:|---:|---|
| AbdAutoPet | 2D | 89.94 | 90.39 | Indirect |
| WORD | 2D | 55.07 | 58.31 | Indirect |
| WORD | 3D Fixfr4 | 59.93 | 60.94 | Indirect |
| BTCV | 2D | 10.39 | 10.43 | Neither is reliable |
| BTCV | 3D Fixfr4 | 34.16 | 12.24 | Direct |

### WORD per-class Dice

| Class | 2D Direct | 2D Indirect | 3D Direct | 3D Indirect |
|---|---:|---:|---:|---:|
| spleen | 86.00 | 88.31 | 89.24 | 89.92 |
| right kidney | 79.30 | 86.83 | 89.49 | 89.05 |
| left kidney | 79.98 | 85.55 | 88.25 | 87.87 |
| gallbladder | 0.00 | 0.00 | 0.00 | 0.00 |
| esophagus | 0.00 | 0.00 | 0.00 | 0.00 |
| pancreas | 38.43 | 43.73 | 47.61 | 54.12 |
| liver | 91.02 | 92.28 | 92.02 | 93.15 |
| stomach | 65.84 | 69.78 | 72.81 | 73.44 |

WORD learns the large organs well. Fixfr4 improves kidney, pancreas, and stomach
performance, but gallbladder and esophagus remain completely unresolved.

### BTCV per-class Dice

| Class | 2D Direct | 2D Indirect | 3D Direct | 3D Indirect |
|---|---:|---:|---:|---:|
| spleen | 0.00 | 8.27 | 60.07 | 14.05 |
| right kidney | 0.00 | 6.82 | 34.68 | 8.85 |
| left kidney | 0.00 | 6.68 | 37.24 | 8.67 |
| gallbladder | 0.00 | 0.25 | 0.00 | 0.27 |
| esophagus | 0.00 | 0.39 | 0.00 | 0.27 |
| pancreas | 0.00 | 0.90 | 8.59 | 0.77 |
| liver | 83.11 | 51.48 | 85.07 | 50.81 |
| stomach | 0.00 | 8.68 | 47.61 | 14.20 |

BTCV 2D collapses primarily to liver. Fixfr4 partially recovers spleen, kidneys,
pancreas, and stomach in Direct mode, but performance remains far below a usable
segmentation baseline.

### AbdAutoPet per-class Dice

| Class | Direct mean Dice | Indirect mean Dice | Direct median | Indirect median |
|---|---:|---:|---:|---:|
| label 1 | 96.45 | 96.51 | 96.70 | 96.76 |
| label 2 | 92.69 | 92.93 | 93.24 | 93.51 |
| label 3 | 92.94 | 93.04 | 94.67 | 94.72 |
| label 4 | 77.66 | 79.06 | 80.66 | 81.97 |

For the smallest AbdAutoPet class (`label_4`):

- no positive case produced an entirely empty prediction;
- Direct had 8/200 cases with Dice below 0.5;
- Indirect had 6/200 cases with Dice below 0.5;
- worst Dice was approximately 0.22;
- small-organ instability is present, but systematic complete misses are not.

## 8. Small-organ Failure Analysis

### Observed complete misses

| Dataset/model | Mode | Observation |
|---|---|---|
| WORD 2D | Direct | gallbladder and esophagus raw predictions empty in 24/24 cases |
| WORD 3D | Direct | gallbladder and esophagus raw predictions empty in 24/24 cases |
| WORD 2D | Indirect | postprocessing empties gallbladder in 23/24 and esophagus in 22/24 cases |
| WORD 3D | Indirect | postprocessing empties both classes in 23/24 cases |
| BTCV 2D | Direct | all classes except liver raw-empty in 6/6 cases |
| BTCV 3D | Direct | gallbladder and esophagus raw-empty in 6/6 cases |
| AbdAutoPet 2D | Direct/Indirect | no empty prediction in any GT-positive case |

WORD Indirect produces non-empty raw signals, but they are spatially incorrect:
raw Dice is approximately 0.0003 for gallbladder and 0.008 for esophagus. The
100-voxel component filter exposes rather than creates the underlying failure.

### Likely causes, ordered by current evidence

1. **Global reconstruction-loss imbalance.** Small organs contribute only a tiny
   fraction of the full-image L2 and LPIPS objectives.
2. **Low effective spatial resolution.** Full-field 224 x 224 input followed by
   16 x 16 patches makes small organs sub-patch structures.
3. **No small-organ foreground sampling.** Most slices do not contain the target
   organ, and the loader samples all slices/windows uniformly.
4. **Weak token identifiability.** OrganCollector has no direct mask-alignment
   supervision, so small-organ information may leak into background or other
   organ token groups.
5. **Shared TGEnc and AHER.** All classes share token processing and spatial
   recovery, allowing large classes to dominate optimization.
6. **Fixed reconstruction threshold.** The 0.15 score threshold is not a calibrated
   class probability and may suppress weak class-specific signals.
7. **Postprocessing.** The 100-voxel filter removes scattered Indirect artifacts,
   but it is not the root cause of Direct raw-empty predictions.

## 9. Relationship to the Incremental-Segmentation Paper

The reference paper uses a supervised 3D Swin UNETR segmentation network with
96 x 96 x 96 training crops, voxel-wise class logits, and explicit incremental
training stages. Current OWT uses a short Fixfr4 MAE-ViT reconstruction model and
derives segmentation from reconstruction amplitudes.

The current results must therefore not be reported as 4-4 or 2-2 incremental
results. Even comparison with the paper's Offline result is only an external
scale reference, not a controlled architecture comparison.

| Dataset | Current best 3D OWT macro Dice | Paper Offline Dice | Uncontrolled gap |
|---|---:|---:|---:|
| BTCV | 34.16 | 90.75 | -56.59 points |
| WORD | 60.94 | 85.47 | -24.53 points |

## 10. Current Conclusions

1. WORD OWT reconstructs complete CT images reasonably well and learns large
   organs, but does not form usable gallbladder or esophagus token signals.
2. BTCV is limited by only 24 training cases and shows broad class collapse,
   especially in 2D.
3. Fixfr4 improves aggregate organ reconstruction and several medium/large-organ
   Dice scores, but four slices are insufficient to solve the smallest classes.
4. AbdAutoPet demonstrates that reconstruction-based organ tokens can produce
   useful pseudo-segmentation when classes are larger, frequent, and supported by
   substantially more training data.
5. Current class-only PSNR/SSIM metrics are biased by zero background and should
   not be used as evidence of small-organ reconstruction quality.
6. No claim about incremental-learning performance is supported yet.

## 11. Required Next Evaluations

- Add foreground-only ROI-L1 and ROI-L2 for class-only reconstruction.
- Add positive-slice and organ-bounding-box PSNR/SSIM.
- Measure reconstruction energy recovered inside the organ and leakage outside it.
- Sweep class-specific reconstruction thresholds before changing the model.
- Inspect OrganCollector attention overlap with ground-truth masks.
- Run a small-organ-positive subset overfit test to separate optimization failure
  from dataset-scale imbalance.
- Establish a supervised segmentation reference under the same preprocessing.
- Only after validating the joint baseline, implement and evaluate matched 4-4
  and 2-2 incremental protocols.

### Small-organ and low-data intervention plan

Increasing the spatial grid from 14 x 14 (196 tokens) to 16 x 16 (256 tokens) is a valid minimal ablation, but it is not expected to solve the current complete misses by itself. With full-field preprocessing, this changes the physical patch width only from approximately 36.6 mm to 32 mm and increases small-organ pixels by about 1.31 times. The global reconstruction-loss imbalance remains unchanged.

Two implementations must be distinguished:

1. Resizing the existing 224 JPG files to 256 in the DataLoader creates no new anatomical information and is not a valid high-resolution experiment.
2. Regenerating 256 x 256 images directly from the 512 x 512 resampled NIfTI data, then training with input size 256 and patch size 16, is the valid 256-token ablation.

Recommended priority:

1. Replace full-field downscaling with native-resolution foreground-centered 224 x 224 crops. This retains the same 196-token cost while reducing physical patch width to approximately 16 mm. Use sliding-window aggregation at test time.
   Do not replace each volume with one fixed crop. Mix low-resolution global views with dynamically jittered native-resolution foreground and random crops, supervise only classes present in each local crop, and ensure whole-volume coverage across epochs. A 224 mm field of view retains local context but not the entire abdomen.
   Generate crops on the fly rather than materializing many duplicate files. For 2D, apply exactly the same crop and geometric transform to the image and mask. For Fixfr4/3D, use one shared XY crop across every slice in the window and crop image/mask with identical XYZ coordinates. These crops are correlated views of the same case, not additional independent patients, so dataset splits and cross-validation must remain case-wise.
2. Add class-balanced ROI reconstruction loss, averaged independently inside each present organ rather than over the complete image.
3. Use class-uniform positive-slice/window sampling, with explicit oversampling of gallbladder, esophagus, and pancreas.
4. Add an OrganCollector attention-alignment loss or an organ-level binary head so that each token group is explicitly tied to its class mask.
5. Run the regenerated 256 x 256, patch-16 experiment as a controlled resolution ablation; evaluate patch-8 or a high-resolution branch only after the loss and sampling problems are addressed.
6. For BTCV, initialize from the WORD Common8 checkpoint and fine-tune with a lower learning rate, or train on WORD and BTCV jointly with dataset-balanced sampling.
7. Report BTCV results with cross-validation because the current six-case test split has high variance.

The resolution ablation must retain the same split, model depth, token factor, loss, epochs, and evaluation protocol so that any improvement can be attributed to spatial resolution rather than a simultaneous training change.

## 12. Result Artifacts

- AbdAutoPet evaluation: `Results/AbdAutoPet_Eval/2D/OWT-Joint-C4-Recon_ckpt1199/`
- WORD 2D evaluation: `Results/Common8_Eval/WORD_2D/OWT-Joint-C8-Recon_ckpt1199/`
- WORD 3D evaluation: `Results/Common8_Eval/WORD_3D_Fixfr4/OWT-Joint-C8-Recon3D_ckpt1199/`
- BTCV 2D evaluation: `Results/Common8_Eval/BTCV_2D/OWT-Joint-C8-Recon_ckpt1199/`
- BTCV 3D evaluation: `Results/Common8_Eval/BTCV_3D_Fixfr4/OWT-Joint-C8-Recon3D_ckpt1199/`

The per-case CSV files remain raw evidence. Future experiment interpretation and
status updates should be added to this document.

## 13. CropMix-v1 Small-Organ Experiment

### Purpose

Test whether preserving native in-plane detail and increasing the frequency of
small-organ-positive views improves WORD gallbladder, esophagus, and pancreas
reconstruction/segmentation without changing the OWT architecture or test input.

### Training intervention

- Source data: WORD resampled to 1 x 1 x 3 mm, HU clipped to [-175, 250],
  mapped to [0, 1], and stored on a 512 x 512 canvas.
- Model input remains 224 x 224; patch size remains 16 and the grid remains
  14 x 14 = 196 patches.
- View mixture: 50% complete 512 x 512 images resized to 224 x 224 and 50%
  native-resolution 224 x 224 XY crops.
- Crop center: a present organ, with priority given to Common8 classes 4, 5,
  and 6 (gallbladder, esophagus, pancreas) and up to 32-pixel XY jitter.
- Focus sampling: 80% of local-crop requests are redirected to a slice/window
  containing at least one focus class.
- 3D geometry: all four Fixfr4 slices and their masks use one shared XY crop.
- A local crop's focus class is forced into that iteration's reconstructed
  organ-token set; at least one class always remains visible.
- Intensity handling: complete-view resizing is clipped to [0, 1] after cubic
  interpolation but is not per-sample min-max normalized.

This is a combined intervention (resolution-preserving crop, focused sampling,
and focus-token selection), not a single-factor crop ablation.

### Controlled settings

The Common8 mapping, case split, model architecture, token factor 20, learning
rate 1e-4, 1200 epochs, full 224 x 224 test input, thresholds, and Step2/Step3
evaluation code remain matched to the existing WORD baseline.

### Slurm chain submitted 2026-07-20

| Stage | 2D Job | 3D Job | Dependency |
|---|---:|---:|---|
| HR512 preprocessing | 1552981 | shared | none |
| 1-epoch smoke | 1552982 | 1552983 | preprocessing success |
| 1200-epoch full training | 1552984 | 1552985 | corresponding smoke success |
| Step2/Step3 evaluation | 1554939 | 1554940 | corresponding full training success |

At submission time, Slurm accepted five jobs and then reached
`QOSMaxSubmitJobPerUserLimit`. The evaluation jobs were submitted after the
smoke jobs released their slots.

Progress check on 2026-07-20 at approximately 21:31 CST:

- preprocessing and both 1-epoch smoke jobs completed successfully;
- 2D full training was in epoch 326/1200 with L2 0.0052, LPIPS 0.0851,
  crop fraction 0.492, and focus-crop fraction 0.432;
- 3D full training was in epoch 225/1200 with L2 0.0057, LPIPS 0.0956,
  crop fraction 0.499, and focus-crop fraction 0.437;
- both Slurm error logs were empty, with no observed OOM or NaN;
- estimated remaining time was about 39 hours for 2D and 49-55 hours for 3D,
  subject to epoch-time variation and checkpoint I/O.

## 14. LossBalance-v1 Reconstruction-Loss Variant

### Isolation boundary

LossBalance-v1 is implemented as a separate training version. It does not edit
or import-switch the legacy `main_pretrain.py`, `engine_pretrain.py`, or
`OWT_models.py` pipeline. The model subclass adds no parameters, so its state
dictionary remains architecture-compatible with legacy OWT checkpoints.

### Loss

```text
L_total = L_global_L2
        + lambda_roi * L_per_sample_per_organ_ROI_L2
        + lambda_lpips * L_LPIPS
```

For each sample, squared reconstruction error is averaged independently inside
each present foreground-organ mask. Those organ losses are averaged so organ
area does not determine gradient weight. Samples with no foreground organ skip
the ROI term. Background is excluded by default (`background_weight = 0`).
No Dice, BCE, CE, classification head, or segmentation head is added.

Default ablation settings are `lambda_roi = 1.0`, `lambda_lpips = 1.0`, and
`background_weight = 0.0`. The loss-only Slurm scripts use the original 224 x
224 WORD data and disable CropMix, isolating loss accounting from resolution
and sampling changes.

### Files

- `util/organ_balanced_loss.py`: vectorized 2D/3D organ-balanced ROI-L2.
- `OWT_models_lossbalance.py`: parameter-free subclass overriding loss/forward.
- `engine_pretrain_lossbalance.py`: passes masks and logs ROI accounting.
- `main_pretrain_lossbalance.py`: independent entry point and loss arguments.
- `tests/test_organ_balanced_loss.py`: area-invariance and Fixfr4 tests.
- `slurm/pretrain_lossbalance_word_2d.sbatch`: isolated WORD 2D run.
- `slurm/pretrain_lossbalance_word_3d.sbatch`: isolated WORD Fixfr4 run.

Synthetic verification produced global L2 0.4444, organ-balanced ROI-L2 2.5,
and total reconstruction loss 2.9444 for a hand-computable example. Both 2D
and 3D tests pass with finite gradients. Slurm training has not yet been
submitted because the four CropMix GPU jobs occupy the current QOS submission
limit.
`````

<a id="psem-results"></a>

### experiment/psem:docs/RESULTS.md

来源：`experiment/psem @ e8ecb1a1fc25f45c5ff93682cf69db4cb00d2fe3`。原文完整保留。

`````text
> 合并自既有分支 README；本次未重新计算指标。文中“当前”限定于原记录时点。

# PSEM 实验总览

本分支统一保存 Per-Sample Exhaustive/Structured Masking（PSEM）系列实现与实验记录。本文只对**相同数据集、相同测试病例和相同正式评估协议**下的结果排序；不同数据集之间不比较绝对数值。

## 结论摘要

- **AbdAutoPET 2D：** PSEM-v1 Indirect 仍是当前 PSEM 系列最优，4 类 post Dice 为 **90.512%**。PSEM-v3 Triplet+Loss3 的 Indirect 为 **89.938%**，与 PSEM-v2a Indirect 基本持平，但没有超过 v1。
- **WORD 2D：** PSEM-v3 同时取得当前 PSEM 系列最优 Direct（**68.353%**）和 Indirect（**68.244%**）。Triplet 的主要收益是修复 Indirect 查询分解，而不是显著提高 Direct。
- **AbdAutoPET 3D Fixfr4：** 目前只有 PSEM-v1 完成正式评估，尚不能判断 v2/v3 在 3D 上是否更好。
- **数值稳定性：** PSEM-v3 AutoPET 原 FP16 路线在约 epoch 500 附近出现 NaN。修复版本采用 BF16、LPIPS FP32、非有限值诊断和梯度裁剪，最终训练 Job `121484` 与 200 病例评估 Job `124936` 均正常完成。

## 统一指标口径

- 排名主指标：`all_cases`、类别等权宏平均 `post Dice`，数值越高越好。
- 辅助指标：`post NSD` 越高越好，`post HD95` 越低越好。
- Direct：只保留目标类别 token 的重建响应。
- Indirect：`input/full reconstruction - without-class reconstruction` 的差分响应。
- 分割结果来自重建响应阈值化及统一后处理，不是额外训练的 segmentation head。
- AutoPET 2D/3D 均为 200 个测试病例、4 个前景类；WORD 2D 为 24 个测试病例、8 个前景类。
- AutoPET 的可靠器官名称映射尚未确认，因此只使用 `label_1`–`label_4`；WORD 类别顺序为脾脏、右肾、左肾、胆囊、食管、胰腺、肝脏、胃。

## AbdAutoPET 2D 排名

相同 200 病例、224 输入、`per_sample` 归一化、固定阈值 0.15 和最小连通域 20。表内类别 Dice 顺序为 `label_1/label_2/label_3/label_4`。

| 排名 | 实验 | 读出 | post Dice | post NSD | post HD95 (mm) | 4 类 post Dice |
|---:|---|---|---:|---:|---:|---|
| 1 | PSEM-v1 | Indirect | **90.512%** | **89.981%** | 7.004 | 96.55 / 93.09 / 93.05 / 79.36 |
| 2 | 原始 OWT（对照） | Indirect | 90.386% | 89.616% | inf\* | 96.51 / 92.93 / 93.04 / 79.06 |
| 3 | PSEM-v1 | Direct | 90.110% | 89.118% | 10.277 | 96.53 / 92.95 / 91.89 / 79.06 |
| 4 | PSEM-v2a Query20 | Direct | 90.063% | 88.731% | **5.932** | 96.41 / 92.73 / 92.85 / 78.27 |
| 5 | PSEM-v3 Triplet+Loss3 | Indirect | 89.938% | 87.639% | 8.074 | 96.36 / 92.56 / 92.69 / 78.15 |
| 6 | 原始 OWT（对照） | Direct | 89.938% | 88.687% | inf\* | 96.45 / 92.69 / 92.94 / 77.66 |
| 7 | PSEM-v2a Query20 | Indirect | 89.888% | 88.421% | 6.904 | 96.38 / 92.70 / 92.70 / 77.77 |
| 8 | PSEM-v3 Triplet+Loss3 | Direct | 88.783% | 83.172% | 7.055 | 96.12 / 91.64 / 91.70 / 75.67 |
| 9 | PSEM+LossBalance-v2 | Indirect | 84.673% | 68.029% | 13.022 | 95.17 / 88.21 / 88.05 / 67.26 |
| 10 | LossBalance-v1（对照） | Direct | 84.186% | 66.665% | 9.672 | 94.82 / 87.68 / 88.01 / 66.23 |
| 11 | LossBalance-v1（对照） | Indirect | 83.299% | 64.387% | 14.753 | 94.64 / 87.17 / 86.98 / 64.40 |
| 12 | PSEM+LossBalance-v2 | Direct | 79.728% | 63.051% | 62.247 | 94.83 / 86.18 / 79.13 / 58.77 |

\* 原 OWT 合并汇总的每种读出各含 1 个无穷 HD95，因此正式宏平均记为 `inf`；Dice/NSD 从完整 200 病例逐病例文件恢复。

### PSEM-v3 的病例配对分析

以下为 10,000 次病例 bootstrap 的“PSEM-v3 减对照”宏平均 Dice 差值；置信区间只反映测试病例抽样，不包含不同训练 seed 的方差。

| 读出 | 对照 | 差值（百分点） | 95% CI | 结论 |
|---|---|---:|---|---|
| Direct | PSEM-v1 Direct | -1.326 | [-1.597, -0.988] | 明确下降 |
| Direct | 原始 OWT Direct | -1.154 | [-1.329, -0.972] | 明确下降 |
| Direct | PSEM-v2a Direct | -1.279 | [-1.436, -1.115] | 明确下降 |
| Direct | PSEM+LossBalance-v2 Direct | +9.056 | [8.675, 9.471] | 明确改善 |
| Indirect | PSEM-v1 Indirect | -0.574 | [-0.718, -0.410] | 明确下降 |
| Indirect | 原始 OWT Indirect | -0.448 | [-0.580, -0.307] | 明确下降 |
| Indirect | PSEM-v2a Indirect | +0.049 | [-0.107, 0.224] | 无法区分，视为持平 |
| Indirect | PSEM+LossBalance-v2 Indirect | +5.265 | [4.978, 5.610] | 明确改善 |

PSEM-v3 的 Indirect 比自身 Direct 高 1.155 个百分点，说明 Triplet/Delta 目标相对更偏向保留 Indirect；但由于 Direct 本身下降更大，这个差值不能单独证明分解质量改善。它没有提升 AutoPET 的总体上限，主要退化来自 `label_4`：Direct 75.67%，比 v1 Direct 低 3.39 个百分点。Direct HD95 从 v1 的 10.277 mm 降到 7.055 mm，但 Dice 和 NSD 同时下降，因此只能说极端边界误差改善，不能说总体分割更好。


## AbdAutoPET 2D 重建指标

按 Whole L2 从低到高排序。L2/LPIPS 越低越好，PSNR/SSIM 越高越好；`label_4-only L2` 用于检查本次最明显退化类别的重建质量。

| 排名 | 实验 | Whole L2 | Whole LPIPS | Whole PSNR | Whole SSIM | label_4-only L2 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | PSEM-v1 | **0.00015039** | **0.01907** | **38.624** | 0.91971 | 0.00019494 |
| 2 | PSEM-v3 Triplet+Loss3 | 0.00017368 | 0.02313 | 37.979 | 0.90798 | 0.00025977 |
| 3 | PSEM-v2a Query20 | 0.00017749 | 0.02356 | 37.908 | 0.92126 | **0.00019280** |
| 4 | PSEM+LossBalance-v2 | 0.00026088 | 0.03334 | 36.373 | **0.92377** | 0.00058690 |

PSEM-v3 的 Whole L2 比 v1 高 15.49%，但比 v2a 低 2.15%，说明总体像素误差与 v2a 接近、仍不如 v1。更关键的是 `label_4-only L2` 比 v1/v2a 分别高 33.25%/34.73%，与 `label_4` Direct Dice 的下降方向一致。另一方面，SSIM 与 Dice 并不单调一致：LossBalance-v2 的 Whole SSIM 最高但分割最差，因此不能用单个重建指标替代类别分解评估。

## AbdAutoPET 3D Fixfr4 排名

目前只有 PSEM-v1 完成同协议 200 病例评估。类别顺序为 `label_1/label_2/label_3/label_4`。

| 排名 | 实验 | 读出 | post Dice | post NSD | post HD95 (mm) | 4 类 post Dice |
|---:|---|---|---:|---:|---:|---|
| 1 | PSEM-v1 | Direct | **87.583%** | 82.450% | 16.677 | 94.54 / 91.71 / 90.33 / 73.75 |
| 2 | PSEM-v1 | Indirect | 87.519% | **83.100%** | **12.238** | 94.99 / 90.77 / 91.90 / 72.41 |

该表不支持“3D 不如 2D”的方法结论，因为 Fixfr4 的输入、训练和空间评估协议与 2D 不同；它只记录当前已完成结果。

## WORD 2D 排名

相同 24 病例、224 输入和 Common8 正式评估。类别顺序为脾脏/右肾/左肾/胆囊/食管/胰腺/肝脏/胃。

| 排名 | 实验 | 读出 | post Dice | post NSD | post HD95 (mm) | 8 类 post Dice |
|---:|---|---|---:|---:|---:|---|
| 1 | PSEM-v3 Triplet+Loss3 | Direct | **68.353%** | 57.516% | 15.133 | 87.84 / 87.39 / 86.65 / 25.11 / 46.65 / 52.98 / 91.49 / 68.72 |
| 2 | PSEM-v3 Triplet+Loss3 | Indirect | **68.244%** | **58.851%** | 26.001 | 86.89 / 85.06 / 82.67 / 32.13 / 41.86 / 56.38 / 91.59 / 69.37 |
| 3 | LossBalance-v3a（对照） | Direct | 68.216% | 55.495% | **15.079** | 87.25 / 86.55 / 85.15 / 24.94 / 46.42 / 53.18 / 91.29 / 70.94 |
| 4 | PSEM-v2 LossBalance-v3 | Direct | 67.209% | 54.177% | 15.080 | 86.79 / 85.38 / 84.09 / 24.11 / 47.05 / 51.20 / 90.58 / 68.46 |
| 5 | LossBalance-v1（对照） | Direct | 64.990% | 47.560% | 16.598 | 85.63 / 82.81 / 82.27 / 24.64 / 33.86 / 53.27 / 89.23 / 68.21 |
| 6 | 原始 OWT（对照） | Indirect | 58.309% | 49.866% | 80.962 | 88.31 / 86.83 / 85.55 / 0.00 / 0.00 / 43.73 / 92.28 / 69.78 |
| 7 | LossBalance-v3a（对照） | Indirect | 56.796% | 47.752% | 69.545 | 86.80 / 86.24 / 83.68 / 0.00 / 0.16 / 40.75 / 91.66 / 65.08 |
| 8 | 原始 OWT（对照） | Direct | 55.072% | 44.279% | inf | 86.00 / 79.30 / 79.98 / 0.00 / 0.00 / 38.43 / 91.02 / 65.84 |
| 9 | PSEM-v2a Query20 | Direct | 52.311% | 40.155% | inf | 83.56 / 66.20 / 79.83 / 0.00 / 2.72 / 31.95 / 90.36 / 63.86 |
| 10 | PSEM-v2 LossBalance-v3 | Indirect | 52.216% | 41.632% | 84.054 | 84.49 / 81.76 / 79.66 / 0.00 / 0.00 / 21.16 / 90.36 / 60.30 |
| 11 | PSEM-v2a Query20 | Indirect | 49.889% | 38.862% | 91.771 | 84.74 / 79.90 / 79.25 / 0.00 / 0.00 / 0.41 / 90.99 / 63.83 |
| 12 | LossBalance-v1（对照） | Indirect | 41.575% | 18.809% | 372.641 | 57.08 / 48.45 / 50.05 / 5.79 / 8.84 / 27.99 / 81.56 / 52.85 |
| 13 | PSEM-v1 | Direct | 36.266% | 29.127% | 319.094 | 37.75 / 43.57 / 45.11 / 10.83 / 9.89 / 45.87 / 46.97 / 50.13 |
| 14 | PSEM-v1 | Indirect | 34.866% | 17.367% | 355.002 | 48.66 / 40.81 / 42.65 / 0.78 / 0.74 / 18.28 / 79.73 / 47.28 |
| 15 | PSEM+LossBalance-v2 | Indirect | 21.129% | 7.103% | 375.216 | 28.79 / 20.57 / 21.84 / 1.49 / 0.18 / 1.17 / 63.60 / 31.39 |
| 16 | PSEM+LossBalance-v2 | Direct | 8.880% | 6.992% | 424.928 | 8.26 / 6.75 / 7.16 / 0.41 / 2.22 / 2.70 / 31.17 / 12.36 |

PSEM-v3 在 WORD 上相对 PSEM-v2+LossBalance-v3 的 Direct 增益为 1.14 个百分点，相对独立 LossBalance-v3a 的 Direct 增益仅 0.14 个百分点；关键变化是 Indirect 从 52.22% 提高到 68.24%（相对独立 LossBalance-v3a 也提高 11.45 点），并与 Direct 只差 0.11 个百分点。这直接支持 Triplet 的设计目标：让 Direct、Context、Plus 在同一源样本上成对训练，使 `Plus - Context` 与目标器官响应一致。Indirect 仍明显依赖后处理，因此报告时必须同时保留 raw/post，而不能把全部提升归因于网络。

## 已完成实验登记

| 版本 | 核心改动 | 数据集与正式评估 | 当前判断 |
|---|---|---|---|
| PSEM-v1 | 只允许 GT-present 类进入逐样本穷举删除日程；保持原 L2+LPIPS | AutoPET 2D、AutoPET 3D Fixfr4、WORD 2D | AutoPET 2D 当前最优；WORD 存在训练/推理查询状态错配 |
| PSEM-v2a Query20 | 20-slot 全 token-bank 查询日程，包含正/负 Direct、leave-one-out、whole、background、organs 和随机子集 | AutoPET 2D、WORD 2D | AutoPET 与 v1 接近；WORD 大器官恢复，但小器官仍严重失败 |
| PSEM+LossBalance-v2 | PSEM-v1 加按样本频率加权的 ROI 损失 | AutoPET 2D、WORD 2D | 两个数据集均明显退化，判定该组合无效 |
| PSEM-v2 LossBalance-v3 | Query20 加只优化 present-and-kept 正 ROI 的 Loss3 | WORD 2D | Direct 提升到 67.21%，但 Indirect 仍只有 52.22% |
| PSEM-v3 Triplet+Loss3 | 每个源样本构造 Direct/Context/Plus，优化三分支 Loss3 与 `0.1 × Delta loss` | WORD 2D；AutoPET 2D | WORD 修复 Indirect；AutoPET 与 v2a 持平、低于 v1，收益不具跨数据集一致性 |

### PSEM-v3 正式运行

| 数据集 | 训练 | 评估 | 资源/规模 | 状态 |
|---|---|---|---|---|
| WORD 2D | XEC Job `114767` | XEC Job `114768` | 2×A800；1200 epochs；24 病例评估 | 完成 |
| AbdAutoPET 2D | sifansong/XEC Job `121484` | sifansong/XEC Job `124936` | 4×A800；有效 source batch 192；1200 epochs；200 病例评估 | 完成；BF16/LPIPS-FP32 数值修复版本 |

AutoPET 原 FP16 训练链曾在约 epoch 500 附近出现非有限值。最终有效版本保留模型、数据、Triplet/Loss3 定义和优化预算，只修改数值执行策略：BF16 autocast、LPIPS FP32、梯度裁剪与逐分量有限值检查。该变化必须随 checkpoint 一起记录，不能把最终模型描述成未经修复的原 FP16 运行。

## 结果文件与详细记录

- PSEM-v1：`archive/HISTORY.md`
- PSEM-v2a：`archive/HISTORY.md`
- PSEM-v3：`archive/HISTORY.md`
- 正式结果以各实验目录的 `step3_segmentation_summary.csv`、`step3_segmentation_per_case.csv` 和 `protocol.json` 为准。

## 下一步

1. PSEM-v3 AutoPET 至少补 2 个 seed；当前病例 bootstrap 不包含训练随机性，不能据单 seed 宣称稳定差异。
2. 优先分析 AutoPET `label_4` 的 false negative、预测体积比和 Direct/Indirect 差分图，确认 Direct 回退来自类 token 读出还是 Triplet 的预算分配。
3. 若要判断 PSEM-v3 是否适用于 3D，需要在 AutoPET 3D Fixfr4 上补同协议 v3 对照；不能拿 2D 与 3D 的当前单次结果直接排名。
4. WORD 继续报告 raw/post 两套结果，并做固定后处理的消融，量化 Indirect 提升中模型与后处理各自的贡献。
`````

<a id="lossbalance-experiments"></a>

### experiment/lossbalance:docs/EXPERIMENTS.md

来源：`experiment/lossbalance @ 144b427c79737257406e86ea3e94d2a33e8fae7e`。原文完整保留。

`````text
# LossBalance 合并说明

v2只改变ROI重建损失：按阳性样本频率N_c^(-alpha)加权，alpha=.5、ratio cap4，
归一化使期望前景权重质量为1；lambda_roi=1。背景不参与，batch分母固定。
v3a仅优化present且kept器官的positive ROI，lambda_positive=.25；removed仅监控，
仍受global L2+LPIPS监督。保持原OWT架构与mask日程；不加PSEM或Delta。
这两者不是OrganSlot的Tversky/BalancedFocal分割损失。

WORD224控制配置：20 tokens/class、有效batch64、blr1e-4、1200epochs。
记录指标必须包括Direct/Indirect raw/post、precision、recall、volume ratio、重建误差。
v2源记录的训练1586123完成，3D full1586124启动前取消；v3源记录停留旧pending，
不能当作今天状态。本分支原docs不含完整正式八类结果，不凭其他协议补排名。
跨版本已汇总结果见experiment/psem分支RESULTS；需要汇报时回到对应原始结果核验。

公式推导、样本频率/权重表、单元测试和历史任务完整保存在
[历史原文](archive/HISTORY.md)。不因归档删除数据、权重或实验可追溯性。
`````


来源提交：`43cad3025610c37d28f8395802157a87ec3d9b7b`。
原文逐字保留；所有状态、指令和路径仅反映各自历史时点。
不要默认读取本文件；先读 ../README.md。

- [docs/ARM_E_MAE_AND_RECON_20260908_CN.md](#source-1)
- [docs/ARM_E_MAE_DDP_RECOVERY_20260910.md](#source-2)
- [docs/ARM_F_2D_3D.md](#source-3)
- [docs/BRANCH_CONSOLIDATION_20260907.md](#source-4)
- [docs/ORGAN_SLOTBANK_ARCHITECTURE_CN.md](#source-5)
- [docs/ORGAN_SLOTBANK_EXPERIMENT_LOG.md](#source-6)
- [docs/ORGAN_SLOTBANK_EXPERIMENT_SUMMARY_CN.md](#source-7)
- [docs/ORGAN_SLOTBANK_IMPLEMENTATION_HANDOFF.md](#source-8)
- [docs/ORGSLOT_112_NATIVE_CROP448_EXPERIMENT_PLAN.md](#source-9)
- [docs/ORGSLOT_ARMB_3D_WORD070_ROI20_CN.md](#source-10)
- [docs/ORGSLOT_ARM_E_CROSS_ATTENTION_CN.md](#source-11)
- [docs/ORGSLOT_AUTOPET_MAE_TRANSFER_CN.md](#source-12)
- [docs/ORGSLOT_CONFIGURATION_CONTRACT_CN.md](#source-13)
- [docs/ORGSLOT_WORD070_ARM_E_MULTISCALE_PIXEL.md](#source-14)
- [docs/ORGSLOT_WORD070_MULTIQUERY_EXPERIMENT.md](#source-15)
- [docs/ORGSLOT_WORD070_QUERYMASK_3D_EXPERIMENT_CN.md](#source-16)
- [docs/ORGSLOT_WORD070_QUERYMASK_EXPERIMENT.md](#source-17)
- [docs/ORGSLOT_WORD070_SMALL_ORGAN_LOSS_CN.md](#source-18)
- [docs/ORGSLOT_WORD448_JOINT_SLOT_SEG_V1.md](#source-19)
- [docs/ORGSLOT_WORD448_RETAINED_MULTISCALE_FOCAL_V2.md](#source-20)
- [docs/ORGSLOT_WORD448_ROI20_LOSS3_EXPERIMENT.md](#source-21)
- [docs/ORGSLOT_WORD_COMMON8_SOTA_CN.md](#source-22)
- [docs/ORGSLOT_WORD_RESOLUTION_ABLATION_LOG.md](#source-23)
- [docs/PIXEL_PE_ABLATION.md](#source-24)
- [docs/PIXEL_PE_NUMERICS_20260908.md](#source-25)
- [docs/THREED_SLICEWISE_DIAGNOSTIC.md](#source-26)
- [README.md](#source-27)
- [README_EXPERIMENTS_CN.md](#source-28)

<a id="source-1"></a>

## docs/ARM_E_MAE_AND_RECON_20260908_CN.md

SHA256: `7c2d369982cfe2fbc16b8d97c1e9838dddf68f6caa1a1bb907a30be0aca9c5f0`

````text
# Arm B MAE recon 补齐与 Arm E MAE 三臂实验（2026-09-08）

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

Head 数字沿用此前已核验汇报；本次新核验的是 recon JSON。
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

## Arm E MAE 三臂
用户明确要求在原始 Arm E 上也比较 scratch / encoder / encoder+decoder。
不加 cross-attention，不加 PE，不加载 pixel decoder、Collector/TGEnc/AHER 权重。

| 组 | 目的 | 账号/集群 | 训练 | 评估 | 状态 |
|---|---|---|---|---|---|
| E0 scratch | 多尺度空间分支无预训练对照 | bolinren19 / SIP | 2892114 | 2921054 | 训练 epoch498，评估 afterok 等待 |
| E1 encoder | 隔离 MAE encoder 收益 | sifansong / XEC | 133896 | 133898 | 训练 PENDING (Priority)，评估 afterok |
| E2 encoder+decoder | 检验共享重建 decoder 额外收益 | sifansong / XEC | 133897 | 133899 | 训练 PENDING (Priority)，评估 afterok |

E0 正式评估已提交：1×A800、10 CPU、128GB、angelosstefanidis/8a800；
afterok:2892114，依次 train calibration / full head / full recon。
E1/E2 每条4×A800、24CPU、192GB、sifansong/8gpus。
两条正式训练于2026-09-08直接提交，没有申请 smoke。

固定：WORD 0.7×0.7×2、input/ROI 448/384、ROI probability0.2、small-organ loss、
lambda_seg0.01、retained、有效 batch192=4×8×6、118800更新、
warmup5940、blr1e-4（实际lr7.5e-5）、seed0、FP16 GradScaler、无新增梯度裁剪。
MAE checkpoint SHA256:
0b3571a3e79095686a0b12649735a298aaeb8a0b3bb93a47f3484439ac4a1ec6

## 兼容快照与复现
本地运行快照：.worktrees/orgslot_arm_e_mae_20260908。
远端运行快照：
/gpfs/work/aac/sifansong/worktrees/orgslot_arm_e_mae_20260908
；上传归档为
/gpfs/work/aac/sifansong/arm_e_mae_20260908.tar
，SHA256 为
d57d709a1822f398450607555c40ba357cd0cbec427f377b240055e49f1d4a1d。
基于原 Arm E 提交81adba2，而非直接使用改变了数值策略的最新版训练引擎。
原始训练引擎和 OrganSlotEmbed.py 与 E0 运行目录 SHA256 完全一致。
仅补 MAE 参数/严格加载/加载记录、Git provenance 容错和源文件哈希。
加载器 util/mae_transfer.py 及数据核验器取自统一分支01f1666。
入口：
- slurm/orgslot/train/arm_e_mae_sifan_xec.sbatch
- slurm/orgslot/eval/arm_e_mae_unified.sbatch
- scripts/orgslot/arm_e_mae_81adba2.patch：旧 main/launcher 的完整兼容补丁。
在81adba2归档上应用补丁，再加入上述加载器和入口即可复现。
不要把训练入口指向任意后续 worktree，否则会失去 E0 可比性。

已完成 Python compile、bash -n 和旧新引擎哈希核验；没有新增 GPU smoke。
最终以三组完整checkpoint802同协议结果判断 Arm E MAE 是否超过既有项目内最好结果；
不能用当前epoch498训练loss推断最终SOTA。
````

<a id="source-2"></a>

## docs/ARM_E_MAE_DDP_RECOVERY_20260910.md

SHA256: `58072f74cf98848962447e7973517a16d1a8927649c45104e9cd359e171ee2a6`

````text
# Arm E MAE encoder+decoder recovery

133897 failed near epoch541. Ranks 0/2/3 report the same NCCL ALLREDUCE sequence
27417850 timeout; follow-on stacks are in DDP buffer synchronization. No initiating
rank1 stack is available. Transport failure versus collective desynchronization is
not proven. Do not describe the issue as a confirmed NaN or fixed root cause.

Mitigations: sorted packed metric reduction with fixed-size schema handshake
(all ranks fail together on schema mismatch), opt-in disabling of per-forward
buffer broadcasts only when buffers are exactly frozen LPIPS shift/scale. The
initial DDP state synchronization and parameter gradient reduction remain enabled.
Resume logs enable distributed DETAIL diagnostics to expose mismatched collectives.

Resume checkpoint500, including optimizer and FP16 GradScaler, at epoch501;
microbatch8 x 4 GPUs x accumulation6 =192, same loss, ROI20, spacing0.7, LR schedule,
118800 total updates. Clipping disabled to preserve historical no-clipping FP16
policy. GradScaler overflow skip remains allowed and logged as amp_overflow.
Save every25 epochs. RNG/worker state was not saved by the old checkpoint;
this is not a bitwise replay of the old trajectory.

Use a new snapshot from latest origin/feature/orgslot. Existing runtime remains
unchanged because encoder-only133/134494 shares it. New afterok evaluation uses
arm_e_mae_unified.sbatch: train calibration, test head, reconstruction.
CPU Gloo regression checks reverse dict insertion order and mismatched schemas;
GPU stability remains unproven until the resumed job crosses the old failure.
````

<a id="source-3"></a>

## docs/ARM_F_2D_3D.md

SHA256: `0b5949136bca918878aadb20bea44c1fca01bd63f1096527995ff8c6027cbf9a`

````text
# Arm F: organ-conditioned two-stage attention

Branch: `experiment/orgslot-arm-f`, based on `247ded90e5d49603f68ace2153b9a6fad6ba2d8d`.
No training results yet. No Slurm jobs submitted by this implementation.

## Model configuration

Use the existing training entry `main_pretrain_orgslot_common8_a100.py`:

```
--slot_head_type arm_f_attention --slot_head_channels 128
--token_factor 20 --query_refinement none --pixel_pe none
--seg_loss_type small_organ --lambda_seg 0.01
--tversky_alpha_fp 0.3 --tversky_beta_fn 0.7
--balanced_focal_weight 0.5 --hard_negative_ratio 0.02
--negative_slice_weight 0.1 --focal_alpha 0.75 --focal_gamma 2
--organ_roi_aug --organ_roi_probability 0.2
--input_size 448 --global_crop_size 448 --roi_crop_size 384
--expected_spacing 0.7 0.7 2 --amp_dtype bf16 --clip_grad 1
```

2D: `--dimension 2D --segmentation_unit slice`.
3D: `--dimension 3D --fix_frame 4 --temp_stride 1 --segmentation_unit slice`.
Linear readout control: replace head type with `arm_f_linear`.
Arm F always uses fixed axial sin/cos attention PE: y/x in 2D, t/y/x in 3D.
`pixel_pe none` refers to the older decoder's optional PE, not disabling Arm F PE.
Arm F uses its own three refinement blocks; `query_refinement cross_attn` remains
an Arm E-only option and must not be used here.

Dataset CSV, ROI index, preprocessing summary, LPIPS weights, output paths and
initialization must be selected for the actual account/cluster. Do not copy
another account's paths. Match effective batch, LR, update budget and initial
encoder/reconstruction weights to the chosen comparison arm; no claim of fair
comparison is made merely from equal epochs across 2D/3D.

## Shape and implementation

Collector/TGEnc retains image-conditioned `[B,20,D]` tokens. Input projection
and the existing per-slot identity produce `[B,20,128]`. Three separately
parameterized blocks read P16, fused P8, fused P4. Each block is pre-norm cross
attention, organ-local self-attention, FFN. All decoder weights are shared across
organs. No Hungarian matching, instance loss or mask gating is introduced.

For 448 input: maps are `[B,128,28,28]`, `[B,128,56,56]`, `[B,128,112,112]`.
For 3D: maps are `[B,128,4,28,28]`, `[B,128,4,56,56]`, `[B,128,4,112,112]`.
The 3D stem applies the original Arm E 2D convolutions slice-wise. Cross-attention
can access all slab positions with fixed t/y/x PE, but there is no added temporal
Conv3D mixing. This new stem is a separate architectural change versus 3D Arm D;
use a matched Arm F linear readout to isolate reverse attention effects.
Temporal encoder stride >1 averages input slices to the encoder time grid before
the stem and interpolates final logits back; the intended baseline uses stride 1.

Stage B: P4 queries read updated tokens; output retains a P4 residual, then FFN,
shared LayerNorm/linear classifier, interpolation and original per-slot calibration.
Every organ has one output channel; raw/calibrated dictionaries and active-row
selection are unchanged. Reconstruction/AHER/fusion are retained.

Linear control: mask MLP -> 20 dot-product scores -> learned linear token fusion.
Only the fused mask is supervised. This is equivalent to a fused dynamic mask
embedding, not pixel-adaptive token selection. Token indices/count must be stable.

Existing 2D/3D head evaluators rebuild from saved `slot_head_type` and channels;
Arm F checkpoints require this branch's code and strict loading. No partial
loading of an Arm E checkpoint as a full Arm F resume is supported.

## Numerical and resource behavior

Standard `nn.MultiheadAttention(batch_first=True, need_weights=False)`, four heads,
FFN ratio 4, dropout 0. Attention/FFN/readout run locally in FP32 for compatibility
with the cluster's older PyTorch; spatial convolutions retain ambient AMP.
Interpolation uses the existing FP32-safe helper. Never call `model.bfloat16()`;
keep FP32 parameters and use autocast. No full attention heatmaps returned in the
training path. Attention weights are not foreground probabilities.

P4 features and FFN activations scale per active organ; loops do not free saved
autograd activations before backward. 3D has four times as many spatial positions
as 2D. Full-size GPU memory, speed and DDP preflight remain unverified; choose
microbatch from measurement, retaining effective batch via accumulation.

## Validation and next experiments

CPU tests cover 2D/3D attention and linear heads, finite backward, all new decoder
parameters receiving gradients, Collector gradients, active rows, unchanged
reconstruction shape, strict checkpoint roundtrip, 20-token sensitivity, temporal
PE identity and BF16 inputs to attention. CPU BF16 whole-stem execution is not
claimed: the installed CPU GroupNorm rejects mixed BF16/FP32 inputs.

Validated on 2026-09-10: 33 unittest cases (`test_arm_f`, `test_orgslot_model`,
`test_query_cross_attention`) passed, plus all 7 standalone slice-wise loss and
3D evaluator regression functions. `git diff --check` passed. No GPU/DDP run.

Compare Arm E, Arm F linear, Arm F attention under the same protocol. Additional
single-scale and depth-matched ablations are not implemented yet. Monitor eight
organ Dice, precision/recall, empty-slice FP, volume ratio, memory and throughput.
No SOTA or efficacy claim before full evaluation.

Future formal jobs use maximum allowed walltime, verified on submission: SIP
4a800/8a800 7 days; XEC 8gpus 5 days, 4gpus 7 days. Existing jobs unchanged.
Formal submission scripts: `slurm/orgslot/train/arm_f.sbatch` and
`slurm/orgslot/eval/arm_f.sbatch`, with explicit `ARM_F_TARGET`,
`ARM_F_DIMENSION`, `EXPERIMENT_WORKDIR`, `ARM_F_COMMIT` exports.
2D: microbatch 2 x 4 GPUs x accumulation 24 = 192 slices.
3D: microbatch 1 x 4 GPUs x accumulation 12 = 48 slabs (192 slices).
Both use scratch initialization, actual LR 7.5e-5, 118800 updates. Smaller
microbatches may change legacy_batch TGR behavior; do not claim exact equivalence
from matching effective batch alone. The first evaluation is fixed threshold 0.5.
The branch is fast-forwarded to origin/feature/orgslot before runtime snapshots
are created and checked; existing worktrees are not updated.
````

<a id="source-4"></a>

## docs/BRANCH_CONSOLIDATION_20260907.md

SHA256: `7f129cd42c9be627ea3173f8bf53808d480a5d9dc9e49c5b1611260d90ab5608`

````text
# Branch consolidation: 2026-09-07

Recommended development branches: main, feature/orgslot, experiment/psem, experiment/lossbalance.
feature/orgslot contains all OrganSlot histories/configurations. Development
worktree: .worktrees/orgslot_integrated. Historical experiment worktrees remain at
their original commits in detached HEAD state, preserving logs, checkpoints and uncommitted changes.
Existing Slurm tasks continue to use their original directories.

Consolidation and eight archive tags have been pushed to GitHub. After explicit
user authorization, all eight old local and six corresponding remote experiment
branch refs were deleted. Remote deletion used atomic push with exact expected
SHAs; local deletion used merged-branch checks. Archive tags remain on GitHub.
The old orgslot_head_ab worktree is detached at 1b104f4; its files are preserved.

Archive tags use prefix archive/20260907/ and the suffix below:

| Former experiment branch / tag suffix | Original commit |
|---|---|
| orgslot-3d-slicewise | 8984c9e |
| orgslot-armb-3d | 6bee802 |
| orgslot-autopet-mae-transfer | b43c7f0 |
| orgslot-pixel-pe | 4e7bf7c |
| orgslot-querymask | 4d7c5e0 |
| orgslot-querymask-3d | ab3bf1f |
| orgslot-querymask-multiquery | 4d7c5e0 |
| orgslot-querymask-multiscale-pixel | 81adba2 |

Recover with `git switch -c <new-name> <archive-tag>` in an unused checkout.
No experiment result or checkpoint is deleted.

Unified controls: dimension, slot_head_type, pixel_pe, segmentation_unit,
mae_init_checkpoint and mae_init_scope. segmentation_unit defaults to slab
for legacy behavior; slice-wise and 3D PE scripts explicitly select slice.
2D loss is unchanged. Arm E remains 2D-only.

Validation: 51 model/PE/Arm E/loss tests, 6 MAE tests, 3 slice-wise checks,
shell syntax and git diff checks passed. GPU experiment validation is separate.
````

<a id="source-5"></a>

## docs/ORGAN_SLOTBANK_ARCHITECTURE_CN.md

SHA256: `b318439569364da96ff37aecf3795ec5383cfc68b3459d8ff2f7ea8e79130092`

````text
# E-OWT-Seg / OrganSlotBank 架构与实现说明

## 1. 一句话说明

OrganSlotBank 把原始 OWT 中“所有器官共用的一条联合 token 路径”，改成了
“每个器官拥有一个可以独立追加、独立训练、独立冻结和独立输出分割结果的
slot”。共享 ViT 图像编码器和共享重建解码器仍然保留，因此新增一个器官时不需要
重建整个网络，也不需要在增量阶段读取旧器官标注。

当前 v0 的目标不是一次完成最终论文实验，而是先证明四件事：

1. 增量训练的数据接口拿不到旧类和未来类 GT；
2. 新器官可以通过 append slot 加入，并且旧参数不会被暗中修改；
3. 每个 slot 既能参与 OWT 重建，也能直接输出器官二值分割；
4. 2D、Fixfr4 3D、checkpoint 和 tiny overfit 的工程链路全部可运行。

## 2. 原始 OWT 的数据流和限制

原始路径为：

```text
image
  -> shared ViT patch encoder (blocks1)
  -> one joint OrganCollector
  -> [B, S*K, C] joint organ token bank
  -> one shared TGEnc (blocks2)
  -> one shared AHER
  -> one spatial patch canvas
  -> shared Transformer decoder
  -> reconstructed image
```

其中 `S` 是类别数，`K` 是每类 token 数。不同类别只是在联合 token 序列中占据
不同的连续区间，后面的 TGEnc 和 AHER 仍由所有类别共享。原模型没有器官专属的
监督分割头，主要依靠 token 删除后的图像重建进行学习。

这会给类别增量学习带来三个结构问题：

- 类别数写进联合 collector 的输出通道和总 token 数，追加类别会改变已有张量形状；
- 新类训练会经过共享 TGEnc、AHER 和 decoder，容易改坏旧类表示；
- 没有每类独立 head，无法直接区分“旧表示是否保持不变”和“最终融合是否发生竞争”。

## 3. 新架构总览

新的数据流为：

```text
image
  -> shared ViT patch encoder
  -> patch feature z [B,N,C]
       |-> OrganSlot_background -> canvas_bg, logit_bg
       |-> OrganSlot_old_1      -> canvas_1,  logit_1
       |-> OrganSlot_old_2      -> canvas_2,  logit_2
       `-> OrganSlot_new        -> canvas_new,logit_new

selected slot canvases
  -> normalized additive fusion
  -> shared reconstruction decoder
  -> reconstructed image

calibrated per-slot logits
  -> stack in a fixed semantic order
  -> argmax
  -> final multiclass segmentation
```

共享部分是 patch encoder 和 reconstruction decoder；器官专属部分被封装为
`OrganSlot`，所有 slot 由 `OrganSlotBank` 管理。

## 4. 一个 OrganSlot 内部有什么

每个器官 slot 拥有以下完整路径：

```text
OrganCollector_s
-> K organ tokens
-> OrganWiseTGEnc_s
-> AHER_s
-> patch canvas_s
-> PatchBinaryHead_s
-> raw binary logit_s
-> scale_s * logit_s + bias_s
-> calibrated logit_s
```

### 4.1 器官专属 OrganCollector

每个 slot 从同一个 patch feature `z` 中独立收集 `K` 个该器官 token。这样新增器官
只会新增一个 collector，不改变任何旧 collector 的形状或权重。

### 4.2 器官专属 TGEnc

原始 OWT 有一组共享 `blocks2`。v0 为每个 slot 放置一组局部 Linear-Attention
block，并把默认深度明确设为 `slot_tg_depth=1`。其残差和归一化语义保持原 OWT：

```text
encoded = tokens + SlotNorm(TGEnc(tokens))
```

这里没有声称复制了原始六层 TGEnc。深度 0/1/2 是后续必须报告的消融项。

### 4.3 器官专属 AHER 和 canvas

每个 slot 的 AHER 把自己的 `K` 个 token 恢复成 `N` 个空间 patch feature：

```text
canvas_s: [B,N,D]
```

旧架构只有一个联合 canvas；新架构能明确观察每个器官对空间特征的独立贡献。

### 4.4 轻量二值分割头

`PatchBinaryHead` 使用 `LayerNorm + Linear(D,1)` 在 patch canvas 上产生一个通道的
logit，再用双线性或三线性插值恢复到输入分辨率：

```text
2D: [B,1,H,W]
3D: [B,1,T,H,W]
```

这个 head 很小，能用 Head-only baseline 检验性能究竟来自完整新 slot，还是只需要
在冻结特征上训练一个读出层。

### 4.5 Calibration

每类都有两个标量：

```text
calibrated_logit_s = scale_s * raw_logit_s + bias_s
```

初始值严格为 scale=1、bias=0。Base 阶段可以从可见 GT 学习 calibration；增量
minimal 默认冻结 calibration，因为只用新类 GT 时没有合法的旧类监督去调整旧类
calibration。只有显式启用时才训练新类 calibration。

## 5. 多个 slot 如何共同重建

设样本 `i` 保留的 slot 集合为 `R_i`。代码先对被保留的 canvas 求和，再除以保留
数量的平方根，最后通过共享 `LayerNorm`：

```text
fused_i = LayerNorm(sum(canvas_s, s in R_i) / sqrt(|R_i|))
```

而不是简单求和，是为了避免 slot 数越多，特征幅值线性变大。不能把全部 slot 都
删除；代码和测试都强制每个样本至少保留一个 slot。融合后的 patch canvas 进入原有
共享 decoder，输出 CT/PETCT 图像重建。

## 6. Slot-aware TGR

Base 阶段仍保留 OWT 的 token-group reconstruction 思想，但 mask 改成逐样本的
slot keep mask：

- 由 `sample_index + epoch + seed` 确定性地产生；
- 同一 batch 的样本可以保留不同 slot 子集；
- 保留数量在 1 到 S 之间，不会出现全删；
- 被删除 slot 对应的可见器官区域在重建 target 中置零；
- 被删除 slot 的分割 logit 不进入 segmentation loss，也没有梯度。

它与 PSEM 的“穷举 present-label 二进制组合”不是同一个算法。当前 v0 使用的是
确定性、逐样本、非空的随机外观调度，目的是先隔离 OrganSlotBank 自身的贡献。

增量 minimal 阶段暂时保留全部 slot，确保新器官 canvas 与冻结 decoder 的兼容性；
后续再单独加入 retain-new TGR，不能在第一轮把多项创新混在一起。

## 7. 严格标签隔离

这是当前实现最重要的实验约束。

原始 manifest reader 只负责读取图像和 raw label。训练 DataLoader 之前必须套上
`StrictVisibilityDataset`：

- Base 阶段只返回 Base 可见器官的二值 mask；
- provisional background 只由 Base 可见前景的补集构造；
- incremental Liver 阶段只返回 `visible_masks={liver}`；
- 包装后的 sample 不包含 `label` 或 `full_label`；
- 只有 evaluation wrapper 被允许返回完整 raw label。

因此增量 loss API 在机械上拿不到旧器官 GT。如果 batch 同时出现 old mask 或 full
mask，objective 会直接抛错，而不是静默使用。

病例拆分工具只解析路径中的 case ID，不读取像素或标签内容。训练和验证病例存在任何
交集时，runner 会在建 DataLoader 前终止。

## 8. 损失函数

### 8.1 Base 阶段

```text
L_base = L_reconstruction
       + lambda_lpips * L_LPIPS
       + lambda_seg * mean_retained_slot(Dice + BCE)
```

background 分割项默认权重为 0.25，避免面积巨大的背景主导所有 slot。分割损失只对
当前样本保留的 slot 计算。

### 8.2 增量 minimal

增量阶段只使用当前新器官 mask：

```text
L_inc_min = DiceBCE(logit_new, M_new)
          + lambda_rec * masked_MSE(reconstruction, image, M_new)
```

第二项只在新器官区域计算，用来检查新 slot 产生的 canvas 是否能被冻结 decoder
理解；它不是让新 slot 负责重建整张图。

### 8.3 Old-confidence suppression 消融

这项不读取旧 GT，而是读取冻结旧 head 的高置信预测：

```text
M_old = union(sigmoid(detach(old_logits)) > 0.7)
M_sup = M_old AND NOT M_new
L_sup = masked_BCEWithLogits(logit_new, 0, M_sup)
```

它抑制“新类侵占旧类高置信区域”的 false positive。若没有满足阈值的旧区域，loss
安全地返回有限的 0。

## 9. 类别增量的实际操作

### 9.1 追加新器官

```text
load Base checkpoint
-> append_slot("liver", raw_id)
-> copy background collector/TGEnc/AHER/head weights
-> reset new-slot calibration to identity (1,0)
-> restore semantic name/raw ID metadata
-> verify every pre-existing tensor byte级不变
```

从 background 初始化的原因是背景路径已经学会覆盖广泛解剖区域，比完全随机初始化更
接近可用的空间 collector/canvas。追加操作会在内部克隆 append 前的整个 state dict，
如果任何旧 tensor 改变就立即报错。

### 9.2 Ours

冻结 shared encoder、shared decoder、background 和所有 old slots；只训练新 Liver
slot。minimal 默认也冻结新 slot 的两个 calibration 标量。

### 9.3 Sequential FT

从同一个 Base checkpoint、同一个 background 初始化的新 slot 开始，但开放全部
参数，只用新类 GT 训练。它是用于观察灾难性遗忘的下界，不是另一个新方法。

### 9.4 Head-only

冻结新 slot 的 collector、TGEnc 和 AHER，只训练新 slot 的 `LayerNorm+Linear`
binary head。它用于排除“仅增加一个小 head 就足够”的解释。

### 9.5 冻结验证

runner 在训练前为每个 `requires_grad=False` tensor 计算 SHA-256；每个 epoch 后再次
计算并逐名称比较。missing、added 或 changed 任一非空都会终止训练。

## 10. Checkpoint 与原 OWT 迁移

OrganSlot checkpoint 保存：

- 完整 model state dict；
- optimizer state；
- epoch；
- 有序 slot names；
- semantic name、sanitized key 和 raw class ID metadata；
- stage/method 等额外信息。

同构 OrganSlot checkpoint 使用 `strict=True` 精确回载。原 OWT 初始化则生成显式迁移
报告：共享 patch encoder/blocks1/decoder 能按形状继承；联合 collector 的类别输出
通道按 `K` 切片给对应 slot；原 shared blocks2 的前 `slot_tg_depth` 层复制到各 slot；
原 shared AHER 复制到各 slot。所有未载入、形状不匹配和故意跳过的 slot 都记录在
JSON 中，不能静默吞掉。

## 11. 2D/3D 数据和评估

`OrganSlotManifestDataset` 支持：

- 2D `[3,H,W]` 图像和 `[1,H,W]` raw label；
- Fixfr4 3D `[3,4,H,W]` 图像和 `[1,4,H,W]` raw label；
- 数字 slice index，而不是字符串排序；
- image/mask case 和 slice 一致性检查；
- 固定空间尺寸检查；
- per-sample min-max 或 fixed-255 归一化。

评估侧已实现 binary Dice、calibrated multiclass argmax、按 case 数字 slice 聚合、
Old/New/All DSC 和 forgetting。完整命令行评估器及正式 before/after 表仍待 Gate-D/F。

## 12. 本轮实际完成的代码

核心架构和训练文件：

- `OrganSlotEmbed.py`：slot、bank、binary head；
- `OWT_models_orgslot.py`：共享编码/解码、slot forward、canvas fusion、append/freeze；
- `losses_orgslot.py`：Base、incremental、suppression 和 background loss primitive；
- `engine_pretrain_orgslot.py`：stage-explicit objective、有限步 smoke、验证；
- `main_pretrain_orgslot.py`：synthetic/real、2D/3D、Ours/Sequential FT/Head-only、
  manifest checksum、病例隔离、best/last checkpoint、冻结哈希审计；
- `datasets/orgslot_manifest.py`：真实 CSV adapter；
- `util/label_visibility.py`：严格标签可见性边界；
- `util/slot_tgr.py`：逐样本 slot schedule 和 target；
- `util/checkpoint_orgslot.py`：保存、精确回载、迁移、append、hash；
- `eval_orgslot.py`：head-based metric primitive；
- `tools/split_orgslot_manifest.py`：确定性病例拆分；
- `tools/run_orgslot_tiny_overfit.py`：tiny memorization 与增量冻结审计。

所有实现均在独立 `feature/orgslotbank-v0` worktree 中。原始
`OWT_models.py`、`OrganEmbed.py`、`engine_pretrain.py` 和 `main_pretrain.py` 未修改，
仍作为受控基线。

## 13. 当前验证结果

- CPU 单元测试：32/32 通过；
- 2D 和 Fixfr4 3D synthetic forward/backward：通过；
- 真实 2D `[B,3,224,224]` 单 batch debug-width forward/backward：通过；
- 真实 Fixfr4 `[B,3,4,224,224]` 单 batch debug-width forward/backward：通过；
- Base -> append Liver -> Ours/Sequential FT/Head-only 完整编排 smoke：通过；
- Ours 冻结张量 138 个、Head-only 冻结张量 157 个，训练后 hash 完全一致；
- tiny overfit：loss `1.7362165 -> 0.1302057`，ratio `0.0749939`，organ Dice
  `0.9354208`；
- tiny checkpoint 精确回载：通过；
- incremental batch 仅有 Liver mask，旧参数 hash 不变：通过；
- 700 个病例按 seed=0 拆为 train 630 / validation 70，2D 和 3D 均无病例交集。

## 14. 尚未完成和不能过度宣称的部分

1. 真实单 batch 当前使用 debug-width 模型；正式 ViT-Base GPU 2D/3D smoke 尚未运行。
2. 尚未运行真实 Base 训练、Liver 增量训练和三方法机制表。
3. 尚未运行 suppression/background-plasticity 的正式消融。
4. 尚未形成完整的 case-level before/after 命令行评估和论文表格。
5. 尚未提交任何 OrganSlot 完整 1200-epoch 任务。
6. 配置中的 raw ID 语义来自计划假设：0=background、1=kidney_combined、
   2=spleen、3=pancreas、4=liver。当前仓库未找到独立数据说明验证这份映射；在确认
   前只能把真实数据 smoke 解释为 ID 级工程验证，不能作为器官语义结果。

因此当前准确结论是：OrganSlotBank v0 的结构、隔离约束、训练编排和 debug 数据链路
已通过门控；模型效果和增量抗遗忘优势仍必须由后续受控真实实验回答。
````

<a id="source-6"></a>

## docs/ORGAN_SLOTBANK_EXPERIMENT_LOG.md

SHA256: `b71b74534d190b55145cd256b2dad298aaa67e150c3c59d8b8d050ef025adf90`

````text
# OrganSlotBank v0 Experiment Log

- Branch: `feature/orgslotbank-v0`
- Starting commit: `6cdf9b575edd6845b7518c18d7ef9be9b89508f7`
- Scope: strict visibility, model/loss/checkpoint tests, 2D/3D forward-backward,
  and tiny overfit only. No full training submission.

## 2026-07-24 Gate-A implementation

Worktree: `/gpfs/work/aac/bolinren19/OD_OWT_orgslot`

Implemented:

- stage/class config and a training wrapper that never returns raw/full labels;
- provisional background built only from base-visible old masks;
- explicit fixed-K `OrganSlot` paths, appendable `OrganSlotBank`, binary heads,
  calibration, normalized additive per-sample canvas fusion, and shared decoder;
- per-sample Slot-Aware TGR, base reconstruction/segmentation loss, minimal
  Liver-only incremental loss, detached suppression ablation primitive;
- exact checkpoint save/load, original OWT migration, visible-slot transfer,
  append-from-background, freeze/hash audits, and head-based metrics;
- isolated synthetic smoke entry point. Formal datasets and long training were
  deliberately not started.

Validation commands:

```text
/gpfs/work/aac/bolinren19/.conda/envs/abdpet/bin/python -m unittest discover -s tests -p 'test_orgslot_*.py' -v
OMP_NUM_THREADS=4 /gpfs/work/aac/bolinren19/.conda/envs/abdpet/bin/python -m tools.run_orgslot_tiny_overfit --steps 250 --output-dir Results/OrganSlotBank/_tiny_overfit/gate_a_cpu --device cpu
sbatch slurm/orgslot/smoke/orgslot_gate_a_cpu.sbatch
```

Observed results before final checkpoint-resume test addition:

- 24/24 CPU tests passed, including 2D and Fixfr4-TS1 3D forward/backward and
  an original OWT forward regression. The final count is recorded below.
- tiny overfit: loss `1.7362165 -> 0.1302058` (ratio `0.0749940`), organ Dice
  `0.9354208`, exact checkpoint reload `true`;
- the same audit appended Liver from background, ran incremental backward with
  `visible_masks={liver}` only, and kept all frozen parameter hashes unchanged;
- final current-code Slurm smoke job `1561641`: `COMPLETED`, exit `0:0`, elapsed
  `00:00:27`; base total loss `1.6614242 -> 1.4783453`, reconstruction MSE
  `0.1312498 -> 0.1153881`, with finite head metrics and saved checkpoint.

Boundary/status:

- Gate A wiring is exercised on synthetic data only.
- Gate B/C, real-data manifests, Sequential-FT/Head-only comparisons,
  suppression/background-plasticity ablations, and full training are not run.
- Original `OWT_models.py`, `OrganEmbed.py`, `engine_pretrain.py`, and
  `main_pretrain.py` are unchanged; the regression test confirms legacy OWT
  forward remains runnable.

Final test result after adding base/incremental resume coverage: **25/25 passed**.

Resolved debug events:

- The first tiny-overfit target used non-patch-aligned, sample-varying regions.
  On the intentionally tiny 2x2 canvas it reached loss ratio `0.7845` and Dice
  `0`, so it failed the memorization gate. The test was corrected to the
  canonical repeated, patch-aligned single-pattern target; the success
  threshold was not relaxed, and the corrected run reached ratio `0.0749940`.
- The first Slurm submission was rejected before job creation because the
  account default QOS was invalid. The script was updated to the existing
  project association `account=sifansong`, `partition/qos=cpudebug`; current
  code then completed as job `1561641` with exit `0:0`.

## 2026-07-26 real-data wiring and controlled-baseline update

Added without modifying the legacy OWT files:

- deterministic 2D/Fixfr4 manifest adapter and strict case/slice checks;
- deterministic case-level train/validation split tool that never reads label
  pixels;
- real/synthetic stage-aware runner with manifest checksums, case-overlap
  rejection, 2D/3D model selection, finite-step smoke mode, best/last
  checkpoints, and per-epoch frozen-parameter SHA-256 audits;
- explicit Ours, Sequential-FT, and Head-only trainable scopes;
- automatic detached frozen-old prediction path for the suppression ablation;
- identity/frozen incremental calibration by default, matching the handoff;
- two new behavior tests for baseline scopes and detached old-confidence
  supervision, plus two case-split tests.

Current validation:

- **32/32** OrganSlot CPU tests pass;
- complete synthetic Base checkpoint -> append Liver -> Ours/Sequential-FT/
  Head-only orchestration passes;
- Ours trains 11,969/127,279 debug parameters and preserves 138 frozen tensor
  hashes; Head-only trains 97/127,279 and preserves 157 frozen hashes;
- current-code tiny overfit again reaches loss `1.7362165 -> 0.1302057`, ratio
  `0.0749939`, Dice `0.9354208`, exact reload true, frozen hashes unchanged;
- real 2D `[B,3,224,224]` debug-width one-batch forward/backward passes with
  finite total loss `1.9208560`;
- real Fixfr4 `[B,3,4,224,224]` debug-width one-batch forward/backward passes
  with finite total loss `1.8933843`;
- seed-0 case split contains 630 train and 70 validation cases. 2D records are
  70,560/7,840 and Fixfr4 records are 68,670/7,630.

The real-data checks above are engineering smokes, not accuracy results. They
use debug-width models and only one training batch. The assumed raw-ID semantic
mapping in `owt_legacy_debug.json` still needs independent dataset evidence
before formal organ-named claims or long training.

## 2026-08-15 WORD448 JointSeg Focal v2

目的：在 ROI20 reconstruction baseline 上加入每个 slot 独立的二分类 Focal
监督，替代出现前景全背景塌缩风险的 Dice+BCE tiny 目标；不启用融合后 Loss3。

固定配置：

- 分支 `experiment/orgslot-jointseg-focal-v0`，提交 `7d8343e`；
- XEC账号 `antengcai23`，Slurm account `sifansong`；
- WORD 2D，448输入，ROI20使用384 crop resize到448；
- `L = global L2 + LPIPS + 0.1 * slot focal`；
- `focal_alpha=0.75`，`focal_gamma=2.0`；
- `seg_supervision=all`，`lambda_bg_seg=0.25`；
- 真实keep/drop继续控制reconstruction target和fusion；
- 118800 optimizer updates，有效batch 192，2张A800。

验证：本地53项OrganSlot回归测试和15项Focal/TGR目标测试通过；XEC同环境15项
目标测试通过。逐slot日志新增预测体积、GT体积和阳性像素概率，防止仅凭总loss
误判全背景输出为成功。

任务链：

- smoke `117005`；
- 正式训练 `117006`，依赖smoke成功；
- reconstruction Direct/Indirect评估 `117007`；
- head训练集阈值校准 `117008`；
- 固定0.5及冻结校准阈值head测试 `117009`。
````

<a id="source-7"></a>

## docs/ORGAN_SLOTBANK_EXPERIMENT_SUMMARY_CN.md

SHA256: `2d33f7c552f054bc1017acc2e7443296f3f73f301cf6602d909f3f057aed0f54`

````text
# OrganSlotBank 实验总览

最后更新：2026-08-15 15:17 CST
分支：`feature/orgslotbank-v0`
工作区：`/home/Anteng/OD_OWT_orgslot`
环境：`/home/Anteng/miniconda3/envs/abdpet`
数据根目录：`/mnt/DATA-4/anteng`

本文档维护迁移到本地 GPU 服务器后完成的正式实验、工程验证和当前运行任务。
数值来自各实验的 `resolved_config.json`、`log.txt` 和评估 `results.json`。
`Results/` 中的 checkpoint、TensorBoard event 和可视化不提交到 Git。

## 1. 评价口径

- AutoPET：完整测试集 22,400 张切片、200 个病例。
- WORD：完整测试集 6,990 张切片、24 个病例、8 个前景器官。
- 表中的 case Dice 是“GT 存在病例的 Dice 均值”，不是 slice Dice 或 global Dice。
- Direct：直接保留/调用目标器官响应得到分割。
- Indirect：通过全器官重建与删除目标 slot 后重建的差分得到分割。
- `post` 表示应用既定形态学后处理；不同推理规则不能混在一起比较。

## 2. AutoPET 224 架构消融

三个正式训练使用相同 AutoPET 2D 训练/测试 CSV、224 输入、legacy batch mask、
有效 batch 96 和 1200 epochs。

| 实验 | 主要变量 | 最终训练 loss | Direct | Indirect |
|---|---|---:|---:|---:|
| OWT-compatible legacy-mask | 受控原始架构基线 | 0.02561 | **88.30%** | **84.89%** |
| OrganSlot linear-sqrt | slot 独立路径；画布除以 `sqrt(保留slot数)` | 0.02404 | 87.98% | 81.33% |
| OrganSlot + DiceBCE head | 联合重建和分割头监督 | 0.59723（seg 0.28084） | 使用 head 评价 | 不作为 head 指标 |

结论：

- OrganSlot 的 reconstruction Direct 只比 OWT 低 0.31 个百分点，但 Indirect 低
  3.55 个百分点，说明组合差分能力比直接重建更容易受架构变化影响。
- 联合监督分割头必须使用 logits 评价。完整测试集上：epoch 600 的
  binary-post/argmax-post 分别为 78.64%/78.42%；epoch 1199 为
  79.06%/77.95%。分割头确实学到语义，但没有超过 reconstruction Direct。
- epoch 500、600、1199 均已推理；binary threshold 和互斥 argmax 是两种不同
  推理规则，后续必须分别报告。

结果位置：

- `Results/OrganSlotBank/evaluation/AbdAutoPet_2D/reconstruction_threshold_v1/`
- `Results/OrganSlotBank/evaluation/AbdAutoPet_2D/head_v1/`

## 3. WORD 1x1x2 mm / 448 消融

三组实验共享 native 1x1x2 mm 预处理、center-448 数据、训练/测试 CSV、有效
batch 192 和 118,800 optimizer updates；均在 epoch 802 达到更新预算。

| 实验 | 变量 | Direct-post | Indirect-post |
|---|---|---:|---:|
| control linear-sqrt | 分母 `sqrt(当前保留slot数)` | 64.44% | 6.92% |
| fixed-sqrt K=9 | 分母固定为 `sqrt(9)` | 64.65% | 6.82% |
| ROI20 linear-sqrt | 20% 小器官 ROI crop | **75.20%** | **7.39%** |

统一使用固定 response threshold 0.02 和完整测试集。

ROI20 epoch 802 的 Direct-post 器官 Dice：

| 脾 | 右肾 | 左肾 | 胆囊 | 食管 | 胰腺 | 肝 | 胃 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 90.22% | 89.96% | 89.47% | 40.60% | 58.67% | 63.92% | 93.26% | 75.48% |

结论：

- 固定分母几乎没有改善 Direct，也没有挽救 Indirect，因此 LayerNorm/动态缩放
  不是 Indirect 崩溃的唯一原因。
- ROI20 将 Direct 提高 10.76 个百分点，并改善部分小器官形状，但 Indirect
  仍很低；定位改善与可组合差分是两个问题。
- ROI crop 仅用于训练。测试图像和 GT 使用同一 deterministic center-448
  变换，不存在“输入被 crop、GT 未 crop”造成的评价错位。
- 当前响应常能学习到相似形状但发生空间偏移；形态学后处理无法修复系统性偏移。

结果位置：

- `Results/OrganSlotBank/evaluation/Common8/WORD_2D/control_ckpt802_fixedthr002/results.json`
- `Results/OrganSlotBank/evaluation/Common8/WORD_2D/fixedk9_ckpt802_fixedthr002/results.json`
- `Results/OrganSlotBank/evaluation/Common8/WORD_2D/roi20_ckpt802_fixedthr002/results.json`
- `Results/OrganSlotBank/visualizations/WORD_2D/raw_small_organs_control_vs_roi20/`

## 4. 第三组：严格 head-only WORD 实验

### 实验问题

固定 ROI20 表征后，只训练专用分割读出，能否得到比 reconstruction threshold
更稳定的定位和小器官分割？该设计同时验证联合分割损失是否曾破坏重建表征。

### 训练设计

- 初始化：WORD ROI20 `checkpoint-802.pth`。
- 数据：同一 WORD native-112/center-448 manifests。
- 增强：同一 20% ROI crop，focus raw IDs 为 4/5/6。
- 输入时 9 个 slot 全部可见。
- 冻结：ViT、collector、TGEnc、AHER、slot token、fusion、reconstruction decoder。
- 训练：9 个 `PatchBinaryHead` 和各 slot 的 calibration scale/bias，共 20,763 参数。
- 损失：Dice+BCE；`lambda_seg=1`，背景权重 0.25。
- reconstruction 和 LPIPS 均关闭，forward 跳过 reconstruction decoder。
- AdamW，lr `1e-3`，weight decay 0。
- GPU 0/1；micro-batch 8/GPU；accumulation 12；有效 batch 192。
- 14,800 updates；warmup 740 updates；约 100 epochs。
- tmux：`word448-headprobe`。

### 启动前验证

- OrganSlot 测试：45/45 passed。
- 真实 WORD 12-slice tiny set 覆盖 raw class 0--8；胆囊/食管/胰腺阳性切片
  分别为 6/4/8。
- 100-update tiny overfit：Dice+BCE `2.0015 -> 0.6458`。
- checkpoint audit：54 个张量发生变化，全部属于 head/calibration；其余冻结
  参数逐张量完全相同。

### 当前状态（2026-08-15 15:17 CST）

- 状态：RUNNING，GPU 0/1。
- epoch 0：segmentation loss 1.42607，148 optimizer updates。
- 实测 ROI fraction 0.20196，与 20% 配置一致。
- epoch 1 进行中，阶段平均 segmentation loss 约 0.915。
- reconstruction loss 和 LPIPS 恒为 0，符合实验定义。
- 未发现 NaN、OOM、traceback 或 checkpoint 不兼容。
- 尚无测试 Dice；训练 loss 下降不能当作最终效果提升。

启动脚本：`scripts/orgslot/run_word_112_448_head_probe.sh`

有效运行目录：

`Results/OrganSlotBank/Common8/WORD_2D/OrgSlot_WORD112_input448_roi20_headonly_DiceBCE_from_roi20ckpt802_eb192_u14800`

异常启动记录：第一次启动的 ROI 增强正确，但 head-only 分支把 ROI 日志错误记为
0；在未完成一个 epoch 前停止并修复。第二次因 25743 端口未释放而没有开始训练。
当前有效任务从 epoch 0 在 25744 端口重新开始；这两次无效启动不计入结果。

### 完成标准

训练完成后，在不变的完整 WORD 测试集上同时评价 calibrated argmax 和 independent
binary threshold，分别给出 raw/post、每器官 Dice 和 8 类均值，重点检查胆囊、
食管、胰腺。正式对照为 ROI20 reconstruction Direct-post 75.20%；在完整推理前
不声明第三组优于现有结果。

## 5. 工程边界和后续维护

- 原始 `OWT_models.py`、`OrganEmbed.py`、`engine_pretrain.py`、`main_pretrain.py`
  保持可运行，作为受控基线。
- PSEM 未叠加进以上 OrganSlot 实验，避免一次改变多个变量。
- `Results/` 是本机实验工件，不进入 Git；复现依赖 launcher、resolved config、
  数据 manifest checksum 和源 checkpoint 路径。
- 每次正式评估后，应更新本文件中的状态、checkpoint、完整测试集规模、评价模式、
  per-organ 指标和结论，并保留失败/无效实验的原因。
- 早期 Gate-A/Gate-B、严格标签隔离和增量学习工程记录见
  `docs/ORGAN_SLOTBANK_EXPERIMENT_LOG.md`。
````

<a id="source-8"></a>

## docs/ORGAN_SLOTBANK_IMPLEMENTATION_HANDOFF.md

SHA256: `6fcb0bed9560aee9359726ab6051c297e770e61167169dcc36522cf409cf4959`

````text
# E-OWT-Seg / OrganSlotBank Implementation Handoff

Updated: 2026-07-24

This document is an implementation specification for a new coding session. It
combines the authoritative planning files with a code-aware audit of the
current OWT repository. It does not mean that OrganSlotBank has already been
implemented.

## 1. Status And Source Of Truth

Current repository:

```text
/gpfs/work/aac/bolinren19/OD_OWT
```

Current branch at handoff time:

```text
experiment/psem-v1
```

Important:

- PSEM and LossBalance are auxiliary OWT pretraining/masking experiments.
- They are not the E-OWT-Seg main method.
- Do not implement OrganSlotBank on top of the PSEM branch by default.
- Create a separate worktree and branch from the clean `main` baseline so
  pending/running PSEM jobs and their source tree are not changed.

Recommended setup:

```bash
git -C /gpfs/work/aac/bolinren19/OD_OWT worktree add \
  -b feature/orgslotbank-v0 \
  /gpfs/work/aac/bolinren19/OD_OWT_orgslot \
  main
```

Before running this command, verify that `main` is the intended clean OWT
baseline and that it contains any required environment compatibility fixes.

Read the plan in this order:

1. `00_README.md`
2. `CONTEXT.md`
3. `07_codeaware_grill_log.md`
4. `08_project3_implementation_handoff.md`
5. `10_integrated_final_plan.md`
6. `11_project3_experiment_protocol_handoff.md`

Plan directory:

```text
/gpfs/work/aac/bolinren19/2026-07/plan/2026-06-29_expandable_owt_incremental_seg
```

Authority rule:

- `10_integrated_final_plan.md` defines the final method and architecture.
- `11_project3_experiment_protocol_handoff.md` defines the newest experiment
  protocol, debug split, initialization comparisons, and execution order.
- `08_project3_implementation_handoff.md` defines code boundaries.
- `03`, `04`, and `05` are historical drafts and must not override `10` or
  `11`.
- `CONTEXT.md` contains useful terminology, but any stale side-path wording is
  superseded by the Minimal OrganSlotBank Refactor in `10`.

## 2. Research Question And Claim Boundary

Task:

```text
partial-label organ-incremental medical image segmentation
```

It is not online learning. At each stage, images may come from the same
dataset, but only a subset of labels is visible.

Base stage:

```text
visible labels = old organs only
future organs = hidden and included in provisional background
```

Incremental stage:

```text
visible labels = current new organ only
old organ GT = forbidden for training
future organ GT = forbidden for training
old organ GT = evaluation/reporting only
```

Main claim to test:

> Organ-wise token/restoration paths can be appended for new organs while old
> paths remain frozen, reducing old-organ drift under new-organ-only labels.

Do not claim:

- generic online learning;
- future-organ discovery;
- complete solution to class-incremental segmentation;
- complete solution to background shift;
- direct implementation-equivalent superiority over PCDD unless its official
  code, split, and runtime protocol are actually reproduced.

## 3. Current OWT Code Audit

The original path must remain runnable as a reference baseline.

### Current model

Relevant files:

```text
OWT_models.py
OrganEmbed.py
engine_pretrain.py
main_pretrain.py
datasets/dataset3D.py
```

Current architecture:

```text
image
-> ViT encoder blocks1
-> patch features [B, N, 768]
-> one joint OrganCollector
-> all class token groups [B, S*K, 768]
-> one shared blocks2 TGEnc
-> one shared AHER
-> one spatial canvas [B, N, 768]
-> shared decoder
-> reconstructed image
```

For the current base model:

```text
input size = 224
patch size = 16
K = token_factor = 20
2D N = 14*14 = 196
3D Fixfr4-TS1 N = 4*14*14 = 784
```

Current joint components:

- `OWT_models.py:190`: shared `blocks2`;
- `OWT_models.py:201`: one `organ_embed`;
- `OWT_models.py:212`: one `decoder_embed` AHER;
- `OWT_models.py:428`: all organ tokens are collected together;
- `OWT_models.py:438`: all retained tokens enter shared TGEnc;
- `OWT_models.py:453`: all retained tokens enter one AHER.

Current training limitations:

- `engine_pretrain.py` samples one dropped-class list for the entire batch.
- `random_selected_class` means dropped classes, despite its ambiguous name.
- Dropped class regions are zeroed in the reconstruction target.
- There is no supervised segmentation head or validation loop.
- Current segmentation scripts infer masks from reconstruction differences and
  hand-selected thresholds. This remains an auxiliary diagnostic only.
- `main_pretrain.py` silently loads checkpoints with `strict=False`.
- Random seeds are currently commented out.
- Train/validation/test manifests are not first-class stage configurations.

Current data details:

- CSV columns are `image_pth` and `mask_pth`.
- 2D images and labels are loaded as three-channel arrays.
- 3D samples are four adjacent numbered slices for `Fixfr4`.
- Runtime `RandomGenerator` may resize, but for the OWT-native v0 path the CSV
  should already point to offline-preprocessed 224x224 slices.
- The data loader currently returns the complete label map to the training
  engine. That is too easy to misuse for strict incremental experiments.

## 4. Target Architecture

### 4.1 Shared encoder

Input:

```text
2D: x [B, 3, 224, 224]
3D: x [B, 3, 4, 224, 224]
```

Output:

```text
z [B, N, C]
C = 768
N = 196 for 2D
N = 784 for Fixfr4-TS1
```

The encoder remains shared. It is trainable during base fitting and frozen
during the main incremental method.

### 4.2 One explicit slot per class

Slot bank:

```text
background
old organ 1
old organ 2
...
new organ slots appended at later stages
```

Each slot owns:

```text
OrganCollector_s
OrganWiseTGEnc_s
AHER_s
BinaryHead_s
calibration scale a_s
calibration bias b_s
```

Per-slot data flow:

```text
z [B,N,C]
-> OrganCollector_s
tokens_s [B,K,C], collector_attention [B,K,N]
-> OrganWiseTGEnc_s
encoded_tokens_s [B,K,C]
-> AHER_s
canvas_s [B,N,D], aher_attention [B,N,K]
-> BinaryHead_s
patch/pixel logit_s
```

Defaults inherited from OWT:

```text
C = 768
D = 768
K = 20
```

### 4.3 Normalized additive canvas

For a per-sample slot keep mask `keep [B,S]`:

```text
weighted_s = canvas_s * keep[:,s,None,None]
canvas_sum = sum(weighted_s)
count = keep.sum(dim=1).clamp_min(1)
canvas_all = canvas_sum / sqrt(count[:,None,None])
canvas_all = LayerNorm(canvas_all)
```

Then:

```text
canvas_all -> existing shared decoder -> reconstruction
```

The keep mask must be per sample. Do not take the union of slot choices across
the batch.

Memory rule:

- Do not stack every `[B,S,N,D]` canvas during normal training.
- Aggregate canvases in a loop and retain only logits/attention requested for
  losses or diagnostics.
- Return all canvases only under `return_diagnostics=True`.
- This matters especially for 3D where `N=784`.

### 4.4 Binary segmentation head

The plan locks the head location to the AHER canvas but does not lock its exact
layers.

Recommended minimal v0:

```text
LayerNorm(D)
-> Linear(D,1)
-> reshape patch logits
-> bilinear/trilinear interpolation to input resolution
```

Equivalent implementation:

```text
1x1 Conv2d/Conv3d after reshaping the canvas
```

Why this default:

- same head works for every slot;
- parameter cost is small;
- it isolates the value of OC/TGEnc/AHER;
- it works for both 2D and 3D;
- a larger convolutional head can be a later ablation.

Expected output:

```text
2D logit_s [B,1,H,W]
3D logit_s [B,1,T,H,W]
```

### 4.5 Organ-wise TGEnc depth

The plan locks organ-wise ownership but does not lock the exact slot depth.
Copying all six original `blocks2` layers for every class may grow parameters
too quickly.

Recommended v0 engineering default:

```text
slot_tg_depth = 1
```

Required ablation if the method becomes a paper result:

```text
depth 0 / 1 / 2
or shared frozen TGEnc + slot adapter
```

Whichever default is selected must be recorded with:

- total parameters;
- added parameters per new organ;
- GPU memory;
- inference time.

Do not silently describe a one-block implementation as a copied six-layer
TGEnc.

## 5. New File Layout

Keep the original files unchanged and runnable.

Recommended additions:

```text
OrganSlotEmbed.py
OWT_models_orgslot.py
losses_orgslot.py
engine_pretrain_orgslot.py
main_pretrain_orgslot.py
eval_orgslot.py
util/label_visibility.py
util/slot_tgr.py
util/checkpoint_orgslot.py
configs/orgslot/
tests/test_orgslot_model.py
tests/test_orgslot_visibility.py
tests/test_orgslot_losses.py
tests/test_orgslot_checkpoint.py
slurm/orgslot/
docs/ORGAN_SLOTBANK_EXPERIMENT_LOG.md
```

### `OrganSlotEmbed.py`

Implement:

```text
PatchBinaryHead
OrganSlot
OrganSlotBank
```

`OrganSlot` should contain `collector`, `tg_encoder`, `aher`, `head`, and
calibration parameters.

Use `nn.ModuleDict` with stable sanitized slot names. Store the semantic
name/raw class ID separately in checkpoint metadata.

Required methods:

```python
forward_tokens(z)
forward_canvas(z)
forward(z, output_size, return_attention=False)
set_trainable(enabled)
copy_from(other_slot, reset_head_bias=False)
```

`add_slot()` must be called before wrapping the model in DDP and before
constructing the optimizer.

### `OWT_models_orgslot.py`

Implement a pure model forward API rather than calculating all losses inside
the model.

Recommended output:

```python
{
    "reconstruction": reconstruction,
    "slot_logits": {name: logits},
    "slot_tokens": optional_dict,
    "slot_canvases": optional_dict,
    "collector_attention": optional_dict,
    "aher_attention": optional_dict,
    "slot_keep_mask": keep_mask,
}
```

Recommended methods:

```text
forward_encoder(x)
forward_slots(z, slot_keep_mask, return_diagnostics)
fuse_canvases(...)
forward_decoder(canvas)
forward(...)
append_slot(name, init_from)
freeze_for_incremental(old_slots, new_slots, background_policy)
parameter_report()
```

Separate encoder normalization and slot normalization. The current OWT model
reuses `self.norm` after both `blocks1` and `blocks2`; an explicit new model
should not accidentally couple those semantics.

### `losses_orgslot.py`

Implement independently testable functions:

```text
soft_dice_loss
dice_bce_loss
masked_mean
base_reconstruction_loss
base_segmentation_loss
new_region_reconstruction_loss
old_confidence_suppression_loss
background_new_complementarity_loss
```

Every masked loss must divide by the number of valid pixels/samples plus an
epsilon, not by the full image size.

### `util/label_visibility.py`

Implement:

```text
ClassSpec
StageSpec
build_visible_binary_masks
build_provisional_background
validate_raw_label_ids
```

Use a config file to map raw dataset IDs to semantic names. Do not assume the
OWT legacy raw IDs or BTCV IDs without checking the preprocessed masks.

Training batches should return only stage-visible masks:

```python
{
    "image": ...,
    "visible_masks": {slot_name: binary_mask},
    "case_id": ...,
    "slice_index": ...,
    "sample_index": ...,
}
```

The full label map may be returned only by an evaluation dataset/loader.
This makes old/future GT leakage mechanically harder.

### `util/slot_tgr.py`

Implement explicit names:

```text
sample_base_slot_keep_mask
sample_retain_new_keep_mask
build_base_reconstruction_target
build_incremental_pseudo_target
```

Never reuse the ambiguous name `random_selected_class`.

### `util/checkpoint_orgslot.py`

Implement:

```text
load_original_owt_initialization
load_orgslot_base_checkpoint
append_and_initialize_new_slots
visible_slot_transfer
hash_frozen_parameters
compare_parameter_hashes
```

All load functions must print and save:

- loaded keys;
- missing keys;
- unexpected keys;
- shape mismatches;
- slots loaded;
- slots intentionally skipped.

Do not use unexplained `strict=False`.

### `engine_pretrain_orgslot.py`

Support explicit stages:

```text
base
incremental_min
incremental_suppress
incremental_formal
offline
sequential_ft
head_only
```

Log each loss component separately. The displayed total must be the exact loss
used for backpropagation.

### `main_pretrain_orgslot.py`

Add configuration for:

```text
stage
slot names and raw IDs
visible old slots
new slots
initialization source
base checkpoint
slot_tg_depth
lambda values
background policy
old confidence threshold
evaluation interval
seed
```

Restore deterministic seed setup for Python, NumPy, PyTorch, distributed
samplers, and DataLoader workers. Save the fully resolved configuration to the
run directory.

### `eval_orgslot.py`

This is the main segmentation evaluator. Do not use reconstruction threshold
as the primary result.

It must support:

- binary per-slot probabilities;
- calibrated multiclass fusion;
- slice-to-case aggregation;
- 2D and 3D input paths;
- per-class and aggregate metrics;
- raw old-path and final fused outputs;
- visual overlays and slot attention diagnostics.

## 6. Label Visibility Implementation

### 6.1 Base stage

For visible old organ masks `M_1...M_n`:

```text
M_bg = NOT(M_1 OR ... OR M_n)
```

Future organs are not removed from `M_bg`, because doing that would use future
labels and invalidate the protocol.

Example OWT legacy debug:

```text
visible old:
  Kidney-combined
  Spleen
  Pancreas

hidden future:
  Liver

base background:
  all pixels outside the three visible old masks, including Liver
```

### 6.2 Incremental stage

The training loader exposes only:

```text
M_new
```

It must not expose:

```text
M_old
full multiclass label map
future masks
```

The evaluator may load complete GT separately after training.

### 6.3 Dataset split rules

- Split at patient/case level before expanding into 2D slices or 3D windows.
- Never let slices/windows from one case appear in more than one split.
- Keep the final test split untouched.
- Use validation for hyperparameters and checkpoint choice.
- Save case lists, CSV checksums, preprocessing version, and class map.
- For OWT debug, preserve the existing official train/test division where
  available and create validation from training cases only.
- For BTCV/WORD, recover and document the PCDD split where feasible. If it
  cannot be recovered, use a fixed case-level split and report that the result
  is protocol-inspired rather than split-identical.

## 7. Slot-Aware TGR

### 7.1 Base sampler

For `S` base slots including provisional background, produce:

```text
keep [B,S] boolean
```

Recommended random v0:

1. For each sample independently, sample a retained count `k` uniformly from
   `1...S`.
2. Generate a deterministic sample/epoch permutation of slots.
3. Keep the first `k`.
4. Ensure at least one slot is retained.

This gives different samples different retained combinations without variable
token lengths, because every slot always has exactly `K` internal tokens.

For each slot:

```text
compute canvas_s [B,N,D]
multiply by keep[:,s,None,None]
```

No token padding is required across samples.

Base target:

```text
target = input.clone()
for every dropped visible old slot:
    zero its GT region in target
if provisional background is dropped:
    zero provisional-background region in target
```

The zeroing is per sample.

Segmentation loss:

- retained slot: supervise its binary mask;
- dropped slot: ignore it;
- do not relabel a dropped organ as background.

### 7.2 Incremental minimal run

For the first mechanism test:

- new slot is always retained;
- keep all frozen old/background slots;
- compute reconstruction loss only in the new-organ GT region;
- background remains fully frozen;
- no suppress loss.

This isolates whether the appended slot can learn.

### 7.3 Retain-New formal sampler

For each sample:

```text
existing = [background, old_1, ..., old_n]
shuffle(existing)
k ~ Uniform(1, len(existing)+1)
retained = [new] + existing[:k-1]
```

New is always retained.

When old/background slots are dropped, reconstruction target masks must come
from detached frozen predictions:

```text
M_old_high = sigmoid(old_logit_frozen) > tau_old
```

Default:

```text
tau_old = 0.7
```

Always remove the current new GT from old/background pseudo masks before using
them:

```text
M_old_high = M_old_high AND NOT(M_new)
```

Old GT must never be used for incremental target construction.

## 8. Losses

### 8.1 Base loss

```text
L_base = L_rec_base + lambda_seg * L_seg_base
```

Reconstruction:

```text
L_rec_base =
    mean((reconstruction - TGR_target)^2)
    + lambda_lpips * LPIPS(reconstruction, TGR_target)
```

For 3D, preserve the existing slice-wise LPIPS behavior unless a validated 3D
perceptual loss is introduced.

Segmentation:

```text
L_seg_base =
    mean retained old-slot DiceBCE
    + lambda_bg_seg * retained background DiceBCE
```

Binary loss:

```text
DiceBCE(logit, mask) =
    soft_dice_loss(sigmoid(logit), mask)
    + BCEWithLogits(logit, mask)
```

Use `lambda_bg_seg < 1` because provisional background contains future organs.

### 8.2 Incremental minimal loss

```text
L_inc_min =
    L_seg_new
    + lambda_rec * L_rec_new_region
```

```text
L_seg_new = DiceBCE(logit_new, M_new)
```

```text
L_rec_new_region =
    sum((reconstruction - input)^2 * M_new)
    / (sum(M_new) * channels + eps)
```

The new-region loss checks that the new slot produces a canvas compatible with
the frozen decoder. It must not become a full-image new-organ decoder.

### 8.3 Suppress ablation

```text
L_inc_sup =
    L_inc_min
    + lambda_sup * L_sup_old_conf
```

Build:

```text
M_old_union = union of frozen old high-confidence masks
M_sup = M_old_union AND NOT(M_new)
```

Then:

```text
L_sup_old_conf =
    masked BCEWithLogits(logit_new, target=0, mask=M_sup)
```

This reduces new-on-old false positives without using old GT.

### 8.4 Background release

Later formal run:

```text
L_bg_new =
    BCEWithLogits(logit_bg[M_new], 0)
```

Only local complementarity on the new GT region is used in P0. Do not add
full-image background/new complementarity.

### 8.5 Formal P0 loss

```text
L_inc =
    L_rec_inc
    + lambda_seg * L_seg_new
    + lambda_bg_new * L_bg_new
    + lambda_sup * L_sup_old_conf
```

P1/P2 losses such as old fused KL, background teacher stability, token
orthogonality, contrastive purity, or register regularization must not be added
before P0 is validated.

### 8.6 Provisional starting values

The plan does not lock most numerical weights. Use the following only as
debugging defaults, then tune on validation:

```text
lambda_seg = 1.0
lambda_lpips = 1.0, matching the current OWT addition
lambda_bg_seg = 0.25
lambda_rec = 0.1
lambda_sup = 0.0 for minimal run, then 1.0 for suppress ablation
lambda_bg_new = 1.0
tau_old = 0.7
```

Required small validation grids:

```text
lambda_bg_seg: 0.1, 0.25, 0.5
lambda_rec: 0.01, 0.1, 1.0
lambda_sup: 0.1, 0.5, 1.0
tau_old: 0.6, 0.7, 0.8
```

Do not tune on the test set.

## 9. Freeze And Optimizer Policies

### 9.1 Base stage

Train:

- shared encoder;
- visible old slots;
- provisional background slot;
- shared decoder;
- segmentation heads;
- identity calibration parameters if included.

If initialized from an original OWT checkpoint, the code-aware plan suggests:

```text
inherited OWT path: lr_scale = 0.1
new slot heads/modules: lr_scale = 1.0
```

The current scheduler supports per-group `lr_scale`.

### 9.2 Incremental minimal Ours

Freeze:

- shared encoder;
- all old collectors;
- all old TGEnc modules;
- all old AHER modules;
- all old heads;
- full background path;
- shared decoder.

Train:

- new collector;
- new TGEnc;
- new AHER;
- new binary head.

### 9.3 Incremental formal background policy

Low-LR trainable:

- background collector;
- background AHER;
- background head.

Frozen:

- background TGEnc.

Recommended LR scales:

```text
new slot = 1.0
background plastic modules = 0.1
calibration = 0.1 or 1.0, but report it
```

Fallback only:

- open background TGEnc at low LR;
- tiny decoder adapter;
- low-LR shared decoder tuning.

### 9.4 Freeze verification

Before and after incremental training:

1. Save SHA256 hashes of every frozen parameter tensor.
2. Assert frozen parameters have `requires_grad=False`.
3. After backward, assert frozen gradients are `None`.
4. After optimizer step, compare hashes.
5. Run a fixed probe batch through raw old paths before and after training and
   assert logits/canvases are numerically unchanged in `eval()` mode.

Do not rely only on optimizer parameter-group printouts.

## 10. New Slot Initialization

Default:

```text
new.collector <- deepcopy(background.collector)
new.tg_encoder <- deepcopy(background.tg_encoder)
new.aher <- deepcopy(background.aher)
new.head <- deepcopy(background.head)
```

Reset/keep head bias consistently and record the choice.

Required ablation:

```text
random new slot initialization
```

The rationale is that future organs were included in provisional background
during base training, so new-slot learning is a background release/refinement
process.

## 11. Original OWT Checkpoint Migration

An original joint OWT checkpoint is an initialization source, not the final
old-knowledge checkpoint. After migration, a new OrganSlotBank base stage must
still be trained under protocol-visible labels.

Recommended mapping:

### Shared encoder

Copy compatible:

```text
patch_embed
cls_token
positional embeddings
blocks1
encoder normalization
```

### Shared decoder

Copy compatible:

```text
decoder positional embeddings
decoder_blocks
decoder_norm
decoder_pred
```

### Joint OrganCollector to slots

The current joint collector emits `S*K` channels.

For slot index `s`:

```text
slot.conv1 <- joint.conv1
slot.conv3 <- joint.conv3
slot.conv2.weight <- joint.conv2.weight[s*K:(s+1)*K]
```

This preserves the class-token output-channel grouping at initialization.

### Shared TGEnc to organ-wise TGEnc

Copy the selected first `slot_tg_depth` blocks from original `blocks2` into
each visible slot and background slot.

If using adapters instead, document the exact mapping and initialization.

### Shared AHER to per-slot AHER

The current AHER linears are token-shared, so copy the complete compatible
AHER state into every visible slot/background AHER.

### Segmentation heads

Initialize new unless loading a previous OrganSlotBank checkpoint.

### Migration tests

- Verify collector output slices match the corresponding joint collector
  outputs before TGEnc.
- Verify all loaded tensor shapes.
- Save a migration report JSON.
- Never silently load future slot weights into a strict Base4/Base7 path.

## 12. Calibration And Inference

Per-slot calibration:

```text
calibrated_logit_s = a_s * logit_s + b_s
```

Initialize:

```text
a_s = 1
b_s = 0
```

Final prediction:

```text
stack [background, old organs, new organs]
argmax over calibrated logits
```

Important unresolved detail:

The plan says old/new/background calibration parameters may be trained during
incremental learning, but old calibration has no valid supervised gradient if
only new GT is used. Therefore:

- minimal debug: keep all calibration identity/frozen;
- first calibrated run: train new/background calibration with new GT;
- train old calibration only if a pseudo-label consistency objective using
  frozen old predictions is explicitly implemented;
- never tune old calibration with old GT during incremental training.

Report:

```text
raw old binary-head DSC
final calibrated fused DSC
```

Raw old path proves representation/output invariance. Final fused DSC measures
the usable multiclass result.

## 13. Validation And Metrics

### 13.1 Primary segmentation metrics

Report:

- per-organ DSC;
- `Old DSC` mean;
- `New DSC` mean;
- `All DSC` mean;
- old DSC before incremental;
- old DSC after incremental;
- forgetting;
- new-on-old false positives.

Forgetting:

```text
F = Old_DSC_before - Old_DSC_after
```

Also report per-old-organ forgetting.

New-on-old FP:

```text
sum(pred_new AND GT_old_union) / (sum(GT_old_union) + eps)
```

Old GT is allowed here because this is evaluation, not training.

### 13.2 OWT diagnostics

Report:

- new-region reconstruction MSE/PSNR;
- full reconstruction MSE/LPIPS where relevant;
- collector attention maps;
- SDTG/token maps;
- AHER maps;
- prediction overlays;
- added parameters per organ;
- peak GPU memory;
- inference time.

Reconstruction-threshold direct/indirect Dice remains auxiliary and cannot
replace head-based segmentation results.

### 13.3 2D case-level evaluation

1. Predict every slice.
2. Recover `case_id` and numeric `slice_index`.
3. Sort slices by numeric index.
4. Stack predictions into a case volume.
5. Compute case-level 3D DSC in the declared preprocessed space.
6. If preprocessing metadata supports inverse mapping, also report
   original-volume-space DSC.

Never call cropped/resized-space Dice original-volume Dice.

For empty GT classes:

- record whether GT and prediction are empty;
- do not silently assign Dice 1 and average it with present-organ cases;
- report presence-conditioned DSC and the chosen empty-case policy.

### 13.4 Checkpoint selection

Base:

- select by validation old-organ mean DSC;
- use reconstruction as a secondary diagnostic.

Incremental:

- select by visible new-organ validation DSC or visible validation loss;
- do not select by old test DSC;
- old validation GT may be reported for analysis only if the protocol allows
  it, but must not drive optimization or checkpoint selection.

Test is run once for the selected checkpoint.

## 14. Required Baselines

### Initial mechanism-debug table

1. Ours minimal.
2. Sequential full fine-tuning.
3. Head-only adapter.

Use the same:

- base checkpoint;
- data split;
- preprocessing;
- incremental labels;
- new-organ loss;
- iteration budget;
- head family where applicable.

### Ours

OrganSlotBank, append new slot, freeze old path and decoder.

### Sequential FT

Start from the same OrganSlotBank base checkpoint, append the same new slot,
then train the full model with new-organ-only labels:

- encoder;
- old slots;
- background;
- decoder;
- all heads.

This is the naive forgetting lower bound.

### Head-only adapter

Recommended precise definition:

- copy/fix the background-derived new slot path;
- freeze its collector, TGEnc, and AHER;
- train only the new binary head on that frozen canvas.

This isolates whether a small readout alone explains the result.

### Full controlled table

Add after the initial debug passes:

- Offline upper bound: all selected labels jointly visible.
- OWT-Seg joint baseline: original joint OC/shared TGEnc/shared AHER plus the
  same supervised head family and visibility protocol.
- OrganSlotBank naive/full FT.
- Ours freeze+append.
- Optional dense adapter from frozen encoder features.

Potential naming ambiguity:

`Sequential FT` and `OrganSlotBank naive` can become identical if both mean
full fine-tuning of the same OrganSlotBank checkpoint. Do not report duplicate
runs as separate methods. For the full table, distinguish:

```text
OWT-Seg Joint FT:
  joint original architecture, full fine-tuning

OrganSlotBank FT:
  organ-wise architecture, full fine-tuning

Ours:
  organ-wise architecture, strict freeze+append
```

## 15. PCDD Role And Formal Protocols

PCDD is:

- a benchmark/protocol reference;
- a reported-number reference;
- not the engineering code base;
- not a direct implementation dependency.

Match where feasible:

- class order;
- train/validation/test split;
- label visibility;
- Old/New/All DSC;
- forgetting.

Do not force-match unless necessary:

- optimizer;
- OWT preprocessing;
- loss recipe;
- backbone internals.

PCDD eight-organ order:

```text
1 spleen
2 right kidney
3 left kidney
4 gallbladder
5 esophagus
6 pancreas
7 liver
8 stomach
```

Formal order:

### BTCV 4-1

```text
Base: spleen, right kidney, left kidney, gallbladder
Step 1: esophagus
Step 2: pancreas
Step 3: liver
Step 4: stomach
```

At each incremental step, only the current new label is visible.

### BTCV 7-1

```text
Base: classes 1-7
Increment: stomach
```

### BTCV 4-4

```text
Base: classes 1-4
Increment: classes 5-8
```

Implement 4-4 only after single-new-organ append works. Define whether four
new slots are trained jointly in one incremental stage, matching the protocol.

### WORD

Run after BTCV is credible. Use the same eight-class semantic mapping and
strict visibility rules.

PCDD comparison table must label rows honestly:

```text
PCDD reported result
our reproduction, only if code/split reproduced
E-OWT-Seg under matched protocol
```

Do not call a protocol-level comparison an implementation reproduction.

## 16. Initialization Experiments

For BTCV/WORD Base4/Base7 compare:

1. Random initialization.
2. External MAE encoder initialization.
3. External all-8 OWT visible-slot transfer initialization.

### External MAE

Load:

- encoder by default;
- decoder only if structurally compatible.

Do not load:

- organ slots;
- segmentation heads.

### External all-8 OWT

External manual/pseudo masks are allowed only to produce the external
initialization checkpoint.

For strict Base4/Base7:

- load shared encoder/decoder;
- load only protocol-visible old slots;
- do not load future slots;
- fit Base4/Base7 again on BTCV/WORD with visible labels only.

The external dataset is not part of the strict BTCV/WORD training data.

Use the same fixed base and incremental budgets across initialization routes.
Longer best-route runs must be reported separately.

## 17. Experiment Execution Order

### Stage A: CPU/unit tests

Pass all model, visibility, loss, and checkpoint tests before Slurm.

### Stage B: one-batch forward/backward

Check:

- 2D and 3D shapes;
- finite losses;
- expected trainable gradients;
- no frozen gradients;
- TGR keep masks;
- checkpoint save/load.

### Stage C: tiny overfit

Use 2-8 cases/slices with augmentation disabled.

Success:

- visible segmentation loss drops strongly;
- new-organ Dice approaches overfit behavior;
- reconstruction loss decreases;
- no NaN/Inf.

If tiny data cannot overfit, do not submit full training.

### Stage D: 2D OWT-legacy smoke

Split:

```text
Base old: Kidney-combined, Spleen, Pancreas
New: Liver
```

Run:

1. Base training smoke.
2. Incremental minimal Ours.
3. Evaluation.

Use one seed and a small subset.

### Stage E: early 3D Fixfr4-TS1 smoke

Run immediately after minimal 2D works. Verify that no 2D-only reshape,
upsampling, or metric assumptions remain.

### Stage F: 2D mechanism table

Run:

- Ours minimal;
- Sequential FT;
- Head-only adapter.

Then:

- suppress-loss ablation;
- background-plasticity ablation;
- random versus background new-slot initialization.

Use at least three seeds for reported conclusions.

### Stage G: full controlled OWT table

Add:

- Offline upper bound;
- OWT-Seg joint;
- OrganSlotBank full FT;
- optional dense adapter.

### Stage H: BTCV

Order:

```text
4-1 -> 7-1 -> 4-4
```

### Stage I: full 3D

Run after 2D mechanism and early 3D smoke are stable.

### Stage J: WORD

Run after BTCV gives a credible mechanism result.

## 18. Suggested Run Budgets

The plan locks fairness, not exact epoch counts.

Recommended development budgets:

```text
unit test: seconds
one-batch smoke: 2-5 iterations
tiny overfit: 200-500 optimizer steps
subset smoke: 2 epochs
mechanism pilot: 20-100 epochs depending on convergence
final: chosen from validation convergence, identical across compared methods
```

The original OWT scripts use:

```text
blr = 1e-4
weight_decay = 0.05
warmup = 60
epochs = 1200
token_factor = 20
input = 224
LA = enabled
2D batch per GPU = 96 in the old two-GPU script
3D batch per GPU = 64 in the old two-GPU script
```

Do not immediately spend 1200 epochs on an unverified new pipeline. Determine
the final fixed budget after smoke and convergence pilots. Keep effective
batch size, optimizer, schedule, augmentation, and iteration count matched
within every comparison table.

Final reporting:

- seeds: at least 3;
- mean and standard deviation;
- exact base checkpoint shared across incremental baselines;
- exact run command and resolved config saved.

## 19. Slurm And Output Isolation

Environment:

```text
/gpfs/work/aac/bolinren19/.conda/envs/abdpet
```

Recommended layout:

```text
slurm/orgslot/smoke/
slurm/orgslot/debug/
slurm/orgslot/btcv/
slurm/orgslot/word/
slurm/orgslot/logs/

Results/OrganSlotBank/
  OWTLegacy/
  BTCV/
  WORD/
```

Run ID should include:

```text
dataset
protocol
stage
method
dimension
initialization
seed
config hash
```

Every result directory should contain:

```text
resolved_config.yaml
command.txt
git_commit.txt
git_diff.patch or dirty-status warning
dataset_manifest_checksums.json
class_map.yaml
train.log
metrics.jsonl
best_checkpoint.pth
last_checkpoint.pth
eval/
visualizations/
parameter_report.json
checkpoint_load_report.json
```

Do not write OrganSlotBank results into PSEM, LossBalance, or original OWT
result directories.

## 20. Required Tests

### Model tests

- 2D forward output shapes.
- 3D Fixfr4 output shapes.
- Every slot produces exactly `K` tokens.
- AHER produces exactly `N` canvas positions.
- normalized additive fusion is correct for 1, 2, and S slots.
- per-sample keep masks do not affect other samples.
- all-dropped input is rejected or repaired to one retained slot.
- appending a slot preserves old state keys and tensors.

### Visibility tests

- base loader exposes only old masks and provisional background.
- hidden Liver is included in base provisional background.
- incremental train loader exposes only Liver mask.
- old/full GT is unavailable from the incremental training batch.
- train/val/test case IDs are disjoint.

### Loss tests

- dropped slots contribute zero segmentation loss.
- retained slots receive gradients.
- new-region reconstruction ignores pixels outside new GT.
- empty masks do not produce NaN.
- suppress loss uses detached pseudo masks.
- old GT never enters incremental loss functions.

### Freeze tests

- frozen parameter hashes are unchanged.
- frozen gradients are `None`.
- raw old logits match before/after incremental training.
- new slot and allowed background modules do update.

### Migration tests

- joint collector `conv2` slices map to the correct slots.
- original checkpoint key report is complete.
- visible-slot transfer skips future slots.
- base and incremental checkpoint resume work.

### Metric tests

- synthetic masks have known Dice.
- numeric slice sorting is correct (`_2` before `_10`).
- case aggregation is correct.
- empty-case policy is explicit.
- Old/New/All means and forgetting are correct.

## 21. Decision Gates

### Gate A: wiring

Pass only if:

- 2D smoke runs;
- all losses are finite;
- freeze/train groups are verified;
- base reconstruction and segmentation are nontrivial;
- checkpoint save/load is exact.

### Gate B: mechanism

Pass only if:

- new Liver learns;
- raw old paths do not drift;
- final old DSC drop is lower than Sequential FT/OrganSlotBank FT;
- new-on-old FP is controlled;
- result is better than head-only adapter in a meaningful way.

### Gate C: paper evidence

Pass only if:

- forgetting improves over OWT-Seg joint FT and OrganSlotBank FT;
- new-class DSC remains acceptable;
- added parameters are defensible;
- slot/reconstruction diagnostics support the mechanism;
- results hold across seeds and formal BTCV protocols.

If Gate C is weak, simplify the claim or improve the P0 mechanism. Do not hide
failure by adding many P1/P2 modules at once.

## 22. P0, P1, And P2 Boundary

P0:

- base reconstruction + segmentation;
- organ-wise slots;
- strict freeze+append;
- new-region reconstruction;
- suppress ablation;
- weak background update;
- calibrated argmax;
- Retain-New TGR.

P1:

- incremental LPIPS crop;
- old fused-logit KL;
- background teacher;
- low-LR background TGEnc;
- threshold sensitivity.

P2:

- register tokens;
- learnable canvas fusion;
- CrossSlotMixer;
- token orthogonality/contrastive objectives;
- decoder adapter.

Implement P0 first. Add one component at a time with a controlled ablation.

## 23. Relationship To PSEM And LossBalance

PSEM:

- tests/fixes per-sample class-token masking in original OWT;
- may inform the deterministic slot TGR sampler;
- does not implement expandable organ slots.

LossBalance:

- tests organ-aware reconstruction weighting for small organs;
- may later be combined with OrganSlotBank after the main mechanism works;
- is not the incremental freeze+append method.

Do not start by combining:

```text
PSEM + LossBalance + OrganSlotBank + background plasticity + suppression
```

That would make failure attribution impossible.

## 24. First Implementation Checklist

1. Create a separate worktree/branch from clean `main`.
2. Copy this handoff and record the starting commit.
3. Add class/stage config and strict visibility dataset wrapper.
4. Add `OrganSlot`, `OrganSlotBank`, and minimal binary head.
5. Add the new model with encoder, per-slot path, normalized fusion, decoder.
6. Add model/loss/visibility/migration unit tests.
7. Implement base loss and per-sample Slot-Aware TGR.
8. Implement base training and validation.
9. Implement OrganSlotBank checkpoint save/load and OWT migration report.
10. Train/overfit a tiny OWT-legacy base sample.
11. Implement append/copy-from-background and freeze verification.
12. Implement incremental minimal Liver loss.
13. Implement head-based evaluator and case aggregation.
14. Run 2D minimal smoke.
15. Run early 3D Fixfr4-TS1 smoke.
16. Run Ours, Sequential FT, and Head-only with one shared base checkpoint.
17. Add suppression as a separate ablation.
18. Add weak background plasticity as a separate ablation.
19. Add full controlled baselines.
20. Only then create strict BTCV 4-1/Base4 manifests and run formal tests.

## 25. Completion Definition For The Next Coding Session

The first coding session is complete only when it delivers:

- new files without breaking original OWT/PSEM paths;
- passing CPU unit tests;
- passing 2D forward/backward;
- passing 3D Fixfr4 forward/backward;
- a tiny-batch overfit result;
- a saved/loaded base checkpoint;
- an appended Liver slot initialized from background;
- verified frozen old parameters;
- one incremental forward/backward using only Liver GT;
- head-based validation metrics;
- a Slurm smoke job that exits successfully;
- an experiment log recording commands, configs, and observed results.

It is not complete merely because the model imports or a long training job has
been submitted.
````

<a id="source-9"></a>

## docs/ORGSLOT_112_NATIVE_CROP448_EXPERIMENT_PLAN.md

SHA256: `debad717f08cfb1e7a300a0abf4deadaeda2773ec3e994eb5785a4e02a4db081`

````text
# OrganSlotBank WORD 1x1x2: 448 vs 448 + 20% Small-Organ Crop

## Locked experiment

Offline preprocessing performs only RAS reorientation, joint image/label
resampling to 1x1x2 mm, HU clipping to [-175,250], normalization, and native
matrix slice export. There is no intermediate 448 canvas and no offline
resize. All paths are rooted at `/mnt/DATA-4/anteng`.

WORD is used first. Its 120 cases have resampled XY sizes from 490x490 through
501x501, so every case supports the declared physical crop windows without
padding. BTCV is excluded from this first experiment because 23/30 cases are
smaller than 448 after resampling and require a separate protocol.

### Arm A: 448 control

```text
1x1x2 native matrix -> deterministic center crop 448x448 -> OrganSlot img_size=448
```

The output represents exactly 1 mm per pixel in XY.

### Arm B: 448 + 20% small-organ ROI

Each requested sample is selected as follows:

- 80%: identical global center crop 448x448;
- 20%: choose gallbladder (ID4), esophagus (ID5), or pancreas (ID6)
  uniformly; draw a positive slice from that class pool; crop a 384x384
  physical window around its bounding-box center with up to 10% bounded
  center jitter; resize image/label to 448x448.

The ROI branch enlarges linear size by 448/384=1.167 and area by about 1.36,
not the stronger 1.96 area factor of a 320 crop. The crop is guaranteed to
retain the entire focus-organ bounding box. It keeps all image context and all
other labels within the selected window; it never masks everything except the
focus organ.

The focus OrganSlot is forced to remain in the TGR keep mask for ROI samples.
Other slots follow the selected ordinary mask schedule. TensorBoard records
the realized ROI fraction, per-focus-class fraction, and retained-slot count.

Validation and test always use the deterministic center 448 crop. Labels are
never used to choose evaluation crops.

## Fairness invariants

Both arms share the exact case split, native 1x1x2 files, model variant,
initialization, loss weights, ordinary mask schedule, seed, per-GPU
micro-batch, accumulation, effective batch, learning rate, optimizer updates,
warmup updates, and evaluation. The intentional Arm-B changes are the 20%
positive-slice mixture, 384 ROI crop, and focus-slot retention.

At patch size16, 448 produces 28x28=784 spatial patch tokens versus196 at
224. Before formal runs, sweep per-GPU micro-batch 2,4,8. Select the largest
safe value with memory headroom and use accumulation to target effective
two-GPU batch192 (micro-batch8 implies accumulation12).

## Required gates

1. Current AutoPET jobs and queued inference finish; one architecture is
   selected and frozen for both arms.
2. Full WORD geometry/label validation and immutable manifests complete.
3. ROI index checksum matches the training manifest and contains nonempty
   pools for IDs4,5,6.
4. Image/label overlays are inspected before/after resampling and after both
   crop modes.
5. Unit tests prove paired 2D/Fixfr4 crop, nearest label resize, strict label
   isolation, positive focus selection, and focus-slot retention.
6. Audit per-class realized sampling, crop clipping, and label IDs.
7. Deterministic 448 forward/backward passes.
8. Four-to-eight real samples overfit with augmentation disabled.
9. Micro-batch memory sweep and short two-GPU Arm-B smoke pass without
   NaN/OOM and produce checkpoint, provenance, logs, and TensorBoard events.
10. Launch Arm A on GPUs0,1 and Arm B on GPUs2,3 only after all gates pass.
````

<a id="source-10"></a>

## docs/ORGSLOT_ARMB_3D_WORD070_ROI20_CN.md

SHA256: `b1e34db7905c60f5af14b1a554407e971029dfbe7ac006373090429923575787`

````text
# OrganSlot Arm B 3D：WORD 0.7 spacing + ROI20

## 目的

把已完成的 2D Arm B 改成短体块 3D 训练，验证相邻轴位层面的上下文是否能改善
小器官连续性和边界，同时保持数据、slot 数、重建目标、分割损失和 ROI20 策略不变。

## 与 2D Arm B 的关系

| 项目 | 2D Arm B | 3D Arm B |
|---|---:|---:|
| spacing | 0.7 x 0.7 x 2.0 mm | 相同 |
| xy 输入 | 448 x 448 | 相同 |
| ROI | 20% 概率，384 裁剪后缩放到 448 | 相同 xy 策略 |
| z 输入 | 单层 | 连续 4 层（约 8 mm） |
| patch embedding | Conv2d，16 x 16 | Conv3d，1 x 16 x 16 |
| head | multiscale Conv2d | multiscale Conv3d |
| loss | small-organ，lambda_seg=0.01 | 相同，按 3D slab 计算 |
| batch | 192 slices/update | 48 slabs/update，约 192 slices/update |
| actual LR | 7.5e-5 | 7.5e-5（显式固定） |
| optimizer updates | 118800 | 118800 |

## 3D ROI20

ROI 索引仍来自阳性 2D anchor 切片。抽中 ROI 样本后，以 anchor 为中心读取连续
4 层；在这 4 层的目标器官 mask 上取 z 向并集，再确定共同的 384 x 384 xy
裁剪框。相同裁剪作用于所有层，最后统一缩放到 448 x 448。病例边界处平移窗口，
不做跨病例填充，也不越界。

这不是完整病例 3D crop，而是可由 A800 训练、并尽量接近原 Arm B 协议的
short-slab volumetric 3D 实验。

## 关键配置

- `dimension=3D`
- `fix_frame=4`
- `temp_stride=1`
- `slot_head_type=multiscale_conv`
- `slot_head_channels=128`
- `seg_loss_type=small_organ`
- Tversky FP/FN = 0.3/0.7
- Balanced Focal weight = 0.5
- hard negative = top 2%
- negative slab weight = 0.1
- `lambda_seg=0.01`
- 4 x A800，micro batch 2/GPU，accumulation 6
- actual LR = 7.5e-5，与 2D Arm B 一致

## 正式评估

训练完成后必须按病例重组完整 3D 预测，报告八类 case-level 3D Dice、IoU、
precision、recall、NSD 和 HD95（mm）。2D slice 指标只作为错误分析，不作为主结果。
````

<a id="source-11"></a>

## docs/ORGSLOT_ARM_E_CROSS_ATTENTION_CN.md

SHA256: `da21d51ee8b6a30f408de0b7169a07de00ad9a6ef916bf0282c3fbdc509e6a39`

````text
# Arm E + Cross-Attention（2026-09-08）

## 实验目的与范围
在Arm E上增加一个共享的query-to-image cross-attention block，测试query在
最终mask匹配前再次读取空间特征是否有帮助。不是替换AHER、不是新增P2、
不是SAM two-way decoder；不增加query数量，不新增绝对位置PE。

统一开关：`--slot_head_type arm_e_multiscale_query --query_refinement cross_attn`。
环境变量为QUERY_REFINEMENT。默认none完全保留旧E路径与state dict；
仅2D Arm E支持本开关。checkpoint args用于统一head/reconstruction evaluator
恢复结构并严格加载，不允许把原E checkpoint伪装成已训练attention checkpoint。

## 实际结构
CT的P4/P8与ViT P16融合 → P4 pixel feature（448输入时112×112）。
原20个TGEnc tokens经LN、均值、投影和器官身份向量产生一个query。
P4平均池化2倍 → 56×56 memory（注意：不是直接使用P8 lateral feature）。
4-head cross-attention，通道128、无dropout：
q + Attention(LN(q), LN(memory)) → 残差FFN（128→256→128）→ refined q。
refined q与原P4 feature做归一化点积 → 112 logits → 插值回448。
attention计算局部FP32，整体训练BF16。attention模块初始化使用fork_rng，
避免改变后续原有organ slot参数初始化；新模块本身不是零初始化。

长宽112/448=1/4，面积1/16；P4每位置仍有128通道，不能等同“信息只剩1/16”。
边界可能受P4上生成logit再插值的限制。真实P2 skip（224）应作为独立实验，
不能与本次attention一起加。224同通道特征元素数是112的4倍。

## 锁定配置及可比性
WORD spacing0.7/0.7/2，448/384，ROI0.2，small-organ loss，
retained，lambda_seg0.01，lambda_bg_seg0，原Tversky/Focal/top2%/negative0.1，
batch192，seed0，118800更新、warmup5940、blr1e-4、AdamW weight decay0.05。
本组BF16、clip1、finite每次更新、每25epoch保存。先同分配2次真实DDP更新验证，
再从seed0正式训练，不从预检查checkpoint续跑。

重要：历史E任务2892114使用旧FP16/scaler流程，因此与本组并非严格单变量比较。
严格attention结论需同一代码、BF16/clip等设置的query_refinement=none对照；
目前只新增用户要求的一组，不擅自多提交一组baseline。

## 热图与交互检查
工具：`python -m tools.export_query_cross_attention`。
严格加载训练后checkpoint，默认固定选择测试CSV中每个目标器官最先出现的
两张阳性切片（4胆囊、5食管、6胰腺），不按预测质量挑选。
每例导出PNG/PDF、NPZ和manifest：
- CT、GT、head均值attention / uniform（1代表均匀水平）；
- normal / uniform-attention / bypass整个refinement的概率图；
- NPZ保存4个head的原始56×56归一化权重，供逐head审查；
- GT attention mass、GT面积占比、二者之比；
- 三种预测的固定0.5 slice Dice和相对normal的平均概率变化；
- checkpoint及CSV哈希、epoch、dataset index与严格加载报告。

uniform保留value、输出投影、FFN，只取消位置选择；bypass连同FFN一起跳过，
两者含义不同。所有干预保持pixel features和原query不变。
注意力图不是分割概率，也不是因果证明；关注器官外上下文不必然错误。
本模块没有新增绝对坐标PE。同步重排keys/values可以不改变聚合结果，因此
空间热图不能证明“理解了解剖绝对坐标”。干预结果也可能是分布外响应，
要和完整病例级指标一起判断，不能靠几个样例宣称提升。

## 运行与评估
训练脚本：slurm/orgslot/train/orgslot_word07072_arm_e_crossattn_bolin_sip.sbatch。
评估脚本：slurm/orgslot/eval/orgslot_word07072_arm_e_crossattn_bolin_sip.sbatch。
运行使用独立detached快照，通过EXPERIMENT_WORKDIR/EVAL_WORKDIR传入，
固定代码后不再原地改动。评估以afterok跟随训练，先导出固定样例热图，
再运行原24病例统一评估。最终无真实训练checkpoint前，不提供“已证明有效”图。

## 验证
已提交：bolinren19/SIP，account angelosstefanidis，QoS 8a800。
训练2920724（4 A800、24 CPU、192 GB）；热图与统一评估2920726
（1 A800、10 CPU、128 GB），afterok:2920724。
运行快照：`.worktrees/orgslot_arm_e_crossattn`，detached commit `4cbd096`。
此次未推送GitHub。5项cross-attention测试、26项整模型回归、2项Arm E回归通过。
PNG/PDF合成渲染测试通过，但图片查看受沙箱限制，未完成视觉排版验收。
真实GPU预检查和训练后热图仍等待调度，不能记为已通过/已产生。

已增加：共享参数初始化一致性、严格state dict roundtrip、
attention逐head空间归一化、normal/uniform/bypass干预差异、
query对memory依赖、部分器官head参与的整模型finite backward。
渲染测试仅用合成未训练输入，并明确标注非实验证据。
GPU预检查通过与否必须读取任务日志，CPU测试不能代替长程稳定性审核。
````

<a id="source-12"></a>

## docs/ORGSLOT_AUTOPET_MAE_TRANSFER_CN.md

SHA256: `355667781dc9ad0e14a2b81b2c7f0917a5028885261eda18484cb8cb6bad1b14`

````text
# AutoPET MAE → WORD OrganSlot Arm B 迁移实验

## 研究问题

在当前最优的 WORD 2D Arm B 训练协议不变时，使用 AutoPET CT 进行纯图像 MAE 预训练，是否能通过共享 image encoder 和共享 reconstruction decoder 改善 WORD Common8，特别是胆囊、食管和胰腺？

## 固定的 WORD 配置

- spacing：0.7×0.7×2.0 mm；
- input/global crop：448；ROI crop：384；ROI probability：0.2；
- head：`multiscale_conv`，128 channels；
- retained-slot supervision；
- small-organ loss：Tversky FP/FN=0.3/0.7，Balanced Focal权重0.5，hard negatives top 2%，negative slice weight 0.1；
- `lambda_seg=0.01`，`lambda_bg_seg=0`；
- effective batch 192，118800 optimizer updates，seed 0。

现有 scratch Arm B 是直接对照。完整对比固定为三臂，新增实验只改变初始化范围：

| Arm | 初始化 | 状态/用途 |
|---|---|---|
| B0 scratch | 全部随机初始化 | 已完成的 Arm B checkpoint-802，作为零预训练基线 |
| B1 encoder-only | AutoPET MAE `patch_embed + cls + 6层LA encoder + encoder_norm` | 隔离 encoder 预训练收益 |
| B2 encoder+decoder | B1全部内容 + 共享 `decoder_blocks/norm/pred` | 判断共享重建 decoder 是否提供额外收益 |

三臂中的 Collector、TGEnc、AHER 和 `multiscale_conv` segmentation head 均从相同随机种子初始化。B0不重复训练，但必须和B1/B2一起使用同一套3D病例级 evaluator 重评。

B0来自提交`b748e59`，本分支起点为`5226496`。两提交间针对该训练主线的差异仅为可选Loss-v3、resume和附加日志；模型、数据与既有small-organ loss代码未改。三臂固定`positive_roi_loss_weight=0`，该可选项不进入总损失或梯度，因此可以复用B0。

跨账号数据一致性由`tools/validate_word070_dataset_identity.py`在启动前检查：train 28586张、test 6990张；规范化清单SHA256分别为`ffbe5e...1fc8c`和`871a31...a9f1`，ROI核心SHA256为`3a2596...fb8b`。规范化会移除`/WORD/`之前的账号根路径。

## AutoPET MAE

AutoPET XEC 训练清单包含78400张224×224 RGB JPEG，三个通道是相同灰度复制。第一版保持原224图像，不放大到448；WORD位置编码按448重新生成，因此不迁移位置编码。

MAE配置：patch16、LA encoder 6层/768维、decoder 8层/768维、mask ratio 0.75、masked-patch pixel MSE、BF16、clip-grad 1.0、effective batch 192、100000 updates、5000 warmup updates。按患者从AutoPET training划分90%/10% train/validation，不使用AutoPET test和器官标签。

## 权重映射

加载：`patch_embed`、`cls_token`、6个encoder blocks、`encoder_norm`、共享`decoder_blocks`、`decoder_norm`和`decoder_pred`。

不加载：MAE位置编码、`mask_token`、`decoder_embed`、optimizer；WORD所有slot的Collector、TGEnc、AHER、segmentation head和校准参数保持随机初始化。加载器对每个映射张量做名称和shape验证，并保存`mae_initial_checkpoint_load.json`。

## 结果判定

主要比较B0/B1/B2的同协议3D病例级结果：八类平均Dice、小器官平均Dice、逐器官Dice、precision/recall、NSD、HD95、预测/GT体积比以及固定/验证集校准阈值。`B1-B0`量化encoder预训练收益，`B2-B1`量化共享decoder的额外贡献；不能只汇报B2相对B0。

单seed只能作为探索性证据。若B1或B2平均Dice提高至少0.5点且小器官平均提高至少1点、同时大器官下降不超过0.5点，再补两个seed。

## 执行链

1. `autopet_mae_transfer_smoke_sifan_xec.sbatch`：真实数据、完整模型做2次更新并验证checkpoint；
2. `autopet_mae_transfer_pretrain_bolin_sip.sbatch`：正式MAE当前选择bolinren19/SIP的`sifansong/4a800`，其调度预估早于XEC；`autopet_mae_transfer_pretrain_sifan_xec.sbatch`保留为备用；
3. `orgslot_word07072_armb_autopet_mae_encoder_sifan_xec.sbatch`：B1 encoder-only；
4. `orgslot_word07072_armb_autopet_mae_sifan_xec.sbatch`：B2 encoder+decoder 的 sifansong/XEC 版本；
5. `orgslot_word07072_armb_autopet_mae_encoder_decoder_anteng_xec.sbatch`：B2 的 antengcai23/XEC 并行版本，必须先复制并校验同一 MAE checkpoint；
6. 两个新 checkpoint-802 完成后，将B0/B1/B2使用相同3D病例级重建/head评估脚本重评。

MAE预训练当前运行在`bolinren19/SIP`，因为该账号已有完整AutoPET，且4卡调度预估比XEC早；它不承担后续WORD训练。MAE完成后仅传输模型checkpoint并记录SHA256，再由`sifansong/XEC`运行B1、`antengcai23/XEC`运行B2。不得把三个账号或SIP/XEC的绝对路径互换。
````

<a id="source-13"></a>

## docs/ORGSLOT_CONFIGURATION_CONTRACT_CN.md

SHA256: `9ea64e2bfa65dc21b7b766f01655280140be460f79acb3ac66512362818affa5`

````text
# OrganSlot 架构、数值精度与提交配置统一规范

更新时间：2026-09-08。适用分支：`feature/orgslot`。
本文件是新实验的配置与审核规范，不表示所有历史脚本已经自动整改。

## 1. 唯一开发入口与运行快照

- 统一开发目录：`OD_OWT/.worktrees/orgslot_integrated`。
- Arm A/B/C/D/E、2D/3D、PE、MAE迁移通过参数选择，不再为每种模型复制实现。
- 历史目录可能处于 detached HEAD；归档tag用于追溯，不能视作实时同步副本。
- 新任务应使用经过核验的独立代码快照，并记录绝对路径、commit与未提交diff。
- 若修复必须应用到旧实验目录，必须分别核验统一入口和实际运行目录；只合并Git分支不算部署完成。
- 正在运行的任务不要原地替换代码。需要变更时建立新快照、新输出目录及新任务。

当前例外：重提的2D PE任务 `2919868` 仍明确使用
`.worktrees/orgslot_pixel_pe`，相关修复也已同步到统一开发目录。
这不意味着其他旧目录已经同步。未提交diff也属于实际实验版本，不能只记录commit。

## 2. 四层配置必须同时核验

| 层级 | 负责内容 | 实际证据 |
|---|---|---|
| 模型 | head、query、PE、维度、token数量 | 模型构建参数、checkpoint args及state dict |
| 优化与数值 | loss、AMP、裁剪、LR、batch、更新预算、初始化/续训 | resolved_config.json、训练日志 |
| 启动与部署 | worktree、环境、数据路径、脚本环境变量、最终CLI | Slurm脚本、command.txt、启动日志 |
| 调度与评估 | 登录用户、Slurm account、QoS、资源、依赖、评估协议 | scontrol、评估结果与完整性标记 |

**模型支持BF16 ≠ 启动任务选择了BF16；分支包含修复 ≠ 运行目录包含修复。**

参数不是简单的“命令行永远覆盖一切”，实际顺序为：

`sbatch导出环境 → 作业脚本export → shell启动器构造CLI → Python解析 → 模型/优化器`

- 作业脚本 `export AMP_DTYPE=bf16` 会覆盖继承环境中的同名变量。
- 启动器 `AMP_DTYPE=${AMP_DTYPE:-fp16}` 在变量未设置或为空时使用FP16。
- 启动器显式传入 `--amp_dtype` 后，Python默认值不再决定该次运行精度。
- 当前通用启动器和Python入口的AMP默认值**仍是FP16**，没有全局改成BF16。
- 当前 `PIXEL_PE` 默认none、`SEGMENTATION_UNIT` 默认slab，是历史兼容行为，不代表新实验推荐值。
- Shell变量只有被启动器转换成CLI参数才会生效；不能凭变量名称推断生效。

审查入口：
[通用启动器](../scripts/orgslot/run_word_112_448_a100.sh)、
[Python入口](../main_pretrain_orgslot_common8_a100.py)、
[训练循环](../engine_pretrain_orgslot_common8_a100.py)。

## 3. 架构参数统一表

| 控制项 | 环境变量 / CLI | 含义与边界 |
|---|---|---|
| 维度 | DIMENSION / --dimension | 2D或3D；3D还需核验FIX_FRAME、TEMP_STRIDE |
| Arm A | SLOT_HEAD_TYPE=linear | AHER canvas后线性分割头 |
| Arm B | SLOT_HEAD_TYPE=multiscale_conv | AHER canvas后多尺度卷积分割头 |
| Arm C | SLOT_HEAD_TYPE=query_dot | 共享pixel decoder；organ tokens汇总为一个query |
| Arm D | SLOT_HEAD_TYPE=multi_query_dot | 共享pixel decoder；保留多个query，log-mean-exp汇总 |
| Arm E | SLOT_HEAD_TYPE=arm_e_multiscale_query | 输入图像P4/P8与ViT P16融合；单query；目前仅2D |
| Pixel PE | PIXEL_PE / --pixel_pe | none、spatial、temporal、spatiotemporal；当前仅C/D |
| 监督单位 | SEGMENTATION_UNIT / --segmentation_unit | slab保留历史3D语义；slice按切片判断阳性/空切片；2D不受此开关影响 |
| MAE初始化 | MAE_INIT_CHECKPOINT、MAE_INIT_SCOPE | encoder或encoder_decoder；必须记录来源checkpoint |
| 训练续跑 | RESUME_CHECKPOINT / --resume | 与MAE初始化互斥；核验epoch、optimizer和更新计数连续性 |

2D只允许none/spatial，不存在独立的切片时间轴。3D可以分别比较四种PE。
PE添加在pixel projection之后、pixel decoder之前；不改OrganCollector的时间压缩，
也不等同于给direct head接入完整OWT reconstruction Transformer decoder。

`token_factor=20` 与 `ROI probability=0.2` 是两个不同参数。
“ROI20”指20%的ROI采样概率，不表示20个token。

## 4. 可比实验必须锁定的项目

2D WORD head对照使用：spacing 0.7×0.7×2.0、输入448、ROI384、
ROI probability 0.2、token factor20、seed0、有效batch192、
118800次optimizer更新。3D不能照搬2D有效batch，应另行记录slab数量、
每slab切片数及更新预算。

分割损失：`small_organ`；retained监督；lambda_seg=0.01；
lambda_bg_seg=0；Tversky FP/FN=0.3/0.7、eps=1e-6；
Balanced Focal权重0.5、alpha=0.75、gamma=2；
hard negatives=top2%、negative slice weight=0.1。
不能把“背景像素负样本”误认为独立background slot监督，两者不同。

总训练目标也不是只有Dice+Focal：当前路线仍有L2 reconstruction及LPIPS，
并加权叠加segmentation loss。核验 `loss_version=L2-LPIPS`、
`lambda_lpips=1`、`fusion_mode=linear_sqrt`、reference count=9，
以及旧Loss3相关权重是否为0。

比较表中必须增加以下列：
**AMP dtype、clip_grad、LR、有效batch、更新次数、监督单位、初始化来源**。
只保持head和loss相同不足以保证公平对照。
本次BF16 PE结果与历史FP16无PE结果只能作带精度差异的比较；
严格PE消融还需同精度无PE对照，不能把差异全部解释成PE收益。

## 5. 新A800任务的数值规范

以下是新任务必须显式配置的要求，不是所有旧任务的当前状态：

```bash
export AMP_DTYPE=bf16
export CLIP_GRAD=1.0
export FINITE_CHECK_INTERVAL=1
```

还需显式设置DIMENSION、SLOT_HEAD_TYPE、PIXEL_PE、SEGMENTATION_UNIT、
loss和采样参数。不得从其他实验的shell环境隐式继承MAE/RESUME配置。

| 保护措施 | 能做什么 | 不能保证什么 |
|---|---|---|
| BF16 autocast | 比FP16更宽的指数范围；当前入口不启用FP16 GradScaler | 不能防止除零、非法运算或所有反向不稳定 |
| 梯度裁剪1.0 | 在梯度有限时限制全局范数 | 不能将已产生的NaN/Inf修复为有效梯度 |
| strict finite checks | 异常时停止，避免污染optimizer/参数 | 不等于训练能够自动恢复 |
| FP32局部fallback | 处理特定算子兼容性或精度问题 | 仅对已覆盖的算子/路径有效，不代表全模型FP32 |
| checkpoint | 提供可追溯恢复点 | 文件存在不等于参数有限、配置正确或可成功续训 |

当前scaler流程：backward → unscale → strict clip → step → scaler update。
FP16下如果strict clip先抛错，就到不了GradScaler正常的skip/backoff。
不要通过关闭 `error_if_nonfinite`、静默跳过异常或nan_to_num掩盖问题。

## 6. 2026-09-08事故记录与修复范围

任务2910849在epoch64 step233出现非有限梯度。启动日志和完整CLI均显示
`amp_dtype=fp16`；原因是PE作业脚本没有显式设置AMP_DTYPE。
首个异常算子未被旧日志记录，**FP16溢出仍是机制假设，不是已复现的算子级根因**。

已实施：
- 2D PE脚本固定BF16、clip1.0、finite interval1，并检查GPU支持BF16。
- backward/update错误报告包含异常梯度参数名称和scaler状态。
- checkpoint从每100改为每25个epoch。
- 同一分配内先进行两次更新DDP预检查，单独输出，通过后从头正式训练。
- CPU侧5项回归测试通过；实际GPU预检查和跨epoch64验证仍需看新任务日志。
- 新训练2919868；评估2919869依赖afterok:2919868；旧失效评估2910855已取消。

详见[数值修复记录](PIXEL_PE_NUMERICS_20260908.md)。
两次更新只能验证启动链路，不能证明长期稳定，不能提前标记“NaN彻底解决”。

## 7. 三账号、两集群：路径与额度分离

每个任务必须记录：
登录用户、集群、Slurm account、QoS、partition、GPU型号/数量、
环境Python绝对路径、worktree、训练/评估CSV、ROI索引、LPIPS权重、
预训练/续训checkpoint、输出目录。

- bolinren19/SIP已用数据根为 `/gpfs/work/aac/bolinren19/2026-07/DATA`。
- sifansong与antengcai23的XEC工作目录分别在各自用户根下；数据路径须现场读取其配置，
  不能通过替换用户名猜测存在性。
- 登录用户名不等于Slurm account；QoS名也不是GPU类型。
- SIP与XEC Job ID属于不同命名空间；不能用SIP sacct核验XEC任务。
- 训练与评估必须各自核验环境和路径，不能只验证训练目录。
- README不保存密码、私钥或token；访问配置使用本地HPC skill。

## 8. 提交前及运行后审核清单

### 提交前

1. 确认统一分支修复与实际运行快照一致，记录commit和dirty diff。
2. 阅读作业脚本及其调用的启动器，检查显式参数与默认回退。
3. 检查真实CSV、ROI索引、权重、Python环境及独立输出目录；不得覆盖旧结果。
4. 核对每卡batch×GPU数×accum_iter；3D同时记录slab单位。
5. 运行配置/PE回归测试及bash语法检查：
   `python -m unittest discover -s tests -p 'test_pixel_pe*.py' -v`。
6. 先分配内验证再正式训练；不要截断ROI池制造无效smoke。
7. sbatch返回Job ID后，通过scontrol确认account、QoS、资源、WorkDir及依赖。

### 启动后

核验同一输出目录中的 `resolved_config.json`、`command.txt`、启动日志：
AMP、head、PE、监督单位、spacing、loss、batch、初始化必须与实验表一致。
不以README、脚本名称或环境变量的“预期值”替代真实解析值。
检查首个epoch有限性、ROI路径、weighted_seg≈0.01×seg_loss；
checkpoint保存后检查args、epoch、可加载性及参数有限性。
本次PE额外要求稳定跨过epoch64，之后检查checkpoint75/100。

### 评估及依赖

- sbatch保存提交时的batch脚本；修改本地脚本不会改写已提交副本。
  但脚本调用的外部Python/shell通常在运行时读取，所以代码快照也必须冻结。
- 旧afterok上游失败后不能自动改为依赖新任务。重新建立训练—评估链并记录新ID。
- evaluator从checkpoint args重建head/PE，严格加载权重；核验八类、病例数和完整性。
- 固定0.5与校准阈值结果分别列出；历史训练集校准协议必须明确标注，
  新阈值选择优先用验证集，不能在测试集挑最佳阈值。
- 3D必须报告病例体积级指标，注明slab重叠融合、后处理及head/reconstruction路径。

## 9. 尚未自动化的部分

当前并没有全仓库唯一的强制配置schema，也没有检查所有sbatch的通用验证器。
现有新测试只防止2D PE精度入口回退；其他实验仍必须逐任务审核。
本规范不修改通用FP16默认、不改变正在运行的实验、不自动重提任务。
后续若实现全局强制校验，应另行测试历史兼容性，不能仅凭文档声称已经完成。

````

<a id="source-14"></a>

## docs/ORGSLOT_WORD070_ARM_E_MULTISCALE_PIXEL.md

SHA256: `71d8fc335d66ffd65f13f5adaa376d277e3297fb9dd0b12551eb2c07d6fd8601`

````text
# OrganSlot Arm E: high-resolution multiscale pixel decoder

## Research question

Arm C and Arm D obtain similar WORD 2D results even though Arm D keeps 20
independent token queries. Arm E therefore tests a different bottleneck:
whether the final 28x28 ViT feature lacks the spatial detail required by small
organs.

The controlled change is:

    Arm C: ViT P16 -> upsampled pixel feature -> single organ query dot product
    Arm E: ViT P16 + image P8 + image P4 -> fused pixel feature
                                        -> same single organ query dot product

The convolutional path never predicts an organ mask by itself. The raw mask is
always the normalized dot product between the shared pixel feature and an
OrganSlot query.

## Exact architecture

For a 448x448 input:

    image -> lightweight Conv/GN/GELU stem -> P4  [B,128,112,112]
                                        -> P8  [B,128, 56, 56]
    ViT final patch tokens                 -> P16 [B,128, 28, 28]

    P16 upsample + projected P8 -> depthwise/pointwise refinement
         upsample + projected P4 -> depthwise/pointwise refinement
                                -> pixel feature [B,128,112,112]

    20 OrganSlot tokens -> Arm C mean aggregation/projection
                        -> one query [B,128]

    FP32 normalized dot product -> [B,1,112,112] -> bilinear 448x448

Arm E adds 256,320 parameters. The formal model has 186,799,826 total
parameters; the Arm E shared pixel decoder has 491,840 parameters.

## Fixed experimental controls

- WORD 2D, 8 foreground organs and one background slot
- spacing 0.7 x 0.7 x 2.0 mm
- input 448, ROI crop 384, ROI probability 0.2
- token factor 20, one aggregated organ query
- retained segmentation supervision, background segmentation weight 0
- MSE + LPIPS + 0.01 x small-organ segmentation loss
- Tversky FP/FN 0.3/0.7
- balanced focal weight 0.5, focal alpha/gamma 0.75/2
- top-2% hard negatives, negative-slice weight 0.1
- effective batch 192, 118800 optimizer updates, seed 0

No query-count, loss, sampling, ViT patch-size, threshold, or reconstruction
change is part of Arm E v1.

## Verification

Local tests:

- Arm C/D/E model regression: 20/20 passed
- retained/small-organ loss regression: 20/20 passed
- Arm E checkpoint compatibility: passed
- B=2 at 448x448, P4/P8/P16 and K=1/3/8 shapes: passed
- stem, P16 projection, OrganCollector, and query gradients: non-zero
- zero query: raw mask is exactly zero
- shuffled query: raw mask changes
- reconstruction with shared weights: Arm C and Arm E are bitwise equal

Single-GPU real-data smoke:

- SIP job 2814754, 1x A800, completed
- 4 optimizer updates completed without NaN/Inf
- peak allocated GPU memory: 12.76 GiB
- mean segmentation loss: 1.0118
- mean weighted segmentation loss: 0.0101
- observed ROI fraction: 0.1875
- checkpoint and formal smoke validator passed

Two-GPU DDP:

- job 2814794 exposed a launcher import-path error before model execution
- launcher fixed in commit 75c3b45
- replacement SIP job 2864374 requests 2x A800 under 8a800
- the synthetic DDP preflight deliberately gives rank 0 kidney and rank 1
  spleen, then performs three optimizer steps with fixed collective schema

## Slurm entry points

- single GPU smoke:
  slurm/orgslot/smoke/orgslot_word07072_arm_e_single_bolin_sip.sbatch
- two GPU DDP smoke:
  slurm/orgslot/smoke/orgslot_word07072_arm_e_ddp_bolin_sip.sbatch
- formal four GPU training:
  slurm/orgslot/train/orgslot_word07072_arm_e_bolin_sip.sbatch

Formal training must not be submitted until job 2864374 passes.

## Success criteria

Compare checkpoint-802 to Arm C with the same 24 test cases, fixed threshold
0.5, and identical postprocessing. Report all eight organs, overall foreground
Dice, and the mean of gallbladder, esophagus, and pancreas. Arm E supports the
hypothesis only if the small organs improve coherently rather than trading one
organ against another.
````

<a id="source-15"></a>

## docs/ORGSLOT_WORD070_MULTIQUERY_EXPERIMENT.md

SHA256: `7e6974940ee9cc442b9bcd2137b74d4598e62a0d2fda3ac91e920bcc035fe429`

````text
# WORD 0.7 ROI20 Multi-Query Pixel-Mask Experiment

## Question

Does preserving all 20 TGEnc token queries improve organ-mask readout over the
single pooled-query Arm C while preserving the same pixel and reconstruction paths?

## Architecture

The reconstruction path is unchanged:

```text
z -> OrganCollector_s -> TGEnc_s -> AHER_s -> canvas_s -> fusion -> reconstruction
```

The new segmentation path is:

```text
z -> shared pixel decoder -> P(x)
TGEnc_s token k -> shared query projection + slot identity -> q_{s,k}
part_logit_{s,k}(x) = sqrt(D) * cosine(P(x), q_{s,k})
mask_logit_s(x) = logmeanexp_k(part_logit_{s,k}(x))
```

The shared pixel decoder upsamples the final 28x28 ViT patch grid to a 112x112
128-channel feature map. All 20 token queries produce part-response maps. Log-mean-exp is used as a
smooth maximum with no token-count logit bias; identical token queries exactly
recover the single-query score. Per-slot logits are bilinearly resized to
448x448.
Segmentation gradients update the shared ViT, shared pixel/query decoder,
OrganCollector, TGEnc, and slot identity, but do not pass through AHER.

## Locked comparison

This Arm D changes only the token-to-query aggregation relative to Arm C.

| Setting | Arm A | Arm B | Arm C | Arm D |
|---|---|---|---|---|
| head | linear | multiscale_conv | pooled query-dot | multi-query-dot |
| head input | AHER canvas | AHER canvas | mean of 20 TGEnc tokens + final ViT pixels | 20 independent TGEnc queries + same pixels |
| query aggregation | n/a | n/a | mean before projection | log-mean-exp after 20 part maps |
| spacing | 0.7x0.7x2.0 | same | same | same |
| input / ROI crop | 448 / 384 | same | same | same |
| ROI probability | 0.2 | same | same | same |
| supervision | retained | same | same | same |
| loss | small_organ | same | same | same |
| lambda_seg | 0.01 | same | same | same |
| effective batch | 192 | same | same | same |
| updates / warmup | 118800 / 5940 | same | same | same |
| optimizer / base LR / WD | AdamW / 1e-4 / 0.05 | same | same | same |
| seed | 0 | same | same | same |

No diversity loss is added in the first controlled run. Token specialization or
collapse must be measured after training instead of changing head and loss at
the same time.

## Run registry

- Branch: `experiment/orgslot-querymask-multiquery`
- Implementation commit: `606c7ed`
- Account/cluster: `bolinren19 / SIP`
- Smoke: Job `2548701`, 2xA800, 16 CPU, 128 GB
- Formal train: Job `2548702`, 4xA800, 24 CPU, 192 GB, `afterok:2548701`
- Formal output: checkpoint-802 after 118,800 optimizer updates

## Acceptance gates

1. Unit tests: finite forward/backward, exact mask shapes, dropped-row zeros,
   shared pixel and slot-query gradients, no AHER segmentation gradient.
2. Real 0.7 ROI20 smoke: eight optimizer updates, finite reconstruction,
   LPIPS, small-organ metrics, positive focus-organ coverage, checkpoint keys.
3. Formal training is submitted with `afterok` on the smoke job.
4. Final evaluation must use the same validation/test inference protocol as
   Arm A/B, including fixed 0.5 head threshold as the primary result.
````

<a id="source-16"></a>

## docs/ORGSLOT_WORD070_QUERYMASK_3D_EXPERIMENT_CN.md

SHA256: `15c965a457b094a32c7afaf4063a84867d3e52531115dc460caef017346b83c8`

````text
# OrganSlot 3D Arm C/D：WORD 0.7 spacing + ROI20

## 研究问题

在4层short-slab 3D协议下，直接保留ViT空间patch特征的query-mask读出是否仍优于
AHER canvas上的multiscale head；保留20个独立token queries是否进一步优于把20个
tokens汇总成一个器官query？

## 严格对照

| Arm | 唯一主要变量 | 分割空间来源 | 器官query |
|---|---|---|---|
| 3D B | `multiscale_conv` | AHER canvas | 无动态query |
| 3D C | `query_dot` | 共享3D ViT pixel decoder | 20 tokens均值后形成1个query |
| 3D D | `multi_query_dot` | 与3D C相同 | 20个独立queries，以无token数偏置的log-mean-exp聚合 |

其余配置固定：0.7×0.7×2.0 mm、448输入、384 ROI crop、ROI概率0.2、连续4层、
stride1、20 tokens/organ、small-organ loss、`lambda_seg=0.01`、BF16、clip-grad1.0、
有效batch 48 slabs（约192 slices）、actual LR 7.5e-5、118800 optimizer updates和seed0。

3D C/D均从随机初始化开始，不加载2D C/D checkpoint，避免把架构收益和2D预训练收益
混在一起。训练集和测试集由账号无关的规范化SHA256门禁验证。

## 3D实现

共享pixel decoder把ViT输出reshape为`[B,C,T,H,W]`，只在xy方向逐级上采样；
每一级3×3×3 depthwise convolution交换相邻层信息。所有3D trilinear resize在FP32
执行后转回原dtype，以兼容XEC PyTorch 1.13的BF16算子集合。

3D C：

[
q_s=W_q\operatorname{Mean}_k(\operatorname{LN}(T_{s,k}))+e_s,
\qquad
l_s(t,x,y)=\sqrt{D}\,\hat P(t,x,y)^\top\hat q_s.
]

3D D：

[
l_{s,k}(t,x,y)=\sqrt{D}\,\hat P(t,x,y)^\top\hat q_{s,k},
]

[
l_s=\operatorname{logsumexp}_k(l_{s,k})-\log K.
]

减去`log K`消除token数量导致的固定logit偏置；当20个queries完全相同时，3D D与
3D C输出严格一致。

## 执行账号

- 3D C：`sifansong/XEC`，账号根目录
  `/gpfs/work/aac/sifansong`；
- 3D D：`antengcai23/XEC`，账号根目录
  `/gpfs/work/aac/antengcai23`。

两个账号分别使用自己的Python、数据、LPIPS和结果目录，不交换绝对路径。分开提交是为了
利用各自的4-GPU QOS并行排队，不代表数据协议不同。

## 验收与判定

训练验收：

1. 4张A800可见；
2. 数据SHA256门禁通过；
3. 3D query/pixel decoder均有非零有限梯度；
4. 无NaN/Inf，`weighted_segmentation_loss≈0.01×segmentation_loss`；
5. 完成118800 updates并生成checkpoint-199。

正式结果只使用真正病例级3D评估：24病例、八类、固定0.5阈值，报告Dice、IoU、
precision、recall、NSD、HD95和预测/GT体积比。主要比较为3D C-B和3D D-C；在三臂
全部完成前不做最终架构结论。
````

<a id="source-17"></a>

## docs/ORGSLOT_WORD070_QUERYMASK_EXPERIMENT.md

SHA256: `e7219d7a5bd9b813f438251c377ee6c2944b5d2508c63e9263bbc072577be135`

````text
# WORD 0.7 ROI20 Shared Pixel Query-Mask Experiment

## Question

Does bypassing the AHER canvas for segmentation improve organ-mask readout while
preserving the existing OrganSlot reconstruction path?

## Architecture

The reconstruction path is unchanged:

```text
z -> OrganCollector_s -> TGEnc_s -> AHER_s -> canvas_s -> fusion -> reconstruction
```

The new segmentation path is:

```text
z -> shared pixel decoder -> P(x)
TGEnc_s tokens -> shared query projection + slot identity -> q_s
mask_logit_s(x) = sqrt(D) * cosine(P(x), q_s)
```

The shared pixel decoder upsamples the final 28x28 ViT patch grid to a 112x112
128-channel feature map. Per-slot logits are bilinearly resized to 448x448.
Segmentation gradients update the shared ViT, shared pixel/query decoder,
OrganCollector, TGEnc, and slot identity, but do not pass through AHER.

## Locked comparison

Arm C changes only the segmentation readout relative to the existing 0.7
spacing Arm A/B runs.

| Setting | Arm A | Arm B | Arm C |
|---|---|---|---|
| head | linear | multiscale_conv on AHER canvas | shared pixel query-dot |
| spacing | 0.7x0.7x2.0 | same | same |
| input / ROI crop | 448 / 384 | same | same |
| ROI probability | 0.2 | same | same |
| supervision | retained | same | same |
| loss | small_organ | same | same |
| lambda_seg | 0.01 | same | same |
| Tversky FP/FN | 0.3/0.7 | same | same |
| Balanced Focal weight | 0.5 | same | same |
| hard-negative ratio | 0.02 | same | same |
| negative-slice weight | 0.1 | same | same |
| effective batch | 192 | same | same |
| updates / warmup | 118800 / 5940 | same | same |
| optimizer / base LR / WD | AdamW / 1e-4 / 0.05 | same | same |
| seed | 0 | same | same |

## Acceptance gates

1. Unit tests: finite forward/backward, exact mask shapes, dropped-row zeros,
   shared pixel and slot-query gradients, no AHER segmentation gradient.
2. Real 0.7 ROI20 smoke: eight optimizer updates, finite reconstruction,
   LPIPS, small-organ metrics, positive focus-organ coverage, checkpoint keys.
3. Formal training is submitted with `afterok` on the smoke job.
4. Final evaluation must use the same validation/test inference protocol as
   Arm A/B, including fixed 0.5 head threshold as the primary result.
````

<a id="source-18"></a>

## docs/ORGSLOT_WORD070_SMALL_ORGAN_LOSS_CN.md

SHA256: `4a8453d84eb2966341558d00519f96b8e917f9e82a4242e25799caf7dc5d1ebe`

````text
# WORD 0.7-space 当前训练损失说明

## 1. 适用实验

本文描述当前 WORD 2D OrganSlotBank 小器官实验使用的训练损失。对应配置为：

- 数据 spacing：`0.7 × 0.7 × 2 mm`；
- 输入尺寸：`448 × 448`；
- segmentation head：`multiscale_conv`；
- segmentation supervision：仅监督 TGR 当前保留的 organ slots（`retained`）；
- ROI sampling probability：`0.2`，与对照实验一致；
- 不使用额外 organ weighting；
- 不使用 PCGrad；
- background slot 的 segmentation weight 为 `0`。

正式训练入口为：

```text
slurm/orgslot/train/orgslot_word07072_roi20_retained_multiconv_smallorgan_xec.sbatch
```

## 2. 总损失

当前仍然是单阶段联合训练，同时优化重建任务和分割任务：

\[
\mathcal L_{total}
=\mathcal L_{rec}
+\lambda_{lpips}\mathcal L_{lpips}
+\lambda_{seg}\mathcal L_{seg}.
\]

当前权重为：

\[
\lambda_{lpips}=1,\qquad \lambda_{seg}=0.01.
\]

因此实际配置是：

\[
\boxed{
\mathcal L_{total}
=\mathcal L_{MSE}
+\mathcal L_{LPIPS}
+0.01\mathcal L_{small\text{-}organ}
}
\]

其中：

- `MSE` 约束像素级重建；
- `LPIPS` 约束感知结构和纹理；
- `small-organ` loss 监督 retained organ slots 的 binary masks。

重建 target 不是无条件复制原始图像，而是由当前 TGR keep mask 和可见器官 mask 构造的重建目标。分割 loss 不改变该重建 target。

## 3. 单个器官、单张切片的分割损失

每个 organ slot 都是一个独立二分类问题。设：

- \(z(x)\)：像素 \(x\) 的预测 logit；
- \(p(x)=\sigma(z(x))\)：前景概率；
- \(y(x)\in\{0,1\}\)：该器官的 GT mask。

根据当前切片是否包含该器官，采用不同的损失。

### 3.1 阳性切片

如果：

\[
\sum_x y(x)>0,
\]

则使用：

\[
\boxed{
\mathcal L_{+}
=\mathcal L_{Tversky}
+0.5\left(
\mathcal L_{Focal}^{fg}
+\mathcal L_{Focal}^{hard\text{-}bg}
\right)
}
\]

#### Tversky loss

\[
TP=\sum_x p(x)y(x),
\]

\[
FP=\sum_x p(x)(1-y(x)),
\]

\[
FN=\sum_x (1-p(x))y(x).
\]

当前定义：

\[
\mathcal L_{Tversky}
=1-
\frac{TP+\epsilon}
{TP+0.3FP+0.7FN+\epsilon},
\qquad \epsilon=10^{-6}.
\]

这里 FN 的系数 `0.7` 大于 FP 的 `0.3`，表示训练初期更重视减少漏分。对于胆囊、食管和胰腺等小器官，漏掉少量前景像素就可能破坏主体结构，因此当前设置偏向提高召回率。

#### 前景 Focal

前景像素单独平均：

\[
\mathcal L_{Focal}^{fg}
=\frac{1}{|\Omega_{fg}|}
\sum_{x\in\Omega_{fg}}
0.75(1-p(x))^2\operatorname{BCE}(z(x),1).
\]

这样前景贡献不再被大量背景像素稀释。

#### 困难背景 Focal

对每个背景像素计算：

\[
h(x)=0.25p(x)^2\operatorname{BCE}(z(x),0).
\]

只选择损失最大的前 `2%` 背景像素：

\[
\mathcal L_{Focal}^{hard\text{-}bg}
=\operatorname{mean}\left(\operatorname{TopK}_{2\%}\{h(x)\}\right).
\]

这部分主要处理器官边界附近、相邻器官区域和模型高置信度误报位置，而不是让大量容易背景主导优化。

### 3.2 阴性切片

如果当前切片完全不含该器官：

\[
\sum_x y(x)=0,
\]

则不计算 Tversky，也不使用全图 BCE 平均。当前损失为：

\[
\boxed{
\mathcal L_{-}
=0.1\cdot
\operatorname{mean}\left(
\operatorname{TopK}_{2\%}
\{\operatorname{BCE}(z(x),0)\}
\right)
}
\]

它只惩罚模型最容易误报的 `2%` 像素，并将整个阴性切片分支乘以 `0.1`。这可以保留对假阳性的约束，同时防止数量众多的阴性切片压倒稀少的阳性切片。

Top-K 数量使用向上取整，并保证至少选择一个像素。

## 4. 一个 batch 内如何聚合

对器官 \(c\)，阳性切片和阴性切片分别求平均：

\[
\mathcal L_c
=\operatorname{mean}_{i\in P_c}(\mathcal L_{+,i})
+\operatorname{mean}_{j\in N_c}(\mathcal L_{-,j}).
\]

如果当前 batch 没有某一类切片，对应项记为 0。关键点是先分别平均，再相加，而不是把所有切片直接混在一起平均。因此增加阴性切片数量不会自动稀释阳性切片的梯度。

随后，对当前 batch 中实际受到监督的非背景 organ slots 等权平均，得到：

\[
\mathcal L_{seg}=\operatorname{mean}_{c\in\mathcal C_{supervised}}\mathcal L_c.
\]

当前没有额外的器官级权重。background slot 虽可记录诊断指标，但其 segmentation weight 为 0，不进入 \(\mathcal L_{seg}\)。

## 5. 为什么总权重使用 0.01

旧 focal loss 对全图像素直接求均值，数值通常较小；当前损失包含一个接近 \([0,1]\) 范围的 Tversky 项，并把前景和困难背景分别平均，因此原始数值自然约为 1，不能沿用旧 loss 的数值尺度直接比较。

已通过的两卡 smoke 中：

- 原始 segmentation loss：约 `1.0407`；
- 加权 segmentation term：约 `0.0104`；
- reconstruction MSE：约 `0.2457`；
- LPIPS：约 `0.8087`。

所以 `lambda_seg=0.01` 的作用是让分割梯度产生明确影响，同时避免在实验开始时突然压过重建目标。这里比较的是乘权重后的 loss/梯度影响，而不是仅比较原始 loss 数值。

## 6. 数值稳定性与多卡语义

- small-organ loss 内部将 logits 转为 FP32 后计算；
- 每一步检查 total loss 是否为有限值，出现 NaN/Inf 会立即终止；
- 所有 DDP ranks 使用固定的 per-organ 诊断字段，避免集合通信次数不一致；
- 没有监督样本的器官指标记录为 0，并配合 `supervised_samples=0` 解读；
- dropped slots 不计算 segmentation gradient；
- 当前仍是普通加权求和，不使用 PCGrad 或两阶段训练。

## 7. 需要重点观察的训练指标

总量指标：

- `reconstruction_loss`；
- `p_loss`；
- `segmentation_loss`；
- `weighted_segmentation_loss`；
- `loss`。

每个器官分别记录：

- `seg_<organ>_positive_samples`；
- `seg_<organ>_negative_samples`；
- `seg_<organ>_tversky_loss`；
- `seg_<organ>_positive_focal_loss`；
- `seg_<organ>_hard_negative_focal_loss`；
- `seg_<organ>_empty_negative_loss`；
- `seg_<organ>_positive_slice_predicted_fraction`；
- `seg_<organ>_empty_slice_predicted_fraction`；
- `seg_<organ>_target_fraction`。

判断是否改善小器官时，不能只看总 loss 是否下降，还要检查：

1. 阳性切片预测比例是否从接近全图前景逐渐收缩到合理区域；
2. 阴性切片预测比例是否下降，说明假阳性受到抑制；
3. Tversky loss 是否下降，说明召回和区域重叠在改善；
4. 前景 Focal 是否下降，说明小器官前景置信度在提高；
5. 困难背景和空切片 loss 是否下降，说明边界误报和无器官误报在减少；
6. 最终仍须用验证集 Dice、HD95/NSD 和逐器官结果判断效果，训练 loss 不能替代分割评估。

## 8. 代码对应关系

- 总损失组合与训练监控：`engine_pretrain_orgslot_common8_a100.py`；
- Tversky、balanced focal、hard-negative mining 和 batch 聚合：`losses_orgslot.py`；
- 参数定义和合法性检查：`main_pretrain_orgslot_common8_a100.py`；
- 0.7-space 正式实验参数：`slurm/orgslot/train/orgslot_word07072_roi20_retained_multiconv_smallorgan_xec.sbatch`；
- smoke 验证器：`tools/validate_orgslot_jointseg_smoke.py`。
````

<a id="source-19"></a>

## docs/ORGSLOT_WORD448_JOINT_SLOT_SEG_V1.md

SHA256: `44fb283eea3bcbdad2c403cbd68abe13748afdb65d708f12f2ed077a5b1b0c00`

````text
# OrganSlot WORD448 Joint Slot Segmentation v1

## 1. 目标与假设

ROI20-only 的 Direct post Dice 为 75.20%，ROI20+融合后 Loss3 为 65.18%。
v1 不再对融合重建结果增加器官 ROI loss，而是用每个 slot 自己的 binary head
约束该 slot canvas 必须能独立解码对应器官。

主目标为：

\[
L=L_{global\ L2}+L_{LPIPS}+0.01L_{slot\ Dice+BCE}.
\]

该损失增强“对应器官可从对应 slot 解码”，但不预先宣称其他 slot 完全不含
该器官。是否真正解决职责混合，要联合观察 head Dice 与 reconstruction Direct。

## 2. 主实验配置

| 参数 | 值 |
|---|---|
| 数据 | WORD 2D native-resampled，448 deterministic eval crop |
| 训练增强 | ROI20，384 crop resize 到 448 |
| ROI probability | 0.20 |
| fusion | `linear_sqrt`，reference count 9 |
| reconstruction | Global L2 + LPIPS，`lambda_lpips=1` |
| fused Loss3 | 不存在；最新版分支已删除该实现 |
| slot segmentation | Dice+BCE，`lambda_seg=0.01` |
| background segmentation | `lambda_bg_seg=0.25` |
| segmentation supervision | `all` |
| reconstruction/fusion mask | 原始真实随机 keep/drop |
| effective batch | 192 |
| optimizer updates | 118800 |

`all` 只改变 segmentation loss 的监督选择：被 drop 的 slot 不参与融合，但它的
binary head、AHER、TGEnc 和 Collector 仍接受自己器官的监督。重建 target 不变，
不增加第二次 decoder forward，Direct/Indirect 协议不变。

## 3. 分阶段实验

### Gate A：单元测试

必须证明：

1. `all` 时 dropped slot 的 head、slot 路径和共享 backbone 获得非零梯度；
2. `retained` 保持旧行为，dropped head 梯度为零；
3. segmentation supervision mode 不改变 reconstruction target；
4. segmentation-only backward 不直接更新 reconstruction decoder；
5. optimizer 能完成一步并改变 dropped slot head 参数；
6. 旧 checkpoint、fusion、head-only probe 测试仍通过。

当前结果：59 项全量测试通过。

### Gate B：deterministic tiny overfit

- 8 个真实训练样本；
- deterministic center crop，关闭随机图像增强；
- 每个样本固定 keep mask；
- 1 GPU，有效 batch 4，400 updates；
- 同时优化 L2、LPIPS 和 `0.01 * segmentation`。

验收：total、reconstruction、segmentation 三项末窗口均低于初始窗口的 90%，
所有 9 个 slot 均有监督且产出 checkpoint。

### Gate C：真实 ROI forward/backward smoke

- 真实 ROI index；
- ROI probability 暂设为 1.0，仅为强制覆盖 ROI 路径；
- 2 GPU，4 updates；
- 检查 loss 有限、每 slot loss/监督数存在、加权项精确为
  `0.01 * segmentation_loss`、无旧 Loss3 指标、产出 checkpoint。

### Gate D：正式训练

只有 Gate B、C 均通过后才提交。正式 ROI probability 恢复为 0.20，其他训练
预算与 ROI20-only 完全一致。

## 4. 正式对照与后续消融

| ROI20 | slot segmentation | supervision | 用途 | 状态 |
|---:|---:|---|---|---|
| 0 | 0 | - | control，Direct 64.44% | 已完成 |
| 1 | 0 | - | ROI20 baseline，Direct 75.20% | 已完成 |
| 1 | fused Loss3 | - | 失败组合，Direct 65.18% | 已完成 |
| 1 | 0.01 | all | v1 主实验 | Gate B 前，不提交正式 |
| 1 | 0.01 | retained | 判断 dropped-slot supervision 的贡献 | 主实验后按需 |
| 0 | 0.01 | all | 判断 ROI 与 seg 的交互 | 只在主实验结果含糊时 |

不把 segmentation-only 从头训练作为 reconstruction 正式对照，因为它不训练
shared reconstruction decoder。已有 frozen head-only probe 仅作为表示诊断。

若主实验明显提升，再以短预算比较 `0.01/0.025/0.05`，不同时提交三组完整训练。

## 5. 评估

### Reconstruction

- 固定 threshold 0.02；
- Direct raw/post；
- Indirect raw/post；
- 每器官 case Dice/global Dice；
- 每类预测体积/GT体积比。

### Binary heads

- 固定 sigmoid threshold 0.5 是正式主结果；
- raw/post Dice；
- 每器官 case/global Dice；
- 每类预测体积/GT体积比。

当前数据只有 Train/Test，没有独立 validation。禁止在 Test 上选阈值。为诊断纯
校准漂移，可以在最多 6000 个训练样本上选择每类 threshold，再冻结到 Test；
该结果明确标注为 `train_calibration`，不能冒充独立 validation 结果，也不能替代
固定 0.5 主结果。

## 6. 判读

1. head Dice 与 Direct 都提高：slot identity 辅助监督成功。
2. head Dice 高、Direct 不提高：瓶颈位于 slot canvas→fusion→shared decoder。
3. Direct 提高、Indirect 下降：共享 encoder/其他 slot 可能重新读取 dropped organ，
   优先比较 `all` 与 `retained`，不要增加 fused ROI loss。
4. head Dice 低且预测体积接近零：大量空切片负监督可能压制稀有器官，先检查
   阳性/空标签统计，再决定是否调整 sampling 或 loss。
5. 固定 0.5 差、train-calibrated threshold 好：主要是 head calibration drift，
   不是 reconstruction Direct 已解决。

## 7. 文件与任务

核心代码：

- `engine_pretrain_orgslot_common8_a100.py`
- `main_pretrain_orgslot_common8_a100.py`
- `losses_orgslot.py`
- `scripts/orgslot/run_word_112_448_a100.sh`

验证/评估：

- `tools/validate_orgslot_jointseg_tiny.py`
- `tools/validate_orgslot_jointseg_smoke.py`
- `tools/eval_common8_orgslot_heads.py`
- `tools/eval_common8_orgslot_reconstruction_threshold.py`

Job IDs：等待 Gate B 提交后填写。

## 8. Focal v2 正式实验

Dice+BCE tiny 的数值门禁虽然通过，但最终 background loss 约为 0.0036，
八个前景 slot loss 均接近 1.0，存在全背景塌缩风险。因此 v2 保留联合重建、
ROI20 和 all-slot supervision，只替换独立 binary head 的损失：

\[
L=L_{global\ L2}+L_{LPIPS}+0.1L_{slot\ focal}.
\]

固定配置：

- `seg_loss_type=focal`；
- `focal_alpha=0.75`，其中 alpha 是前景正类权重；
- `focal_gamma=2.0`；
- `lambda_seg=0.1`；
- `seg_supervision=all`；
- `organ_roi_probability=0.20`；
- fused Loss3 关闭。

Focal 的原始数值约为 Dice+BCE 的十分之一，因此使用 0.1 而不是直接沿用
0.01。该设置预期使加权辅助项约为 0.004，仍小于旧 Dice+BCE 联合实验的
0.01--0.02，避免分割目标主导重建。

训练日志额外记录每个 slot 的 `predicted_fraction`、`target_fraction` 和
`positive_probability`。smoke 必须验证这些字段有限，正式判断还必须依赖固定
0.5 threshold 的逐 head Dice/预测体积，以及相同 0.02 threshold 的
reconstruction Direct/Indirect Dice。

XEC `antengcai23` 任务链（2026-08-15）：

| 阶段 | Job ID | 依赖/状态 |
|---|---:|---|
| ROI100 Focal smoke | 117005 | 已提交，等待调度 |
| ROI20正式训练 | 117006 | `afterok:117005` |
| reconstruction评估 | 117007 | `afterok:117006` |
| head train calibration | 117008 | `afterok:117006` |
| head test | 117009 | `afterok:117008` |

代码分支：`experiment/orgslot-jointseg-focal-v0`；提交：`7d8343e`。
````

<a id="source-20"></a>

## docs/ORGSLOT_WORD448_RETAINED_MULTISCALE_FOCAL_V2.md

SHA256: `3be6490f8629bf3a2e4a00b1bc592e266efe069dba7f89935883add0572a4e41`

````text
# WORD448 OrganSlot retained multiscale Focal v2

## Objective

Test whether a stronger local segmentation decoder can improve small-organ
readout without breaking OrganSlot retained-query semantics.

## Locked configuration

- dataset: WORD 2D native-resample, 448 input;
- augmentation: ROI20 (384 crop resized to 448), focus classes 4/5/6;
- reconstruction: global L2 + LPIPS;
- segmentation: independent binary sigmoid focal loss;
- focal alpha/gamma: 0.75/2.0;
- segmentation weight: 0.1;
- background segmentation weight: 0;
- supervision: retained slots only;
- head: four-stage multiscale depthwise-separable convolution decoder;
- fusion: linear_sqrt;
- no fused-image Loss3.

## Execution semantics

The shared ViT encodes the complete image batch once. For each slot, only rows
whose slot is retained are gathered into Collector, slot TGEnc, AHER and the
binary head. Results are scattered back to batch layout for fusion and loss.
Dropped rows therefore consume no slot-path/head compute and receive no
segmentation gradient. Legacy all-slot supervision remains available explicitly
and computes the union of reconstruction-retained and head-supervised rows.

## Gates

1. Unit tests verify 2D shape, finite backward, sparse Collector/head batch
   sizes, zero dropped logits, and no-head reconstruction mode.
2. Real ROI100 smoke uses the formal 8 images/GPU microbatch for four optimizer
   updates and must produce finite reconstruction, LPIPS and focal metrics.
3. Formal ROI20 training starts only after the smoke succeeds.
4. Formal evaluation reports fixed-threshold reconstruction Direct/Indirect,
   fixed 0.5 binary-head Dice, and validation-only calibrated head thresholds.

## Job registry

Submitted from login account antengcai23 on XEC using Slurm project account
sifansong:

| Stage | Job ID | Dependency at submission |
|---|---:|---|
| real ROI100 smoke | 119640 | Priority |
| formal ROI20 training | 119641 | afterok:119640 |
| reconstruction Direct/Indirect | 119642 | afterok:119641 |
| head train-calibration | 119643 | afterok:119641 |
| head test | 119644 | afterok:119643 |

Submission audit:

- An initial account-name attempt was rejected before creating jobs.
- Jobs 119609-119613 were cancelled. The smoke file had accidentally been
  overwritten by the formal file because both had the same basename in a
  temporary patch-generation directory; 119609 ran 8 minutes of the formal
  configuration before cancellation and is not an experiment result.
- Commit ad98843 restores a distinct four-update ROI100 smoke and adds shell
  assertions for MAX_UPDATES=4, ACCUM_ITER=1, ROI probability 1.0 and a unique
  smoke tag before launching Python.


## Smoke acceptance and formal start

Corrected smoke 119640 completed on xgpua800n6 in 28 seconds with exit code 0.
It used two A800 GPUs, microbatch 8/GPU, accumulation 1, ROI probability 1.0
and exactly four optimizer updates. The real-data validator passed and saved
checkpoint-0.pth. Averaged finite metrics were:

- total loss: 1.0856;
- reconstruction L2: 0.2633;
- LPIPS: 0.8160;
- focal segmentation: 0.0632;
- weighted focal term: 0.0063;
- peak GPU memory: about 16.3 GiB on the first logged step.

All three focus classes were exercised and FP32 probability monitoring remained
finite. The afterok dependency released formal job 119641, which started on
xgpua800n6. Its first logged batches are finite, use ROI probability 0.2,
effective batch 192 and peak memory about 18.3 GiB. Jobs 119642-119644 remain
blocked behind the formal training/evaluation dependencies.
````

<a id="source-21"></a>

## docs/ORGSLOT_WORD448_ROI20_LOSS3_EXPERIMENT.md

SHA256: `63aefe557217b9aab5a481a2487cc9b0ed0ff1926c11a2cfe89c39a61507bc12`

````text
# OrganSlot WORD 448 ROI20 + LossBalance-v3

## Objective

Test whether OrganSlotBank benefits from combining the 448 native-resolution
protocol and 20% small-organ ROI sampling with the positive-state-only
LossBalance-v3 objective. This is a new combination experiment, not a
loss-matched replacement for the existing OrganSlot 448 control.

## Locked configuration

- Dataset: WORD Common8 2D, 96 training and 24 test cases.
- Preprocessing: RAS, 1x1x2 mm, native resampled matrix, fixed-255 intensity.
- Input: global 448 center crop; 20% ROI branch uses 384 crop resized to 448.
- ROI focus classes: gallbladder 4, esophagus 5, pancreas 6.
- Architecture: OrganSlotBank, 20 tokens/class, slot TG depth 1.
- Fusion: `linear_sqrt`.
- Mask: `legacy_batch`; an ROI focus slot is always retained.
- Segmentation head supervision is disabled: `lambda_seg=0.0`.
- Reconstruction: global L2 + LPIPS.
- Loss-v3: present-and-kept foreground ROI-L2 only, weight 0.25.
- Removed foreground ROI-L2 is monitoring only.
- Frequency parameters: alpha 0.5, maximum weight ratio 4.0.
- Two A800 GPUs, micro-batch 8/GPU, accumulation 12, effective batch 192.
- Optimizer update budget: 118800; warmup updates: 5940.

Natural 448-center-crop positive-slice counts for raw IDs 1..8:

```text
4392 4875 5070 1631 3827 3753 7327 5556
```

These counts are frozen before ROI mixing so ROI20 remains an explicit data
intervention rather than silently redefining Loss-v3 weights.

## Isolation and implementation

Loss-v3 consumes only `visible_masks` returned by `StrictVisibilityDataset`;
the raw multi-class label is never exposed at the training boundary. Positive
and removed states use the exact OrganSlot `[B,9]` keep mask after focus-slot
retention. No trainable model parameter or checkpoint key is added. When the
Loss-v3 coefficient is zero, its per-mask computation is skipped.

## Validation

- Four manifests: 70,792 records checked, zero missing referenced files.
- 29 focused regression tests passed.
- Real WORD gallbladder ROI 448 forward/backward passed with finite backbone
  gradients and nonzero positive supervision.
- Frozen LPIPS state extracted from the completed WORD OWT checkpoint; SHA256
  `a99ef0ef436727d809148f5e9cbc09c1f0218079020918b0b1cc5167493c4287`.

## Jobs

Current retry implementation commit: `2231cd0`. The retry uses
`sifansong/4a800` and two typed A800 GPUs.

| Job | Stage | Status at submission |
|---:|---|---|
| 1651723 | Initial ROI100 Loss-v3 smoke | FAILED before Python: no devices |
| 1651724 | Initial ROI20 Loss-v3 formal | CANCELLED: DependencyNeverSatisfied |
| 1655555 | Corrected seg=0 ROI100 smoke + validator | PENDING (Priority), estimated 2026-08-11 09:33 |
| 1655556 | Corrected seg=0 ROI20 formal training | PENDING, `afterok:1655555` |

The full job uses an `afterok` dependency on a two-GPU smoke whose validator
requires visible CUDA devices, finite global/LPIPS/positive ROI metrics,
exercised ROI samples, and a checkpoint. The retry excludes `gpua800n2` and
`gpua800n6`.

## GPU visibility incident and retry

Jobs 1651723/1651724 are invalid: smoke 1651723 failed before Python with
`No devices were found` on `gpua800n2`; formal 1651724 became
`DependencyNeverSatisfied` and was cancelled. The initial scripts also used
`lambda_seg=1.0`, which did not isolate the planned pure ROI20 + Loss-v3
combination. Retry scripts use `lambda_seg=0.0`, typed A800 GRES, exclude the
two nodes implicated in CUDA invisibility, and require both PyTorch CUDA
discovery and a real CUDA tensor operation before training.

## 2026-08-11 second retry

Smoke `1655555` also started without an allocated GPU, this time on
`gpua800n1`, and failed at the CUDA preflight before any model forward.
Dependent formal job `1655556` was cancelled.

The replacement chain is:

| Stage | Job | Dependency | Initial state |
|---|---:|---|---|
| ROI100 smoke + validator | 1658405 | none | PENDING (Priority) |
| ROI20 formal training | 1658406 | afterok:1658405 | PENDING (Dependency) |

The `sbatch` command line explicitly requests `gpu:a800:2` and excludes
`gpua800n1`, `gpua800n2`, and `gpua800n6`. Post-submission inspection confirmed
`ReqTRES=...gres/gpu=2`, `TresPerNode=gres:gpu:a800:2`, and
`ExcNodeList=gpua800n[1-2,6]` for both jobs.
````

<a id="source-22"></a>

## docs/ORGSLOT_WORD_COMMON8_SOTA_CN.md

SHA256: `32257a4ac336a7ca5cb2293c0367d3ca6c525e0ba0d2248a59b49d3d562a4458`

````text
# OrganSlot WORD Common8 八类结果与项目内 SOTA

更新时间：2026-08-28

## 排名口径

本表只比较 WORD 同一完整测试集上的 OrganSlot 实验：24 个病例、6990 张切片，
八类顺序固定为脾脏、右肾、左肾、胆囊、食管、胰腺、肝脏、胃。指标为
case-level Dice presence mean，表中乘以 100。

主排名要求同一个 checkpoint、同一种输出和一套预先固定的后处理，不允许按测试集
为不同器官临时挑选不同方法。小器官均值固定为胆囊、食管、胰腺三类的算术平均。
这里的“SOTA”仅指本项目内已完成且协议匹配的实验，不代表公开 WORD benchmark
或 PCDD 的外部 SOTA。

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

## 相对 0.7 ROI20 无 head 基线的增量

| 实验 / 输出 | 脾脏 | 右肾 | 左肾 | 胆囊 | 食管 | 胰腺 | 肝脏 | 胃 | 小器官均值 | 八类均值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Arm B reconstruction | +0.63 | +0.45 | +0.40 | **+13.19** | +2.16 | +0.83 | +0.74 | -0.12 | **+5.39** | **+2.29** |
| Arm B head fixed 0.5 | +0.52 | +0.02 | +0.28 | +9.18 | +0.54 | -0.50 | +0.57 | +3.25 | +3.07 | +1.73 |
| Arm B head train-calibrated | +0.87 | +0.70 | +0.87 | +10.91 | +1.43 | -0.81 | +0.57 | +3.07 | +3.84 | +2.20 |
| 0.7 spacing 相对 1.0 spacing | +1.89 | +1.07 | +2.04 | -0.06 | **+8.65** | **+5.61** | +0.58 | +3.92 | +4.73 | +2.96 |

主要结论：

1. 0.7 spacing 的主要收益在食管、胰腺和胃，证明更高平面分辨率对细长/低对比结构有效；
   胆囊并未仅靠 spacing 改善。
2. Arm B joint small-organ training 让 reconstruction 平均提升 2.29，胆囊提升
   13.19，说明分割监督确实改善了共享表示与 AHER canvas。
3. Arm B 的 multiscale head 固定阈值结果比同 checkpoint reconstruction 低 0.56；
   小器官均值低 2.32。因此目前证据支持“联合监督有效”，尚不支持
   “AHER-canvas multiscale head 优于 reconstruction”。
4. train calibration 改善了 head 总均值，但未改变上述结论，且胰腺仍低于固定阈值结果。

## 八类逐器官观测最优

| 器官 | 当前最高 Dice | 来源 |
|---|---:|---|
| 脾脏 | **92.98** | Arm B head，train-calibrated |
| 右肾 | **91.73** | Arm B head，train-calibrated |
| 左肾 | **92.38** | Arm B head，train-calibrated |
| 胆囊 | **53.74** | Arm B reconstruction Direct-post |
| 食管 | **69.48** | Arm B reconstruction Direct-post |
| 胰腺 | **70.35** | Arm B reconstruction Direct-post |
| 肝脏 | **94.58** | Arm B reconstruction Direct-post |
| 胃 | **82.65** | Arm B head，fixed 0.5 |

这张表用于定位每类上限，不是一个可直接部署的“拼接模型”。若要把逐器官选择变成
正式方法，必须先在验证集确定每类输出源，再对测试集做一次冻结评估。

## 未完成的同协议 head 对照

| Arm | 唯一主要变量 | 账号 / 集群 | Job | 2026-08-28 状态 | 八类结果 |
|---|---|---|---:|---|---|
| A | AHER canvas + linear head | bolinren19 / SIP | 2413564 | PENDING (Resources) | 尚无 |
| B | AHER canvas + multiscale head | antengcai23 / XEC | 121398 | COMPLETED | 已列入排名 |
| C | final ViT pixels + 20 tokens 先汇总成 1 query | bolinren19 / SIP | 2415372 | RUNNING | 尚无 |
| D | final ViT pixels + 20 个独立 queries + log-mean-exp | bolinren19 / SIP | 2548701 / 2548702 | smoke 等待；formal dependency | 尚无 |

A/C/D 完成前不得写入估计值。最终 A/B/C/D 的主要 head 对比必须统一使用
checkpoint-802、fixed-0.5 head、相同测试 CSV 和相同后处理。

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
  [ORGAN_SLOTBANK_EXPERIMENT_SUMMARY_CN.md](ORGAN_SLOTBANK_EXPERIMENT_SUMMARY_CN.md)。

Arm B 两个正式 JSON 均加载 epoch 802 checkpoint，`exact=true`；reconstruction
结果标记完整测试集，head 结果标记 `complete_split=true`，样本数均为 6990。

## 更新规则

每次新结果只在同时满足以下条件时进入主排名：

1. checkpoint 精确加载且 epoch、head 类型与配置匹配；
2. 24 病例 / 6990 切片完整评估；
3. 阈值和后处理在测试前固定；
4. 记录账号、集群、Job ID 和真实结果路径；
5. 重新计算八类均值、小器官均值和排名，不手工挑选有利器官。
````

<a id="source-23"></a>

## docs/ORGSLOT_WORD_RESOLUTION_ABLATION_LOG.md

SHA256: `48a917d087a8489287780e97f8b901f03d0f614af5599c9c9ff39c132b0d8d9e`

````text
# OrganSlot WORD Resolution Ablation Log

Last updated: 2026-08-12 18:43 CST

## Objective

Test whether the large zero/black field in the 1.0×1.0×2.0 mm, 448×448
pipeline wastes model capacity. Keep the OrganSlot architecture, Loss3, optimizer,
effective batch, update budget, split, threshold, and evaluation modes fixed.

## Code provenance

- Branch: `experiment/orgslot-resolution-ablation-v0`
- Commit: `4fd6260`
- Base local Loss3/ROI line: `749cbec`
- Merged latest remote OrganSlot fixed-reference support through `fdca982`
- XEC account/worktree:
  `sifansong:/gpfs/work/aac/sifansong/worktrees/orgslot_resolution`
- XEC environment:
  `/gpfs/work/aac/sifansong/envs/abdpet`
- XEC OpenCV was changed from GUI `opencv-python 4.6.0.66` to
  `opencv-python-headless 4.6.0.66` because login/compute nodes do not provide
  `libGL.so.1`.

## Full-dataset geometry audit

The audit used all 35,576 2D slices from the fixed WORD training and test
manifests at 1.0×1.0×2.0 mm.

| Center crop | Median black pixels | Foreground-positive slices lost | Foreground pixel retention |
|---:|---:|---:|---:|
| 448 | 74.55% | 0 | approximately 100% |
| 384 | 66.28% | 0 | approximately 100% |
| 336 | 57.49% | 0 | approximately 100% |

The focus-organ maximum 2D boxes at 1 mm were:

| Class | Organ | Maximum box |
|---:|---|---:|
| 4 | gallbladder | 77×75 |
| 5 | esophagus | 44×46 |
| 6 | pancreas | 164×95 |

No focus-organ slice exceeded ROI224. Therefore 336 was selected over 384:
it removes more uninformative border without losing a single organ-positive
slice in this fixed dataset.

Machine-readable audits:

- `docs/WORD_CROP_GEOMETRY_ANALYSIS.json`
- `docs/WORD_TRAIN_CROP_GEOMETRY_ANALYSIS.json`

## Controlled arms

All arms use OrganSlot, `linear_sqrt` fusion, nine slots, ROI sampling
probability 0.2, Loss3 positive weight 0.25, alpha 0.5, maximum class-weight
ratio 4, segmentation head loss 0, effective batch 192, and 118,800 optimizer
updates.

| Arm | Spacing (mm) | Global/input | ROI | Purpose |
|---|---:|---:|---:|---|
| Current control | 1.0×1.0×2.0 | 448 | 384 | Existing running reference |
| A | 1.0×1.0×2.0 | 336 | 224 | Remove black border by runtime crop |
| B | 0.7×0.7×2.0 | 448 | 384 | Keep 448 while increasing in-plane sampling density |

Per user decision, Arm B retains ROI384 exactly like the current experiment.
This makes B a direct spacing ablation, although the physical ROI field becomes
268.8 mm instead of 384 mm.

Loss3 training positive-slice counts for the fixed split remain:

`4392 4875 5070 1631 3827 3753 7327 5556`

The 336 audit confirmed no positive slice is lost, so Arm A uses the same
counts. Arm B counts will be read from the completed 0.7 mm training audit and
must match the configuration before its smoke is submitted.

## Validation status

- 51 OrganSlot unit tests passed after merging the latest architecture.
- Three reconstruction-evaluator regression tests passed.
- 336 model forward/backward passed (21×21 ViT patch grid).
- 336 global + ROI224 paired transform passed.
- 448 global + ROI384 at 0.7 mm proxy geometry passed.
- Python compilation, Bash syntax, and `git diff --check` passed.
- XEC repeated all 51 OrganSlot tests successfully.
- Both launchers leave Slurm-provided `CUDA_VISIBLE_DEVICES` unchanged and
  fail immediately unless the exact allocated GPU count is visible.

## Jobs and acceptance gates

### Arm A: 1 mm / 336 / ROI224

| Stage | Account/cluster | Job | State at submission | Acceptance |
|---|---|---:|---|---|
| Smoke | sifansong/XEC | 115127 | PENDING (Priority) | 2 GPUs visible; 4 updates; finite losses; ROI and positive supervision exercised; checkpoint exists |
| Formal train | sifansong/XEC | 115128 | dependency | Smoke succeeds; 118,800 updates; no NaN/OOM |
| Evaluation | sifansong/XEC | 115129 | dependency | Exact checkpoint load; all 24 cases; all 8 classes and 4 modes; 100% complete |

### Arm B: 0.7 mm / 448 / ROI384

The 0.7×0.7×2.0 mm native-matrix dataset is being generated from the original
120 WORD volumes using the fixed seed-42 case split. Submit the same
smoke→train→evaluation chain only after preprocessing, ROI audit, crop audit,
checksums, and XEC transfer complete.

## Interpretation rule

Compare each completed arm to the existing 1 mm/448/ROI384 OrganSlot run using
the same primary metric (`direct_post`, fixed threshold 0.02), especially the
gallbladder, esophagus, and pancreas Dice. Do not treat training L2/LPIPS values
across different input resolutions as segmentation improvement by themselves.
````

<a id="source-24"></a>

## docs/PIXEL_PE_ABLATION.md

SHA256: `378ad9e6ce8b8a87ac32d3adb1d3e80423de1e5e7402c0f85e7a7997d0fe05bc`

````text
# Arm D pixel PE controlled ablation

## Restart after provenance failure — 2026-09-08

Supersedes the earlier job matrix. All jobs below: antengcai23/XEC,
account sifansong, QoS 8gpus; four A800/24 CPU/192 GB per training,
one A800/10 CPU/128 GB per evaluation. Training queues independently;
each evaluation has afterok on its own training. No smoke job submitted.

| Mode | Training | Evaluation |
|---|---|---|
| none, slice-wise baseline | 133848 | 133849 |
| spatial | 133850 | 133851 |
| temporal | 133852 | 133853 |
| spatiotemporal | 133854 | 133855 |

Frozen code: /gpfs/work/aac/antengcai23/worktrees/orgslot_3d_restart_01f1666.
Commit 01f1666; archive SHA256 1ae368a12e0495937a1c65d5f4729c8a414fddeab0241794fd0be9f75730893b.
Git metadata is optional with a 15-second timeout; actual source hashes
and dataset checksums remain recorded. Full real non-Git provenance writing,
three provenance tests, three slice-loss checks and target data identity passed.
Submission exports conda lib plus shared Mesa lib in LD_LIBRARY_PATH.
All eight new jobs verified pending (training Priority, evaluation Dependency).
Old 132071/132072/132097/132099/132101 cancelled; failed runs and files retained.

## Scheduling update — 2026-09-08 (supersedes original dependencies below)

Released baseline 132071 from user hold. Removed the baseline-evaluation
dependency from PE training 132096, 132098 and 132100: all four now queue
independently. Each evaluation retains afterok on its own training job.
Changed these eight antengcai23/XEC jobs to QoS 8gpus using scontrol:
132071/132072, 132096/132097, 132098/132099, 132100/132101.
Training still requests four A800 GPUs, not eight; no model/data/loss change.
The submitted script may still say 4gpus: scontrol's current job record is
authoritative for this scheduling override. New submissions must select QoS
explicitly after checking permissions, wall-time and live capacity.

At this audit SIP and XEC A800 nodes had no unallocated GPUs. bolinren19/SIP
retains Arm E and 2D PE; sifansong/XEC retains MAE encoder-only;
antengcai23/XEC retains MAE encoder-decoder plus these 3D jobs.
No cross-account data/code migration or duplicate training was submitted.
Eight-GPU per-user QoS permits up to two four-GPU jobs within that QoS,
subject to other limits and physical availability; it does not reserve GPUs.

2D: existing no-PE Arm D 82.12% (historical reported score; match its exact
threshold/postprocess protocol before comparing) versus spatial PE, same seed 0,
WORD 0.7/0.7/2, ROI20, original small-organ loss, batch192, 118800 updates.

3D: slice-wise-loss Arm D 132071 without PE versus spatial-only, temporal-only,
and spatial+temporal. All share slice-wise loss, seed0, batch48 slabs, 4 frames,
118800 updates, BF16, clipping1 and LR7.5e-5. Do NOT use old slab-loss 32.01%
as the isolated PE control. Three PE jobs depend on baseline completion.

CLI --pixel_pe defaults to none. Added after projection/reshape, before the
existing pixel decoder. Spatial PE [1,D,H,W] broadcasts over time, initialized
from OWT 2D sin/cos; temporal [1,D,T,1,1] broadcasts over H,W, normal std0.02.
Both learnable, as in OWT 3D decoder. This is local slab time, not absolute CT z.
OWT encoder already has PE; this tests reintroducing explicit position in the
pixel branch. It does not add a Transformer or change convolutions/collector.

Checkpoints store pixel_pe in args; 2D and 3D evaluators instantiate the matching
mode and strictly load state. Legacy checkpoints default to none. Temporal PE
initialization preserves the random stream used by existing parameters.

Compare fixed-threshold and original calibrated protocol separately. Report
eight organs, small-organ mean, precision/recall, volume ratio, and per-z counts.
Spatial-only vs none isolates space; temporal-only vs none isolates time;
both vs each single mode tests combination. An improvement does not by itself
prove PE absence was the sole cause of collapse.

## Submitted matrix (2026-09-06)

Code snapshot ab7742b. 23 unit/regression tests passed; full tiny 3D PE
forward/backward passed. Allocated-GPU execution remains pending.

| Arm | Account/cluster | Train | Eval | Dependency |
|---|---|---|---|---|
| 2D spatial | bolinren19/SIP | 2910849 | 2910855 | none |
| 3D none, slice loss | antengcai23/XEC | 132071 | 132072 | held for diagnostics review |
| 3D spatial | antengcai23/XEC | 132096 | 132097 | afterok:132072 |
| 3D temporal | antengcai23/XEC | 132098 | 132099 | afterok:132072 |
| 3D spatial+temporal | antengcai23/XEC | 132100 | 132101 | afterok:132072 |

All training requests are 4 A800 / 24 CPU / 192 GB, four-day limit.
No historical job was cancelled or modified. No new accuracy results yet.
````

<a id="source-25"></a>

## docs/PIXEL_PE_NUMERICS_20260908.md

SHA256: `df4f2f855d9c7aaa2d9463916b6ea0797eb232494cba619caf8cb236befdbdd2`

````text
# 2D spatial PE restart: 2026-09-08

Confirmed: SIP job 2910849 used FP16 (both launch banner and argv), not BF16.
The PE sbatch omitted AMP_DTYPE and inherited the common launcher's fp16 default.
At epoch 64 step 233, gradient norm was nonfinite; strict clipping aborted
before optimizer.step. This also prevents GradScaler from reaching its usual
skip/backoff path. Loss clipping cannot repair an already nonfinite gradient.
The first offending operator is NOT known: old logs did not retain per-parameter
gradient diagnostics or the failing state. FP16 overflow is a hypothesis, not
a proven operator-level diagnosis.

Fix: explicitly select BF16 (no FP16 scaler), retain clipping 1.0 and strict
failure, check parameter finiteness every update, and report bad gradient
parameter names plus scaler state on any future backward/update failure.
Save every 25 epochs. Use a two-update, full-data, same-batch DDP preflight
inside the training allocation; only start formal training if validation passes.
Restart seed 0 from scratch in a new output directory (only checkpoint-0 survived).
Do not alter PE, loss, sampling, LR, or optimizer update budget.

This restart changes numeric precision relative to the historical FP16 no-PE
control; a strictly isolated PE claim ultimately needs a matched BF16 no-PE
control. A passed preflight is NOT proof of stability past epoch 64.
Keep failed output intact; replace only its permanently blocked evaluation job.

Submitted: SIP bolinren19 / account angelosstefanidis / QoS 8a800.
Training 2919868 (4 A800, 24 CPU, 192 GB); unified evaluation 2919869
(1 A800, 10 CPU, 128 GB), afterok:2919868. Old dead evaluation 2910855
cancelled; old training logs/checkpoint preserved. Five unittest checks passed
(PE shape/gradients/initialization plus launch-precision guards), bash syntax
and git diff checks passed. GPU preflight remains pending with training.
````

<a id="source-26"></a>

## docs/THREED_SLICEWISE_DIAGNOSTIC.md

SHA256: `cc0746c609fb8d51ac766488706ca95591b8adee65013503a9ffd8f4e1b03a18`

````text
# 3D controlled diagnosis, 2026-09-06

Parent: ab3bf1f (Arm D, 128095). Existing C/D jobs are unchanged.

Confirmed: direct-head case Dice is B 24.32%, D 32.01%; full test coverage,
exact checkpoint-199 load. Strong false positives are measured. These figures
do not evaluate reconstruction. The original small-organ objective classifies
an entire [T,H,W] slab as positive or empty.

Not confirmed: whether z-smearing is the dominant error; whether overlap fusion,
missing temporal identity, temporal convolutions, or representation cause it.

Diagnostics: same checkpoint and deterministic crop; raw logits, checkpoint
calibrated logits (not test-fitted thresholds), and reconstruction via the
existing per-slot canvas -> fusion -> spatial/temporal PE -> decoder. Fixed
thresholds: head 0.5, reconstruction 0.02 (legacy). Report mean probability,
mean logit and center-only fusion, before/after identical 3D postprocessing.
per_slice.csv includes GT and predicted counts for every organ and z position.
These are cropped-grid case metrics, not native full-FOV claims.

Loss-only baseline: Arm D from the original random seed 0, no checkpoint resume.
Only 5D small-organ inputs are flattened B,T into supervision samples. Average
positive planes and empty planes separately, then add; empty weight remains 0.1.
2D and other loss types unchanged. Positive/negative diagnostic counts now count
slices for this loss. All optimizer, data, ROI20, update budget, BF16, head and
architecture settings match the parent. 118800 updates, LR 7.5e-5, batch 48 slabs.

Tests: mixed slab 1 positive + 3 empty, exact separate means, finite backward,
equivalence to the old 2D formula, and excluded-slab zero gradient.

No temporal PE, convolution, collector, or reconstruction architecture change.
Implementation snapshot: 678f97e. Local tests: 3 new loss checks and 20 existing
loss/TGR checks passed. Tiny 3D reconstruction via collected canvases is equal
to the existing selected-slot reconstruction forward, with finite output.

Submitted jobs (XEC): B head pilot 131959 -> full head/reconstruction 132069;
D head pilot 131960 -> full head/reconstruction 132070. Pilot cases=2, full=24.
Arm D loss-only training 132071 is USER HOLD: release only after diagnostics
are reviewed, preserving the requested no-retrain-first order. It is registered,
not training. No claim of loss correction improving Dice has been established.
Its same-protocol formal head evaluation is 132072, afterok:132071.
At 2026-09-06 21:30 CST, both pilots await QOSMaxGRESPerUser (MAE occupies
each account GPU quota). Full diagnostics await pilot success. No new
checkpoint diagnostic result is available yet.

Only consider temporal PE, then 1x3x3 conv as separate later experiments after
the loss-only result and evaluator controls are reviewed.
````

<a id="source-27"></a>

## README.md

SHA256: `d879604e2dfd440dddebd68baa5188f5cd13ac057ba7cc80dd5ae49e2bf46c6f`

````text
# OrganSlot：统一开发入口

本地与GitHub统一使用 `feature/orgslot`。已整合2D/3D Arm A/B/C/D/E、
AutoPET MAE迁移、3D诊断、逐slice损失和Pixel PE。旧实验使用归档tag追溯。

新增实验：[Arm E + Cross-Attention与热图诊断](docs/ORGSLOT_ARM_E_CROSS_ATTENTION_CN.md)。
使用`--query_refinement cross_attn`；默认none保留原E。只改query refinement，
输出仍为P4（112×112）；新增实验BF16与历史E的FP16存在数值配置差异，
严格结论需同精度对照。热图显示读取位置，不自动构成因果或定位正确的证明。

## 配置与部署必读（2026-09-08）

**统一代码不等于所有任务自动使用同一配置。** 新实验必须遵循
[架构、数值精度与提交配置统一规范](docs/ORGSLOT_CONFIGURATION_CONTRACT_CN.md)。
该文档列出A–E架构参数、loss/采样配置、覆盖顺序、账号路径及训练—评估审核清单。

| 必须统一的层级 | 核验重点 |
|---|---|
| 模型选择 | dimension、slot_head_type、pixel_pe、token数量 |
| 训练语义 | segmentation_unit、loss、采样、MAE初始化或resume |
| 数值与优化 | AMP dtype、梯度裁剪、finite检查、LR、有效batch、更新预算 |
| 实际部署 | 运行worktree/commit/diff、Python环境、数据及权重绝对路径 |
| 评估 | checkpoint配置、严格加载、病例完整性、阈值/后处理、新afterok依赖 |

**已确认的风险：** 通用shell/Python入口仍默认FP16。新A800任务必须显式设置
`AMP_DTYPE=bf16 CLIP_GRAD=1.0 FINITE_CHECK_INTERVAL=1`，不能仅凭模型支持BF16
就认为启用了它。最终以运行目录的`resolved_config.json`、`command.txt`和日志为准。
这是新任务规范，不代表全部历史脚本已自动整改。

2D PE旧任务2910849遗漏精度设置，在epoch64出现非有限梯度。
[修复记录](docs/PIXEL_PE_NUMERICS_20260908.md)记录了配置证据、验证范围及
新训练2919868→评估2919869；修复已同步实际PE目录和统一入口，
尚需GPU预检查及跨epoch64验证。其他旧worktree不会自动获得修复。

新开发以`.worktrees/orgslot_integrated`为入口；运行使用可追溯快照。
本次配置文档更新不修改运行任务，不替代历史实验的原始配置。

| 配置 | 参数 |
|---|---|
| 2D / 3D | `--dimension 2D/3D` |
| Arm A/B/C/D/E | `--slot_head_type linear/multiscale_conv/query_dot/multi_query_dot/arm_e_multiscale_query` |
| 无PE / Spatial / Temporal / 两者 | `--pixel_pe none/spatial/temporal/spatiotemporal`，仅C/D；2D不支持Temporal |
| 旧3D监督 / 新逐slice监督 | `--segmentation_unit slab/slice`；默认slab保持旧实验语义，2D不受影响 |
| MAE迁移 | `--mae_init_checkpoint`、`--mae_init_scope` |

Arm E目前仅支持2D。各实验现有Slurm脚本记录原账号路径；整合不会自动迁移
正在运行或排队的任务。新实验应从统一分支建立代码快照并显式设置上述参数。

详见 [PE排表](docs/PIXEL_PE_ABLATION.md)、
[3D诊断](docs/THREED_SLICEWISE_DIAGNOSTIC.md)。以下保留Arm E历史说明。

## Arm E历史说明（2026-09-03快照，非实时任务状态）

本分支 `experiment/orgslot-querymask-multiscale-pixel` 建立在统一的 Arm C/D
参数化实现之上，并加入 Arm E。C、D、E 共用同一模型入口，不复制 OrganSlot
语义与 reconstruction 路径。Arm E 在 Arm C 单 query 路径上增加来自输入图像的
P4/P8 空间特征：

```text
image -> Conv stem -> P4/P8 --+
final ViT z -> P16 -----------+-> top-down pixel decoder -> P(x)
20 TGEnc tokens -> mean pool -> organ query q_s
mask_s(x) = sqrt(D) * cosine(P(x), q_s)
```

reconstruction 路径保持 `OrganCollector -> TGEnc -> AHER -> canvas -> fusion` 不变。
空间支路不能独立输出 mask，最终 mask 始终由 organ query 点积控制。详细设计、
验证证据和任务记录见
[docs/ORGSLOT_WORD070_ARM_E_MULTISCALE_PIXEL.md](docs/ORGSLOT_WORD070_ARM_E_MULTISCALE_PIXEL.md)。

## 统一模型配置

| Arm | `--slot_head_type` | 空间特征 | query 聚合 |
|---|---|---|---|
| C | `query_dot` | ViT P16，28 x 28 | 20 tokens 均值为 1 query |
| D | `multi_query_dot` | ViT P16，28 x 28 | 保留 20 queries，log-mean-exp 汇总 |
| E | `arm_e_multiscale_query` | Conv P4/P8 + ViT P16，输出 112 x 112 | 与 C 相同的单 query |

## 锁定的可比配置

spacing 0.7 x 0.7 x 2.0、input/ROI 448/384、ROI probability 0.2、token
factor 20、retained supervision、small-organ loss、segmentation weight 0.01、
effective batch 192、118800 updates、AdamW、seed 0 均与 Arm A/B 相同。

## MAE 迁移与 recon 最新补充（2026-09-08）

Arm B MAE encoder / encoder+decoder 的 recon 已完整核验，分别80.57% / 81.42%。
Arm E MAE 三臂配置、八类表及任务编号见 [专项记录](docs/ARM_E_MAE_AND_RECON_20260908_CN.md)。
下面部分状态是历史记录，最新三臂提交状态以专项记录为准。

## 状态

| Arm | 账号 / 集群 | Job | 状态 | 正式结果 |
|---|---|---:|---|---:|
| C query-dot | bolinren19 / SIP | 2415372 | COMPLETED | mean Dice 82.01% |
| D multi-query dot | bolinren19 / SIP | 2548702 | COMPLETED | mean Dice 82.12% |
| E multiscale query-dot 单卡 smoke | bolinren19 / SIP | 2814754 | COMPLETED | 通过 |
| E multiscale query-dot 双卡 smoke | bolinren19 / SIP | 2864374 | PENDING | 尚无 |

Arm E 第一版固定沿用 Arm C 的 20 token 平均单 query；Arm D 已说明增加 query
数量不是主要瓶颈，因此本实验只检验高分辨率空间信息。

完成后必须使用和 Arm A/B 相同的 24 病例、6990 切片、post-processing 协议；
head 主结果使用固定 0.5 阈值，训练集校准阈值只能作为次要结果。

当前已完成基线、Arm B 八类结果及项目内 SOTA 见
[docs/ORGSLOT_WORD_COMMON8_SOTA_CN.md](docs/ORGSLOT_WORD_COMMON8_SOTA_CN.md)。
更详细的结构说明见
[docs/ORGSLOT_WORD070_QUERYMASK_EXPERIMENT.md](docs/ORGSLOT_WORD070_QUERYMASK_EXPERIMENT.md)。

## 结果路径

C/D 的历史结果与 E 的新结果必须分别写入运行账号对应的 worktree
`Results/OrganSlotBank/...`。不得混用 `bolinren19 / SIP`、`sifansong / XEC`
或其他账号的数据绝对路径。运行日志和 checkpoint 不纳入 Git。
# Arm F experimental branch

2D/3D organ-conditioned two-stage attention is available through
`--slot_head_type arm_f_attention`; `arm_f_linear` is the matched linear-readout
control. See [configuration, architecture and validation](docs/ARM_F_2D_3D.md).
No formal Arm F training or benchmark results are available yet.
````

<a id="source-28"></a>

## README_EXPERIMENTS_CN.md

SHA256: `89e73c06dd6cdc598e491267d377e393baf96f4f337550bc152a6ca7cf691f64`

````text
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
````
