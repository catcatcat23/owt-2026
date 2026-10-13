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
