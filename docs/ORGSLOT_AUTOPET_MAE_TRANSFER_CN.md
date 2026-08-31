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
2. `autopet_mae_transfer_pretrain_sifan_xec.sbatch`：仅在smoke成功后生成固定路径`checkpoint-final.pth`；
3. `orgslot_word07072_armb_autopet_mae_encoder_sifan_xec.sbatch`：B1 encoder-only；
4. `orgslot_word07072_armb_autopet_mae_sifan_xec.sbatch`：B2 encoder+decoder 的 sifansong/XEC 版本；
5. `orgslot_word07072_armb_autopet_mae_encoder_decoder_anteng_xec.sbatch`：B2 的 antengcai23/XEC 并行版本，必须先复制并校验同一 MAE checkpoint；
6. 两个新 checkpoint-802 完成后，将B0/B1/B2使用相同3D病例级重建/head评估脚本重评。

MAE预训练运行在`sifansong/XEC`：这是当前唯一确认同时拥有AutoPET和WORD 0.7的空闲账号环境。B1可在同账号依赖MAE启动；B2在MAE完成并复制同一checkpoint后可转到当前无任务的`antengcai23/XEC`并行。不得把sifansong、antengcai23或SIP/XEC的绝对路径互换，每次复制都要记录SHA256。
