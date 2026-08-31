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

现有 scratch Arm B 是直接对照。新增实验只改变初始化。

## AutoPET MAE

AutoPET XEC 训练清单包含78400张224×224 RGB JPEG，三个通道是相同灰度复制。第一版保持原224图像，不放大到448；WORD位置编码按448重新生成，因此不迁移位置编码。

MAE配置：patch16、LA encoder 6层/768维、decoder 8层/768维、mask ratio 0.75、masked-patch pixel MSE、BF16、clip-grad 1.0、effective batch 192、100000 updates、5000 warmup updates。按患者从AutoPET training划分90%/10% train/validation，不使用AutoPET test和器官标签。

## 权重映射

加载：`patch_embed`、`cls_token`、6个encoder blocks、`encoder_norm`、共享`decoder_blocks`、`decoder_norm`和`decoder_pred`。

不加载：MAE位置编码、`mask_token`、`decoder_embed`、optimizer；WORD所有slot的Collector、TGEnc、AHER、segmentation head和校准参数保持随机初始化。加载器对每个映射张量做名称和shape验证，并保存`mae_initial_checkpoint_load.json`。

## 结果判定

主要比较scratch Arm B与AutoPET-MAE Encoder+Decoder Arm B的同协议3D病例级结果：八类平均Dice、小器官平均Dice、逐器官Dice、precision/recall、NSD、HD95、预测/GT体积比以及固定/验证集校准阈值。若主实验有效，再补encoder-only消融以分离decoder贡献。

单seed只能作为探索性证据。若平均Dice提高至少0.5点且小器官平均提高至少1点、同时大器官下降不超过0.5点，再补两个seed。

## 执行链

1. `autopet_mae_transfer_smoke_sifan_xec.sbatch`：真实数据、完整模型做2次更新并验证checkpoint；
2. `autopet_mae_transfer_pretrain_sifan_xec.sbatch`：仅在smoke成功后生成固定路径`checkpoint-final.pth`；
3. `orgslot_word07072_armb_autopet_mae_sifan_xec.sbatch`：依赖MAE成功完成后启动WORD训练；
4. WORD checkpoint-802完成后使用与scratch Arm B完全相同的统一重建/head评估脚本。

任务应运行在`sifansong/XEC`：该环境同时拥有AutoPET和WORD 0.7数据。不得把sifansong、antengcai23或SIP/XEC的绝对路径互换。
