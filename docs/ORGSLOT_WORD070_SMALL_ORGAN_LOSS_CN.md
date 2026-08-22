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
