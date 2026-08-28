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

## 正式评估

训练完成后必须按病例重组完整 3D 预测，报告八类 case-level 3D Dice、IoU、
precision、recall、NSD 和 HD95（mm）。2D slice 指标只作为错误分析，不作为主结果。
