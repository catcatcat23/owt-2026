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
