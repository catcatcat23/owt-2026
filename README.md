# OrganSlot WORD Common8：Arm E 多尺度 Pixel Decoder

更新时间：2026-09-03

本分支 `experiment/orgslot-querymask-multiscale-pixel` 在 Arm C 单 query
语义路径上加入真正来自输入图像的 P4/P8 空间特征：

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

## 锁定的可比配置

spacing 0.7 x 0.7 x 2.0、input/ROI 448/384、ROI probability 0.2、token
factor 20、retained supervision、small-organ loss、segmentation weight 0.01、
effective batch 192、118800 updates、AdamW、seed 0 均与 Arm A/B 相同。

## 状态

| Arm | 账号 / 集群 | Job | 状态 | 正式结果 |
|---|---|---:|---|---:|
| E multiscale query-dot 单卡 smoke | bolinren19 / SIP | 2814754 | COMPLETED | 通过 |
| E multiscale query-dot 双卡 smoke | bolinren19 / SIP | 2864218 | PENDING | 尚无 |

Arm E 第一版固定沿用 Arm C 的 20 token 平均单 query；Arm D 已说明增加 query
数量不是主要瓶颈，因此本实验只检验高分辨率空间信息。

完成后必须使用和 Arm A/B 相同的 24 病例、6990 切片、post-processing 协议；
head 主结果使用固定 0.5 阈值，训练集校准阈值只能作为次要结果。

当前已完成基线、Arm B 八类结果及项目内 SOTA 见
[docs/ORGSLOT_WORD_COMMON8_SOTA_CN.md](docs/ORGSLOT_WORD_COMMON8_SOTA_CN.md)。
更详细的结构说明见
[docs/ORGSLOT_WORD070_QUERYMASK_EXPERIMENT.md](docs/ORGSLOT_WORD070_QUERYMASK_EXPERIMENT.md)。

## 结果路径

Arm C 在 `bolinren19 / SIP` 运行，结果只应写入本 worktree 的
`Results/OrganSlotBank/...`。不得套用 `antengcai23 / XEC` 的数据或结果路径。
运行日志不纳入 Git。
