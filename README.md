# OrganSlot WORD Common8：Arm C / D 参数化 Query-Mask 基线

更新时间：2026-08-28

本分支 `experiment/orgslot-querymask` 已统一 Arm C 与 Arm D。两种结构共享
`SharedPixelQueryDecoder2D`，只通过 `--slot_head_type` 选择 query 聚合方式：

```text
final ViT z -> shared pixel decoder -> P(x)
Arm C (`query_dot`): 20 TGEnc tokens -> mean pool -> one query
Arm D (`multi_query_dot`): 20 TGEnc tokens -> 20 queries -> log-mean-exp
mask_s(x) = query-conditioned cosine similarity with P(x)
```

reconstruction 路径保持 `OrganCollector -> TGEnc -> AHER -> canvas -> fusion` 不变。
因此 Arm C 只改变 segmentation readout，直接检验 AHER canvas 是否是分割空间信息的
瓶颈。

## 锁定的可比配置

spacing 0.7 x 0.7 x 2.0、input/ROI 448/384、ROI probability 0.2、token
factor 20、retained supervision、small-organ loss、segmentation weight 0.01、
effective batch 192、118800 updates、AdamW、seed 0 均与 Arm A/B 相同。

## 状态

| Arm | 账号 / 集群 | Job | 状态 | 正式结果 |
|---|---|---:|---|---:|
| C pooled query-dot | bolinren19 / SIP | 2415372 | COMPLETED | 82.01% |
| D multi-query dot | bolinren19 / SIP | 2548702 | COMPLETED | 82.12% |

Arm C 与 D 的 small-organ mean 分别为 65.87% 和 65.88%。D 没有形成实质提升，
说明“单 query 聚合”不是当前主要瓶颈，但两种行为都保留用于可复现消融。

完成后必须使用和 Arm A/B 相同的 24 病例、6990 切片、post-processing 协议；
head 主结果使用固定 0.5 阈值，训练集校准阈值只能作为次要结果。

当前已完成基线、Arm B 八类结果及项目内 SOTA 见
[docs/ORGSLOT_WORD_COMMON8_SOTA_CN.md](docs/ORGSLOT_WORD_COMMON8_SOTA_CN.md)。
更详细的结构说明见
[docs/ORGSLOT_WORD070_QUERYMASK_EXPERIMENT.md](docs/ORGSLOT_WORD070_QUERYMASK_EXPERIMENT.md)。

## 结果路径

Arm C/D 在 `bolinren19 / SIP` 运行，结果只应写入运行账号对应 worktree 的
`Results/OrganSlotBank/...`。不得混用不同账号或集群的数据绝对路径。运行日志不纳入 Git。
