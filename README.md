# OrganSlot WORD Common8：Arm C Single-Query Mask Head

更新时间：2026-08-28

本分支 `experiment/orgslot-querymask` 测试 Arm C：分割路径绕过 AHER canvas，
让最终 ViT patch 特征提供空间信息，让器官 token 提供器官身份。

```text
final ViT z -> shared pixel decoder -> P(x)
20 TGEnc tokens -> mean pool -> organ query q_s
mask_s(x) = sqrt(D) * cosine(P(x), q_s)
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
| C pooled query-dot | bolinren19 / SIP | 2415372 | RUNNING | 尚无，禁止预填 |

当前实现把 20 个 TGEnc tokens 先平均为一个 query。它保留器官级语义，但可能把
token 间的部位分工再次压缩。Arm D 是只改变这一点的后续对照。

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
