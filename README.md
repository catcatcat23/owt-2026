# OrganSlot WORD Common8：Arm D Multi-Query Mask Head

更新时间：2026-08-28

本分支 `experiment/orgslot-querymask-multiquery` 测试 Arm D：保留 20 个 TGEnc
tokens 作为 20 个独立 query，而不是像 Arm C 一样先均值汇总为一个 query。

```text
final ViT z -> shared pixel decoder -> P(x)
TGEnc token k -> q_{s,k}
part_logit_{s,k}(x) = sqrt(D) * cosine(P(x), q_{s,k})
mask_logit_s(x) = logmeanexp_k(part_logit_{s,k}(x))
```

这样每个 token 可以响应器官的不同部位、形态或边界，再用无 token-count 偏置的
log-mean-exp 汇总。reconstruction 路径保持不变；第一版不加入 diversity loss，
确保相对 Arm C 只改变 query aggregation。

## 锁定的可比配置

spacing 0.7 x 0.7 x 2.0、input/ROI 448/384、ROI probability 0.2、token
factor 20、retained supervision、small-organ loss、segmentation weight 0.01、
effective batch 192、118800 updates、AdamW、seed 0 均与 Arm A/B/C 相同。

## 状态

| 阶段 | 账号 / 集群 | Job | 状态 | 结果 |
|---|---|---:|---|---:|
| smoke | bolinren19 / SIP | 2548701 | PENDING (Priority) | 尚无 |
| formal train | bolinren19 / SIP | 2548702 | PENDING (Dependency) | 等待 smoke 后启动 |

实现提交为 `606c7ed`，作业登记提交为 `730664d`。最终目标为 checkpoint-802。
完成后必须与 Arm C 使用同一评估协议；固定 0.5 head 阈值是主结果，训练集校准阈值
只能作为次要结果。还应检查 20 个 part maps 是否发生完全塌缩。

当前已完成基线、Arm B 八类结果及项目内 SOTA 见
[docs/ORGSLOT_WORD_COMMON8_SOTA_CN.md](docs/ORGSLOT_WORD_COMMON8_SOTA_CN.md)。
完整设计见
[docs/ORGSLOT_WORD070_MULTIQUERY_EXPERIMENT.md](docs/ORGSLOT_WORD070_MULTIQUERY_EXPERIMENT.md)。

## 结果路径

Arm D 在 `bolinren19 / SIP` 运行，结果只应写入本 worktree 的
`Results/OrganSlotBank/...`。不得套用其他账号或 XEC 的绝对路径。运行日志不纳入 Git。
