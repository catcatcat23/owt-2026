# OrganSlot：统一开发入口

本地与GitHub统一使用 `feature/orgslot`。已整合2D/3D Arm A/B/C/D/E、
AutoPET MAE迁移、3D诊断、逐slice损失和Pixel PE。旧实验使用归档tag追溯。

| 配置 | 参数 |
|---|---|
| 2D / 3D | `--dimension 2D/3D` |
| Arm A/B/C/D/E | `--slot_head_type linear/multiscale_conv/query_dot/multi_query_dot/arm_e_multiscale_query` |
| 无PE / Spatial / Temporal / 两者 | `--pixel_pe none/spatial/temporal/spatiotemporal`，仅C/D；2D不支持Temporal |
| 旧3D监督 / 新逐slice监督 | `--segmentation_unit slab/slice`；默认slab保持旧实验语义，2D不受影响 |
| MAE迁移 | `--mae_init_checkpoint`、`--mae_init_scope` |

Arm E目前仅支持2D。各实验现有Slurm脚本记录原账号路径；整合不会自动迁移
正在运行或排队的任务。新实验应从统一分支建立代码快照并显式设置上述参数。

详见 [PE排表](docs/PIXEL_PE_ABLATION.md)、
[3D诊断](docs/THREED_SLICEWISE_DIAGNOSTIC.md)。以下保留Arm E历史说明。

更新时间：2026-09-03

本分支 `experiment/orgslot-querymask-multiscale-pixel` 建立在统一的 Arm C/D
参数化实现之上，并加入 Arm E。C、D、E 共用同一模型入口，不复制 OrganSlot
语义与 reconstruction 路径。Arm E 在 Arm C 单 query 路径上增加来自输入图像的
P4/P8 空间特征：

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

## 统一模型配置

| Arm | `--slot_head_type` | 空间特征 | query 聚合 |
|---|---|---|---|
| C | `query_dot` | ViT P16，28 x 28 | 20 tokens 均值为 1 query |
| D | `multi_query_dot` | ViT P16，28 x 28 | 保留 20 queries，log-mean-exp 汇总 |
| E | `arm_e_multiscale_query` | Conv P4/P8 + ViT P16，输出 112 x 112 | 与 C 相同的单 query |

## 锁定的可比配置

spacing 0.7 x 0.7 x 2.0、input/ROI 448/384、ROI probability 0.2、token
factor 20、retained supervision、small-organ loss、segmentation weight 0.01、
effective batch 192、118800 updates、AdamW、seed 0 均与 Arm A/B 相同。

## 状态

| Arm | 账号 / 集群 | Job | 状态 | 正式结果 |
|---|---|---:|---|---:|
| C query-dot | bolinren19 / SIP | 2415372 | COMPLETED | mean Dice 82.01% |
| D multi-query dot | bolinren19 / SIP | 2548702 | COMPLETED | mean Dice 82.12% |
| E multiscale query-dot 单卡 smoke | bolinren19 / SIP | 2814754 | COMPLETED | 通过 |
| E multiscale query-dot 双卡 smoke | bolinren19 / SIP | 2864374 | PENDING | 尚无 |

Arm E 第一版固定沿用 Arm C 的 20 token 平均单 query；Arm D 已说明增加 query
数量不是主要瓶颈，因此本实验只检验高分辨率空间信息。

完成后必须使用和 Arm A/B 相同的 24 病例、6990 切片、post-processing 协议；
head 主结果使用固定 0.5 阈值，训练集校准阈值只能作为次要结果。

当前已完成基线、Arm B 八类结果及项目内 SOTA 见
[docs/ORGSLOT_WORD_COMMON8_SOTA_CN.md](docs/ORGSLOT_WORD_COMMON8_SOTA_CN.md)。
更详细的结构说明见
[docs/ORGSLOT_WORD070_QUERYMASK_EXPERIMENT.md](docs/ORGSLOT_WORD070_QUERYMASK_EXPERIMENT.md)。

## 结果路径

C/D 的历史结果与 E 的新结果必须分别写入运行账号对应的 worktree
`Results/OrganSlotBank/...`。不得混用 `bolinren19 / SIP`、`sifansong / XEC`
或其他账号的数据绝对路径。运行日志和 checkpoint 不纳入 Git。
