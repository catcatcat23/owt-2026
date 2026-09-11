# OrganSlot：研究主线与架构

## 研究主线

主要 idea 是器官专属权重的可插拔增量学习，以及解耦后重接的器官 tokens 如何辅助下游任务。
分割头 A–F 是检验下游读出与空间信息利用的实验，不是替代增量主线的六个独立贡献。
参数隔离、表示解耦、独立模块可组合是三个不同主张；联合 WORD 指标不能自动证明后两者。

## 共享与专属路径

共享 ViT -> z；每器官 Collector_s -> K tokens -> TGEnc_s -> AHER_s -> canvas_s。
选定 canvases 经 fusion -> 共享 reconstruction Transformer decoder -> 图像重建。
分割路径按 head 配置选择 canvas 或共享 pixel features，输出每器官 logit 与 calibration。
原始 OWT 的 joint token 路径与新的器官专属模块应明确区分，不能将继承组件称为新增贡献。
AHER 后重建路径的 spatial/temporal PE 和 Transformer 不等同于 direct head 的解码路径。

| Arm | head 配置 | 空间路径 / token 使用 |
|---|---|---|
| A | `linear` | AHER canvas -> 线性读出 |
| B | `multiscale_conv` | AHER canvas -> 多尺度卷积读出 |
| C | `query_dot` | ViT 空间分支；K tokens 汇总为单 query |
| D | `multi_query_dot` | 保留 K queries；点积 masks 经 log-mean-exp 汇总 |
| E | `arm_e_multiscale_query` | 输入 P4/P8 + ViT P16；单 query；当前 E 为 2D |
| E+CA | E + `query_refinement=cross_attn` | query 读取空间特征，再点积；不等于 F |
| F | `arm_f_attention` | K tokens 顺序读取 P16/P8/P4，再由 P4 反向读取 tokens |
| F 对照 | `arm_f_linear` | 相同多尺度 token refinement，改用线性点积读出 |

448 输入对应 P16=28、P8=56、P4=112；112 是边长的 1/4、面积的 1/16。
F 保留 20 个图像条件化 tokens；三层 cross-attention/self-attention/FFN 后反向读取，
P4 residual + FFN + 共享分类器生成 mask。解码器跨器官共享，不做 Hungarian matching。
F 3D stem 为逐 slice 2D 卷积，attention 读取整个 slab，不新增 temporal Conv3D。
因此 F 与旧 3D D 不只差一个反向 attention；应使用 F linear 匹配对照隔离该因素。
Attention 权重不是前景概率；热图不能单独证明语义解耦或因果作用。

## 增量学习的边界与待验证主张

原始 v0 包含 append_slot、可见标签隔离、冻结/hash 审计与 tiny 工程验证。
这些不替代正式多阶段增量实验。新增器官时必须列明共享 ViT、pixel decoder、
reconstruction decoder、旧 slots、calibration 的冻结/更新状态和旧数据访问权限。
仅冻结旧 slots 而更新共享路径，仍可能改变旧器官输出；独立 binary 输出不变也不保证
新增类别后 multiclass argmax 不受竞争影响。

论文最小验证：不同增量顺序的 Old/New/All 指标；独立模块直接重接 vs 微调；
普通分割头/固定类别 query vs 图像 organ tokens；有无重建监督；参数增长与旧数据需求。
这些是待验证问题，不在本次文档整理中标记为完成。

## 实现入口

- `OWT_models_orgslot.py`、`OrganSlotEmbed.py`：主模型与器官模块。
- `ArmFDecoder.py`：F 的多尺度双向解码与 linear 对照。
- `main_pretrain_orgslot_common8_a100.py`：训练配置入口。
- `engine_pretrain_orgslot_common8_a100.py`：训练、数值执行与 loss 集成。
- 各历史设计的公式、测试和路径在历史档案中按原文件名查询。
