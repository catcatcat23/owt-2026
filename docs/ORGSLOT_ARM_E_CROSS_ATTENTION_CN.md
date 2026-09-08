# Arm E + Cross-Attention（2026-09-08）

## 实验目的与范围
在Arm E上增加一个共享的query-to-image cross-attention block，测试query在
最终mask匹配前再次读取空间特征是否有帮助。不是替换AHER、不是新增P2、
不是SAM two-way decoder；不增加query数量，不新增绝对位置PE。

统一开关：`--slot_head_type arm_e_multiscale_query --query_refinement cross_attn`。
环境变量为QUERY_REFINEMENT。默认none完全保留旧E路径与state dict；
仅2D Arm E支持本开关。checkpoint args用于统一head/reconstruction evaluator
恢复结构并严格加载，不允许把原E checkpoint伪装成已训练attention checkpoint。

## 实际结构
CT的P4/P8与ViT P16融合 → P4 pixel feature（448输入时112×112）。
原20个TGEnc tokens经LN、均值、投影和器官身份向量产生一个query。
P4平均池化2倍 → 56×56 memory（注意：不是直接使用P8 lateral feature）。
4-head cross-attention，通道128、无dropout：
q + Attention(LN(q), LN(memory)) → 残差FFN（128→256→128）→ refined q。
refined q与原P4 feature做归一化点积 → 112 logits → 插值回448。
attention计算局部FP32，整体训练BF16。attention模块初始化使用fork_rng，
避免改变后续原有organ slot参数初始化；新模块本身不是零初始化。

长宽112/448=1/4，面积1/16；P4每位置仍有128通道，不能等同“信息只剩1/16”。
边界可能受P4上生成logit再插值的限制。真实P2 skip（224）应作为独立实验，
不能与本次attention一起加。224同通道特征元素数是112的4倍。

## 锁定配置及可比性
WORD spacing0.7/0.7/2，448/384，ROI0.2，small-organ loss，
retained，lambda_seg0.01，lambda_bg_seg0，原Tversky/Focal/top2%/negative0.1，
batch192，seed0，118800更新、warmup5940、blr1e-4、AdamW weight decay0.05。
本组BF16、clip1、finite每次更新、每25epoch保存。先同分配2次真实DDP更新验证，
再从seed0正式训练，不从预检查checkpoint续跑。

重要：历史E任务2892114使用旧FP16/scaler流程，因此与本组并非严格单变量比较。
严格attention结论需同一代码、BF16/clip等设置的query_refinement=none对照；
目前只新增用户要求的一组，不擅自多提交一组baseline。

## 热图与交互检查
工具：`python -m tools.export_query_cross_attention`。
严格加载训练后checkpoint，默认固定选择测试CSV中每个目标器官最先出现的
两张阳性切片（4胆囊、5食管、6胰腺），不按预测质量挑选。
每例导出PNG/PDF、NPZ和manifest：
- CT、GT、head均值attention / uniform（1代表均匀水平）；
- normal / uniform-attention / bypass整个refinement的概率图；
- NPZ保存4个head的原始56×56归一化权重，供逐head审查；
- GT attention mass、GT面积占比、二者之比；
- 三种预测的固定0.5 slice Dice和相对normal的平均概率变化；
- checkpoint及CSV哈希、epoch、dataset index与严格加载报告。

uniform保留value、输出投影、FFN，只取消位置选择；bypass连同FFN一起跳过，
两者含义不同。所有干预保持pixel features和原query不变。
注意力图不是分割概率，也不是因果证明；关注器官外上下文不必然错误。
本模块没有新增绝对坐标PE。同步重排keys/values可以不改变聚合结果，因此
空间热图不能证明“理解了解剖绝对坐标”。干预结果也可能是分布外响应，
要和完整病例级指标一起判断，不能靠几个样例宣称提升。

## 运行与评估
训练脚本：slurm/orgslot/train/orgslot_word07072_arm_e_crossattn_bolin_sip.sbatch。
评估脚本：slurm/orgslot/eval/orgslot_word07072_arm_e_crossattn_bolin_sip.sbatch。
运行使用独立detached快照，通过EXPERIMENT_WORKDIR/EVAL_WORKDIR传入，
固定代码后不再原地改动。评估以afterok跟随训练，先导出固定样例热图，
再运行原24病例统一评估。最终无真实训练checkpoint前，不提供“已证明有效”图。

## 验证
已增加：共享参数初始化一致性、严格state dict roundtrip、
attention逐head空间归一化、normal/uniform/bypass干预差异、
query对memory依赖、部分器官head参与的整模型finite backward。
渲染测试仅用合成未训练输入，并明确标注非实验证据。
GPU预检查通过与否必须读取任务日志，CPU测试不能代替长程稳定性审核。

