# 任务交接：检查入口与已知风险

队列查询快照：2026-09-11，Asia/Shanghai。只核对 squeue，未重新读取每个训练日志。
不在这里沿用旧 epoch 或把不在队列解释为完成；最新状态必须查询 Slurm/log/results。

| 实验目的 | 账号/集群 | 训练 → 评估 Job | 本次队列状态 / 下次核验 |
|---|---|---|---|
| E scratch 续训，MAE零预训练对照 | bolinren19/SIP | 2922770 → 2922772 | 训练 RUNNING，评估 PENDING；查ckpt、loss、head/recon |
| D 2D spatial PE | bolinren19/SIP | 2922771 → 2922773 | PENDING；查BF16插值修复与真实配置 |
| E cross-attention | bolinren19/SIP | 2923562 → 2923565 | PENDING；查梯度、显存、热图生成 |
| F 2D 双向交互 | bolinren19/SIP | 2925500 → 2925501 | PENDING；运行快照 arm_f_runtime_0ba5304 |
| E MAE encoder-only 从500恢复 | sifansong/XEC | 135283 → 135284 | 9月11日23:41核验：训练PENDING Priority；评估afterok:135283；旧134495已取消 |
| E MAE encdec 从500恢复 | sifansong/XEC | 134915 → 134916 | PENDING；确认501续跑与跨原541故障区间 |
| F 3D 双向交互 | sifansong/XEC | 134901 → 134902 | 训练RUNNING；查slab监督、速度、最高时限内可达预算 |
| D 3D slice-wise baseline | antengcai23/XEC | 133848 → 133849 | 训练RUNNING；先看precision/recall/volume ratio恢复情况 |
| D 3D spatial PE | antengcai23/XEC | 133850 → 133851 | 训练RUNNING；与同精度slice baseline比较 |
| D 3D temporal PE | antengcai23/XEC | 133852 → 133853 | PENDING |
| D 3D spatial+temporal PE | antengcai23/XEC | 133854 → 133855 | PENDING |

表内箭头是既有登记配对；本次未逐条重新核验 afterok，不能替代 scontrol。
资源、epoch、最新checkpoint、loss、八类结果均需下一次日志审计补齐，而非凭历史填数。
XEC D运行目录为该用户 worktrees/orgslot_3d_restart_01f1666；F为
worktrees/arm_f_runtime_0ba5304；E encdec为worktrees/arme_mae_runtime_43cad30。
SIP squeue WorkDir可能只是父目录，实际脚本引用的源码快照需另外核验。

## 下次按顺序检查

Encoder-only恢复提交：代码`f70b8319d10a719772e08ee85ec4ba6e7bd6b1cc`，不可变目录
`/gpfs/work/aac/sifansong/worktrees/arme_mae_encoder_runtime_f70b831`；入口
`slurm/orgslot/train/arm_e_mae_encoder_resume500.sbatch`。训练4×A800/24CPU/192GB，
评估1×A800/10CPU/128GB；均sifansong账户、8gpus QoS、5天。
版本门禁、checkpoint模型/优化器有限性、双进程CPU Gloo测试通过；GPU尚未启动。
评估包含checkpoint802的head固定/校准阈值和reconstruction；输出名
`ArmE_MAE_encoder_resume500_135283`。此记录为提交后文档更新，不修改已排队运行源码。

1. 134494已确认epoch541附近NCCL超时，134495依赖失效。checkpoint-500已CPU加载核验，模型/优化器有限，GradScaler完整。使用arm_e_mae_encoder_resume500.sbatch从501恢复；保持FP16、有效batch192、loss、采样和更新预算，复用固定schema指标归约与冻结LPIPS buffer免广播修复。新运行必须检查跨过541并保存550；根因尚未完全确认，不能只凭提交成功宣称修复验证完成。
2. 读取E/F/3D正在训练组的最新日志，核对epoch、checkpoint、finite与剩余时间。
3. 核验所有训练—评估依赖是否有效；失败链不会自己改成新训练。
4. 完成评估后同时收录head/recon、八类、完整性和阈值，更新RESULTS而非再开日期文档。
5. 区分confirmed与hypothesis：NCCL根因、token信息损失、PE解决z-smearing均不能提前定论。

## 本次文档整理的边界

只更新开发分支文档；不修改任务、运行快照、checkpoint、训练代码。
旧 orgslot_integrated 工作区存在未提交改动，原样保留，不能自动当作最新分支。
定位当前分支工作区请用 git worktree list，而不是依赖历史绝对路径。
# 2026-09-13：batch6替换排队3D任务

用户约定后续四卡2D每卡16×累积3=有效192，3D每卡6×累积2=有效48。
运行中2D快照不改。两组3D新运行代码均为2697b05642d527bfd137debc0f9ed175e9ae345f。

| 实验 | 账号/集群 | 新训练 → 评估 | 旧链 | 配置 |
|---|---|---|---|---|
| Arm F 3D topk，从checkpoint30续训 | sifansong/XEC | 137410 → 137411 | 136312/136313已取消 | 4A800/20CPU/192GB，4gpus，7天 |
| Arm F 3D mean背景，从头训练 | antengcai23/XEC | 137412 → 137413 | 136355/136356已取消 | 4A800/20CPU/192GB，8gpus，5天 |

各账号独立目录`/gpfs/work/aac/<用户名>/worktrees/arm_f_batch6_20260913`。
评估均1A800/6CPU/128GB，afterok指向对应新训练；队列已解除hold，提交后PENDING。
CPU配置检查与版本门禁通过；未新增GPU smoke，batch6显存/吞吐尚待正式启动验证。
已有checkpoint和日志保留；运行快照不随后续文档提交更新。
