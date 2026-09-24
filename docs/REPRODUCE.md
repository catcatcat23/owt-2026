# OWT baseline 复现入口（main）

本分支保留 OWT 模型及本项目全类别联合训练基线，不含 OrganSlot head。
本次发布原本未跟踪的 Slurm 脚本和配套工具；保留历史参数，不重新训练或宣称已复现论文全部数字。

## 1. 获取代码与环境

```bash
git clone --branch main https://github.com/catcatcat23/owt-2026.git
cd owt-2026
git rev-parse HEAD
```

参考 `image_CL_mae2.yml`、`requirements.txt` 配置环境。它们是历史环境记录，包含平台相关依赖，不保证在新机器直接安装成功。训练依赖 PyTorch、torchvision、timm、einops、LPIPS 相关实现等；预处理额外使用 nibabel、NumPy、Pillow，评估使用 OpenCV、SciPy。先检查依赖导入与 CUDA 可用性，再正式启动。
模型见 `OWT_models.py`、`OrganEmbed.py`；训练入口 `main_pretrain.py` 和 `engine_pretrain.py`。
LPIPS 相关权重路径见 `VQ/lpips.py`，运行前检查所需权重存在。不要把这里的 OWT 重建训练误称为普通无标签 MAE。

## 2. 数据与协议

数据集需自行按授权获取，仓库不提供患者数据或训练 checkpoint。
WORD/BTCV Common8 预处理入口：`tools/preprocess_owt_common8.py`。
使用 `python tools/preprocess_owt_common8.py --help` 查看参数；批处理模板为 `tools/run_preprocess_owt_common8_local.sh` 或 `slurm/preprocess_owt_common8.sbatch`。

历史 Common8 配置：spacing 1×1×3、HU [-175,250]、canvas 512、输出224、seed42、Fixfr4（4帧）。类别映射、划分和 CSV 生成以预处理代码及输出元数据为准。CSV 内的图像/标签路径也必须是接收方机器上的有效路径。
**这不是 OrganSlot 的 0.7×0.7×2、448/384 ROI20 协议，不能直接混入其排行榜。**

## 3. 选择训练与评估脚本

| 数据/模式 | 训练 | 评估 |
|---|---|---|
| WORD Common8 2D | `slurm/pretrain_owt_common8_full_word.sbatch` | `slurm/evaluate_owt_common8_word.sbatch` |
| BTCV Common8 2D | `slurm/pretrain_owt_common8_full_btcv.sbatch` | `slurm/evaluate_owt_common8_btcv.sbatch` |
| WORD Common8 3D Fixfr4 | `slurm/pretrain_owt_common8_3d_word.sbatch` | `slurm/evaluate_owt_common8_3d_word.sbatch` |
| BTCV Common8 3D Fixfr4 | `slurm/pretrain_owt_common8_3d_btcv.sbatch` | `slurm/evaluate_owt_common8_3d_btcv.sbatch` |
| AbdAutoPET 2D | `slurm/pretrain_owt_2d_abdpet.sbatch` | `slurm/evaluate_owt_abdautopet_2d.sbatch` |

`pretrain_owt_2d_word.sbatch`、`pretrain_owt_2d_btcv.sbatch` 是另一些历史配置，不要替代上面的 Common8 full 配方。`smoke_*` 只用于小规模检查，不代表正式训练。

## 4. 提交前必须修改

- 脚本中的 `WORKDIR`、`CONDA_ENV`、CSV/数据根目录、输出目录、评估 checkpoint 路径。
- `#SBATCH` 的 account、partition、QoS、GPU、时限与日志路径；这些是原 SIP 账号配置，不适用于所有机器。
- **提交前创建日志父目录**，因为 Slurm 打开日志早于脚本中的 mkdir。
- 修改 GPU 数或 batch 会改变有效 batch / 学习率含义，不属于严格等配置复现。保存实际命令、环境、commit 与数据划分。

完成上述本机适配后，例如运行 WORD Common8 2D：

```bash
sbatch slurm/pretrain_owt_common8_full_word.sbatch
# 等训练成功结束，核实 checkpoint-1199.pth 可加载，并修改评估脚本中的路径，再提交：
sbatch slurm/evaluate_owt_common8_word.sbatch
```

没有 Slurm 时，在已配置 GPU 环境中按脚本执行其 Python/DDP 命令；不要直接照搬集群绝对路径。脚本保留历史 torch.distributed.launch 入口。

## 5. 评估解释与验证范围

2D 评估实现 `tools/evaluate_owt_common8.py`；3D 为 `tools/evaluate_owt_common8_3d.py`。这条基线使用重建读出，不是 OrganSlot direct segmentation head。阈值、连通域与 spacing 参数均应随结果报告；距离指标必须使用与实际预处理一致的 spacing。
历史结果及限制见 [RESULTS.md](RESULTS.md)，不要将 joint 全类别结果解释为增量抗遗忘结果。

本次发布仅检查脚本 Bash 语法与 Python 语法；没有在全新环境重跑 GPU 训练/评估。公开脚本保留历史路径以便追溯，不意味着开箱即跑。
