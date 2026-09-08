# OrganSlot 架构、数值精度与提交配置统一规范

更新时间：2026-09-08。适用分支：`feature/orgslot`。
本文件是新实验的配置与审核规范，不表示所有历史脚本已经自动整改。

## 1. 唯一开发入口与运行快照

- 统一开发目录：`OD_OWT/.worktrees/orgslot_integrated`。
- Arm A/B/C/D/E、2D/3D、PE、MAE迁移通过参数选择，不再为每种模型复制实现。
- 历史目录可能处于 detached HEAD；归档tag用于追溯，不能视作实时同步副本。
- 新任务应使用经过核验的独立代码快照，并记录绝对路径、commit与未提交diff。
- 若修复必须应用到旧实验目录，必须分别核验统一入口和实际运行目录；只合并Git分支不算部署完成。
- 正在运行的任务不要原地替换代码。需要变更时建立新快照、新输出目录及新任务。

当前例外：重提的2D PE任务 `2919868` 仍明确使用
`.worktrees/orgslot_pixel_pe`，相关修复也已同步到统一开发目录。
这不意味着其他旧目录已经同步。未提交diff也属于实际实验版本，不能只记录commit。

## 2. 四层配置必须同时核验

| 层级 | 负责内容 | 实际证据 |
|---|---|---|
| 模型 | head、query、PE、维度、token数量 | 模型构建参数、checkpoint args及state dict |
| 优化与数值 | loss、AMP、裁剪、LR、batch、更新预算、初始化/续训 | resolved_config.json、训练日志 |
| 启动与部署 | worktree、环境、数据路径、脚本环境变量、最终CLI | Slurm脚本、command.txt、启动日志 |
| 调度与评估 | 登录用户、Slurm account、QoS、资源、依赖、评估协议 | scontrol、评估结果与完整性标记 |

**模型支持BF16 ≠ 启动任务选择了BF16；分支包含修复 ≠ 运行目录包含修复。**

参数不是简单的“命令行永远覆盖一切”，实际顺序为：

`sbatch导出环境 → 作业脚本export → shell启动器构造CLI → Python解析 → 模型/优化器`

- 作业脚本 `export AMP_DTYPE=bf16` 会覆盖继承环境中的同名变量。
- 启动器 `AMP_DTYPE=${AMP_DTYPE:-fp16}` 在变量未设置或为空时使用FP16。
- 启动器显式传入 `--amp_dtype` 后，Python默认值不再决定该次运行精度。
- 当前通用启动器和Python入口的AMP默认值**仍是FP16**，没有全局改成BF16。
- 当前 `PIXEL_PE` 默认none、`SEGMENTATION_UNIT` 默认slab，是历史兼容行为，不代表新实验推荐值。
- Shell变量只有被启动器转换成CLI参数才会生效；不能凭变量名称推断生效。

审查入口：
[通用启动器](../scripts/orgslot/run_word_112_448_a100.sh)、
[Python入口](../main_pretrain_orgslot_common8_a100.py)、
[训练循环](../engine_pretrain_orgslot_common8_a100.py)。

## 3. 架构参数统一表

| 控制项 | 环境变量 / CLI | 含义与边界 |
|---|---|---|
| 维度 | DIMENSION / --dimension | 2D或3D；3D还需核验FIX_FRAME、TEMP_STRIDE |
| Arm A | SLOT_HEAD_TYPE=linear | AHER canvas后线性分割头 |
| Arm B | SLOT_HEAD_TYPE=multiscale_conv | AHER canvas后多尺度卷积分割头 |
| Arm C | SLOT_HEAD_TYPE=query_dot | 共享pixel decoder；organ tokens汇总为一个query |
| Arm D | SLOT_HEAD_TYPE=multi_query_dot | 共享pixel decoder；保留多个query，log-mean-exp汇总 |
| Arm E | SLOT_HEAD_TYPE=arm_e_multiscale_query | 输入图像P4/P8与ViT P16融合；单query；目前仅2D |
| Pixel PE | PIXEL_PE / --pixel_pe | none、spatial、temporal、spatiotemporal；当前仅C/D |
| 监督单位 | SEGMENTATION_UNIT / --segmentation_unit | slab保留历史3D语义；slice按切片判断阳性/空切片；2D不受此开关影响 |
| MAE初始化 | MAE_INIT_CHECKPOINT、MAE_INIT_SCOPE | encoder或encoder_decoder；必须记录来源checkpoint |
| 训练续跑 | RESUME_CHECKPOINT / --resume | 与MAE初始化互斥；核验epoch、optimizer和更新计数连续性 |

2D只允许none/spatial，不存在独立的切片时间轴。3D可以分别比较四种PE。
PE添加在pixel projection之后、pixel decoder之前；不改OrganCollector的时间压缩，
也不等同于给direct head接入完整OWT reconstruction Transformer decoder。

`token_factor=20` 与 `ROI probability=0.2` 是两个不同参数。
“ROI20”指20%的ROI采样概率，不表示20个token。

## 4. 可比实验必须锁定的项目

2D WORD head对照使用：spacing 0.7×0.7×2.0、输入448、ROI384、
ROI probability 0.2、token factor20、seed0、有效batch192、
118800次optimizer更新。3D不能照搬2D有效batch，应另行记录slab数量、
每slab切片数及更新预算。

分割损失：`small_organ`；retained监督；lambda_seg=0.01；
lambda_bg_seg=0；Tversky FP/FN=0.3/0.7、eps=1e-6；
Balanced Focal权重0.5、alpha=0.75、gamma=2；
hard negatives=top2%、negative slice weight=0.1。
不能把“背景像素负样本”误认为独立background slot监督，两者不同。

总训练目标也不是只有Dice+Focal：当前路线仍有L2 reconstruction及LPIPS，
并加权叠加segmentation loss。核验 `loss_version=L2-LPIPS`、
`lambda_lpips=1`、`fusion_mode=linear_sqrt`、reference count=9，
以及旧Loss3相关权重是否为0。

比较表中必须增加以下列：
**AMP dtype、clip_grad、LR、有效batch、更新次数、监督单位、初始化来源**。
只保持head和loss相同不足以保证公平对照。
本次BF16 PE结果与历史FP16无PE结果只能作带精度差异的比较；
严格PE消融还需同精度无PE对照，不能把差异全部解释成PE收益。

## 5. 新A800任务的数值规范

以下是新任务必须显式配置的要求，不是所有旧任务的当前状态：

```bash
export AMP_DTYPE=bf16
export CLIP_GRAD=1.0
export FINITE_CHECK_INTERVAL=1
```

还需显式设置DIMENSION、SLOT_HEAD_TYPE、PIXEL_PE、SEGMENTATION_UNIT、
loss和采样参数。不得从其他实验的shell环境隐式继承MAE/RESUME配置。

| 保护措施 | 能做什么 | 不能保证什么 |
|---|---|---|
| BF16 autocast | 比FP16更宽的指数范围；当前入口不启用FP16 GradScaler | 不能防止除零、非法运算或所有反向不稳定 |
| 梯度裁剪1.0 | 在梯度有限时限制全局范数 | 不能将已产生的NaN/Inf修复为有效梯度 |
| strict finite checks | 异常时停止，避免污染optimizer/参数 | 不等于训练能够自动恢复 |
| FP32局部fallback | 处理特定算子兼容性或精度问题 | 仅对已覆盖的算子/路径有效，不代表全模型FP32 |
| checkpoint | 提供可追溯恢复点 | 文件存在不等于参数有限、配置正确或可成功续训 |

当前scaler流程：backward → unscale → strict clip → step → scaler update。
FP16下如果strict clip先抛错，就到不了GradScaler正常的skip/backoff。
不要通过关闭 `error_if_nonfinite`、静默跳过异常或nan_to_num掩盖问题。

## 6. 2026-09-08事故记录与修复范围

任务2910849在epoch64 step233出现非有限梯度。启动日志和完整CLI均显示
`amp_dtype=fp16`；原因是PE作业脚本没有显式设置AMP_DTYPE。
首个异常算子未被旧日志记录，**FP16溢出仍是机制假设，不是已复现的算子级根因**。

已实施：
- 2D PE脚本固定BF16、clip1.0、finite interval1，并检查GPU支持BF16。
- backward/update错误报告包含异常梯度参数名称和scaler状态。
- checkpoint从每100改为每25个epoch。
- 同一分配内先进行两次更新DDP预检查，单独输出，通过后从头正式训练。
- CPU侧5项回归测试通过；实际GPU预检查和跨epoch64验证仍需看新任务日志。
- 新训练2919868；评估2919869依赖afterok:2919868；旧失效评估2910855已取消。

详见[数值修复记录](PIXEL_PE_NUMERICS_20260908.md)。
两次更新只能验证启动链路，不能证明长期稳定，不能提前标记“NaN彻底解决”。

## 7. 三账号、两集群：路径与额度分离

每个任务必须记录：
登录用户、集群、Slurm account、QoS、partition、GPU型号/数量、
环境Python绝对路径、worktree、训练/评估CSV、ROI索引、LPIPS权重、
预训练/续训checkpoint、输出目录。

- bolinren19/SIP已用数据根为 `/gpfs/work/aac/bolinren19/2026-07/DATA`。
- sifansong与antengcai23的XEC工作目录分别在各自用户根下；数据路径须现场读取其配置，
  不能通过替换用户名猜测存在性。
- 登录用户名不等于Slurm account；QoS名也不是GPU类型。
- SIP与XEC Job ID属于不同命名空间；不能用SIP sacct核验XEC任务。
- 训练与评估必须各自核验环境和路径，不能只验证训练目录。
- README不保存密码、私钥或token；访问配置使用本地HPC skill。

## 8. 提交前及运行后审核清单

### 提交前

1. 确认统一分支修复与实际运行快照一致，记录commit和dirty diff。
2. 阅读作业脚本及其调用的启动器，检查显式参数与默认回退。
3. 检查真实CSV、ROI索引、权重、Python环境及独立输出目录；不得覆盖旧结果。
4. 核对每卡batch×GPU数×accum_iter；3D同时记录slab单位。
5. 运行配置/PE回归测试及bash语法检查：
   `python -m unittest discover -s tests -p 'test_pixel_pe*.py' -v`。
6. 先分配内验证再正式训练；不要截断ROI池制造无效smoke。
7. sbatch返回Job ID后，通过scontrol确认account、QoS、资源、WorkDir及依赖。

### 启动后

核验同一输出目录中的 `resolved_config.json`、`command.txt`、启动日志：
AMP、head、PE、监督单位、spacing、loss、batch、初始化必须与实验表一致。
不以README、脚本名称或环境变量的“预期值”替代真实解析值。
检查首个epoch有限性、ROI路径、weighted_seg≈0.01×seg_loss；
checkpoint保存后检查args、epoch、可加载性及参数有限性。
本次PE额外要求稳定跨过epoch64，之后检查checkpoint75/100。

### 评估及依赖

- sbatch保存提交时的batch脚本；修改本地脚本不会改写已提交副本。
  但脚本调用的外部Python/shell通常在运行时读取，所以代码快照也必须冻结。
- 旧afterok上游失败后不能自动改为依赖新任务。重新建立训练—评估链并记录新ID。
- evaluator从checkpoint args重建head/PE，严格加载权重；核验八类、病例数和完整性。
- 固定0.5与校准阈值结果分别列出；历史训练集校准协议必须明确标注，
  新阈值选择优先用验证集，不能在测试集挑最佳阈值。
- 3D必须报告病例体积级指标，注明slab重叠融合、后处理及head/reconstruction路径。

## 9. 尚未自动化的部分

当前并没有全仓库唯一的强制配置schema，也没有检查所有sbatch的通用验证器。
现有新测试只防止2D PE精度入口回退；其他实验仍必须逐任务审核。
本规范不修改通用FP16默认、不改变正在运行的实验、不自动重提任务。
后续若实现全局强制校验，应另行测试历史兼容性，不能仅凭文档声称已经完成。

