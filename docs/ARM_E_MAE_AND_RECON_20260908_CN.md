# Arm B MAE recon 补齐与 Arm E MAE 三臂实验（2026-09-08）

## 已完成的 Arm B MAE：统一八类结果
指标：24 病例 / 6990 切片，checkpoint-802 exact=true，case Dice presence mean ×100。
Reconstruction 为 Direct-post，固定阈值 0.02，min_size=20、opening_radius=1。
两组 MAE 原始 JSON 已归档到 artifacts/mae_recon_audit_20260908/。
原始 Arm B 数字来自既有 ORGSLOT_WORD_COMMON8_SOTA_CN.md 的已验证记录。

| 初始化 / recon | 脾脏 | 右肾 | 左肾 | 胆囊 | 食管 | 胰腺 | 肝脏 | 胃 | 八类均值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Encoder+decoder | 93.19 | 92.12 | 92.21 | 57.05 | 69.83 | 71.33 | 94.70 | 80.95 | **81.42** |
| Encoder-only | 92.56 | 92.61 | 91.78 | 50.98 | 71.00 | 71.10 | 94.25 | 80.25 | **80.57** |
| Scratch Arm B | 92.74 | 91.48 | 91.92 | 53.74 | 69.48 | 70.35 | 94.58 | 79.28 | **80.45** |

| 初始化 | Head fixed-0.5 post | Head train-calibrated post | Recon Direct-post |
|---|---:|---:|---:|
| Scratch Arm B | 79.89 | 80.36 | 80.45 |
| Encoder-only | 80.60 | 80.97 | 80.57 |
| Encoder+decoder | 81.09 | 81.69 | 81.42 |

Head 数字沿用此前已核验汇报；本次新核验的是 recon JSON。
Encoder+decoder 相对 encoder-only 的 recon 提高 0.8552 个百分点。
相对 scratch recon 约 +0.97；胆囊提升最明显，但 encoder-only 胆囊低于 scratch。
不能声称所有器官同步获益，或凭单 seed 证明收益机制。
以上是 Arm B MAE 内部排名，不是全项目新 SOTA；Arm C/D 的既有约 82% 结果仍需同列比较。
Head/recon 各自单独排名，不在测试集逐器官挑输出拼接。

## 数据来源
- Encoder-only：sifansong / XEC，训练 128402，评估 128404。
  worktree: /gpfs/work/aac/sifansong/worktrees/orgslot_autopet_mae
  result: Results/OrganSlotBank/evaluation/Common8/WORD_2D/OrgSlot_WORD07072_ROI20_RetainedMultiConv_SmallOrgan_L001_AutoPETMAE_EncoderOnly_128402_ckpt802_reconstruction_fixedthr002/results.json
- Encoder+decoder：antengcai23 / XEC，训练 128401，评估 128403。
  worktree: /gpfs/work/aac/antengcai23/worktrees/orgslot_autopet_mae
  result: Results/OrganSlotBank/evaluation/Common8/WORD_2D/OrgSlot_WORD07072_ROI20_RetainedMultiConv_SmallOrgan_L001_AutoPETMAE_EncDec_128401_ckpt802_reconstruction_fixedthr002/results.json
- 两文件 complete_test_set=true，samples=6990，八类 case_count 均为24。

## Arm E MAE 三臂
用户明确要求在原始 Arm E 上也比较 scratch / encoder / encoder+decoder。
不加 cross-attention，不加 PE，不加载 pixel decoder、Collector/TGEnc/AHER 权重。

| 组 | 目的 | 账号/集群 | 训练 | 评估 | 状态 |
|---|---|---|---|---|---|
| E0 scratch | 多尺度空间分支无预训练对照 | bolinren19 / SIP | 2892114 | 2921054 | 训练 epoch498，评估 afterok 等待 |
| E1 encoder | 隔离 MAE encoder 收益 | sifansong / XEC | 133896 | 133898 | 训练 PENDING (Priority)，评估 afterok |
| E2 encoder+decoder | 检验共享重建 decoder 额外收益 | sifansong / XEC | 133897 | 133899 | 训练 PENDING (Priority)，评估 afterok |

E0 正式评估已提交：1×A800、10 CPU、128GB、angelosstefanidis/8a800；
afterok:2892114，依次 train calibration / full head / full recon。
E1/E2 每条4×A800、24CPU、192GB、sifansong/8gpus。
两条正式训练于2026-09-08直接提交，没有申请 smoke。

固定：WORD 0.7×0.7×2、input/ROI 448/384、ROI probability0.2、small-organ loss、
lambda_seg0.01、retained、有效 batch192=4×8×6、118800更新、
warmup5940、blr1e-4（实际lr7.5e-5）、seed0、FP16 GradScaler、无新增梯度裁剪。
MAE checkpoint SHA256:
0b3571a3e79095686a0b12649735a298aaeb8a0b3bb93a47f3484439ac4a1ec6

## 兼容快照与复现
本地运行快照：.worktrees/orgslot_arm_e_mae_20260908。
远端运行快照：
/gpfs/work/aac/sifansong/worktrees/orgslot_arm_e_mae_20260908
；上传归档为
/gpfs/work/aac/sifansong/arm_e_mae_20260908.tar
，SHA256 为
d57d709a1822f398450607555c40ba357cd0cbec427f377b240055e49f1d4a1d。
基于原 Arm E 提交81adba2，而非直接使用改变了数值策略的最新版训练引擎。
原始训练引擎和 OrganSlotEmbed.py 与 E0 运行目录 SHA256 完全一致。
仅补 MAE 参数/严格加载/加载记录、Git provenance 容错和源文件哈希。
加载器 util/mae_transfer.py 及数据核验器取自统一分支01f1666。
入口：
- slurm/orgslot/train/arm_e_mae_sifan_xec.sbatch
- slurm/orgslot/eval/arm_e_mae_unified.sbatch
- scripts/orgslot/arm_e_mae_81adba2.patch：旧 main/launcher 的完整兼容补丁。
在81adba2归档上应用补丁，再加入上述加载器和入口即可复现。
不要把训练入口指向任意后续 worktree，否则会失去 E0 可比性。

已完成 Python compile、bash -n 和旧新引擎哈希核验；没有新增 GPU smoke。
最终以三组完整checkpoint802同协议结果判断 Arm E MAE 是否超过既有项目内最好结果；
不能用当前epoch498训练loss推断最终SOTA。
