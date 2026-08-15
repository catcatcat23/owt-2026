# OrganSlotBank v0 Experiment Log

- Branch: `feature/orgslotbank-v0`
- Starting commit: `6cdf9b575edd6845b7518c18d7ef9be9b89508f7`
- Scope: strict visibility, model/loss/checkpoint tests, 2D/3D forward-backward,
  and tiny overfit only. No full training submission.

## 2026-07-24 Gate-A implementation

Worktree: `/gpfs/work/aac/bolinren19/OD_OWT_orgslot`

Implemented:

- stage/class config and a training wrapper that never returns raw/full labels;
- provisional background built only from base-visible old masks;
- explicit fixed-K `OrganSlot` paths, appendable `OrganSlotBank`, binary heads,
  calibration, normalized additive per-sample canvas fusion, and shared decoder;
- per-sample Slot-Aware TGR, base reconstruction/segmentation loss, minimal
  Liver-only incremental loss, detached suppression ablation primitive;
- exact checkpoint save/load, original OWT migration, visible-slot transfer,
  append-from-background, freeze/hash audits, and head-based metrics;
- isolated synthetic smoke entry point. Formal datasets and long training were
  deliberately not started.

Validation commands:

```text
/gpfs/work/aac/bolinren19/.conda/envs/abdpet/bin/python -m unittest discover -s tests -p 'test_orgslot_*.py' -v
OMP_NUM_THREADS=4 /gpfs/work/aac/bolinren19/.conda/envs/abdpet/bin/python -m tools.run_orgslot_tiny_overfit --steps 250 --output-dir Results/OrganSlotBank/_tiny_overfit/gate_a_cpu --device cpu
sbatch slurm/orgslot/smoke/orgslot_gate_a_cpu.sbatch
```

Observed results before final checkpoint-resume test addition:

- 24/24 CPU tests passed, including 2D and Fixfr4-TS1 3D forward/backward and
  an original OWT forward regression. The final count is recorded below.
- tiny overfit: loss `1.7362165 -> 0.1302058` (ratio `0.0749940`), organ Dice
  `0.9354208`, exact checkpoint reload `true`;
- the same audit appended Liver from background, ran incremental backward with
  `visible_masks={liver}` only, and kept all frozen parameter hashes unchanged;
- final current-code Slurm smoke job `1561641`: `COMPLETED`, exit `0:0`, elapsed
  `00:00:27`; base total loss `1.6614242 -> 1.4783453`, reconstruction MSE
  `0.1312498 -> 0.1153881`, with finite head metrics and saved checkpoint.

Boundary/status:

- Gate A wiring is exercised on synthetic data only.
- Gate B/C, real-data manifests, Sequential-FT/Head-only comparisons,
  suppression/background-plasticity ablations, and full training are not run.
- Original `OWT_models.py`, `OrganEmbed.py`, `engine_pretrain.py`, and
  `main_pretrain.py` are unchanged; the regression test confirms legacy OWT
  forward remains runnable.

Final test result after adding base/incremental resume coverage: **25/25 passed**.

Resolved debug events:

- The first tiny-overfit target used non-patch-aligned, sample-varying regions.
  On the intentionally tiny 2x2 canvas it reached loss ratio `0.7845` and Dice
  `0`, so it failed the memorization gate. The test was corrected to the
  canonical repeated, patch-aligned single-pattern target; the success
  threshold was not relaxed, and the corrected run reached ratio `0.0749940`.
- The first Slurm submission was rejected before job creation because the
  account default QOS was invalid. The script was updated to the existing
  project association `account=sifansong`, `partition/qos=cpudebug`; current
  code then completed as job `1561641` with exit `0:0`.

## 2026-07-26 real-data wiring and controlled-baseline update

Added without modifying the legacy OWT files:

- deterministic 2D/Fixfr4 manifest adapter and strict case/slice checks;
- deterministic case-level train/validation split tool that never reads label
  pixels;
- real/synthetic stage-aware runner with manifest checksums, case-overlap
  rejection, 2D/3D model selection, finite-step smoke mode, best/last
  checkpoints, and per-epoch frozen-parameter SHA-256 audits;
- explicit Ours, Sequential-FT, and Head-only trainable scopes;
- automatic detached frozen-old prediction path for the suppression ablation;
- identity/frozen incremental calibration by default, matching the handoff;
- two new behavior tests for baseline scopes and detached old-confidence
  supervision, plus two case-split tests.

Current validation:

- **32/32** OrganSlot CPU tests pass;
- complete synthetic Base checkpoint -> append Liver -> Ours/Sequential-FT/
  Head-only orchestration passes;
- Ours trains 11,969/127,279 debug parameters and preserves 138 frozen tensor
  hashes; Head-only trains 97/127,279 and preserves 157 frozen hashes;
- current-code tiny overfit again reaches loss `1.7362165 -> 0.1302057`, ratio
  `0.0749939`, Dice `0.9354208`, exact reload true, frozen hashes unchanged;
- real 2D `[B,3,224,224]` debug-width one-batch forward/backward passes with
  finite total loss `1.9208560`;
- real Fixfr4 `[B,3,4,224,224]` debug-width one-batch forward/backward passes
  with finite total loss `1.8933843`;
- seed-0 case split contains 630 train and 70 validation cases. 2D records are
  70,560/7,840 and Fixfr4 records are 68,670/7,630.

The real-data checks above are engineering smokes, not accuracy results. They
use debug-width models and only one training batch. The assumed raw-ID semantic
mapping in `owt_legacy_debug.json` still needs independent dataset evidence
before formal organ-named claims or long training.

## 2026-08-15 WORD448 JointSeg Focal v2

目的：在 ROI20 reconstruction baseline 上加入每个 slot 独立的二分类 Focal
监督，替代出现前景全背景塌缩风险的 Dice+BCE tiny 目标；不启用融合后 Loss3。

固定配置：

- 分支 `experiment/orgslot-jointseg-focal-v0`，提交 `7d8343e`；
- XEC账号 `antengcai23`，Slurm account `sifansong`；
- WORD 2D，448输入，ROI20使用384 crop resize到448；
- `L = global L2 + LPIPS + 0.1 * slot focal`；
- `focal_alpha=0.75`，`focal_gamma=2.0`；
- `seg_supervision=all`，`lambda_bg_seg=0.25`；
- 真实keep/drop继续控制reconstruction target和fusion；
- 118800 optimizer updates，有效batch 192，2张A800。

验证：本地53项OrganSlot回归测试和15项Focal/TGR目标测试通过；XEC同环境15项
目标测试通过。逐slot日志新增预测体积、GT体积和阳性像素概率，防止仅凭总loss
误判全背景输出为成功。

任务链：

- smoke `117005`；
- 正式训练 `117006`，依赖smoke成功；
- reconstruction Direct/Indirect评估 `117007`；
- head训练集阈值校准 `117008`；
- 固定0.5及冻结校准阈值head测试 `117009`。
