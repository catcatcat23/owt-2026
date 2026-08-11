# PSEM-v3 Triplet + LossBalance-v3 Experiment

## Status

- Branch: `experiment/psem-v3-triplet-loss3-word-v0`
- Base: `experiment/psem-v2-lossbalance-v3-word-v0` at `199f6d1`
- Datasets: WORD Common8 2D and AbdAutoPET 2D, both at 224
- Experiment scripts commit: `ced65cc`
- Stage: recovery chains submitted on 2026-08-10
- Both full training jobs are gated by their dataset-specific smoke validators.

## Original WORD chain

| Job | Stage | Final state | Cause |
|---:|---|---|---|
| 1651604 | 2-GPU smoke | FAILED | Node exposed no CUDA GPU |
| 1651606 | 1200-epoch WORD 2D training | CANCELLED | DependencyNeverSatisfied |
| 1651607 | common WORD 2D evaluation | CANCELLED | DependencyNeverSatisfied |

This chain was superseded by the recovery submission below.

## Recovery chains submitted 2026-08-10

The original WORD chain failed before model training because job 1651604 saw
no CUDA GPU. Its dependent jobs 1651606 and 1651607 were cancelled; this was
not a model, memory, or Loss3 failure.

| Dataset | Job | Stage | Resources | Dependency | State at submission |
|---|---:|---|---|---|---|
| WORD | 1655585 | smoke | 2 GPU, `sifansong/4a800` | none | PENDING (Priority) |
| WORD | 1655586 | 1200-epoch training | 2 GPU, `sifansong/4a800` | afterok:1655585 | PENDING |
| WORD | 1655587 | evaluation | 1 GPU, `angelosstefanidis/8a800` | afterok:1655586 | PENDING |
| AbdAutoPET | 1655588 | smoke | 4 GPU, `angelosstefanidis/8a800` | none | PENDING (Priority) |
| AbdAutoPET | 1655589 | 1200-epoch training | 4 GPU, `angelosstefanidis/8a800` | afterok:1655588 | PENDING |
| AbdAutoPET | 1655590 | evaluation | 1 GPU, `angelosstefanidis/8a800` | afterok:1655589 | PENDING |

Every stage has a strict CUDA device-count preflight. Each smoke validator is
the final command in its job, so `afterok` release requires a valid
`checkpoint-0.pth`, finite required metrics, complete anchor coverage, and
both present and absent anchor cases.

The AbdAutoPET Loss3 frequency counts computed from all 78,400 training slices
are `[59698, 43715, 34418, 30768]`. Its four-GPU source batch is
`12 x 4 x 4 = 192`, exactly matching the completed PSEM-v2 run's
`96 x 2 = 192`. LR scaling and optimizer batch semantics therefore remain
controlled while the triplet workload finishes within the QOS wall-time.

The PSEM-v2 AbdAutoPET comparator is already complete: smoke 1629559, training
1629561, and evaluation 1629564 all completed successfully. Its checkpoint and
full evaluation output are present and will not be recomputed.

## Motivation

PSEM-v2 mixes seven manually weighted query modes in a 20-slot cycle. Loss3
optimizes present-and-kept organs, while Whole and Leave-one-out are trained in
unpaired samples. Low individual reconstruction errors therefore do not ensure
that `Whole - Without(c)` isolates organ `c`, which is required by Indirect
inference.

## Query construction

For each source sample, choose one foreground anchor `c` from only `epoch` and
`sample_index`; GT presence never selects the anchor. Construct three queries:

1. `Direct = {c}`
2. `Context = S`, where `c` is excluded
3. `Plus = S union {c}`

Half of sample/epoch pairs use `S = all classes except c`, exactly matching the
Whole/Without inference pair. The other half use a deterministic non-empty
subset of the complete class bank to retain arbitrary-composition training.
The targets satisfy exactly:

```text
Target(Plus) - Target(Context) = Target(Direct)
```

When `c` is absent, the Direct target and target difference are both zero.
When `c` is present, both equal the CT content inside the organ label.

## Objective

The three branches are concatenated along the batch dimension and passed
through the unchanged PSEM-v2 + Loss3 model. Existing Loss3 terms are averaged
over the resulting `3B` query batch:

```text
L_branch = average Loss3(Direct, Context, Plus)
```

The only new optimized term applies the Loss3 global-L2 plus positive-ROI
structure to the query increment:

```text
Delta = Prediction(Plus) - Prediction(Context)
L_delta = L2(Delta, Target(Direct))
          + 0.25 * PositiveROI(Delta, Target(Direct))
L_total = L_branch + LPIPS + 0.1 * L_delta
```

No extra LPIPS is applied to the difference image. No separate negative or
remove objective is added in v0: the Direct all-zero target and zero target
difference already provide negative-query and negative-addition supervision.

## Fairness and compute

- Architecture, trainable parameters, data, augmentation, LR rule, epochs,
  Loss3 class weights, and evaluator are unchanged.
- Each source image creates three query forwards. The v0 implementation repeats
  the full forward for correctness and isolation; it does not yet reuse the ViT
  token bank.
- Per-GPU source batch is 8 with accumulation 4 on two GPUs. The effective
  source-image batch is 64, equal to the PSEM-v2 + Loss3 baseline (`32 x 2`).
- Query compute is approximately three times larger and must be reported.

## Verification completed

- Deterministic schedule uses no GT labels.
- Anchors cover every foreground class.
- Direct contains exactly one class.
- Context is non-empty and excludes the anchor.
- Plus equals Context plus exactly the anchor.
- Present and absent target identities are exact.
- Difference loss is zero for exact predictions and backpropagates finite
  gradients after perturbation.
- Small 2D model forward/backward passes.
- One-sample triplet intentionally overfits by more than 20% in 20 steps.
- All 14 original PSEM tests and existing PSEM-v2 + Loss3 tests still pass.

## Files

- `util/triplet_query_schedule.py`
- `util/triplet_query_loss.py`
- `engine_pretrain_psem_triplet_loss3.py`
- `main_pretrain_psem_triplet_loss3.py`
- `tests/test_psem_triplet_loss3.py`
- `tools/validate_psem_v3_triplet_smoke.py`
- `slurm/psem_v3_triplet/smoke_word_2d.sbatch`
- `slurm/psem_v3_triplet/pretrain_word_2d.sbatch`
- `slurm/psem_v3_triplet/evaluate_word_2d.sbatch`
- `slurm/psem_v3_triplet/smoke_abdautopet_2d.sbatch`
- `slurm/psem_v3_triplet/pretrain_abdautopet_2d.sbatch`
- `slurm/psem_v3_triplet/evaluate_abdautopet_2d.sbatch`

## Progress audit entry points

Snapshot time: 2026-08-10 CST. Both smoke jobs are `PENDING (Priority)`;
training and evaluation jobs are `PENDING (Dependency)`. No PSEM-v3
performance result exists yet.

### Queue and accounting

```bash
squeue -j 1655585,1655586,1655587,1655588,1655589,1655590 \
  -o '%.18i %.28j %.10T %.18a %.10q %.10M %.30R'
sacct -j 1655585,1655586,1655587,1655588,1655589,1655590 -X \
  --format=JobID,JobName,State,ExitCode,Elapsed,Start,End,NodeList
```

### Smoke evidence

- WORD Slurm stdout/stderr:
  `slurm/logs/PSEM_v3/WORD_2D_Smoke/psemv3_triplet_smoke_1655585.{out,err}`
- WORD smoke run:
  `Results/PSEM_v3_Triplet_Loss3/_smoke/WORD_2D/PSEMv3_TripletContext_Loss3_Delta01_WORD_2D_C8_T20_GPU2_B8A4_E1_N256`
- AbdAutoPET Slurm stdout/stderr:
  `slurm/logs/PSEM_v3/AbdAutoPET_2D_Smoke/psemv3_auto_smoke_1655588.{out,err}`
- AbdAutoPET smoke run:
  `Results/PSEM_v3_Triplet_Loss3/_smoke/AbdAutoPET_2D/PSEMv3_TripletContext_Loss3_Delta01_AbdAutoPET_2D_C4_T20_GPU4_B12A4_E1_N512`

A smoke passes only when the Slurm state is `COMPLETED`, `checkpoint-0.pth`
and `log.txt` exist, the corresponding validator prints
`PSEM-v3 triplet smoke validation passed`, and all required losses and
negative energies are finite. A failed CUDA preflight is a cluster/device
failure, not evidence about method quality.

### Formal training and evaluation evidence

- WORD training run:
  `Results/PSEM_v3_Triplet_Loss3/WORD_2D/PSEMv3_TripletContext_Loss3_Delta01_Common8_WORD_2D_C8_T20_GPU2_B8A4_E1200`
- WORD evaluation:
  `Results/PSEM_v3_Triplet_Loss3/Eval/WORD_2D/ckpt1199`
- AbdAutoPET training run:
  `Results/PSEM_v3_Triplet_Loss3/AbdAutoPET_2D/PSEMv3_TripletContext_Loss3_Delta01_AbdAutoPET_2D_C4_T20_GPU4_B12A4_E1200`
- AbdAutoPET evaluation:
  `Results/PSEM_v3_Triplet_Loss3/Eval/AbdAutoPET_2D/ckpt1199`

For each training job, review `Result.txt`, the final JSON line in `log.txt`,
`checkpoint-1199.pth`, elapsed time, and maximum GPU memory. For each
evaluation, verify `protocol.json`, expected test-case completeness, Step2
reconstruction, and Step3 Direct/Indirect raw/post per-organ Dice and NSD.

The final comparison must use PSEM-v2 on the same dataset as the primary mask
baseline. WORD additionally compares with OWT and standalone LossBalance-v3a.
AbdAutoPET additionally compares with OWT and PSEM-v1. Training losses across
different objectives are diagnostics only and must not be used as performance
rankings.

## Completion gate

Each GPU smoke must produce finite branch, LPIPS, delta-global, delta-positive,
negative Direct-energy, and negative Delta-energy metrics; cover all eight
WORD anchors or all four AbdAutoPET anchors plus present/absent cases; save
`checkpoint-0.pth`; and pass `tools/validate_psem_v3_triplet_smoke.py`. Only
then may the corresponding 1200-epoch training start.

## 2026-08-11 second retry

The first recovery smoke jobs `1655585` (WORD) and `1655588` (AbdAutoPET)
started on `gpua800n1` without a GPU allocation and failed at the strict CUDA
preflight. Their dependent chains became unusable and were cancelled. No model
forward or loss computation occurred, so these failures are infrastructure
incidents rather than method results.

The replacement submission explicitly passes typed A800 GRES on the `sbatch`
command line and excludes `gpua800n1`, `gpua800n2`, and `gpua800n6`:

| Dataset | Stage | Job | Dependency | Initial state |
|---|---|---:|---|---|
| WORD | smoke | 1658407 | none | PENDING (Priority) |
| WORD | train | 1658408 | afterok:1658407 | PENDING (Dependency) |
| WORD | evaluation | 1658409 | afterok:1658408 | PENDING (Dependency) |
| AbdAutoPET | smoke | 1658410 | none | PENDING (Priority) |
| AbdAutoPET | train | 1658411 | afterok:1658410 | PENDING (Dependency) |
| AbdAutoPET | evaluation | 1658412 | afterok:1658411 | PENDING (Dependency) |

Immediately after submission, every job showed the expected
`ReqTRES=...gres/gpu=N`, typed `TresPerNode=gres:gpu:a800:N`, and
`ExcNodeList=gpua800n[1-2,6]`. Future audits should use these six job IDs in
place of the superseded `1655585`--`1655590` chain.
