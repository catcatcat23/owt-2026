# 2D spatial PE restart: 2026-09-08

Confirmed: SIP job 2910849 used FP16 (both launch banner and argv), not BF16.
The PE sbatch omitted AMP_DTYPE and inherited the common launcher's fp16 default.
At epoch 64 step 233, gradient norm was nonfinite; strict clipping aborted
before optimizer.step. This also prevents GradScaler from reaching its usual
skip/backoff path. Loss clipping cannot repair an already nonfinite gradient.
The first offending operator is NOT known: old logs did not retain per-parameter
gradient diagnostics or the failing state. FP16 overflow is a hypothesis, not
a proven operator-level diagnosis.

Fix: explicitly select BF16 (no FP16 scaler), retain clipping 1.0 and strict
failure, check parameter finiteness every update, and report bad gradient
parameter names plus scaler state on any future backward/update failure.
Save every 25 epochs. Use a two-update, full-data, same-batch DDP preflight
inside the training allocation; only start formal training if validation passes.
Restart seed 0 from scratch in a new output directory (only checkpoint-0 survived).
Do not alter PE, loss, sampling, LR, or optimizer update budget.

This restart changes numeric precision relative to the historical FP16 no-PE
control; a strictly isolated PE claim ultimately needs a matched BF16 no-PE
control. A passed preflight is NOT proof of stability past epoch 64.
Keep failed output intact; replace only its permanently blocked evaluation job.

Submitted: SIP bolinren19 / account angelosstefanidis / QoS 8a800.
Training 2919868 (4 A800, 24 CPU, 192 GB); unified evaluation 2919869
(1 A800, 10 CPU, 128 GB), afterok:2919868. Old dead evaluation 2910855
cancelled; old training logs/checkpoint preserved. Five unittest checks passed
(PE shape/gradients/initialization plus launch-precision guards), bash syntax
and git diff checks passed. GPU preflight remains pending with training.
