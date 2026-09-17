# Controlled query/readout ablation

Existing E, `arm_f_attention`, and `arm_f_linear` implementations are retained.
New heads are `arm_f_query_dot` and `arm_f_reverse_dot`.

| Head | Token refinement | Reverse pixel update | Readout |
| --- | --- | --- | --- |
| arm_e_multiscale_query | Existing mean-pool query | None | Existing normalized dot |
| arm_f_query_dot | Existing F P16/P8/P4 blocks | None | Mean refined tokens, normalized dot with original P4 |
| arm_f_reverse_dot | Same F blocks | Existing reverse attention + residual + FFN | Same query, normalized dot with updated P4 |
| arm_f_attention | Unchanged | Unchanged | Existing LayerNorm + linear classifier |

New dot heads use `sqrt(channels) * cosine(pixel, mean(refined_tokens))`.
No learned pooling or added query projection. Slot identity is already injected
before refinement and is not added again. E's own query construction is unchanged.
The two new paths support 2D and 3D; the controlled E/F experiment is 2D because
the existing E baseline supports only 2D.

Planned common protocol: WORD native07072, 448/global448/ROI384, ROI probability
0.2, scratch, seed0, 20 tokens, 128 channels, small-organ loss, lambda_seg0.01,
LR7.5e-5, 118800 updates, warmup5940, 4 GPUs, microbatch8/accum6/effective192.
Use identical background reduction and optimizer settings across runs. No resume
or MAE initialization. Retain token self-attention, FFN and F attention PE.

Use the existing unified 2D evaluator with EXPECTED_SLOT_HEAD_TYPE set explicitly:
training first6000 calibration, test fixed0.5 and calibrated, raw/post, plus recon.
Do not compare calibrated results against an uncalibrated historical baseline.

Same seed alone does not ensure identical common-module initialization across
architectures (constructor RNG consumption differs). Before claiming an exact
initial-weight-matched ablation, audit or explicitly synchronize shared initial
weights in a separate experiment initialization step. Legacy initialization is
not changed here. E vs F also changes query aggregation/projection and PE, so it
tests the refinement design as a package, not cross-attention alone.

CPU tests cover new paths in 2D/3D, finite gradients and strict round-trip loading;
legacy decoder tests compare keys, initialization and outputs against fd9cf91.
No GPU training is implied by these tests.
