# WORD 0.7 ROI20 Multi-Query Pixel-Mask Experiment

## Question

Does preserving all 20 TGEnc token queries improve organ-mask readout over the
single pooled-query Arm C while preserving the same pixel and reconstruction paths?

## Architecture

The reconstruction path is unchanged:

```text
z -> OrganCollector_s -> TGEnc_s -> AHER_s -> canvas_s -> fusion -> reconstruction
```

The new segmentation path is:

```text
z -> shared pixel decoder -> P(x)
TGEnc_s token k -> shared query projection + slot identity -> q_{s,k}
part_logit_{s,k}(x) = sqrt(D) * cosine(P(x), q_{s,k})
mask_logit_s(x) = logmeanexp_k(part_logit_{s,k}(x))
```

The shared pixel decoder upsamples the final 28x28 ViT patch grid to a 112x112
128-channel feature map. All 20 token queries produce part-response maps. Log-mean-exp is used as a
smooth maximum with no token-count logit bias; identical token queries exactly
recover the single-query score. Per-slot logits are bilinearly resized to
448x448.
Segmentation gradients update the shared ViT, shared pixel/query decoder,
OrganCollector, TGEnc, and slot identity, but do not pass through AHER.

## Locked comparison

This Arm D changes only the token-to-query aggregation relative to Arm C.

| Setting | Arm A | Arm B | Arm C | Arm D |
|---|---|---|---|---|
| head | linear | multiscale_conv | pooled query-dot | multi-query-dot |
| head input | AHER canvas | AHER canvas | mean of 20 TGEnc tokens + final ViT pixels | 20 independent TGEnc queries + same pixels |
| query aggregation | n/a | n/a | mean before projection | log-mean-exp after 20 part maps |
| spacing | 0.7x0.7x2.0 | same | same | same |
| input / ROI crop | 448 / 384 | same | same | same |
| ROI probability | 0.2 | same | same | same |
| supervision | retained | same | same | same |
| loss | small_organ | same | same | same |
| lambda_seg | 0.01 | same | same | same |
| effective batch | 192 | same | same | same |
| updates / warmup | 118800 / 5940 | same | same | same |
| optimizer / base LR / WD | AdamW / 1e-4 / 0.05 | same | same | same |
| seed | 0 | same | same | same |

No diversity loss is added in the first controlled run. Token specialization or
collapse must be measured after training instead of changing head and loss at
the same time.

## Acceptance gates

1. Unit tests: finite forward/backward, exact mask shapes, dropped-row zeros,
   shared pixel and slot-query gradients, no AHER segmentation gradient.
2. Real 0.7 ROI20 smoke: eight optimizer updates, finite reconstruction,
   LPIPS, small-organ metrics, positive focus-organ coverage, checkpoint keys.
3. Formal training is submitted with `afterok` on the smoke job.
4. Final evaluation must use the same validation/test inference protocol as
   Arm A/B, including fixed 0.5 head threshold as the primary result.
