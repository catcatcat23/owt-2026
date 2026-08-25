# WORD 0.7 ROI20 Shared Pixel Query-Mask Experiment

## Question

Does bypassing the AHER canvas for segmentation improve organ-mask readout while
preserving the existing OrganSlot reconstruction path?

## Architecture

The reconstruction path is unchanged:

```text
z -> OrganCollector_s -> TGEnc_s -> AHER_s -> canvas_s -> fusion -> reconstruction
```

The new segmentation path is:

```text
z -> shared pixel decoder -> P(x)
TGEnc_s tokens -> shared query projection + slot identity -> q_s
mask_logit_s(x) = sqrt(D) * cosine(P(x), q_s)
```

The shared pixel decoder upsamples the final 28x28 ViT patch grid to a 112x112
128-channel feature map. Per-slot logits are bilinearly resized to 448x448.
Segmentation gradients update the shared ViT, shared pixel/query decoder,
OrganCollector, TGEnc, and slot identity, but do not pass through AHER.

## Locked comparison

Arm C changes only the segmentation readout relative to the existing 0.7
spacing Arm A/B runs.

| Setting | Arm A | Arm B | Arm C |
|---|---|---|---|
| head | linear | multiscale_conv on AHER canvas | shared pixel query-dot |
| spacing | 0.7x0.7x2.0 | same | same |
| input / ROI crop | 448 / 384 | same | same |
| ROI probability | 0.2 | same | same |
| supervision | retained | same | same |
| loss | small_organ | same | same |
| lambda_seg | 0.01 | same | same |
| Tversky FP/FN | 0.3/0.7 | same | same |
| Balanced Focal weight | 0.5 | same | same |
| hard-negative ratio | 0.02 | same | same |
| negative-slice weight | 0.1 | same | same |
| effective batch | 192 | same | same |
| updates / warmup | 118800 / 5940 | same | same |
| optimizer / base LR / WD | AdamW / 1e-4 / 0.05 | same | same |
| seed | 0 | same | same |

## Acceptance gates

1. Unit tests: finite forward/backward, exact mask shapes, dropped-row zeros,
   shared pixel and slot-query gradients, no AHER segmentation gradient.
2. Real 0.7 ROI20 smoke: eight optimizer updates, finite reconstruction,
   LPIPS, small-organ metrics, positive focus-organ coverage, checkpoint keys.
3. Formal training is submitted with `afterok` on the smoke job.
4. Final evaluation must use the same validation/test inference protocol as
   Arm A/B, including fixed 0.5 head threshold as the primary result.
