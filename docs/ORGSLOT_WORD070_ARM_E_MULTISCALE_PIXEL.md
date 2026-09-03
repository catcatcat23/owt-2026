# OrganSlot Arm E: high-resolution multiscale pixel decoder

## Research question

Arm C and Arm D obtain similar WORD 2D results even though Arm D keeps 20
independent token queries. Arm E therefore tests a different bottleneck:
whether the final 28x28 ViT feature lacks the spatial detail required by small
organs.

The controlled change is:

    Arm C: ViT P16 -> upsampled pixel feature -> single organ query dot product
    Arm E: ViT P16 + image P8 + image P4 -> fused pixel feature
                                        -> same single organ query dot product

The convolutional path never predicts an organ mask by itself. The raw mask is
always the normalized dot product between the shared pixel feature and an
OrganSlot query.

## Exact architecture

For a 448x448 input:

    image -> lightweight Conv/GN/GELU stem -> P4  [B,128,112,112]
                                        -> P8  [B,128, 56, 56]
    ViT final patch tokens                 -> P16 [B,128, 28, 28]

    P16 upsample + projected P8 -> depthwise/pointwise refinement
         upsample + projected P4 -> depthwise/pointwise refinement
                                -> pixel feature [B,128,112,112]

    20 OrganSlot tokens -> Arm C mean aggregation/projection
                        -> one query [B,128]

    FP32 normalized dot product -> [B,1,112,112] -> bilinear 448x448

Arm E adds 256,320 parameters. The formal model has 186,799,826 total
parameters; the Arm E shared pixel decoder has 491,840 parameters.

## Fixed experimental controls

- WORD 2D, 8 foreground organs and one background slot
- spacing 0.7 x 0.7 x 2.0 mm
- input 448, ROI crop 384, ROI probability 0.2
- token factor 20, one aggregated organ query
- retained segmentation supervision, background segmentation weight 0
- MSE + LPIPS + 0.01 x small-organ segmentation loss
- Tversky FP/FN 0.3/0.7
- balanced focal weight 0.5, focal alpha/gamma 0.75/2
- top-2% hard negatives, negative-slice weight 0.1
- effective batch 192, 118800 optimizer updates, seed 0

No query-count, loss, sampling, ViT patch-size, threshold, or reconstruction
change is part of Arm E v1.

## Verification

Local tests:

- Arm C/D/E model regression: 20/20 passed
- retained/small-organ loss regression: 20/20 passed
- Arm E checkpoint compatibility: passed
- B=2 at 448x448, P4/P8/P16 and K=1/3/8 shapes: passed
- stem, P16 projection, OrganCollector, and query gradients: non-zero
- zero query: raw mask is exactly zero
- shuffled query: raw mask changes
- reconstruction with shared weights: Arm C and Arm E are bitwise equal

Single-GPU real-data smoke:

- SIP job 2814754, 1x A800, completed
- 4 optimizer updates completed without NaN/Inf
- peak allocated GPU memory: 12.76 GiB
- mean segmentation loss: 1.0118
- mean weighted segmentation loss: 0.0101
- observed ROI fraction: 0.1875
- checkpoint and formal smoke validator passed

Two-GPU DDP:

- job 2814794 exposed a launcher import-path error before model execution
- launcher fixed in commit 75c3b45
- replacement SIP job 2864374 requests 2x A800 under 8a800
- the synthetic DDP preflight deliberately gives rank 0 kidney and rank 1
  spleen, then performs three optimizer steps with fixed collective schema

## Slurm entry points

- single GPU smoke:
  slurm/orgslot/smoke/orgslot_word07072_arm_e_single_bolin_sip.sbatch
- two GPU DDP smoke:
  slurm/orgslot/smoke/orgslot_word07072_arm_e_ddp_bolin_sip.sbatch
- formal four GPU training:
  slurm/orgslot/train/orgslot_word07072_arm_e_bolin_sip.sbatch

Formal training must not be submitted until job 2864374 passes.

## Success criteria

Compare checkpoint-802 to Arm C with the same 24 test cases, fixed threshold
0.5, and identical postprocessing. Report all eight organs, overall foreground
Dice, and the mean of gallbladder, esophagus, and pancreas. Arm E supports the
hypothesis only if the small organs improve coherently rather than trading one
organ against another.
