# SAM-tail + MAE encoder: soft prior ablation

Modes: soft_prior (routing only), soft_prior_aux (same routing plus supervision).
No P2, no tail modification, no added parameters. Before each P16/P8/P4 token
read, mean current tokens, L2 normalize query/pixels, dot times sqrt(C).
Use detached log(0.2+0.8*sigmoid(logits)) as additive attention bias shared by
20 tokens and four heads. Tail itself receives no mask prior in these modes.

Auxiliary logits retain gradients, are upsampled directly to GT resolution,
and use existing small-organ loss. Average three scale losses, eta=0.25:
L = L_reconstruction_related + 0.01*(L_final + 0.25*mean(L16,L8,L4)).
Log each scale, average and weighted auxiliary loss. Retained supervision;
background-only batches use zero auxiliary loss without skipping backward.
Same seed0, WORD07072 ROI20, microbatch16 x4GPU xaccum3=192, 118800 updates,
MAE encoder initialization; unified head calibration/test and recon evaluation.

## Literature and scope

Mask2Former official maskformer_model.py copies final loss weights to every
auxiliary layer and sums terms. Its Cityscapes panoptic config uses class2,
mask BCE5, Dice5. These are not relative auxiliary weights for our loss.
https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/maskformer_model.py
https://github.com/facebookresearch/Mask2Former/blob/main/configs/cityscapes/panoptic-segmentation/maskformer2_R50_bs16_90k.yaml

H-SAM official trainer.py uses w=0.6**(0.990**epoch), loss=(1-w)*coarse+w*final.
Thus coarse/final starts0.4/0.6, then coarse decays. It does not use fixed0.25.
https://github.com/Cccccczh404/H-SAM/blob/main/trainer.py

Our fixed0.25 on the mean is an explicit adaptation to retain the established
final segmentation/reconstruction balance, NOT an official paper parameter.
Do not change it based on test results. Only one seed; no significance claim.
CPU regression validation is separate from GPU/DDP validation.
