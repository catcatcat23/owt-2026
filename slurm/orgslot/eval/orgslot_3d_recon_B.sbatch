#!/bin/bash
#SBATCH --job-name=os3d_recon_B
#SBATCH --partition=gpua800,gpua8001t
#SBATCH --account=sifansong
#SBATCH --qos=4gpus
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/work/aac/sifansong/worktrees/orgslot_3d_slicewise/os3d_recon_%j.out
#SBATCH --error=/gpfs/work/aac/sifansong/worktrees/orgslot_3d_slicewise/os3d_recon_%j.err
set -euo pipefail
cd /gpfs/work/aac/sifansong/worktrees/orgslot_3d_slicewise
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
/gpfs/work/aac/sifansong/envs/abdpet/bin/python -m tools.eval_common8_orgslot_heads_3d \
 --checkpoint /gpfs/work/aac/sifansong/worktrees/orgslot_armb_3d/Results/OrganSlotBank/Common8/WORD_3D/OrgSlot_WORD07072_ROI20_RetainedMultiConv3D_FixFr4_SmallOrgan_L0.01_Resume50_BF16_Clip1_128275/checkpoint-199.pth \
 --test-csv /gpfs/work/aac/sifansong/data/OWT_Common8_07072_NATIVE/WORD/csv/WORD_Test_2D_native07072.csv \
 --preprocess-summary /gpfs/work/aac/sifansong/data/OWT_Common8_07072_NATIVE/WORD/metadata/preprocess_summary.json \
 --expected-spacing 0.7 0.7 2 --expected-cases 24 \
 --output-dir /gpfs/work/aac/sifansong/worktrees/orgslot_3d_slicewise/diagnostics_B_${SLURM_JOB_ID} \
 --class-map configs/orgslot/common8_offline.json --fix-frame 4 --temp-stride 1 \
 --batch-size 1 --workers 4 --binary-threshold 0.5 --min-component-voxels 20 \
 --opening-radius-mm 0 --fusion-mode linear_sqrt --diagnostics --amp-dtype fp16
