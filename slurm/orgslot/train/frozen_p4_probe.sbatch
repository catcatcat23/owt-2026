#!/usr/bin/env bash
#SBATCH --job-name=F_P4_probe_pancreas
#SBATCH --partition=gpu4090,gpu40901t
#SBATCH --account=sifansong
#SBATCH --qos=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
set -euo pipefail
: "${PROBE_ROOT:?immutable extracted source parent required}"
TASK_ENV=/gpfs/work/aac/sifansong/envs/abdpet
export PATH="$TASK_ENV/bin:$PATH"
export PYTHONPATH="$PROBE_ROOT/source"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
export LD_LIBRARY_PATH="$TASK_ENV/lib:/gpfs/spack/opt/spack/linux-icelake/mesa-25.0.5-zygiclacf7ay3uqqvqizl3lsvuzkk4od/lib:${LD_LIBRARY_PATH:-}"
cd "$PROBE_ROOT/source"
sha256sum "$PROBE_ROOT/source.tar"
python tools/frozen_p4_probe.py --checkpoint /gpfs/work/aac/sifansong/worktrees/arm_f_mae_encoder_df7ed39/Results/OrganSlotBank/Common8/WORD_2D/ArmF_2D_137867/checkpoint-802.pth --train-csv /gpfs/work/aac/sifansong/data/OWT_Common8_07072_NATIVE/WORD/csv/WORD_Training_2D_native07072.csv --class-id 6 --batch-size 16 --epochs 10 --workers 4 --output "$PROBE_ROOT/results_${SLURM_JOB_ID}"
