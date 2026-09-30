#!/usr/bin/env bash
#SBATCH --job-name=WORD96_prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -euo pipefail
: "${EXPERIMENT_WORKDIR:?}" "${NEW_DATA_ROOT:?}"
cd "$EXPERIMENT_WORKDIR"
export ARM_F_TARGET=bolin-sip ARM_F_DIMENSION=2D
source scripts/orgslot/arm_f_environment.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PYTHON" tools/prepare_word_official96.py build \
 --plan configs/orgslot/word_official96_seed42.json \
 --raw /gpfs/work/aac/bolinren19/2026-07/DATA/WORD/raw/WORD-V0.1.0 \
 --labels-zip /gpfs/work/aac/bolinren19/2026-07/DATA/WORD/raw/WORD_V0.1.0_labelsTs.zip \
 --old "$PROCESSED_ROOT" --output "$NEW_DATA_ROOT"
