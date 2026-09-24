#!/bin/bash
#SBATCH --job-name=owt_common8
#SBATCH --partition=cpudebug
#SBATCH --account=sifansong
#SBATCH --qos=cpudebug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/work/aac/bolinren19/2026-07/OD_OWT/slurm/logs/preprocess_common8/%x_%j.out
#SBATCH --error=/gpfs/work/aac/bolinren19/2026-07/OD_OWT/slurm/logs/preprocess_common8/%x_%j.err

set -euo pipefail

WORKDIR=/gpfs/work/aac/bolinren19/2026-07/OD_OWT
DATA_ROOT=/gpfs/work/aac/bolinren19/2026-07/DATA
CONDA_ENV=/gpfs/work/aac/bolinren19/.conda/envs/abdpet
OUTPUT_ROOT=${DATA_ROOT}/processed/OWT_Common8

mkdir -p ${WORKDIR}/slurm/logs/preprocess_common8
export PATH=${CONDA_ENV}/bin:${PATH}
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

run_dataset() {
    local dataset=$1
    local source_root=$2
    local output_dir=$3

    echo "Started ${dataset} Common8 preprocessing at $(date)"
    echo "Source: ${source_root}"
    echo "Output: ${output_dir}"

    python tools/preprocess_owt_common8.py \
        --dataset ${dataset} \
        --root ${source_root} \
        --out ${output_dir} \
        --spacing 1 1 3 \
        --hu-min -175 \
        --hu-max 250 \
        --canvas-size 512 \
        --image-size 224 \
        --jpeg-quality 95 \
        --fix-frame 4 \
        --seed 42 \
        --workers ${SLURM_CPUS_PER_TASK}

    echo "Finished ${dataset} Common8 preprocessing at $(date)"
}

cd ${WORKDIR}
run_dataset \
    word \
    ${DATA_ROOT}/WORD/raw/WORD-V0.1.0 \
    ${OUTPUT_ROOT}/WORD
run_dataset \
    btcv \
    ${DATA_ROOT}/BTCV/Abdomen/RawData \
    ${OUTPUT_ROOT}/BTCV
