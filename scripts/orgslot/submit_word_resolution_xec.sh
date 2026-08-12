#!/usr/bin/env bash
set -euo pipefail

ARM=${1:?Usage: $0 crop336_roi224|spacing07072_448_roi384}
case "${ARM}" in
  crop336_roi224|spacing07072_448_roi384) ;;
  *) echo "Unknown ARM: ${ARM}" >&2; exit 2 ;;
esac
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

SMOKE_JOB=$(sbatch --parsable --export=ALL,ARM="${ARM}"   slurm/orgslot/resolution/orgslot_word_resolution_smoke.sbatch)
TRAIN_JOB=$(sbatch --parsable --dependency=afterok:"${SMOKE_JOB}"   --export=ALL,ARM="${ARM}"   slurm/orgslot/resolution/orgslot_word_resolution_train.sbatch)
EVAL_JOB=$(sbatch --parsable --dependency=afterok:"${TRAIN_JOB}"   --export=ALL,ARM="${ARM}",TRAIN_JOB_ID="${TRAIN_JOB}"   slurm/orgslot/resolution/orgslot_word_resolution_eval.sbatch)

printf 'arm=%s smoke=%s train=%s eval=%s\n'   "${ARM}" "${SMOKE_JOB}" "${TRAIN_JOB}" "${EVAL_JOB}"
