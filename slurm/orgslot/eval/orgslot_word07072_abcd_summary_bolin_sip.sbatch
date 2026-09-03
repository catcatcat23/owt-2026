#!/bin/bash
#SBATCH --job-name=sip_word070_abcd_table
#SBATCH --partition=gpua800
#SBATCH --account=angelosstefanidis
#SBATCH --qos=8a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/gpfs/work/aac/bolinren19/OD_OWT/.worktrees/orgslot_querymask_multiquery/slurm/orgslot/logs/WORD070_UnifiedEval/%x_%j.out
#SBATCH --error=/gpfs/work/aac/bolinren19/OD_OWT/.worktrees/orgslot_querymask_multiquery/slurm/orgslot/logs/WORD070_UnifiedEval/%x_%j.err

set -euo pipefail

ROOT=/gpfs/work/aac/bolinren19
WORKDIR=${ROOT}/OD_OWT/.worktrees/orgslot_querymask_multiquery
A=${ROOT}/OD_OWT/.worktrees/orgslot_head_ab/Results/OrganSlotBank/evaluation/Common8/WORD_2D/OrgSlot_WORD07072_ROI20_RetainedLinear_SmallOrgan_L0.01_Bolin_2413564_ckpt802_heads_test/results.json
B=${ROOT}/OD_OWT/artifacts/armb_121398_reference/heads_test_results.json
C=${ROOT}/OD_OWT/.worktrees/orgslot_querymask/Results/OrganSlotBank/evaluation/Common8/WORD_2D/OrgSlot_WORD07072_ROI20_RetainedQueryDot_SmallOrgan_L0.01_Bolin_2415372_ckpt802_heads_test/results.json
D=${ROOT}/OD_OWT/.worktrees/orgslot_querymask_multiquery/Results/OrganSlotBank/evaluation/Common8/WORD_2D/OrgSlot_WORD07072_ROI20_RetainedMultiQueryDot_SmallOrgan_L0.01_Bolin_2548702_ckpt802_heads_test/results.json
OUTPUT=${WORKDIR}/artifacts/word070_head_abcd_ckpt802
PYTHON=${ROOT}/.conda/envs/abdpet/bin/python

cd "${WORKDIR}"
test ! -e "${OUTPUT}"
"${PYTHON}" tools/summarize_common8_head_abcd.py \
  --arm A_linear="${A}" \
  --arm B_multiscale_conv="${B}" \
  --arm C_single_query_dot="${C}" \
  --arm D_multi_query_dot="${D}" \
  --output-dir "${OUTPUT}"

"${PYTHON}" - "${OUTPUT}" <<'PY'
import csv
import json
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
rows = list(csv.DictReader((output / "abcd_head_comparison.csv").open()))
payload = json.loads((output / "abcd_head_comparison.json").read_text())
assert len(rows) == 8
assert len(payload["provenance"]) == 4
assert {row["mode"] for row in rows} == {
    "fixed_0.5_post",
    "train_calibrated_post",
}
print((output / "abcd_head_comparison.md").read_text())
PY
