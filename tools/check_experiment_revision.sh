#!/usr/bin/env bash
set -euo pipefail
runtime_dir=${1:?usage: bash tools/check_experiment_revision.sh RUNTIME_WORKTREE}
expected_ref=${2:-origin/feature/orgslot}
git -C "$runtime_dir" fetch origin
runtime_head=$(git -C "$runtime_dir" rev-parse HEAD)
expected_head=$(git -C "$runtime_dir" rev-parse "${expected_ref}^{commit}")
if [[ -n $(git -C "$runtime_dir" status --porcelain --untracked-files=normal) ]]; then
  printf 'ERROR: runtime worktree has uncommitted changes: %s\n' "$runtime_dir" >&2
  exit 1
fi
if [[ "$runtime_head" != "$expected_head" ]]; then
  printf 'ERROR: runtime HEAD %s differs from %s (%s)\n' "$runtime_head" "$expected_ref" "$expected_head" >&2
  exit 1
fi
printf 'Verified runtime=%s commit=%s reference=%s\n' "$runtime_dir" "$runtime_head" "$expected_ref"
