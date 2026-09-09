# Experiment Version Policy

- Before editing an experiment, fetch origin and inspect the development branch,
  the actual runtime worktree, and queued/running jobs.
- Commit and push fixes before submitting or resubmitting training. A fix in
  another worktree does not repair the runtime worktree.
- For new OrganSlot submissions, use the latest origin/feature/orgslot commit.
  Run bash tools/check_experiment_revision.sh RUNTIME_WORKTREE first. It must pass.
  Resolve dirty changes without discarding user work.
- Use an immutable runtime worktree or source snapshot. Do not update source
  underneath running or queued jobs. Existing experiments retain their pinned
  revision; replacing a queued job requires updating its evaluation dependency.
- Before sbatch, record the full commit, runtime path, account/cluster, and script.
  Verify remote HEAD or the transferred archive checksum against the local source.
- After sbatch, verify job state, runtime path, and afterok evaluation dependency.
  Report CPU tests and GPU preflight separately. Do not claim GPU validation
  passed while it is pending.
