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

# XEC access and status reporting

- User-designated XEC endpoint: `xecdtn.hpc.xjtlu.edu.cn`. On 2026-11-09,
  sifansong and antengcai23 both authenticated successfully there; hostname
  `xpxecdtn1` and Slurm queries were verified. Prefer this endpoint for XEC
  status/log checks. Continue using the xjtlu-hpc-access helper; its hardcoded
  xeclogin SSH destination requires an explicit endpoint override for this route.
- `xeclogin.hpc.xjtlu.edu.cn` closed connections during public-key authentication
  that day. Do not assume an account/key failure without checking DTN. Never
  disable host-key verification to work around an authentication failure.
- Verify hostname, username and cluster before interpreting job IDs. An SSH
  failure is not a Slurm job failure. If all approved endpoints fail, label
  cached progress with its actual observation time; do not present it as current.
- Queue start times are scheduler estimates, not guarantees. Report planned
  future stages separately from jobs that actually have Slurm IDs.
