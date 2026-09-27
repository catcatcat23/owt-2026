# A/B/C/D retirement (2026-09-27)

User authorized retiring A/B/C/D code, with results preserved.
Removed 18 dedicated WORD07072 A-D training/smoke entrypoints from the current
development tree. Recover any removed file from the remote tag
`archive/20260927/abcd-before-retirement` (7b250c9).

Checkpoints, experiment metrics, inference outputs, historical runtime worktrees,
and all queued/running jobs are unchanged. No experiment data was deleted.

Partial cleanup only: core head removal/default changes/test migration were
blocked by approval review due to shared E/F dependencies. Those operations
were NOT executed. A-D model implementations and evaluation compatibility
remain pending explicit approval for that shared-path refactor. E/F still
inherit SharedPixelQueryDecoder; never delete that class without preserving
its initialization/state_dict behavior. Existing defaults are unchanged.

Historical docs may name the deleted launchers; use the archive tag to reproduce
those commands. This retirement is a maintenance choice, not a declaration
that the scientific results were invalid.
