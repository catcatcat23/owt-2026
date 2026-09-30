"""Select a completed Stage1 by training budget, never by test performance."""
import argparse
import json
import math
from pathlib import Path

import torch


def validate_stage1_completion(checkpoint, budget=59400):
    args = checkpoint.get("args", {})
    args = args if isinstance(args, dict) else vars(args)
    if args.get("incremental_stage") != "stage1" or args.get("max_optimizer_updates") != budget:
        raise ValueError("Require completed incremental Stage1 at the declared budget")
    steps = [float(s["step"]) for s in checkpoint.get("optimizer", {}).get("state", {}).values() if "step" in s]
    if not steps or not all(math.isfinite(s) for s in steps) or max(steps) != budget:
        raise ValueError("Stage1 optimizer state does not prove completion")
    return args


def final_checkpoint(run, budget=59400):
    run = Path(run)
    records = [json.loads(line) for line in (run / "log.txt").read_text().splitlines() if line.strip()]
    if not records or records[-1].get("optimizer_updates") != budget:
        raise ValueError("Stage1 log has not reached its update budget")
    last = records[-1]
    if any(isinstance(v, (int, float)) and not math.isfinite(v) for v in last.values()):
        raise ValueError("Non-finite final Stage1 metrics")
    path = run / f"checkpoint-{last['epoch']}.pth"
    checkpoint = torch.load(path, map_location="cpu")
    args = validate_stage1_completion(checkpoint, budget)
    if checkpoint["epoch"] != last["epoch"]:
        raise ValueError("Checkpoint/log epoch mismatch")
    if (args.get("slot_head_type"), args.get("query_refinement"), args.get("mae_init_scope")) != ("arm_e_multiscale_query", "cross_attn", "encoder"):
        raise ValueError("Unexpected Stage1 architecture/initialization")
    for name, tensor in checkpoint["model"].items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError("Non-finite checkpoint tensor: " + name)
    return path.resolve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("run")
    args = parser.parse_args()
    print(final_checkpoint(args.run))
