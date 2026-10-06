"""Select Stage2 by completed training budget and verify frozen-parameter audit."""
import argparse
import json
import math
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch


def final_checkpoint(run, budget=None):
    run = Path(run)
    last = json.loads((run / "log.txt").read_text().splitlines()[-1])
    budget = last.get("optimizer_updates") if budget is None else budget
    if budget not in (59400, 118800) or last.get("optimizer_updates") != budget:
        raise ValueError("Stage2 has not completed a supported update budget")
    if any(not math.isfinite(v) for v in last.values() if isinstance(v, (int, float))):
        raise ValueError("Non-finite final training record")
    path = run / f"checkpoint-{last['epoch']}.pth"
    state = torch.load(path, map_location="cpu")
    args = state["args"]
    args = args if isinstance(args, dict) else vars(args)
    expected = {"incremental_stage": "stage2", "max_optimizer_updates": budget,
                "mae_init_scope": "encoder", "background_policy": "separation"}
    from util.orgslot_incremental44 import validate_incremental_architecture
    head = validate_incremental_architecture(args)
    if head == "arm_f_sam_tail" and args.get("stage2_shared_segmentation") != "slot_private":
        raise ValueError("Unexpected SAM-tail Stage2 policy")
    if any(args.get(k) != v for k, v in expected.items()) or state["epoch"] != last["epoch"]:
        raise ValueError("Unexpected Stage2 checkpoint configuration/epoch")
    steps = [float(s["step"]) for s in state["optimizer"]["state"].values() if "step" in s]
    if not steps or not all(math.isfinite(s) for s in steps) or max(steps) != budget:
        raise ValueError("Optimizer state does not prove completion")
    for name, value in state["model"].items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError("Non-finite tensor: " + name)
    audit = json.loads((run / "freeze_audit.json").read_text())
    if audit.get("epoch") != last["epoch"] or audit.get("match") is not True:
        raise ValueError("Final frozen-parameter audit did not pass")
    return path.resolve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("run")
    print(final_checkpoint(parser.parse_args().run))
