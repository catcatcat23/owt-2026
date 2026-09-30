"""Resolve final checkpoint by completed update budget, never by test performance."""
import json
from pathlib import Path
import sys
import torch


def final_checkpoint(run):
    records = [json.loads(line) for line in (run / "log.txt").read_text().splitlines() if line.strip()]
    last = records[-1]
    assert int(last["train_optimizer_updates"]) == 118800, last
    path = run / ("checkpoint-%d.pth" % last["epoch"])
    state = torch.load(path, map_location="cpu")
    args = state["args"]
    assert state["epoch"] == last["epoch"]
    assert args.slot_head_type == "arm_e_multiscale_query" and args.query_refinement == "cross_attn"
    assert args.mae_init_scope == "encoder" and not args.resume
    assert args.batch_size == 16 and args.accum_iter == 3 and args.seed == 0
    assert args.max_optimizer_updates == 118800 and abs(args.lambda_seg - 0.01) < 1e-12
    assert list(args.expected_spacing) == [0.7, 0.7, 2.0] and args.organ_roi_probability == 0.2
    assert args.input_size == 448 and args.global_crop_size == 448
    assert Path(args.val_data_path).name == "WORD_Validation_2D_native07072.csv"
    assert (run / "mae_initial_checkpoint_load.json").is_file()
    return path


if __name__ == "__main__":
    print(final_checkpoint(Path(sys.argv[1])))
