#!/usr/bin/env python3
"""Run the shared Fixfr4 evaluator with PSEM AbdAutoPET preprocessing."""

import importlib.util
from pathlib import Path


SHARED_EVALUATOR = Path(
    "/gpfs/work/aac/bolinren19/2026-07/OD_OWT/"
    "tools/evaluate_owt_common8_3d.py"
)


def load_shared_evaluator():
    spec = importlib.util.spec_from_file_location(
        "shared_owt_3d_evaluator", SHARED_EVALUATOR
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load shared evaluator: {SHARED_EVALUATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    evaluator = load_shared_evaluator()
    shared_load_case = evaluator.load_case

    def load_case_per_sample(rows, input_size):
        return shared_load_case(
            rows, input_size, intensity_norm="per_sample"
        )

    evaluator.load_case = load_case_per_sample
    evaluator.CLASS_NAMES = {
        0: "background",
        1: "label_1",
        2: "label_2",
        3: "label_3",
        4: "label_4",
    }
    evaluator.main()


if __name__ == "__main__":
    main()
