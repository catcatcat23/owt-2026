#!/usr/bin/env python3
"""Extract the frozen LPIPS submodule from an original OWT checkpoint."""

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    prefix = "perceptual_loss."
    state = {
        key[len(prefix):]: value
        for key, value in checkpoint["model"].items()
        if key.startswith(prefix)
    }
    if not state:
        raise RuntimeError("checkpoint contains no perceptual_loss state")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output)
    print("saved {} tensors to {}".format(len(state), output))


if __name__ == "__main__":
    main()
