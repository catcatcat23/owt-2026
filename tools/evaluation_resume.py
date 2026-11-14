"""Opt-in local metric checkpoints for time-limited evaluation jobs."""
import json
import os
import pickle
import time
from pathlib import Path


class EvaluationResume:
    def __init__(self, output_dir, config, enabled):
        self.path = Path(output_dir) / "metric_resume.pkl"
        self.enabled = enabled
        self.signature = json.dumps(config, sort_keys=True)
        self.state = {}
        self.started = time.time()
        if enabled and self.path.exists():
            with self.path.open("rb") as handle:
                saved = pickle.load(handle)  # Locally generated evaluation state only.
            if saved["signature"] != self.signature:
                raise ValueError("Resume configuration or input hashes differ")
            self.state = saved["state"]
        self.offset = self.state.get("processed", 0)
        self.previous_elapsed = self.state.get("elapsed", 0.0)

    def save(self, processed, counters, background=None):
        if not self.enabled:
            return
        state = dict(processed=processed, counters=counters, background=background,
                     elapsed=self.previous_elapsed + time.time() - self.started)
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("wb") as handle:
            pickle.dump(dict(signature=self.signature, state=state), handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self.path)

    def yield_if_due(self, processed, total, counters, background=None):
        deadline = float(os.environ.get("EVAL_DEADLINE_EPOCH", "inf"))
        if self.enabled and processed < total and time.time() >= deadline:
            self.save(processed, counters, background)
            print("Evaluation progress saved: {}/{}; continuing in next job".format(processed, total), flush=True)
            raise SystemExit(75)
