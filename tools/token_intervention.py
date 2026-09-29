"""Frozen Arm E token interventions. No optimizer, no threshold selection."""
import argparse
import csv
import json
import random
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader
from tools.eval_common8_orgslot_reconstruction_threshold import (
    build_dataset, build_model, parse_class_configuration, sha256, atomic_json,
    new_counter, update_counter, postprocess_slice)


class TokenBank:
    def __init__(self, capacity=128, seed=0):
        self.capacity = capacity
        self.rng = random.Random(seed)
        self.sums, self.counts, self.samples = {}, {}, {}

    def add(self, name, positive, token, case, index):
        key = (name, bool(positive))
        value = token.detach().cpu().float()
        self.sums[key] = self.sums.get(key, torch.zeros_like(value)) + value
        self.counts[key] = self.counts.get(key, 0) + 1
        pool = self.samples.setdefault(key, [])
        record = (value.clone(), str(case), int(index))
        if len(pool) < self.capacity:
            pool.append(record)
        else:
            j = self.rng.randrange(self.counts[key])
            if j < self.capacity:
                pool[j] = record

    def mean(self, name, positive=None):
        keys = [(name, False), (name, True)] if positive is None else [(name, bool(positive))]
        return sum(self.sums[k] for k in keys) / sum(self.counts[k] for k in keys)

    def donor(self, name, positive, excluded_cases=()):
        pool = [r for r in self.samples[(name, bool(positive))] if r[1] not in excluded_cases]
        if not pool:
            raise ValueError("No independent donor case")
        # Equal probability per cached donor case, then per cached slice.
        cases = sorted(set(r[1] for r in pool))
        case = self.rng.choice(cases)
        return self.rng.choice([r for r in pool if r[1] == case])


def case_ids(path):
    with open(path) as f:
        return {Path(r["mask_pth"]).parent.name for r in csv.DictReader(f)}


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ("checkpoint", "train-csv", "test-csv", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--class-map", default="configs/orgslot/common8_offline.json")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--donor-repeats", type=int, default=3)
    p.add_argument("--bank-capacity", type=int, default=128)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    train_ids, test_ids = case_ids(args.train_csv), case_ids(args.test_csv)
    assert len(train_ids) == 96 and len(test_ids) == 24 and not train_ids & test_ids
    torch.manual_seed(args.seed)
    specs, _ = parse_class_configuration(Path(args.class_map))
    foreground = [(int(s["raw_class_id"]), s["name"]) for s in specs if int(s["raw_class_id"]) > 0]
    model, report = build_model("orgslot", Path(args.checkpoint), specs, 448, "linear_sqrt")
    assert report["exact"] and report["slot_head_type"] == "arm_e_multiscale_query"
    model.requires_grad_(False).eval().cuda()
    # Preserve decoder query_norm, projection, slot identity and calibration.
    decoder = model.pixel_query_decoder
    original_pixels = decoder.forward_pixels
    captured = {}
    def capture_pixels(*a, **kw):
        value = original_pixels(*a, **kw)
        captured["pixels"] = value
        return value
    datasets = {k: build_dataset(Path(path), 448, 448)
                for k, path in [("train", args.train_csv), ("test", args.test_csv)]}
    provenance = dict(vars(args), load_report=report,
        checkpoint_sha256=sha256(Path(args.checkpoint)),
        train_sha256=sha256(Path(args.train_csv)), test_sha256=sha256(Path(args.test_csv)),
        protocol="FP32; original crop448; fixed0.5 raw and post(min20,opening1); no threshold fitting",
        warning="Exploratory test-set intervention, NOT a new segmentation method. Matched conditions use recipient GT presence only for diagnostic donor selection.",
        train_cases=sorted(train_ids), test_cases=sorted(test_ids))
    atomic_json(out / "provenance.json", provenance)
    bank = TokenBank(args.bank_capacity, args.seed)
    def forward(images, heads):
        keep = torch.ones(images.shape[0], len(specs), dtype=torch.bool, device="cuda")
        return model(images, slot_keep_mask=keep, head_compute_mask=keep,
                     decode_heads=heads, decode_reconstruction=False, return_diagnostics=True)
    with torch.inference_mode():
        processed = 0
        for batch in DataLoader(datasets["train"], batch_size=args.batch_size,
                                shuffle=False, num_workers=args.workers):
            result = forward(batch["image"].cuda(), False)
            for cid, name in foreground:
                pos = (batch["full_label"] == cid).flatten(1).any(1)
                for i, case in enumerate(batch["case_id"]):
                    bank.add(name, bool(pos[i]), result["slot_tokens"][name][i], case, processed+i)
            processed += len(batch["case_id"])
            if processed % 100 == 0:
                atomic_json(out / "progress.json", dict(stage="donor_cache", processed=processed, total=len(datasets["train"])))
                print("donor_cache", processed, flush=True)
        for _, name in foreground:
            for state in (False, True):
                assert len({r[1] for r in bank.samples[(name, state)]}) >= 3
        torch.save(dict(sums=bank.sums, counts=bank.counts, samples=bank.samples), out / "token_bank.pt")
        modes = ["original", "mean_all", "mean_matched"] + ["donor_" + str(i) for i in range(args.donor_repeats)]
        counters, changes, slice_rows = {}, {}, []
        with open(out / "donor_assignments.jsonl", "w") as donors, patch.object(decoder, "forward_pixels", capture_pixels):
            processed = 0
            for batch in DataLoader(datasets["test"], batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.workers):
                images = batch["image"].cuda()
                result = forward(images, True)
                pixels = captured.pop("pixels")
                labels = batch["full_label"][:, 0].numpy()
                for cid, name in foreground:
                    slot = model.slot_bank.get_slot(name)
                    tokens = result["slot_tokens"][name]
                    baseline = result["calibrated_logits"][name]
                    reread = slot.forward_query_head(tokens, pixels, decoder, images.shape[-2:])[1]
                    torch.testing.assert_close(reread, baseline, rtol=1e-5, atol=1e-5)
                    positive = [(labels[i] == cid).any() for i in range(len(labels))]
                    for mode in modes:
                        if mode == "original":
                            logits = baseline
                        else:
                            replacements = []
                            for i, case in enumerate(batch["case_id"]):
                                if mode.startswith("donor"):
                                    token, donor_case, donor_index = bank.donor(name, positive[i], (str(case),))
                                    donors.write(json.dumps(dict(mode=mode, organ=name, recipient_case=str(case),
                                        recipient_index=processed+i, positive=bool(positive[i]),
                                        donor_case=donor_case, donor_index=donor_index)) + "\n")
                                else:
                                    token = bank.mean(name, positive[i] if mode == "mean_matched" else None)
                                replacements.append(token)
                            replacement = torch.stack(replacements).to(tokens)
                            logits = slot.forward_query_head(replacement, pixels, decoder, images.shape[-2:])[1]
                        assert torch.isfinite(logits).all()
                        delta = (logits-baseline).abs().flatten(1).mean(1).cpu().tolist()
                        predictions = (logits[:, 0] > 0).cpu().numpy()
                        for i, case in enumerate(batch["case_id"]):
                            target = labels[i] == cid
                            ck = (mode, name, str(case))
                            entry = changes.setdefault(ck, [0., 0])
                            entry[0] += delta[i]; entry[1] += 1
                            for post in (False, True):
                                pred = postprocess_slice(predictions[i], 20, 1) if post else predictions[i]
                                key = (mode, "post" if post else "raw", name, str(case))
                                update_counter(counters.setdefault(key, new_counter()), pred, target)
                                slice_rows.append(dict(mode=mode, processing=key[1], organ=name, case=str(case),
                                    sample_index=processed+i, gt=int(target.sum()), pred=int(pred.sum()),
                                    tp=int((pred & target).sum()), logit_mae=delta[i]))
                processed += len(labels)
                if processed % 100 == 0:
                    atomic_json(out / "progress.json", dict(stage="interventions", processed=processed, total=len(datasets["test"])))
                    print("interventions", processed, flush=True)
        rows = []
        for (mode, processing, organ, case), c in sorted(counters.items()):
            pred, gt, tp = c["prediction"], c["target"], c["intersection"]
            change = changes[(mode, organ, case)]
            rows.append(dict(mode=mode, processing=processing, organ=organ, case=case, tp=tp,
                fp=pred-tp, fn=gt-tp, pred=pred, gt=gt, dice=2*tp/(pred+gt) if pred+gt else 1.,
                precision=tp/pred if pred else None, recall=tp/gt if gt else None,
                volume_ratio=pred/gt if gt else None, empty_prediction=bool(gt and not pred),
                zero_overlap=bool(gt and not tp), logit_mae=change[0]/change[1]))
        for filename, records in [("per_case.csv", rows), ("per_slice.csv", slice_rows)]:
            with open(out / filename, "w") as f:
                writer = csv.DictWriter(f, fieldnames=list(records[0]))
                writer.writeheader(); writer.writerows(records)
        summary = {}
        for mode in modes:
            summary[mode] = {}
            for processing in ("raw", "post"):
                summary[mode][processing] = {name: float(np.mean([r["dice"] for r in rows
                    if r["mode"] == mode and r["processing"] == processing and r["organ"] == name]))
                    for _, name in foreground}
        atomic_json(out / "results.json", dict(complete=True, samples=processed, metrics=summary,
            peak_memory_gib=torch.cuda.max_memory_allocated()/2**30, no_training=True))
        atomic_json(out / "progress.json", dict(stage="complete", complete=True, processed=processed))
    assert all(p.grad is None for p in model.parameters())


if __name__ == "__main__":
    main()

