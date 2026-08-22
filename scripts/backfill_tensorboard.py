"""Backfill TensorBoard scalars from immutable DSF JSON/manifest logs.

This is deliberately one-way: JSON remains the authoritative experiment
record, while TensorBoard is a convenient view. Re-running with a clean output
directory recreates the same scalar series without touching checkpoints.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--start-after", type=int, default=-1,
                        help="append only records and manifests newer than this step")
    args = parser.parse_args()
    run, destination = Path(args.run), Path(args.out)
    destination.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(destination))
    args_path = run / "args.json"
    if args_path.exists() and args.start_after < 0:
        writer.add_text("run/args", "```json\n" + args_path.read_text() + "\n```", 0)

    jsonl = run / "log.jsonl"
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            step = int(row["step"])
            if step <= args.start_after:
                continue
            for key, value in row.items():
                if key not in ("step", "elapsed") and isinstance(value, (int, float)):
                    writer.add_scalar(f"train/{key}", value, step)
            if "elapsed" in row:
                writer.add_scalar("system/elapsed_seconds", row["elapsed"], step)

    for path in sorted(run.glob("manifest_*.json")):
        row = json.loads(path.read_text())
        step = int(row["step"])
        if step <= args.start_after:
            continue
        for key, value in row.get("metrics", {}).items():
            if isinstance(value, (int, float)):
                writer.add_scalar(f"fixed_validation/{key}", value, step)
        if row.get("gif"):
            writer.add_text("artifacts/reconstruction_gif", str(run / row["gif"]), step)
    writer.close()
    print(json.dumps({"run": str(run.resolve()), "tensorboard": str(destination.resolve())}, indent=2))


if __name__ == "__main__":
    main()
