"""In cờ --meta cho scripts/score.py từ <run>/run.json:  python -m lab.meta_args work/runs/e1"""
import json
import sys
from pathlib import Path

KEYS = ("trainable_params", "peak_vram_gb", "train_minutes")


def meta_args(run_dir):
    run = json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8"))
    return " ".join(f"--meta {k}={run[k]}" for k in KEYS)


if __name__ == "__main__":
    print(meta_args(sys.argv[1]))
