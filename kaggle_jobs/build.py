"""Đóng gói repo thành một kernel Kaggle (script) cho từng stage rồi (tuỳ chọn) push.

  python kaggle_jobs/build.py smoke            # chỉ sinh kaggle_jobs/build/smoke/
  python kaggle_jobs/build.py main --push      # sinh và push lên Kaggle (GPU T4 x2)

Mỗi stage gồm 2 "lane" chạy song song, lane i dùng GPU i; lệnh trong một lane chạy tuần tự.
"""
import argparse
import base64
import io
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER = "wanghui1101"
DATASET = "lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes"
INCLUDE = ["scripts", "lab", "configs", "tests", "pytest.ini", "requirements-colab.txt"]


def score(name, pred, out, run=None, coord="norm1000", split="test"):
    meta = ""
    if run:
        meta = (f" $(python -m lab.meta_args {run})")
    return (f"python scripts/score.py --pred {pred} --name {name} --out {out} --split {split} "
            f"--coord {coord}{meta}")


def train_and_eval(setup, run, pred, name, ev, extra=""):
    return [f"python -m lab.train --setup {setup} --out {run} {extra}".strip(),
            f"python -m lab.infer --run-dir {run} --out {pred}" + (" --coord pixel" if "pixel" in extra else ""),
            score(name, pred, ev, run=run, coord="pixel" if "pixel" in extra else "norm1000")]


STAGES = {
    # Kiểm tra nhanh toàn bộ đường ống trước khi tốn GPU cho thí nghiệm thật.
    "smoke": {
        "lanes": [
            ["python -m lab.inspect_model --out work/inspect.json",
             "python -m lab.train --setup E3 --out work/runs/smoke_e3 --max-steps 4 --save-every 2",
             "python -m lab.check_reload work/runs/smoke_e3",
             "python -m lab.infer --run-dir work/runs/smoke_e3 --split val --limit 8 "
             "--out work/preds/smoke_e3_val.jsonl"],
            ["python -m lab.infer --split val --limit 16 --out work/preds/smoke_e0_val.jsonl",
             "python -m lab.train --setup E1 --out work/runs/smoke_e1 --max-steps 3",
             "python -m lab.grpo --init-run work/runs/smoke_e1 --out work/runs/smoke_e5 --max-steps 2 "
             "--limit 16"],
        ],
        "prev": [], "timeout": 3 * 3600,
    },
    "smoke2": {
        "lanes": [
            ["python -m lab.train --setup E3 --out work/runs/smoke_e3 --max-steps 4 --save-every 2",
             "python -m lab.check_reload work/runs/smoke_e3",
             "python -m lab.infer --run-dir work/runs/smoke_e3 --split val --limit 16 "
             "--out work/preds/smoke_e3_val.jsonl"],
            ["python -m lab.inspect_model --out work/inspect.json",
             "python -m lab.train --setup E2 --out work/runs/smoke_e2 --max-steps 4",
             "python -m lab.check_reload work/runs/smoke_e2"],
        ],
        "prev": [], "timeout": 3 * 3600,
    },
    "main": {
        "lanes": [
            train_and_eval("E1", "work/runs/e1", "work/preds/e1.jsonl", "E1", "work/evidence/e1.json")
            + train_and_eval("E3", "work/runs/e3", "work/preds/e3.jsonl", "E3", "work/evidence/e3.json"),
            train_and_eval("E2", "work/runs/e2", "work/preds/e2.jsonl", "E2", "work/evidence/e2.json")
            + ["python -m lab.infer --out work/preds/e0.jsonl",
               score("E0", "work/preds/e0.jsonl", "work/evidence/e0.json")],
        ],
        "prev": [], "timeout": 8 * 3600,
    },
}


def payload():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for item in INCLUDE:
            tf.add(ROOT / item, arcname=item,
                   filter=lambda t: None if "__pycache__" in t.name or t.name.endswith(".pyc") else t)
    return base64.b64encode(buf.getvalue()).decode()


def build(stage, spec):
    out = ROOT / "kaggle_jobs" / "build" / stage
    out.mkdir(parents=True, exist_ok=True)
    tpl = (ROOT / "kaggle_jobs" / "runner_template.py").read_text(encoding="utf-8")
    code = (tpl.replace("__STAGE__", stage).replace("__LANES__", json.dumps(spec["lanes"]))
            .replace("__PREV__", json.dumps([f"{USER}/{p}" for p in spec["prev"]]))
            .replace("__PAYLOAD__", payload()))
    (out / "run.py").write_text(code, encoding="utf-8")
    slug = f"day10-vlm-{stage}"
    meta = {"id": f"{USER}/{slug}", "title": slug, "code_file": "run.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True,
            "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [DATASET],
            "kernel_sources": [f"{USER}/{p}" for p in spec["prev"]],
            "competition_sources": [], "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out, slug


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=sorted(STAGES))
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    spec = STAGES[a.stage]
    out, slug = build(a.stage, spec)
    print(f"built {out} ({(out / 'run.py').stat().st_size / 1e3:.0f} kB)")
    for i, lane in enumerate(spec["lanes"]):
        print(f"lane{i}:", *lane, sep="\n  ")
    if a.push:
        subprocess.run(["kaggle", "kernels", "push", "-p", str(out), "-t", str(spec["timeout"]),
                        "--accelerator", "NvidiaTeslaT4"], check=True)
        print(f"https://www.kaggle.com/code/{USER}/{slug}")


if __name__ == "__main__":
    main()
