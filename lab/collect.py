"""Gom kết quả nhỏ từ output kernel Kaggle (work/kaggle/<stage>/) vào report/ để commit. KHÔNG chép trọng số.

  python -m lab.collect main stage2

- work/evidence/e*.json  → report/evidence/      (bằng chứng chấm bởi scripts/score.py)
- work/extra/*.json      → report/extra/         (thí nghiệm ngoài rubric, vd. không mask loss)
- work/preds/*.jsonl     → report/preds/ và work/preds/  (output thô của model, dùng cho phân tích)
- work/runs/<r>/{run.json, train_log.jsonl, grpo_log.json, reward_stats.jsonl, reload_check.json}
                         → report/runs/<r>/ và work/runs/<r>/
- logs/*.log             → report/logs/ (bỏ dòng thanh tiến trình tqdm)
"""
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_FILES = ("run.json", "train_log.jsonl", "grpo_log.json", "reward_stats.jsonl", "reload_check.json")
_TQDM = re.compile(r"(Loading weights|Fetching|Downloading|it/s\]|s/it\])")


def _copy(src, *dsts):
    for d in dsts:
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, d)


def collect(stage):
    base = ROOT / "work" / "kaggle" / stage
    w = base / "work"
    n = 0
    for p in sorted((w / "evidence").glob("e*.json")):
        _copy(p, ROOT / "report" / "evidence" / p.name)
        n += 1
    for p in sorted((w / "extra").glob("*.json")):
        _copy(p, ROOT / "report" / "extra" / p.name)
        n += 1
    for p in sorted((w / "preds").glob("*.jsonl")):
        if p.stem.startswith("smoke"):
            continue
        _copy(p, ROOT / "report" / "preds" / p.name, ROOT / "work" / "preds" / p.name)
        n += 1
    for run in sorted((w / "runs").glob("*")) if (w / "runs").exists() else []:
        if run.name.startswith("smoke"):
            continue
        for f in RUN_FILES:
            if (run / f).exists():
                _copy(run / f, ROOT / "report" / "runs" / run.name / f, ROOT / "work" / "runs" / run.name / f)
                n += 1
    for p in sorted((base / "logs").glob("*.log")):
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        keep = [ln for ln in lines if not _TQDM.search(ln)]
        out = ROOT / "report" / "logs" / p.name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(keep) + "\n", encoding="utf-8")
        n += 1
    print(f"{stage}: chép {n} file")


if __name__ == "__main__":
    for s in sys.argv[1:] or ["main"]:
        collect(s)
