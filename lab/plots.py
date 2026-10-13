"""Biểu đồ cho báo cáo: F1 kèm CI95 từng thí nghiệm, đường loss E1–E3, reward GRPO.

  python -m lab.plots --evidence report/evidence --runs work/runs --out report/figures
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # 3 slot đầu của bảng màu phân loại (đạt kiểm tra CVD)
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_f1(evidence_dir, out):
    rows = []
    for p in sorted(Path(evidence_dir).glob("e*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        rows.append((p.stem, d["micro"]["f1"], *d["f1_ci95"]))
    rows.sort(key=lambda r: r[0])
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(rows) + 1.2), facecolor=SURFACE)
    _style(ax)
    ys = list(range(len(rows)))[::-1]
    for y, (name, f1, lo, hi) in zip(ys, rows):
        ax.plot([lo, hi], [y, y], color=SERIES[0], linewidth=2, solid_capstyle="round")
        ax.plot([f1], [y], "o", color=SERIES[0], markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(hi + 0.01, y, f"{f1:.3f}  [{lo:.3f}–{hi:.3f}]", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(ys, [r[0] for r in rows], color=INK)
    ax.set_xlim(0, min(1.0, max(r[3] for r in rows) + 0.22))
    ax.set_xlabel("F1@IoU0.5 trên test (311 ảnh) — chấm tròn: F1, vạch: CI95 bootstrap", color=MUTED, fontsize=9)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "f1_ci95.png", dpi=160)
    plt.close(fig)


def _read_log(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def plot_loss(runs_dir, out, names=("e1", "e2", "e3")):
    fig, ax = plt.subplots(figsize=(7, 3.6), facecolor=SURFACE)
    _style(ax)
    for color, name in zip(SERIES, names):
        p = Path(runs_dir) / name / "train_log.jsonl"
        if not p.exists():
            continue
        log = _read_log(p)
        xs = [r["step"] for r in log]
        # trung bình trượt 10 bước cho dễ đọc
        ys = [sum(r["loss"] for r in log[max(0, i - 9):i + 1]) / len(log[max(0, i - 9):i + 1])
              for i in range(len(log))]
        ax.plot(xs, ys, color=color, linewidth=2, label=name.upper())
        ax.text(xs[-1] + 3, ys[-1], name.upper(), color=INK, fontsize=8.5, va="center")
    ax.set_yscale("log")
    ax.set_xlabel("bước tối ưu", color=MUTED, fontsize=9)
    ax.set_ylabel("loss (trung bình trượt 10 bước, thang log)", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / "loss_e1_e3.png", dpi=160)
    plt.close(fig)


def plot_grpo(runs_dir, out, name="e5"):
    p = Path(runs_dir) / name / "grpo_log.json"
    if not p.exists():
        return
    log = [r for r in json.loads(p.read_text(encoding="utf-8")) if "reward" in r]
    if not log:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), facecolor=SURFACE)
    for ax, key, label in ((axes[0], "reward", "reward trung bình / bước"),
                           (axes[1], "completions/mean_length", "độ dài completion (token)")):
        _style(ax)
        xs = [r.get("step", i + 1) for i, r in enumerate(log)]
        ax.plot(xs, [r.get(key, float("nan")) for r in log], color=SERIES[0], linewidth=2)
        ax.set_title(label, color=INK, fontsize=9.5, loc="left")
        ax.set_xlabel("bước GRPO", color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "grpo_reward.png", dpi=160)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence", default="report/evidence")
    ap.add_argument("--runs", default="work/runs")
    ap.add_argument("--out", default="report/figures")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plot_f1(a.evidence, out)
    plot_loss(a.runs, out)
    plot_grpo(a.runs, out)
    print("đã vẽ", sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()
