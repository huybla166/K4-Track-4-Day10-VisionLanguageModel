"""Phân tích cho REPORT.md, dùng đúng hàm chấm của scripts/score.py.

  python -m lab.analysis --preds work/preds --evidence report/evidence --out report/analysis
Sinh analysis.json + analysis.md: bảng tổng hợp, bootstrap cặp (chênh lệch F1 có vượt CI không),
bootstrap theo phiên quay, chấm sai thang toạ độ (câu 3), phân loại lỗi và recall theo kích thước box (câu 9).
"""
import argparse
import json
import random
from pathlib import Path

from lab.common import CLASSES, load_split, read_jsonl
from scripts.score import _gt_boxes, _match, _prf, _to_norm1000, evaluate, iou_xyxy, parse_output

ORDER = ["e0", "e1", "e2", "e3", "e4_coord_norm1000", "e4_coord_pixel", "e5", "nomask_e1"]


def load_preds(path):
    return {r["image_id"]: r["raw"] for r in read_jsonl(path)}


def per_image(entries, preds, coord="norm1000"):
    """(tp, fp, fn) mỗi ảnh, giống hệt score.evaluate."""
    rows = []
    for e in entries:
        parsed = parse_output(preds.get(e["image_id"]))
        pred = _to_norm1000(parsed, coord, e["width"], e["height"]) if parsed is not None else []
        per, _ = _match(_gt_boxes(e), pred)
        rows.append(tuple(sum(v[k] for v in per.values()) for k in range(3)))
    return rows


def _f1(rows, idx):
    tp = sum(rows[i][0] for i in idx)
    fp = sum(rows[i][1] for i in idx)
    fn = sum(rows[i][2] for i in idx)
    return _prf(tp, fp, fn)["f1"]


def paired_bootstrap(rows_a, rows_b, n_boot=2000, seed=0):
    """Phân phối F1(b) − F1(a) khi lấy mẫu lại CÙNG tập ảnh cho cả hai setup."""
    rng, n = random.Random(seed), len(rows_a)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(_f1(rows_b, idx) - _f1(rows_a, idx))
    diffs.sort()
    full = range(n)
    return {"diff": _f1(rows_b, full) - _f1(rows_a, full),
            "ci95": [diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot) - 1]],
            "p_b_not_better": sum(d <= 0 for d in diffs) / n_boot}


def session_bootstrap(entries, rows, n_boot=2000, seed=0):
    """Bootstrap theo cụm phiên quay: test chỉ có vài phiên nên CI theo ảnh lạc quan quá mức."""
    by = {}
    for i, e in enumerate(entries):
        by.setdefault(e["session"], []).append(i)
    keys = sorted(by)
    rng, vals = random.Random(seed), []
    for _ in range(n_boot):
        idx = [i for _ in keys for i in by[keys[rng.randrange(len(keys))]]]
        vals.append(_f1(rows, idx))
    vals.sort()
    per_session = {k: round(_f1(rows, by[k]), 4) for k in keys}
    return {"sessions": {k: len(v) for k, v in by.items()}, "f1_per_session": per_session,
            "ci95": [vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot) - 1]]}


def error_breakdown(entries, preds, coord="norm1000"):
    """FP: nhầm lớp / sai vị trí / bịa; FN: nhầm lớp / sai vị trí / bỏ sót hẳn (theo từng lớp)."""
    fp = {c: {"nham_lop": 0, "sai_vi_tri": 0, "bia": 0} for c in (*CLASSES, "other")}
    fn = {c: {"nham_lop": 0, "sai_vi_tri": 0, "bo_sot": 0} for c in CLASSES}
    for e in entries:
        gt = _gt_boxes(e)
        parsed = parse_output(preds.get(e["image_id"]))
        pred = _to_norm1000(parsed, coord, e["width"], e["height"]) if parsed is not None else []
        tp_p, tp_g = set(), set()
        for lab in {x["label"] for x in gt} | {p["label"] for p in pred}:
            gi = [i for i, x in enumerate(gt) if x["label"] == lab]
            pj = [j for j, x in enumerate(pred) if x["label"] == lab]
            pairs = sorted(((iou_xyxy(gt[i]["bbox"], pred[j]["bbox"]), i, j) for i in gi for j in pj),
                           reverse=True)
            for v, i, j in pairs:
                if v < 0.5:
                    break
                if i in tp_g or j in tp_p:
                    continue
                tp_g.add(i)
                tp_p.add(j)
        for j, p in enumerate(pred):
            if j in tp_p:
                continue
            key = p["label"] if p["label"] in CLASSES else "other"
            best_same = max((iou_xyxy(g["bbox"], p["bbox"]) for g in gt if g["label"] == p["label"]), default=0)
            best_other = max((iou_xyxy(g["bbox"], p["bbox"]) for g in gt if g["label"] != p["label"]), default=0)
            if best_other >= 0.5:
                fp[key]["nham_lop"] += 1
            elif best_same >= 0.1:
                fp[key]["sai_vi_tri"] += 1
            else:
                fp[key]["bia"] += 1
        for i, g in enumerate(gt):
            if i in tp_g:
                continue
            best_same = max((iou_xyxy(g["bbox"], p["bbox"]) for p in pred if p["label"] == g["label"]), default=0)
            best_other = max((iou_xyxy(g["bbox"], p["bbox"]) for p in pred if p["label"] != g["label"]), default=0)
            if best_other >= 0.5:
                fn[g["label"]]["nham_lop"] += 1
            elif best_same >= 0.1:
                fn[g["label"]]["sai_vi_tri"] += 1
            else:
                fn[g["label"]]["bo_sot"] += 1
    return {"fp": fp, "fn": fn}


def recall_by_size(entries, preds, coord="norm1000", edges=(0.25, 1.0, 4.0)):
    """Recall theo diện tích box đúng (% khung hình), từng lớp."""
    names = [f"<{edges[0]}%"] + [f"{a}-{b}%" for a, b in zip(edges, edges[1:])] + [f">={edges[-1]}%"]
    stat = {c: {n: [0, 0] for n in names} for c in CLASSES}
    for e in entries:
        gt = _gt_boxes(e)
        parsed = parse_output(preds.get(e["image_id"]))
        pred = _to_norm1000(parsed, coord, e["width"], e["height"]) if parsed is not None else []
        used = set()
        for i, g in sorted(enumerate(gt), key=lambda t: 0):
            area = (g["bbox"][2] - g["bbox"][0]) * (g["bbox"][3] - g["bbox"][1]) / 1e4
            b = sum(area >= x for x in edges)
            hit = None
            cands = sorted(((iou_xyxy(g["bbox"], p["bbox"]), j) for j, p in enumerate(pred)
                            if p["label"] == g["label"] and j not in used), reverse=True)
            if cands and cands[0][0] >= 0.5:
                hit = cands[0][1]
                used.add(hit)
            stat[g["label"]][names[b]][0] += hit is not None
            stat[g["label"]][names[b]][1] += 1
    return {c: {n: {"recall": round(h / t, 3) if t else None, "n": t} for n, (h, t) in v.items()}
            for c, v in stat.items()}


def pred_stats(entries, preds, coord="norm1000"):
    n_pred, n_gt, giant, fenced = 0, 0, 0, 0
    for e in entries:
        raw = preds.get(e["image_id"]) or ""
        fenced += "```" in raw
        parsed = parse_output(raw)
        n_gt += len(e["objects"])
        if parsed is None:
            continue
        pred = _to_norm1000(parsed, coord, e["width"], e["height"])
        n_pred += len(pred)
        giant += sum((p["bbox"][2] - p["bbox"][0]) >= 900 for p in pred)
    return {"pred_boxes_per_image": n_pred / len(entries), "gt_boxes_per_image": n_gt / len(entries),
            "full_width_boxes": giant, "fenced_outputs": fenced}


def table(evidence):
    head = ("| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | "
            "F1 pothole | F1 crack | F1 manhole | P | R | valid | mean IoU |")
    lines = [head, "|" + "---|" * 15]
    names = {"e0": "zero-shot", "e1": "LoRA LLM", "e2": "+ projector", "e3": "+ LoRA ViT",
             "e4_coord_norm1000": "E4 coord=norm1000", "e4_coord_pixel": "E4 coord=pixel", "e5": "GRPO",
             "nomask_e1": "E1 không mask loss (đối chứng câu 4)"}
    for k in ORDER:
        if k not in evidence:
            continue
        d = evidence[k]
        m, pc, meta = d["micro"], d["per_class"], d.get("meta", {})
        lines.append(
            f"| {d['name']} | {names.get(k, k)} | {meta.get('trainable_params', 0):,} | "
            f"{meta.get('peak_vram_gb', '–')} | {meta.get('train_minutes', '–')} | {m['f1']:.3f} | "
            f"[{d['f1_ci95'][0]:.3f}, {d['f1_ci95'][1]:.3f}] | {d['fp_per_image']:.2f} | "
            f"{pc['pothole']['f1']:.3f} | {pc['crack']['f1']:.3f} | {pc['manhole']['f1']:.3f} | "
            f"{m['precision']:.3f} | {m['recall']:.3f} | {d['parse_valid_rate']:.3f} | {d['mean_iou']:.3f} |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", default="work/preds")
    ap.add_argument("--evidence", default="report/evidence")
    ap.add_argument("--extra", default="report/extra", help="bằng chứng ngoài rubric (vd. không mask loss)")
    ap.add_argument("--out", default="report/analysis")
    a = ap.parse_args(argv)
    entries = load_split("test")
    files = [*Path(a.evidence).glob("e*.json"), *Path(a.extra).glob("*.json")]
    evidence = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in files}
    coord_of = {k: d.get("coord", "norm1000") for k, d in evidence.items()}
    preds, rows = {}, {}
    for k in ORDER:
        f = Path(a.preds) / f"{k}.jsonl"
        if f.exists():
            preds[k] = load_preds(f)
            rows[k] = per_image(entries, preds[k], coord_of.get(k, "norm1000"))
    res = {"table_md": table(evidence), "pred_stats": {}, "session_bootstrap": {}, "paired": {},
           "errors": {}, "recall_by_size": {}, "wrong_scale": {}}
    for k, p in preds.items():
        c = coord_of.get(k, "norm1000")
        res["pred_stats"][k] = pred_stats(entries, p, c)
        res["session_bootstrap"][k] = session_bootstrap(entries, rows[k])
        res["errors"][k] = error_breakdown(entries, p, c)
        res["recall_by_size"][k] = recall_by_size(entries, p, c)
    trained = [k for k in ("e1", "e2", "e3") if k in rows]
    best = max(trained, key=lambda k: evidence[k]["micro"]["f1"]) if trained else None
    res["best_sft"] = best
    for a_, b_ in (("e0", "e1"), ("e1", "e2"), ("e2", "e3"), ("e1", "e3"), (best, "e5"),
                   (best, "e4_coord_pixel"), ("e1", "nomask_e1")):
        if a_ in rows and b_ in rows:
            res["paired"][f"{b_}-{a_}"] = paired_bootstrap(rows[a_], rows[b_])
    # Câu 3: chấm cùng một file dự đoán với thang toạ độ SAI.
    for k in ("e0", best, "e4_coord_pixel"):
        if k in preds:
            wrong = "pixel" if coord_of.get(k, "norm1000") == "norm1000" else "norm1000"
            r = evaluate(entries, preds[k], coord=wrong, n_boot=200)
            res["wrong_scale"][k] = {"scored_as": wrong, "f1": r["micro"]["f1"], "mean_iou": r["mean_iou"],
                                     "right_f1": evidence[k]["micro"]["f1"] if k in evidence else None}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "table.md").write_text(res["table_md"] + "\n", encoding="utf-8")
    print(res["table_md"])
    print(json.dumps({k: v for k, v in res.items() if k != "table_md"}, indent=1, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    main()
