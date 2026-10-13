"""Vẽ ảnh minh hoạ dự đoán đúng/sai: box đúng (xanh lá, nét liền) và box dự đoán (TP xanh dương, FP đỏ).

  python -m lab.viz --pred work/preds/e2.jsonl --name E2 --out report/figures
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from lab.common import ROOT, load_split, read_jsonl
from scripts.score import IOU_THR, _gt_boxes, _to_norm1000, iou_xyxy, parse_output

GT_C, TP_C, FP_C, FN_C = (40, 200, 70), (40, 120, 255), (235, 40, 40), (255, 200, 0)


def match_image(gt, pred):
    """Ghép như score.py; trả (tp_pred_idx, fp_pred_idx, fn_gt_idx, confusions)."""
    tp, matched_g = set(), set()
    for lab in {x["label"] for x in gt} | {p["label"] for p in pred}:
        gi = [i for i, x in enumerate(gt) if x["label"] == lab]
        pi = [j for j, x in enumerate(pred) if x["label"] == lab]
        pairs = sorted(((iou_xyxy(gt[i]["bbox"], pred[j]["bbox"]), i, j) for i in gi for j in pi), reverse=True)
        for v, i, j in pairs:
            if v < IOU_THR:
                break
            if i in matched_g or j in tp:
                continue
            matched_g.add(i)
            tp.add(j)
    fp = [j for j in range(len(pred)) if j not in tp]
    fn = [i for i in range(len(gt)) if i not in matched_g]
    conf = [(gt[i]["label"], pred[j]["label"]) for i in fn for j in fp
            if iou_xyxy(gt[i]["bbox"], pred[j]["bbox"]) >= IOU_THR]
    return sorted(tp), fp, fn, conf


def _font(size):
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw(entry, raw, title, coord="norm1000", scale=2):
    W, H = entry["width"], entry["height"]
    img = Image.open(ROOT / "data" / entry["image_path"]).convert("RGB").resize((W * scale, H * scale))
    gt = _gt_boxes(entry)
    parsed = parse_output(raw)
    pred = _to_norm1000(parsed, coord, W, H) if parsed is not None else []
    tp, fp, fn, conf = match_image(gt, pred)
    d = ImageDraw.Draw(img)
    f = _font(13 * scale // 2 + 4)

    def px(b):
        return [b[0] / 1000 * W * scale, b[1] / 1000 * H * scale, b[2] / 1000 * W * scale, b[3] / 1000 * H * scale]

    for i, g in enumerate(gt):
        d.rectangle(px(g["bbox"]), outline=FN_C if i in fn else GT_C, width=2 * scale)
        x, y = px(g["bbox"])[:2]
        d.text((x + 2, max(0, y - 18)), f"GT {g['label']}", fill=FN_C if i in fn else GT_C, font=f)
    for j, p in enumerate(pred):
        c = TP_C if j in tp else FP_C
        b = px(p["bbox"])
        d.rectangle([b[0] + 3 * scale, b[1] + 3 * scale, b[2] - 3 * scale, b[3] - 3 * scale],
                    outline=c, width=scale)  # thụt vào trong để không che box đúng
        d.text((b[0] + 2, b[3] + 2), p["label"], fill=c, font=f)
    status = "JSON hỏng" if parsed is None else f"TP={len(tp)} FP={len(fp)} FN={len(fn)}"
    bar = Image.new("RGB", (img.width, 34 * scale // 2 + 20), (20, 20, 20))
    ImageDraw.Draw(bar).text((8, 6), f"{title} | {entry['image_id']} | {status}   "
                             "[GT khớp: xanh lá, GT bỏ sót: vàng | dự đoán TP: xanh dương, FP: đỏ]",
                             fill=(240, 240, 240), font=f)
    canvas = Image.new("RGB", (img.width, img.height + bar.height))
    canvas.paste(img, (0, 0))
    canvas.paste(bar, (0, img.height))
    return canvas, {"tp": len(tp), "fp": len(fp), "fn": len(fn), "conf": conf, "valid": parsed is not None,
                    "n_gt": len(gt)}


def pick_cases(entries, preds, coord="norm1000"):
    """Chọn ảnh tiêu biểu: đúng hoàn toàn, bỏ sót, bịa box, nhầm lớp, JSON hỏng."""
    rows = []
    for e in entries:
        raw = preds.get(e["image_id"], "")
        parsed = parse_output(raw)
        pred = _to_norm1000(parsed, coord, e["width"], e["height"]) if parsed is not None else []
        tp, fp, fn, conf = match_image(_gt_boxes(e), pred)
        rows.append((e, len(tp), len(fp), len(fn), conf, parsed is None))
    cases = []

    def take(kind, cond, key, n):
        got = [r for r in rows if cond(r) and r[0]["image_id"] not in {c[1]["image_id"] for c in cases}]
        for r in sorted(got, key=key)[:n]:
            cases.append((kind, r[0]))

    take("dung_hoan_toan", lambda r: r[2] == 0 and r[3] == 0 and r[1] >= 2, lambda r: -r[1], 2)
    take("bo_sot", lambda r: r[3] >= 2 and r[2] <= 1, lambda r: -r[3], 2)
    take("bia_box", lambda r: r[2] >= 2, lambda r: -r[2], 2)
    take("nham_lop", lambda r: bool(r[4]), lambda r: -len(r[4]), 1)
    take("json_hong", lambda r: r[5], lambda r: 0, 1)
    return cases


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="report/figures")
    ap.add_argument("--coord", default="norm1000")
    ap.add_argument("--ids", nargs="*", help="vẽ đúng các image_id này thay vì tự chọn")
    a = ap.parse_args(argv)
    entries = load_split("test")
    preds = {r["image_id"]: r["raw"] for r in read_jsonl(a.pred)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.ids:
        by_id = {e["image_id"]: e for e in entries}
        cases = [("chon", by_id[i]) for i in a.ids]
    else:
        cases = pick_cases(entries, preds, a.coord)
    index = []
    for k, (kind, e) in enumerate(cases, 1):
        img, info = draw(e, preds.get(e["image_id"], ""), a.name, a.coord)
        fn = out / f"{a.name.lower()}_{k:02d}_{kind}_{e['image_id']}.jpg"
        img.save(fn, quality=88)
        index.append({"file": fn.name, "kind": kind, "image_id": e["image_id"], **info,
                      "raw": preds.get(e["image_id"], "")[:400]})
        print(fn.name, info)
    (out / f"{a.name.lower()}_index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False),
                                                      encoding="utf-8")


if __name__ == "__main__":
    main()
