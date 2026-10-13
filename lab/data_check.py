"""Kiểm tra GUIDE bước 1: số ảnh/box từng split, không phiên nào nằm ở 2 split, vẽ 6 ảnh có box.

  python -m lab.data_check --out report/figures
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from lab.common import CLASSES, ROOT, load_split

COLORS = {"pothole": (235, 104, 52), "crack": (42, 120, 214), "manhole": (27, 175, 122)}


def stats():
    out, sessions = {}, {}
    for s in ("train", "val", "test"):
        rows = load_split(s)
        boxes = Counter(o["label"] for r in rows for o in r["objects"])
        areas = sorted(o["yolo"][2] * o["yolo"][3] for r in rows for o in r["objects"])
        out[s] = {"images": len(rows), "boxes": {c: boxes[c] for c in CLASSES},
                  "empty_images": sum(not r["objects"] for r in rows),
                  "sessions": len({r["session"] for r in rows}),
                  "median_box_area_pct": round(100 * areas[len(areas) // 2], 3)}
        sessions[s] = {r["session"] for r in rows}
    overlap = {f"{a}&{b}": sorted(sessions[a] & sessions[b])
               for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    return out, overlap


def grid(entries, out_path, cols=3, scale=1):
    tiles = []
    for e in entries:
        im = Image.open(ROOT / "data" / e["image_path"]).convert("RGB")
        d = ImageDraw.Draw(im)
        W, H = im.size
        for o in e["objects"]:
            xc, yc, w, h = o["yolo"]
            b = [(xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H]
            d.rectangle(b, outline=COLORS[o["label"]], width=2)
            d.text((b[0] + 2, max(0, b[1] - 11)), o["label"], fill=COLORS[o["label"]])
        d.rectangle([0, H - 14, W, H], fill=(20, 20, 20))
        d.text((4, H - 13), f"{e['image_id']} ({e['session']})", fill=(240, 240, 240))
        tiles.append(im)
    W, H = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * W, rows * H), (255, 255, 255))
    for i, t in enumerate(tiles):
        canvas.paste(t, ((i % cols) * W, (i // cols) * H))
    canvas.save(out_path, quality=88)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="report/figures")
    a = ap.parse_args(argv)
    st, overlap = stats()
    print(json.dumps({"splits": st, "session_overlap": overlap}, indent=2, ensure_ascii=False))
    assert not any(overlap.values()), "có phiên quay nằm ở hai split!"
    train = load_split("train")
    # 6 ảnh từ 6 phiên khác nhau, ưu tiên ảnh có đủ nhiều lớp
    picked, seen = [], set()
    for e in sorted(train, key=lambda e: (-len({o["label"] for o in e["objects"]}), e["image_id"])):
        if e["session"] not in seen:
            picked.append(e)
            seen.add(e["session"])
        if len(picked) == 6:
            break
    Path(a.out).mkdir(parents=True, exist_ok=True)
    grid(picked, Path(a.out) / "data_gt_samples.jpg")
    (Path(a.out) / "data_stats.json").write_text(
        json.dumps({"splits": st, "session_overlap": overlap}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
