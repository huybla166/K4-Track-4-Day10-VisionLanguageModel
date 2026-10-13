"""Câu 6: đo rò rỉ giữa các frame liền kề bằng một baseline KHÔNG học gì (chạy CPU, vài chục giây).

  python -m lab.leakage --out report/analysis/leakage.json

Baseline "chép nhãn láng giềng gần nhất": với mỗi ảnh test, tìm ảnh train giống nhất (ảnh xám thu nhỏ,
khoảng cách L2) rồi chép nguyên box của nó làm dự đoán; chấm bằng đúng scripts.score.evaluate.
- Split chuẩn (theo phiên quay): ảnh train không cùng cung đường → chép nhãn gần như vô dụng.
- Split ngẫu nhiên cùng kích thước (train 1.422 / test 311, gộp cả 2.009 ảnh): frame liền kề của cùng
  phiên rơi vào train → chép nhãn "trúng" → F1 cao giả, dù baseline không hề nhìn thấy ổ gà.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from lab.common import ROOT, load_split, objects_to_answer
from scripts.score import evaluate

THUMB = (64, 36)  # giữ tỉ lệ 16:9 của ảnh 640×360


def features(entries):
    feats = []
    for e in entries:
        im = Image.open(ROOT / "data" / e["image_path"]).convert("L").resize(THUMB, Image.BILINEAR)
        v = np.asarray(im, dtype=np.float32).ravel()
        v = (v - v.mean()) / (v.std() + 1e-6)  # bỏ khác biệt độ sáng tổng thể
        feats.append(v)
    return np.stack(feats)


def nn_copy(train_idx, test_idx, feats, entries):
    """Dự đoán = box của ảnh train gần nhất. Trả (preds, khoảng cách NN, tỉ lệ NN cùng phiên)."""
    tr = feats[train_idx]
    preds, dists, same = {}, [], 0
    for i in test_idx:
        d = ((tr - feats[i]) ** 2).mean(axis=1)
        j = train_idx[int(d.argmin())]
        dists.append(float(d.min()))
        same += entries[j]["session"] == entries[i]["session"]
        preds[entries[i]["image_id"]] = objects_to_answer(entries[j]["objects"])
    return preds, dists, same / len(test_idx)


def run(name, train_idx, test_idx, feats, entries, dup_thr):
    preds, dists, same = nn_copy(train_idx, test_idx, feats, entries)
    r = evaluate([entries[i] for i in test_idx], preds, n_boot=300)
    return {"split": name, "n_train": len(train_idx), "n_test": len(test_idx),
            "f1": round(r["micro"]["f1"], 4), "f1_ci95": [round(x, 4) for x in r["f1_ci95"]],
            "precision": round(r["micro"]["precision"], 4), "recall": round(r["micro"]["recall"], 4),
            "per_class_f1": {c: round(v["f1"], 4) for c, v in r["per_class"].items() if c != "other"},
            "nn_same_session_frac": round(same, 4),
            "near_duplicate_frac": round(float(np.mean(np.array(dists) < dup_thr)), 4),
            "nn_dist_median": round(float(np.median(dists)), 4)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="report/analysis/leakage.json")
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args(argv)

    splits = {s: load_split(s) for s in ("train", "val", "test")}
    entries = splits["train"] + splits["val"] + splits["test"]
    feats = features(entries)
    n_tr, n_va = len(splits["train"]), len(splits["val"])
    train_idx = list(range(n_tr))
    test_idx = list(range(n_tr + n_va, len(entries)))

    # Ngưỡng "gần trùng": phân vị 25% khoảng cách giữa hai frame LIỀN KỀ trong cùng phiên (theo tên file).
    by_sess = {}
    for i, e in enumerate(entries):
        by_sess.setdefault(e["session"], []).append(i)
    adj = []
    for idx in by_sess.values():
        order = sorted(idx, key=lambda k: entries[k]["image_id"])
        adj += [float(((feats[i] - feats[j]) ** 2).mean()) for i, j in zip(order, order[1:])]
    dup_thr = float(np.percentile(adj, 25))

    res = {"thumb": THUMB, "near_duplicate_threshold": round(dup_thr, 4),
           "adjacent_frame_dist_median": round(float(np.median(adj)), 4),
           "session_split": run("session (chuẩn)", train_idx, test_idx, feats, entries, dup_thr),
           "random_split": []}
    for seed in range(a.seeds):
        perm = list(range(len(entries)))
        random.Random(seed).shuffle(perm)
        res["random_split"].append(run(f"random seed={seed}", perm[:n_tr], perm[-len(test_idx):], feats,
                                       entries, dup_thr))
    rs = res["random_split"]
    res["random_split_mean_f1"] = round(sum(r["f1"] for r in rs) / len(rs), 4)
    res["random_split_mean_near_dup"] = round(sum(r["near_duplicate_frac"] for r in rs) / len(rs), 4)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
