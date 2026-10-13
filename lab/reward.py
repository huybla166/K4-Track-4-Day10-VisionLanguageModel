"""Reward IoU kiểm chứng được cho GRPO (GUIDE bước 9)."""
from scripts.score import IOU_THR, iou_xyxy, parse_output


def gt_from_objects(objects):
    """objects YOLO của split → [{"label", "bbox"}] trên thang 0–1000 (giống score._gt_boxes)."""
    out = []
    for o in objects:
        xc, yc, w, h = o["yolo"]
        out.append({"label": o["label"], "bbox": ((xc - w / 2) * 1000, (yc - h / 2) * 1000,
                                                  (xc + w / 2) * 1000, (yc + h / 2) * 1000)})
    return out


def match_iou(gt_boxes, pred):
    """Ghép tham lam theo IoU giảm dần, cùng lớp, chỉ nhận IoU ≥ 0.5. Trả list IoU các cặp."""
    ious = []
    for lab in {g["label"] for g in gt_boxes} | {p["label"] for p in pred}:
        g = [x["bbox"] for x in gt_boxes if x["label"] == lab]
        p = [x["bbox"] for x in pred if x["label"] == lab]
        pairs = sorted(((iou_xyxy(a, b), i, j) for i, a in enumerate(g) for j, b in enumerate(p)),
                       reverse=True)
        used_g, used_p = set(), set()
        for v, i, j in pairs:
            if v < IOU_THR:
                break
            if i in used_g or j in used_p:
                continue
            used_g.add(i)
            used_p.add(j)
            ious.append(v)
    return ious


def compute_reward(completion, gt_boxes):
    """0 nếu JSON hỏng; 1 nếu cả hai rỗng; còn lại 0.2 + 0.8·ΣIoU / max(n_gt, n_pred, 1)."""
    pred = parse_output(completion)
    if pred is None:
        return 0.0
    if not gt_boxes and not pred:
        return 1.0
    total = sum(match_iou(gt_boxes, pred))
    return 0.2 + 0.8 * total / max(len(gt_boxes), len(pred), 1)


def completion_text(c):
    """TRL truyền completion dạng chuỗi hoặc list message hội thoại."""
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c and isinstance(c[-1], dict):
        content = c[-1].get("content", "")
        if isinstance(content, list):
            return "".join(x.get("text", "") for x in content if isinstance(x, dict))
        return content
    return str(c)


def iou_reward(completions, gt, **kwargs):
    """Hàm reward theo giao diện trl.GRPOTrainer: cột `gt` của dataset là list box 0–1000."""
    out = []
    for c, g in zip(completions, gt):
        boxes = [{"label": b["label"], "bbox": tuple(b["bbox"])} for b in g]
        out.append(compute_reward(completion_text(c), boxes))
    return out
