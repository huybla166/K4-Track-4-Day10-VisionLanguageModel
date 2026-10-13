"""Hằng số và hàm dùng chung: prompt, chuyển nhãn YOLO → JSON grounding, nạp model 4-bit."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.labcommon import CLASSES, read_jsonl, write_jsonl  # noqa: E402,F401

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
COORDS = ("norm1000", "pixel")

# Tên tham số của projector (merger chính + 3 merger DeepStack lấy đặc trưng ViT lớp 5, 11, 17).
PROJECTOR_KEYS = ("visual.merger", "visual.deepstack_merger_list")

_PROMPTS = {
    "norm1000": (
        "Locate every road damage instance in this dashcam image. "
        "Classes: pothole, crack, manhole. "
        "Answer with ONLY a JSON list in the form "
        '[{{"label": "pothole", "bbox_2d": [x1, y1, x2, y2]}}], '
        "where x1, y1, x2, y2 are integers on a 0-1000 scale relative to the image width and height "
        "(0,0 is the top-left corner, 1000,1000 is the bottom-right corner). "
        "If there is no damage, answer []."
    ),
    "pixel": (
        "Locate every road damage instance in this dashcam image. "
        "Classes: pothole, crack, manhole. "
        "Answer with ONLY a JSON list in the form "
        '[{{"label": "pothole", "bbox_2d": [x1, y1, x2, y2]}}], '
        "where x1, y1, x2, y2 are integer pixel coordinates in the {width}x{height} image "
        "(x from 0 to {width}, y from 0 to {height}, 0,0 is the top-left corner). "
        "If there is no damage, answer []."
    ),
}


def build_prompt(coord="norm1000", width=640, height=360):
    return _PROMPTS[coord].format(width=width, height=height)


def _scale(coord, width, height):
    if coord == "norm1000":
        return 1000, 1000
    if coord == "pixel":
        return width, height
    raise ValueError(f"unknown coord {coord!r}")


def yolo_to_box(label, yolo, coord="norm1000", width=640, height=360):
    """YOLO (tâm + kích thước, chuẩn hoá [0,1]) → {"label", "bbox_2d": [x1,y1,x2,y2]} nguyên."""
    xc, yc, w, h = yolo
    sx, sy = _scale(coord, width, height)

    def clamp(v, hi):
        return int(min(max(round(v), 0), hi))

    x1, y1 = clamp((xc - w / 2) * sx, sx), clamp((yc - h / 2) * sy, sy)
    x2, y2 = clamp((xc + w / 2) * sx, sx), clamp((yc + h / 2) * sy, sy)
    # score.parse_output loại CẢ ảnh nếu một box có x2 <= x1: không bao giờ dạy model box suy biến.
    if x2 <= x1:
        x1, x2 = (x1, x1 + 1) if x1 < sx else (sx - 1, sx)
    if y2 <= y1:
        y1, y2 = (y1, y1 + 1) if y1 < sy else (sy - 1, sy)
    return {"label": label, "bbox_2d": [x1, y1, x2, y2]}


def objects_to_boxes(objects, coord="norm1000", width=640, height=360):
    boxes = [yolo_to_box(o["label"], o["yolo"], coord, width, height) for o in objects]
    boxes.sort(key=lambda b: (b["bbox_2d"][1], b["bbox_2d"][0]))  # trên→dưới, trái→phải
    return boxes


def objects_to_answer(objects, coord="norm1000", width=640, height=360):
    return json.dumps(objects_to_boxes(objects, coord, width, height), separators=(",", ":"))


def box_to_pixels(bbox, coord, width, height):
    sx, sy = _scale(coord, width, height)
    x1, y1, x2, y2 = bbox
    return (x1 / sx * width, y1 / sy * height, x2 / sx * width, y2 / sy * height)


def load_split(split, splits_dir=None):
    splits_dir = Path(splits_dir or ROOT / "data" / "splits")
    return read_jsonl(splits_dir / f"{split}.jsonl")


def load_model_and_processor(model_id=MODEL_ID, device_map=None):
    """Qwen3-VL-4B: LLM lượng tử hoá NF4; ViT (gồm projector) và lm_head giữ fp16."""
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    # transformers 5.x so khớp skip-module bằng re.match từ ĐẦU tên đầy đủ ("model.visual.blocks.0...")
    # nên "visual" một mình KHÔNG khớp → ViT và projector bị lượng tử hoá 4-bit. Phải ghi "model.visual".
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True, llm_int8_skip_modules=["model.visual", "visual", "lm_head"])
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, quantization_config=quant, dtype=torch.float16,
        device_map=device_map or {"": 0})
    assert_vit_not_quantized(model)
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    return model, processor


def assert_vit_not_quantized(model):
    """ViT + projector + lm_head phải ở fp16; nếu là Linear4bit thì projector không huấn luyện đầy đủ được."""
    bad = [n for n, m in model.named_modules()
           if type(m).__name__ in ("Linear4bit", "Linear8bitLt") and (".visual." in f".{n}" or "lm_head" in n)]
    if bad:
        raise RuntimeError(f"{len(bad)} module ViT/lm_head bị lượng tử hoá, ví dụ {bad[:3]}: "
                           "kiểm tra llm_int8_skip_modules")


def count_params(model):
    """Đếm tham số theo khối. Với Linear4bit, numel() trả số phần tử đã đóng gói → nhân 2."""
    out = {"visual.blocks+patch_embed": 0, "visual.merger": 0, "visual.deepstack_merger_list": 0,
           "language_model": 0, "lm_head": 0}
    for name, p in model.named_parameters():
        n = p.numel() * (2 if p.dtype.itemsize == 1 and p.__class__.__name__ == "Params4bit" else 1)
        if "visual.merger" in name:
            out["visual.merger"] += n
        elif "visual.deepstack_merger_list" in name:
            out["visual.deepstack_merger_list"] += n
        elif ".visual." in f".{name}":
            out["visual.blocks+patch_embed"] += n
        elif "lm_head" in name:
            out["lm_head"] += n
        else:
            out["language_model"] += n
    out["total"] = sum(out.values())
    return out
