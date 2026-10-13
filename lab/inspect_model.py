"""Kiểm tra bước 2, 4, 5 của GUIDE: nhãn ↔ pixel, 220 token ảnh, mask loss, số module LoRA, tham số.

  python -m lab.inspect_model --cpu   # chỉ processor + model trên meta device (không tải trọng số)
  python -m lab.inspect_model         # trên GPU: model 4-bit thật
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from lab.common import MODEL_ID, ROOT, box_to_pixels, build_prompt, load_split, objects_to_answer
from lab.data import IGNORE, encode_example
from lab.lora_setup import LLM_TARGETS, SETUPS, matched_modules, target_regex


def check_roundtrip(entries):
    """YOLO → JSON 0–1000 → pixel phải trùng box gốc đã kẹp vào khung (sai số làm tròn ≤ 0,32 px)."""
    from lab.common import yolo_to_box
    worst, degenerate, out_of_frame = 0.0, 0, 0
    for e in entries:
        W, H = e["width"], e["height"]
        for o in e["objects"]:
            xc, yc, w, h = o["yolo"]
            degenerate += w == 0 or h == 0
            out_of_frame += xc - w / 2 < 0 or yc - h / 2 < 0 or xc + w / 2 > 1 or yc + h / 2 > 1
            ref = (max(0, (xc - w / 2) * W), max(0, (yc - h / 2) * H),
                   min(W, (xc + w / 2) * W), min(H, (yc + h / 2) * H))
            back = box_to_pixels(yolo_to_box(o["label"], o["yolo"])["bbox_2d"], "norm1000", W, H)
            worst = max(worst, max(abs(a - b) for a, b in zip(back, ref)))
    return {"max_px_error": round(worst, 3), "zero_area_boxes": degenerate,
            "out_of_frame_boxes": out_of_frame}


def check_encoding(processor, entry, coord="norm1000"):
    image = Image.open(ROOT / "data" / entry["image_path"]).convert("RGB")
    prompt = build_prompt(coord, entry["width"], entry["height"])
    answer = objects_to_answer(entry["objects"], coord, entry["width"], entry["height"])
    ex = encode_example(processor, image, prompt, answer)
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    kept = ex["labels"][ex["labels"] != IGNORE]
    t, h, w = ex["image_grid_thw"][0].tolist()
    merge = processor.image_processor.merge_size
    return {
        "image_grid_thw": [t, h, w],
        "n_image_tokens": int((ex["input_ids"] == image_token_id).sum()),
        "n_image_tokens_from_grid": t * h * w // merge ** 2,
        "seq_len": int(ex["input_ids"].shape[0]),
        "n_answer_tokens": int(kept.numel()),
        "answer": answer,
        "decoded_labels": processor.tokenizer.decode(kept),
        "full_text": processor.tokenizer.decode(ex["input_ids"]),
        "keys": sorted(ex),
    }


def answer_length_stats(processor, coord="norm1000"):
    out = {}
    for split in ("train", "val", "test"):
        lens = sorted(len(processor.tokenizer(objects_to_answer(e["objects"], coord))["input_ids"]) + 1
                      for e in load_split(split))
        nbox = sorted(len(e["objects"]) for e in load_split(split))
        out[split] = {"max_tokens": lens[-1], "p99_tokens": lens[int(0.99 * (len(lens) - 1))],
                      "mean_tokens": sum(lens) / len(lens), "max_boxes": nbox[-1],
                      "images_over_256_tokens": sum(v > 256 for v in lens)}
    return out


def block_params(model):
    """Tham số theo khối, tính trên model KHÔNG lượng tử (meta device) để ra số thật."""
    out = {}
    for name, p in model.named_parameters():
        if "visual.merger" in name:
            k = "projector.merger"
        elif "visual.deepstack_merger_list" in name:
            k = "projector.deepstack_merger_list"
        elif "visual" in name:
            k = "vit (patch_embed+pos_embed+blocks)"
        elif "lm_head" in name:
            k = "lm_head"
        elif "embed_tokens" in name:
            k = "llm.embed_tokens"
        else:
            k = "llm.layers+norm"
        out[k] = out.get(k, 0) + p.numel()
    out["total"] = sum(out.values())
    return out


def lora_params(model, setup, r=16):
    total = 0
    for n in matched_modules(model, target_regex(setup)):
        m = model.get_submodule(n)
        total += r * (m.in_features + m.out_features)
    return total


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", default="work/inspect.json")
    a = ap.parse_args(argv)
    from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    train = load_split("train")
    res = {"roundtrip": {s: check_roundtrip(load_split(s)) for s in ("train", "val", "test")},
           "encoding": check_encoding(processor, next(e for e in train if len(e["objects"]) >= 2)),
           "encoding_empty": check_encoding(processor, next(e for e in train if not e["objects"]))
           if any(not e["objects"] for e in train) else None,
           "answer_lengths": answer_length_stats(processor)}

    cfg = AutoConfig.from_pretrained(MODEL_ID)
    with torch.device("meta"):
        meta = Qwen3VLForConditionalGeneration._from_config(cfg)
    res["block_params"] = block_params(meta)
    res["tie_word_embeddings"] = bool(getattr(cfg, "tie_word_embeddings", False)
                                      or getattr(cfg.text_config, "tie_word_embeddings", False))
    res["lora_modules"] = {s: len(matched_modules(meta, target_regex(s))) for s in SETUPS}
    res["lora_params_r16"] = {s: lora_params(meta, s) for s in SETUPS}
    res["lora_params_by_r"] = {r: lora_params(meta, "E1", r) for r in (4, 16, 64)}
    res["llm_module_example"] = matched_modules(meta, LLM_TARGETS)[:7]
    res["visual_tree_top"] = sorted({".".join(n.split(".")[:4]) for n, _ in meta.named_modules()
                                     if ".visual." in f".{n}"})[:40]
    del meta

    if not a.cpu:
        from lab.common import load_model_and_processor
        model, _ = load_model_and_processor()
        res["gpu_dtypes"] = {
            "visual.blocks.0.attn.qkv": str(model.model.visual.blocks[0].attn.qkv.weight.dtype),
            "visual.merger.linear_fc1": str(model.model.visual.merger.linear_fc1.weight.dtype),
            "lm_head": str(model.lm_head.weight.dtype),
            "llm q_proj": type(model.model.language_model.layers[0].self_attn.q_proj).__name__,
        }
        res["gpu_memory_after_load_gb"] = round(torch.cuda.memory_allocated() / 2**30, 2)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
