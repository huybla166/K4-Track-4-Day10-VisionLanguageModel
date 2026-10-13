"""Suy luận grounding trên một split → predictions.jsonl (GUIDE bước 3 và 7).

E0:    python -m lab.infer --out work/e0.jsonl
E1–E3: python -m lab.infer --run-dir work/runs/e2 --out work/e2.jsonl
Nạp model gốc 4-bit → adapter LoRA → projector đã huấn luyện (nếu run có projector.pt).
"""
import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

from lab.common import MODEL_ID, ROOT, build_prompt, load_model_and_processor, load_split
from lab.data import user_message
from lab.lora_setup import load_projector


def load_finetuned(run_dir=None, model_id=MODEL_ID, adapter_dir=None):
    model, processor = load_model_and_processor(model_id)
    if run_dir or adapter_dir:
        from peft import PeftModel
        adapter_dir = Path(adapter_dir or Path(run_dir) / "adapter")
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        proj = Path(run_dir) / "projector.pt" if run_dir else None
        if proj and proj.exists():
            n = load_projector(model, torch.load(proj, map_location="cpu"))
            print(f"đã nạp lại {n} tensor projector từ {proj}", flush=True)
    model.eval()
    model.config.use_cache = True
    return model, processor


@torch.no_grad()
def generate(model, processor, images, prompts, max_new_tokens=256):
    msgs = [[user_message(im, p)] for im, p in zip(images, prompts)]
    inputs = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt",
                                           processor_kwargs={"padding": True})
    inputs = inputs.to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                         temperature=None, top_p=None, top_k=None)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--adapter-dir", default=None)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--coord", default="norm1000", choices=["norm1000", "pixel"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model-id", default=MODEL_ID)
    a = ap.parse_args(argv)

    entries = load_split(a.split)
    if a.limit:
        entries = entries[:a.limit]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():  # chạy tiếp nếu bị ngắt giữa chừng
        done = {json.loads(line)["image_id"] for line in out.read_text(encoding="utf-8").splitlines()
                if line.strip()}
    todo = [e for e in entries if e["image_id"] not in done]
    print(f"{a.split}: {len(entries)} ảnh, còn {len(todo)}", flush=True)

    model, processor = load_finetuned(a.run_dir, a.model_id, a.adapter_dir)
    t0 = time.time()
    with open(out, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), a.batch_size):
            chunk = todo[i:i + a.batch_size]
            images = [Image.open(ROOT / "data" / e["image_path"]).convert("RGB") for e in chunk]
            prompts = [build_prompt(a.coord, e["width"], e["height"]) for e in chunk]
            raws = generate(model, processor, images, prompts, a.max_new_tokens)
            for e, raw in zip(chunk, raws):
                f.write(json.dumps({"image_id": e["image_id"], "raw": raw}, ensure_ascii=False) + "\n")
            f.flush()
            if i == 0:
                print("ví dụ output:", repr(raws[0][:300]), flush=True)
            print(f"  {min(i + a.batch_size, len(todo))}/{len(todo)} "
                  f"{(time.time() - t0) / 60:.1f}min", flush=True)
    print(f"infer_minutes={(time.time() - t0) / 60:.2f} peak_vram="
          f"{torch.cuda.max_memory_allocated() / 2**30:.2f}GB", flush=True)


if __name__ == "__main__":
    main()
