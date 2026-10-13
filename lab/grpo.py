"""E5 (tuỳ chọn): GRPO với reward IoU, khởi tạo từ adapter SFT tốt nhất (GUIDE bước 9, configs/e5.yaml).

  python -m lab.grpo --init-run work/runs/e2 --out work/runs/e5
Sinh: <out>/adapter/, <out>/projector.pt (chép từ run gốc, projector đóng băng), <out>/run.json,
      <out>/grpo_log.json (log_history của Trainer), <out>/reward_stats.jsonl (theo dõi reward hacking).
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from datasets import Dataset
from PIL import Image

from lab.common import ROOT, build_prompt, load_model_and_processor, load_split, objects_to_answer
from lab.data import subsample
from lab.lora_setup import load_projector
from lab.reward import compute_reward, completion_text, gt_from_objects
from scripts.score import parse_output


class RewardWithStats:
    """Hàm reward cho GRPOTrainer + ghi thống kê để phát hiện reward hacking ([] hoặc box khổng lồ)."""
    __name__ = "iou_reward"

    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.calls = 0

    def __call__(self, completions, gt, **kwargs):
        rewards, n_boxes, empty, invalid, giant, n_gt = [], [], 0, 0, 0, []
        for c, g in zip(completions, gt):
            text = completion_text(c)
            boxes = [{"label": b["label"], "bbox": tuple(b["bbox"])} for b in json.loads(g)]
            rewards.append(compute_reward(text, boxes))
            pred = parse_output(text)
            n_gt.append(len(boxes))
            if pred is None:
                invalid += 1
                continue
            n_boxes.append(len(pred))
            empty += not pred
            giant += any((p["bbox"][2] - p["bbox"][0]) * (p["bbox"][3] - p["bbox"][1]) > 0.25e6
                         for p in pred)
        self.calls += 1
        n = len(rewards)
        self.f.write(json.dumps({
            "call": self.calls, "n": n, "reward_mean": sum(rewards) / n,
            "invalid_frac": invalid / n, "empty_frac": empty / n, "giant_box_frac": giant / n,
            "pred_boxes_mean": sum(n_boxes) / max(len(n_boxes), 1), "gt_boxes_mean": sum(n_gt) / n,
            "sample": completion_text(completions[0])[:200]}) + "\n")
        self.f.flush()
        return rewards


def make_dataset(entries, processor, max_answer_tokens):
    rows = []
    for e in entries:
        ans = objects_to_answer(e["objects"])
        if len(processor.tokenizer(ans)["input_ids"]) + 1 > max_answer_tokens:
            continue  # đáp án dài hơn max_completion_length: luôn bị cắt → JSON hỏng, phạt oan
        gt = [{"label": b["label"], "bbox": list(b["bbox"])} for b in gt_from_objects(e["objects"])]
        rows.append({"prompt": [{"role": "user", "content": build_prompt("norm1000", e["width"], e["height"])}],
                     "image": Image.open(ROOT / "data" / e["image_path"]).convert("RGB"),
                     "gt": json.dumps(gt), "image_id": e["image_id"]})
    return Dataset.from_list(rows)


def main(argv=None):
    from peft import PeftModel
    from trl import GRPOConfig, GRPOTrainer

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init-run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--max-completion-length", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8, help="số completion mỗi micro-batch")
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    out, init = Path(a.out), Path(a.init_run)
    out.mkdir(parents=True, exist_ok=True)

    model, processor = load_model_and_processor()
    model = PeftModel.from_pretrained(model, str(init / "adapter"), is_trainable=True)
    if (init / "projector.pt").exists():  # projector của SFT được giữ nguyên (đóng băng) trong GRPO
        load_projector(model, torch.load(init / "projector.pt", map_location="cpu"))
        shutil.copy(init / "projector.pt", out / "projector.pt")
    processor.tokenizer.padding_side = "left"

    entries = subsample(load_split("train"), 1.0)
    if a.limit:
        entries = entries[:a.limit]
    ds = make_dataset(entries, processor, a.max_completion_length)
    print(f"GRPO prompts: {len(ds)} / {len(entries)} ảnh train (lọc đáp án ≤ {a.max_completion_length} token)",
          flush=True)

    cfg = GRPOConfig(
        output_dir=str(out / "trainer"), max_steps=a.max_steps, learning_rate=a.lr, beta=a.beta,
        num_generations=a.num_generations, max_completion_length=a.max_completion_length,
        per_device_train_batch_size=a.batch_size, gradient_accumulation_steps=a.grad_accum,
        fp16=True, bf16=False, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        lr_scheduler_type="constant", warmup_steps=0, max_grad_norm=1.0, weight_decay=0.0,
        mask_truncated_completions=True, logging_steps=1, save_strategy="no", report_to="none",
        seed=a.seed, log_completions=True, num_completions_to_print=2, use_vllm=False,
        remove_unused_columns=False, dataloader_num_workers=0)
    reward = RewardWithStats(out / "reward_stats.jsonl")
    trainer = GRPOTrainer(model=model, reward_funcs=[reward], args=cfg, train_dataset=ds,
                          processing_class=processor)
    # TRL ép tham số huấn luyện của model lượng tử về bf16; với lr 5e-6 cập nhật sẽ bị làm tròn mất → về fp32.
    n_cast = 0
    for p in trainer.model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
            n_cast += 1
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"trainable={trainable:,} (ép {n_cast} tensor về fp32)", flush=True)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    minutes = (time.time() - t0) / 60
    trainer.model.save_pretrained(out / "adapter")
    (out / "grpo_log.json").write_text(json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")
    run = {"setup": "E5-GRPO", "init_run": str(init), "trainable_params": trainable,
           "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
           "train_minutes": round(minutes, 2), "train_prompts": len(ds),
           **{k: v for k, v in vars(a).items() if k not in ("out", "init_run")},
           "gpu": torch.cuda.get_device_name(0)}
    (out / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(json.dumps(run), flush=True)


if __name__ == "__main__":
    main()
