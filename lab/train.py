"""Huấn luyện QLoRA cho E1/E2/E3 (và biến thể E4) theo GUIDE bước 6.

Ví dụ:  python -m lab.train --setup E2 --out work/runs/e2
Sinh:   <out>/adapter/ (LoRA), <out>/projector.pt (E2/E3), <out>/run.json, <out>/train_log.jsonl
"""
import argparse
import json
import math
import platform
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from lab.common import MODEL_ID, load_model_and_processor, load_split
from lab.data import GroundingDataset, make_collate, subsample
from lab.lora_setup import SETUPS, apply_setup, projector_state


def infinite(loader):
    while True:
        yield from loader


def save(model, out, setup):
    model.save_pretrained(out / "adapter")  # peft: CHỈ lưu adapter LoRA
    if SETUPS[setup]["train_projector"]:
        torch.save(projector_state(model), out / "projector.pt")  # projector lưu riêng


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--setup", required=True, choices=sorted(SETUPS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--train-fraction", type=float, default=1.0)
    ap.add_argument("--coord", default="norm1000", choices=["norm1000", "pixel"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed)
    torch.manual_seed(a.seed)

    t_load = time.time()
    model, processor = load_model_and_processor(a.model_id)
    model, info = apply_setup(model, a.setup, r=a.r, alpha=a.alpha, dropout=a.dropout)
    print(json.dumps(info), flush=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    processor.tokenizer.padding_side = "right"

    entries = subsample(load_split("train"), a.train_fraction, seed=a.seed)
    ds = GroundingDataset(entries, processor, coord=a.coord)
    g = torch.Generator().manual_seed(a.seed)
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=True, drop_last=True, generator=g,
                        collate_fn=make_collate(processor.tokenizer.pad_token_id),
                        num_workers=a.workers, persistent_workers=a.workers > 0)
    batches = infinite(loader)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, max(1, round(a.warmup_frac * a.max_steps)),
                                            a.max_steps)
    scaler = torch.amp.GradScaler("cuda")
    dev = next(p.device for p in params)
    print(f"train images={len(entries)} trainable={info['trainable_params']:,} "
          f"load_minutes={(time.time() - t_load) / 60:.1f}", flush=True)

    model.train()
    torch.cuda.reset_peak_memory_stats()
    log = open(out / "train_log.jsonl", "w", encoding="utf-8")
    t0 = time.time()
    for step in range(1, a.max_steps + 1):
        loss_sum = 0.0
        for _ in range(a.grad_accum):
            batch = {k: v.to(dev) for k, v in next(batches).items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss / a.grad_accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"loss không hữu hạn ở bước {step}: {loss.item()} "
                                         "(nguy cơ tràn số fp16 → giảm lr / kiểm tra mask)")
            scaler.scale(loss).backward()
            loss_sum += loss.item()
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        rec = {"step": step, "loss": loss_sum, "grad_norm": gnorm if math.isfinite(gnorm) else None,
               "lr": sched.get_last_lr()[0], "scale": scaler.get_scale(),
               "elapsed_min": (time.time() - t0) / 60}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if step % 10 == 0 or step == 1:
            print(f"[{a.setup}] step {step}/{a.max_steps} loss={loss_sum:.4f} gn={gnorm:.3f} "
                  f"lr={rec['lr']:.2e} {rec['elapsed_min']:.1f}min "
                  f"vram={torch.cuda.max_memory_allocated() / 2**30:.2f}GB", flush=True)
        if step % a.save_every == 0 or step == a.max_steps:
            save(model, out, a.setup)
    train_minutes = (time.time() - t0) / 60
    log.close()

    run = {
        **info, **{k: v for k, v in vars(a).items() if k != "out"},
        "train_images": len(entries),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "train_minutes": round(train_minutes, 2),
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "python": platform.python_version(),
    }
    (out / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(json.dumps(run), flush=True)


if __name__ == "__main__":
    main()
