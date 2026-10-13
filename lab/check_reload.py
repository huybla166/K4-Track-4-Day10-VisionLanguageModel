"""Kiểm tra nạp lại run đã huấn luyện: adapter LoRA khác 0 và projector đã lưu thực sự được nạp lại.

  python -m lab.check_reload work/runs/e2
"""
import json
import sys
from pathlib import Path

import torch

from lab.common import load_model_and_processor
from lab.lora_setup import is_projector_param, load_projector, strip_peft_prefix


def main(run_dir):
    from peft import PeftModel
    run_dir = Path(run_dir)
    model, _ = load_model_and_processor()
    base = {strip_peft_prefix(n): p.detach().float().cpu().clone()
            for n, p in model.named_parameters() if is_projector_param(n)}
    model = PeftModel.from_pretrained(model, str(run_dir / "adapter"))
    lora_b = [p.abs().max().item() for n, p in model.named_parameters() if "lora_B" in n]
    res = {"lora_tensors": len(lora_b), "lora_B_nonzero": sum(v > 0 for v in lora_b),
           "lora_vit_tensors": sum(1 for n, _ in model.named_parameters() if "lora_" in n and "visual" in n)}
    proj = run_dir / "projector.pt"
    if proj.exists():
        saved = torch.load(proj, map_location="cpu")
        res["projector_tensors_saved"] = len(saved)
        res["projector_params_saved"] = sum(v.numel() for v in saved.values())
        res["max_change_vs_base"] = max((saved[k] - base[k]).abs().max().item() for k in saved)
        load_projector(model, saved)
        now = {strip_peft_prefix(n): p.detach().float().cpu()
               for n, p in model.named_parameters() if is_projector_param(n)}
        res["max_err_after_reload"] = max((now[k] - saved[k]).abs().max().item() for k in saved)
        assert res["max_change_vs_base"] > 0, "projector không đổi so với gốc: có thật sự được huấn luyện?"
        assert res["max_err_after_reload"] < 1e-2, "projector nạp lại không khớp bản đã lưu"
    print(json.dumps(res, indent=2))
    (run_dir / "reload_check.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
