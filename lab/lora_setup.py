"""Setup LoRA / đóng băng cho E1, E2, E3 (GUIDE bước 5)."""
import re

import torch
from peft import LoraConfig, get_peft_model

from lab.common import PROJECTOR_KEYS

LLM_TARGETS = r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
VIT_TARGETS = r"model\.visual\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.linear_fc(1|2))"

SETUPS = {
    "E1": {"vit_lora": False, "train_projector": False},
    "E2": {"vit_lora": False, "train_projector": True},
    "E3": {"vit_lora": True, "train_projector": True},
}


def target_regex(setup):
    return f"({LLM_TARGETS}|{VIT_TARGETS})" if SETUPS[setup]["vit_lora"] else LLM_TARGETS


def matched_modules(model, regex):
    pat = re.compile(regex)
    return [n for n, _ in model.named_modules() if pat.fullmatch(n)]


def is_projector_param(name):
    return any(k in name for k in PROJECTOR_KEYS) and "lora_" not in name


def apply_setup(model, setup, r=16, alpha=32, dropout=0.05):
    for p in model.parameters():
        p.requires_grad = False
    model.enable_input_require_grads()  # cần cho gradient checkpointing
    regex = target_regex(setup)
    matched = matched_modules(model, regex)
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=regex,
                     bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    extra = []
    if SETUPS[setup]["train_projector"]:
        for name, p in model.named_parameters():
            if is_projector_param(name):
                p.data = p.data.to(torch.float32)  # tối ưu ở fp32: cập nhật nhỏ không bị nuốt bởi fp16
                p.requires_grad = True
                extra.append(name)
    for p in model.parameters():  # GradScaler không unscale được gradient fp16: mọi tham số học đều fp32
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.to(torch.float32)
    trainable =sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "lora_" in n)
    info = {"setup": setup, "r": r, "alpha": alpha, "n_lora_modules": len(matched),
            "n_lora_modules_vit": sum(".visual." in f".{m}" for m in matched),
            "trainable_params": trainable, "lora_params": lora,
            "projector_params": trainable - lora, "projector_tensors": len(extra)}
    return model, info


def strip_peft_prefix(name):
    for pre in ("base_model.model.", "base_model."):
        if name.startswith(pre):
            return name[len(pre):]
    return name


def projector_state(model):
    """Các tham số projector đã huấn luyện (peft.save_pretrained KHÔNG lưu chúng)."""
    return {strip_peft_prefix(n): p.detach().to("cpu", torch.float32).clone()
            for n, p in model.named_parameters() if is_projector_param(n)}


def load_projector(model, state):
    """Nạp lại projector vào model (gốc hoặc đã bọc peft); ép về dtype hiện tại của model."""
    params = {strip_peft_prefix(n): p for n, p in model.named_parameters()}
    missing = [k for k in state if k not in params]
    if missing:
        raise KeyError(f"không tìm thấy {len(missing)} tham số projector, ví dụ {missing[:3]}")
    with torch.no_grad():
        for k, v in state.items():
            params[k].copy_(v.to(params[k].dtype))
    return len(state)
