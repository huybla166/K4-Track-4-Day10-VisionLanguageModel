"""Dựng ví dụ huấn luyện: chat template, mask loss (chỉ học phần trả lời), collate (GUIDE bước 4)."""
import random
from pathlib import Path

import torch
from PIL import Image

from lab.common import ROOT, build_prompt, objects_to_answer

IGNORE = -100


def user_message(image, prompt):
    return {"role": "user", "content": [{"type": "image", "image": image},
                                        {"type": "text", "text": prompt}]}


def encode_example(processor, image, prompt, answer, mask_prompt=True):
    """Trả dict tensor 1 mẫu; labels = -100 trên token ảnh + prompt, giữ nguyên token trả lời.

    mask_prompt=False chỉ dùng cho thí nghiệm đối chứng (câu 4): tính loss trên MỌI token.
    """
    user = user_message(image, prompt)
    kw = dict(tokenize=True, return_dict=True, return_tensors="pt")
    prompt_enc = processor.apply_chat_template([user], add_generation_prompt=True, **kw)
    full_enc = processor.apply_chat_template(
        [user, {"role": "assistant", "content": [{"type": "text", "text": answer}]}],
        add_generation_prompt=False, **kw)
    n = prompt_enc["input_ids"].shape[1]
    ids = full_enc["input_ids"][0]
    if not torch.equal(ids[:n], prompt_enc["input_ids"][0]):
        raise ValueError("prompt không phải tiền tố của chuỗi đầy đủ: kiểm tra chat template")
    labels = ids.clone()
    labels[:n] = IGNORE
    # Qwen3 chèn "\n" sau <|im_end|> ở cuối lượt assistant: không cần học token xuống dòng đó.
    end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    ends = (ids == end_id).nonzero().flatten()
    if len(ends) and ends[-1] >= n:
        labels[ends[-1] + 1:] = IGNORE
    if not mask_prompt:
        labels = ids.clone()
    out = {k: v[0] for k, v in full_enc.items() if k not in ("pixel_values", "image_grid_thw")}
    out["pixel_values"] = full_enc["pixel_values"]
    out["image_grid_thw"] = full_enc["image_grid_thw"]
    out["labels"] = labels
    return out


def make_collate(pad_id):
    def collate(batch):
        out = {}
        seq_keys = [k for k in batch[0] if k not in ("pixel_values", "image_grid_thw")]
        for k in seq_keys:
            pad = {"labels": IGNORE, "input_ids": pad_id}.get(k, 0)
            out[k] = torch.nn.utils.rnn.pad_sequence([b[k] for b in batch], batch_first=True,
                                                     padding_value=pad)
        # Token ảnh có số lượng thay đổi theo ảnh: nối theo chiều 0, KHÔNG pad.
        out["pixel_values"] = torch.cat([b["pixel_values"] for b in batch], dim=0)
        out["image_grid_thw"] = torch.cat([b["image_grid_thw"] for b in batch], dim=0)
        return out
    return collate


def subsample(entries, fraction, seed=42):
    if fraction >= 1.0:
        return list(entries)
    k = max(1, round(len(entries) * fraction))
    return random.Random(seed).sample(list(entries), k)


class GroundingDataset(torch.utils.data.Dataset):
    def __init__(self, entries, processor, coord="norm1000", data_dir=None, mask_prompt=True):
        self.entries, self.processor, self.coord = entries, processor, coord
        self.data_dir = Path(data_dir or ROOT / "data")
        self.mask_prompt = mask_prompt

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        e = self.entries[i]
        image = Image.open(self.data_dir / e["image_path"]).convert("RGB")
        prompt = build_prompt(self.coord, e["width"], e["height"])
        answer = objects_to_answer(e["objects"], self.coord, e["width"], e["height"])
        return encode_example(self.processor, image, prompt, answer, self.mask_prompt)
