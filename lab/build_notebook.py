"""Dựng lab_day10.ipynb từ mã trong lab/ và kết quả thật (report/, work/).

  python -m lab.build_notebook

Ô GPU (huấn luyện, suy luận) là lệnh đã chạy trên Kaggle T4 x2 qua kaggle_jobs/build.py, KHÔNG chạy lại
ở đây; log thật nằm trong report/logs/. Ô CPU (dữ liệu, nhãn, chấm điểm, phân tích, test reward) được chạy
thật bằng trình thực thi nhỏ bên dưới và output được ghi vào notebook.
"""
import contextlib
import io
import json
import os
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s, run=True):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True), "_run": run}


CELLS = [
    md("""
# Lab Day 10: Finetune Qwen3-VL-4B (LoRA) để định vị hư hại mặt đường

Mã thật nằm trong `lab/` (mỗi bước của GUIDE là một module); notebook này gọi lại các module đó theo đúng thứ tự
GUIDE và in số liệu thật.

- **GPU:** mọi bước huấn luyện/suy luận chạy trên **Kaggle T4 x2** (2 thí nghiệm song song, mỗi GPU một thí nghiệm)
  bằng `kaggle_jobs/build.py` → log trong `report/logs/`. Các ô đó để `# (đã chạy trên Kaggle)` và không có output.
- **CPU:** các ô còn lại (dữ liệu, nhãn, chấm điểm, phân tích, test reward) được chạy thật khi dựng notebook
  (`python -m lab.build_notebook`).
"""),
    md("## 0. Môi trường"),
    code("""
# (đã chạy trên Kaggle) — runner cài đúng các bản đã ghim rồi gỡ torchao cũ của image Kaggle
# !pip -q install transformers==5.19.0 peft==0.21.2 trl==1.9.2 accelerate==1.15.0 bitsandbytes==0.50.2
# !pip uninstall -y torchao
# !python -m pytest -q
""", run=False),
    code("""
import json, os, subprocess, sys
sys.path.insert(0, '.')
r = subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests', 'lab/tests'], capture_output=True, text=True,
                   encoding='utf-8', errors='replace')
print(r.stdout.strip().splitlines()[-1])
"""),
    md("## 1. Dữ liệu: split theo phiên quay"),
    code("""
# !python scripts/prepare_data.py --kaggle      (trên Kaggle: --source /kaggle/input/...)
from lab.data_check import stats
st, overlap = stats()
print(json.dumps(st, indent=1, ensure_ascii=False))
print('phiên nằm ở 2 split:', overlap)
print('dataset_version:', json.load(open('data/splits/meta.json'))['dataset_version'])
"""),
    md("![6 ảnh train kèm box gốc](report/figures/data_gt_samples.jpg)"),
    md("""
**Chia ngẫu nhiên thì sao?** Baseline *không học gì*: chép box của ảnh train giống nhất (ảnh xám 64×36, L2).
Split theo phiên: láng giềng không bao giờ cùng phiên → F1 ≈ 0. Split ngẫu nhiên: láng giềng là frame liền kề → F1 cao giả.
"""),
    code("""
# python -m lab.leakage --out report/analysis/leakage.json
lk = json.load(open('report/analysis/leakage.json', encoding='utf-8'))
s = lk['session_split']
print(f"split theo phiên : F1={s['f1']:.3f}  láng giềng cùng phiên={s['nn_same_session_frac']:.0%}  gần trùng={s['near_duplicate_frac']:.0%}")
for r in lk['random_split']:
    print(f"{r['split']:<20}: F1={r['f1']:.3f}  láng giềng cùng phiên={r['nn_same_session_frac']:.0%}  gần trùng={r['near_duplicate_frac']:.0%}")
"""),
    md("## 2. Nhãn YOLO → JSON thang 0–1000"),
    code("""
from lab.common import build_prompt, objects_to_answer, yolo_to_box, box_to_pixels, load_split
e = load_split('train')[0]
print(e['image_id'], e['objects'][:2])
print(objects_to_answer(e['objects']))
# Kiểm tra: quy ngược về pixel phải trùng box YOLO gốc (sai số chỉ do làm tròn số nguyên)
err, skipped = 0, 0
for split in ('train', 'val', 'test'):
    for x in load_split(split):
        for o in x['objects']:
            xc, yc, w, h = o['yolo']
            if w * h == 0 or not (0 <= xc - w/2 and xc + w/2 <= 1 and 0 <= yc - h/2 and yc + h/2 <= 1):
                skipped += 1  # box suy biến / tràn khung: bị kẹp có chủ đích
                continue
            px = box_to_pixels(yolo_to_box(o['label'], o['yolo'])['bbox_2d'], 'norm1000', 640, 360)
            ref = ((xc-w/2)*640, (yc-h/2)*360, (xc+w/2)*640, (yc+h/2)*360)
            err = max(err, max(abs(p - q) for p, q in zip(px, ref)))
print(f'sai số quy đổi lớn nhất: {err:.2f} px ({skipped} box suy biến/tràn khung được kẹp)')
print(build_prompt('norm1000'))
"""),
    md("## 3. E0 zero-shot"),
    code("""
# (đã chạy trên Kaggle) — lab/common.py: llm_int8_skip_modules=["model.visual", "visual", "lm_head"]
# !python -m lab.infer --out work/preds/e0.jsonl
# !python scripts/score.py --pred work/preds/e0.jsonl --name E0 --out report/evidence/e0.json
""", run=False),
    code("""
from scripts.score import parse_output
e0 = json.load(open('report/evidence/e0.json'))
print('parse_valid_rate =', round(e0['parse_valid_rate'], 3), '| F1 =', round(e0['micro']['f1'], 3))
raws = [json.loads(l) for l in open('report/preds/e0.jsonl', encoding='utf-8')]
bad = [r for r in raws if parse_output(r['raw']) is None]
print(f'JSON hỏng: {len(bad)} ảnh, trong đó {sum(len(r["raw"]) >= 400 for r in bad)} bị cắt ở 256 token vì lặp box')
for r in raws[:3] + bad[:2]:
    print('-', r['image_id'], repr(r['raw'][:160]))
"""),
    md("## 4. Ví dụ huấn luyện và mask loss"),
    code("""
# (đã chạy trên Kaggle: python -m lab.inspect_model) — mã: lab/data.py::encode_example, make_collate
ins = json.load(open('report/analysis/inspect_model.json', encoding='utf-8'))
enc = ins['encoding']
print('image_grid_thw =', enc['image_grid_thw'], '→ token ảnh =', enc['n_image_tokens'])
print('độ dài chuỗi =', enc['seq_len'], '| token có loss =', enc['n_answer_tokens'])
print('giải mã labels != -100:', enc['decoded_labels'][:120], '...', enc['decoded_labels'][-30:])
"""),
    md("**Đối chứng không mask** (`python -m lab.train --setup E1 --no-mask-prompt`, đã chạy trên Kaggle): cùng E1, chỉ đổi mask."),
    code("""
for k in ('e4_coord_pixel', 'nomask_e1'):
    log = [json.loads(l) for l in open(f'report/runs/{k}/train_log.jsonl')]
    t, s = sum(x['tokens'] for x in log), sum(x['supervised_tokens'] for x in log)
    print(f"{k:<15} token/mẫu={t/len(log)/8:.1f}  token có loss/mẫu={s/len(log)/8:.1f} ({s/t:.1%})  "
          f"loss {sum(x['loss'] for x in log[:10])/10:.2f} → {sum(x['loss'] for x in log[-10:])/10:.2f}")
for k in ('report/evidence/e1.json', 'report/extra/nomask_e1.json'):
    d = json.load(open(k))
    print(f"{d['name']:<10} F1={d['micro']['f1']:.3f} CI95={[round(x, 3) for x in d['f1_ci95']]} "
          f"P={d['micro']['precision']:.3f} R={d['micro']['recall']:.3f} FP/ảnh={d['fp_per_image']:.2f}")
"""),
    md("## 5. Setup LoRA / đóng băng"),
    code("""
print(json.dumps(ins['block_params'], indent=1))
print('số module LoRA:', ins['lora_modules'], '| dtype trên GPU:', ins['gpu_dtypes'])
for k in ('e1', 'e2', 'e3'):
    r = json.load(open(f'report/runs/{k}/run.json'))
    print(k.upper(), {x: r[x] for x in ('n_lora_modules', 'trainable_params', 'lora_params', 'projector_params')})
"""),
    md("## 6. Huấn luyện (300 bước, batch 2 × tích luỹ 4, lr 2e-4, cosine, fp16 + GradScaler)"),
    code("""
# (đã chạy trên Kaggle, GPU0: E1 rồi E2; GPU1: E3 rồi E0)
# !python -m lab.train --setup E1 --out work/runs/e1
# !python -m lab.train --setup E2 --out work/runs/e2      # + projector.pt lưu riêng
# !python -m lab.train --setup E3 --out work/runs/e3
""", run=False),
    code("""
for k in ('e1', 'e2', 'e3'):
    r = json.load(open(f'report/runs/{k}/run.json'))
    log = [json.loads(l) for l in open(f'report/runs/{k}/train_log.jsonl')]
    print(f"{k.upper()}: loss {sum(x['loss'] for x in log[:10])/10:.3f} → {sum(x['loss'] for x in log[-10:])/10:.3f} | "
          f"VRAM đỉnh {r['peak_vram_gb']} GB | {r['train_minutes']} phút | {r['gpu']}")
"""),
    md("![loss](report/figures/loss_e1_e3.png)"),
    md("## 7. Suy luận và chấm E1–E3"),
    code("""
# (đã chạy trên Kaggle) nạp 4-bit → adapter LoRA → projector.pt (E2/E3)
# !python -m lab.infer --run-dir work/runs/e3 --out work/preds/e3.jsonl
# !python scripts/score.py --pred work/preds/e3.jsonl --name E3 --out report/evidence/e3.json $(python -m lab.meta_args work/runs/e3)
""", run=False),
    code("""
print(open('report/analysis/table.md', encoding='utf-8').read())
"""),
    md("![F1 kèm CI95](report/figures/f1_ci95.png)"),
    code("""
an = json.load(open('report/analysis/analysis.json', encoding='utf-8'))
for k, v in an['paired'].items():
    print(f"{k:<22} ΔF1={v['diff']:+.3f}  CI95 cặp=[{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
"""),
    md("## 8. E4: ablation thang toạ độ (norm1000 vs pixel) trên setup tốt nhất (E3)"),
    code("""
# (đã chạy trên Kaggle)
# !python -m lab.train --setup E3 --coord pixel --out work/runs/e4_coord_pixel
# !python -m lab.infer --run-dir work/runs/e4_coord_pixel --coord pixel --out work/preds/e4_coord_pixel.jsonl
# !python scripts/score.py --pred work/preds/e4_coord_pixel.jsonl --name E4 --coord pixel --out report/evidence/e4_coord_pixel.json ...
""", run=False),
    code("""
for k in ('e4_coord_norm1000', 'e4_coord_pixel'):
    p = f'report/evidence/{k}.json'
    if os.path.exists(p):
        d = json.load(open(p))
        print(k, '| coord =', d['coord'], '| F1 =', round(d['micro']['f1'], 3), '| CI95 =', [round(x, 3) for x in d['f1_ci95']])
v = an['paired']['e4_coord_pixel-e3']
print(f"pixel − norm1000: ΔF1={v['diff']:+.3f}  CI95 cặp=[{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
print('chấm sai thang:', json.dumps(an['wrong_scale'], indent=1))
"""),
    md("## 9. E5 (bonus): GRPO với reward IoU"),
    code("""
from lab.reward import compute_reward
GT = [{'label': 'pothole', 'bbox': (100, 100, 200, 200)}]
for name, out in [('hoàn hảo', '[{"label":"pothole","bbox_2d":[100,100,200,200]}]'),
                  ('JSON hỏng', '[{"label":"pothole","bbox_2d":[100,'),
                  ('sai lớp', '[{"label":"crack","bbox_2d":[100,100,200,200]}]'),
                  ('thừa box', '[{"label":"pothole","bbox_2d":[100,100,200,200]},{"label":"pothole","bbox_2d":[800,800,900,900]}]'),
                  ('box khổng lồ', '[{"label":"pothole","bbox_2d":[0,0,1000,1000]}]'),
                  ('trả []', '[]')]:
    print(f'{name:<12} reward={compute_reward(out, GT):.2f}')
print('cùng rỗng    reward=%.2f' % compute_reward('[]', []))
r = subprocess.run([sys.executable, '-m', 'pytest', '-q', 'lab/tests'], capture_output=True, text=True,
                   encoding='utf-8', errors='replace')
print(r.stdout.strip().splitlines()[-1])
"""),
    code("""
# (đã chạy trên Kaggle) !python -m lab.grpo --init-run work/runs/e3 --out work/runs/e5
p = 'report/runs/e5/run.json'
if os.path.exists(p):
    print(json.dumps({k: v for k, v in json.load(open(p)).items() if k != 'gpu'}, indent=1))
    rs = [json.loads(l) for l in open('report/runs/e5/reward_stats.jsonl', encoding='utf-8')]
    q = max(1, len(rs) // 4)
    for name, part in (('1/4 đầu', rs[:q]), ('1/4 cuối', rs[-q:])):
        avg = lambda key: sum(x[key] for x in part) / len(part)
        print(f"{name}: reward={avg('reward_mean'):.3f} rỗng={avg('empty_frac'):.1%} box khổng lồ={avg('giant_box_frac'):.1%} "
              f"JSON hỏng={avg('invalid_frac'):.1%} box/ảnh={avg('pred_boxes_mean'):.2f} (GT {avg('gt_boxes_mean'):.2f})")
    e5 = json.load(open('report/evidence/e5.json'))
    v = an['paired']['e5-e3']
    print(f"E5: F1={e5['micro']['f1']:.3f} CI95={[round(x, 3) for x in e5['f1_ci95']]} P={e5['micro']['precision']:.3f} "
          f"R={e5['micro']['recall']:.3f} | E5 − E3: ΔF1={v['diff']:+.3f} CI95 cặp=[{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
"""),
    md("![GRPO](report/figures/grpo_reward.png)"),
    md("## 10. Phân tích lỗi (câu 9) và ảnh minh hoạ"),
    code("""
best = an['best_sft']
print('setup SFT tốt nhất:', best)
print('FP:', json.dumps(an['errors'][best]['fp']))
print('FN:', json.dumps(an['errors'][best]['fn']))
print('recall theo diện tích box:', json.dumps(an['recall_by_size'][best], ensure_ascii=False))
print('F1 theo phiên quay:', an['session_bootstrap'][best]['f1_per_session'])
"""),
    md("Ảnh minh hoạ: `report/figures/e3_*.jpg` và `e0_*.jpg` (xanh lá: GT khớp, vàng: GT bỏ sót, xanh dương: dự đoán đúng, đỏ: dự đoán sai)."),
]


def execute(cells):
    """Chạy tuần tự các ô CPU trong cùng một namespace, ghi stdout/traceback làm output."""
    ns, n = {"__name__": "__main__"}, 0
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        for c in cells:
            run = c.pop("_run", None)
            if c["cell_type"] != "code" or not run:
                continue
            n += 1
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    exec(compile("".join(c["source"]), f"<cell {n}>", "exec"), ns)
                c["outputs"] = [{"name": "stdout", "output_type": "stream", "text": buf.getvalue().splitlines(True)}]
            except Exception:
                c["outputs"] = [{"name": "stdout", "output_type": "stream", "text": buf.getvalue().splitlines(True)},
                                {"name": "stderr", "output_type": "stream",
                                 "text": traceback.format_exc().splitlines(True)}]
                print(f"ô {n} lỗi:\n{traceback.format_exc()}")
            c["execution_count"] = n
    finally:
        os.chdir(cwd)
    for c in cells:
        c.pop("_run", None)
    return cells


def main():
    nb = {"cells": execute([dict(c) for c in CELLS]),
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    (ROOT / "lab_day10.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("đã ghi lab_day10.ipynb")


if __name__ == "__main__":
    main()
