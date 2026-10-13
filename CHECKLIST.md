# Checklist Lab Day 10

Tiến độ theo GUIDE/vlearn, kèm bằng chứng. Phần cứng: Kaggle Tesla T4 (2 GPU, mỗi thí nghiệm một GPU).

## Chuẩn bị
- [x] Fork + clone repo cá nhân, `python -m pytest -q` → 68 passed, 1 skipped (test dữ liệu thật bỏ qua khi chưa có data). Sau khi điền REPORT, test `test_report_template_has_student_markers...` fail có chủ đích (xem `BAO_CAO_LOI_VA_LUU_Y.md` mục 10)
- [x] Khoá Kaggle không nằm trong repo/notebook (Kaggle CLI đăng nhập OAuth; `.env` nằm trong `.gitignore`)

## Chặng 1 — Dữ liệu (commit `654f31c`)
- [x] `prepare_data.py` → 1.422 / 276 / 311 ảnh, `dataset_version = d10-21e4070301`
- [x] Số box từng lớp khớp README; không phiên nào ở 2 split (`report/figures/data_stats.json`)
- [x] Vẽ 6 ảnh có box gốc (`report/figures/data_gt_samples.jpg`)
- [x] Đo rò rỉ của split ngẫu nhiên bằng baseline chép nhãn (`lab/leakage.py`, `report/analysis/leakage.json`)

## Chặng 2 — Nhãn 0–1000 và E0 (commit `0c96471`)
- [x] `yolo_to_box` / `objects_to_answer` (sắp theo y1, x1), quy ngược về pixel sai số ≤ 0,32 px
- [x] Prompt nêu 3 lớp và "0-1000 scale"
- [x] E0 zero-shot trên 311 ảnh test → `report/evidence/e0.json` (parse_valid_rate 0,907, F1 0,101)

## Chặng 3 — Chat template và mask loss (commit `0c96471`)
- [x] 220 token ảnh cho 640×360 (`image_grid_thw = [1, 22, 40]`)
- [x] Giải mã `labels != -100` = đúng JSON đáp án + `<|im_end|>`
- [x] Collate: pad `input_ids/attention_mask/mm_token_type_ids/labels`, nối `pixel_values/image_grid_thw`
- [x] Đối chứng không mask loss (`--no-mask-prompt`): 16,5% token có nhãn khi mask so với 100% khi không; F1 0,255 so với 0,268 (`report/extra/nomask_e1.json`) → REPORT câu 4

## Chặng 4 — E1, E2, E3 (commit `ba57ae2`, `2e29c6e`, `0d91a38`)
- [x] 252 module LoRA (E1, E2), 348 (E3); E2 − E1 = 109,105,152 tham số projector
- [x] ViT + projector + lm_head ở fp16 (`llm_int8_skip_modules=["model.visual", ...]` + `assert_vit_not_quantized`)
- [x] Huấn luyện 300 bước, loss hữu hạn và giảm (`report/figures/loss_e1_e3.png`)
- [x] Lưu riêng `projector.pt` (E2/E3) và nạp lại lúc suy luận
- [x] `e1.json` – `e3.json` có meta `trainable_params`, `peak_vram_gb`, `train_minutes`
- [x] E1: parse_valid_rate 0,994 ≥ 0,95; F1(E1) − F1(E0) = 0,168 ≥ 0,10

## Chặng 5 — E4 ablation (Kaggle kernel `day10-vlm-stage2`)
- [x] Đổi đúng một biến (`coord_scale`: norm1000 vs pixel) trên setup tốt nhất E3, chạy cả hai giá trị
- [x] `report/evidence/e4_coord_norm1000.json` (F1 0,429), `report/evidence/e4_coord_pixel.json` (F1 0,363); Δ cặp −0,065 [−0,102; −0,029] → REPORT câu 10

## Chặng 6 — E5 GRPO (bonus)
- [x] Hàm reward IoU + 12 test (`lab/reward.py`, `lab/tests/test_reward.py`): hoàn hảo 1,0 · JSON hỏng 0,0 · sai lớp 0,2 · thừa box 0,6 · cùng rỗng 1,0
- [x] GRPO từ adapter E3, 60 bước, lr 5e-6, `num_generations=4` → `report/runs/e5/` (log) + `report/evidence/e5.json` (F1 0,448)
- [x] Phân tích reward hacking (`[]`, box khổng lồ, số box theo thời gian) + so với E3 bằng bootstrap cặp: +0,019 [−0,002; +0,039]

## Báo cáo và nộp bài
- [x] `REPORT.md`: họ tên, `dataset_version`, bảng tổng hợp E0–E5, 10 câu + phần E5, không còn `<điền`
- [x] ≥ 6 ảnh minh hoạ có box dự đoán + box đúng (7 ảnh `report/figures/e3_*.jpg`, 7 ảnh `e0_*.jpg`)
- [x] `lab_day10.ipynb` dựng từ mã thật (`python -m lab.build_notebook`): ô CPU chạy thật, ô GPU là lệnh đã chạy trên Kaggle
- [x] `python scripts/check_submission.py` → 0 FAIL, 0 WARN
- [x] Không commit `data/`, `work/`, `.env`, `*.safetensors`, `*.pt`; ≥ 5 commit
- [x] Push lên GitHub (public): https://github.com/huybla166/K4-Track-4-Day10-VisionLanguageModel
- [x] Nộp link trên LMS (sinh viên tự nộp)
