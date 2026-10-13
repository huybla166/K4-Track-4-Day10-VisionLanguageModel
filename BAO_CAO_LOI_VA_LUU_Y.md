# Báo cáo các bẫy kỹ thuật và lưu ý (Lab Day 10)

Các lỗi dưới đây là lỗi **tôi thực sự gặp** khi chạy lab trên Kaggle T4 x2 với đúng các phiên bản đã ghim
(`transformers==5.19.0`, `peft==0.21.2`, `trl==1.9.2`, `bitsandbytes==0.50.2`), kèm cách phát hiện và cách sửa.
Các con số lấy từ `work/inspect.json`, log smoke test và `report/analysis/analysis.json`.

## 1. `llm_int8_skip_modules=["visual", "lm_head"]` KHÔNG giữ được ViT ở fp16 (transformers 5.x)

- **Triệu chứng:** không có lỗi nào. Smoke test đầu tiên chạy E1 bình thường, nhưng `lab.inspect_model` in
  `visual.blocks.0.attn.qkv: torch.uint8` và `visual.merger.linear_fc1: torch.uint8` (tức `Linear4bit`). Ở E3,
  số "tham số projector" đếm được chỉ còn 54.579.200 (một nửa của 109.105.152 vì 2 trọng số 4-bit đóng gói
  trong 1 byte) và huấn luyện dừng với `ValueError: Attempting to unscale FP16 gradients`.
- **Nguyên nhân:** transformers 5.x so khớp skip-module bằng `re.match(pattern, full_name)` **neo ở đầu** tên đầy đủ
  (`quantizers_utils.should_convert_module`). Tên module thật là `model.visual.blocks.0.attn.qkv`, nên mẫu `"visual"`
  không khớp; chỉ `"lm_head"` (tên trùng khít) được bỏ qua. Kết quả: ViT + projector bị lượng tử hoá 4-bit, projector
  không thể huấn luyện đầy đủ (đúng cái bẫy GUIDE bước 3 cảnh báo, nhưng xảy ra *dù đã làm đúng theo GUIDE*).
- **Sửa:** `llm_int8_skip_modules=["model.visual", "visual", "lm_head"]` và thêm `assert_vit_not_quantized()`
  trong `lab/common.py`: nếu còn bất kỳ `Linear4bit` nào trong `visual`/`lm_head` thì dừng ngay.
  Đã kiểm chứng bằng chính hàm của transformers (`should_convert_module`): mẫu cũ → *convert*, mẫu mới → *SKIP*
  cho `model.visual.blocks.*`, `model.visual.merger.*`, `model.visual.deepstack_merger_list.*`.
- **Bài học:** luôn in dtype của vài module đại diện sau khi nạp model; công cụ chấm không phát hiện được lỗi này
  (E0/E1 vẫn chạy ra số, chỉ là số của một model khác cấu hình).

## 2. `torchao` cũ trên image Kaggle làm hỏng E3

- **Triệu chứng:** `ImportError: Found an incompatible version of torchao. Found version 0.10.0, but only versions above 0.16.0 are supported`
  khi `get_peft_model` gắn LoRA lên các `nn.Linear` fp16 của ViT (chỉ E3 mới có LoRA trên Linear không lượng tử).
- **Sửa:** `pip uninstall -y torchao` trước khi chạy (peft chỉ kiểm tra torchao nếu nó được cài).

## 3. GradScaler và tham số fp16

- `GradScaler.unscale_` từ chối gradient fp16. Mọi tham số được huấn luyện (LoRA, projector) phải ở **float32**;
  `apply_setup()` ép tất cả tham số `requires_grad` về fp32 sau khi gắn LoRA và mở projector.
- Projector ở fp32 còn vì cập nhật lr·grad rất nhỏ (≈1e-6 tương đối) bị **làm tròn mất** nếu cộng vào trọng số fp16
  (fp16 chỉ có ~3 chữ số có nghĩa).
- Một số bước có `grad_norm = inf` và `scale` giảm một nửa: đó là GradScaler bỏ qua bước bị tràn số, không phải lỗi.

## 4. TRL `GRPOTrainer` tự ép tham số sang bf16

- Với model lượng tử hoá, `GRPOTrainer.__init__` chạy `param.data = param.data.to(torch.bfloat16)` cho mọi tham số
  huấn luyện, và `GRPOConfig` mặc định `bf16=True`. T4 không hỗ trợ bf16 gốc; ngoài ra lr 5e-6 cộng vào trọng số
  bf16 (8 bit mantissa) gần như **không đổi gì**.
- **Sửa:** `fp16=True, bf16=False` và sau khi tạo trainer, ép lại 504 tensor LoRA về fp32 (`lab/grpo.py`).

## 5. `max_completion_length=128` của GRPO ngắn hơn đáp án thật

- Đáp án dài nhất ở train là 366 token (13 box), p99 = 219 token. Với 128 token, ảnh nhiều box luôn bị cắt → JSON hỏng
  → reward 0, tức là GRPO sẽ **phạt oan** việc liệt kê đủ box và đẩy model về phía trả ít box (một dạng reward hacking
  do cấu hình chứ không do model).
- **Sửa:** chỉ dùng ảnh train có đáp án ≤ 128 token làm prompt GRPO và bật `mask_truncated_completions=True`.

## 6. Chat template, mask loss và token kết thúc

- Qwen3-VL chèn `"\n"` sau `<|im_end|>` của lượt assistant; tôi mask cả token này (chỉ học JSON + `<|im_end|>`).
  Kiểm chứng: giải mã các token có `labels != -100` ra đúng chuỗi JSON đáp án + `<|im_end|>`.
- Ảnh 640×360 → `image_grid_thw = [1, 22, 40]` (patch 16 px, ảnh được resize về 640×352) → 880 patch / (2×2 merge)
  = **220 token** ảnh. Trong chuỗi 605 token của ví dụ kiểm tra, chỉ 249 token đáp án được tính loss.
- Collate: `input_ids/attention_mask/mm_token_type_ids/labels` được pad (labels pad = -100);
  `pixel_values` (1760×1536 cho 2 ảnh) và `image_grid_thw` được **nối** theo chiều 0, không pad.

## 7. Projector không nằm trong adapter

- `peft.save_pretrained()` chỉ lưu LoRA. Projector của E2/E3 được lưu riêng `projector.pt` (24 tensor, 109.105.152
  tham số) và nạp lại khi suy luận. `lab.check_reload` xác nhận: projector đã lưu khác bản gốc và sau khi nạp lại
  sai số chỉ bằng độ chính xác fp16.

## 8. Bộ chấm và dữ liệu: những điều công cụ chấm không bắt

- `score.parse_output` loại **cả ảnh** nếu chỉ một box có `x2 <= x1`. Vì vậy nhãn huấn luyện không bao giờ chứa box
  suy biến (`yolo_to_box` nới box rộng 0 thành rộng 1 đơn vị).
- Dữ liệu có 1 box diện tích 0 (`vlcsnap-00058`, train, `h = 0`) và 63 box tràn ra ngoài khung (được kẹp lại).
- Nhãn nhiễu: trong train có **123 cặp box manhole/pothole trùng nhau** (IoU ≥ 0,5, cùng một vật được gán 2 lớp)
  và 20 cặp crack/pothole; test chỉ có 1 cặp manhole/pothole. Model học "nắp cống hỏng thì kèm thêm pothole" từ train
  và bị tính FP pothole trên test.
- **Không ảnh nào rỗng** ở cả 3 split, nên đáp án `[]` không bao giờ đúng: một model "lười" trả `[]` bị phạt ngay.
- Test chỉ gồm **3 phiên quay** (213 + 83 + 15 ảnh). Bootstrap theo ảnh (CI95 của `score.py`) coi 311 ảnh là độc
  lập, trong khi các frame cùng phiên rất giống nhau → CI thật rộng hơn. Tôi bổ sung bootstrap theo phiên và bootstrap
  cặp cho chênh lệch F1 trong `lab/analysis.py`.

## 9. Môi trường Windows / Kaggle

- Console Windows (cp1252) lỗi `UnicodeEncodeError` khi in tiếng Việt → `PYTHONIOENCODING=utf-8`.
- Đường dẫn venv quá dài trên Windows (`WinError 206`) khi cài torch → dùng thư mục ngắn.
- Kaggle CLI 2.x: đăng nhập bằng `kaggle auth login` (OAuth) — không cần chép khoá vào repo hay notebook.
- Chạy kernel script không xem được log trực tiếp: tôi chạy 3 smoke test ngắn (≤ 4 bước) trước khi tốn ~2 giờ GPU
  cho E1–E3, nhờ vậy phát hiện lỗi 1 và 2 sớm.
- Token OAuth của Kaggle CLI hết hạn giữa chừng: `kaggle kernels status` báo `Permission 'kernels.get' was denied`
  dù CLI vẫn nói đã đăng nhập → chạy lại `kaggle auth login --force`.
- `kaggle kernels output` tải cả adapter/projector (hàng trăm MB) và hay đứt `IncompleteRead` → chỉ tải file nhỏ bằng
  `--file-pattern '\.(json|jsonl|log|txt|md)$'`. Trọng số ở lại trên Kaggle, kernel sau dùng lại qua `kernel_sources`
  (stage2 nạp `work/runs/e3` từ output của kernel chính), nên không có trọng số nào đi qua máy cá nhân hay vào git.

## 10. Suy luận và đánh giá: những chỗ dễ đọc sai số

- **E0 lặp vô hạn:** 25/311 ảnh, model gốc "lát gạch" box crack liền nhau cho đến khi bị cắt ở `max_new_tokens=256`,
  thành JSON hỏng và **cả ảnh** tính là bỏ sót. Tôi giữ nguyên 256 cho mọi setup để so sánh công bằng thay vì tăng
  riêng cho E0. Model E4-pixel cũng có 1 ảnh lặp box pothole kiểu này.
- **Thang toạ độ phải đi cùng nhau ở 3 chỗ:** nhãn huấn luyện, `infer.py --coord` và `score.py --coord`. Lệch một chỗ
  là F1 về đúng 0,000 (REPORT câu 3), nên mọi file bằng chứng đều ghi trường `coord`, và `lab.analysis` đọc trường này
  trước khi chấm lại.
- **Loss khi không mask không so được với loss khi có mask:** 4,14 so với 0,59 không có nghĩa là model tệ gấp 7 lần.
  Phần lớn loss đến từ 220 token `<|image_pad|>` (REPORT câu 4).
- **GRPO: reward trên train rất nhiễu** (16 completion/bước, độ lệch chuẩn khoảng 0,2). Đừng kết luận từ đường reward,
  hãy chấm lại trên test và so bằng bootstrap cặp. Luôn ghi tỉ lệ `[]` và tỉ lệ box khổng lồ mỗi bước
  (`reward_stats.jsonl`) để bắt reward hacking sớm.
- **Hai kiểm tra của repo mâu thuẫn nhau về `REPORT.md`:** `tests/test_configs_and_docs.py::test_report_template_has_student_markers_and_ten_questions`
  đòi file **còn** chuỗi `<điền` (test này viết cho bản template), trong khi `scripts/check_submission.py` báo FAIL nếu
  chuỗi đó **còn**. Sau khi điền xong báo cáo, test này fail là điều đúng phải xảy ra. Tôi không sửa `tests/` (đề cấm),
  và lấy `check_submission.py` (0 FAIL) làm chuẩn nộp bài. Mọi test còn lại đều pass (79 passed, 1 skipped).
