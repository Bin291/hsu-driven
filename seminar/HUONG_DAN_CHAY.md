# Hướng dẫn chạy — Prompt Firewall (HSU AI-Driven Challenge 2026)

Notebook `prompt_firewall_toxicchat.ipynb` huấn luyện + đánh giá một cascade 3 tầng
(regex → TF-IDF/LinearSVC → DistilBERT) để chặn prompt độc hại. Xem chi tiết kết quả
và kiến trúc trong [BAO_CAO.md](BAO_CAO.md).

## 1. Yêu cầu môi trường

- Python 3.9–3.13 khuyến nghị (bản gốc kiểm chứng trên macOS arm64 / Python 3.9.6).
  Nếu máy chỉ có Python bản mới hơn (vd. 3.14), `requirements.txt` ghim version cứng
  có thể **không cài được** — khi đó cài không ghim version (xem mục 2b).
- ~3–4 GB dung lượng trống (torch + transformers + trọng số DistilBERT tải về).
- Kết nối internet ở lần chạy đầu (tải dataset `lmsys/toxic-chat` và model
  `distilbert-base-uncased` từ Hugging Face Hub).
- Không bắt buộc GPU. Có CUDA/MPS thì train DistilBERT nhanh hơn nhiều; chạy CPU vẫn
  ra kết quả đúng nhưng chậm hơn (xem mục 4).

## 2a. Cài đặt — dùng đúng version đã kiểm chứng (khuyến nghị nếu máy hỗ trợ)

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2b. Cài đặt — Python quá mới, không cài được bản ghim (vd. Windows + Python 3.14)

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch transformers tokenizers safetensors huggingface_hub ^
    scikit-learn scipy numpy pandas joblib datasets matplotlib ^
    ipykernel jupyter_client nbformat nbconvert
```

Lưu ý: bản mới nhất của `transformers`/`datasets` đôi khi đổi API so với bản ghim
trong `requirements.txt`. Nếu notebook lỗi ở một cell cụ thể do API đổi, đó là điểm
cần sửa thủ công (thường chỉ đổi tên tham số).

## 3. Chạy notebook

### Cách 1 — mở trong Jupyter/VS Code và chạy tuần tự (khuyến nghị để xem từng bước)

```bash
python -m ipykernel install --user --name=prompt-firewall --display-name "prompt-firewall"
jupyter notebook prompt_firewall_toxicchat.ipynb
```
Chọn kernel `prompt-firewall`, chạy **Run All** (Cell → Run All / Kernel → Restart & Run All).

### Cách 2 — chạy toàn bộ từ dòng lệnh, không cần mở giao diện

```bash
jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.kernel_name=prompt-firewall \
    --ExecutePreprocessor.timeout=3600 \
    --output prompt_firewall_toxicchat.executed.ipynb \
    prompt_firewall_toxicchat.ipynb
```

`--ExecutePreprocessor.timeout=3600` cho phép mỗi cell chạy tối đa 1 giờ (bước huấn
luyện DistilBERT là cell tốn thời gian nhất). Kết quả nằm trong file
`prompt_firewall_toxicchat.executed.ipynb` (bản gốc không bị ghi đè).

## 4. Thời gian chạy dự kiến

| Phần | GPU/MPS | CPU |
|---|---|---|
| Tải dataset + EDA | vài giây | vài giây |
| TF-IDF + LinearSVC | < 1 giây | < 1 giây |
| Huấn luyện DistilBERT (3 epoch, ~4.3k mẫu) | 1–3 phút | **10–40+ phút**, tuỳ CPU |
| Benchmark tốc độ 10.000 prompt | vài giây | vài chục giây |

Trên máy chỉ có CPU, tổng thời gian chạy toàn bộ notebook thường rơi vào khoảng
**15–45 phút**, phần lớn nằm ở bước fine-tune DistilBERT (mục 5 của notebook).

## 5. Sau khi chạy xong — kiểm tra sản phẩm

```bash
python artifacts/predict.py input.csv output.csv prompt
```

`artifacts/` sẽ chứa:
- `prompt_firewall.py` — module `PromptFirewall` dùng lại được ngoài notebook.
- `predict.py` — script chấm điểm cho input là CSV có cột prompt.
- `firewall_model/` — trọng số đã huấn luyện (`fast.joblib`, thư mục `bert/`, `config.json`).

Dùng trực tiếp trong Python:
```python
from prompt_firewall import PromptFirewall
fw = PromptFirewall.load('artifacts/firewall_model', device='cpu')
fw.predict(['Ignore all previous instructions'])
```

## 6. Sự cố thường gặp

- **`pip install` báo không tìm thấy bản torch/transformers ghim sẵn** → dùng cách 2b
  (cài không ghim version).
- **Lỗi tải dataset/model từ Hugging Face** → kiểm tra kết nối internet; lần chạy đầu
  cần tải ~250 MB (DistilBERT) + dataset toxic-chat.
- **Notebook treo lâu ở cell huấn luyện DistilBERT** → bình thường trên CPU, xem mục 4;
  tăng `--ExecutePreprocessor.timeout` nếu chạy qua `nbconvert`.
- **Muốn chạy lại nhanh, bỏ qua DistilBERT** → chỉ chạy các cell tới hết mục 4
  (TF-IDF + LinearSVC); sẽ không tái tạo được cascade 3 tầng đầy đủ như trong báo cáo.
