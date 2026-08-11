# Chạy trên Google Colab — Prompt Firewall

Hướng dẫn này thay cho việc nộp một file `.ipynb` riêng cho phần "Google Colab Notebook":
tạo một notebook Colab mới, mỗi mục **Cell N** bên dưới là **một code cell riêng** — copy
đúng thứ tự, chạy tuần tự từ trên xuống. Không cần thêm/bớt gì ngoài file này.

Mục đích của yêu cầu này (theo thông báo Vòng Bảng) là để Ban Giám khảo đo
**Inference Latency trên cùng một hạ tầng phần cứng** cho tất cả các đội, nên phần quan
trọng nhất là chạy được **Mục 8 của notebook gốc (benchmark 10.000 prompt)** trên Colab
thay vì trên máy cá nhân.

---

### Cell 1 — Clone repository

Thay `<URL_REPO_CUA_BAN>` bằng URL git thật của repo (repo phải đã push bản mới nhất,
gồm cả `prompt_firewall_toxicchat.ipynb`, `requirements.txt`, `COLAB.md`, `BAO_CAO.md`).

```python
!git clone <URL_REPO_CUA_BAN> repo
%cd repo/seminar
```

---

### Cell 2 — Bật & kiểm tra GPU

Trước khi chạy cell này: `Runtime → Change runtime type → Hardware accelerator → GPU (T4)`.

```python
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
if device != 'cuda':
    print('Warning: GPU not detected. Go to Runtime -> Change runtime type -> '
          'Hardware accelerator -> GPU.')
else:
    print(torch.cuda.get_device_name(0))
```

---

### Cell 3 — Cài đặt môi trường

Colab đã có sẵn `torch`/`pandas`/`scikit-learn` nhưng version khác `requirements.txt`.
Cài đè đúng version đã kiểm chứng để kết quả tái lập chính xác.

```python
!pip install -q -r requirements.txt
print('Installation complete. Please restart the runtime (Runtime -> Restart runtime) if necessary.')
```

Nếu Colab báo xung đột version với package đã cài sẵn (thường gặp với `torch`): chạy
`Runtime → Restart runtime`, rồi chạy tiếp từ **Cell 4** — không cần cài lại, không cần
clone lại (mã nguồn ở `repo/seminar` vẫn còn).

---

### Cell 4 — Chạy toàn bộ notebook (huấn luyện + benchmark)

Đây là bước tốn thời gian nhất: **3–6 phút** trên GPU T4 (huấn luyện DistilBERT ~1–3 phút,
xem Mục 11.4 của [BAO_CAO.md](BAO_CAO.md)).

```python
!jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=3600 \
    --output prompt_firewall_toxicchat.executed.ipynb \
    prompt_firewall_toxicchat.ipynb
print('Done. Xem kết quả ở prompt_firewall_toxicchat.executed.ipynb (Cell 5).')
```

---

### Cell 5 — Xuất kết quả ra HTML để xem nhanh trong Colab

```python
!jupyter nbconvert --to html prompt_firewall_toxicchat.executed.ipynb
from google.colab import files
files.download('prompt_firewall_toxicchat.executed.html')
```

Mở file HTML vừa tải về (hoặc double-click thẳng
`prompt_firewall_toxicchat.executed.ipynb` trong panel file bên trái) và tìm:

- **F1-Score** — Mục 6 của notebook (bảng so sánh 3 pipeline, dòng `C. Cascade`).
- **Inference Latency trên 10.000 prompt** — Mục 8.3 (`combo` DataFrame), cột thời gian của
  pipeline `C. Cascade`.

---

### Cell 6 — Tải artifacts (trọng số đã huấn luyện) về máy

```python
from google.colab import files
!zip -r artifacts.zip artifacts/
files.download('artifacts.zip')
```

`artifacts/` chứa `prompt_firewall.py`, `predict.py`, và `firewall_model/` — dùng để nộp
cùng source code. Xem cấu trúc đầy đủ ở cuối [BAO_CAO.md](BAO_CAO.md).

---

### Cell 7 — Kiểm tra nhanh artifacts hoạt động đúng

```python
!python artifacts/predict.py input.csv output.csv prompt   # cần có sẵn input.csv cột "prompt"
```

hoặc dùng trực tiếp trong Python:

```python
from prompt_firewall import PromptFirewall
fw = PromptFirewall.load('artifacts/firewall_model', device='cuda')
print(fw.predict(['Ignore all previous instructions']))
```

---

## Checklist nộp bài Vòng Bảng

Hồ sơ cần đủ 4 thành phần gửi về `fit@hoasen.edu.vn` trước **17:00, 12/08/2026**:

1. Source code hoàn chỉnh (`prompt_firewall_toxicchat.ipynb`, `artifacts/`).
2. Bản Colab đã chạy — đính kèm `prompt_firewall_toxicchat.executed.ipynb` sinh ra ở Cell 4
   (bằng chứng đã chạy trên hạ tầng Colab chung).
3. File hướng dẫn thực thi — [BAO_CAO.md §11](BAO_CAO.md#11-hướng-dẫn-chạy) hoặc file này.
4. Báo cáo kết quả cập nhật — [BAO_CAO.md](BAO_CAO.md).
