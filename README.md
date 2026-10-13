# Lane Detection for Driver Assistance

Prototype nhận diện vạch làn đường theo hướng ADAS (Advanced Driver Assistance System), so sánh định lượng ba pipeline trên cùng một bài toán:

| | Phương pháp |
|---|---|
| **Pipeline A** | Canny/Sobel edge detection + color prefilter + Hough Transform |
| **Pipeline B** | Inverse Perspective Mapping (bird's-eye view) + color/gradient threshold + sliding window + polynomial fit |
| **Pipeline C** | Ultra-Fast-Lane-Detection (UFLD) — ResNet-18, pretrained trên CULane, chỉ inference |

Dữ liệu: tập con [CULane](https://xingangpan.github.io/projects/CULane.html) (`driver_193_90frame`, 342 clip, 13.621 frame).

## Kết quả (toàn bộ 13.621 ảnh)

| Pipeline | Accuracy | Precision | Recall | F1 | ms/frame (CPU) |
|---|---|---|---|---|---|
| A | 60,2% | 89,4% | 9,0% | 16,3% | 11,9 |
| B | 61,8% | 83,9% | 34,3% | 48,7% | 45,2 |
| C | **80,8%** | 95,8% | 69,9% | **80,8%** | 124,6 |

Chi tiết số liệu, phân tích lỗi và case study: xem [`report/report.md`](report/report.md) và [`eval/results.md`](eval/results.md).

## Cài đặt

```bash
pip install -r requirements.txt
```

Yêu cầu Python 3.10+. `requirements.txt` cài PyTorch bản CPU; nếu máy có GPU CUDA, pipeline C sẽ tự động dùng GPU (code device-agnostic), không cần chỉnh gì thêm.

## Chuẩn bị dữ liệu

CULane không thể tải trực tiếp bằng script (phân phối qua Google Drive/Baidu, không có mirror chính thức dạng wget-able):

1. Vào trang chính thức: https://xingangpan.github.io/projects/CULane.html
2. Tải phần ảnh (ví dụ `driver_193_90frame`) và giải nén vào `data/raw/culane/driver_193_90frame/`, giữ đúng cấu trúc gốc (`<clip>/<frame>.jpg` + `<frame>.lines.txt`).
3. (Tùy chọn) Tạo lại tập mẫu nhỏ dùng để phát triển/debug:
   ```bash
   python src/prepare_dataset.py
   ```
   Script này copy một subset nhỏ (50 ảnh điều kiện thuận lợi + 10 ảnh ban đêm) vào `data/samples/`. Tập mẫu này đã có sẵn trong repo, không bắt buộc chạy lại trừ khi muốn đổi clip khác.

## Tải trọng số mô hình (Pipeline C)

```bash
pip install gdown
python -m gdown "https://drive.google.com/uc?id=1zXBRTw50WOzvUp6XKsi8Zrk3MUC3uFuq" -O weights/culane_18.pth
```

Checkpoint chính thức (CULane, ResNet-18) từ repo gốc [cfzd/Ultra-Fast-Lane-Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection).

## Chạy đánh giá

```bash
# Tập phát triển (60 ảnh, nhanh)
python -m eval.run_eval

# Toàn bộ dataset đã tải (13.6k ảnh, ~30-40 phút trên CPU)
python -m eval.run_eval --full
```

## Cấu trúc mã nguồn

```
src/
├── io_utils.py           # Đọc/ghi ảnh, video
├── preprocessing.py      # Grayscale, Gaussian blur
├── edge_extraction.py    # Canny, Sobel, color threshold
├── roi.py                # Region of interest (hình thang)
├── perspective.py        # Inverse Perspective Mapping (IPM)
├── lane_candidates.py    # Hough Transform, sliding window
├── curve_fit.py          # Polynomial fit
├── temporal_smoothing.py # EMA + outlier rejection cho video
├── deviation.py          # Tính lane-center deviation
├── visualize.py          # HUD overlay + debug-view
├── pipeline_a.py          # Pipeline A end-to-end
├── pipeline_b.py          # Pipeline B end-to-end
├── pipeline_c.py          # Pipeline C end-to-end (UFLD)
├── ufld_model.py          # Kiến trúc UFLD (ResNet-18 + row-anchor head)
├── culane_loader.py        # Parse annotation CULane (.lines.txt)
└── prepare_dataset.py     # Script chọn/copy subset mẫu từ dataset gốc

eval/
├── metrics.py     # Accuracy/F1, jitter, runtime
├── run_eval.py    # Chạy đánh giá trên tập mẫu hoặc toàn bộ dataset
└── results.md     # Bảng kết quả

data/
├── raw/          # Dataset CULane gốc đã tải (không commit, tự tải theo hướng dẫn trên)
└── samples/      # Subset nhỏ (easy/challenging) dùng để phát triển, đã có sẵn trong repo

weights/           # Checkpoint UFLD pretrained (không commit, tự tải theo hướng dẫn trên)
report/            # Báo cáo kết quả (report.md) kèm ảnh minh họa
```

## Ghi chú

- Toàn bộ pipeline chạy device-agnostic (tự dùng GPU nếu có, fallback CPU).
- ROI/tham số hình học được hiệu chỉnh cho khung hình dashcam kiểu CULane; với nguồn camera khác có thể cần tinh chỉnh lại (xem phân tích trong `report/report.md`, mục "Phân tích lỗi và thảo luận").
- Dữ liệu CULane dùng cho mục đích nghiên cứu/phi thương mại theo giấy phép gốc của bộ dữ liệu.
