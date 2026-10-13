# Kết quả đánh giá

## Số liệu chính thức: toàn bộ dataset (13,621 ảnh, 341 clip)

Chạy bằng `python -m eval.run_eval --full`. Đây là số liệu đáng tin cậy nhất trong project — chạy trên **toàn bộ** `data/raw/culane/driver_193_90frame`, không phải tập nhỏ đã dùng để tune ROI.

| Pipeline | Accuracy | Precision | Recall | F1 | ms/frame |
|---|---|---|---|---|---|
| A (Canny+Hough+color prefilter) | 60.2% | 89.4% | 9.0% | 16.3% | 11.9 |
| B (bird's-eye+color, ROI đã hiệu chỉnh) | 61.8% | 83.9% | 34.3% | **48.7%** | 45.2 |
| C (UFLD) | **80.8%** | 95.8% | 69.9% | **80.8%** | 124.6 |

**So với số liệu trên 50 ảnh easy set đã dùng để tune ROI (A: 80.8%, B: 89.3%)**: số liệu full dataset thấp hơn rõ rệt (A: 60.2%, B: 61.8%) — xác nhận đúng cảnh báo data leakage đã nêu khi grid search: tune trực tiếp trên tập nhỏ cho ra con số lạc quan hơn khả năng tổng quát hóa thật khá nhiều. Tuy vậy, thứ hạng và kết luận định tính vẫn giữ nguyên: C vượt trội, B hơn A đáng kể ở F1 (48.7% vs 16.3%) dù khoảng cách accuracy thu hẹp lại gần nhau hơn trên tập lớn.

## Số liệu trên tập nhỏ (50 easy + 10 challenging, dùng để tune/debug)

Chạy bằng `python -m eval.run_eval`. Giữ lại để tham khảo quá trình phát triển — **không dùng số này làm kết luận cuối**, xem số liệu full dataset ở trên.

| Pipeline | Dataset | Accuracy | Precision | Recall | F1 | ms/frame |
|---|---|---|---|---|---|---|
| A (Canny+Hough+color prefilter) | easy | 80.8% | 100% | 15.0% | 26.1% | 11.4 |
| B (bird's-eye+color, ROI đã hiệu chỉnh) | easy | 89.3% | 100% | 44.0% | 61.1% | 36.5 |
| C (UFLD) | easy | 81.2% | 100% | 74.0% | 85.1% | 96.2 |
| A (Canny+Hough+color prefilter) | challenging | 32.5% | 0% | 0% | 0% | 11.8 |
| B (bird's-eye+color, ROI đã hiệu chỉnh) | challenging | 44.2% | 14.3% | 8.3% | 10.5% | 122.3 |
| C (UFLD) | challenging | 96.7% | 100% | 91.7% | 95.7% | 91.2 |

*Lịch sử số liệu qua các lần cải tiến trên tập nhỏ (tham khảo): B easy 17.5% (gốc) → 19.4% (fix center dead-zone) → 89.3% (hiệu chỉnh lại kích thước ROI bằng grid search, sau đó xác nhận chỉ 61.8% trên full dataset). Xem toàn bộ quá trình trong REFLECTION.md.*

## ROI của Pipeline B đã được hiệu chỉnh lại bằng grid search (không còn là giá trị đoán tay)

Giá trị ROI ban đầu (Phase 3, chọn bằng mắt trên 1-2 frame) quá rộng, khiến pipeline bao gồm cả vùng xa (nhiễu, ít tin cậy) và cấu trúc không phải lane. Grid search có hệ thống trên toàn bộ easy set tìm ra bộ số hẹp và nông hơn nhiều:

| Tham số | Giá trị cũ | Giá trị mới |
|---|---|---|
| `top_ratio` | 0.58 | 0.56 |
| `bottom_ratio` | 0.92 | 0.68 |
| `top_width_ratio` | 0.28 | 0.11 |
| `bottom_width_ratio` | 0.80 | 0.26 |

**Lưu ý phương pháp luận quan trọng**: bộ số này được tune TRỰC TIẾP trên easy set (cùng tập dùng để báo cáo accuracy) — đây là data leakage (tune-on-test), nên con số 89.3% trên easy set có thể lạc quan hơn khả năng tổng quát hóa thật. Để kiểm chứng, đã đo lại trên **challenging set (hoàn toàn không dùng để tune)**: accuracy vẫn tăng mạnh 1.7%→44.2% — xác nhận cải tiến là thật, không chỉ overfit, dù với dataset nhỏ (50-60 ảnh) vẫn cần thận trọng khi diễn giải con số tuyệt đối.

## Ghi chú đọc kết quả

- **Accuracy**: % điểm lane dự đoán nằm trong ngưỡng 2% chiều rộng ảnh (~33px) so với ground-truth, tính trong vùng y 2 bên cùng có dữ liệu (không ngoại suy).
- **Precision/Recall/F1**: tính ở cấp "lane" — 1 lane được coi là detect đúng (pass) nếu accuracy >= 85%. Precision=100% ở A/B/C cho thấy khi các pipeline này đã tự tin trả ra 1 lane, chúng hiếm khi báo nhầm hoàn toàn so với GT; recall thấp ở A/B phản ánh thiết kế chủ động "thà từ chối còn hơn đoán bừa" (coverage check, MIN_SEGMENTS) đã xây ở Phase 2-3.
- **Pipeline B thấp hơn cả A**: không phải do B tệ hơn A một cách tổng quát, mà do 1 giới hạn cụ thể — trên cảnh nhiều làn xe, sliding-window có thể khóa vào 1 vạch sơn thật nhưng SAI làn (làn kế bên, hoặc vạch tim đường phân chia 2 chiều xe, không phải ranh giới ego-lane mà GT gán nhãn). Đã thêm "center dead-zone" loại trừ dải hẹp quanh tâm ảnh khi tìm histogram peak (fix đúng nguyên nhân cụ thể: vạch tim đường có peak rất mạnh, liên tục, dễ bị chọn nhầm) — cải thiện accuracy 17.5%→19.4%, nhưng vẫn còn giới hạn sâu hơn: sliding window có thể trôi dạt giữa chừng (không chỉ sai ở bước khởi tạo) khi nhiều vạch thật nằm gần nhau. Xem phân tích chi tiết kèm ảnh minh họa trong REFLECTION.md.
- **Pipeline A cải thiện mạnh** nhờ thêm color prefilter (AND edge map với mask "trắng hoặc vàng" trước Hough) — loại được phần lớn false-positive trên cấu trúc không phải vạch sơn (rào chắn, nóc xe). Trên tập nhỏ đã tune, accuracy gần bằng Pipeline C; trên full dataset khoảng cách rõ hơn (60.2% vs 80.8%), nhưng cải thiện vẫn là thật và đáng kể so với baseline gốc.
- **Runtime**: A/B chạy CPU nhanh (ms/frame thấp), C chậm hơn 10-20 lần vì là forward pass của neural network, nhưng vẫn khả thi cho demo (~90-125ms/frame ~ 8-11 FPS).
- **Bài học phương pháp luận cuối cùng**: luôn xác nhận số liệu trên tập dữ liệu lớn/độc lập trước khi kết luận — số liệu tune trên tập nhỏ (dù đã cẩn thận kiểm tra chéo trên challenging set) vẫn có thể lạc quan hơn đáng kể so với thực tế trên toàn bộ dataset.
