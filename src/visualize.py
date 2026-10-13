"""Trung tam truc quan hoa cua project - 2 nhom ham tach biet:
- draw_hud_overlay: danh cho nguoi xem cuoi, giong demo ADAS that.
- draw_debug_steps: danh cho muc tieu hoc/hieu, xem tung buoc xu ly noi bo.
Ca 2 deu nhan input la format diem lane dung chung (list (x,y)), nen dung chung duoc
cho ca 3 pipeline A/B/C."""
import cv2
import numpy as np

from src.deviation import compute_deviation

LANE_FILL_COLOR = (0, 200, 0)
LANE_LINE_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)
ARROW_COLOR = (0, 215, 255)


def _sort_by_y(points):
    return sorted(points, key=lambda p: p[1])


def draw_hud_overlay(img, left_points, right_points, pipeline_name=""):
    """Ve polygon ban trong suot giua 2 lane + mui ten huong lech + text deviation gon gang.
    Khong sua img goc, tra ve anh moi."""
    out = img.copy()
    h, w = img.shape[:2]

    if left_points and right_points:
        left_sorted = _sort_by_y(left_points)
        right_sorted = _sort_by_y(right_points)
        polygon = np.array([[int(x), int(y)] for x, y in left_sorted] +
                            [[int(x), int(y)] for x, y in reversed(right_sorted)], dtype=np.int32)
        overlay = out.copy()
        cv2.fillPoly(overlay, [polygon], LANE_FILL_COLOR)
        out = cv2.addWeighted(overlay, 0.3, out, 0.7, 0)

    for points, is_valid in [(left_points, left_points is not None), (right_points, right_points is not None)]:
        if not is_valid:
            continue
        pts = [(int(x), int(y)) for x, y in _sort_by_y(points)]
        for i in range(len(pts) - 1):
            cv2.line(out, pts[i], pts[i + 1], LANE_LINE_COLOR, 5)

    deviation = compute_deviation(left_points, right_points, w) if (left_points and right_points) else None

    if deviation is not None:
        offset_m, direction = deviation
        arrow_y = int(h * 0.90)
        center_x = w // 2
        # he so va min-length de deviation nho (vai cm) van thay ro mui ten, khong bi "bien mat"
        # dau TRU vi offset_m>0 nghia la lech TRAI (xem deviation.py) -> mui ten phai tro sang trai
        # tren man hinh (dx am), nguoc dau voi offset_m
        arrow_dx = int(np.clip(-offset_m * 150, -180, 180))
        if 0 < abs(arrow_dx) < 40:
            arrow_dx = 40 if arrow_dx > 0 else -40
        if arrow_dx != 0:
            cv2.arrowedLine(out, (center_x, arrow_y), (center_x + arrow_dx, arrow_y),
                             (0, 0, 255), 6, tipLength=0.35)

        label = f"{abs(offset_m):.2f} m {direction}" if direction != "CENTER" else "ON CENTER"
    else:
        label = "LANE NOT DETECTED"

    title = f"{pipeline_name}  {label}" if pipeline_name else label
    cv2.putText(out, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, TEXT_COLOR, 2, cv2.LINE_AA)
    return out


def draw_debug_steps(panels, cols=2, cell_size=(480, 270)):
    """panels: dict {label: anh (grayscale hoac BGR)}. Ghep thanh 1 luoi anh, moi anh resize ve
    cung kich thuoc va co title. Dung de xem 'may nhin thay gi' qua tung buoc xu ly."""
    cw, ch = cell_size
    cells = []
    for label, img in panels.items():
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        resized = cv2.resize(img, (cw, ch))
        cv2.putText(resized, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(resized, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
        cells.append(resized)

    rows = int(np.ceil(len(cells) / cols))
    pad = np.zeros((ch, cw, 3), dtype=np.uint8)
    while len(cells) < rows * cols:
        cells.append(pad)

    row_imgs = [np.hstack(cells[r * cols:(r + 1) * cols]) for r in range(rows)]
    return np.vstack(row_imgs)
