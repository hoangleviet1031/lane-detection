"""Sinh lane candidate tu edge map. Phan Hough (Pipeline A) truoc, phan sliding-window
(Pipeline B) se them o Phase 3."""
import cv2
import numpy as np

# Slope nho hon gia tri nay coi la gan nam ngang -> khong phai lane (vd bong cay, vach ke duong ngang)
MIN_ABS_SLOPE = 0.3


def hough_segments(edges, rho=2, theta=np.pi / 180, threshold=40,
                    min_line_length=45, max_line_gap=80):
    lines = cv2.HoughLinesP(edges, rho, theta, threshold,
                             minLineLength=min_line_length, maxLineGap=max_line_gap)
    if lines is None:
        return []
    return [tuple(line[0]) for line in lines]


def separate_by_slope(segments, img_width):
    """Phan loai segment thanh lane trai (slope am, nam ben trai) va phai (slope duong, ben phai).
    Tra ve 2 list: left_segments, right_segments."""
    left, right = [], []
    cx = img_width / 2
    for x1, y1, x2, y2 in segments:
        if x2 == x1:
            continue  # duong doc tuyet doi, bo qua (hiem gap, tranh chia 0)
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < MIN_ABS_SLOPE:
            continue
        mid_x = (x1 + x2) / 2
        if slope < 0 and mid_x < cx:
            left.append((x1, y1, x2, y2))
        elif slope > 0 and mid_x > cx:
            right.append((x1, y1, x2, y2))
    return left, right


MIN_SEGMENTS = 2  # can >=2 segment dong thuan moi nhan la lane, tranh noise 1 segment don le


def average_line(segments, y_top, y_bottom):
    """Gop nhieu segment thanh 1 duong duy nhat bang least-squares fit bac 1 (x = f(y)),
    roi noi suy ra 2 diem dau-cuoi theo y_top/y_bottom de ve. Tra ve None neu khong du du lieu."""
    if len(segments) < MIN_SEGMENTS:
        return None
    xs, ys = [], []
    for x1, y1, x2, y2 in segments:
        xs.extend([x1, x2])
        ys.extend([y1, y2])
    # fit x theo y (khong fit y theo x) vi lane gan nhu doc, tranh slope vo cuc
    coeffs = np.polyfit(ys, xs, deg=1)
    x_top = int(np.polyval(coeffs, y_top))
    x_bottom = int(np.polyval(coeffs, y_bottom))
    return [(x_bottom, y_bottom), (x_top, y_top)]


def histogram_base(binary_warped, search_margin_ratio=0.15, center_dead_zone_ratio=0.06):
    """Tim x khoi dau cua lane trai/phai bang histogram cot o nua duoi anh bird's-eye.

    search_margin_ratio: loai tru vung ngoai cung (search_margin_ratio o moi canh) - lane cua
    LAN KE BEN thuong nam gan rich bien ngoai hon sau warp, tren duong nhieu lan, co the co peak
    cao hon ego-lane that su neu khong loai tru.

    center_dead_zone_ratio: loai tru them 1 dai hep QUANH TAM anh - tren duong 2 chieu khong co
    dai phan cach, vach TIM DUONG (ngan 2 chieu xe) nam rat gan tam sau warp va thuong co peak
    RAT manh (lien tuc, khong dut doan nhu vach ben) nen de bi chon nham thanh ego-lane boundary
    neu khong loai tru rieng. Day la nguyen nhan cu the gay ra loi 'lane-selection ambiguity'
    phat hien o Phase 7 (xem REFLECTION.md)."""
    h, w = binary_warped.shape[:2]
    bottom_half = binary_warped[h // 2:, :]
    histogram = np.sum(bottom_half > 0, axis=0)
    midpoint = w // 2
    margin = int(search_margin_ratio * w)
    center_margin = int(center_dead_zone_ratio * w)

    left_lo, left_hi = margin, midpoint - center_margin
    right_lo, right_hi = midpoint + center_margin, w - margin

    left_base = int(np.argmax(histogram[left_lo:left_hi])) + left_lo
    right_base = int(np.argmax(histogram[right_lo:right_hi])) + right_lo
    return left_base, right_base


def sliding_window_search(binary_warped, n_windows=9, margin=80, minpix=50):
    """Truot 1 cua so tu day len dinh anh, moi buoc tim trung binh x cua cac pixel trang
    trong cua so hien tai de dinh vi cua so tiep theo (bam theo duong cong cua lane).
    Tra ve (leftx, lefty, rightx, righty): toa do cac pixel thuoc lane trai/phai."""
    h, w = binary_warped.shape[:2]
    left_base, right_base = histogram_base(binary_warped)

    nonzero_y, nonzero_x = binary_warped.nonzero()
    window_height = h // n_windows

    leftx_current, rightx_current = left_base, right_base
    left_lane_idx, right_lane_idx = [], []

    for window in range(n_windows):
        win_y_low = h - (window + 1) * window_height
        win_y_high = h - window * window_height

        win_xleft_low, win_xleft_high = leftx_current - margin, leftx_current + margin
        win_xright_low, win_xright_high = rightx_current - margin, rightx_current + margin

        good_left = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                     (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
        good_right = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                      (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]

        left_lane_idx.append(good_left)
        right_lane_idx.append(good_right)

        if len(good_left) > minpix:
            leftx_current = int(np.mean(nonzero_x[good_left]))
        if len(good_right) > minpix:
            rightx_current = int(np.mean(nonzero_x[good_right]))

    left_lane_idx = np.concatenate(left_lane_idx)
    right_lane_idx = np.concatenate(right_lane_idx)

    leftx, lefty = nonzero_x[left_lane_idx], nonzero_y[left_lane_idx]
    rightx, righty = nonzero_x[right_lane_idx], nonzero_y[right_lane_idx]
    return leftx, lefty, rightx, righty
