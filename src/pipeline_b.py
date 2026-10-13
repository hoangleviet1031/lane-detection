"""Pipeline B: bird's-eye view (IPM) + color/Sobel threshold + sliding window + polynomial fit."""
import cv2

from src.preprocessing import to_grayscale, gaussian_blur
from src.edge_extraction import combine_color_gradient
from src.perspective import get_perspective_transform, perspective_src_vertices, warp, unwarp_points
from src.lane_candidates import sliding_window_search
from src.curve_fit import fit_polynomial, generate_points


def run_pipeline_b(img, n_points=10, roi_vertices=None, **roi_kwargs):
    """Tra ve dict: {"lanes": [left_points_or_None, right_points_or_None], "binary_warped": ...}.
    left/right_points la list (x, y) TRONG KHONG GIAN ANH GOC (da unwarp), cung format voi Pipeline A.

    roi_vertices / roi_kwargs (vd bottom_width_ratio=...): tuy chinh trapezoid dung cho warp,
    tuong tu tham so ROI cua Pipeline A - khong hardcode 1 bo so cho moi nguon camera."""
    h, w = img.shape[:2]
    vertices = roi_vertices if roi_vertices is not None else perspective_src_vertices(img.shape, **roi_kwargs)
    M, Minv = get_perspective_transform(img.shape, src_vertices=vertices)

    gray = gaussian_blur(to_grayscale(img), ksize=5)
    binary = combine_color_gradient(img, gray)
    binary_warped = warp(binary, M)

    leftx, lefty, rightx, righty = sliding_window_search(binary_warped)

    left_coeffs = fit_polynomial(leftx, lefty, img_height=h)
    right_coeffs = fit_polynomial(rightx, righty, img_height=h)

    # da qua coverage check (>=45% chieu cao) nen tin duoc fit, ve full range ke ca qua
    # khoang trong tu nhien giua cac vach dash (polynomial noi suy qua duoc, khong phai ngoai suy mu)
    y_top, y_bottom = 0, h
    left_warped_pts = generate_points(left_coeffs, y_top, y_bottom, n_points)
    right_warped_pts = generate_points(right_coeffs, y_top, y_bottom, n_points)

    left_pts = unwarp_points(left_warped_pts, Minv) if left_warped_pts else None
    right_pts = unwarp_points(right_warped_pts, Minv) if right_warped_pts else None

    return {
        "lanes": [left_pts, right_pts],
        "binary_warped": binary_warped,
        "roi_vertices": vertices,
    }


def draw_overlay_simple(img, result):
    """Overlay tam thoi de kiem tra Phase 3 bang mat; Phase 6 se thay bang visualize.py."""
    out = img.copy()
    for lane in result["lanes"]:
        if lane is None:
            continue
        pts = [(int(x), int(y)) for x, y in lane]
        for i in range(len(pts) - 1):
            cv2.line(out, pts[i], pts[i + 1], (0, 255, 0), 6)
    cv2.polylines(out, result["roi_vertices"], isClosed=True, color=(255, 0, 0), thickness=2)
    return out
