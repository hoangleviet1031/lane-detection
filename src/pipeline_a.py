"""Pipeline A: Canny edge + Hough transform tren khong gian anh goc (khong warp)."""
import cv2
import numpy as np

from src.preprocessing import to_grayscale, gaussian_blur
from src.roi import apply_roi_mask, trapezoid_vertices
from src.edge_extraction import canny_edges, sobel_edges, lane_color_mask
from src.lane_candidates import hough_segments, separate_by_slope, average_line


def run_pipeline_a(img, edge_method="canny", roi_vertices=None, roi_top_ratio=0.58, roi_bottom_ratio=0.92,
                    use_color_prefilter=True):
    """Tra ve dict: {"lanes": [left_points_or_None, right_points_or_None], "edges": edge_map}.
    left/right_points la [(x_bottom, y_bottom), (x_top, y_top)] dung chung format diem lane.

    roi_vertices: truyen vao neu muon tu chinh ROI (vd video co ty le khung hinh/vi tri camera
    khac CULane) thay vi dung default ratio - ROI la gia dinh ve camera, khong nen hardcode cung
    cho moi nguon video (bai hoc tu Phase 2).

    use_color_prefilter: AND edge map voi mask 'trang hoac vang' (lane_color_mask) truoc Hough,
    de loai cau truc hinh hoc giong lane nhung sai mau (vd rao chan kim loai xam) - fix cho loi
    phat hien o Phase 2 (xem REFLECTION.md)."""
    h, w = img.shape[:2]
    gray = gaussian_blur(to_grayscale(img), ksize=5)

    if edge_method == "canny":
        edges = canny_edges(gray, 50, 150)
    elif edge_method == "sobel":
        edges = sobel_edges(gray, ksize=3, thresh=50)
    else:
        raise ValueError(f"edge_method khong hop le: {edge_method}")

    if use_color_prefilter:
        color_mask = lane_color_mask(img, gray)
        color_mask = cv2.dilate(color_mask, np.ones((5, 5), np.uint8))  # no rong vai px, tranh mat edge
        # sat canh vach that do anti-aliasing
        edges = cv2.bitwise_and(edges, color_mask)

    if roi_vertices is None:
        roi_vertices = trapezoid_vertices(img.shape, top_ratio=roi_top_ratio, bottom_ratio=roi_bottom_ratio)
    vertices = roi_vertices
    roi_edges = apply_roi_mask(edges, vertices)

    segments = hough_segments(roi_edges)
    left_segs, right_segs = separate_by_slope(segments, w)

    y_top = int(h * roi_top_ratio)
    y_bottom = int(h * roi_bottom_ratio)
    left_line = average_line(left_segs, y_top, y_bottom)
    right_line = average_line(right_segs, y_top, y_bottom)

    return {"lanes": [left_line, right_line], "edges": roi_edges, "roi_vertices": vertices}


def draw_overlay_simple(img, result):
    """Overlay tam thoi de kiem tra ket qua Phase 2 bang mat; Phase 6 se thay bang visualize.py."""
    out = img.copy()
    for line in result["lanes"]:
        if line is None:
            continue
        (x1, y1), (x2, y2) = line
        cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 6)
    cv2.polylines(out, result["roi_vertices"], isClosed=True, color=(255, 0, 0), thickness=2)
    return out
