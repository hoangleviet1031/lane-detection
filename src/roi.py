"""Region of interest: mask hinh thang chi giu vung duong phia truoc xe,
loai bo bau troi, cay hai ben, va dashboard o day anh."""
import numpy as np
import cv2


def trapezoid_vertices(img_shape, top_ratio=0.58, bottom_ratio=0.92,
                        top_width_ratio=0.10, bottom_width_ratio=0.80):
    h, w = img_shape[:2]
    top_y = int(h * top_ratio)
    bottom_y = int(h * bottom_ratio)
    top_half_w = int(w * top_width_ratio / 2)
    bottom_half_w = int(w * bottom_width_ratio / 2)
    cx = w // 2
    return np.array([[
        (cx - bottom_half_w, bottom_y),
        (cx - top_half_w, top_y),
        (cx + top_half_w, top_y),
        (cx + bottom_half_w, bottom_y),
    ]], dtype=np.int32)


def apply_roi_mask(img, vertices=None):
    """Tra ve anh da mask (vung ngoai ROI = 0). img co the la anh xam/edge (2D) hoac mau (3D)."""
    if vertices is None:
        vertices = trapezoid_vertices(img.shape)
    mask = np.zeros_like(img)
    fill_color = 255 if img.ndim == 2 else (255,) * img.shape[2]
    cv2.fillPoly(mask, vertices, fill_color)
    return cv2.bitwise_and(img, mask)
