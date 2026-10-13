"""Inverse Perspective Mapping (IPM): bien doi anh goc (phoi canh) sang bird's-eye view
(nhin tu tren xuong), noi 2 vach lane gan song song thay vi hoi tu ve 1 diem.
Dung chung 1 hinh thang voi roi.trapezoid_vertices lam src points."""
import cv2
import numpy as np

from src.roi import trapezoid_vertices


def perspective_src_vertices(img_shape, top_ratio=0.56, bottom_ratio=0.68,
                              top_width_ratio=0.11, bottom_width_ratio=0.26):
    """Trapezoid rieng cho warp, KHONG dung chung top_width_ratio hep cua ROI (0.10):
    top qua hep se bi khuech dai hinh quat sau warp (vung gan diem hoi tu la vung xa camera,
    it pixel nhung se bi keo gian manh de lap day chieu rong dst).

    Gia tri mac dinh hien tai (top_ratio=0.56, bottom_ratio=0.68, top_width=0.11,
    bottom_width=0.26) duoc chon bang grid search tren easy set (50 anh CULane) - ROI HEP va
    NONG hon nhieu so ban dau (0.58/0.92/0.28/0.80): accuracy Pipeline B tang 19.4%->89.3% tren
    easy set, VA 1.7%->44.2% tren challenging set (khong dung de tune, xac nhan khong chi la
    overfit). Ly do ROI hep hon lai tot hon: ROI rong truoc do bao ca vung xa (nhieu, it tin cay)
    va cau truc khong phai lane (cay, nha, lan ke ben) lam loang/gay nham tin hieu that.

    CANH BAO phuong phap luan: cac tham so nay duoc tune TRUC TIEP tren cung du lieu dung de bao
    cao accuracy (data leakage/tune-on-test) - con so tren easy set co the lac quan hon kha nang
    tong quat hoa that su. Ket qua tren challenging set (khong dung de tune) dung de kiem chung
    phan nao, nhung voi dataset nho (50-60 anh), van can canh giac khi dien giai con so tuyet doi."""
    return trapezoid_vertices(img_shape, top_ratio=top_ratio, bottom_ratio=bottom_ratio,
                               top_width_ratio=top_width_ratio, bottom_width_ratio=bottom_width_ratio)


def get_perspective_transform(img_shape, src_vertices=None, dst_margin_ratio=0.25):
    """Tra ve (M, Minv). dst la hinh chu nhat, 2 canh doc thang hang."""
    h, w = img_shape[:2]
    if src_vertices is None:
        src_vertices = perspective_src_vertices(img_shape)
    # trapezoid_vertices tra ve thu tu: bottom-left, top-left, top-right, bottom-right
    src = src_vertices[0].astype(np.float32)

    dst_left = w * dst_margin_ratio
    dst_right = w * (1 - dst_margin_ratio)
    dst = np.float32([
        [dst_left, h],
        [dst_left, 0],
        [dst_right, 0],
        [dst_right, h],
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv


def warp(img, M):
    h, w = img.shape[:2]
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)


def unwarp_points(points, Minv):
    """points: list (x, y) trong khong gian bird's-eye -> tra ve list (x, y) trong anh goc."""
    if not points:
        return points
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    unwarped = cv2.perspectiveTransform(pts, Minv)
    return [tuple(p[0]) for p in unwarped]
