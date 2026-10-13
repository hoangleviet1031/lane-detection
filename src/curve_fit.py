"""Fit duong cong bac 2 (x = a*y^2 + b*y + c) cho tap pixel lane tim duoc tu sliding window,
roi sinh ra list diem (x, y) theo dung format dung chung toan bo project."""
import numpy as np

MIN_POINTS_TO_FIT = 50  # can toi thieu ngan ngua fit tren qua it pixel (vd lane bi che khuat)
MIN_Y_COVERAGE_RATIO = 0.45  # pixel tim duoc phai trai dai >=45% chieu cao anh warp moi tin duoc


def fit_polynomial(xs, ys, deg=2, img_height=None):
    """Tra ve None neu qua it diem HOAC cac diem qua tap trung o 1 doan ngan (y-range hep):
    fit tren doan ngan roi ngoai suy ra ca anh se cho duong cong vo ly ngoai vung co du lieu that."""
    if len(xs) < MIN_POINTS_TO_FIT:
        return None
    if img_height is not None:
        y_coverage = (ys.max() - ys.min()) / img_height
        if y_coverage < MIN_Y_COVERAGE_RATIO:
            return None
    return np.polyfit(ys, xs, deg=deg)


def generate_points(coeffs, y_top, y_bottom, n_points=10):
    """Sinh n_points diem (x, y) doc theo duong cong tu y_bottom len y_top (gan -> xa),
    dung np.polyval de noi suy x tai moi y."""
    if coeffs is None:
        return None
    ys = np.linspace(y_bottom, y_top, n_points)
    xs = np.polyval(coeffs, ys)
    return [(float(x), float(y)) for x, y in zip(xs, ys)]
