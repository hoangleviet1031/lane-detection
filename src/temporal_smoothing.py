"""On dinh lane qua cac frame video bang EMA (Exponential Moving Average).

Hoat dong tren dung format diem lane dung chung toan bo project (list (x,y)), nen dung chung
duoc cho ca 3 pipeline ma khong can biet chi tiet ben trong (duong thang hay polynomial bac 2).
Cach lam: resample moi lane ve 1 luoi y co dinh (noi suy tuyen tinh giua cac diem), roi EMA
rieng gia tri x tai tung y trong luoi do qua cac frame.
"""
import numpy as np

EMA_ALPHA = 0.3  # trong so frame moi; alpha nho = muot hon nhung tre hon theo thay doi that
SANITY_MAX_SHIFT_RATIO = 0.12  # lane lech qua 12% chieu rong anh so voi EMA truoc -> coi la outlier
N_REF_POINTS = 10


def _resample(points, ref_ys):
    """Noi suy tuyen tinh x theo y tai cac diem trong ref_ys. points da sap theo y bat ky thu tu."""
    pts = sorted(points, key=lambda p: p[1])
    ys = np.array([p[1] for p in pts])
    xs = np.array([p[0] for p in pts])
    return np.interp(ref_ys, ys, xs)  # np.interp can ys tang dan


class LaneSmoother:
    """1 instance quan ly smoothing cho 1 lane (vd 'left' hoac 'right') cua 1 pipeline tren 1 video.

    y_min, y_max: PHAI la dung khoang y ma pipeline nguon thuc su sinh diem (vd y_top/y_bottom
    cua ROI trong Pipeline A), KHONG duoc dung toan bo chieu cao anh - neu luoi resample rong hon
    khoang du lieu that, np.interp se ngoai suy phang (flat extrapolation) ra ngoai, tao thanh
    doan thang dung vo ly o 2 dau (da gap loi nay luc test tren video moi)."""

    def __init__(self, y_min, y_max, alpha=EMA_ALPHA, n_ref=N_REF_POINTS):
        self.ref_ys = np.linspace(y_min, y_max, n_ref)
        self.alpha = alpha
        self.ema_xs = None  # None = chua co frame nao duoc chap nhan

    def update(self, points, img_width):
        """points: list (x,y) cua frame hien tai, hoac None neu pipeline khong detect duoc.
        Tra ve: list (x,y) da smooth, hoac None neu chua tung co frame nao hop le."""
        if points is None or len(points) < 2:
            return self._current_points()  # giu nguyen gia tri EMA truoc (dong vai tro "du doan")

        current_xs = _resample(points, self.ref_ys)

        if self.ema_xs is None:
            self.ema_xs = current_xs  # frame dau tien, chua co gi de so sanh, nhan luon
        else:
            max_shift = SANITY_MAX_SHIFT_RATIO * img_width
            shift = np.abs(current_xs - self.ema_xs).mean()
            if shift > max_shift:
                pass  # outlier: bo qua frame nay, giu nguyen EMA cu (khong cap nhat)
            else:
                self.ema_xs = self.alpha * current_xs + (1 - self.alpha) * self.ema_xs

        return self._current_points()

    def _current_points(self):
        if self.ema_xs is None:
            return None
        return [(float(x), float(y)) for x, y in zip(self.ema_xs, self.ref_ys)]
