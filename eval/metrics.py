"""Metric danh gia: accuracy/F1 kieu TuSimple (don gian hoa), jitter (frame stability), runtime.

Ghi chu ve accuracy: day la ban DON GIAN HOA tu protocol chinh thuc cua TuSimple (khong phai
CULane IoU-based eval, vi CULane eval goc can binary C++ bien dich rieng va qua phuc tap cho
muc tieu prototype). Thay vi angle-adaptive threshold nhu TuSimple that, dung threshold co dinh
theo ty le chieu rong anh - don gian hon nhung van do duoc dung tinh than "point-level accuracy".
"""
import time

import numpy as np


def _resample_x(points, ref_ys):
    """Noi suy x theo y tai cac moc ref_ys. points: list (x,y) bat ky thu tu."""
    pts = sorted(points, key=lambda p: p[1])
    ys = np.array([p[1] for p in pts])
    xs = np.array([p[0] for p in pts])
    if ys.min() == ys.max():
        return None
    return np.interp(ref_ys, ys, xs, left=np.nan, right=np.nan)


def lane_point_accuracy(gt_points, pred_points, img_width, threshold_ratio=0.02, n_samples=10):
    """So sanh 1 lane du doan voi 1 lane ground-truth tai cac moc y chung (chi trong vung
    ca 2 cung co du lieu - tranh so sanh ngoai suy). Tra ve accuracy trong [0,1], hoac None
    neu khong co vung y chung nao de so sanh.
    threshold_ratio: nguong sai so tinh theo % chieu rong anh (vd 0.02 = 2% = ~33px tren anh 1640px)."""
    if not gt_points or not pred_points:
        return None

    gt_ys = [p[1] for p in gt_points]
    pred_ys = [p[1] for p in pred_points]
    y_lo = max(min(gt_ys), min(pred_ys))
    y_hi = min(max(gt_ys), max(pred_ys))
    if y_hi <= y_lo:
        return None

    ref_ys = np.linspace(y_lo, y_hi, n_samples)
    gt_xs = _resample_x(gt_points, ref_ys)
    pred_xs = _resample_x(pred_points, ref_ys)
    if gt_xs is None or pred_xs is None:
        return None

    threshold_px = threshold_ratio * img_width
    correct = np.abs(gt_xs - pred_xs) < threshold_px
    return float(np.mean(correct))


def evaluate_frame(gt_left, gt_right, pred_left, pred_right, img_width,
                    threshold_ratio=0.02, lane_pass_threshold=0.85):
    """Danh gia 1 frame: tra ve dict voi accuracy tung lane + co detect dung (pass) khong,
    dung de gop thanh precision/recall/F1 o cap dataset (run_eval.py)."""
    left_acc = lane_point_accuracy(gt_left, pred_left, img_width, threshold_ratio)
    right_acc = lane_point_accuracy(gt_right, pred_right, img_width, threshold_ratio)

    accs = [a for a in (left_acc, right_acc) if a is not None]
    overall_acc = float(np.mean(accs)) if accs else None

    return {
        "left_acc": left_acc,
        "right_acc": right_acc,
        "overall_acc": overall_acc,
        "left_pass": (left_acc is not None and left_acc >= lane_pass_threshold),
        "right_pass": (right_acc is not None and right_acc >= lane_pass_threshold),
        "gt_has_left": gt_left is not None and len(gt_left) > 0,
        "gt_has_right": gt_right is not None and len(gt_right) > 0,
        "pred_has_left": pred_left is not None,
        "pred_has_right": pred_right is not None,
    }


def precision_recall_f1(frame_results):
    """Gop ket qua nhieu frame (tu evaluate_frame) thanh precision/recall/F1 cap dataset.
    TP: GT co lane VA pred detect dung (pass). FP: pred detect (co ra lane) nhung GT khong co
    HOAC detect sai (khong pass). FN: GT co lane nhung pred khong detect hoac detect sai."""
    tp = fp = fn = 0
    for r in frame_results:
        for side in ("left", "right"):
            gt_has = r[f"gt_has_{side}"]
            pred_has = r[f"pred_has_{side}"]
            passed = r[f"{side}_pass"]
            if gt_has and pred_has and passed:
                tp += 1
            elif gt_has and (not pred_has or not passed):
                fn += 1
            elif (not gt_has) and pred_has:
                fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def compute_jitter(x_series):
    """Do 'giat' cua vi tri lane qua cac frame: std cua sai khac x giua 2 frame lien tiep.
    x_series: list cac gia tri x (vd x tai y=bottom), co the chua np.nan cho frame mat detect."""
    arr = np.array(x_series, dtype=float)
    valid = arr[~np.isnan(arr)]
    if len(valid) < 2:
        return None
    return float(np.std(np.diff(valid)))


def time_call(fn, *args, **kwargs):
    """Chay fn va do thoi gian (ms). Tra ve (ket_qua, elapsed_ms)."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result, elapsed_ms
