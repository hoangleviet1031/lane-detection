"""Chay ca 3 pipeline tren easy set + challenging set, xuat bang so sanh accuracy/F1, jitter
(uoc luong tu Phase 5), runtime. Chay: python -m eval.run_eval"""
import glob
import os

from src.culane_loader import parse_lines_txt
from src.io_utils import read_image
from src.pipeline_a import run_pipeline_a
from src.pipeline_b import run_pipeline_b
from src.pipeline_c import load_model, run_pipeline_c
from eval.metrics import evaluate_frame, precision_recall_f1, time_call

IMG_WIDTH = 1640  # anh CULane goc


def pick_ego_lanes(gt_lanes, img_width):
    """Chon 2 lane GT gan tam anh nhat (1 ben trai, 1 ben phai) lam ego-lane boundary,
    thay vi gia dinh cung slot index co dinh - vi so luong lane ghi trong .lines.txt
    thay doi theo tung anh (chi ghi lane nao con nhin thay duoc)."""
    cx = img_width / 2
    left_candidates, right_candidates = [], []
    for lane in gt_lanes:
        if not lane:
            continue
        bottom_x = max(lane, key=lambda p: p[1])[0]
        if bottom_x < cx:
            left_candidates.append((cx - bottom_x, lane))
        else:
            right_candidates.append((bottom_x - cx, lane))
    left = min(left_candidates, key=lambda t: t[0])[1] if left_candidates else None
    right = min(right_candidates, key=lambda t: t[0])[1] if right_candidates else None
    return left, right


def load_dataset(folder, recursive=False):
    """recursive=True: quet ca cac thu muc con (vd data/raw/culane/driver_193_90frame/<clip>/*.jpg) -
    dung cho eval tren toan bo dataset da tai, khac voi data/samples/ (flat, subset nho)."""
    pattern = os.path.join(folder, "**", "*.jpg") if recursive else os.path.join(folder, "*.jpg")
    samples = []
    for jpg_path in sorted(glob.glob(pattern, recursive=recursive)):
        txt_path = jpg_path.rsplit(".", 1)[0] + ".lines.txt"
        if not os.path.exists(txt_path):
            continue
        gt_lanes = parse_lines_txt(txt_path)
        samples.append((jpg_path, gt_lanes))
    return samples


def run_pipeline_c_wrapper(img, model):
    result = run_pipeline_c(img, model)
    return {"lanes": [result["lanes"][1], result["lanes"][2]], "infer_ms": result["infer_ms"]}


def evaluate_dataset(name, samples, model_c, progress_every=None):
    print(f"\n=== {name} ({len(samples)} anh) ===", flush=True)
    pipeline_results = {"A": [], "B": [], "C": []}
    pipeline_runtime = {"A": [], "B": [], "C": []}

    for i, (jpg_path, gt_lanes) in enumerate(samples):
        img = read_image(jpg_path)
        gt_left, gt_right = pick_ego_lanes(gt_lanes, IMG_WIDTH)

        out_a, ms_a = time_call(run_pipeline_a, img, edge_method="canny")
        out_b, ms_b = time_call(run_pipeline_b, img)
        out_c, ms_c = time_call(run_pipeline_c_wrapper, img, model_c)

        for key, out, ms in [("A", out_a, ms_a), ("B", out_b, ms_b), ("C", out_c, ms_c)]:
            left, right = out["lanes"][0], out["lanes"][1]
            res = evaluate_frame(gt_left, gt_right, left, right, IMG_WIDTH)
            pipeline_results[key].append(res)
            pipeline_runtime[key].append(ms)

        if progress_every and (i + 1) % progress_every == 0:
            print(f"  ... {i + 1}/{len(samples)} anh", flush=True)

    rows = []
    for key in ("A", "B", "C"):
        prf = precision_recall_f1(pipeline_results[key])
        accs = [r["overall_acc"] for r in pipeline_results[key] if r["overall_acc"] is not None]
        mean_acc = sum(accs) / len(accs) if accs else 0.0
        mean_ms = sum(pipeline_runtime[key]) / len(pipeline_runtime[key])
        rows.append({
            "pipeline": key, "dataset": name, "mean_acc": mean_acc,
            "precision": prf["precision"], "recall": prf["recall"], "f1": prf["f1"],
            "mean_ms": mean_ms,
        })
    return rows


def print_table(rows):
    header = f"{'Pipeline':8} {'Dataset':12} {'Acc':>6} {'Prec':>6} {'Recall':>6} {'F1':>6} {'ms/frame':>9}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['pipeline']:8} {r['dataset']:12} {r['mean_acc']*100:5.1f}% "
              f"{r['precision']*100:5.1f}% {r['recall']*100:5.1f}% {r['f1']*100:5.1f}% "
              f"{r['mean_ms']:8.1f}")


def main():
    model_c = load_model("weights/culane_18.pth")

    easy_samples = load_dataset("data/samples/easy")
    challenging_samples = load_dataset("data/samples/challenging")

    all_rows = []
    all_rows += evaluate_dataset("easy", easy_samples, model_c)
    all_rows += evaluate_dataset("challenging", challenging_samples, model_c)

    print("\n=== Bang tong hop ===")
    print_table(all_rows)


def main_full():
    """Chay tren TOAN BO dataset da tai (342 clip, ~13.6k frame) - khac voi main() chi dung
    subset nho da chon tay o data/samples/. Chay: python -m eval.run_eval --full"""
    model_c = load_model("weights/culane_18.pth")
    samples = load_dataset("data/raw/culane/driver_193_90frame", recursive=True)
    rows = evaluate_dataset("full_dataset (driver_193_90frame)", samples, model_c, progress_every=500)
    print("\n=== Bang tong hop (full dataset) ===")
    print_table(rows)
    return rows


if __name__ == "__main__":
    import sys
    if "--full" in sys.argv:
        main_full()
    else:
        main()
