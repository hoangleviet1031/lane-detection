"""Pipeline C: UFLD pretrained (CULane, ResNet-18) - chi inference, khong train.
Decode logic (softmax expectation tren grid cell) copy dung cong thuc tu demo.py goc
de dam bao doc dung output cua checkpoint pretrained."""
import time

import numpy as np
import scipy.special
import torch
import torchvision.transforms as T

from src.ufld_model import ParsingNet, load_pretrained

CULANE_ROW_ANCHOR = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]
GRIDING_NUM = 200
NUM_LANES = 4
INPUT_H, INPUT_W = 288, 800
MIN_POINTS_PER_LANE = 2  # can >2 row anchor co detect moi nhan la 1 lane hop le (giong demo.py goc)

_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((INPUT_H, INPUT_W)),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path, device=None):
    device = device or get_device()
    model = ParsingNet(cls_dim=(GRIDING_NUM + 1, len(CULANE_ROW_ANCHOR), NUM_LANES))
    load_pretrained(model, checkpoint_path, device)
    model = model.to(device).eval()
    return model


def decode_output(out, img_w, img_h):
    """out: tensor (1, 201, 18, 4) tu model. Tra ve list 4 lane, moi lane la list (x, y)
    (chi cac diem hop le, da loc nhung row anchor model du doan 'khong co lane')."""
    out_np = out[0].detach().cpu().numpy()  # (201, 18, 4)
    out_np = out_np[:, ::-1, :]  # dao chieu row anchor (model doc tu duoi len, ta muon tren xuong)

    prob = scipy.special.softmax(out_np[:-1, :, :], axis=0)  # bo lop cuoi (no-lane) khi tinh expectation
    idx = np.arange(GRIDING_NUM) + 1
    idx = idx.reshape(-1, 1, 1)
    loc = np.sum(prob * idx, axis=0)  # (18, 4) - vi tri grid cell ky vong (expectation), lien tuc

    argmax = np.argmax(out_np, axis=0)  # (18, 4) - class co xac suat cao nhat
    loc[argmax == GRIDING_NUM] = 0  # grid cell cuoi (index 200) = lop "khong co lane" -> loai

    col_sample = np.linspace(0, INPUT_W - 1, GRIDING_NUM)
    col_sample_w = col_sample[1] - col_sample[0]

    lanes = []
    for lane_i in range(NUM_LANES):
        col = loc[:, lane_i]
        if np.sum(col != 0) <= MIN_POINTS_PER_LANE:
            lanes.append(None)
            continue
        points = []
        for k in range(len(CULANE_ROW_ANCHOR)):
            if col[k] <= 0:
                continue
            x = col[k] * col_sample_w * img_w / INPUT_W - 1
            # da dao nguoc row anchor o tren nen dung chi so doi xung k -> len(row_anchor)-1-k
            y = img_h * (CULANE_ROW_ANCHOR[len(CULANE_ROW_ANCHOR) - 1 - k] / INPUT_H) - 1
            points.append((float(x), float(y)))
        lanes.append(points if points else None)
    return lanes


def run_pipeline_c(img, model, device=None):
    """img: BGR (nhu cv2.imread). Tra ve dict {"lanes": [lane0,...,lane3], "infer_ms": float}.
    lane0..3 la 4 lane CULane chuan (thu tu trai-xa, trai-gan, phai-gan, phai-xa);
    Pipeline A/B chi co 2 lane (trai/phai gan nhat) nen khi so sanh se lay lane index 1 va 2."""
    device = device or get_device()
    img_rgb = img[:, :, ::-1]  # BGR -> RGB vi model train tren anh RGB
    h, w = img.shape[:2]

    tensor = _transform(img_rgb).unsqueeze(0).to(device)

    start = time.perf_counter()
    with torch.no_grad():
        out = model(tensor)
    infer_ms = (time.perf_counter() - start) * 1000

    lanes = decode_output(out, img_w=w, img_h=h)
    return {"lanes": lanes, "infer_ms": infer_ms}


def draw_overlay_simple(img, result):
    """Overlay tam thoi de kiem tra Phase 4 bang mat; Phase 6 se thay bang visualize.py."""
    import cv2
    out = img.copy()
    colors = [(255, 0, 255), (0, 255, 0), (0, 255, 0), (255, 0, 255)]  # highlight 2 lane giua (trai/phai gan nhat)
    for lane, color in zip(result["lanes"], colors):
        if lane is None:
            continue
        for x, y in lane:
            cv2.circle(out, (int(x), int(y)), 5, color, -1)
    return out
