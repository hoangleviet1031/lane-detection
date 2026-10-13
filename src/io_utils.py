"""Hàm đọc/ghi ảnh và video dùng chung cho các pipeline A/B/C."""
import os
import cv2


def read_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Khong doc duoc anh: {path}")
    return img


def write_image(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)


def read_video_frames(path, max_frames=None):
    """Trả về generator các frame (BGR) từ video."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Khong mo duoc video: {path}")
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
        count += 1
        if max_frames is not None and count >= max_frames:
            break
    cap.release()


class VideoWriter:
    """Wrapper ghi video overlay ra file, giữ nguyên fps/kich thuoc cua video nguon."""

    def __init__(self, path, fps, frame_size):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(path, fourcc, fps, frame_size)

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()


def get_video_fps(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 0 else 25.0
