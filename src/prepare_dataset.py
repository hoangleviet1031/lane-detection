"""Chon va copy mot subset nho tu CULane da tai (data/raw/culane/driver_193_90frame)
thanh data/samples/easy va data/samples/challenging, dung cho prototype (khong can full 13k frame).

Da chon bang mat (xem vai frame dai dien tu cac clip khac nhau):
- easy: clip ban ngay, duong thang, vach ro.
- challenging: clip quay ban dem (thieu sang, loa den pha) -> giong category "Night" cua CULane goc.
"""
import os
import shutil

RAW_DIR = "data/raw/culane/driver_193_90frame"
SAMPLES_DIR = "data/samples"

EASY_CLIPS = [
    "06051502_0708.MP4",
    "06051600_0726.MP4",
    "06051147_0643.MP4",
    "06051740_0759.MP4",
    "06051532_0718.MP4",
]

CHALLENGING_CLIPS = [
    "06042110_0531.MP4",  # night, thieu sang + loa den pha
]

FRAMES_PER_CLIP = 10  # lay toi da N frame dau moi clip, du dung cho prototype


def collect_frame_stems(clip_dir, max_frames=FRAMES_PER_CLIP):
    files = sorted(f for f in os.listdir(clip_dir) if f.endswith(".jpg"))
    return [f[:-4] for f in files[:max_frames]]


def copy_subset(clip_names, dest_subdir):
    dest_dir = os.path.join(SAMPLES_DIR, dest_subdir)
    os.makedirs(dest_dir, exist_ok=True)
    count = 0
    for clip in clip_names:
        clip_dir = os.path.join(RAW_DIR, clip)
        if not os.path.isdir(clip_dir):
            print(f"[canh bao] khong thay clip: {clip_dir}")
            continue
        for stem in collect_frame_stems(clip_dir):
            jpg_src = os.path.join(clip_dir, f"{stem}.jpg")
            txt_src = os.path.join(clip_dir, f"{stem}.lines.txt")
            tag = clip.replace(".MP4", "")
            jpg_dst = os.path.join(dest_dir, f"{tag}_{stem}.jpg")
            txt_dst = os.path.join(dest_dir, f"{tag}_{stem}.lines.txt")
            shutil.copy2(jpg_src, jpg_dst)
            if os.path.exists(txt_src):
                shutil.copy2(txt_src, txt_dst)
            count += 1
    print(f"Da copy {count} frame vao {dest_dir}")


if __name__ == "__main__":
    copy_subset(EASY_CLIPS, "easy")
    copy_subset(CHALLENGING_CLIPS, "challenging")
