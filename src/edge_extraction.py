"""Hai cach trich xuat edge evidence de so sanh: Canny vs Sobel."""
import cv2
import numpy as np


def canny_edges(gray, low_thresh=50, high_thresh=150):
    return cv2.Canny(gray, low_thresh, high_thresh)


def sobel_edges(gray, ksize=3, thresh=50):
    """Gradient magnitude theo ca x va y, threshold ra binary edge map."""
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    magnitude = np.uint8(255 * magnitude / (magnitude.max() + 1e-6))
    _, binary = cv2.threshold(magnitude, thresh, 255, cv2.THRESH_BINARY)
    return binary


def sobel_x_threshold(gray, ksize=3, thresh=(20, 100)):
    """Gradient theo rieng truc x (vach lane gan nhu doc -> gradient x manh hon gradient y)."""
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    abs_sobel = np.abs(sobel_x)
    scaled = np.uint8(255 * abs_sobel / (abs_sobel.max() + 1e-6))
    binary = np.zeros_like(scaled)
    binary[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 255
    return binary


def combine_color_gradient(img_bgr, gray, sobel_thresh=(20, 100), s_thresh=(120, 255)):
    """Hop nhat 2 nguon evidence bang OR: pixel duoc coi la lane neu thoa 1 trong 2 dieu kien.
    Sobel-x bat duoc truong hop mau nhat nhung tuong phan do sang ro (vach trang).
    S-channel bat duoc truong hop tuong phan do sang thap nhung khac mau ro (vach vang tren nen xam)."""
    sobel_bin = sobel_x_threshold(gray, thresh=sobel_thresh)
    s_bin = hls_s_channel_threshold(img_bgr, thresh=s_thresh)
    combined = np.zeros_like(sobel_bin)
    combined[(sobel_bin > 0) | (s_bin > 0)] = 255
    return combined


def lane_color_mask(img_bgr, gray, brightness_thresh=170, s_thresh=(120, 255)):
    """Mask 'co the la mau vach lane' (trang HOAC vang), dung de AND voi edge map truoc Hough
    (Pipeline A) nham loai cac cau truc hinh hoc giong lane nhung khong dung mau (vd rao chan
    kim loai mau xam): trang = sang (gray > brightness_thresh), vang = bao hoa mau cao (S-channel).
    Khac voi combine_color_gradient (OR voi sobel, dung lam evidence chinh cho Pipeline B), ham nay
    chi dung lam MASK LOC, khong tu sinh ra edge."""
    bright = np.zeros_like(gray)
    bright[gray > brightness_thresh] = 255
    yellow = hls_s_channel_threshold(img_bgr, thresh=s_thresh)
    mask = np.zeros_like(gray)
    mask[(bright > 0) | (yellow > 0)] = 255
    return mask


def hls_s_channel_threshold(img_bgr, thresh=(120, 255)):
    """Threshold tren S-channel (HLS) de bat vach trang/vang on dinh hon duoi bong/anh sang manh."""
    hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]
    binary = np.zeros_like(s_channel)
    binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 255
    return binary
