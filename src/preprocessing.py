"""Buoc tien xu ly anh: grayscale + lam mo de giam nhieu truoc khi tim edge."""
import cv2


def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def gaussian_blur(gray, ksize=5):
    return cv2.GaussianBlur(gray, (ksize, ksize), 0)
