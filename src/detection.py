# src/detection.py
import cv2
import numpy as np
from scipy import ndimage

def detect_people_via_subtraction(image_gray, background_gray, use_clahe=True,
                                  blur_ksize=5, thresh_method='otsu',
                                  min_area=30, morph_kernel_size=3):
    """
    Returns a list of (x,y) centroids of connected components detected as people.
    """
    img = image_gray.copy()
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)
    diff = cv2.absdiff(img, background_gray)
    # optional Laplacian enhancement
    lap = cv2.Laplacian(diff, cv2.CV_8U, ksize=3)
    fused = cv2.addWeighted(diff, 0.8, lap, 0.2, 0)
    # blur
    fused = cv2.GaussianBlur(fused, (blur_ksize, blur_ksize), 0)
    # threshold
    if thresh_method == 'otsu':
        _, th = cv2.threshold(fused, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # fixed threshold
        _, th = cv2.threshold(fused, 30, 255, cv2.THRESH_BINARY)
    # morphological clean
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    # label
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(th, connectivity=8)
    points = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cx, cy = centroids[i]
            points.append((float(cx), float(cy)))
    return points, th

def detect_people_via_blobs(image_gray, params=None):
    # alternative using SimpleBlobDetector
    if params is None:
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 20
        params.maxArea = 5000
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(image_gray)
    pts = [(kp.pt[0], kp.pt[1]) for kp in keypoints]
    return pts
