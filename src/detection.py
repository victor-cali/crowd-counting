# src/detection.py
import cv2
import numpy as np
from scipy import ndimage

def detect_people_via_subtraction(
    image_gray,
    bg1_gray, bg2_gray, mean_bg,     # ← two empty images + mean background
    region_mask=None,
    probability_map=None,
    use_clahe=True,
    blur_ksize=5, thresh_method='otsu',
    min_area=20, min_area_map=None,  # y-dependent minimum area
    max_area=40, max_area_map=None,  # y-dependent maximum area
    morph_kernel_size=3):

    img = image_gray.copy()

    # CLAHE
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)

    # ------------------------------------------------------
    # Background subtraction with three backgrounds
    # ------------------------------------------------------
    # Compute differences
    diff1 = cv2.absdiff(img, bg1_gray)
    diff2 = cv2.absdiff(img, bg2_gray)
    diff3 = cv2.absdiff(img, mean_bg)

    # Apply higher weight to diff3
    # For example: weight_diff3 = 1.5, weight_diff1/2 = 1.0
    weight_diff1 = 1.0
    weight_diff2 = 1.0
    weight_diff3 = 2

    # Weighted combination
    diff_combined = np.minimum(
        (diff1 * weight_diff1).astype(np.float32),
        (diff2 * weight_diff2).astype(np.float32)
    )
    diff = np.minimum(diff_combined, (diff3 * weight_diff3).astype(np.float32))

    # Clip to 0-255 and convert to uint8
    diff = np.clip(diff, 0, 255).astype(np.uint8)

    # Laplacian sharpening
    lap = cv2.Laplacian(diff, cv2.CV_8U, ksize=3)
    fused = cv2.addWeighted(diff, 0.8, lap, 0.2, 0)

    # Apply probability map
    if probability_map is not None:
        fused = (fused.astype(np.float32) * probability_map).astype(np.uint8)

    # Blur
    fused = cv2.GaussianBlur(fused, (blur_ksize, blur_ksize), 0)

    # Threshold
    if thresh_method == 'otsu':
        _, th = cv2.threshold(fused, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, th = cv2.threshold(fused, 30, 255, cv2.THRESH_BINARY)

    # Apply region mask
    if region_mask is not None:
        th = cv2.bitwise_and(th, th, mask=region_mask)

    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(th, connectivity=8)

    points = []
    boxes = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        # Use progressive min area if map is provided
        effective_min_area = min_area
        if min_area_map is not None and 0 <= y < len(min_area_map):
            effective_min_area = min_area_map[y]

        # Use progressive max area if map is provided
        effective_max_area = max_area
        if max_area_map is not None and 0 <= y < len(max_area_map):
            effective_max_area = max_area_map[y]

        # Filter by area
        if effective_min_area <= area <= effective_max_area:
            cx, cy = centroids[i]
            points.append((float(cx), float(cy)))
            boxes.append((x, y, w, h))

    return points, boxes, th
