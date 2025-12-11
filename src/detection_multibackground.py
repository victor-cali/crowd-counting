# src/detection.py
import cv2
import numpy as np
from typing import List, Tuple, Optional, Union

def detect_people_via_subtraction(
    image_gray: np.ndarray,
    bg1_gray: np.ndarray, bg2_gray: np.ndarray, mean_bg: np.ndarray,     # ← two empty images + mean background
    region_mask: Optional[np.ndarray] = None,
    probability_map: Optional[np.ndarray] = None,
    use_clahe: bool = True,
    blur_ksize: int = 5, thresh_method: str = 'otsu',
    min_area: Union[int, float] = 20, min_area_map: Optional[Union[List[float], np.ndarray]] = None,  # y-dependent minimum area
    max_area: Union[int, float] = 40, max_area_map: Optional[Union[List[float], np.ndarray]] = None,  # y-dependent maximum area
    morph_kernel_size: int = 3
) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int, int, int]], np.ndarray]:
    """
    Detects objects (people) in a grayscale image using a multi-background subtraction technique.

    This pipeline applies CLAHE contrast enhancement, computes weighted differences against
    three background models, enhances edges with Laplacian, applies spatial probability maps,
    and filters resulting blobs based on size (potentially varying by Y-coordinate).

    Args:
        image_gray (np.ndarray): The current frame in grayscale (2D array, uint8).
        bg1_gray (np.ndarray): First static background reference (2D array, uint8).
        bg2_gray (np.ndarray): Second static background reference (2D array, uint8).
        mean_bg (np.ndarray): Mean/Rolling average background (2D array, usually uint8 or float32).
        region_mask (Optional[np.ndarray]): Binary mask where 0 indicates ignored regions (2D array, uint8).
        probability_map (Optional[np.ndarray]): Float mask (0.0-1.0) weighting detection probability per pixel.
        use_clahe (bool): Whether to apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
        blur_ksize (int): Kernel size for Gaussian Blur (must be odd).
        thresh_method (str): Thresholding method, either 'otsu' or fixed.
        min_area (Union[int, float]): Global minimum blob area in pixels.
        min_area_map (Optional[Union[List, np.ndarray]]): Array mapping Y-coordinate to specific min_area values.
        max_area (Union[int, float]): Global maximum blob area in pixels.
        max_area_map (Optional[Union[List, np.ndarray]]): Array mapping Y-coordinate to specific max_area values.
        morph_kernel_size (int): Size of the structuring element for morphological operations.

    Returns:
        Tuple containing:
            1. points (List[Tuple[float, float]]): Centroids (x, y) of detected blobs.
            2. boxes (List[Tuple[int, int, int, int]]): Bounding boxes (x, y, w, h).
            3. th (np.ndarray): The final binary threshold image used for detection (visualization/debug).
    """

    img = image_gray.copy()

    # CLAHE
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
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
    weight_diff3 = 2.0  # Changed to float for consistency

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

    points: List[Tuple[float, float]] = []
    boxes: List[Tuple[int, int, int, int]] = []

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
