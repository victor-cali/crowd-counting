from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np

# -------------------------
# LoG High-Frequency Head Detection + Peak Detection
# -------------------------


def detect_heads_log2(
    image_gray: np.ndarray,
    valid_mask:  Optional[np.ndarray] = None,
    bg_log:  Optional[np.ndarray] = None,
    sigma: float = 2.0,
    nms_radius: int = 6,
    response_threshold: float = 20.0,
    min_area: int = 3,
    max_area: int = 40,
    clahe_clip: float = 2.0,
    clahe_grid: Tuple[int, int] = (8, 8),
    border_width: int = 15,  # NEW: border exclusion width
) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int, int, int]], np.ndarray]:
    """
    LoG-inspired high-frequency peak detection with area-based filtering.
    
    This version includes:
        - CLAHE
        - Gaussian smoothing (scale selection)
        - Laplacian (LoG approximation)
        - Response normalization
        - Local Maxima Detection (NMS)
        - Area estimation around each peak
        - Area thresholding
        - Border exclusion to prevent edge artifacts
        - Output format: same as Variant D → (x, y, area)

    Parameters
    ----------
    image_gray : np.ndarray
        The current frame in grayscale (2D array, uint8).
    bg_log : Optional[np.ndarray]
        Background LoG response to subtract (shape (H, W), float32).
        If provided, it will be subtracted from the LoG response to suppress background patterns.
    valid_mask : np.ndarray
        Binary mask (H, W), where 1 = valid region, 0 = invalid.
        If None, entire image_gray is considered valid (except borders).
    sigma : float
        Standard deviation for Gaussian smoothing.
    nms_radius : int
        Radius for Non-Maximum Suppression.
    response_threshold : float
        Minimum LoG response (normalized) to consider a peak.
    min_area : int
        Minimum estimated peak-support area to keep.
    max_area : int
        Maximum estimated peak-support area to keep.
    clahe_clip : float
        CLAHE clip limit.
    clahe_grid : tuple
        CLAHE tile grid size.
    border_width : int
        Width of border region to exclude from detection (pixels).

    Returns
    -------
        Tuple containing:
            1. points (List[Tuple[float, float]]): List of detected head centers (x, y)
            2. boxes (List[Tuple[int, int, int, int]]): Bounding boxes (x, y, w, h)
            3. resp_norm_masked (np.ndarray): Final LoG response map (H, W), float32
    """

    H, W = image_gray.shape
    
    # ----------------------------------------------------------------------
    # 1) Create border mask to exclude edge regions
    # ----------------------------------------------------------------------
    border_mask = np.ones((H, W), dtype=np.uint8)
    border_mask[:border_width, :] = 0  # top
    border_mask[-border_width:, :] = 0  # bottom
    border_mask[:, :border_width] = 0  # left
    border_mask[:, -border_width:] = 0  # right
    
    # Combine with user-provided valid_mask
    if valid_mask is not None:
        combined_mask = valid_mask & border_mask
    else:
        combined_mask = border_mask

    # ----------------------------------------------------------------------
    # 2) CLAHE with border padding to reduce edge artifacts
    # ----------------------------------------------------------------------
    # Pad image before CLAHE to reduce edge effects
    pad = max(10, int(sigma * 3))
    img_padded = cv2.copyMakeBorder(
        image_gray, pad, pad, pad, pad, 
        cv2.BORDER_REFLECT_101
    )
    
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    img_clahe_padded = clahe.apply(img_padded)
    
    # Remove padding
    img_clahe = img_clahe_padded[pad:-pad, pad:-pad]

    # ----------------------------------------------------------------------
    # 3) Gaussian smoothing (LoG scale selection)
    # ----------------------------------------------------------------------
    ksize = int(6 * sigma) | 1
    blurred = cv2.GaussianBlur(img_clahe, (ksize, ksize), sigmaX=sigma)

    # ----------------------------------------------------------------------
    # 4) Laplacian → approximate LoG
    # ----------------------------------------------------------------------
    lap = cv2.Laplacian(blurred.astype(np.float32), cv2.CV_32F, ksize=3)
    log_resp = -lap  # make blob centers positive
    
    # Optional background LoG subtraction
    if bg_log is not None:
        if bg_log.shape == log_resp.shape:
            log_resp = log_resp - bg_log
        else:
            raise ValueError("bg_log must match log_resp dimensions")

    # ----------------------------------------------------------------------
    # 5) Normalize LoG to [0,255]
    # ----------------------------------------------------------------------
    lo_min, lo_max = log_resp.min(), log_resp.max()
    if lo_max - lo_min < 1e-6:
        resp_norm = np.zeros_like(log_resp, dtype=np.float32)
    else:
        resp_norm = (log_resp - lo_min) / (lo_max - lo_min) * 255.0
    
    # ----------------------------------------------------------------------
    # 6) Apply combined valid-region + border mask
    # ----------------------------------------------------------------------
    resp_norm_masked = resp_norm * combined_mask

    # ----------------------------------------------------------------------
    # 7) NMS (non-maximum suppression)
    # ----------------------------------------------------------------------
    radius = nms_radius
    nms_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    dilated = cv2.dilate(resp_norm_masked, nms_kernel)

    is_peak = (resp_norm_masked == dilated) & (resp_norm_masked > response_threshold)

    ys, xs = np.where(is_peak)

    # ----------------------------------------------------------------------
    # 8) Area estimation and filtering
    # ----------------------------------------------------------------------
    points: List[Tuple[float, float]] = []
    boxes: List[Tuple[int, int, int, int]] = []

    window = 4  # 9×9 patch for area estimation

    # Final binary detection mask
    th = np.zeros((H, W), dtype=np.uint8)

    for (x, y) in zip(xs, ys):
        # Double-check: skip if in border region (redundant but safe)
        if (x < border_width or x >= W - border_width or 
            y < border_width or y >= H - border_width):
            continue

        # Extract local patch for area estimation
        x1 = max(0, x - window)
        x2 = min(W, x + window + 1)
        y1 = max(0, y - window)
        y2 = min(H, y + window + 1)

        patch = resp_norm[y1:y2, x1:x2]
        peak_val = resp_norm[y, x]

        # Area = number of pixels above 40% of peak
        area = int(np.sum(patch > peak_val * 0.4))

        # Area filtering
        if area < min_area or area > max_area:
            continue

        # Store point
        points.append((float(x), float(y)))

        # Store bounding box (centered window)
        w = x2 - x1
        h = y2 - y1
        boxes.append((x1, y1, w, h))

        # Mark detection in binary mask
        th[y, x] = 255

    return points, boxes, resp_norm_masked


def detect_heads_log2_improved(
    image_gray: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    sigma_list: List[float] = [1.5, 2.5, 3.5],   # multi-scale sigmas
    response_threshold: float = 25.0,
    border_width: int = 15,
    contour_min_area: int = 5,
    contour_max_area: int = 200,
) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int, int, int]], np.ndarray]:

    H, W = image_gray.shape

    # ----------------------------------------------------------------------
    # 1) Build combined mask (border + valid region)
    # ----------------------------------------------------------------------
    border_mask = np.ones((H, W), dtype=np.uint8)
    border_mask[:border_width, :] = 0
    border_mask[-border_width:, :] = 0
    border_mask[:, :border_width] = 0
    border_mask[:, -border_width:] = 0

    if valid_mask is None:
        combined_mask = border_mask
    else:
        combined_mask = (valid_mask.astype(np.uint8) & border_mask)

    # ----------------------------------------------------------------------
    # 2) Multi-scale DoG (Difference-of-Gaussians)
    # ----------------------------------------------------------------------
    responses = []

    for sigma in sigma_list:
        k1 = int(6 * sigma) | 1
        k2 = int(6 * (sigma * 1.6)) | 1  # classic DoG ratio ~1.6

        g1 = cv2.GaussianBlur(image_gray, (k1, k1), sigmaX=sigma)
        g2 = cv2.GaussianBlur(image_gray, (k2, k2), sigmaX=sigma * 1.6)

        dog = (g1.astype(np.float32) - g2.astype(np.float32))

        responses.append(dog)

    # stack and take max response across scales
    resp_all = np.stack(responses, axis=-1)         # H × W × S
    resp_max = np.max(resp_all, axis=-1)            # H × W

    # mask invalid regions
    resp_max *= combined_mask

    # normalize to 0–255
    rmin, rmax = resp_max.min(), resp_max.max()
    resp_norm = (
        (resp_max - rmin) / (rmax - rmin + 1e-6) * 255
    ).astype(np.float32)

    # ----------------------------------------------------------------------
    # 3) Smoothed response before NMS
    # ----------------------------------------------------------------------
    resp_smooth = cv2.GaussianBlur(resp_norm, (5, 5), sigmaX=1.0)

    # ----------------------------------------------------------------------
    # 4) Plateau-safe NMS using connected component maxima
    # ----------------------------------------------------------------------
    # Threshold first
    binary = (resp_smooth > response_threshold).astype(np.uint8)

    # Connected components
    cc = cv2.connectedComponentsWithStats(binary, connectivity=8)
    num_labels, _, stats, centroids = cc

    peaks = []
    for i in range(1, num_labels):  # skip background
        x, y = centroids[i]
        x = int(x)
        y = int(y)

        # find true max inside this component
        y1, y2 = stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]
        x1, x2 = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]

        patch = resp_smooth[y1:y2, x1:x2]
        (py, px) = np.unravel_index(np.argmax(patch), patch.shape)
        px += x1
        py += y1

        peaks.append((px, py))

    # ----------------------------------------------------------------------
    # 5) Area estimation using Canny → closed contour
    # ----------------------------------------------------------------------
    points = []
    boxes = []

    for (x, y) in peaks:

        # validate border
        if (x < border_width or x >= W - border_width or
            y < border_width or y >= H - border_width):
            continue

        ROI_SIZE = 25
        x1 = max(0, x - ROI_SIZE)
        x2 = min(W, x + ROI_SIZE + 1)
        y1 = max(0, y - ROI_SIZE)
        y2 = min(H, y + ROI_SIZE + 1)

        roi = image_gray[y1:y2, x1:x2]

        # ---- Edge-based closed contour detection ----
        edges = cv2.Canny(roi, 60, 120)

        # Close gaps
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3)))

        # find contours
        contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue

        # pick the largest closed contour
        cont = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cont)

        # area filtering
        if area < contour_min_area or area > contour_max_area:
            continue

        # ensure the contour is approximately closed & compact
        perimeter = cv2.arcLength(cont, True)
        if perimeter == 0:
            continue

        compactness = 4 * np.pi * (area / (perimeter * perimeter))
        if compactness < 0.25:  # elongated → reject
            continue

        # store head detection
        points.append((float(x), float(y)))
        boxes.append((x1, y1, x2 - x1, y2 - y1))

    return points, boxes, resp_norm
