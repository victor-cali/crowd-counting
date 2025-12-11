from typing import List, Tuple, Optional
import cv2
import numpy as np

# -------------------------
# Helper Utilities
# -------------------------

def ensure_gray(img: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to grayscale if needed.

    Input:
      - img: np.ndarray with shape (H, W, 3) or (H, W)
    Output:
      - gray: np.ndarray shape (H, W), dtype=np.uint8 with range 0..255
    """
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.ndim == 2:
        return img
    raise ValueError("Unsupported image shape: %s" % (img.shape,))


def build_background_from_empty_images(empty_images: List[np.ndarray]) -> np.ndarray:
    """
    Build a background model (grayscale float32 image) from empty-beach images.

    Input:
      - empty_images: list of images; each image is BGR or grayscale numpy array.
                      All images must have the same shape (H, W) or (H, W, 3).
    Output:
      - bg: np.ndarray shape (H, W), dtype=np.float32, values roughly 0..255
            This is the pixel-wise average of the provided empty images.
    """
    if len(empty_images) == 0:
        raise ValueError("Need at least one empty image to build background")

    # Convert to grayscale float32
    gray_imgs = [ensure_gray(im).astype(np.float32) for im in empty_images]
    # Stack and compute mean across axis=0 -> shape (H, W)
    stacked = np.stack(gray_imgs, axis=0)  # shape (N, H, W)
    bg = np.mean(stacked, axis=0).astype(np.float32)
    return bg


def nms_local_max(response: np.ndarray, radius: int = 3, threshold: float = 0.0) -> List[Tuple[int, int, float]]:
    """
    Simple Non-Maximum Suppression via dilation to find local maxima in a 2D response map.

    Input:
      - response: 2D float array shape (H, W) — higher = stronger response (can be negative)
      - radius: int, radius of the local neighborhood (in pixels). window size = 2*radius+1
      - threshold: float, minimum response value to be considered

    Output:
      - points: list of (x, y, score) for every local maximum where score > threshold
        coordinates are integer pixel indices (x = col, y = row).
    """
    # Convert to float32
    resp = response.astype(np.float32)
    H, W = resp.shape

    # Dilate to get local max in neighborhood
    ksize = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    dilated = cv2.dilate(resp, kernel)

    # Keep only those equal to dilated and above threshold
    is_local_max = (resp == dilated) & (resp > threshold)

    ys, xs = np.where(is_local_max)
    points = [(int(x), int(y), float(resp[y, x])) for (y, x) in zip(ys, xs)]
    # Sort by score descending (optional)
    points.sort(key=lambda t: t[2], reverse=True)
    return points


def centroids_from_binary_mask(mask: np.ndarray, min_area: int = 5) -> List[Tuple[int, int, int]]:
    """
    Given a binary mask (0 or 255 / 0 or 1), find connected components and return centroids.

    Input:
      - mask: np.ndarray shape (H, W), dtype uint8 or bool. Non-zero pixels are foreground.
      - min_area: minimum pixel area for a connected component to be considered valid

    Output:
      - centroids: list of (x, y, area) where (x,y) is centroid integer coordinates (col, row)
    """
    if mask.dtype != np.uint8:
        mask_u8 = (mask > 0).astype(np.uint8) * 255
    else:
        mask_u8 = (mask > 0).astype(np.uint8) * 255

    # connectedComponentsWithStats returns:
    # num_labels, labels, stats, centroids
    num_labels, labels, stats, cents = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    result: List[Tuple[int, int, int]] = []
    for label in range(1, num_labels):  # skip background label 0
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        cx, cy = cents[label]  # floats (x, y)
        result.append((int(round(cx)), int(round(cy)), area))
    return result


# -------------------------
# Variant B: LoG High-Frequency Head Detection + Peak Detection
# -------------------------

def detect_heads_log(
    image: np.ndarray,
    bg_log_mask: Optional[np.ndarray] = None,
    clahe_clip: float = 2.0,
    clahe_tile: int = 8,
    gaussian_sigma: float = 1.0,
    laplacian_ksize: int = 3,
    nms_radius: int = 6,
    response_threshold: float = 10.0,
    normalize: bool = True,
) -> List[Tuple[int, int, float]]:
    """
    Variant B: Detect head-like blobs using LoG (approximated by Gaussian blur + Laplacian)
    followed by local-peak detection (NMS).

    Inputs:
      - image: np.ndarray, input image in BGR or grayscale. Shape (H, W, 3) or (H, W), dtype=np.uint8.
      - bg_log_mask: Optional[np.ndarray], background LoG response to subtract (shape (H, W), float32).
                     If provided, it will be subtracted from the LoG response to suppress background patterns.
      - clahe_clip, clahe_tile: CLAHE parameters (clipLimit & tileGridSize)
      - gaussian_sigma: sigma for Gaussian blur before Laplacian. Controls scale of detection (approx head size).
      - laplacian_ksize: aperture size for Laplacian operator (1,3,5). Usually 3 is good.
      - nms_radius: radius for local maxima NMS (in pixels).
      - response_threshold: minimum LoG response (after optional subtraction and normalization) to accept a peak.
      - normalize: whether to normalize LoG response before thresholding (recommended True)

    Output:
      - detections: list of (x, y, score) coordinates of detected head peaks (x = col, y = row).
    """

    # --- Step 0: Grayscale conversion
    # Input: image (H, W, 3) or (H, W)
    # Output: gray (H, W), dtype uint8
    gray = ensure_gray(image)  # shape: (H, W), dtype=uint8

    H, W = gray.shape

    # --- Step 1: CLAHE (contrast limited adaptive histogram equalization)
    # Input: gray (H, W)
    # Output: clahe_img (H, W), dtype uint8
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
    clahe_img = clahe.apply(gray)

    # --- Step 2: Gaussian blur (reduce noise, implement LoG)
    # Input: clahe_img (H, W)
    # Output: blurred float32 (H, W)
    ksize = int(round(gaussian_sigma * 6)) | 1  # make odd
    blurred = cv2.GaussianBlur(clahe_img, (ksize, ksize), gaussian_sigma).astype(np.float32)

    # --- Step 3: Laplacian (approx LoG)
    # Input: blurred (H, W) float32
    # Output: lap (H, W) float32 (can be negative & positive)
    lap = cv2.Laplacian(blurred, ddepth=cv2.CV_32F, ksize=laplacian_ksize)

    # LoG typically gives negative valley at blobs (depending on sign), so we often use absolute response.
    log_resp = -lap  # invert sign if necessary so blob centers are positive peaks

    # --- Step 4: Subtract background LoG mask (optional)
    # Input: log_resp (H, W), bg_log_mask (H, W)
    # Output: log_clean (H, W)
    if bg_log_mask is not None:
        if bg_log_mask.shape != log_resp.shape:
            raise ValueError("bg_log_mask must match image shape")
        log_resp = log_resp - bg_log_mask.astype(np.float32)

    # --- Step 5: Normalize (optional)
    # Input: log_resp
    # Output: log_norm
    if normalize:
        # Remove mean and scale to 0..255 roughly for stable thresholding
        lo_min, lo_max = float(np.min(log_resp)), float(np.max(log_resp))
        if lo_max - lo_min > 1e-6:
            log_norm = (log_resp - lo_min) / (lo_max - lo_min) * 255.0
        else:
            log_norm = np.zeros_like(log_resp)
    else:
        log_norm = log_resp

    # --- Step 6: Local maxima via NMS
    # Input: log_norm (H, W) float32
    # Output: candidate peak list [(x,y,score), ...]
    candidates = nms_local_max(log_norm, radius=nms_radius, threshold=response_threshold)

    # candidates is list of (x, y, score)
    return candidates


# -------------------------
# Variant C: Hybrid (Background Mask ∧ Edges)
# -------------------------

def detect_heads_hybrid(
    image: np.ndarray,
    background: np.ndarray,
    diff_threshold: float = 30.0,
    clahe_clip: float = 2.0,
    clahe_tile: int = 8,
    morph_kernel_size: int = 5,
    min_blob_area: int = 15,
    edge_low_thresh: int = 50,
    edge_high_thresh: int = 150,
    use_head_kernel: bool = True,
    head_kernel_radius: int = 4,
) -> List[Tuple[int, int, int]]:
    """
    Variant C: Hybrid pipeline combining background subtraction (foreground mask)
    with edge detection. Output is centroids of candidate head blobs.

    Inputs:
      - image: np.ndarray, input image in BGR or grayscale (H, W, 3) or (H, W), dtype=uint8
      - background: np.ndarray, grayscale float32 background model shape (H, W) (see build_background_from_empty_images)
      - diff_threshold: float, pixel-intensity absolute difference threshold to create foreground mask
      - clahe_clip, clahe_tile: CLAHE parameters applied before background subtraction
      - morph_kernel_size: int, size of morphological kernel used to clean up foreground (odd)
      - min_blob_area: int, minimum connected-component area (in px) to be considered a person
      - edge_low_thresh, edge_high_thresh: Canny thresholds
      - use_head_kernel: bool, whether to perform a circular template match on blobs to prefer circular heads
      - head_kernel_radius: int, radius in px of the circular head kernel used for matching (if enabled)

    Output:
      - centroids: list of (x, y, area) where x,y are ints (col,row) and area is component area in px
    """

    # --- Step 0: Input checks and conversion
    # Inputs:
    #  - image: (H, W, 3) or (H, W) uint8
    #  - background: (H, W) float32 (average of empty images)
    img_gray = ensure_gray(image).astype(np.float32)  # shape (H, W)
    bg = background.astype(np.float32)
    if img_gray.shape != bg.shape:
        raise ValueError("image and background must have same (H, W)")

    H, W = img_gray.shape

    # --- Step 1: CLAHE (optional, helps contrast)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
    img_clahe = clahe.apply(img_gray.astype(np.uint8)).astype(np.float32)  # (H, W), float32

    # --- Step 2: Background subtraction -> foreground mask
    # Input: img_clahe (H, W), bg (H, W)
    # Output: diff (H, W) float32, mask_binary (H, W) uint8 {0,255}
    diff = cv2.absdiff(img_clahe.astype(np.uint8), np.clip(bg, 0, 255).astype(np.uint8)).astype(np.uint8)
    _, mask = cv2.threshold(diff, int(diff_threshold), 255, cv2.THRESH_BINARY)
    # mask: uint8 (H, W) with values {0,255}

    # --- Step 3: Morphological operations to clean mask
    # Input: mask (H, W)
    # Output: mask_clean (H, W)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)  # remove small noise
    mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, k)  # close small holes
    mask_clean = mask_close  # (H, W) uint8

    # --- Step 4: Edge detection (Canny) on CLAHE image
    # Input: img_clahe (H, W) float32
    # Output: edges (H, W) uint8 {0,255}
    edges = cv2.Canny(img_clahe.astype(np.uint8), threshold1=edge_low_thresh, threshold2=edge_high_thresh)

    # --- Step 5: Combine mask and edges: keep edges only where foreground mask exists
    # Input: mask_clean (H, W), edges (H, W)
    # Output: edges_fg (H, W)
    edges_fg = cv2.bitwise_and(edges, mask_clean)  # keeps only edges in foreground regions

    # --- Step 6: Morphological refinement on edges_fg (thicken and close)
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_fg_close = cv2.morphologyEx(edges_fg, cv2.MORPH_CLOSE, k2)
    edges_fg_fill = cv2.dilate(edges_fg_close, k2, iterations=1)
    # Convert edges to a filled mask via flood fill on contours: find contours and fill them
    contours, _ = cv2.findContours(edges_fg_fill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask_clean)
    cv2.drawContours(filled, contours, -1, color=255, thickness=cv2.FILLED)  # (H, W) uint8

    # --- Step 7: Combine filled regions with original foreground mask to get final candidate blobs
    candidates_mask = cv2.bitwise_and(mask_clean, filled)

    # --- Step 8 (optional): Filter connected components and compute centroids
    centroids = centroids_from_binary_mask(candidates_mask, min_area=min_blob_area)
    # centroids: list of (x, y, area)

    # --- Step 9 (optional): Head kernel matching (prefer circular shapes)
    if use_head_kernel and len(centroids) > 0:
        # Build circular kernel normalized
        r = head_kernel_radius
        kr = 2 * r + 1
        y, x = np.ogrid[-r:r + 1, -r:r + 1]
        circle = (x * x + y * y) <= (r * r)
        circle_kernel = circle.astype(np.uint8)

        # For each centroid, compute overlap between kernel placed at centroid and candidates_mask
        filtered = []
        for (cx, cy, area) in centroids:
            x0 = cx - r
            y0 = cy - r
            x1 = cx + r
            y1 = cy + r
            # skip if kernel goes out of bounds
            if x0 < 0 or y0 < 0 or x1 >= W or y1 >= H:
                continue
            patch = candidates_mask[y0:y1 + 1, x0:x1 + 1]
            if patch.shape != circle_kernel.shape:
                continue
            # compute how many kernel pixels overlap with mask
            overlap = int(np.sum((patch > 0).astype(np.uint8) * circle_kernel))
            # require at least 30% of circle kernel overlapped, or area-based threshold
            if overlap >= 0.3 * np.sum(circle_kernel):
                filtered.append((cx, cy, area))
        centroids = filtered

    return centroids



