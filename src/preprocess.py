# src/preprocess.py
import cv2
import numpy as np

def apply_clahe(gray, clipLimit=2.0, tileGridSize=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(gray)

def compute_background_median(image_paths, sample_limit=None):
    """
    Compute median background from list of grayscale image paths.
    If sample_limit is set, uses at most that many images (randomly sampled).
    """
    import random
    if sample_limit and len(image_paths) > sample_limit:
        image_paths = random.sample(image_paths, sample_limit)
    stacks = []
    for p in image_paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        stacks.append(img.astype(np.uint8))
    if not stacks:
        raise ValueError("No images available to compute background")
    arr = np.stack(stacks, axis=0)
    med = np.median(arr, axis=0).astype(np.uint8)
    return med

def background_subtraction(image_gray, background_gray):
    # both uint8
    diff = cv2.absdiff(image_gray, background_gray)
    return diff
