# src/main.py
import glob
import os
from pathlib import Path
import cv2
import numpy as np
from utils import list_images, load_image, find_annotation_csv, parse_makesense_csv, overlay_points
from preprocess import compute_background_median, apply_clahe, background_subtraction
from detection import detect_people_via_subtraction
from evaluation import match_points, compute_image_level_mse, person_level_metrics
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
IMG_GLOB = "imatges/*.jpg"
ANNOTATION_CSV = None  # if None, try to auto-find in project root
OUTPUT_DIR = "results"
BACKGROUND_SAMPLE_LIMIT = 40  # number of images used to compute median background
MATCH_RADIUS = 18.0  # pixels for matching predicted point to GT point
MIN_AREA = 20
# --------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

def list_images(img_glob="imatges/*.jpg"):
    return sorted(glob.glob(img_glob))

def main():
    print("Listing images...")
    img_paths = list_images(IMG_GLOB)
    if not img_paths:
        raise RuntimeError(f"No images found with glob {IMG_GLOB}")
    print(f"Found {len(img_paths)} images.")

    # annotations
    csv_path = ANNOTATION_CSV or find_annotation_csv(".")
    if csv_path:
        print(f"Using annotation CSV: {csv_path}")
        annotations = parse_makesense_csv(csv_path)
    else:
        print("No annotation CSV found. Proceeding without ground truth.")
        annotations = {}

    # compute background (median)
    print("Computing median background from images...")
    background = compute_background_median(img_paths, sample_limit=BACKGROUND_SAMPLE_LIMIT)

    per_image_gt_counts = []
    per_image_pred_counts = []
    total_tp = total_fp = total_fn = 0

    for p in img_paths:
        fname = Path(p).name
        gray = load_image(p, gray=True)
        # detect
        pts, mask = detect_people_via_subtraction(gray, background,
                                                 use_clahe=True,
                                                 blur_ksize=5,
                                                 thresh_method='otsu',
                                                 min_area=MIN_AREA,
                                                 morph_kernel_size=3)
        pred_count = len(pts)
        gt_pts = annotations.get(fname, [])
        gt_count = len(gt_pts)
        per_image_gt_counts.append(gt_count)
        per_image_pred_counts.append(pred_count)

        # match
        if gt_pts:
            res = match_points(gt_pts, pts, radius=MATCH_RADIUS)
            total_tp += res["tp"]; total_fp += res["fp"]; total_fn += res["fn"]
            print(f"{fname}: GT={gt_count}, Pred={pred_count}, TP={res['tp']}, FP={res['fp']}, FN={res['fn']}")
        else:
            print(f"{fname}: GT=NA, Pred={pred_count}")

        # visualization save
        color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if gt_pts:
            color = overlay_points(color, gt_pts, color=(0,255,0), radius=4)  # green = GT
        color = overlay_points(color, pts, color=(0,0,255), radius=3)  # red = pred
        overlay_path = os.path.join(OUTPUT_DIR, f"vis_{fname}")
        cv2.imwrite(overlay_path, color)
        # also save mask
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"mask_{fname}"), mask)

    # evaluation
    if per_image_gt_counts:
        mse = compute_image_level_mse(per_image_gt_counts, per_image_pred_counts)
        print(f"\nImage-level MSE (counts): {mse:.3f}")
    else:
        mse = None
        print("\nNo ground-truth counts to compute image-level MSE.")

    if (total_tp+total_fp+total_fn) > 0 or (len(per_image_gt_counts)>0):
        metrics = person_level_metrics(total_tp, total_fp, total_fn)
        print(f"Person-level precision: {metrics['precision']:.3f}, recall: {metrics['recall']:.3f}, f1: {metrics['f1']:.3f}")
    else:
        metrics = None
        print("No person-level matches computed (no annotations).")

    # Save a small report
    rep = {
        "num_images": len(img_paths),
        "mse": mse,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "precision": metrics['precision'] if metrics else None,
        "recall": metrics['recall'] if metrics else None,
        "f1": metrics['f1'] if metrics else None,
    }
    import json
    with open(os.path.join(OUTPUT_DIR, "report.json"), "w") as f:
        json.dump(rep, f, indent=2)

    print("\nSaved visualizations and report to 'results/'.")

if __name__ == "__main__":
    main()
