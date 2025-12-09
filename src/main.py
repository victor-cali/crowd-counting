# src/main.py
import glob
import os
from pathlib import Path
import cv2
import numpy as np
from utils import list_images, load_image, find_annotation_csv, parse_makesense_csv, overlay_points
from preprocess import apply_clahe
from detection import detect_people_via_subtraction
from evaluation import compute_image_level_mse, person_level_metrics
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
IMG_GLOB = "imatges/*.jpg"
ANNOTATION_CSV = None
OUTPUT_DIR = "results"
ACCEPTANCE_RADIUS = 50
MIN_AREA = 100
MIN_AREA_FAR = MIN_AREA//4
MAX_AREA = 2500
MAX_AREA_FAR = MAX_AREA//4
# --------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

def list_images(img_glob="imatges/*.jpg"):
    return sorted(glob.glob(img_glob))

# -------------------------------
# Bounding box evaluation function
# -------------------------------
def evaluate_boxes(gt_points, pred_boxes):
    matched_gt = set()
    matched_boxes = set()

    for bi, (x, y, w, h) in enumerate(pred_boxes):
        for gi, (gx, gy) in enumerate(gt_points):
            if gi in matched_gt:
                continue
            if (gx >= x - ACCEPTANCE_RADIUS and gx <= x + w + ACCEPTANCE_RADIUS and
                gy >= y - ACCEPTANCE_RADIUS and gy <= y + h + ACCEPTANCE_RADIUS):
                matched_gt.add(gi)
                matched_boxes.add(bi)
                break

    tp = len(matched_gt)
    fp = len(pred_boxes) - tp
    fn = len(gt_points) - tp

    return tp, fp, fn



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

    # ------------------------------------------------------
    # Firstly, we select the first two images of an empty beach to subtract them later
    # We will also generate a mask using the mean between all the images
    # ------------------------------------------------------
    print("Loading first two images as background reference...")
    empty_paths = img_paths[:2]
    empty_imgs = [load_image(p, gray=True) for p in empty_paths]

    mean_imgs = []
    for p in img_paths:
        img = load_image(p, gray=True).astype(np.float32)
        mean_imgs.append(img)

    mean_bg = np.mean(mean_imgs, axis=0).astype(np.uint8)

    # ------------------------------------------------------
    # Region mask + probability map
    # ------------------------------------------------------
    h, w = empty_imgs[0].shape
    region_mask = np.ones((h, w), dtype=np.uint8)
    probability_map = np.ones((h, w), dtype=np.float32)

    # Ignore the top 35% of the image (mountain)
    top_h = int(h * 0.35)
    region_mask[:top_h, :] = 0

    # Reduce probability in bottom part
    bottom_h = int(h * 0.65)
    probability_map[bottom_h:, :] = 0.5

    # evaluation accumulators
    per_image_gt_counts = []
    per_image_pred_counts = []
    total_tp = total_fp = total_fn = 0

    for p in img_paths:
        fname = Path(p).name
        gray = load_image(p, gray=True)

        # detect
        pts, boxes, mask = detect_people_via_subtraction(
            gray,
            empty_imgs[0],  # bg1
            empty_imgs[1],  # bg2
            mean_bg,
            region_mask=region_mask,
            probability_map=probability_map,
            use_clahe=True,
            blur_ksize=5,
            thresh_method='otsu',
            min_area=MIN_AREA,
            min_area_map=np.linspace(MIN_AREA_FAR, MIN_AREA, h).astype(np.int32),
            max_area=MAX_AREA,
            max_area_map=np.linspace(MAX_AREA_FAR, MAX_AREA, h).astype(np.int32),
            morph_kernel_size=3
        )

        pred_count = len(boxes)
        gt_pts = annotations.get(fname, [])
        gt_count = len(gt_pts)

        per_image_gt_counts.append(gt_count)
        per_image_pred_counts.append(pred_count)

        # ---------------------------
        # NEW: bounding box based evaluation
        # ---------------------------
        if gt_pts:
            tp, fp, fn = evaluate_boxes(gt_pts, boxes)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            print(f"{fname}: GT={gt_count}, Pred={pred_count}, TP={tp}, FP={fp}, FN={fn}")
        else:
            print(f"{fname}: GT=NA, Pred={pred_count}")

        # visualization
        color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # draw ground truth points
        if gt_pts:
            color = overlay_points(color, gt_pts, color=(0,255,0), radius=4)

        # NEW: draw predicted bounding boxes
        for (x, y, w, h) in boxes:
            cv2.rectangle(color, (x, y), (x+w, y+h), (0,0,255), 2)

        # save
        overlay_path = os.path.join(OUTPUT_DIR, f"vis_{fname}")
        cv2.imwrite(overlay_path, color)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"mask_{fname}"), mask)

    # evaluation
    if per_image_gt_counts:
        mse = compute_image_level_mse(per_image_gt_counts, per_image_pred_counts)
        print(f"\nImage-level MSE (counts): {mse:.3f}")
    else:
        mse = None

    if (total_tp + total_fp + total_fn) > 0:
        metrics = person_level_metrics(total_tp, total_fp, total_fn)
        print(f"Person-level precision: {metrics['precision']:.3f}, recall: {metrics['recall']:.3f}, f1: {metrics['f1']:.3f}")
    else:
        metrics = None

    # Save report
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
