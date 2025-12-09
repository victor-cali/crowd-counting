# src/evaluation.py
import numpy as np
from math import hypot

def match_points(gt_points, pred_points, radius=15.0):
    """
    Greedy matching: for each predicted point, find nearest unmatched gt within radius.
    Returns counts: tp, fp, fn and lists of matched pairs.
    """
    gt = [tuple(p) for p in list(gt_points)]
    pr = [tuple(p) for p in list(pred_points)]
    matched_gt = set()
    matched_pairs = []
    tp = 0
    for i, p in enumerate(pr):
        best_j, best_d = None, None
        for j, g in enumerate(gt):
            if j in matched_gt:
                continue
            d = hypot(p[0]-g[0], p[1]-g[1])
            if d <= radius and (best_d is None or d < best_d):
                best_d = d
                best_j = j
        if best_j is not None:
            tp += 1
            matched_gt.add(best_j)
            matched_pairs.append((gt[best_j], p))
    fp = len(pr) - tp
    fn = len(gt) - tp
    return {"tp": tp, "fp": fp, "fn": fn, "matches": matched_pairs}

def compute_image_level_mse(gt_counts, pred_counts):
    arr_gt = np.array(gt_counts, dtype=float)
    arr_pr = np.array(pred_counts, dtype=float)
    assert arr_gt.shape == arr_pr.shape
    mse = np.mean((arr_gt - arr_pr)**2)
    return float(mse)

def person_level_metrics(total_tp, total_fp, total_fn):
    precision = total_tp / (total_tp + total_fp) if (total_tp+total_fp)>0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp+total_fn)>0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision+recall)>0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
