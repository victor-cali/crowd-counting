# src/utils.py
import os
import glob
import cv2
import numpy as np
import pandas as pd
from ast import literal_eval

def list_images(img_glob="imatges/*.jpg"):
    return sorted(glob.glob(img_glob))

def load_image(path, gray=True):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    if gray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def find_annotation_csv(root="."):
    # search common names
    candidates = []
    for name in ["annotations.csv", "labels.csv", "annotations_export.csv", "makesense.csv"]:
        p = os.path.join(root, name)
        if os.path.exists(p):
            candidates.append(p)
    # fallback: any csv in root
    if not candidates:
        candidates = glob.glob(os.path.join(root, "*.csv"))
    return candidates[0] if candidates else None

def parse_makesense_csv(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)

    cols = [c.lower() for c in df.columns]

    # detect your special format
    # example columns: ['person','240','867','1660806000.jpg','1920','1080']
    if len(df.columns) == 6:
        # assume format:
        # col0 = label
        # col1 = x
        # col2 = y
        # col3 = image filename
        points_by_image = {}
        for _, row in df.iterrows():
            label = row[df.columns[0]]
            x = float(row[df.columns[1]])
            y = float(row[df.columns[2]])
            fname = str(row[df.columns[3]])
            points_by_image.setdefault(fname, []).append((x, y))
        return points_by_image

    # OTHERWISE fall back to old logic
    raise ValueError(
        f"Unsupported CSV format. Columns: {df.columns.tolist()}"
    )

def overlay_points(img, points, color=(255,0,0), radius=4):
    # img: BGR, points: list of (x,y)
    out = img.copy()
    for (x,y) in points:
        cv2.circle(out, (int(round(x)), int(round(y))), radius, color, -1)
    return out
