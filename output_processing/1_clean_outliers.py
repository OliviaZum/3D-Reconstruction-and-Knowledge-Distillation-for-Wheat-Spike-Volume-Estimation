import cv2
import numpy as np 
import matplotlib.pyplot as plot
import os
import pandas as pd
from pathlib import Path
from datetime import datetime
import re

def is_overexposed_img(img, bright_threshold=240, overexposed_ratio=0.07, underexposed_ratio=0.0001):

    """
    Args:
        img: grayscale or BGR image (np.ndarray)
        bright_threshold: pixel value considered "very bright" (0..255 for uint8)
        overexposed_ratio: fraction of bright pixels to flag overexposure
        returns: (overexposed_bool, bright_ratio_float)
    """

    if img is None:
        raise ValueError("Image is None. Check if the path was loaded correctly.")
    
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bright_pixels = np.sum(img >= bright_threshold)
    total_pixels = img.size
    ratio = bright_pixels / total_pixels

    return ratio > overexposed_ratio, ratio < underexposed_ratio, ratio 

def crop_img(img, top=0, bottom=0, left=0, right=0):
    """
    Crop borders from the image.
    Args:
        img: numpy array
        top, bottom, left, right: number of pixels to remove from each side
    Returns:
        Cropped image
    """
    h, w = img.shape[:2]
    return img[top:h-bottom if bottom > 0 else h, left:w-right if right > 0 else w]


def scan_folder(
        root_dir, 
        imread_flag = cv2.IMREAD_GRAYSCALE, 
        bright_threshold=240, 
        overexposed_ratio = 0.07, 
        underexposed_ratio=0.0001,
        save_csv_path=None):
    
    """
    Args:
        Recursively scan root_dir, compute metrics for each image, and return a DataFrame.
        Optionally saves to CSV when save_csv_path is provided.
    """


    rows = []
    root = Path(root_dir)
    counter = 0

    for dirpath, subfolder, files in os.walk(root):
        for fname in files:
            

            if fname.lower() != "cam_07.png":  # exact match
                continue



            fpath = Path(dirpath) / fname
            img = cv2.imread(str(fpath), imread_flag)


            if img is None:
                print(f"Warning: failed to load {fpath}")
                continue

            img_cropped = crop_img(img, top=900, bottom=500, left=300, right=1000)
            # optional: save cropped image for inspection
            #counter += 1
            #debug_dir = Path("/home/zumstego/volume_prediction_fip/FIP_test/check")
            #debug_dir.mkdir(exist_ok=True)
            #out_path = debug_dir / f"{counter}_{fname}"
            #cv2.imwrite(str(out_path), img_cropped)

            mean_val = float(np.mean(img_cropped))
            std_val = float(np.std(img_cropped))
            over, under, ratio = is_overexposed_img(img_cropped, bright_threshold, overexposed_ratio, underexposed_ratio)

            rows.append({
                "path": str(fpath), 
                "subfolder": str(subfolder),
                "rel_path": str(fpath.relative_to(root)), 
                "folder": str(Path(dirpath).relative_to(root)), 
                "filename": fname,
                "mean": mean_val, 
                "std": std_val, 
                "bright_ratio": ratio, 
                "overexposed": bool(over), 
                "underexposed": bool(under),
                "bright_threshold": bright_threshold, 
                "overexposed_ratio_thresh": overexposed_ratio
            })

    df = pd.DataFrame(rows).sort_values("rel_path").reset_index(drop=True)

    if save_csv_path:
        out = Path(save_csv_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"Saved metrics to: {out}")

    return df

DATE_RE = re.compile(r"(\d{4})_(\d{2})_(\d{2})")

def list_target_dirs(root, min_date_str="2023_05_15"):
    root = Path(root)
    min_date = datetime.strptime(min_date_str, "%Y_%m_%d").date()
   
    targets = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        m = DATE_RE.search(p.name)
        if not m:
            continue
        y, mo, d = map(int, m.groups())
        folder_date = datetime(y, mo, d).date()
        if folder_date >= min_date:
            targets.append(p)
    # sort by date/name for reproducibility
    targets.sort(key=lambda x: x.name)
    print(targets)
    return targets

if __name__ == "__main__": 
    ROOT = "/data-kp/FIP/Analysis/2023/WW034/debayered/"
    #ROOT = "/home/zumstego/volume_prediction_fip/FIP_test/1_Messung_2023/FIP_feld1_datum2"
    MIN_DATE = "2023_05_15"
    BRIGHT_THRESHOLD = 240      # try 235..245 depending on your dataset
    OVEREXPOSED_RATIO = 0.05  
    UNDEREXPOSED_RATIO = 0.0001  # 5% bright pixels => overexposed
    CSV_OUT = "/home/zumstego/volume_prediction_fip/output_processing/outliers/all_2023.csv"

    subdirs = list_target_dirs(ROOT, MIN_DATE)
    print(f"Found {len(subdirs)} target folders (>= {MIN_DATE}).")

    # scan each and combine
    all_dfs = []
    for sd in subdirs:
        print(f"Scanning: {sd}")
        df = scan_folder(
            sd,
            imread_flag=cv2.IMREAD_GRAYSCALE,
            bright_threshold=BRIGHT_THRESHOLD,
            overexposed_ratio=OVEREXPOSED_RATIO,
            underexposed_ratio=UNDEREXPOSED_RATIO
        )
        if not df.empty:
            # add top-level lot folder name as a column for grouping later
            df.insert(0, "lot_folder", sd.name)
            all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    print(combined.head())
    print("\nCounts (overexposed):")
    if not combined.empty:
        print(combined["overexposed"].value_counts())

    # save combined CSV
    if not combined.empty:
        out = Path(CSV_OUT)
        out.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out, index=False)
        print(f"\nSaved combined metrics to: {out}")
    else:
        print("No images found/matched.")


    


#load images
#normal_img1 = cv2.imread("FIP_test/1_Messung_2023/FIP_feld1_datum1/FPWW0340840_FIP2_20230616_123456/cam_07.png", cv2.IMREAD_GRAYSCALE)
#normal_img2 = cv2.imread("FIP_test/1_Messung_2023/FIP_feld1_datum1/FPWW0340843_FIP2_20230616_123456/cam_07.png", cv2.IMREAD_GRAYSCALE)
#normal_img3 = cv2.imread("FIP_test/1_Messung_2023/FIP_feld1_datum1/FPWW0340844_FIP2_20230616_123456/cam_07.png", cv2.IMREAD_GRAYSCALE)
#overexposed_img = cv2.imread("FIP_test/1_Messung_2023/FIP_feld1_datum1/FPWW0340500_FIP2_20230616_123456/cam_07.png", cv2.IMREAD_GRAYSCALE)



