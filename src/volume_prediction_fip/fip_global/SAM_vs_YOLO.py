import os
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import os
from typing import List
from ultralytics import YOLO
from ultralytics.engine.results import Results
import torch
import numpy as np
from ultralytics import SAM
from ultralytics.data.converter import convert_coco
from pathlib import Path
import shutil
import cv2
from volume_prediction_fip.utils import helpers
from volume_prediction_fip.fip_global import fip_detection
import time



######################################################

import json
CURRENT_IMAGE = None


def load_coco_annotations_for_image(coco_json_path, image_filename):
    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    image_id = None
    for img in coco["images"]:
        if img["file_name"] == image_filename:
            image_id = img["id"]
            break

    assert image_id is not None, "Image not found in COCO"

    anns = [
        ann for ann in coco["annotations"]
        if ann["image_id"] == image_id
    ]
    return anns


def save_comparison_crops(img_path, coco_annotations, snippets_dir):
    """
    Saves aligned (image, GT mask, prediction mask) crops
    using orig_box from detection.
    """

    out_dir = snippets_dir / "gt_comparison"
    out_dir.mkdir(exist_ok=True)

    ious = []
    num_valid = 0


    full_img = cv2.imread(str(img_path))
    H, W = full_img.shape[:2]

    # -------------------------------------------------
    # build full GT mask once
    # -------------------------------------------------
    gt_mask = np.zeros((H, W), dtype=np.uint8)
    for ann in coco_annotations:
        for poly in ann["segmentation"]:
            pts = np.array(poly).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(gt_mask, [pts], 1)

    # -------------------------------------------------
    # loop over snippets / detections
    # -------------------------------------------------
    for meta_path in sorted(snippets_dir.glob("snippet_*_meta.npy")):
        idx = meta_path.stem.split("_")[1]
        meta = np.load(meta_path, allow_pickle=True).item()

        # --- boxes
        ox1, oy1, ox2, oy2 = meta["orig_box"]
        cx1, cy1, cx2, cy2 = meta["cutbox"]

        # --- crops from full image
        img_crop = full_img[oy1:oy2, ox1:ox2]
        # --- crops from full image
        img_crop = full_img[oy1:oy2, ox1:ox2]
        gt_crop = gt_mask[oy1:oy2, ox1:ox2]

        # -------------------------------------------------
        # GT cleanup: keep only largest connected component
        # -------------------------------------------------
        if gt_crop.sum() > 0:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                gt_crop.astype(np.uint8),
                connectivity=8
            )

            # skip background (label 0)
            if num_labels > 1:
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                gt_crop = (labels == largest_label).astype(np.uint8)

                if img_crop.size == 0:
                    continue

        # -------------------------------------------------
        # load predicted patch mask (300x300)
        # -------------------------------------------------
        seg_path = snippets_dir / f"snippet_{idx}_seg.png"
        if not seg_path.exists():
            continue

        pred = cv2.imread(str(seg_path))
        if pred is None:
            continue

        # binary patch mask
        pred_mask = (np.max(pred, axis=2) > 0).astype(np.uint8)

        if pred_mask.shape == gt_crop.shape:
            mask_orig = pred_mask
            mask_cut = None  



        # -------------------------------------------------
        # PATCH → CUTBOX (REMOVE PATCH PADDING ONLY)
        # -------------------------------------------------
        else: 
            cut_w = cx2 - cx1
            cut_h = cy2 - cy1
            if cut_w <= 0 or cut_h <= 0:
                continue

            # put_image_on_patch centers the crop in the patch
            patch_h, patch_w = pred_mask.shape

            start_y = (patch_h - cut_h) // 2
            start_x = (patch_w - cut_w) // 2

            mask_cut = pred_mask[
                start_y : start_y + cut_h,
                start_x : start_x + cut_w
            ]

            if mask_cut.sum() == 0:
                print(f"[{idx}] Pred empty after unpadding → skipping")
                continue


            # -------------------------------------------------
            # CUTBOX → ORIG_BOX (remove padding)
            # -------------------------------------------------
            mask_orig = np.zeros_like(gt_crop, dtype=np.uint8)

            dx = ox1 - cx1
            dy = oy1 - cy1

            h, w = gt_crop.shape

            y0 = max(dy, 0)
            x0 = max(dx, 0)
            y1 = min(dy + h, mask_cut.shape[0])
            x1 = min(dx + w, mask_cut.shape[1])

            oy0 = y0 - dy
            ox0 = x0 - dx
            oy1 = oy0 + (y1 - y0)
            ox1 = ox0 + (x1 - x0)

            mask_orig = np.zeros_like(gt_crop, dtype=np.uint8)
            mask_orig[oy0:oy1, ox0:ox1] = mask_cut[y0:y1, x0:x1]


        # -------------------------------------------------
        # sanity checks
        # -------------------------------------------------
        if mask_orig.sum() == 0:
            print(f"[{idx}] Pred empty → skipping")
            continue

        if gt_crop.sum() == 0:
            print(f"[{idx}] GT empty → skipping")
            continue

        assert mask_orig.shape == gt_crop.shape

        # -------------------------------------------------
        # save aligned crops
        # -------------------------------------------------
        cv2.imwrite(out_dir / f"{idx}_img.png", img_crop)
        cv2.imwrite(out_dir / f"{idx}_gt.png", gt_crop * 255)
        cv2.imwrite(out_dir / f"{idx}_pred.png", mask_orig * 255)

        # -------------------------------------------------
        # IoU
        # -------------------------------------------------
        intersection = np.logical_and(mask_orig, gt_crop).sum()
        union = np.logical_or(mask_orig, gt_crop).sum()
        iou = intersection / (union + 1e-6)

        if iou > 0:
            ious.append(iou)
            num_valid += 1

        print(
            f"[{idx}] OK | "
            f"patch {pred_mask.shape} | "
            f"cut {None if mask_cut is None else mask_cut.shape} | "
            f"orig {mask_orig.shape} | "
            f"IoU = {iou:.3f}"
        )



    if num_valid > 0:
        mean_iou = float(np.mean(ious))
        print(f"\nImage mean IoU over {num_valid} spikes: {mean_iou:.3f}")
    else:
        print("\nImage mean IoU: no valid IoUs (>0)")

    print(f"Saved GT comparison crops to {out_dir}")

    return ious



def detection():
    
    # detection: 
    def get_package_folder():
        return Path(__file__).parent.parent.parent.parent
    def get_assets_path():
        return get_package_folder() / "assets"
    def get_detection_model():
        #return YOLO(get_assets_path() / "model-weights" / "yolo-medium-detect-mAp50-0766.pt")
        return YOLO(get_assets_path() / "model-weights" / "best_yolo11l_40ep.pt")
    det_model = get_detection_model()


    img_path = CURRENT_IMAGE


    # -----------------------
    # input
    # -----------------------

    img = cv2.imread(str(img_path))
    assert img is not None

    # -----------------------
    # output folder
    # -----------------------
    out_dir = img_path.parent / "snipplets" / img_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------
    # detection
    # -----------------------
    det_r = det_model(str(img_path))[0]

    if det_r.boxes is None or det_r.boxes.shape[0] == 0:
        print("No detections found")
        return

    boxes = det_r.boxes.xyxy.cpu().numpy().astype(int)

    # save raw boxes (for SAM later)
    np.save(out_dir / "boxes_xyxy.npy", boxes)

    # -----------------------
    # parameters (EXACTLY as original)
    # -----------------------
    PADDING = 8
    PATCH_SIZE = 300

    # -----------------------
    # crop using ORIGINAL helpers
    # -----------------------
    for i, box in enumerate(boxes):

        # 1) extend box (padding + clipping)
        crop_img, cutbox = helpers.extend_image_box(
            box,
            img,
            padding=PADDING,
            return_box=True
        )
        

        # 2) compute crop params (scaling + centering)
        crop_params = helpers.get_crop_params(
            PATCH_SIZE,
            crop_img,
        )

        # 3) place on fixed-size patch
        patch, patchbox = helpers.put_image_on_patch(
            PATCH_SIZE,
            crop_img,
            crop_params,
        )

        # save snippet
        cv2.imwrite(
            str(out_dir / f"snippet_{i:03d}.png"),
            patch,
        )

        # save FULL metadata for exact reversibility
        meta = {
            "orig_box": [int(x) for x in box],
            "cutbox": [int(x) for x in cutbox],
            "crop_params": crop_params,
            "patchbox": patchbox,
            "patch_size": PATCH_SIZE,
        }

        np.save(out_dir / f"snippet_{i:03d}_meta.npy", meta)





######################################################

def segmentation_yolo():
    
    # -----------------------
    # model
    # -----------------------
    seg_model = helpers.get_segmentation_model()

    # -----------------------
    # paths
    # -----------------------
    img_path = CURRENT_IMAGE

    snippets_dir = img_path.parent / "snipplets" / img_path.stem
    assert snippets_dir.exists(), f"Snipplets folder not found: {snippets_dir}"

    # -----------------------
    # load full image once
    # -----------------------
    full_img = cv2.imread(str(img_path))
    assert full_img is not None
    H, W = full_img.shape[:2]

    # full-image combined mask
    full_mask_combined = np.zeros((H, W), dtype=np.uint8)


    # -----------------------
    # Preload snippets (NO I/O in timing)
    # -----------------------
    snippets = []
    for snippet_path in sorted(snippets_dir.glob("snippet_[0-9][0-9][0-9].png")):
        img = cv2.imread(str(snippet_path))
        assert img is not None
        snippets.append(img)

    
    if len(snippets) == 0:
        print("No snippets found.")
        return []


    # -----------------------
    # Warmup (important!)
    # -----------------------
    _ = seg_model(snippets[0], imgsz=288, verbose=False)
    torch.cuda.synchronize()

    # -----------------------
    # Batched inference
    # -----------------------
    torch.cuda.synchronize()
    total_start = time.perf_counter()

    results = seg_model(
    snippets,
    imgsz=288,
    verbose=False
    )

    for snippet_path, r in zip(
        sorted(snippets_dir.glob("snippet_[0-9][0-9][0-9].png")),
        results
    ):

        if r.masks is None:
            continue

        mask = r.masks.data[0].cpu().numpy()
        mask = (mask > 0).astype(np.uint8)

        # resize mask back to 300x300
        mask = cv2.resize(
            mask.astype(np.uint8),
            (300, 300),
            interpolation=cv2.INTER_NEAREST
        )

        combined_mask = np.zeros((300, 300, 3), dtype=np.uint8)
        combined_mask[mask == 1] = [255, 255, 255]


        out_path = snippet_path.with_name(
            snippet_path.stem + "_seg.png"
        )

        cv2.imwrite(str(out_path), combined_mask)

        # -----------------------
        # reconstruct snippet mask into full-image coordinates
        # -----------------------
        idx = snippet_path.stem.split("_")[1]
        meta_path = snippets_dir / f"snippet_{idx}_meta.npy"
        meta = np.load(meta_path, allow_pickle=True).item()

        ox1, oy1, ox2, oy2 = meta["orig_box"]
        cx1, cy1, cx2, cy2 = meta["cutbox"]

        cut_w = cx2 - cx1
        cut_h = cy2 - cy1
        if cut_w <= 0 or cut_h <= 0:
            continue

        # PATCH -> CUTBOX: remove patch padding
        patch_h, patch_w = mask.shape
        start_y = (patch_h - cut_h) // 2
        start_x = (patch_w - cut_w) // 2

        mask_cut = mask[
            start_y:start_y + cut_h,
            start_x:start_x + cut_w
        ]

        if mask_cut.sum() == 0:
            continue

        # CUTBOX -> ORIG_BOX: remove crop padding
        mask_orig = np.zeros((oy2 - oy1, ox2 - ox1), dtype=np.uint8)

        dx = ox1 - cx1
        dy = oy1 - cy1

        h, w = mask_orig.shape

        y0 = max(dy, 0)
        x0 = max(dx, 0)
        y1 = min(dy + h, mask_cut.shape[0])
        x1 = min(dx + w, mask_cut.shape[1])

        out_y0 = y0 - dy
        out_x0 = x0 - dx
        out_y1 = out_y0 + (y1 - y0)
        out_x1 = out_x0 + (x1 - x0)

        if y1 > y0 and x1 > x0:
            mask_orig[out_y0:out_y1, out_x0:out_x1] = mask_cut[y0:y1, x0:x1]

        if mask_orig.sum() == 0:
            continue

        # paste into full-image mask
        full_mask_combined[oy1:oy2, ox1:ox2] = np.logical_or(
            full_mask_combined[oy1:oy2, ox1:ox2],
            mask_orig
        ).astype(np.uint8)

    cv2.imwrite(
    str(snippets_dir / "full_image_seg.png"),
    full_mask_combined * 255
    )

    torch.cuda.synchronize()
    total_end = time.perf_counter()

    print(f"\nYOLO batched segmentation time (full image): {(total_end - total_start)*1000:.2f} ms")

        

    # -----------------------
    # load GT + save comparison crops
    # -----------------------
    coco_annotations = load_coco_annotations_for_image(
        coco_json_path=Path("/projects/zumstego/volume_prediction_fip/FIP-coco-instance/annotations/instances_default.json"),
        image_filename=img_path.name,
    )

    ious = save_comparison_crops(
    img_path=img_path,
    coco_annotations=coco_annotations,
    snippets_dir=snippets_dir
    )

    return ious




def segmentation_SAM():
    """
    Run SAM on the SAME detections, save results into *_sam folder,
    then run the same GT comparison.
    """

    # -------------------------------------------------
    # model
    # -------------------------------------------------
    model = SAM("assets/model-weights/sam2_s.pt")

    # -------------------------------------------------
    # paths
    # -------------------------------------------------
    img_path = CURRENT_IMAGE
    img_name = img_path.stem


    base_dir = Path(
    "/projects/zumstego/volume_prediction_fip/FIP-coco-instance/images/snipplets"
    )


    snippets_dir = base_dir / img_name
    assert snippets_dir.exists(), f"Snipplets folder not found: {snippets_dir}"

    snippets_dir_sam = base_dir / f"{img_name}_sam"
    snippets_dir_sam.mkdir(exist_ok=True)

    # -------------------------------------------------
    # copy snippets + meta ONCE
    # -------------------------------------------------
    for p in snippets_dir.glob("snippet_[0-9][0-9][0-9].png"):
        shutil.copy(p, snippets_dir_sam / p.name)

    for p in snippets_dir.glob("snippet_*_meta.npy"):
        shutil.copy(p, snippets_dir_sam / p.name)

    # -------------------------------------------------
    # LOAD FULL IMAGE ONCE 
    # -------------------------------------------------
    full_img = cv2.imread(str(img_path))
    assert full_img is not None

    full_mask_combined = np.zeros(full_img.shape[:2], dtype=np.uint8)

    # -------------------------------------------------
    # run SAM segmentation
    # -------------------------------------------------

    # collect all boxes first
    PADDING = 8
    PATCH_SIZE = 300

    meta_paths = sorted(snippets_dir.glob("snippet_*_meta.npy"))

    torch.cuda.synchronize()
    total_start = time.perf_counter()

    for meta_path in meta_paths:

        meta = np.load(meta_path, allow_pickle=True).item()
        x1, y1, x2, y2 = meta["orig_box"]

        # ---- RUN SAM FOR THIS BOX ONLY ----
        r = model(
            source=full_img,
            bboxes=[[x1, y1, x2, y2]],
        )[0]

        if r.masks is None:
            continue

        full_mask = (r.masks.data[0].cpu().numpy() > 0).astype(np.uint8)

        full_mask_combined = np.logical_or(full_mask_combined, full_mask).astype(np.uint8)

        # ---- reproduce SAME geometry as detection() ----
        crop_mask, _ = helpers.extend_image_box(
            [x1, y1, x2, y2],
            full_mask,
            padding=PADDING,
            return_box=True
        )

        crop_params = helpers.get_crop_params(
            PATCH_SIZE,
            crop_mask,
        )

        mask_patch, _ = helpers.put_image_on_patch(
            PATCH_SIZE,
            crop_mask,
            crop_params,
        )

        combined_mask = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
        combined_mask[mask_patch == 1] = [255, 255, 255]

        snippet_name = meta_path.stem.replace("_meta", "")
        out_mask_path = snippets_dir_sam / f"{snippet_name}_seg.png"

        cv2.imwrite(str(out_mask_path), combined_mask)

    cv2.imwrite(
    str(snippets_dir_sam / "full_image_seg.png"),
    full_mask_combined * 255
    )

    torch.cuda.synchronize()
    total_end = time.perf_counter()

    print(f"SAM total inference time (full image): {(total_end - total_start)*1000:.2f} ms")

    
    # -------------------------------------------------
    # GT comparison (UNCHANGED)
    # -------------------------------------------------
    coco_annotations = load_coco_annotations_for_image(
        coco_json_path=Path(
            "/projects/zumstego/volume_prediction_fip/FIP-coco-instance/annotations/instances_default.json"
        ),
        image_filename=f"{img_name}.png",
    )

    ious = save_comparison_crops(
    img_path=img_path,
    coco_annotations=coco_annotations,
    snippets_dir=snippets_dir_sam
    )

    return ious


IMAGES_DIR = Path(
    "/projects/zumstego/volume_prediction_fip/FIP-coco-instance/images"
)

if __name__ == "__main__":
    all_ious = []
    for img_path in sorted(IMAGES_DIR.glob("*.png")):

        print(f"\nProcessing {img_path.name}")

        CURRENT_IMAGE = img_path

        detection()
        #ious = segmentation_yolo() 
        
        #GLOBAL IoU (non-zero only):
        #Mean     : 0.7341
        #Variance : 0.028426
        #Std Dev  : 0.1686
        #Used 2464 / 2464 masks

        ious = segmentation_SAM()


        if ious is not None:
            all_ious.extend(ious)

    # remove zero IoUs
    valid_ious = [iou for iou in all_ious if iou > 0]

    all_ious_np = np.array(all_ious)

    if len(all_ious_np) == 0:
        print("\nNo IoUs collected at all.")
    else:
        # ---- ALL IoUs ----
        mean_all = float(all_ious_np.mean())
        var_all = float(all_ious_np.var())
        std_all = float(all_ious_np.std())

        print(f"\nGLOBAL IoU (all):")
        print(f"Mean     : {mean_all:.4f}")
        print(f"Variance : {var_all:.6f}")
        print(f"Std Dev  : {std_all:.4f}")

        # ---- NON-ZERO IoUs ONLY ----
        valid_ious = all_ious_np[all_ious_np > 0]

        if len(valid_ious) > 0:
            mean_valid = float(valid_ious.mean())
            var_valid = float(valid_ious.var())
            std_valid = float(valid_ious.std())

            print(f"\nGLOBAL IoU (non-zero only):")
            print(f"Mean     : {mean_valid:.4f}")
            print(f"Variance : {var_valid:.6f}")
            print(f"Std Dev  : {std_valid:.4f}")
            print(f"Used {len(valid_ious)} / {len(all_ious_np)} masks")
        else:
            print("\nNo non-zero IoUs found.")
