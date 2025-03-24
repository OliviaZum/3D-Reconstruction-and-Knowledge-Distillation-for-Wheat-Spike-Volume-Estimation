from pathlib import Path
from typing import Any, Dict, List, Tuple
from ultralytics import YOLO
from ultralytics.engine.results import Results
from fip_global import fip_detection
from utils.helpers import put_image_on_patch, extend_image_box, get_crop_params
import json
import argparse
import time
import torch
import torch.nn.functional as F
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from experiments_volume_models.shared.datasets import distance_norm_transform, get_transform, vol_unorm
from experiments_volume_models.shared.models import unflat
import pandas as pd
import gc

class Stopwatch:
    def __init__(self):
        self.start_time = None
        self.start()

    def start(self):
        self.start_time = time.time()

    def elapsed(self):
        a = time.time() - self.start_time
        self.reset()
        return a

    def reset(self):
        self.start()

def export_plant_images(folder: Path, patches, cluster_ids):
    import cv2

    c_to_p = cluster_to_patches(patches, cluster_ids)

    for id, patches in c_to_p.items():
        for i, patch in enumerate(patches):
            cv2.imwrite(folder / f"{id}_{i}.jpg", patch)
    
class BatchedAccessor:
    def __init__(self, batch_size, total_len):
        self.batch_size = batch_size
        self.total_len = total_len

    def get_range(self):
        return range(0, self.total_len, self.batch_size)

    def get(self, i, data):
        return data[i:min(i + self.batch_size, self.total_len)]
    

# Consider: 
# Due to batch processing of the segmentation your images end up having size 288 instead of 300 (Probs irrelevant, later anyway scaled down)
# Maybe do a better way of selecting the segmented spike?

def infer_volume(calibration_file: Path | str,
                 image_folder: Path | str,
                 output_folder: Path,
                 min_view: int,
                 detection_model: YOLO,
                 segmentation_model: YOLO,
                 dinov2_model,
                 volume_model):
    
    with torch.no_grad():
        batch_size_seg = 70
        batch_size_dino = 500
        batch_size_volpred = 256
        patch_box_size = 300
        imgsz_seg = 288
        min_img_after_seg = 6
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        """
        if output_folder.exists():
            print("The ouput folder already exists. Aborting.")
            exit(1)
        output_folder.mkdir()
        """
        # DEBUG
        output_folder.mkdir(exist_ok=True)



        with open(calibration_file) as f:
            calibration = json.load(f)

        s = Stopwatch()

        # Detection of spikes

        print("Detecting spikes") 
        boxes, results = fip_detection.find_objects_yolo(detection_model, image_folder)
        img_map = {f"cam_{i:02}.png": results[i-1].orig_img for i in range(1, 13)}

        detection_model.cpu()
        torch.cuda.empty_cache() 

        print(f"Spike pairing (Detection took: {s.elapsed()})") 
        # Spike pairing
        connected_boxes = fip_detection.connect_boxes(boxes, calibration, min_view)
        torch.cuda.empty_cache()

        # Padding detections and removing small clusters
        patches = []
        connected_filtered = []
        spike_counter = set()
        for box in connected_boxes:
            if box["cluster"] is not None:
                img = extend_image_box(box["box"], img_map[box["image"]], 20)
                cp = get_crop_params(patch_box_size, img)
                patch, _ = put_image_on_patch(patch_box_size, img, cp)
                
                patches.append(patch)
                connected_filtered.append(box)
                spike_counter.add(box["cluster"])

        print(f"Found {len(spike_counter)} spikes with >= {min_view} observations.")
        print(f"Preprocessing (Spike pairing took: {s.elapsed()})")

        # Preprocessing
        seg_accessor = BatchedAccessor(batch_size_seg, len(patches))
        img_transform = v2.Compose(get_transform(False))
        patches_preprocessed = []
        for i in seg_accessor.get_range():
            img_list = seg_accessor.get(i, patches)
            meta_list = seg_accessor.get(i, connected_filtered)
            
            w, h, _ = img_list[0].shape
            torch_img = torch.zeros((len(img_list), 3, w, h))
            for i, img in enumerate(img_list):
                torch_img[i] = torch.tensor(img).permute(2, 0, 1)
            torch_img = torch_img.to(device)
            torch_img /= 255
            torch_img = F.interpolate(torch_img, (imgsz_seg, imgsz_seg), mode="bilinear")

            res: List[Results] = segmentation_model(torch_img, imgsz=imgsz_seg, verbose=False, conf=0.2)

            # Avoid this dangling around after the loop. Also we anyway need masks on cpu.
            del torch_img
            for i in range(len(res)):
                res[i] = res[i].cpu()

            # Yeah yeah not particularly efficient. Segmentation accounts for roughly three times as much time than this, so kept simple.
            for r, meta in zip(res, meta_list):
                seg_patch = fip_detection.clear_background_segmented(r)
                if seg_patch is None:
                    patches_preprocessed.append(None)
                else:
                    torch_image = torch.tensor(seg_patch).permute(2, 0, 1)
                    torch_image = distance_norm_transform(torch_image, meta["distance"])
                    torch_image = img_transform(torch_image)
                    patches_preprocessed.append(torch_image)

        segmentation_model.cpu()
        torch.cuda.empty_cache()

        # Put into clusters
        cluster_to_patch = {}
        seg_fail_total = 0
        seg_succ_total = 0
        for patch, meta in zip(patches_preprocessed, connected_filtered):
            id = meta["cluster"]
            if patch is None:
                seg_fail_total += 1
                continue
            cluster_to_patch.setdefault(id, {})
            cluster_to_patch[id][meta["image"]] = (patch, meta)
            seg_succ_total += 1

        ctopsize = len(cluster_to_patch)
        cluster_to_patch = {key: value for key, value in cluster_to_patch.items() if len(value) >= min_img_after_seg}
        num_removed_plants_seg = ctopsize - len(cluster_to_patch)

        print(f"""Segmentation failed for {seg_fail_total} and worked for {seg_succ_total} patches. {num_removed_plants_seg} spikes where removed 
                for having less than {min_img_after_seg} observations after segmentation. Remaining plants: {len(cluster_to_patch)}""")

        print(f"Inference (Preprocessing took {s.elapsed()})")

        # Build dino input list and mask tensor
        mask = torch.ones((len(cluster_to_patch), 12), dtype=torch.bool)
        dino_patch_list = []
        for cidx, v in enumerate(cluster_to_patch.values()):
            patches, _ = zip(*v.values())
            for pidx, patch in enumerate(patches):
                dino_patch_list.append(patch)
                mask[cidx, pidx] = False

        # Extract dino features
        dinofeatures = []
        dino_accessor = BatchedAccessor(batch_size_dino, len(dino_patch_list))
        dinov2_model = dinov2_model.eval().to(device)
        for i in dino_accessor.get_range():
            batch = dino_accessor.get(i, dino_patch_list)
            batch = torch.stack(batch).to(device)
            f = dinov2_model(batch).cpu()
            dinofeatures.append(f)

        dinofeatures = torch.concat(dinofeatures)
        paired_features = unflat((*mask.shape, 384), mask, dinofeatures)

        del batch
        dinov2_model.cpu()
        torch.cuda.empty_cache()

        # Predict volume
        vol_accessor = BatchedAccessor(batch_size_volpred, paired_features.shape[0])
        volumes = []
        volume_model = volume_model.eval().to(device)
        for i in vol_accessor.get_range():
            batch = vol_accessor.get(i, paired_features)
            mask_batch = vol_accessor.get(i, mask)
            batch = batch.to(device)
            mask_batch = mask_batch.to(device)
            v, _ = volume_model(batch, mask_batch)
            v = v.cpu().squeeze()
            volumes.append(v)
        volumes = torch.concat(volumes)
        volumes = vol_unorm(volumes)

        del batch
        del mask_batch
        volume_model.cpu()
        torch.cuda.empty_cache()
        print(f"Done. (Inference took {s.elapsed()})")

        print(volumes)
        pass

    
    """

    img_folder = output_folder / "image_export"
    img_folder.mkdir()
    export_plant_images(img_folder, patches, cluster_ids)

    exit(0)
    """



if __name__ == "__main__":
    yolo_det = YOLO("assets/model-weights/yolo-medium-detect-mAp50-0766.pt")
    yolo_seg = YOLO("assets/model-weights/yolo-medium-segment.pt")
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    vol_model = torch.load("assets/model-weights/self-distill-regulated_transformer.pth", weights_only=False)

    infer_volume("assets/poses/2023_07_10_12_56_Lot1.json",
                 r"F:\FIP-data\images\2023\WW034\debayered\2023_07_10_12_56_Lot1\FPWW0340091_FIP2_20230710_120851",
                 Path("local_stuff/tmp_output"), 10,
                 yolo_det, yolo_seg, dino, vol_model)
    exit(0)


    parser = argparse.ArgumentParser(description="Process images with multiple models.")

    parser.add_argument("-c", "--calibration_file", type=Path, required=True, help="Path to calibration file")
    parser.add_argument("-i", "--image_folder", type=Path, required=True, help="Path to the folder containing images")
    parser.add_argument("-o", "--output_folder", type=Path, required=True, help="Path to the output folder")
    parser.add_argument("-mv", "--min_view", type=int, required=True, help="Minimum number of views")
    parser.add_argument("-dm", "--detection_model", type=Path, required=True, help="Path to detection model")
    parser.add_argument("-sm", "--segmentation_model", type=Path, required=True, help="Path to segmentation model")
    parser.add_argument("-vm", "--volume_model", type=Path, required=True, help="Path to volume model")

    args = parser.parse_args()