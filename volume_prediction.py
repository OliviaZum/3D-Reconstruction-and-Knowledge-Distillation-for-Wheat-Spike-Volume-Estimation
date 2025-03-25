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
from experiments_volume_models.shared.datasets import distance_norm_transform, get_transform, vol_unorm
from experiments_volume_models.shared.models import unflat
import pandas as pd
import accelerate
import cv2

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

def infer_volume(image_folder: Path | str,
                 calibration: Dict,
                 min_view: int,
                 detection_model: YOLO,
                 segmentation_model: YOLO,
                 dinov2_model: torch.nn.Module,
                 volume_model: torch.nn.Module,
                 verbose = True,
                 set_seed=True) -> Tuple[pd.DataFrame, Dict[int, Dict[str, Tuple]]]:
    
    if set_seed:
        accelerate.utils.set_seed(0, deterministic=True)
    
    with torch.no_grad():
        # Essentially settings for computation. Probably never worthwile changing, unless you want to optimize runtime/memory usage
        # on a particular system.
        batch_size_seg = 70 # Batch size for segmentation
        batch_size_dino = 500 # Batch size for dino feature extraction
        batch_size_volpred = 256 # Batch size for volume prediction
        min_img_after_seg = 6 # The minimum number of images a cluster should have after segmentation (if below the cluster is removed)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # Move models between cpu and gpu to decrease gpu mem usage. If there is lots of memory, one could make this
        # slightly faster by disabling.
        move_models_to_cpu = True 

        # Keep fixed
        patch_box_size = 300
        imgsz_seg = 288
        s = Stopwatch()

        # Detection of spikes
        if verbose:
            print("Detecting spikes") 
        boxes, results = fip_detection.find_objects_yolo(detection_model, image_folder)
        img_map = {f"cam_{i:02}.png": results[i-1].orig_img for i in range(1, 13)}

        if move_models_to_cpu:
            detection_model.cpu()
            torch.cuda.empty_cache()

        if verbose:
            print(f"Spike pairing (Detection took: {s.elapsed()})")
        # Spike pairing
        connected_boxes = fip_detection.connect_boxes(boxes, calibration, min_view)

        if move_models_to_cpu:
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

        if verbose:
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

        if move_models_to_cpu:
            segmentation_model.cpu()
            torch.cuda.empty_cache()

        # Put into clusters
        cluster_to_patch = {}
        seg_fail_total = 0
        seg_succ_total = 0
        for patch, meta, patch_no_preprocess in zip(patches_preprocessed, connected_filtered, patches):
            id = meta["cluster"]
            if patch is None:
                seg_fail_total += 1
                continue
            cluster_to_patch.setdefault(id, {})
            cluster_to_patch[id][meta["image"]] = (patch, meta, patch_no_preprocess)
            seg_succ_total += 1

        ctopsize = len(cluster_to_patch)
        cluster_to_patch = {key: value for key, value in cluster_to_patch.items() if len(value) >= min_img_after_seg}
        num_removed_plants_seg = ctopsize - len(cluster_to_patch)

        if verbose:
            print(f"""Segmentation failed for {seg_fail_total} and worked for {seg_succ_total} patches. {num_removed_plants_seg} spikes where removed 
                    for having less than {min_img_after_seg} observations after segmentation. Remaining plants: {len(cluster_to_patch)}""")
            print(f"Inference (Preprocessing took {s.elapsed()})")

        # Build dino input list and mask tensor
        mask = torch.ones((len(cluster_to_patch), 12), dtype=torch.bool)
        dino_patch_list = []
        for cidx, v in enumerate(cluster_to_patch.values()):
            patches, _, _ = zip(*v.values())
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
        if move_models_to_cpu:
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
        volumes = vol_unorm(volumes).to(torch.int32)

        del batch
        del mask_batch
        if move_models_to_cpu:
            volume_model.cpu()
            torch.cuda.empty_cache()
        if verbose:
            print(f"Done. (Inference took {s.elapsed()})")

        results = []
        for cidx, v in enumerate(cluster_to_patch.values()):
            _, metas, _ = zip(*v.values())
            r = {"volume": volumes[cidx].item(), "num_observations": len(metas)}
            for meta in metas:
                r[meta["image"]] = meta["box"]
            results.append(r)

        results = pd.DataFrame(results)
        return results, cluster_to_patch

def get_default_models():
    yolo_det = YOLO("assets/model-weights/yolo-medium-detect-mAp50-0766.pt")
    yolo_seg = YOLO("assets/model-weights/yolo-medium-segment.pt")
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    vol_model = torch.load("assets/model-weights/self-distill-regulated_transformer.pth", weights_only=False)
    return yolo_det, yolo_seg, dino, vol_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
        Estimates volumes of wheat spikes for a FIP set of images. 
        You need properly calibrated and scaled cameras (the calibration file).
        Calibration can be reused for any number of FIP scans, as long as the FIP does not change.
        If output_directory is specified the patches after preprocessing will be exported to a folder. This is useful
        for debugging - if e.g. all the spike pairs seem wrong, the calibration is probably of.
    """)

    parser.add_argument("-c", "--calibration_file", type=Path, required=True, help="Path to calibration file (Create via generate_scaled_calibration in fip global)")
    parser.add_argument("-i", "--image_folder", type=Path, required=True, help="Path to the folder containing the images")
    parser.add_argument("-o", "--output_file", type=Path, required=True, help="Path to store the output file to")
    parser.add_argument("-od", "--output_directory", type=Path, required=False, help="A folder to which all preprocessed spikes are exported")
    parser.add_argument("-mv", "--min_view", type=int, required=False, default=12, help="Minimum number of observations for a spike to estimate volume")
    parser.add_argument("-v", "--verbose", action="store_true", required=False, help="Disable printing")

    args = parser.parse_args()

    with open(args.calibration_file) as f:
        calibration = json.load(f)

    if args.output_directory is not None and not args.output_directory.is_dir():
        print(f"{args.output_directory} is not a directory. Aborting.")
        exit(1)
    if not args.output_file.parent.is_dir():
        print(f"The folder for {args.output_file} seems to not exists. Aborting.")
        exit(1)

    yolo_det, yolo_seg, dino, vol_model = get_default_models()
    r, cluster_to_patch = infer_volume(args.image_folder, calibration, 
            args.min_view, yolo_det, yolo_seg, dino, vol_model, verbose=args.verbose)
    
    r.to_csv(args.output_file)

    if args.output_directory is not None:
        for i, (vol, cluster) in enumerate(zip(r["volume"].to_list(), cluster_to_patch.values())):
            _, _, patches = zip(*cluster.values())
            for j, patch in enumerate(patches):
                cv2.imwrite(args.output_directory / f"{i}_{j}_{vol}.jpg", patch)
