"""
The actual end to end pipeline for computing volumes on a FIP scan.
See help text for how to use.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
from ultralytics import YOLO
from ultralytics.engine.results import Results
import json
import argparse
import time
import torch
import torch.nn.functional as F
from torchvision.transforms import v2
import pandas as pd
import accelerate
import cv2
import numpy as np


from volume_prediction_fip.fip_global import fip_detection
from volume_prediction_fip.fip_global import volumes_in_square
from volume_prediction_fip.utils import helpers
from volume_prediction_fip.experiments_volume_models.shared.datasets import distance_norm_transform, get_transform, vol_unorm
from volume_prediction_fip.experiments_volume_models.shared.models import unflat


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

"""
Computes the volume over spikes for a FIP scan. For parameters see help text in main.

Returns: A pd dataframe containing info about each spike (the volume, the bounding boxes of detections on each camera,
how many detections (with successfull segmentation), and the cluster_id). 

The second returned value (which probably useless in most cases) is a dict mapping from clusterid to a dict mapping
from camera name to a tuple: First element is the preprocessed patch (i.e. before passing to DINO),
second element is meta info about the detection (a dict containing e.g. distance, bounding box, camera name and so on), last element is the 
patch without preprocessing. (I.e. padding, but no other things like segmentation, distance normalization or color normalization).
You are guarranted that when iterating over the dict the elements appear in the same order than when iterating over the dataframe.
Also you can always get the patches corresponding to a result spike by accessing the dict at cluster_id.
"""
def infer_volume(image_folder: Path | str,
                 output_folder: Path,
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
        detection_model = detection_model.to(device)
        boxes, results = fip_detection.find_objects_yolo(detection_model, image_folder)
        img_map = {f"cam_{i:02}.png": results[i-1].orig_img for i in range(1, 13)}

        if move_models_to_cpu:
            detection_model.cpu()
            torch.cuda.empty_cache()

        if verbose:
            print(f"Spike pairing (Detection took: {s.elapsed()})")

        # Spike pairing, distances to cameras and estimated 3d values
        connected_boxes, distances, estimated_3d_pos = fip_detection.connect_boxes(boxes, calibration, min_view)
        print("estimated_3d_pos")
        print(estimated_3d_pos)

        # average distance to camera 7
        dist_cam = {k: v for k, v in distances.items() if k == 'cam_07.png'}
        distances_cam = dist_cam['cam_07.png']
        #remove none
        filtered_cam_dist = {k: v for k, v in distances_cam.items() if v is not None}
        #mean distance: 
        mean_value = np.mean(list(filtered_cam_dist.values()))   
        print("mean height")
        print(mean_value)

        #outputs all labels (spikes) in a square of 40 x 40 cm in camera 7
        labels_in_square, reconstructed_3d, square_3d = volumes_in_square.fn_labels_in_square(estimated_3d_pos, mean_value, output_folder, save_image=True)
     
        
        if move_models_to_cpu:
            torch.cuda.empty_cache()

        # Padding detections and removing small clusters
        patches = []
        connected_filtered = []
        spike_counter = set()
        for box in connected_boxes:
            if box["cluster"] is not None:
                img = helpers.extend_image_box(box["box"], img_map[box["image"]], 20)
                cp = helpers.get_crop_params(patch_box_size, img)
                patch, _ = helpers.put_image_on_patch(patch_box_size, img, cp)
                
                patches.append(patch)
                connected_filtered.append(box)
                spike_counter.add(box["cluster"])

        if verbose:
            print(f"Found {len(spike_counter)} spikes with >= {min_view} observations.")
            print(f"Preprocessing (Spike pairing took: {s.elapsed()})")

        # Preprocessing
        seg_accessor = BatchedAccessor(batch_size_seg, len(patches))
        img_transform = v2.Compose(get_transform(False))
        segmentation_model = segmentation_model.to(device)
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
            v = v.cpu().view(-1)
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
        for cidx, (cluster_id, v) in enumerate(cluster_to_patch.items()):
            _, metas, _ = zip(*v.values())
            r = {"index": cidx, "volume": volumes[cidx].item(), "num_observations": len(metas), "cluster_id": cluster_id}
            for meta in metas:
                r[meta["image"]] = meta["box"]
            results.append(r)

        results = pd.DataFrame(results)
      
        #filter all the clusters / labels that are in the 40 x 40 cm square
        filtered_df = results[results["cluster_id"].isin(labels_in_square)].copy()

        return results, cluster_to_patch, reconstructed_3d, square_3d, filtered_df

def get_default_models():
    yolo_det = helpers.get_detection_model()
    yolo_seg = helpers.get_segmentation_model()
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    vol_model = torch.load(helpers.get_volume_model_path(), weights_only=False)
    return yolo_det, yolo_seg, dino, vol_model

def main():
    parser = argparse.ArgumentParser(description="""
        Estimates volumes of wheat spikes for a FIP set of images. 
        You need properly calibrated and scaled cameras (the calibration file).
        Calibration can be reused for any number of FIP scans, as long as the FIP does not change.
        If output_directory is specified the patches after preprocessing will be exported to a folder. This is useful
        for debugging - if e.g. all the spike pairs seem wrong, the calibration is probably of.
    """)

    parser.add_argument("-c", "--calibration_file", type=Path, required=True, help="Path to calibration file (Create via generate_scaled_calibration in fip global)")
    parser.add_argument("-i", "--image_folder", type=Path, required=True, help="Path to the folder containing the images")
    parser.add_argument("-od", "--output_directory", type=Path, required=False, help="A folder to which all preprocessed spikes, dataframes, and figures are exported (Folder with plot name)")
    parser.add_argument("-mv", "--min_view", type=int, required=False, default=12, help="Minimum number of observations for a spike to estimate volume")
    parser.add_argument("-dv", "--disable_verbose", action="store_true", required=False, help="Disable printing")

    args = parser.parse_args()
    verbose = not args.disable_verbose


    output_directory = Path(args.output_directory)

    #create folder with plot name
    if args.output_directory:
        args.output_directory.mkdir(parents=True, exist_ok=True)
    if args.output_directory is not None and not args.output_directory.is_dir():
        print(f"{args.output_directory} is not a directory. Aborting.")
        exit(1)
    

    #do something if output is already there, eg jump to next plot
    #output_file = args.output_directory / "predicted_volume.csv"
    #if output_file.is_file():
    #        input(f"The output file exists. Press enter to continue overwrite, or Ctrl+C to abort")


    with open(args.calibration_file) as f:
        calibration = json.load(f)


    yolo_det, yolo_seg, dino, vol_model = get_default_models()
    r, cluster_to_patch, reconstruced_3d, square_3d, filtered_df = infer_volume(args.image_folder, output_directory, calibration, 
            args.min_view, yolo_det, yolo_seg, dino, vol_model, verbose=verbose)
    
    
    # Get plot name from image folder
    plot_name = args.image_folder.name

    # Save dataframes with plot name
    r.to_csv(output_directory / f"{plot_name}_predicted_volume.csv", index=False)
    filtered_df.to_csv(output_directory / f"{plot_name}_filtered_results.csv", index=False)

    print("output directory")
    print(output_directory)

    #plot final BB, total and in square --> take a lot of space!! (50 MB)
    volumes_in_square.plot_BB_cam7(r, reconstruced_3d, filtered_df, square_3d, output_directory, args.image_folder)
    volumes_in_square.plot_BB_cam7_2colors(r, reconstruced_3d, filtered_df, square_3d, output_directory, args.image_folder)

    # create a subfolder so save spikes
    subfolder = output_directory / "spikes"
    subfolder.mkdir(parents=True, exist_ok=True)

    if args.output_directory is not None:
        for i, (vol, cluster) in enumerate(zip(r["volume"].to_list(), cluster_to_patch.values())):
            _, _, patches = zip(*cluster.values())
            for j, patch in enumerate(patches):
                output_path = subfolder / f"{i}_{j}_{vol}.jpg"
                cv2.imwrite(str(output_path), patch)
                

if __name__ == "__main__":
    main()