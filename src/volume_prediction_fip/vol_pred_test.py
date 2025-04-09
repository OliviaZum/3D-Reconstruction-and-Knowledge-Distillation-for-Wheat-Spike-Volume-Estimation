"""
A quick test for the volume_prediction. Purpose is not to measure accuracy or so,
just to check if something seems wrong with the pipeline as such. For that the pipeline is
run on a few scans followed by checking if the pipeline agrees with the ground truth more or less.
(There is no differentiation between test and train, since the aim is not to estimate population risk)
"""
from fip_dataset.FIPDataset import FIPDataset
import volume_prediction
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    nscans = 10

    config = {
            "annotation_file": r"F:\FIP-data\csv\labeled_spikes.csv",
            "csv_folder": r"F:\FIP-data\csv",
            "img_folder": r"F:\FIP-data\images",
            "ply_folder": r"F:\FIP-data\wheat-scans",
            "precompute_file": r"F:\FIP-data\csv\precomputed.json",
            "pose_folder": r"C:\Users\Admin\Desktop\master_thesis\volume_prediction_fip\assets\poses"
        }
    data: FIPDataset = FIPDataset(config["csv_folder"], config["img_folder"], config["ply_folder"], config["precompute_file"], config["annotation_file"], config["pose_folder"])

    year2023 = data.spikescans[data.spikescans["year"] == "2023"] # Just ignore everything from 2024, because of last day
    img_dirs = year2023["image_dir"].unique()
    yolo_det, yolo_seg, dino, vol_model = volume_prediction.get_default_models()

    vol_pred = []
    vol_gt = []
    nopredfound_counter = 0
    for i in range(nscans):
        img_dir = img_dirs[i]
        print(img_dir)
        sub = data.spikescans[data.spikescans["image_dir"] == img_dir]

        r, _ = volume_prediction.infer_volume(img_dir, data.get_pose_config(img_dir), 9, yolo_det, yolo_seg, dino, vol_model, set_seed=False)

        for _, spike in sub.iterrows():
            main_cam = spike["label_selectedon"]
            main_box = spike[main_cam]

            det_boxes = r[main_cam].to_list()
            vols = r["volume"].to_list()
            max_iou = 0
            max_box = None
            for i, w in enumerate(det_boxes):
                if not isinstance(w, list): # nan, no detection for this camera
                    continue
                iou = FIPDataset.iou(main_box, w)
                if iou > max_iou:
                    max_iou = iou
                    max_box = i
            if max_iou > 0.8:
                vol_pred.append(vols[max_box])
                vol_gt.append(spike["spikevolume"])
            else:
                nopredfound_counter += 1
    print(f"Did not find corresponding box for {nopredfound_counter} spikes.")
    print(f"Correlation: {np.corrcoef(vol_gt, vol_pred)[0, 1]}")
    plt.scatter(vol_gt, vol_pred)
    plt.plot((2000, 9000), (2000, 9000))
    plt.xlabel("GT")
    plt.ylabel("Pred")
    plt.show()

        


        

