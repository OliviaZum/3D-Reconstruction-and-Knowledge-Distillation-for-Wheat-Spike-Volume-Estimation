"""
Generates a dataset of (single) noisefree depth and image spikes
with cameraposes as they are on the fip.
"""

from typing import Literal
import tqdm
import numpy as np
from pathlib import Path
import cv2
import pandas as pd

from volume_prediction_fip.synthetic_fip_render.gen_scene import FIPScene
from volume_prediction_fip.utils import helpers

def generate_dataset(conf_path, plant_mapping: pd.DataFrame, meshes_path: str, out_path: str, spike_pose: Literal["uniform", "normal"] = "uniform"):
    np.random.seed(1)
    scene = FIPScene(conf_path, (4096, 3000), realistic=True, until_borders=False, spike_pose=spike_pose)
    out_path = Path(out_path)
    conf_path = Path(conf_path).name

    ply_files = helpers.get_ply_files(meshes_path)
    ply_files = [p for p in ply_files if (plant_mapping["plant_id"] == Path(p).stem).any()]

    data = []
    for i, ply_file in enumerate(tqdm.tqdm(ply_files)):
        ply_file = Path(ply_file)
        row = plant_mapping.loc[plant_mapping["plant_id"] == ply_file.stem, :]
        row = row.iloc[0]

        node = scene.set_single_spike(ply_file)
        for j, cam in enumerate(scene.camnames()):
            img, depth = scene.make_image(cam)
            dist = depth[depth != 0].mean()

            mask = img == 0
            if np.all(mask):
                continue
            mask = ~np.all(mask, axis=2)
            coords = np.nonzero(mask)
            y_min, x_min = coords[0].min().item(), coords[1].min().item()
            y_max, x_max = coords[0].max().item(), coords[1].max().item()
            box = (x_min, y_min, x_max, y_max)

            img = helpers.extend_image_box(box, img, 20)
            crop_params = helpers.get_crop_params(300, img)
            img_patch, _ = helpers.put_image_on_patch(300, img, crop_params)
            depth_box, cutbox = helpers.extend_image_box(box, depth, 20, return_box=True)
            depth_patch, patchbox = helpers.put_image_on_patch(300, depth_box, crop_params)
            cutbox = cutbox[0:2] + patchbox[0:2]

            img_name = f"{row["plant_id"]}_{j}_b.jpg"
            depth_name = f"{row["plant_id"]}_{j}_b_d.npy"

            cv2.imwrite(out_path / img_name, img_patch)
            np.save(out_path / depth_name, depth_patch)

            data.append(
                (img_name, row["volume"], dist, depth_name, (cutbox[0], cutbox[1]), conf_path, cam)
            )

        scene.remove_node(node)
    
    columns = ["img_name", "volume", "distance", "depth_name", "corner", "pose_file", "pose_key"]
    df = pd.DataFrame(data, columns=columns)

    df.to_csv(out_path / "vol_mapping.csv", index=False)

if __name__ == "__main__":
    plants_to_generate = r"F:\Boxes-ds\segmented_distance_depth\split_without2024\mapping_all_plants.json"
    calibration = r"F:\Boxes-ds\artifical\artificial_fippose_toprot\2023_06_08_13_11_Lot1.json"
    ply_dir = r"F:\FIP-data\wheat-scans"
    out_dir = r"F:\Boxes-ds\artifical\artificial_fippose_toprot"

    plant_mapping = pd.read_json(plants_to_generate,
                                  orient='index', convert_axes=False, dtype={"plant_id" : str})
    plant_mapping = plant_mapping.rename_axis("plant_id")
    plant_mapping = plant_mapping.reset_index()
    generate_dataset(calibration, plant_mapping,
                     ply_dir, out_dir, "normal")
