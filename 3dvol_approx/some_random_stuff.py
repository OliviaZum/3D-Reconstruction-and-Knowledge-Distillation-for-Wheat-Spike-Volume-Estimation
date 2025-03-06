import utils_experiment
import utils_3d
import numpy as np
import reproject_spike_3d
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

def watch_ply_files():
    train, _, _ = utils_experiment.get_real_dataset3d()
    base = Path(r"F:\Boxes-ds\segmented_distance_depth")
    reprojector = reproject_spike_3d.ReprojectSpike3d(base, train.plant_mapping["pose_file"].explode(True).unique())
    for i in range(len(train)):
        #_, b, _ = train.__getitem__(i)
        row = train.plant_mapping.iloc[[i]]
        rows = row.explode(column=["depth_name", "pose_file", "pose_key", "distance", "corner", "images"])

        """
        for i, row in rows.iterrows():
            fig, axes = plt.subplots(1, 2)
            # Load image
            img_path = base / row["images"]
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

            # Load depth map
            depth_path = base / row["depth_name"]
            depth = np.load(depth_path)

            # Normalize depth (ignoring 0 values)
            valid_mask = depth > 0
            if np.any(valid_mask):
                depth_min, depth_max = depth[valid_mask].min(), depth[valid_mask].max()
                depth[valid_mask] = (depth[valid_mask] - depth_min) / (depth_max - depth_min)

            # Plot image
            axes[0].imshow(img)
            axes[0].set_title(f"Image {i}")
            axes[0].axis("off")

            # Plot depth map
            axes[1].imshow(depth, cmap="gray", vmin=0, vmax=1)
            axes[1].set_title(f"Depth Map {i}")
            axes[1].axis("off")

            plt.tight_layout()
            plt.show()
        """

        points, normals, weights = reprojector.reproject_images_3d(rows, voxel_size=0.002, min_view=3)
        print(len(points))
        #points, normals, weights = reprojector.voxelize(points, normals, voxel_size=0.001, weight_sorted=False)
        utils_3d.visualize_point_clouds(points)

def watch_artifical_fip_plyfiles():
    train, _, _ = utils_experiment.get_artifical_fiplike_dataset()
    base = Path(r"F:\Boxes-ds\artifical_fippose_ds")
    reprojector = reproject_spike_3d.ReprojectSpike3d(base, train.plant_mapping["pose_file"].explode(True).unique())
    for i in range(len(train)):
        row = train.plant_mapping.iloc[[i]]
        rows = row.explode(column=["depth_name", "pose_file", "pose_key", "distance", "corner", "images"])

        points, normals, weights = reprojector.reproject_images_3d(rows, voxel_size=0.001)
        print(len(points))
        #points, normals, weights = reprojector.voxelize(points, normals, voxel_size=0.001, weight_sorted=False)
        utils_3d.visualize_point_clouds(points)

watch_artifical_fip_plyfiles()