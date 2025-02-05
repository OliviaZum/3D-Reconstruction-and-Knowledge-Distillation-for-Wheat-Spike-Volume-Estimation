import numpy as np
import pandas as pd
from pathlib import Path
import json
import ast
from scipy.signal import convolve2d

class ReprojectSpike3d:
    def __init__(self, folder_path, pose_files):
        # Load the projection matrix
        self.folder_path = Path(folder_path)
        self.poses = {}
        for pose_file in pose_files:
            path = self.folder_path / pose_file
            with open(path, 'r') as f:
                pose_data = json.load(f)
            for key in pose_data:
                K, Rt_ = ReprojectSpike3d.get_projection(pose_data, key)
                Rt = np.eye(4)
                Rt[0:3, :] = Rt_
                self.poses[(str(pose_file), key)] = (K, Rt)

    @staticmethod
    def load_k(intrinsics):
        f = intrinsics["focal_length"]
        pp = np.array(intrinsics["principal_point"])
        return np.array([
            [f, 0, pp[0]],
            [0, f, pp[1]],
            [0, 0, 1]
        ])

    @staticmethod
    def get_projection(poses_conf, cam: str):
        cam_data = poses_conf[cam]

        R = np.array(cam_data["extrinsics"]["rotation"])
        t = np.array(cam_data["extrinsics"]["center"])
        K = ReprojectSpike3d.load_k(cam_data["intrinsics"])

        extrinsics = np.zeros((3, 4))
        extrinsics[0:3, 0:3] = R
        extrinsics[0:3, 3] = -R @ t

        return K, extrinsics

    def reproject_to_3d(self, row: pd.Series, compute_normals=True):
        """
        Reproject points to 3D using the given folder path and CSV row information.
        
        Args:
            folder_path (Path): The base folder path containing all the files.
            row (pd.Series): A row from the CSV containing data about the image and metadata.
        
        Returns:
            np.ndarray: The 3D points in world coordinates of shape (n, 3).
        """
        depth_path = self.folder_path / row["depth_name"]
        depth_map = np.load(depth_path)
        mask = np.logical_and(np.logical_and(depth_map != 0, depth_map > row["distance"] - 0.2), depth_map < row["distance"] + 0.2)

        # Generate normals
        if compute_normals:
            grad_x = convolve2d(depth_map, np.array([[-1, 0, 1],
                                                        [-1, 0, 1],
                                                        [-1, 0, 1]]), mode='same')
            grad_y = convolve2d(depth_map, np.array([[-1, -1, -1],
                                                    [0, 0, 0],
                                                    [1, 1, 1]]), mode='same')
            norm_dist = np.ones_like(grad_x) * 0.005 # a guess, in theory would require to take the actual distance in 3d
            v_x = np.stack((norm_dist, np.zeros_like(grad_x), -grad_x), axis=2)
            v_y = np.stack((np.zeros_like(grad_y), norm_dist, -grad_y), axis=2)
            normal_map = - np.cross(v_x, v_y)
            length = np.sqrt(np.sum((normal_map ** 2), axis=2, keepdims=True))
            normal_map /= length

        K, Rt = self.poses[(str(row["pose_file"]), row["pose_key"])]

        corner = ast.literal_eval(row["corner"])
        corner_x, corner_y = corner
        
        # Reproject to 3D
        height, width = depth_map.shape
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
        x_coords = x_coords[mask] + corner_x  # Offset by the crop corner
        y_coords = y_coords[mask] + corner_y  # Offset by the crop corner
        
        points_camera = np.vstack((x_coords, y_coords, np.ones_like(x_coords))) * depth_map[mask]
        points_3d_view = np.linalg.inv(K) @ points_camera

        points_3d_world = np.linalg.inv(Rt) @ np.vstack((points_3d_view, np.ones_like(x_coords)))
        if compute_normals:
            normals_3d_world = Rt[0:3, 0:3].T @ normal_map[mask].T
        points_3d_world = points_3d_world[:3].T
        
        if compute_normals:
            return points_3d_world, normals_3d_world.T  # Shape: (n, 3)
        else:
            return points_3d_world

    def voxel_filter(points, normals, voxel_size: float, minhits: int):
        # Removes points with less than minhits in their voxel
        w = points * (1 / voxel_size)
        w = w.astype(np.int64)
        voxels, unique_counts  = np.unique(w, axis=0, return_counts=True)
        selected_voxels = voxels[unique_counts >= minhits]
        mask = np.isin(voxels, selected_voxels)
        return points[mask, :], normals[mask, :]

    def reproject_images_3d(self, rows: pd.DataFrame, selection_size = 2000):
        points = []
        normals = []
        for _, row in rows.iterrows():
            p, normal = self.reproject_to_3d(row)
            points.append(p)
            normals.append(normal)

        points = np.concatenate(points, axis=0)
        normals = np.concatenate(normals, axis=0)

        #points, normals = voxel_filter(points, normals, voxel_size, min_hits) # Remove noise and sample down in a regular fashion
        points -= points.mean(axis=0, keepdims=True)
        # Can lead to same point picked multiple times. Arbitrary unrealistic in practice (also not a big issue), unless there is a very low number of points
        selection = np.random.choice(np.arange(0, len(points)), selection_size, True) 
        points, normals = points[selection, :], normals[selection, :]
        return points, normals

