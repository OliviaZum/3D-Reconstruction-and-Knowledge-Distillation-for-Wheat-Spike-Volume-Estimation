import warnings
import reproject_spike_3d
from torch.utils.data import Dataset
from pathlib import Path
import torch
import trimesh
import pandas as pd
import numpy as np
import tqdm

def random_rigid(data, translate = True):
    axis = np.random.normal(size=3)
    axis /= np.linalg.norm(axis)
    K = np.array([[    0, -axis[2],  axis[1]],
            [ axis[2],     0, -axis[0]],
            [-axis[1],  axis[0],     0]])
    angle = np.random.rand() * 2 * np.pi
    rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    rotation_matrix = torch.tensor(rotation_matrix, dtype=torch.float32)
    data[:, 0:3] = torch.matmul(rotation_matrix, data[:, 0:3].T).T
    if translate:
        translation = (torch.rand(3) - 0.5) * 10
        data[:, 0:3] += translation.unsqueeze(0)
    data[:, 3:6] = torch.matmul(rotation_matrix, data[:, 3:6].T).T
    return data

class PlyDataset(Dataset):
    """
    Loads a dataset of ply files and samples point_cloud_size points on them.
    models: Path to a csv containing a column ply_file listing the full path to the ply files to be loaded
    tensor_path: A path to cache the tensor with the samples. If load_tensor is true will attempt to load the samples from
        cache instead of resampling ply files
    Returns:
        The index of the requested file and a tensor of shape (point_cloud_size, 6) where the first 3 entries are point
            position, next 3 are normals.
    """
    def __init__(self, ply_folder: Path, plant_mapping_path: str | Path, point_cloud_size: int = 2000, augment=False):
        self.plant_mapping = pd.read_json(plant_mapping_path, orient='index', convert_axes=False, dtype={"plant_id" : str})
        self.plant_mapping = self.plant_mapping.rename_axis("plant_id")
        self.plant_mapping = self.plant_mapping.reset_index()

        ply_files = {p.stem: p for p in ply_folder.rglob("*.ply")}
        self.plant_mapping["ply_file"] = self.plant_mapping["plant_id"].map(ply_files)
        if self.plant_mapping["ply_file"].isna().any():
            missing = self.plant_mapping[self.plant_mapping["ply_file"].isna()]["plant_id"].tolist()
            raise ValueError(f"No .ply file found for plant_id(s): {missing}")

        self.point_cloud_size = point_cloud_size
        self.augment = augment
        self.data = None

    def create_or_load_cache(self, path: Path, force_recompute: bool = False):
        if not path.exists() or force_recompute:
            tensors = []
            for file in tqdm.tqdm(self.plant_mapping["ply_file"].apply(Path).tolist(), "Creating cache"):
                pcl = self.load_and_process(file, True)
                tensors.append(torch.Tensor(pcl).unsqueeze(0))
            tensors = torch.concat(tensors, dim=0)
            torch.save(tensors, path)
        else:
            tensors = torch.load(path, weights_only=True)
        self.data = tensors

    def load_and_process(self, file, get_normals=False):
        ply_data = trimesh.load(file)
        ply_data.vertices -= ply_data.vertices.mean(axis=0, keepdims=True)
        pcl, face_index = trimesh.sample.sample_surface(ply_data, self.point_cloud_size)
        if get_normals:
            normals = ply_data.face_normals[face_index]
            pcl = np.concatenate((pcl, normals), 1)
        return pcl

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.data is not None:
            samples = self.data[idx]
        else:
            file = self.plant_mapping.loc[idx, "ply_file"]
            samples = torch.Tensor(self.load_and_process(file, True)).unsqueeze(0)
        if self.augment:
            samples = random_rigid(samples, False)
        return idx, samples, torch.tensor(DepthMapDataset.vol_norm(self.plant_mapping.loc[idx, "volume"]), dtype=torch.float32)
    
class DepthMapDataset(Dataset):
    def __init__(self, base_folder: Path | str, plant_mapping_path: Path | str, point_cloud_size: int = 1000):
        self.point_cloud_size = point_cloud_size
        plant_mapping = pd.read_json(plant_mapping_path, orient='index', convert_axes=False, dtype={"plant_id" : str})
        plant_mapping = plant_mapping.rename_axis("plant_id")
        self.plant_mapping = plant_mapping.reset_index()
        self.reprojector = reproject_spike_3d.ReprojectSpike3d(base_folder, self.plant_mapping["pose_file"].explode(True).unique())
        self.cache = None

    def __len__(self):
        return len(self.plant_mapping)
    
    @staticmethod
    def vol_norm(v):
        return (v - 4500) / 1000
    
    @staticmethod
    def vol_unorm(v):
        return v * 1000 + 4500
    
    def create_or_load_cache(self, path: Path, force_recompute = False):
        if path.exists() and not force_recompute:
            self.cache = torch.load(path, weights_only=True)
        else:
            self.cache = torch.zeros((len(self.plant_mapping), self.point_cloud_size, 7))
            for idx in tqdm.tqdm(range(len(self.plant_mapping)), "Creating cache"):
                row = self.plant_mapping.iloc[[idx]]
                rows = row.explode(column=["depth_name", "pose_file", "pose_key", "distance", "corner"])
                points, normals = self.reprojector.reproject_images_3d(rows)
                points, normals, weight = self.reprojector.voxelize(points, normals, weight_sorted=True)
                data = torch.tensor(np.concatenate((points, normals, np.expand_dims(weight, 1)), axis=1), dtype=torch.float32)
                data[:, 0:3] = (data[:, 0:3] - torch.mean(data[:, 0:3], dim=0, keepdim=True)) * 1000

                inclen = min(self.point_cloud_size, data.shape[0])
                self.cache[idx, 0:inclen, :] = data[0:inclen]

            torch.save(self.cache, path)
    
    def __getitem__(self, index):
        if self.cache is None:
            raise NotImplemented
        else:
            data = self.cache[index]
            
        return index, data, torch.tensor(DepthMapDataset.vol_norm(self.plant_mapping.loc[index, "volume"]), dtype=torch.float32)
    

class Combined3dDataset(Dataset):
    def __init__(self, depthmap_dataset: DepthMapDataset, ply_dataset: PlyDataset):
        self.depthmap_dataset = depthmap_dataset
        self.ply_dataset = ply_dataset

        assert set(self.ply_dataset.plant_mapping["plant_id"]) == set(self.depthmap_dataset.plant_mapping["plant_id"]), "Dataset have to be equal"

    def __len__(self):
        return len(self.ply_dataset)
    
    def __getitem__(self, index):
        pid = self.ply_dataset.plant_mapping.loc[index, "plant_id"]
        dmidx = self.depthmap_dataset.plant_mapping.index[self.depthmap_dataset.plant_mapping["plant_id"] == pid][0]

        _, data_ply, vol = self.ply_dataset.__getitem__(index)
        _, data_depthmap, _ = self.depthmap_dataset.__getitem__(dmidx)
        return index, data_ply, data_depthmap, vol
        