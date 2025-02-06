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
    def __init__(self, base_folder: Path | str, plant_mapping_path: Path | str, point_cloud_size: int = 2000, min_seq_len: int = 3, max_seq_len: int = 12, augment=True):
        self.point_cloud_size = point_cloud_size
        plant_mapping = pd.read_json(plant_mapping_path, orient='index', convert_axes=False, dtype={"plant_id" : str})
        plant_mapping = plant_mapping.rename_axis("plant_id")
        self.plant_mapping = plant_mapping.reset_index()
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.reprojector = reproject_spike_3d.ReprojectSpike3d(base_folder, self.plant_mapping["pose_file"].explode(True).unique())
        self.cache = None
        self.augment = augment
        self.torch_generator = torch.Generator()
        self.reset_generator()
    
    def reset_generator(self):
        self.torch_generator.manual_seed(0)
        self.np_generator = np.random.default_rng(0)

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
            self.cache = torch.zeros((len(self.plant_mapping), 12, self.point_cloud_size, 6))
            for idx in tqdm.tqdm(range(len(self.plant_mapping)), "Creating cache"):
                row = self.plant_mapping.iloc[[idx]]
                rows = row.explode(column=["depth_name", "pose_file", "pose_key", "distance", "corner"])
                for img_idx, (_, row) in enumerate(rows.iterrows()):
                    points, normals = self.reprojector.reproject_to_3d(row)
                    data = torch.tensor(np.concatenate((points, normals), axis=1), dtype=torch.float32)
                    subset = torch.randperm(data.shape[0])[:min(self.point_cloud_size, data.shape[0])]
                    self.cache[idx, img_idx, 0:len(subset), :] = data[subset]
            torch.save(self.cache, path)
    
    def __getitem__(self, index):
        row = self.plant_mapping.iloc[[index]]
        images = row["depth_name"].tolist()[0]
        if self.augment:
            num_img = self.np_generator.integers(min(self.min_seq_len, len(images)), min(self.max_seq_len, len(images)) + 1)
        else:
            num_img = len(images)
        selection = self.np_generator.choice(np.arange(0, len(images)), num_img, replace=False)
        if self.cache is None:
            rows = row.explode(column=["depth_name", "pose_file", "pose_key", "distance", "corner"])
            rows = rows.iloc[selection]
            points, normals = self.reprojector.reproject_images_3d(rows, self.point_cloud_size)
            data = torch.tensor(np.concatenate((points, normals), axis=1), dtype=torch.float32)
        else:
            points = self.cache[index, selection, ]
            points = points.flatten(0, 1)
            points = points[~(points == 0).all(dim=1)]
            if points.shape[0] < self.point_cloud_size: # More of outlier handling than a real thing, in practice expected to rarely ever happen
                data = torch.zeros((self.point_cloud_size, 6))
                data[0:points.shape[0]] = points
                #return self.__getitem__(torch.randint(0, len(self), (1,)).item())
            else:
                selected_points = torch.randperm(points.shape[0], generator=self.torch_generator)[:self.point_cloud_size]
                data = points[selected_points]
            
        data[:, 0:3] = (data[:, 0:3] - torch.mean(data[:, 0:3], dim=0, keepdim=True)) * 1000
        #if self.augment:
        #    data = random_affine(data, False)
            
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
        