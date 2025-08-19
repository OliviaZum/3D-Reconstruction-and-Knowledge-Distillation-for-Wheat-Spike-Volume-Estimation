import warnings
from pathlib import Path
import torch
import trimesh
import pandas as pd
import numpy as np
import tqdm
import hashlib
import numpy as np
import numpy.typing as npt

from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision.io import read_image
from torchvision.transforms import v2, functional as FT
from torch import nn
from pathlib import Path
from typing import List, Literal, Tuple
from volume_prediction_fip.experiments_volume_models.shared import reproject_spike_3d

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

def vol_norm(v):
        return (v - 4763) / 1000
    
def vol_unorm(v):
    return v * 1000 + 4763

class EqualBinSampler(Sampler):
    """
    Creates bins with at least min_num_samples (rather exactly min_num_samples if possible),
    and samples them with equal probability. Can be used as sampler or to get the weight of a sample
    """
    def __init__(self, volumes: pd.Series, num_samples: int = 40, min_num_samples: int = 20):
        self.volumes = volumes
        self.num_samples = num_samples
        self.min_num_samples = min_num_samples
        bins = self._create_bins()
        weights = self._calculate_weights(bins)

        self.samples = []
        for i in weights:
            w, ind  = weights[i], bins[i]
            if pd.isna(w):
                continue
            x = (np.max(np.array(self.volumes)[ind]) + np.min(np.array(self.volumes)[ind])) * 0.5
            self.samples.append((x, w))

        self.weighted_index = []
        self.index_to_weight = {}
        for i, v in enumerate(self.volumes):
            if pd.isna(v):
                continue
            w = self.evaluate(v)
            for _ in range(int(w)):
                self.weighted_index.append(i)
            self.index_to_weight[i] = w
        max_v = max(self.index_to_weight.values())
        for j in self.index_to_weight:
            self.index_to_weight[j] = max(self.index_to_weight[j] / max_v, 0.1)

    def get_weight(self, index):
        return self.index_to_weight[index]

    def lin(self, x, i):
        x0, y0 = self.samples[i]
        x1, y1 = self.samples[i+1]
        return (y0 * (x1 - x) + y1 * (x - x0)) / (x1 - x0)

    def evaluate(self, x):
        for i in range(len(self.samples) - 1):
            if self.samples[i][0] <= x and self.samples[i+1][0] >= x:
                return self.lin(x, i)
        if x <= self.samples[0][0]:
            return self.lin(x, 0)
        else:
            return self.lin(x, -2)

    def _create_bins(self):
        """Create bins and iteratively merge bins with fewer than min_num_samples."""

        sorted_vals = self.volumes[~pd.isna(self.volumes)].sort_values().values
        bin_edges = [0]

        start_idx = 0
        while start_idx < len(sorted_vals) - 1:
            end_idx = start_idx + 1
            while end_idx < len(sorted_vals) - 1 and (end_idx - start_idx) < self.num_samples:
                end_idx += 1
            bin_edges.append(end_idx)
            start_idx = end_idx
        if len(bin_edges) >= 3 and bin_edges[-1] - bin_edges[-2] < self.min_num_samples:
            bin_edges.pop(-2)
        
        bin_edges = [sorted_vals[e] for e in bin_edges]
        bin_edges[-1] += 1
        bins = np.digitize(self.volumes, bin_edges, right=False) # Creates an additional bin for outside values (e.g. nan)
        unique_bins = np.unique(bins)
        bin_indices = {bin_idx: np.where(bins == bin_idx)[0] for bin_idx in unique_bins}

        return bin_indices

    def _calculate_weights(self, bin_indices):
        """Calculate equal sampling probability for all bins."""
        bin_counts = {i: len(indices) for i, indices in bin_indices.items()}
        bin_sizes = {i: np.max(np.array(self.volumes)[indices]) - np.min(np.array(self.volumes)[indices]) for i, indices in bin_indices.items()}
        min_bin = min(bin_sizes.values())
        max_count = max(bin_counts.values())
        weights = {i: (max_count / count) * (bin_size / min_bin) for (i, count), bin_size in zip(bin_counts.items(), bin_sizes.values())}
        return weights

    def __iter__(self):
        """Yield indices with equal probability for all bins."""
        np.random.shuffle(self.weighted_index)
        return iter(self.weighted_index)

    def __len__(self):
        return len(self.weighted_index)

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
    def __init__(self, ply_folder: Path, plant_mapping_path: str | Path, point_cloud_size: int = 1000, voxelize = False):
        self.plant_mapping = pd.read_json(plant_mapping_path, orient='index', convert_axes=False, dtype={"plant_id" : str})
        self.plant_mapping = self.plant_mapping.rename_axis("plant_id")
        self.plant_mapping = self.plant_mapping.reset_index()
        self.voxelize = voxelize

        print(ply_folder)

        ply_files = {p.stem: p for p in ply_folder.rglob("*.ply")}
        self.plant_mapping["ply_file"] = self.plant_mapping["plant_id"].map(ply_files)
        if self.plant_mapping["ply_file"].isna().any():
            missing = self.plant_mapping[self.plant_mapping["ply_file"].isna()]["plant_id"].tolist()
            raise ValueError(f"No .ply file found for plant_id(s): {missing}")

        self.point_cloud_size = point_cloud_size
        self.data = None

    def create_or_load_cache(self, path: Path, force_recompute: bool = False):
        if not path.exists() or force_recompute:
            tensors = []
            for file in tqdm.tqdm(self.plant_mapping["ply_file"].apply(Path).tolist(), "Creating cache"):
                pcl = self.load_and_process(file, True)
                pcl = np.concatenate((pcl, np.ones((pcl.shape[0], 1))), axis=1)
                if self.voxelize:
                    pcl = reproject_spike_3d.ReprojectSpike3d.voxelize_packed(pcl, voxel_size=2.0, weight_sorted=True)
                    pcl.shape[0]
                    if pcl.shape[0] > self.point_cloud_size:
                        pcl = pcl[:self.point_cloud_size]
                    elif pcl.shape[0] < self.point_cloud_size:
                        padding = np.zeros((self.point_cloud_size - pcl.shape[0], pcl.shape[1]))
                        pcl = np.vstack((pcl, padding))
                tensors.append(torch.Tensor(pcl).unsqueeze(0))
            tensors = torch.concat(tensors, dim=0)
            torch.save(tensors, path)
        else:
            tensors = torch.load(path, weights_only=True)
        self.data = tensors

    def load_and_process(self, file, get_normals=False):
        ply_data = trimesh.load(file)
        ply_data.vertices -= ply_data.vertices.mean(axis=0, keepdims=True)
        sample_size = 30000 if self.voxelize else self.point_cloud_size
        pcl, face_index = trimesh.sample.sample_surface(ply_data, sample_size)
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
            raise NotImplemented
        
        if self.voxelize:
            mask = samples[:, 6] != 0
        else:
            mask = torch.ones(len(samples), dtype=torch.bool)
        return idx, samples, torch.tensor(vol_norm(self.plant_mapping.loc[idx, "volume"]), dtype=torch.float32), mask, torch.tensor(1)
    
class DepthMapDataset(Dataset):
    def __init__(self, base_folder: Path | str, plant_mapping_path: Path | str, point_cloud_size: int = 1000, filter_minview = False):
        self.point_cloud_size = point_cloud_size
        plant_mapping = pd.read_json(plant_mapping_path, orient='index', convert_axes=False, dtype={"plant_id" : str})
        plant_mapping = plant_mapping.rename_axis("plant_id")
        self.plant_mapping = plant_mapping.reset_index()
        self.reprojector = reproject_spike_3d.ReprojectSpike3d(base_folder, self.plant_mapping["pose_file"].explode(True).unique(), filter_minview)
        self.cache = None
        if plant_mapping["labeled"].any():
            self.loss_scaling = EqualBinSampler(self.plant_mapping["volume"])

    def __len__(self):
        return len(self.plant_mapping)
    
    def set_errors(self, errors: torch.Tensor):
        self.plant_mapping["error_estimate"] = errors.numpy()
    
    def create_or_load_cache(self, path: Path, force_recompute = False):
        if path.exists() and not force_recompute:
            self.cache = torch.load(path, weights_only=True)
        else:
            self.cache = torch.zeros((len(self.plant_mapping), self.point_cloud_size, 7))
            for idx in tqdm.tqdm(range(len(self.plant_mapping)), "Creating cache"):
                row = self.plant_mapping.iloc[[idx]]
                rows = row.explode(column=["depth_name", "pose_file", "pose_key", "distance", "corner"])
                points, normals, weight = self.reprojector.reproject_images_3d(rows, voxel_size=0.002)
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

        if "error_estimate" in self.plant_mapping.columns:
            vol = torch.tensor([vol_norm(self.plant_mapping.loc[index, "volume"]), self.plant_mapping.loc[index, "error_estimate"]], dtype=torch.float32)
        else:
            vol = torch.tensor(vol_norm(self.plant_mapping.loc[index, "volume"]), dtype=torch.float32)

        mask = ~(data == 0).all(dim=1)
        if "labeled" in self.plant_mapping.columns and self.plant_mapping.loc[index, "labeled"]:
            weight = self.loss_scaling.get_weight(index)
        else:
            weight = 1
        weight = torch.tensor(weight, dtype=torch.float32)

        return index, data, vol, mask, weight
    

class Combined3dDataset(Dataset):
    def __init__(self, depthmap_dataset: DepthMapDataset, ply_dataset: PlyDataset):
        self.depthmap_dataset = depthmap_dataset
        self.ply_dataset = ply_dataset
        self.plant_mapping = self.ply_dataset.plant_mapping

        assert (self.ply_dataset.plant_mapping["plant_id"] == self.depthmap_dataset.plant_mapping["plant_id"]).all(), "Dataset have to be equal"

    def __len__(self):
        return len(self.ply_dataset)
    
    def __getitem__(self, index):
        pid = self.ply_dataset.plant_mapping.loc[index, "plant_id"]
        dmidx = self.depthmap_dataset.plant_mapping.index[self.depthmap_dataset.plant_mapping["plant_id"] == pid][0]

        _, data_ply, vol, plymask, _ = self.ply_dataset.__getitem__(index)
        _, data_depthmap, _, mask, weight  = self.depthmap_dataset.__getitem__(dmidx)
        return index, data_ply, data_depthmap, plymask, mask, vol, weight

class PlantDataError(Exception):
    """Error meant to point at missing data"""
    pass

def extract_image_features(pretrained_model, features, mask):
    """
    Apply pretrained model to images stored in features.
    """
    with torch.no_grad():
        features = pretrained_model(features[~mask])
        features = nn.utils.rnn.pad_sequence(
            features.split((~mask).sum(dim=1).tolist()),
            batch_first=True,
        )
        mask = ~(features.any(dim=-1))
        return features, mask

# Internal use
def _dnt_l2(tin):
    t = tin // 2
    if 2 * t == tin:
        l, r = t, t
    else:
        l, r = t, t+1
    return l, r

# Performs distance normalization on an image
def distance_norm_transform(torch_image, distance):
    orig_size = torch_image.shape[-2:]
    factor = distance / 3.2
    torch_image = FT.resize(torch_image, (int(torch_image.shape[-2] * factor), int(torch_image.shape[-1] * factor)), antialias=True)
    if factor <= 1:
        dw, dh = orig_size[-2] - torch_image.shape[-2], orig_size[-1] - torch_image.shape[-1]
        wl, wr = _dnt_l2(dw)
        hl, hr = _dnt_l2(dh)
        torch_image = FT.pad(torch_image, [wl, hl, wr, hr])
    else:
        torch_image = FT.center_crop(torch_image, orig_size)
    return torch_image

# Get image level transforms
# Operates under the assumption that images have been padded to some uniform size already. If this is not true
# performance will be bad!!
# Also the distance_norm_transform is not done as part of this, since additonal info (the distance) is required
def get_transform(train: bool) -> List:
     start = [v2.Resize(224, antialias=True)] 
     augmentation = [
          v2.RandomRotation(180, v2.InterpolationMode.BILINEAR),
     ] 
     end = [v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
     return start + augmentation + end if train else start + end

class SingleImgDataset(Dataset):
    """
    Returns all the images in a MultiImageTrainDataset.
    Mainly used to create cache in MultiImageTrainDataset.
    No inheritance, since then the parent would use its own child, which is conceptually weird.
    """
    def __init__(self, parent_multiImageTrainDataset) -> None:
        super().__init__()
        self.index_to_plant_and_index = {}
        self.parent_ds = parent_multiImageTrainDataset
        key = 0
        # Map index to a tuple of (plant_id, image index)
        for plant_index in range(parent_multiImageTrainDataset.plant_mapping.shape[0]):
            image_list : List[str] = parent_multiImageTrainDataset.plant_mapping.loc[plant_index, 'images']
            for index, _ in enumerate(image_list):
                self.index_to_plant_and_index[key] = (plant_index, index)
                key += 1

    def __len__(self):
        return len(self.index_to_plant_and_index)
    
    def __getitem__(self, index):
        w = self.index_to_plant_and_index[index]
        return self.parent_ds._get_image(*w)

class MultiImageTrainDataset(Dataset):

    def __init__(self,
                 image_dir : Path | str,
                 plant_mapping_path : Path | str,
                 transform: list,
                 value_column : str,
                 max_seq_len : int,
                 min_seq_len : int,
                 random_seq_len : bool = True,
                 random_generator : np.random.Generator = np.random.default_rng(10),
                 dtype : torch.dtype = torch.float32,
                 validation_mode = False, 
                 distance_normlization = True):
        """Image loader with lazy image loading.
        
        For image to be recognized, it must exist in the directory
        under the name given in the plant_mapping file
        file in value_mapping_path.
        
        Downsizes images to 256 times 256 as resnet only works
        Images are. Herefore we first do a center crop
        so it is advantageous to place the desired object in the middle
        of the image. After the corping image gets rescaled.
        
        Randomly rotates the image if `rotate` true is given.
        This can be done for exaple in combination with n_duplicates
        to 'upsample' dataset.
        

        Args:
            image_dir (Path | str): image direcotry containing images
            plant_mapping (Dataframe): dataframe file with columns: value_column and `images`
            containing a list of images for the index, which is the `plant_id`
            value_column (str): Name of the column to be read for the value
            rotate (bool, optional): Flag indicating whether or not to rotate images. Defaults to False.
            transform (list): Model specific transform settings which should include crop, normalize and resize.
            dtype (dtype, optional): dtype of image. Defaults to float32.
        
        Raises:
            FileExistsError: If Image directory does not exists
        """
        image_dir = Path(image_dir)
        plant_mapping_path = Path(plant_mapping_path)

        self.max_seq_len = max_seq_len
        self.min_seq_len = np.maximum(min_seq_len, 1)
        if self.min_seq_len > self.max_seq_len:
            raise IndexError(("Chosen min seqence lenght is larger than chosen "
                             "Max seqence lenght in MiltiImageTrainDataset."))
        self.random_sequence_len = random_seq_len
        
        self.image_dir = Path(image_dir)
        self.value_column = value_column
        self.dtype = dtype
        self.generator = random_generator
        self.transform = v2.Compose(transform)
        self.validation_mode = validation_mode
        self.distance_normalization = distance_normlization

        # fix torch seed for image rotations
        #torch.manual_seed(self.generator.integers(0,1000000))

        plant_mapping = pd.read_json(plant_mapping_path, orient='index', convert_axes=False, dtype={"plant_id" : str})
        plant_mapping = plant_mapping.rename_axis("plant_id")
        if "labeled" not in plant_mapping.columns:
            plant_mapping["labeled"] = True
        self.plant_mapping = plant_mapping.reset_index()
        self.cache = None
        self.no_label_scale = 1.0
        if plant_mapping["labeled"].any():
            self.loss_scaler = EqualBinSampler(self.plant_mapping.loc[:, "volume"])

        if not image_dir.exists() or not image_dir.is_dir():
            raise FileExistsError(f"Image directory, {str(image_dir)}, does not exist.")

    def __len__(self):
        return self.plant_mapping.shape[0]

    def _get_image(self, plant_idx : int, image_idx : int) -> torch.Tensor:
        """Loads image and transforms it.

        Args:
            plant_idx (int): Chosen plant index.
            image_idx (int): Chosen image index.

        Raises:
            Index_Error: For indexes out of bounds the existing plants / images corresponding to the plant
            PlantDataError: No images are found in the image folder which correspond to the plant.
            
        Warnings:
            If the chosen image does not exist in `image_dir` the next existing image
            corresponding to the given `plant_idx` will be chosen

        Returns:
            torch.Tensor: Transformed image as torch tensor of size (3 x 256 x 256).
        """
        image_list : List[str] = self.plant_mapping.loc[plant_idx, 'images']
        if 0 > plant_idx or plant_idx >= self.__len__():
            raise IndexError(f"Plant index {image_idx} out of boundes [0, {self.__len__()}.")
        if 0 > image_idx or image_idx >= len(image_list):
            raise IndexError(f"Image index {image_idx} out of boundes [0, {len(image_list)}) for {image_list}.")
        
        if self.cache:
            w = self.cache["index"][(plant_idx, image_idx)]
            feature = self.cache["features"][w, :, self.generator.integers(0, self.cache["features"].shape[2])]
            return feature
        else:

            image_name = image_list[image_idx]
            image_path = self.image_dir / image_name

            if not image_path.exists():
                raise PlantDataError(f"{image_path} listed in  {image_list} for plant {plant_idx} does not exist.")
            
            # read image and transform to floating point type
            torch_image = read_image(str(image_path))

            if self.distance_normalization:
                distance = self.plant_mapping.loc[plant_idx, "distance"][image_idx]
                torch_image = distance_norm_transform(torch_image, distance)

            torch_image = self.transform(torch_image)

            return torch_image
        
    def update_labels(self):
        """
        Should be called after any label has been changed
        """
        #self.loss_scaler = EqualBinSampler(self.plant_mapping.loc[:, "volume"])
        df = self.plant_mapping.copy(True)
        df.loc[df["labeled"], "volume"] = pd.NA
        self.no_label_vol_scaler = EqualBinSampler(df["volume"])
        pass
    
    def _get_random_image_sequence(self, plant_idx : int) -> npt.NDArray:
        """Generate a random image subset corresponding the plant

        Args:
            plant_idx (int): Index of the plant

        Returns:
            npt.NDArray: Array with plant indices
        """
        image_list = self.plant_mapping.loc[plant_idx,'images']
        image_indexes = np.arange(len(image_list))
        n_images = len(image_indexes)
        
        # make sure to choose a valid number of images
        max_seq_len = np.minimum(self.max_seq_len, n_images)
        min_seq_len = np.minimum(self.min_seq_len, max_seq_len)
        
        if min_seq_len < self.min_seq_len:
            warning_string = (f"The chosen min sequence length of {self.min_seq_len} is "
                           f"larger than the available max seqence lenght {max_seq_len} items.")
            warnings.warn(UserWarning(warning_string))
            
        if self.random_sequence_len and min_seq_len < max_seq_len:
            num_out = self.generator.integers(min_seq_len, max_seq_len + 1)
        else:
            num_out = max_seq_len

        image_index_choice = self.generator.choice(image_indexes, size=num_out, replace=False)
        img_names = [image_list[i] for i in image_index_choice]
        return image_index_choice, img_names

    def create_or_load_cache(self, pretrained_model: nn.Module, cache_path: Path | str, n_duplicates: int, device = 'cuda' if torch.cuda.is_available() else 'cpu',
                             num_workers: int = 2):
        
        if n_duplicates == 0:
            return
        cache_path = Path(cache_path)
        
        # Create hash for current setup
        plant_mapping = self.plant_mapping.map(lambda x : tuple(x) if isinstance(x,list) else x)
        hash_ = hashlib.sha256(pd.util.hash_pandas_object(plant_mapping, index=True).values)
        pretrained_model.train()
        pretrained_model = pretrained_model.to('cpu')
        hash_.update(str(pretrained_model).encode())
        hash_.update(str(pretrained_model.state_dict()).encode())
        hash_list = [self.image_dir, n_duplicates, self.transform]
        for x in hash_list:
            hash_.update(str(x).encode()) 
        hash_ = hash_.hexdigest()

        # Check if cache is generated
        generate_embeddings = True
        if cache_path.is_file():
            dict_ = torch.load(cache_path, weights_only=True)
            if 'hash' in dict_.keys():
                generate_embeddings = dict_['hash'] != hash_

        if generate_embeddings:
            self.cache = None # Cache needs to be disabled so dataloader loads actual images
            ds = SingleImgDataset(self)
            dataloader = DataLoader(ds, 500, False, pin_memory=True, pin_memory_device=device, num_workers=num_workers, persistent_workers=(n_duplicates > 1 and num_workers > 0))
            pretrained_model.eval()
            pretrained_model = pretrained_model.to(device)

            with torch.no_grad():
                all_features = []
                for _ in range(n_duplicates):
                    epoch_features = []
                    for img in tqdm.tqdm(dataloader, f"Creating {cache_path}, #duplicates {n_duplicates}"):
                            img = img.to(device)
                            features = pretrained_model(img)
                            epoch_features.append(features.cpu())
                    epoch_features = torch.concat(epoch_features, dim=0)
                    all_features.append(epoch_features)
                all_features = torch.stack(all_features, dim=2)
            
            self.cache = {"features": all_features, "index": {y: x for x, y in ds.index_to_plant_and_index.items()}, "hash": hash_}
            torch.save(self.cache, cache_path)
        else:
            self.cache = dict_
                            
    def __getitem__(self, idx : int) -> Tuple[torch.Tensor, float]:
        """Get a training seqence of images for one plant

        Args:
            idx (int): plant index

        Returns:
            Tuple[torch.Tensor, float, torch.Tensor]: torch image seqence (self.max_seq_len x 3 x 256 x 256),
                scaled and shifted output label, unused mask (self.max_seq_len) (true if seqence element is not used)
        """
        
        # chose image indices for the given plant index
        # consider the random number of images as well

        image_index_choice, image_names = self._get_random_image_sequence(idx)
        num_out = image_index_choice.shape[0]

        key_padding_mask = torch.zeros(self.max_seq_len, dtype=bool)
        key_padding_mask[num_out:] = True # mask all entries not to be considered
        
        # load images
        torch_image_list = []
        for image_idx in image_index_choice:
            torch_image = self._get_image(plant_idx=idx, image_idx=image_idx)
            torch_image_list.append(torch_image)
        
        torch_images_ = torch.stack(torch_image_list,dim=0)
        torch_images = torch.zeros((self.max_seq_len,*tuple(torch_images_.shape[1:])),dtype=torch_images_.dtype)
        torch_images[:num_out] = torch_images_
        
        plant: pd.Series = self.plant_mapping.loc[idx,]
        out_label = plant[self.value_column]
        out_label = torch.tensor(vol_norm(out_label), dtype=self.dtype)
        if "labeled" in self.plant_mapping.columns and not plant["labeled"]:
            #loss_weight = self.no_label_scale * self.no_label_vol_scaler.get_weight(idx)
            loss_weight = self.no_label_scale
        else:
            loss_weight = self.loss_scaler.get_weight(idx)
        loss_weight = torch.tensor(loss_weight, dtype=self.dtype)

        if self.validation_mode:
            plant["images"] = image_names
            return torch_images, plant, key_padding_mask, out_label, loss_weight
        else:
            return torch_images, out_label, key_padding_mask, loss_weight

def img_validation_collate_fn(batch : List[Tuple[torch.Tensor, pd.Series, torch.Tensor, torch.Tensor]]):
    tensors, dict_tuple, masks, labels, weights = zip(*batch)
    tensors = torch.stack(tensors=tensors, dim=0)
    masks = torch.stack(tensors=masks, dim=0)
    labels = torch.stack(tensors=labels, dim=0)
    weights = torch.stack(weights)
    return tensors, list(dict_tuple), masks, labels, weights

    
class Image3dCombidataset(Dataset):
    def __init__(self, image_dataset: MultiImageTrainDataset, point_dataset: DepthMapDataset | Combined3dDataset, return_ply=False):
        super().__init__()
        self.image_dataset = image_dataset
        self.point_dataset = point_dataset
        self.has_ply = isinstance(point_dataset, Combined3dDataset)
        self.return_ply = return_ply
        self.validation_mode = image_dataset.validation_mode

        assert (self.point_dataset.plant_mapping["plant_id"] == self.image_dataset.plant_mapping["plant_id"]).all(), "Dataset have to be equal"

    def __len__(self):
        return len(self.image_dataset)
    
    def __getitem__(self, index):
        if self.validation_mode:
            #torch_images, plant, key_padding_mask, out_label, loss_weight
            images, _, imagemask, label, _ = self.image_dataset.__getitem__(index)
        else:
            images, label, imagemask, _ = self.image_dataset.__getitem__(index)
        
        if self.has_ply:
            _, points_ply, points, plymask, pointmask, label2, weight = self.point_dataset.__getitem__(index)
        else:
            _, points, label2, pointmask, weight  = self.point_dataset.__getitem__(index)

        assert (torch.isnan(label) and torch.isnan(label2)) or abs(label - label2) < 0.001
        if self.return_ply:
            return images, points, points_ply, imagemask, pointmask, plymask, label, weight
        else:
            return images, points, imagemask, pointmask, label, weight
