"""
Contains mostly methods to quickly get certain types of dataset instances.
If running experiments, file paths have to be adapted to point to the actual dataset.
"""

from typing import Literal
from experiments_volume_models.shared import datasets
from pathlib import Path
import torch

def _get_image_dataset(split_folder: Path, base_folder: Path, cache_folder_image: Path, ndupli_train = 10, enable_distance = True):
    split_folder, base_folder, cache_folder_image = [Path(s) for s in [split_folder, base_folder, cache_folder_image]]
    pretrained_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    cache_folder_image.mkdir(exist_ok=True)
    train_dataset_image = datasets.MultiImageTrainDataset(base_folder, split_folder / "mapping_train.json", 
                                                          datasets.get_transform(True), "volume", 12, 4, True, distance_normlization=enable_distance)
    train_dataset_image.create_or_load_cache(pretrained_model, cache_folder_image / "traincache.pth", ndupli_train, num_workers=4)
    val_dataset_image = datasets.MultiImageTrainDataset(base_folder, split_folder / "mapping_val.json", datasets.get_transform(False),
                                                        "volume", 12, 6, False, validation_mode=True, distance_normlization=enable_distance)
    val_dataset_image.create_or_load_cache(pretrained_model, cache_folder_image / "valcache.pth", 1, num_workers=0)
    test_dataset_image = datasets.MultiImageTrainDataset(base_folder, split_folder / "mapping_test.json", datasets.get_transform(False),
                                                         "volume", 12, 6, False, validation_mode=True, distance_normlization=enable_distance)
    test_dataset_image.create_or_load_cache(pretrained_model, cache_folder_image / "testcache.pth", 1, num_workers=0)

    return train_dataset_image, val_dataset_image, test_dataset_image

def _get_depthmap_dataset(split_folder, base_folder, cache_folder, force_recompute = False, filter_minview = False):
    split_folder, base_folder, cache_folder = [Path(s) for s in [split_folder, base_folder, cache_folder]]
    cache_folder.mkdir(exist_ok=True)
    train_dataset = datasets.DepthMapDataset(base_folder, split_folder / "mapping_train.json", filter_minview=filter_minview)
    train_dataset.create_or_load_cache(cache_folder / "dmap_cache_train.pth", force_recompute=force_recompute)
    val_dataset = datasets.DepthMapDataset(base_folder, split_folder / "mapping_val.json", filter_minview=filter_minview)
    val_dataset.create_or_load_cache(cache_folder / "dmap_cache_val.pth", force_recompute=force_recompute)
    test_dataset = datasets.DepthMapDataset(base_folder,split_folder / "mapping_test.json", filter_minview=filter_minview)
    test_dataset.create_or_load_cache(cache_folder / "dmap_cache_test.pth", force_recompute=force_recompute)

    return train_dataset, val_dataset, test_dataset

def _get_ply_dataset(split_folder, base_folder, cache_folder, force_recompute = False):
    split_folder, base_folder, cache_folder = [Path(s) for s in [split_folder, base_folder, cache_folder]]
    cache_folder.mkdir(exist_ok=True)
    train_dataset = datasets.PlyDataset(base_folder, split_folder / "mapping_train.json")
    train_dataset.create_or_load_cache(cache_folder / "ply_train.pth", force_recompute=force_recompute)
    val_dataset = datasets.PlyDataset(base_folder, split_folder / "mapping_val.json")
    val_dataset.create_or_load_cache(cache_folder / "ply_val.pth", force_recompute=force_recompute)
    test_dataset = datasets.PlyDataset(base_folder, split_folder / "mapping_test.json")
    test_dataset.create_or_load_cache(cache_folder / "ply_test.pth", force_recompute=force_recompute)

    return train_dataset, val_dataset, test_dataset

# Get a dataset of real 3d reconstructed spikes
def get_real_dataset3d(force_recompute = False):
    split_folder = Path("split_without2024")
    base_folder = Path(r"F:\Boxes-ds\segmented_distance_depth")
    split_folder = base_folder / split_folder
    cache = Path(r"experiments_volume_models\local_stuff\dmap_cache")

    return _get_depthmap_dataset(split_folder, base_folder, cache, force_recompute)

# Get artificial point clouds dataset
def get_ply_dataset(force_recompute = False):
    split_folder = Path(r"F:\Boxes-ds\segmented_distance_depth\split_without2024")
    base_folder = Path(r"F:\FIP-data\wheat-scans")
    cache_path = Path(r"experiments_volume_models\local_stuff\ply_cache")

    return _get_ply_dataset(split_folder, base_folder, cache_path, force_recompute)
    
# Returns real and corresponding artifical pointcloud at the same time
def get_combined_point_real_arti_dataset():
    r_train_ds, r_val_ds, r_test_ds = get_real_dataset3d()
    a_train_ds, a_val_ds, a_test_ds = get_ply_dataset()
    train_dataset = datasets.Combined3dDataset(r_train_ds, a_train_ds)
    val_dataset = datasets.Combined3dDataset(r_val_ds, a_val_ds)
    test_dataset = datasets.Combined3dDataset(r_test_ds, a_test_ds)
    return train_dataset, val_dataset, test_dataset

# Unlabeled image and 3d dataset
def get_unlabeled_dataset(force_recompute = False):
    split_folder = Path(r"F:\Boxes-ds\unlabeled-5000-depth\split")
    base_folder = Path(r"F:\Boxes-ds\unlabeled-5000-depth")
    cache_folder = Path(r"experiments_volume_models\local_stuff\dmap_cache")

    train_dataset_dm = datasets.DepthMapDataset(base_folder, split_folder / "mapping_train.json")
    train_dataset_dm.create_or_load_cache(cache_folder / "dmap_cache_unlabeled_train.pth", force_recompute=force_recompute)
    val_dataset_dm = datasets.DepthMapDataset(base_folder, split_folder / "mapping_test.json")
    val_dataset_dm.create_or_load_cache(cache_folder / "dmap_cache_unlabeled_val.pth", force_recompute=force_recompute)
    pretrained_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    train_dataset_img = datasets.MultiImageTrainDataset(base_folder, split_folder / "mapping_train.json",
                                                           datasets.get_transform(True), "volume", 12, 4, True)
    train_dataset_img.create_or_load_cache(pretrained_model, cache_folder / "img_cache_unlabeled_train.pth", 5, num_workers=4)
    val_dataset_img = datasets.MultiImageTrainDataset(base_folder, split_folder / "mapping_test.json", 
                                                         datasets.get_transform(False), "volume", 12, 6, False, validation_mode=True)
    val_dataset_img.create_or_load_cache(pretrained_model, cache_folder / "img_cache_unlabeled_val.pth", 1, num_workers=0)
    
    return train_dataset_dm, val_dataset_dm, train_dataset_img, val_dataset_img

# Default image dataset
def get_default_image_dataset():
    split_folder = Path(r"F:\Boxes-ds\segmented_distance_depth\split_without2024")
    base_folder = Path(r"F:\Boxes-ds\segmented_distance_depth")
    cache_folder_image = Path(r"F:\Boxes-ds\_cache_dataset_embeddings")

    return _get_image_dataset(split_folder, base_folder, cache_folder_image)

def get_default_image_dataset_wo_distance():
    split_folder = Path(r"F:\Boxes-ds\segmented_distance_depth\split_without2024")
    base_folder = Path(r"F:\Boxes-ds\segmented_distance_depth")
    cache_folder = Path(r"experiments_volume_models\local_stuff\nodistancenorm_cache")

    return _get_image_dataset(split_folder, base_folder, cache_folder, enable_distance=False)

def get_default_image_dataset_wo_distance_and_seg(disable_augmentation = False):
    split_folder = Path(r"F:\Boxes-ds\spike_dataset_manual_20_pad\split_without2024")
    base_folder = Path(r"F:\Boxes-ds\spike_dataset_manual_20_pad")
    if disable_augmentation:
        cache_folder = Path(r"experiments_volume_models\local_stuff\nodistancenosegnoaug_cache")
        ndupli = 1
    else:
        cache_folder = Path(r"experiments_volume_models\local_stuff\nodistancenoseg_cache")
        ndupli = 10

    return _get_image_dataset(split_folder, base_folder, cache_folder, enable_distance=False, ndupli_train=ndupli)

def get_artifical_image_dataset_sideviews():
    split = r"F:\Boxes-ds\artifical\bestpose_noshift_12\split_without2024_noextend"
    base = r"F:\Boxes-ds\artifical\bestpose_noshift_12"
    cache = r"experiments_volume_models\local_stuff\artificial_sideviews"
    return _get_image_dataset(split, base, cache, enable_distance=False)

def get_artificial_image_dataset_randompose():
    split = r"F:\Boxes-ds\artifical\randompose_noshift_12\split_without2024_noextend"
    base = r"F:\Boxes-ds\artifical\randompose_noshift_12"
    cache = r"experiments_volume_models\local_stuff\artificial_randomviews"
    return _get_image_dataset(split, base, cache, enable_distance=False)

def get_artificial_dataset_fiplike_uniform(type: Literal["depth", "image"], force_recompute = False):
    split = r"F:\Boxes-ds\artifical\artifical_fippose_randomrot\split_without2024_noextend"
    base = r"F:\Boxes-ds\artifical\artifical_fippose_randomrot"
    
    if type == "image":
        cache = r"experiments_volume_models\local_stuff\artificial_fiplike_uniform_image"
        return _get_image_dataset(split, base, cache, enable_distance=False)
    else:
        cache = r"experiments_volume_models\local_stuff\artificial_fiplike_uniform_depth"
        return _get_depthmap_dataset(split, base, cache, force_recompute, True)

def get_artificial_dataset_fiplike_normal(type: Literal["depth", "image"], force_recompute = False):
    split = r"F:\Boxes-ds\artifical\artificial_fippose_toprot\split_without2024_noextend"
    base = r"F:\Boxes-ds\artifical\artificial_fippose_toprot"

    if type == "image":
        cache = r"experiments_volume_models\local_stuff\artificial_fiplike_normal_image"
        return _get_image_dataset(split, base, cache, enable_distance=False)
    else:
        cache = r"experiments_volume_models\local_stuff\artificial_fiplike_normal_depth"
        return _get_depthmap_dataset(split, base, cache, force_recompute, True)

# Get a dataset that returns images and real 3d data. if include ply, additionally ply files are returned
def get_combined_image_point_dataset(include_ply = False):
    trainimg, valimg, testimg = get_default_image_dataset()
    if include_ply:
        train3d, val3d, test3d = get_combined_point_real_arti_dataset()
    else:    
        train3d, val3d, test3d = get_real_dataset3d()

    train_dataset = datasets.Image3dCombidataset(trainimg, train3d)
    val_dataset = datasets.Image3dCombidataset(valimg, val3d)
    test_dataset = datasets.Image3dCombidataset(testimg, test3d)

    return train_dataset, val_dataset, test_dataset

def get_auto_split_dataset():
    base = Path(r"F:\Boxes-ds\auto_split")
    split = Path(r"F:\Boxes-ds\auto_split\split_without2024")
    cache_folder = Path(r"experiments_volume_models\local_stuff\auto_split_img_cache")

    return _get_image_dataset(split, base, cache_folder, 1)