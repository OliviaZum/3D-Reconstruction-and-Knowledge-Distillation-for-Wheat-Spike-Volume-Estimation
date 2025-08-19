"""
Contains mostly methods to quickly get certain types of dataset instances.
If running experiments, file paths have to be adapted to point to the actual dataset.
"""

from typing import Literal
from pathlib import Path
import torch
from volume_prediction_fip.experiments_volume_models.shared import datasets
from volume_prediction_fip.utils import helpers

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

def _get_ply_dataset(split_folder, base_folder, cache_folder, force_recompute = False, voxelize = False):
    split_folder, base_folder, cache_folder = [Path(s) for s in [split_folder, base_folder, cache_folder]]
    cache_folder.mkdir(exist_ok=True)
    train_dataset = datasets.PlyDataset(base_folder, split_folder / "mapping_train.json", voxelize=voxelize)
    train_dataset.create_or_load_cache(cache_folder / "ply_train.pth", force_recompute=force_recompute)
    val_dataset = datasets.PlyDataset(base_folder, split_folder / "mapping_val.json", voxelize=voxelize)
    val_dataset.create_or_load_cache(cache_folder / "ply_val.pth", force_recompute=force_recompute)
    test_dataset = datasets.PlyDataset(base_folder, split_folder / "mapping_test.json", voxelize=voxelize)
    test_dataset.create_or_load_cache(cache_folder / "ply_test.pth", force_recompute=force_recompute)

    return train_dataset, val_dataset, test_dataset

# Get a dataset of real 3d reconstructed spikes
def get_real_dataset3d(force_recompute = False):
    split_folder = Path("split_without2024")
    base_folder = Path("/projects/zumstego/volume_prediction_fip/Boxes-ds/segmented_distance_depth")
    split_folder = base_folder / split_folder
    cache = Path(helpers.get_exp_path() / "dmap_real_default")

    return _get_depthmap_dataset(split_folder, base_folder, cache, force_recompute)

# Get artificial point clouds dataset
def get_ply_dataset(force_recompute = False, voxelize = False):
    split_folder = Path("/projects/zumstego/volume_prediction_fip/Boxes-ds/segmented_distance_depth/split_without2024")
    base_folder = Path("/projects/zumstego/volume_prediction_fip/Fip-data/wheat-scans")

    
    if voxelize:
        cache_path = Path(helpers.get_exp_path() / r"ply_cache_voxelized")
    else:
        cache_path = Path(helpers.get_exp_path() / r"ply_cache")

    return _get_ply_dataset(split_folder, base_folder, cache_path, force_recompute, voxelize)
    
# Returns real and corresponding artifical pointcloud at the same time
def get_combined_point_real_arti_dataset():
    r_train_ds, r_val_ds, r_test_ds = get_real_dataset3d()
    a_train_ds, a_val_ds, a_test_ds = get_ply_dataset(voxelize=True)
    train_dataset = datasets.Combined3dDataset(r_train_ds, a_train_ds)
    val_dataset = datasets.Combined3dDataset(r_val_ds, a_val_ds)
    test_dataset = datasets.Combined3dDataset(r_test_ds, a_test_ds)
    return train_dataset, val_dataset, test_dataset

# Unlabeled image and 3d dataset
def get_unlabeled_dataset(force_recompute = False):
    split_folder = Path("/projects/zumstego/volume_prediction_fip/Boxes-ds/unlabeled-5000-depth/split")
    base_folder = Path("/projects/zumstego/volume_prediction_fip/Boxes-ds/unlabeled-5000-depth")
    cache_folder = Path(helpers.get_exp_path() / "cache_unlabeled")

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
    split_folder = Path("/projects/zumstego/volume_prediction_fip/Boxes-ds/segmented_distance_depth/split_without2024")
    base_folder = Path("/projects/zumstego/volume_prediction_fip/Boxes-ds/segmented_distance_depth")
    cache_folder_image = Path(helpers.get_exp_path() / "default_image_cache")

    return _get_image_dataset(split_folder, base_folder, cache_folder_image)

def get_default_image_dataset_wo_distance():
    split_folder = Path(r"F:\Boxes-ds\segmented_distance_depth\split_without2024")
    base_folder = Path(r"F:\Boxes-ds\segmented_distance_depth")
    cache_folder = Path(helpers.get_exp_path() / "nodistancenorm_cache")

    return _get_image_dataset(split_folder, base_folder, cache_folder, enable_distance=False)

def get_default_image_dataset_wo_distance_and_seg(disable_augmentation = False):
    split_folder = Path(r"F:\Boxes-ds\spike_dataset_manual_20_pad\split_without2024")
    base_folder = Path(r"F:\Boxes-ds\spike_dataset_manual_20_pad")
    if disable_augmentation:
        cache_folder = Path(helpers.get_exp_path() / "nodistancenosegnoaug_cache")
        ndupli = 1
    else:
        cache_folder = Path(helpers.get_exp_path() / "nodistancenoseg_cache")
        ndupli = 10

    return _get_image_dataset(split_folder, base_folder, cache_folder, enable_distance=False, ndupli_train=ndupli)

def get_artifical_image_dataset_sideviews():
    split = r"F:\Boxes-ds\artifical\bestpose_noshift_12\split_without2024_noextend"
    base = r"F:\Boxes-ds\artifical\bestpose_noshift_12"
    cache = helpers.get_exp_path() / "artificial_sideviews"
    return _get_image_dataset(split, base, cache, enable_distance=False)

def get_artifical_image_dataset_sideviews6():
    split = r"F:\Boxes-ds\artifical\bestpose_noshift_12\split_without2024_noextend_6img"
    base = r"F:\Boxes-ds\artifical\bestpose_noshift_12"
    cache = helpers.get_exp_path() / "artificial_sideviews6"
    return _get_image_dataset(split, base, cache, enable_distance=False)

def get_artificial_image_dataset_randompose():
    split = r"F:\Boxes-ds\artifical\randompose_noshift_12\split_without2024_noextend"
    base = r"F:\Boxes-ds\artifical\randompose_noshift_12"
    cache = helpers.get_exp_path() / "artificial_randomviews"
    return _get_image_dataset(split, base, cache, enable_distance=False)

def get_artificial_dataset_fiplike_uniform(type: Literal["depth", "image"], force_recompute = False):
    split = r"F:\Boxes-ds\artifical\artifical_fippose_randomrot\split_without2024_noextend"
    base = r"F:\Boxes-ds\artifical\artifical_fippose_randomrot"
    
    if type == "image":
        cache = helpers.get_exp_path() / "artificial_fiplike_uniform_image"
        return _get_image_dataset(split, base, cache, enable_distance=False)
    else:
        cache = helpers.get_exp_path() / "artificial_fiplike_uniform_depth"
        return _get_depthmap_dataset(split, base, cache, force_recompute, True)

def get_artificial_dataset_fiplike_normal(type: Literal["depth", "image"], force_recompute = False):
    split = r"F:\Boxes-ds\artifical\artificial_fippose_toprot\split_without2024_noextend"
    base = r"F:\Boxes-ds\artifical\artificial_fippose_toprot"

    if type == "image":
        cache = helpers.get_exp_path() / "artificial_fiplike_normal_image"
        return _get_image_dataset(split, base, cache, enable_distance=False)
    else:
        cache = helpers.get_exp_path() / "artificial_fiplike_normal_depth"
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
    #base = Path(r"F:\Boxes-ds\auto_split")
    #split = Path(r"F:\Boxes-ds\auto_split\split_without2024")
    base = Path("/projects/zumstego/volume_prediction_fip/Boxes-ds/auto_split")
    split = Path("/projects/zumstego/volume_prediction_fip/Boxes-ds/auto_split/split_without2024")
    cache_folder = Path(helpers.get_exp_path() / "auto_split_img_cache")

    return _get_image_dataset(split, base, cache_folder, 1)

def get_auto_split_dataset_second():
    base = Path("/projects/zumstego/volume_prediction_fip/Boxes-ds/auto_split_new_pair")
    split = Path("/projects/zumstego/volume_prediction_fip/Boxes-ds/auto_split_new_pair/split_without2024")
    #base = Path(r"F:\Boxes-ds\auto_split_new_pair")
    #split = Path(r"F:\Boxes-ds\auto_split_new_pair\split_without2024")
    cache_folder = Path(helpers.get_exp_path() / "auto_split_new_pair_cache")

    return _get_image_dataset(split, base, cache_folder, 1)