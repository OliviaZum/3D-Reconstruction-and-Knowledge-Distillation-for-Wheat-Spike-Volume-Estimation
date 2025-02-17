import datasets_3d
from pathlib import Path
import torch

def get_real_dataset3d(force_recompute = False):
    split_folder = Path("split_without2024")
    base_folder = Path(r"F:\Boxes-ds\segmented_distance_depth")
    train_dataset = datasets_3d.DepthMapDataset(base_folder, base_folder / split_folder / "mapping_train.json")
    train_dataset.create_or_load_cache(Path(r"3dvol_approx\local_stuff\dmap_cache\dmap_cache_train.pth"), force_recompute=force_recompute)
    val_dataset = datasets_3d.DepthMapDataset(base_folder, base_folder / split_folder / "mapping_val.json")
    val_dataset.create_or_load_cache(Path(r"3dvol_approx\local_stuff\dmap_cache\dmap_cache_val.pth"), force_recompute=force_recompute)
    test_dataset = datasets_3d.DepthMapDataset(base_folder, base_folder / split_folder / "mapping_test.json")
    test_dataset.create_or_load_cache(Path(r"3dvol_approx\local_stuff\dmap_cache\dmap_cache_test.pth"), force_recompute=force_recompute)

    return train_dataset, val_dataset, test_dataset

def get_ply_dataset(force_recompute = False):
    split_folder = Path(r"F:\Boxes-ds\segmented_distance_depth\split_without2024")
    base_folder = Path(r"F:\FIP-data\wheat-scans")
    train_dataset = datasets_3d.PlyDataset(base_folder, split_folder / "mapping_train.json")
    train_dataset.create_or_load_cache(Path(r"3dvol_approx\local_stuff\ply_cache\ply_train.pth"), force_recompute=force_recompute)
    val_dataset = datasets_3d.PlyDataset(base_folder, split_folder / "mapping_val.json")
    val_dataset.create_or_load_cache(Path(r"3dvol_approx\local_stuff\ply_cache\ply_val.pth"), force_recompute=force_recompute)
    return train_dataset, val_dataset

def get_combined_point_real_arti_dataset():
    r_tds, r_vds = get_real_dataset3d()
    p_tds, p_vds = get_ply_dataset()
    train_dataset = datasets_3d.Combined3dDataset(r_tds, p_tds)
    val_dataset = datasets_3d.Combined3dDataset(r_vds, p_vds)
    return train_dataset, val_dataset

def get_unlabeled_dataset(force_recompute = False):
    split_folder = Path(r"F:\Boxes-ds\unlabeled-5000-depth\split")
    base_folder = Path(r"F:\Boxes-ds\unlabeled-5000-depth")
    cache_folder = Path(r"3dvol_approx\local_stuff\dmap_cache")
    train_dataset = datasets_3d.DepthMapDataset(base_folder, split_folder / "mapping_train.json")
    train_dataset.create_or_load_cache(cache_folder / "dmap_cache_unlabeled_train.pth", force_recompute=force_recompute)
    val_dataset = datasets_3d.DepthMapDataset(base_folder, split_folder / "mapping_test.json")
    val_dataset.create_or_load_cache(cache_folder / "dmap_cache_unlabeled_val.pth", force_recompute=force_recompute)
    return train_dataset, val_dataset

def get_image_dataset():
    split_folder = Path(r"F:\Boxes-ds\segmented_distance_depth\split_without2024")
    base_folder = Path(r"F:\Boxes-ds\segmented_distance_depth")
    cache_folder_image = Path(r"F:\Boxes-ds\_cache_dataset_embeddings")

    pretrained_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    train_dataset_image = datasets_3d.MultiImageTrainDataset(base_folder, split_folder / "mapping_train.json", datasets_3d.get_transform(True), "volume", 12, 4, True)
    train_dataset_image.create_or_load_cache(pretrained_model, cache_folder_image / "traincache.pth", 10, num_workers=5)
    val_dataset_image = datasets_3d.MultiImageTrainDataset(base_folder, split_folder / "mapping_val.json", datasets_3d.get_transform(False), "volume", 12, 6, False, validation_mode=True)
    val_dataset_image.create_or_load_cache(pretrained_model, cache_folder_image / "valcache.pth", 1, num_workers=0)
    test_dataset_image = datasets_3d.MultiImageTrainDataset(base_folder, split_folder / "mapping_test.json", datasets_3d.get_transform(False), "volume", 12, 6, False, validation_mode=True)
    test_dataset_image.create_or_load_cache(pretrained_model, cache_folder_image / "testcache.pth", 1, num_workers=0)

    return train_dataset_image, val_dataset_image, test_dataset_image

def get_combined_image_point_dataset():
    trainimg, valimg, testimg = get_image_dataset()
    train3d, val3d, test3d = get_real_dataset3d()

    train_dataset = datasets_3d.Image3dCombidataset(trainimg, train3d)
    val_dataset = datasets_3d.Image3dCombidataset(valimg, val3d)
    test_dataset = datasets_3d.Image3dCombidataset(testimg, test3d)

    return train_dataset, val_dataset, test_dataset
