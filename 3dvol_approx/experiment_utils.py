import datasets_3d
from pathlib import Path

def get_real_dataset():
    split_folder = Path("split_without2024")
    base_folder = Path(r"F:\Boxes-ds\segmented_distance_depth")
    train_dataset = datasets_3d.DepthMapDataset(base_folder, base_folder / split_folder / "mapping_train.json")
    train_dataset.create_or_load_cache(Path(r"3dvol_approx\local_stuff\dmap_cache\dmap_cache_train.pth"), force_recompute=False)
    val_dataset = datasets_3d.DepthMapDataset(base_folder, base_folder / split_folder / "mapping_val.json")
    val_dataset.create_or_load_cache(Path(r"3dvol_approx\local_stuff\dmap_cache\dmap_cache_val.pth"), force_recompute=False)
    return train_dataset, val_dataset

def get_ply_dataset():
    split_folder = Path(r"F:\Boxes-ds\segmented_distance_depth\split_without2024")
    base_folder = Path(r"F:\FIP-data\wheat-scans")
    train_dataset = datasets_3d.PlyDataset(base_folder, split_folder / "mapping_train.json", augment=False)
    train_dataset.create_or_load_cache(Path(r"3dvol_approx\local_stuff\ply_cache\ply_train.pth"), force_recompute=False)
    val_dataset = datasets_3d.PlyDataset(base_folder, split_folder / "mapping_val.json")
    val_dataset.create_or_load_cache(Path(r"3dvol_approx\local_stuff\ply_cache\ply_val.pth"), force_recompute=False)
    return train_dataset, val_dataset

def get_combined_dataset():
    r_tds, r_vds = get_real_dataset()
    p_tds, p_vds = get_ply_dataset()
    train_dataset = datasets_3d.Combined3dDataset(r_tds, p_tds)
    val_dataset = datasets_3d.Combined3dDataset(r_vds, p_vds)
    return train_dataset, val_dataset