import experiments.shared.datasets_3d as datasets_3d
from pathlib import Path
import experiments.shared.utils_3d as utils_3d
import open3d as o3d
import numpy as np

def align_point_clouds(source_points, target_points, use_normals=False):
    # Convert numpy arrays to Open3D point clouds
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()
    source_points = np.asarray(source_points, dtype=np.float64)  # Ensure correct dtype
    target_points = np.asarray(target_points, dtype=np.float64)
    source.points = o3d.utility.Vector3dVector(source_points)
    target.points = o3d.utility.Vector3dVector(target_points)

    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # 1. Global Registration (Coarse alignment)
    source_down, target_down, source_fpfh, target_fpfh = preprocess_for_global_registration(source, target)
    global_transformation = global_registration(source_down, target_down, source_fpfh, target_fpfh)

    return global_transformation

    # 2. Local Refinement (ICP)
    refined_transformation = local_icp_refinement(source, target, np.eye(4))

    return refined_transformation

def preprocess_for_global_registration(source, target, voxel_size=0.5):
    """ Downsamples point clouds and computes FPFH features. """
    #source_down = source.voxel_down_sample(voxel_size)
    #target_down = target.voxel_down_sample(voxel_size)

    radius_feature = voxel_size * 5
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    return source, target, source_fpfh, target_fpfh

def global_registration(source_down, target_down, source_fpfh, target_fpfh):
    """ Performs RANSAC-based global registration. """
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=1.0,  # Adjust for more relaxed matching
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 100)
    )
    
    return result.transformation

def local_icp_refinement(source, target, init_transformation, threshold=2):
    """ Refines alignment using ICP. """
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, threshold, init_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    
    return result_icp.transformation

if __name__ == "__main__":
    base_folder = Path(r"F:\Boxes-ds\segmented_distance_depth")
    real_ds = datasets_3d.DepthMapDataset(base_folder, base_folder / "split_without2024" / "mapping_all_val.json", augment=False)
    real_ds.create_or_load_cache(Path(r"3dvol_approx\local_stuff\dmap_cache\dmap_cache_all_val.pth"), force_recompute=False)
    arti_ds = datasets_3d.PlyDataset(Path(r"3dvol_approx\local_stuff\ply_split") / "train_ply.csv", 
                                    Path(r"3dvol_approx\local_stuff\ply_cache\ply_train_cache.pth"), True, augment=False)
    

    real_index = 0
    pid = real_ds.plant_mapping.loc[real_index, "plant_id"]
    arti_index = arti_ds.df.index[arti_ds.df["plant_id"] == pid].to_list()[0]

    _, arti_data, _ = arti_ds.__getitem__(arti_index)
    _, real_data, _ = real_ds.__getitem__(real_index)

    #utils_3d.visualize_point_clouds(arti_data[:, 0:3], real_data[:, 0:3])

    transformation = align_point_clouds(arti_data[:, 0:3], real_data[:, 0:3])

    arti_data_align = (arti_data[:, 0:3] @ transformation[:3, :3].T) + transformation[:3, 3]

    utils_3d.visualize_point_clouds(arti_data_align, real_data[:, 0:3])
