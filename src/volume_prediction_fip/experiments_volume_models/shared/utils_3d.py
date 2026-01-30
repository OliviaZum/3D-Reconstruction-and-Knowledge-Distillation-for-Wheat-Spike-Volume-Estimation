import torch
import trimesh
import pyrender
import numpy as np

def visualize_point_clouds(points_red, points_green = None):
    scene = pyrender.Scene()

    if points_red is not None:
        red_cloud = trimesh.points.PointCloud(points_red, colors=[255, 0, 0])
        red_mesh = pyrender.Mesh.from_points(points_red, colors=red_cloud.colors)
        scene.add(red_mesh)

    if points_green is not None:
        green_cloud = trimesh.points.PointCloud(points_green, colors=[0, 255, 0])
        blue_mesh = pyrender.Mesh.from_points(points_green, colors=green_cloud.colors)
        scene.add(blue_mesh)

    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=7)

def visualize_point_cloud_single(points: np.ndarray, weights: np.ndarray):
    scene = pyrender.Scene()
    arr_min, arr_max = weights.min(), weights.max()
    rcol = (weights - arr_min) / (arr_max - arr_min)
    col = np.zeros((weights.shape[0], 3))
    col[:, 0] = rcol

    red_cloud = trimesh.points.PointCloud(points, colors=col)
    red_mesh = pyrender.Mesh.from_points(points, colors=red_cloud.colors, )
    scene.add(red_mesh)
    pyrender.Viewer(scene, use_raymond_lighting=True, point_size=7)


def chamfer_dist_slow(x1, x2):
    sub = x1.unsqueeze(1) - x2.unsqueeze(2)
    dist = (sub ** 2).sum(dim=3)
    md1, _ = torch.min(dist, dim=2)
    md2, _ = torch.min(dist, dim=1)
    cd = md1.sqrt().mean(dim=1) + md2.sqrt().mean(dim=1)
    return cd.mean()

def chamfer_dist_slow_normal(pred, target, outlier_cutoff = 0, normal_loss = True):
    p1, normalpred = pred[:, :, 0:3], pred[:, :, 3:6]
    p2, normaltarget = target[:, :, 0:3], target[:, :, 3:6]

    sub = p1.unsqueeze(1) - p2.unsqueeze(2)
    dist = (sub ** 2).sum(dim=3)
    md1, _ = torch.min(dist, dim=2)
    md2, sel_idx = torch.min(dist, dim=1)
    if outlier_cutoff > 0:
        m1 = md1 > 1
        m2 = md2 > 1
        md1[m1] = md1[m1] * 0.001 + outlier_cutoff
        md2[m2] = md2[m2] * 0.001 + outlier_cutoff
    cd = md1.sqrt().mean(dim=1) + md2.sqrt().mean(dim=1)

    if normal_loss:
        batch_idx = torch.repeat_interleave(torch.arange(0, len(sel_idx)), sel_idx.shape[1])
        sel_idx = sel_idx.flatten()
        nt = normaltarget[batch_idx, sel_idx, :].reshape((target.shape[0], -1, 3))
        normallength = (normalpred ** 2).sum(dim=2).sqrt()
        targetlength = (nt ** 2).sum(dim=2).sqrt()
        nldir = (nt * normalpred).sum(dim=2) / (normallength * targetlength)
        normalloss = - nldir.mean(dim=1) + ((normallength - 1) ** 2).mean(dim=1)
        loss = cd.mean() + normalloss.mean() * 0.2
    else:
        loss = cd.mean()
    
    return loss

def save_ply(data, output_path="point_cloud.ply"):
    positions = data[:, :3]
    normals = data[:, 3:]
    point_cloud_trimesh = trimesh.Trimesh(vertices=positions, faces=np.zeros((0, 3)), vertex_normals=normals)
    point_cloud_trimesh.export(output_path)

def measure_volume(data, save_path = None, save_name = None):
    import pymeshlab
    positions = data[:, :3]
    normals = data[:, 3:]
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=positions, v_normals_matrix=normals))
    #ms.save_current_mesh(f'3dvol_approx/tests/mesh{tapi}_points.ply')
    ms.generate_surface_reconstruction_screened_poisson(
        depth=8
    )
    volume = ms.get_geometric_measures()["mesh_volume"]
    if save_path:
        ms.save_current_mesh(str(save_path / f"{save_name}_{volume}.ply"))
    return volume

def distance_sample(points: torch.Tensor, num_samples = 200, generator = None) -> torch.Tensor:
    if num_samples is not None:
        batch, n, d = points.shape
        sampled_indices = torch.stack([torch.randperm(n, device=points.device, generator=generator)[:num_samples] for _ in range(batch)])
        sampled_points = torch.gather(points, 1, sampled_indices.unsqueeze(-1).expand(-1, -1, d))
    else:
        sampled_points = points
    
    distances = torch.cdist(sampled_points, points, p=2)  # (b, num_samples, n)
    return distances

def batched_torch_hist(x: torch.Tensor, weigths, start: float, end: float, bins: int):
    bins = bins - 1 # 9 bins
    bin_width = (end - start) / bins
    end = end - start # 60
    x = x - start
    x = (x / bin_width).floor().to(torch.int32) #nxn, bins index of all points j to point i 
    binlist = torch.arange(bins, device=x.device)
    mask_eq = x.unsqueeze(-1) == binlist #add another dimension, one-hot endoding, true for bin index, nxnxbins
    mask_gt = (x >= bins).unsqueeze(-1) #distances outside of 9 bins, nxnx1 
    mask_eq = torch.concat((mask_eq, mask_gt), dim=-1) #merge bools of first 9 bins with bool of overflow bin, across last dimension
    mask_eq = torch.as_tensor(mask_eq, device=weigths.device)
    mask_eq = mask_eq * weigths.reshape((weigths.shape[0], 1, weigths.shape[1], 1)) #batch dim, 1, n_points, 1, broadcast over bins, zero out all columns j that were padded
    counts = mask_eq.sum(dim=-2) #how many neighbours fall into a bin of point i
    return counts

def to_rigid_invariant_representation(x: torch.Tensor, weights: torch.Tensor, num_samples = None, bins = 10, generator = None):
    with torch.no_grad():
        w = distance_sample(x, num_samples, generator) #nxn, distances
        u = batched_torch_hist(w, weights, 0, 60, bins).to(x.dtype)
        ns = u.sum(dim=2, keepdim=True)
        ns[ns == 0] = 1
        u = u / ns

        return u

def compare_rigid_invariant_mse(hist_in, hist_target, average = True):
    sub = torch.square(hist_in.unsqueeze(-3) - hist_target.unsqueeze(-2) )
    wss = sub.sum(dim=-1) # (b, n, n) # Correct dim?
    min1, _ = wss.min(dim=-1) # (b, n)
    min2, _ = wss.min(dim=-2)
    if average:
        return torch.mean(min1) + torch.mean(min2)
    else:
        return torch.mean(min1, dim=1) + torch.mean(min2, dim=1)
    
def mean_where(x: torch.Tensor, mask: torch.Tensor, dim=0):
    masked_mean = torch.sum(x * mask, dim=dim) / mask.sum(dim=dim)
    return masked_mean