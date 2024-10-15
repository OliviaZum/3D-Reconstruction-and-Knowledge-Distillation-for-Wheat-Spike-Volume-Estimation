import sys
sys.path.append("C:/Users/Admin/Desktop/master_thesis/volume_prediction_fip")

from typing import Dict, List
import numpy as np
import epipolar_geometry
import torch
import warnings

def build_epipolar_graph_opt(poses_conf, bounding_boxes: Dict[str, Dict[str, List[float]]], num_samples=20,
                             device = 'cuda' if torch.cuda.is_available() else 'cpu', normalize_graph = True) -> torch.Tensor:

    # Precompute and build mapping from node idx to instance
    idx_to_instance = {}
    instance_to_idx = {}
    num_instance_per_image = []
    img_name_to_start_end = {}

    # Lines of boxes for a single image (3 * #boxes, 3)
    box_lines_img = {}
    # Zero points of connecting lines for a single image
    v_zero_img = {}
    # Connecting vectors of boxes one image (3 * #boxes, 3)
    v_conn_img = {} 
    # Sampled points for each bounding box one image  (num_sample * #boxes, 3)
    sample_points_img = {}

    with torch.no_grad():
        i = 0
        for (n, v) in bounding_boxes.items():
            num_instance_per_image.append(len(v))
            img_name_to_start_end[n] = (i, i + len(v))
            box_lines_img[n] = []
            v_conn_img[n] = []
            v_zero_img[n] = []
            sample_points_img[n] = []
            for id, box in v.items():
                xmin, ymin, xmax, ymax = box
                p = torch.tensor([
                    [xmin, ymin, 1],
                    [xmin, ymax, 1],
                    [xmax, ymax, 1],
                    [xmax, ymin, 1]
                    ])
                pa = torch.stack([p[0, :], p[0, :], p[1, :]])
                pb = torch.stack([p[1, :], p[3, :], p[2, :]])
                v_zero_img[n].append(pa)
                bl = torch.cross(pa, pb, dim=1)
                box_lines_img[n].append(bl)

                bc = pb - pa
                v_conn_img[n].append(bc)

                vert = (p[1, :] - p[0, :]).unsqueeze(0)
                hor = (p[3, :] - p[0, :]).unsqueeze(0)
                mulp = torch.rand((num_samples, 2))
                samples = mulp[:, 0].unsqueeze(1) * hor + mulp[:, 1].unsqueeze(1) * vert + p[0, :].unsqueeze(0)
                sample_points_img[n].append(samples)

                idx_to_instance[i] = (n, id)
                instance_to_idx[(n, id)] = i
                i += 1

            box_lines_img[n] = torch.concat(box_lines_img[n], dim=0).to(device)
            v_conn_img[n] = torch.concat(v_conn_img[n], dim=0).to(device)
            v_zero_img[n] = torch.concat(v_zero_img[n], dim=0).to(device)
            sample_points_img[n] = torch.concat(sample_points_img[n], dim=0).to(device)
        
        cam_names = list(bounding_boxes.keys())
        graph = torch.zeros([len(idx_to_instance)] * 2, device=device)

        for i in range(len(cam_names)):
            for j in range(len(cam_names)):
                if i == j:
                    continue
                im1 = cam_names[i]
                im2 = cam_names[j]
                F = torch.tensor(epipolar_geometry.build_fundamental(poses_conf, im1, im2), dtype=torch.float32, device=device)

                epi_lines = (F @ sample_points_img[im1].T).T
                intersect = torch.cross(epi_lines.unsqueeze(0), box_lines_img[im2].unsqueeze(1), dim=2) # (n_boxline, n_epi_line, 3)
                # Can result in nans, but handled later together with lambda (l) computation
                intersect = intersect / intersect[:, :, 2].view((intersect.shape[0], intersect.shape[1], 1))
                zeroed_intersect = intersect - v_zero_img[im2].unsqueeze(1)
                l = zeroed_intersect[:, :, :2] / v_conn_img[im2][:, :2].unsqueeze(1)
                mask = torch.logical_and(torch.logical_and(torch.logical_not(torch.isnan(l)), l > 0), l < 1)
                l_reduced = mask.sum(dim=2) > 0 # (#n_boxline, n_epi_line)
                # At this point_lreduced = true at i, j if the i'th boxline intersects with the j epiline

                box_intersects = torch.logical_or(torch.logical_or(l_reduced[::3, :], l_reduced[1::3, :]), l_reduced[2::3, :])
                counts = torch.zeros((box_intersects.shape[0], box_intersects.shape[1] // num_samples), device=device) # (#n_boxes_im2, #n_boxes_im1)
                for k in range(num_samples):
                    counts += box_intersects[:, k::num_samples]

                pos_1, end_1 = img_name_to_start_end[im1]
                pos_2, end_2 = img_name_to_start_end[im2]
                graph[pos_1:end_1, pos_2:end_2] += counts.T
                graph[pos_2:end_2, pos_1:end_1] += counts

        if normalize_graph:
            graph *= (1 / (2 * num_samples))
            graph[graph == 0] = -5
            graph[torch.logical_and(graph >= 0, graph <= 0.8)] = 0
            graph.fill_diagonal_(0)

        return graph, idx_to_instance

def lp_cluster_torch(graph: np.ndarray, device = 'cuda' if torch.cuda.is_available() else 'cpu') -> torch.Tensor:
    with torch.no_grad():
        
        # Essentially reimplemented from networkx (with some degree of parallelism)
        # With an entirely parallel implementation oscilations occur which lead to the 
        # graph slowly or not converging at all (and worse results). As a
        # tradeoff using just some degree of parallelism.
        graph = torch.tensor(graph, device=device)
        labels = torch.eye(graph.shape[0], device=device)
        no_converge = True
        max_iter = 200
        iters = 0
        batches = graph.shape[0] // 20
        if batches < 1:
            num_parallel = 1
        else:
            num_parallel = batches
        while no_converge and iters <= max_iter:
            iters += 1
            nodes = torch.randperm(graph.shape[0])
            no_converge = False
            for i in range(0, len(nodes), num_parallel):
                n = nodes[i:min(i+num_parallel, len(nodes))]
                v = graph[n]
                freq = v @ labels
                # Technically labels should be picked at random. 
                # But complicated and slow, practically the algorithm does not seem to work
                # worse with a fixed label.
                max = torch.argmax(freq, dim=1)
                if torch.any(labels[n, max] == 0):
                    no_converge = True
                    labels[n, :] = 0
                    labels[n, max] = 1

        if iters == max_iter:
            warnings.warn("label propagation failed to converge")

        labels = torch.argmax(labels, dim=1).cpu()
        return labels