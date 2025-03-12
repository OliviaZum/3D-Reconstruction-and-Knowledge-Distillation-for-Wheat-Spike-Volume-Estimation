from typing import Dict, List, Tuple
import numpy as np
import torch
import warnings
from utils import calibration_helpers

def build_epipolar_graph_opt(poses_conf, bounding_boxes: Dict[str, Dict[str, List[int]]], num_samples=20,
                             device = 'cuda' if torch.cuda.is_available() else 'cpu', normalize_graph = True) -> Tuple[torch.Tensor, Dict[int, Tuple[str, str]]]:
    """
    Given a set of bounding boxes on multiple cameras, builds a weighted undirected graph, where each bounding box corresponds to one node
    and the edge weight between two nodes is the number of epipolar lines intersecting the bounding boxes. (In practice: Choose random samples in a
    bounding box, check how many time the epipolar lines defined by it intersect another bounding box. It goes both ways.)
    There is a normalization for clustering applied if normalize_graph is true. No hits at all receive negative weight, weights are scaled to
    lie in the 0-1 range. Weak links (below 0.8) are set to 0. Additionally nodes of the same image receive a larger negative weight.
    """

    # Precompute and build mapping from node idx to instance
    idx_to_instance = {}
    instance_to_idx = {}
    num_instance_per_image = []
    img_name_to_start_end = {}

    # Lines of boxes for a single image (3 * #boxes, 3)
    box_lines_img = {}
    # Zero points of box lines for a single image [A corner of the box on the line] (3 * #boxes, 3) 
    v_zero_img = {}
    # Connecting vectors of boxes one image [The vector in direction of the other corner of the box from the zero point] (3 * #boxes, 3)
    v_conn_img = {} 
    # Sampled points for each bounding box on one image  (num_sample * #boxes, 3)
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
                    ], dtype=torch.float32)
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

            if len(box_lines_img[n]) > 0:
                box_lines_img[n] = torch.concat(box_lines_img[n], dim=0).to(device)
                v_conn_img[n] = torch.concat(v_conn_img[n], dim=0).to(device)
                v_zero_img[n] = torch.concat(v_zero_img[n], dim=0).to(device)
                sample_points_img[n] = torch.concat(sample_points_img[n], dim=0).to(device)
            else:
                box_lines_img[n] = torch.zeros((0, 3)).to(device)
                v_conn_img[n] = torch.zeros((0, 3)).to(device)
                v_zero_img[n] = torch.zeros((0, 3)).to(device)
                sample_points_img[n] = torch.zeros((0, 3)).to(device)
        
        cam_names = list(bounding_boxes.keys())
        graph = torch.zeros([len(idx_to_instance)] * 2, device='cpu')

        for i in range(len(cam_names)):
            for j in range(len(cam_names)):
                if i == j:
                    continue
                im1 = cam_names[i]
                im2 = cam_names[j]
                F = torch.tensor(calibration_helpers.build_fundamental(poses_conf, im1, im2), dtype=torch.float32, device=device)

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
                counts = counts.cpu()
                graph[pos_1:end_1, pos_2:end_2] += counts.T
                graph[pos_2:end_2, pos_1:end_1] += counts
                

        if normalize_graph:
            graph *= (1 / (2 * num_samples))
            graph[graph == 0] = -2
            start = 0
            for w in num_instance_per_image:
                end = start + w
                graph[start:end, start:end] = -5
                start = end
            graph[torch.logical_and(graph >= 0, graph <= 0.8)] = 0
            graph.fill_diagonal_(0)

        return graph, idx_to_instance
    
def estimate_distances(poses_conf, bounding_boxes: Dict[str, Dict[str, List[int]]], labels: torch.Tensor,
                       idx_to_instance: Dict[int, Tuple[str, str]], min_view: int = 6,
                       num_samples: int = 100, num_view: int = 3) -> Dict[str, Dict[str, float | None]]:
    """
    Estimate the position of each cluster (if correct: A single spike) by choosing random samples on a subset of the bounding 
    boxes of a cluster then triangulate. In practice this is repeated a few times and the mean position is taken as the position
    of the spike. Then return the distance of the spike on each image.
    """

    # This implementation is quite inefficient. But anyhow since this is O(n) in contrast to the other two functions here it is "fast enough" in practice.
    box_id_to_box = {}
    for (n, v) in bounding_boxes.items():
        for id, box in v.items():
            box_id_to_box[(n, id)] = box
    cluster_to_boxes = {}
    for i, (cam_name, box_id) in idx_to_instance.items():
        cluster = labels[i]
        cluster_to_boxes.setdefault(cluster, [])
        box = box_id_to_box[(cam_name, box_id)]
        cluster_to_boxes[cluster].append((cam_name, box_id, box))

    Ps = {cam_name: calibration_helpers.get_projection(poses_conf, cam_name) for cam_name in bounding_boxes.keys()}

    assert num_view <= min_view, f"Used number of views for triangulation ({num_view}) has to be smaller that min_view"
    bounding_boxes_to_distance: Dict[str, Dict[str, float | None]] = {}
    def set_cluster_unknown(boxes):
        for cam_name, _, _ in boxes:
            bounding_boxes_to_distance.setdefault(cam_name, {})
            bounding_boxes_to_distance[cam_name][cluster] = None

    for cluster, boxes in cluster_to_boxes.items():
        if len(boxes) < min_view:
            set_cluster_unknown(boxes)
            continue

        estimates = []
        weights = []
        for k in range(num_samples):
            subset = np.random.choice(len(boxes), num_view, False)
            A = np.zeros((2*num_view, 4))

            for i, idx in enumerate(subset):
                cam_name, _, box = boxes[idx]
                box = np.array(box)
                sample = box[0:2] + np.array([np.random.rand() * (box[2] - box[0]), np.random.rand() * (box[3] - box[1])])
                P = Ps[cam_name]
                # https://temugeb.github.io/computer_vision/2021/02/06/direct-linear-transorms.html
                A[2*i, :] = sample[1] * P[2, :] - P[1, :]
                A[2*i+1, :] = P[0, :] - sample[0] * P[2, :]
            _, s, Vt = np.linalg.svd(A, full_matrices=False)
            e = Vt[-1]
            hits = 0
            for cam_name, _, (x1, y1, x2, y2) in boxes:
                P = Ps[cam_name]
                t = P @ e
                t /= t[2]
                if t[0] >= x1 and t[1] >= y1 and t[0] <= x2 and t[1] <= y2:
                    hits += 1

            estimates.append(e)
            weights.append(hits / len(boxes))
        
        estimates = np.stack(estimates, axis=0)
        weights = np.array(weights)
        p70 = np.percentile(weights, 70)
        mask = np.logical_not(np.logical_or(estimates[:, 3] < 0.00001, weights < p70))
        estimates = estimates / np.expand_dims(estimates[:, 3], 1)
        count = weights[mask].sum()
        if count == 0:
            set_cluster_unknown(boxes)
            continue
        estimate = (estimates[mask, :] * np.expand_dims(weights[mask], axis=1)).sum(axis=0) / count

        for cam_name, _, _ in boxes:
            cam_pos = np.array(poses_conf[cam_name]["extrinsics"]["center"])
            distance = np.sqrt(np.sum((estimate[0:3] - cam_pos) ** 2))
            bounding_boxes_to_distance.setdefault(cam_name, {})
            bounding_boxes_to_distance[cam_name][cluster] = distance

    return bounding_boxes_to_distance

def lp_cluster_torch(graph: np.ndarray, device = 'cuda' if torch.cuda.is_available() else 'cpu',
                     n_repeats = 3, min_prob = 0.8) -> torch.Tensor:
    """
    Label propagation to cluster the graph created by build_epipolar_graph_opt.
    Label propagation is run n_repeats times, building a "cluster" matrix containing the probabilities
        that a node is paired with another node. min_prob indicates the minimum probability to pair
    """
    with torch.no_grad():
        
        # Essentially reimplemented from networkx (with some degree of parallelism)
        # With an entirely parallel implementation oscilations occur which lead to the 
        # propagation slowly or not converging at all (and worse results). As a
        # tradeoff using just some degree of parallelism.
        graph = torch.tensor(graph, device=device)
        all_labels = []
        for _ in range(n_repeats):
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

            labels = torch.argmax(labels, dim=1)

            if iters == max_iter:
                warnings.warn("label propagation failed to converge")
            else:
                all_labels.append(labels.cpu())


        ksum = torch.zeros((len(labels), len(labels)), dtype=torch.float32, device=device)
        for i in range(len(all_labels)):
            eq = all_labels[i].to(device) == all_labels[i].to(device).unsqueeze(1)
            ksum += eq
        ksum *= (1 / len(all_labels))

        labelprob = ksum >= min_prob
        labels = torch.argmax(labelprob.to(torch.float32), dim=1).cpu()

        return labels