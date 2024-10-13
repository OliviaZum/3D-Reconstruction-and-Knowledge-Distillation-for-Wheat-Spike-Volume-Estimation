import sys
sys.path.append("C:/Users/Admin/Desktop/master_thesis/volume_prediction_fip")

from typing import Dict, List
import numpy as np
import epipolar_geometry
from synthetic_fip_render.gen_scene import load_scene_data
import json
from pyvis.network import Network
import networkx as nx
from networkx.algorithms import community
from sklearn import metrics
from sklearn.cluster import MeanShift
import torch
from torch.optim.adam import Adam
import matplotlib.pyplot as plt
import joblib
import tqdm
import time

def build_epipolar_graph_opt(poses_conf, bounding_boxes: Dict[str, Dict[str, List[float]]], num_samples=5):

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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    start = time.time()
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

        # Actual do not match matrix is given by do_not_match @ do_not_match.T, it's rank #images
        do_not_match = np.ndarray((np.sum(num_instance_per_image), len(num_instance_per_image)))
        current_write = 0
        for i, v in enumerate(num_instance_per_image):
            base = np.zeros((1, len(num_instance_per_image)))
            base[0, i] = 1
            for _ in range(v):
                do_not_match[current_write, :] = base
                current_write += 1
        
        cam_names = list(bounding_boxes.keys())
        graph = torch.zeros([len(idx_to_instance)] * 2, device=device)

        for i in range(len(cam_names)):
            for j in range(i+1, len(cam_names)):
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
                graph[pos_1:end_1, pos_2:end_2] = counts.T
                graph[pos_2:end_2, pos_1:end_1] = counts

        return graph.cpu().numpy(), idx_to_instance, do_not_match
    
def lp_cluster_torch(graph: np.ndarray, do_not_match):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        graph = torch.tensor(graph, device=device)
        graph[graph == 0] = -10
        graph[do_not_match @ do_not_match.T == 1] = -50
        graph.fill_diagonal_(0)
        
        # Essentially reimplemented from networkx (with some degree of parallelism)
        # With an entirely parallel implementation oscilations occur which lead to the 
        # graph slowly or not converging at all (and worse results). As a
        # tradeoff using just some degree of parallelism.
        labels = torch.eye(graph.shape[0], device=device)
        no_converge = True
        num_parallel = graph.shape[0] // 20
        while no_converge:
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

        labels = torch.argmax(labels, dim=1).cpu()
        return labels

# Expects a json of camera_name: {instance_id1: bounding_box, ...}, ...
def build_epipolar_graph(poses_conf, bounding_boxes: Dict[str, Dict[str, List[float]]], num_samples=5):
    idx_to_instance = {}
    instance_to_idx = {}
    i = 0
    num_instance_per_image = []
    for (n, v) in bounding_boxes.items():
        num_instance_per_image.append(len(v))
        for id in v.keys():
            idx_to_instance[i] = (n, id)
            instance_to_idx[(n, id)] = i
            i += 1

    # Actual do not match matrix is given by do_not_match @ do_not_match.T, it's rank #images
    do_not_match = np.ndarray((np.sum(num_instance_per_image), len(num_instance_per_image)))
    current_write = 0
    for i, v in enumerate(num_instance_per_image):
        base = np.zeros((1, len(num_instance_per_image)))
        base[0, i] = 1
        for _ in range(v):
            do_not_match[current_write, :] = base
            current_write += 1

    cam_names = list(bounding_boxes.keys())
    graph = np.zeros([len(idx_to_instance)] * 2)


    for i in range(len(cam_names)):
        for j in range(i+1, len(cam_names)):
            im1 = cam_names[i]
            im2 = cam_names[j]
            F = epipolar_geometry.build_fundamental(poses_conf, im1, im2)

            for id1, (xmin, ymin, xmax, ymax) in tqdm.tqdm(bounding_boxes[im1].items(), f"IMG {i} - IMG {j}"):
                scale_array = np.array([xmax - xmin, ymax - ymin])
                corner = np.array([xmin, ymin])
                for _ in range(num_samples):
                    p = np.array([0, 0, 1])
                    p[0:2] = np.random.uniform(0, 1, 2) * scale_array + corner
                    for id2, box in bounding_boxes[im2].items():
                        hit = epipolar_geometry.check_epipolar_intersect_bbox(F, p, box)
                        if hit:
                            idx1 = instance_to_idx[(im1, id1)]
                            idx2 = instance_to_idx[(im2, id2)]
                            graph[idx1, idx2] += 1
                            graph[idx2, idx1] += 1

    return graph, idx_to_instance, do_not_match

def visualize_graph(adj_matrix, idx_to_instance):
    net = Network(notebook=False)

    # Define a color palette for the img_name (up to 12 different img names)
    colors = [
        'red', 'blue', 'green', 'purple', 'orange', 'yellow', 'cyan', 'magenta',
        'pink', 'lime', 'lightblue', 'lightgreen'
    ]
    
    # Get a list of unique img_names and map them to colors
    unique_img_names = list(set([instance[0] for instance in idx_to_instance.values()]))
    unique_img_names.sort()
    img_name_to_color = {img_name: colors[i % len(colors)] for i, img_name in enumerate(unique_img_names)}

    # Draw
    for idx, (img_name, node_id) in idx_to_instance.items():
        net.add_node(idx, label=str(node_id), color=img_name_to_color[img_name], shape="circle")

    rows, cols = np.nonzero(adj_matrix)
    
    for i, j in zip(rows, cols):
        if i < j:
            weight = adj_matrix[i, j]
            net.add_edge(int(i), int(j), value=int(weight))

    net.show("g.html", notebook=False)

def create_networkx_graph(adj_matrix):
    G = nx.Graph()

    for i in range(adj_matrix.shape[0]):
        G.add_node(i)

    rows, cols = np.nonzero(adj_matrix)
    
    for i, j in zip(rows, cols):
        if i < j:
            weight = adj_matrix[i, j]
            G.add_edge(int(i), int(j), weight=int(weight))

    return G

def to_sklearn_metric_format(idx_to_instance, communities):
    pred_idx = []
    pred_class_label = []
    gt_class_label = []

    current_class = 0
    for community in communities:
        for c in community:
            pred_idx.append(c)
            pred_class_label.append(current_class)
        current_class += 1
    
    for w in pred_idx:
        _, id = idx_to_instance[w]
        gt_class_label.append(int(id))
    return gt_class_label, pred_class_label

def SymNMF(g: np.ndarray):
    def cd(g):
        d = torch.sqrt(g.sum(axis=1, keepdims=True))
        d[d == 0] = 1
        d = 1 / d
        d = d.squeeze()
        g = torch.diag(d) @ g @ torch.diag(d)
        return g

    g = torch.tensor(g, dtype=torch.float)
    g = cd(g)
    print(g.max())
    u = torch.rand((g.shape[0], 8), requires_grad=False) * float(torch.sqrt(g.max()).item())
    w = u.clone().detach().requires_grad_(True)

    optim = Adam([w], lr=0.01)

    for _ in range(5000):
        l = torch.norm(w @ w.T - g)
        l.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)
        with torch.no_grad():
            w[w < 0] = 0
        #print(l)

    wmax = torch.argmax(w, dim=1)
    communities = [[] for _ in range(w.shape[1])]
    for i, v in enumerate(wmax):
        communities[v].append(i)

    return communities


def force_opt(g: np.ndarray, do_not_match: np.ndarray):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        g_attract = torch.tensor(g, dtype=torch.float32)
        # At distance 10 force should be zero. c * 10 - u = 0 => c = u / 10
        g_retract_sub: torch.Tensor = (g_attract == 0).to(torch.float32) * 50
        g_retract_sub[do_not_match @ do_not_match.T == 1] = 3000
        g_retract_mul: torch.Tensor = g_retract_sub / 10

        k = 10
        lr = 0.0002
        embedding = torch.rand((g.shape[0], k)) * 10

        g_attract = g_attract.to(device)
        g_retract_sub = g_retract_sub.to(device)
        g_retract_mul = g_retract_mul.to(device)
        embedding = embedding.to(device)

        for e in tqdm.tqdm(range(200)):
            v_pairwise = embedding.unsqueeze(1) - embedding.unsqueeze(0)
            dists = torch.sqrt((v_pairwise ** 2).sum(dim=2))
            v_dir = v_pairwise / dists.unsqueeze(2)
            v_dir[torch.isnan(v_dir)] = 0

            forceAttr = g_attract * dists
            #forceRep = dists * g_retract - 10
            forceRep = dists * g_retract_mul - g_retract_sub
            forceRep[forceRep > 0] = 0

            force = forceAttr + forceRep
            v_force = v_dir * force.unsqueeze(2)
            c = v_force.sum(dim=0)
            embedding += lr * c
            totalForce = torch.sum(torch.abs(force))
            print(f"{e}: {totalForce}")

        clustering = MeanShift(bandwidth=5).fit(embedding.cpu().numpy())

        #plt.scatter(embedding.cpu().numpy()[:, 0], embedding.cpu().numpy()[:, 1], c=clustering.labels_, cmap="viridis")
        #plt.show()
        return clustering.labels_

def load_and_build_graph(file: str, recompute = False):
    data = load_scene_data(file)
    with open("assets/fip_poses_configuration.json") as f:
        conf = json.load(f)

    fname = "graphcache.cache"
    if not recompute:
        try:
            v = joblib.load(fname)
            return v
        except:
            recompute = True
    if recompute:
        print("Building graph")
        v = build_epipolar_graph_opt(conf, data)
        print("Done")

        joblib.dump(v, fname)
    return v
    

def run_cluster_test(graph: np.ndarray, idx_to_instance: Dict, do_not_match: np.ndarray, task : str = 'force_opt'):
    tasks = {'force_opt', 'sym_nmf', 'vis', 'lovain', 'lpa', 'lpatorch'}
    if task not in tasks:
        print(f"Accepted Tasks are {tasks}")

    if task == 'vis':
        print("Visualization")
        visualize_graph(graph, idx_to_instance)

    if task == 'louvain':
        g = create_networkx_graph(graph)
        #communities = community.greedy_modularity_communities(g, weight="weight", resolution=3)
        communities = community.louvain_communities(g, weight="weight", resolution=1)

    if task == 'lpa':
        graph[graph == 0] = -10
        mask = do_not_match @ do_not_match.T == 1
        graph[mask] = -50
        np.fill_diagonal(graph, 0)
        g = create_networkx_graph(graph)
        communities = community.label_propagation.asyn_lpa_communities(g, weight="weight")

    if task == 'force_opt' or task == 'lpatorch':
        if task == 'force_opt':
            r = force_opt(graph, do_not_match)
        elif task == 'lpatorch':
            r = lp_cluster_torch(graph, do_not_match).numpy()
        u = {}
        for j, v in enumerate(r):
            u.setdefault(v, [])
            u[v].append(j)
        communities = list(u.values())

    if task == 'sym_nmf':
        communities = SymNMF(graph)

    if task != 'vis':
        gt, pred = to_sklearn_metric_format(idx_to_instance, communities)

        confusion = metrics.pair_confusion_matrix(gt, pred)
        print(f"All pairs: {(len(gt) - 1) * len(gt)}")
        print("Correct confusion matrix (pred = gt): ")
        print(metrics.pair_confusion_matrix(gt, gt))
        print("confusion matrix (considering all pairs) (C00: TN, C10: FN, C11: TP, C01: FP): ")
        print(confusion)
    
#graph, idx_to_instance, do_not_match
v = load_and_build_graph("F:/wheat-scans-simplyfied-fast/bboxes_ground_truth/simple/f46eabfb-2775-49bf-8c78-967d5c43ccf2.json", True)
run_cluster_test(*v,'lpatorch')