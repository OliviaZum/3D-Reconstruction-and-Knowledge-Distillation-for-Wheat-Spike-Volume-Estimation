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
from multiview_graph import build_epipolar_graph_opt, lp_cluster_torch
import os

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

    return graph, idx_to_instance

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

"""
Converts a clustering based on list of lists with indizes into a 1d array
with a class label for each cluster. Changes the order of ground truth.
"""
def to_sklearn_metric_format(idx_to_instance, communities):
    pred_gt_idx = []

    current_class = 0
    for community in communities:
        for c in community:
            _, id = idx_to_instance[c]
            pred_gt_idx.append((int(id), current_class, c))
        current_class += 1

    pred_gt_idx.sort(key=lambda x : x[2])
    
    return zip(*pred_gt_idx)

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


def force_opt(g: np.ndarray):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        g_attract = torch.tensor(g, dtype=torch.float32)
        # At distance 10 force should be zero. c * 10 - u = 0 => c = u / 10
        g_retract_sub: torch.Tensor = (g_attract == 0).to(torch.float32) * 50
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

def load_and_build_graph(file: str, recompute = False, store_cache = True):
    data = load_scene_data(file)
    with open("assets/fip_poses_configuration.json") as f:
        conf = json.load(f)

    fname = "graphcache.cache"
    if not recompute:
        try:
            v, data = joblib.load(fname)
            return v, data
        except:
            recompute = True
    if recompute:
        print("Building graph")
        v = build_epipolar_graph_opt(conf, data)
        print("Done")

        if store_cache:
            joblib.dump((v, data), fname)
    return v, data

def show_stats(tp_fn_fp_views: np.ndarray, num_views: np.ndarray, mean_deviation_views: np.ndarray):
    def to_rates(sums):
        return {"FN rate": sums[1] / sums[0], "FP rate": sums[2] / sums[0], "Error rate": (sums[2] + sums[1]) / sums[0]}
    rates_6_10 = to_rates(tp_fn_fp_views[:, 5:9].sum(axis=1))
    rates_10_12 = to_rates(tp_fn_fp_views[:, 9:].sum(axis=1))
    rates_12 = to_rates(tp_fn_fp_views[:, 11])
    print(f"Rates 6-10 Views: {rates_6_10}")
    print(f"Rates 10-12 Views: {rates_10_12}")
    print(f"Rates 12 Views: {rates_12}")

    x = np.arange(1, 13)
    width = 0.2
    fig, ax = plt.subplots(1, 3, figsize=(8, 6))
    ax[0].bar(x - width, tp_fn_fp_views[1, :], width, label='FN', color='red')
    ax[0].bar(x + width, tp_fn_fp_views[2, :], width, label='FP', color='yellow')
    ax[0].bar(x, tp_fn_fp_views[0, :], width, label='TP', color='green')
    ax[0].set_xlabel('# Views')
    ax[0].set_ylabel('# Pairs')
    ax[0].set_title('FN, FP, TP by Views')
    ax[0].set_xticks(x)
    ax[0].legend()

    ax[1].bar(x, num_views, width)
    ax[1].set_title("#Views per distinct object")
    ax[1].set_xticks(x)

    ax[2].bar(x, mean_deviation_views, width, label="MAE cluster size deviation")
    ax[2].set_xlabel("# Views")
    ax[2].set_ylabel("MAE")
    ax[2].set_title("Mean absolute cluster size error per node")
    ax[2].set_xticks(x)
    plt.tight_layout()
    plt.show()

def run_cluster_test(graph: np.ndarray, idx_to_instance: Dict, ground_truth, task : str = 'force_opt', verbose=True):
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
        np.fill_diagonal(graph, 0)
        g = create_networkx_graph(graph)
        communities = community.label_propagation.asyn_lpa_communities(g, weight="weight")

    if task == 'force_opt' or task == 'lpatorch':
        if task == 'force_opt':
            r = force_opt(graph)
        elif task == 'lpatorch':
            r = lp_cluster_torch(graph).numpy()
        u = {}
        for j, v in enumerate(r):
            u.setdefault(v, [])
            u[v].append(j)
        communities = list(u.values())

    if task == 'sym_nmf':
        communities = SymNMF(graph)

    if task != 'vis':
        gt, pred, _ = to_sklearn_metric_format(idx_to_instance, communities)

        # Compute #observations per object
        num_view_per_obj = {}
        for _, instance in ground_truth.items():
            for id, _ in instance.items():
                num_view_per_obj.setdefault(id, 0)
                num_view_per_obj[id] += 1

        # Build reverse idx <-> id dict
        instance_to_idx = {}
        for idx, (_, id) in idx_to_instance.items():
            instance_to_idx.setdefault(id, [])
            instance_to_idx[id].append(idx)

        # Compute error per ground truth cluster size (num observations) and deviation from true cluster size per node
        tp_fn_fp_views = np.zeros((3, 0))
        mean_deviation_views = []
        gt_np = np.array(gt)
        pred_np = np.array(pred)
        indices = np.arange(0, len(gt))
        for current_num_view in range(1, 13):
            objs = [x for x, v in num_view_per_obj.items() if v == current_num_view]
            idxgt = set([x for obj in objs for x in instance_to_idx[obj]])
            tp_fn_fp = np.zeros(3)
            size_deviation = 0
            for idx in idxgt:
                mtp = np.logical_and(gt_np[idx] == gt_np[indices], pred_np[idx] == pred_np[indices])
                mfn = np.logical_and(gt_np[idx] == gt_np[indices], pred_np[idx] != pred_np[indices])
                mfp = np.logical_and(gt_np[idx] != gt_np[indices], pred_np[idx] == pred_np[indices])
                pred_cluster_size = (pred_np[idx] == pred_np[indices]).sum()
                size_deviation += np.abs(pred_cluster_size - current_num_view)
                tp_fn_fp += np.array([mtp.sum() - 1, mfn.sum(), mfp.sum()]) # If idx == i will count one tp to much
            if len(idxgt) > 0:
                mean_deviation_views.append(size_deviation / len(idxgt))
            else:
                mean_deviation_views.append(0)
            tp_fn_fp_views = np.concatenate((tp_fn_fp_views, np.expand_dims(tp_fn_fp, 1)), axis=1)
        mean_deviation_views = np.array(mean_deviation_views)

        def accumulate_frequencies(arr):
            max_value = max(arr)
            frequency_array = np.zeros(max_value)
            for number in arr:
                frequency_array[number-1] += 1
            return frequency_array
        
        
        if verbose:
            show_stats(tp_fn_fp_views, accumulate_frequencies(num_view_per_obj.values()), mean_deviation_views)
            print("Correct confusion matrix (pred = gt): ")
            print(metrics.pair_confusion_matrix(gt, gt))
            print("confusion matrix (considering all pairs) (C00: TN, C10: FN, C11: TP, C01: FP): ")
            print(metrics.pair_confusion_matrix(gt, pred))

        return tp_fn_fp_views, accumulate_frequencies(num_view_per_obj.values()), mean_deviation_views
        
def run_cluster_test_multiple(dir):
    tp_fn_fp_views_agg = np.zeros((3, 12))
    num_view_agg = np.zeros(12)
    mean_deviation_views_agg = np.zeros(12)
    counter = 0
    for file in os.listdir(dir):
        if os.path.splitext(file)[1] == ".json":
            (graph, idx_to_instance), data = load_and_build_graph(os.path.join(dir, file), True, False)
            tp_fn_fp_views, num_view, mean_deviation_views = run_cluster_test(graph.cpu().numpy(), idx_to_instance, data, 'lpatorch', False)
            tp_fn_fp_views_agg += tp_fn_fp_views
            num_view_agg += num_view
            mean_deviation_views_agg += mean_deviation_views
            counter += 1
    mean_deviation_views_agg /= counter
    num_view_agg /= counter
    
    show_stats(tp_fn_fp_views_agg, num_view_agg, mean_deviation_views_agg)


run_cluster_test_multiple("F:/wheat-scans-simplyfied-fast/bboxes_ground_truth/val")
"""
(graph, idx_to_instance), data = load_and_build_graph("F:/wheat-scans-simplyfied-fast/bboxes_ground_truth/val/cdc2f6a9-ea1f-4cfb-8b00-9c8d25b13cdb.json", False)
graph = graph.cpu().numpy()
for _ in range(1):
    run_cluster_test(graph, idx_to_instance, data, 'lpatorch')
"""