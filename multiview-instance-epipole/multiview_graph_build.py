import sys
sys.path.append("C:/Users/Admin/Desktop/master_thesis/volume_prediction_fip")

from typing import Dict, List
import numpy as np
import epipolar_geometry
from synthetic_fip_render.gen_scene import load_scene_data
import json
from pyvis.network import Network

# Expects a json of camera_name: {instance_id1: bounding_box, ...}, ...
def build_epipolar_graph(poses_conf, bounding_boxes: Dict[str, Dict[str, List[float]]], num_samples=5):
    idx_to_instance = {}
    instance_to_idx = {}
    i = 0
    for (n, v) in bounding_boxes.items():
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

            for id1, (xmin, ymin, xmax, ymax) in bounding_boxes[im1].items():
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

def build_networkx_graph():
    pass

def load_and_run_test():
    _, data = next(load_scene_data("F:/wheat-scans-simplyfied-fast/bboxes_ground_truth/simple/"))
    with open("assets/fip_poses_configuration.json") as f:
        conf = json.load(f)

    print("Building graph")
    graph, idx_to_instance = build_epipolar_graph(conf, data)

    print("Visualization")
    visualize_graph(graph, idx_to_instance)

load_and_run_test()