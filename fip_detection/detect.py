import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

from typing import Dict, List, Tuple
from ultralytics import YOLO
from ultralytics.engine.results import Results
from multiview_instance_epipole import multiview_graph
import json
import time

def find_objects(yolo_weights_path: str, poses_conf, scan_folder: str, min_view = 6, min_conf = 0.25, batch_size = 5) -> Tuple[Dict[str, List[Tuple[int, List]]], List]:
    model = YOLO(yolo_weights_path, task='detect')
    imgs = [os.path.join(scan_folder, f"cam_{i:02}.png") for i in range(1, 13)]
    for img in imgs:
        if not os.path.exists(img):
            raise Exception(f"One of the required images {img} has not been found")
    results: list[Results] = []
    for i in range(0, len(imgs), batch_size):
        tmp = model(imgs[i:min(i+batch_size, len(imgs))], imgsz=3008, max_det=800, iou=0.5, batch=2, conf=min_conf)
        results.extend(tmp)

    # This conversion is somewhat innefficient. One could go without it later on
    converted = {}
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        
        current = {}
        for i in range(boxes.shape[0]):
            current[i] = list(boxes[i, :])
        converted[os.path.basename(result.path)] = current
    
    graph, idx_to_instance = multiview_graph.build_epipolar_graph_opt(poses_conf, converted)
    graph = graph.cpu().numpy()
    labels = multiview_graph.lp_cluster_torch(graph).numpy()

    labelcount = {}
    for v in labels:
        labelcount.setdefault(v, 0)
        labelcount[v] += 1

    # Dict mapping from image name to id, bounding box. Unsure bounding boxes (less than min_view predicted views get label -1)
    combined_results: Dict[str, Tuple[int, List]] = {} 
    for i in range(len(idx_to_instance)):
        image_name, id = idx_to_instance[i]
        box = converted[image_name][id]
        combined_results.setdefault(image_name, [])
        label = labels[i]
        if labelcount[label] < min_view:
            label = -1
        combined_results[image_name].append((label, box))

    return combined_results, results


def save_results(boxes, results, dir):
    import numpy as np
    import cv2

    colors = {-1 : (1.0, 0.0, 0.0)}

    # Setup the plot for 12 images in a 4x3 grid

    # Iterate through all 12 result objects
    for i, result in enumerate(results):
        img = result.orig_img  # Get the image (h, w)

        # Get the bounding boxes and IDs for the current image
        for id, bbox in boxes[os.path.basename(result.path)]:
            x1, y1, x2, y2 = bbox  # Extract the coordinates of the bounding box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw the bounding box
            colors.setdefault(id, np.random.rand(3))
            color = colors[id]
            img = cv2.rectangle(img, (x1, y1), (x2, y2), color=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), thickness=2)
            
            # Annotate with the ID (inside the bounding box)
            img = cv2.putText(img, str(id), (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), 2)

        cv2.imwrite(os.path.join(dir, os.path.splitext(os.path.basename(result.path))[0] + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 30])

if __name__ == "__main__":
    with open("assets/fip_poses_configuration.json") as f:
        conf = json.load(f)
    print("Starting find objects")
    start = time.time()
    boxes, results = find_objects("C:/Users/Admin/Desktop/master_thesis/volume_prediction_fip/detection-yolo/weights/detect/medium-train+val/weights/best.pt", 
                                    conf,
                                    "F:/FIP-data/2023/WW034/debayered/2023_06_08_13_11_Lot1/FPWW0340091_FIP2_20230608_122303", batch_size=1)
    print(f"Find objects ended. Time {time.time() - start}")
    
    save_results(boxes, results, "tmpstuff/tmp_results")