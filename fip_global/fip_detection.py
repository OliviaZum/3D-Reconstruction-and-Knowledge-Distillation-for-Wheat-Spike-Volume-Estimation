import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

from typing import Dict, List, Tuple
from ultralytics import YOLO
from ultralytics.engine.results import Results
from image_pairing import multiview_graph
import numpy as np
import cv2

# Find all spikes on all images
def find_objects_yolo(model: YOLO, scan_folder: str, min_conf = 0.25, batch_size = 5) -> Tuple[Dict[str, List[Tuple[int, List]]], List]:
    imgs = [os.path.join(scan_folder, f"cam_{i:02}.png") for i in range(1, 13)]
    for img in imgs:
        if not os.path.exists(img):
            raise Exception(f"One of the required images {img} has not been found")
    results: list[Results] = []
    for i in range(0, len(imgs), batch_size):
        tmp = model(imgs[i:min(i+batch_size, len(imgs))], imgsz=3008, max_det=800, iou=0.5, conf=min_conf, verbose=False)
        results.extend(tmp)

    # This conversion is somewhat innefficient. One could go without it later on
    converted = {}
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy().astype(int) # astype untested currently
        
        current = {}
        for i in range(boxes.shape[0]):
            current[i] = list(boxes[i, :])
        converted[os.path.basename(result.path)] = current

    return converted, results

# Segment a spike (cut out, returns the largest spike on the image with background blacked out)
def segment_spikes(seg_model, img):
    r = seg_model(img, imgsz=288, verbose=False, conf=0.2)
    r: Results = r[0].cpu()
    if r.boxes.shape[0] > 0:
        boxes = r.boxes.xywh
        area = boxes[:, 2] * boxes[:, 3]
        max_box = np.argmax(area)
        scaled_mask = cv2.resize(r.masks.data[max_box].numpy(), list(reversed(r.orig_img.shape[0:2])), interpolation=cv2.INTER_NEAREST).astype(np.bool_)
        r.orig_img[~scaled_mask] = 0
        return r.orig_img
    else:
        return None

# Find which boxes correspond to the same object (spike)
def connect_boxes(boxes: Dict[str, Dict[str, List[int]]], poses_conf, min_view = 6):
    graph, idx_to_instance = multiview_graph.build_epipolar_graph_opt(poses_conf, boxes)
    graph = graph.detach().cpu().numpy()
    labels = multiview_graph.lp_cluster_torch(graph).detach().cpu().numpy()
    distances = multiview_graph.estimate_distances(poses_conf, boxes, labels, idx_to_instance, min_view if min_view >= 3 else 3)

    labelcount = {}
    for v in labels:
        labelcount.setdefault(v, 0)
        labelcount[v] += 1

    combined_results = []
    for i in range(len(idx_to_instance)):
        image_name, id = idx_to_instance[i]
        box = boxes[image_name][id]
        box = [int(round(x)) for x in box] # in case not already int convert it here as last
        label = int(labels[i])
        current_dist = distances[image_name][label]
        current_dist = float(current_dist) if current_dist else None
        if labelcount[label] < min_view:
            label = None
        combined_results.append({"image": image_name, "cluster": label, "box": box, "distance": current_dist})

    return combined_results

# Helper function to export the results of detection and pairing
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

        cv2.imwrite(os.path.join(dir, os.path.splitext(os.path.basename(result.path))[0] + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 70])