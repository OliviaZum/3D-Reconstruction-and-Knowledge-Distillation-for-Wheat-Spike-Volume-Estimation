"""
Takes some scan and outputs the cropped, paired bounding boxes to a folder.
Can also be used to output the full images with annotated boxes, cluster ids and distance.
"""

import json
import os
import cv2
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from volume_prediction_fip.fip_global import fip_detection
from volume_prediction_fip.utils import helpers

def export_individual_spikes(cluster_dict, output_folder):
    for cluster_id, items in cluster_dict.items():
        for item in items:
            image_path = os.path.join(input_scan, item["image"])
            
            if image_path not in loaded_images:
                img = cv2.imread(image_path)
                assert img is not None
                loaded_images[image_path] = img
            else:
                img = loaded_images[image_path]

            x1, y1, x2, y2 = item["box"]
            cropped = img[y1:y2, x1:x2]

            output_path = os.path.join(output_folder, f"{cluster_id}_{item['image']}")
            cv2.imwrite(output_path, cropped)

def export_global_images(combined_results, dir_in, dir_out, write_distance = True, write_id = True, shade_distance = False,
                         max_dist=3.5, min_dist=2.5):
    dist_cmap = plt.get_cmap("Greens")
    colors = {-1 : (1.0, 0.0, 0.0)}

    boxes = {}
    for b in combined_results:
        boxes.setdefault(b["image"], [])
        boxes[b["image"]].append((b["cluster"], b["box"], b["distance"]))

    # Iterate through all 12 result objects
    imgs = [f"cam_{i:02}.png" for i in range(1, 13)]
    for img_name in imgs:
        img = cv2.imread(os.path.join(dir_in, img_name))

        # Get the bounding boxes and IDs for the current image
        for id, bbox, distance in boxes[img_name]:
            x1, y1, x2, y2 = bbox  # Extract the coordinates of the bounding box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw the bounding box
            if shade_distance:
                if distance:
                    n_dist = np.clip((distance - min_dist) / (max_dist - min_dist), 0, 1)
                    color = dist_cmap(n_dist)
                else:
                    color = (1.0, 0, 0)
            else:
                colors.setdefault(id, np.random.rand(3))
                color = colors[id]
            img = cv2.rectangle(img, (x1, y1), (x2, y2), color=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), thickness=2)
            
            # Annotate with the ID (inside the bounding box)
            if distance and write_distance and write_id:
                text = f"{id}, {distance:.2f}"
            elif distance and write_distance:
                text = f"{distance:.2f}"
            elif write_id:
                text = f"{id}"
            else:
                text = ""
            if write_id or write_distance:
                img = cv2.putText(img, text, (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), 2)

        out_path = os.path.join(dir_out, f"visu_{os.path.splitext(img_name)[0]}.jpg")
        cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 70])

if __name__ == "__main__":
    base_date = "2023_07_10_12_56_Lot1"
    pose_config = str(helpers.get_assets_path() / rf"poses/{base_date}.json")
    input_scan = rf"F:\FIP-data\images\2023\WW034\debayered\{base_date}\FPWW0340106_FIP2_20230710_120713"
    output_folder = r"C:\Users\Admin\Downloads/qualitative_out"
    single_spikes = False
    distance_estimates = True

    with open(pose_config) as f:
        pose_config = json.load(f)
    os.makedirs(output_folder, exist_ok=True)

    model = helpers.get_detection_model()
    boxes, _ = fip_detection.find_objects_yolo(model, input_scan)
    clustered = fip_detection.connect_boxes(boxes, pose_config, min_view=10)
    
    cluster_dict = defaultdict(list)
    for obj in clustered:
        if obj["cluster"] is None:
            continue
        cluster_dict[obj['cluster']].append(obj)

    print(f"Num clusters: {len(cluster_dict)}")

    loaded_images = {}

    if distance_estimates:
        export_global_images(clustered, input_scan, output_folder, write_distance=True, write_id=False, shade_distance=True,
                             max_dist=3.5, min_dist=3.0)
    if single_spikes:
        export_individual_spikes(cluster_dict, output_folder)