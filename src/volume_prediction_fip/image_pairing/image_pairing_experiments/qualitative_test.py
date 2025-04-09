from fip_global import fip_detection
import json
import os
import cv2
from collections import defaultdict

if __name__ == "__main__":
    base_date = "2023_07_10_12_56_Lot1"
    pose_config = rf"assets/poses/{base_date}.json"
    with open(pose_config) as f:
        pose_config = json.load(f)
    input_scan = rf"F:\FIP-data\images\2023\WW034\debayered\{base_date}\FPWW0340106_FIP2_20230710_120713"
    output_folder = r"F:\wheat-scans-simplyfied-fast\bboxes_ground_truth\qualit_out"

    print("lg")
    boxes, _ = fip_detection.find_objects_yolo("assets/model-weights/yolo-medium-detect-mAp50-0766.pt", input_scan, batch_size=1)
    print(f"Yolo")
    clustered = fip_detection.connect_boxes(boxes, pose_config, min_view=10)

    print(f"Done")
    
    os.makedirs(output_folder, exist_ok=True)
    cluster_dict = defaultdict(list)
    for obj in clustered:
        if obj["cluster"] is None:
            continue
        cluster_dict[obj['cluster']].append(obj)

    print(f"Num clusters: {len(cluster_dict)}")

    loaded_images = {}

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




