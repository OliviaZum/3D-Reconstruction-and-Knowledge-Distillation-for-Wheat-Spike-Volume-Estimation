import sys
import os
from typing import Dict, List
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

import json
import numpy as np
import re
import pandas as pd
import random
from utils import helpers
import cv2
import tqdm
from ultralytics import YOLO
from fip_detection import detect

class FIPDataset:
    def __init__(self, csv_folder, img_folder, ply_folder, precompute_file = None, spikelabels_file = None, pose_folder = None) -> None:
        if precompute_file:
            with open(precompute_file) as f:
                self.precomputed = json.load(f)
        else:
            self.precomputed = {}
        
        # Read design files
        df2023 = pd.read_csv(os.path.join(csv_folder, "Design_Olivia_Zumsteg_2023.csv"), encoding="ISO-8859-1")
        m2023 = self.inversemapping(self.labels2num2023mapping())
        df2023["year"] = "2023"
        df2023["label"] = df2023["plant_id"].apply(lambda x: m2023[x])

        df2024 = pd.read_csv(os.path.join(csv_folder, "Design_Olivia_Zumsteg_2024.csv"), encoding="ISO-8859-1")
        m2024 = self.inversemapping(self.labels2num2024mapping())
        df2024["label"] = df2024["plant_id"].apply(lambda x: m2024[x])
        df2024["year"] = "2024"
        self.spikescans = pd.concat((df2023, df2024), ignore_index=True)

        # Set corresponding scan and volume
        self.spikescans["ply"] = None
        self.spikescans["spikevolume"] = None
        for current_dir, _, files in os.walk(ply_folder):
            for p in files:
                fullpath = os.path.join(current_dir, p)
                if os.path.isfile(fullpath) and os.path.splitext(p)[1] == ".ply":
                    range_lot, row_lot, plant = str.split(os.path.splitext(p)[0], "_")
                    cond = (self.spikescans["range_lot"] == int(range_lot)) & (self.spikescans["row_lot"] == int(row_lot)) & (self.spikescans["plant_id"] == int(plant))
                    self.spikescans.loc[cond, "ply"] = fullpath
                    with open(os.path.join(current_dir, os.path.splitext(p)[0] + ".txt")) as f:
                        l = f.read()
                    self.spikescans.loc[cond, "spikevolume"] = float(l)

        # Remove spikes without volume
        preclean = len(self.spikescans)
        self.spikescans = self.spikescans.dropna(subset=["ply"])
        print(f"Deleted {preclean - len(self.spikescans)} records where no scan has been found. Left: {len(self.spikescans)}")
        
        # Add image folder
        uid_to_folder = {}
        pattern = r"FPWW\d*(\d{3})_FIP2_(\d*)_\d*"
        for current_dir, _, _ in os.walk(img_folder):
            n = os.path.basename(current_dir)
            match = re.search(pattern, n)
            if match:
                uid_to_folder[(int(match.group(1)), match.group(2))] = current_dir

        def set_folder(row):
            y = row["year"]
            p = row["plant_id"]
            if y == "2023":
                if p in [1, 10, 2]:
                    d = "20230608"
                elif p in [3, 4]:
                    d = "20230626"
                else:
                    d = "20230710"
            elif y == "2024":
                if p in [1, 2, 3]:
                    d = "20240610"
                elif p in [4, 5]:
                    d = "20240704"
                else:
                    d = "20240718"
            else:
                assert False
            return uid_to_folder[(row["plot"], d)]
        
        self.spikescans["image_dir"] = self.spikescans.apply(set_folder, axis=1)

        # Load poses
        if pose_folder is not None:
            self.poses = {}
            for file in os.listdir(pose_folder):
                with open(os.path.join(pose_folder, file)) as f:
                    self.poses[os.path.splitext(file)[0]] = json.load(f)

        # Load annotations
        if spikelabels_file:
            camkeys = [f"cam_{i:02}.png" for i in range(1, 13)]
            df = pd.read_csv(spikelabels_file)
            new_rows = []
            for _, row in df.iterrows():
                image_dir = row['image_dir']
                labeled_data = json.loads(row['labeled_spike'])
                verified = labeled_data.get("verified", False)
                
                for label in self.spikescans.loc[self.spikescans["image_dir"] == image_dir, "label"]:
                    new_row = {
                        "image_dir": image_dir,
                        "label": label,
                        "verified": verified,
                    }
                    
                    for cam_key in camkeys:
                        if cam_key in labeled_data and label in labeled_data[cam_key]:
                            new_row[cam_key] = labeled_data[cam_key][label]
                        else:
                            new_row[cam_key] = None

                    if label in labeled_data["label_selectedon"]:
                        new_row["label_selectedon"] = labeled_data["label_selectedon"][label]
                    else:
                        new_row["label_selectedon"] = None
                    
                    new_rows.append(new_row)

            df = pd.DataFrame(new_rows)
            before = len(df)
            df = df.dropna(subset=camkeys, how='all')
            print(f"Ignoring {before - len(df)} of {before} spikes because no label is set")

            self.spikescans = pd.merge(self.spikescans, df, how='inner', on=["image_dir", "label"])

    def get_pose_config(self, folder):
        for k, v in self.poses.items():
            if k in folder:
                return v

    def precompute_boxes(self, precompute_file, update_connections_only = False):
        # Precompute bounding boxes and connections between them
        # If update_connections_only just change the connections between the boxes, don't repredict the boxes
        if update_connections_only:
            assert self.precomputed is not None
        else:
            self.precomputed = {}
        from fip_detection import detect
        import torch
        import tqdm
        
        result = {}
        for folder in tqdm.tqdm(self.spikescans["image_dir"].unique()):
            conf = self.get_pose_config(folder)
            if update_connections_only:
                w = self.precomputed[folder]
                boxes = {}
                id = 0 
                for boxes_img in w:
                    boxes.setdefault(boxes_img["image"], {})
                    boxes[boxes_img["image"]][id] = boxes_img["box"]
                    id += 1
                print(len(w))
            else:
                boxes, _ = detect.find_objects_yolo("detection-yolo/weights/detect/medium-train+val/weights/best.pt", 
                                                folder, batch_size=1)
            connected_boxes = detect.connect_boxes(boxes, conf, min_view=0)
            if torch.cuda.memory_reserved() // (1024**2) > 3500:
                torch.cuda.empty_cache()
            result[folder] = connected_boxes
        with open(precompute_file, "w") as f:
            json.dump(result, f)
        self.precomputed = result

    @staticmethod
    def iou(box1, box2):
        # box1 and box2 should be in (x1, y1, x2, y2) format
        x1, y1, x2, y2 = box1
        x1_b, y1_b, x2_b, y2_b = box2

        inter_x1 = max(x1, x1_b)
        inter_y1 = max(y1, y1_b)
        inter_x2 = min(x2, x2_b)
        inter_y2 = min(y2, y2_b)
        if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_b - x1_b) * (y2_b - y1_b)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area
    
    def export_plant_images(self, input_folder: str, output_folder: str, export: Dict, padding: int = 20,
                            box_size: int = 300, seg_model = None):
        camkeys = [f"cam_{i:02}.png" for i in range(1, 13)]
        table = pd.DataFrame(columns=["img_name", "volume", "distance"])
        # Load all images
        all_imgs = {}
        for img_name in camkeys:
            img = cv2.imread(os.path.join(input_folder, img_name))
            all_imgs[img_name] = img

        # Do the actual export
        for export_value in export.values():
            
            for img_name, box in export_value["boxes"]:
                new_row = {"img_name": img_name, "volume": export_value["volume"], "distance": box["distance"]}
                out_path = os.path.join(output_folder, img_name)
                img = helpers.extend_image_box(box["box"], all_imgs[box["image"]], padding)
                patch = helpers.put_image_on_patch(box_size, img)
                if seg_model:
                    patch = detect.segment_spikes(seg_model, patch)
                    if patch is None:
                        continue
                cv2.imwrite(out_path, patch)
                table.loc[len(table)] = new_row
        return table
    
    def get_segmodel(self):
        return YOLO(r"C:\Users\Admin\Desktop\master_thesis\volume_prediction_fip\jupyter\runs\segment\train3\weights\best.pt")

    def precompute_image_dataset(self, base_folder: str, padding = 20, automatic_inferred=True, box_size=300, segment=True):
        # Compute an image dataset of the individual spikes

        if os.path.exists(base_folder):
            input("The base folder exists. Press enter to overwrite...")
        os.makedirs(base_folder, exist_ok=True)
        csv_path = os.path.join(base_folder, "vol_mapping.csv")
        tables = []

        def get_box_with_max_overlap(box: List[int], boxes: List[Dict]):
            max_iou = 0
            max_box = None
            for w in boxes:
                iou = self.iou(box, w["box"])
                if iou > max_iou:
                    max_iou = iou
                    max_box = w
            return max_box, max_iou
        
        camkeys = [f"cam_{i:02}.png" for i in range(1, 13)]
        # for now segmentation is done as a post step.
        seg_model = self.get_segmodel()
        for folder in tqdm.tqdm(self.spikescans["image_dir"].unique()):
            scans: pd.DataFrame = self.spikescans.loc[self.spikescans["image_dir"] == folder]
            precomp_json = {}
            for w in self.precomputed[folder]:
                precomp_json.setdefault(w["image"], [])
                precomp_json[w["image"]].append(w)

            export = {}
            for _, scan in scans.iterrows():
                export[scan["plant_id"]] = {"volume": scan["spikevolume"], "boxes": []}
                if automatic_inferred:
                    # Create branch of images which were automatically selected. The ground truth here is the box which was selected first
                    selected = scan[scan["label_selectedon"]]
                    max_box, max_iou = get_box_with_max_overlap(selected, precomp_json[scan["label_selectedon"]])
                    if max_iou < 0.6:
                        max_id = None
                        print(f"For the scan {scan["range_lot"]}_{scan["row_lot"]}, label {scan["label"]} no matching bounding box was found")
                    else:
                        max_id = max_box["cluster"]

                for i, cam_name in enumerate(camkeys):
                    img_name = f"{scan["range_lot"]}_{scan["row_lot"]}_{scan["plant_id"]}_{i + 1}_a.jpg"

                    if not automatic_inferred:
                        # Write manual images (i.e. boxes which were selected by hand)
                        if scan[cam_name]:
                            max_box, max_iou = get_box_with_max_overlap(scan[cam_name], precomp_json[cam_name])
                            if max_iou < 0.6:
                                print(f"For the scan {scan["range_lot"]}_{scan["row_lot"]}, label {scan["label"]} no matching bounding box was found")
                                continue
                            if max_box["distance"] is None:
                                continue
                            export[scan["plant_id"]]["boxes"].append((img_name, max_box))
                    else:
                        if max_id:
                            c = 0
                            for w in precomp_json[cam_name]:
                                if max_id == w["cluster"] and w["distance"] is not None:
                                    export[scan["plant_id"]]["boxes"].append((img_name, w))
                                    break # Ignore the case where multiple boxes appear on one image for now
                                    c += 1
            
            table = self.export_plant_images(folder, base_folder, export, padding, box_size, seg_model if segment else None)
            tables.append(table)

        table = pd.concat(tables, axis=1)
        table.to_csv(csv_path, index=False)

    def generate_unlabeled_spikes(self, base_folder: str, num_plants = None, num_plants_per_scan = None, min_image_per_plant = 10, 
                                  padding: int = 20, box_size: int = 300, segment: bool = True):
        if os.path.exists(base_folder):
            input("The base folder exists. Press enter to overwrite...")
        os.makedirs(base_folder, exist_ok=True)
        scans: pd.DataFrame = self.spikescans.drop_duplicates(subset=["image_dir", "row_lot", "range_lot"])
        scans = (
            scans.groupby(["row_lot", "range_lot"], as_index=False)
            .agg({"image_dir": list})
        )
        scans = scans.values.tolist()
        csv_path = os.path.join(base_folder, "vol_mapping.csv")
        tables = []
        seg_model = self.get_segmodel() if segment else None

        count_plants = 0

        for row_lot, range_lot, dirs in scans:
            plantindex = 0
            for scan in dirs:
                prec = self.precomputed[scan]
                t = {}
                for p in prec:
                    t.setdefault(p["cluster"], [])
                    t[p["cluster"]].append(p)
                prec = {}
                for x, y in t.items():
                    if len(y) >= min_image_per_plant:
                        prec[x] = y

                export = {}
                scan_plant_counter = 0
                for _, cluster in prec.items():
                    plant_id = f"{range_lot}_{row_lot}_{plantindex+15}"
                    export[plant_id] = {"volume": None, "boxes": []}
                    for box, i in zip(cluster, range(len(cluster))):
                        img_name = f"{plant_id}_{i}_b.jpg"
                        export[plant_id]["boxes"].append((img_name, box))
                    count_plants += 1
                    plantindex += 1
                    scan_plant_counter += 1
                    if count_plants >= num_plants or scan_plant_counter >= num_plants_per_scan:
                        break
                        
                table = self.export_plant_images(scan, base_folder, export, padding, box_size, seg_model)
                tables.append(table)
                if count_plants >= num_plants:
                    break
            if count_plants >= num_plants:
                    break

        table = pd.concat(tables, ignore_index=True)
        table.to_csv(csv_path, index=False)

    @staticmethod
    def labels2num2023mapping():
        return {
            "green": 1,
            "red": 10,
            "yellowgreen": 2,
            "black": 3,
            "silver": 4,
            "violet": 5,
            "yellow": 6,
            "brown": 7,
            "white": 8,
            "blue": 9,
        }
        
    @staticmethod
    def labels2num2024mapping():
        return {
            "green": 1,
            "red": 2,
            "yellowgreen": 3,
            "black": 4,
            "silver": 5,
            "violet": 6,
            "yellow": 7,
            "brown": 8,
            "white": 9,
            "blue": 10,
        }

    @staticmethod
    def inversemapping(mapping):
        return {x: y for y, x in mapping.items()}

    @staticmethod
    def labels2numgeneralmapping():
        return FIPDataset.labels2num2024mapping()


if __name__ == "__main__":
    config = {
            "annotation_file": r"F:\FIP-data\csv\labeled_spikes.csv",
            "csv_folder": r"F:\FIP-data\csv",
            "img_folder": r"F:\FIP-data\images",
            "ply_folder": r"F:\FIP-data\wheat-scans",
            "precompute_file": r"F:\FIP-data\csv\precomputed.json",
            "pose_folder": r"C:\Users\Admin\Desktop\master_thesis\volume_prediction_fip\assets\poses"
        }

    data = FIPDataset(config["csv_folder"], config["img_folder"], config["ply_folder"], config["precompute_file"], config["annotation_file"], config["pose_folder"])
    #data.spikescans.to_csv(r"F:\FIP-data\csv\fip_data_export.csv")
    #data.precompute_image_dataset(r"F:\Boxes-ds\spike_dataset_test", 20, False, segment=True)
    data.generate_unlabeled_spikes(r"F:\Boxes-ds\unlabeled", 1000, 10)
    #data.precompute_boxes(r"F:\FIP-data\csv\precomputed_test.json", update_connections_only=True)