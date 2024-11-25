import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

import json
import numpy as np
import re
import pandas as pd
import random

class FIPDataset:
    def __init__(self, csv_folder, img_folder, ply_folder, precompute_file = None, spikelabels_file = None) -> None:
        if precompute_file:
            self.precomputed = pd.read_csv(precompute_file, encoding="ISO-8859-1")
        else:
            self.precomputed = pd.DataFrame()
        
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
        self.poses = {}
        pose_folder = 'assets/poses'
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
            self.precomputed = pd.DataFrame()
            self.precomputed["image_dir"] = self.spikescans["image_dir"].unique()
        from fip_detection import detect
        import torch
        import tqdm
        def converter(obj):
            if isinstance(obj, np.int64):
                return int(obj)
            elif isinstance(obj, np.float32):
                return float(obj)
            
        for index, row in tqdm.tqdm(self.precomputed.iterrows()):
            conf = self.get_pose_config(row['image_dir'])
            if update_connections_only:
                w = json.loads(row['spikes'])
                boxes = {} # Ensure ids are unique
                for img, boxes_img in w.items():
                    n = {}
                    for i, (_, v) in enumerate(boxes_img):
                        n[i] = v
                    boxes[img] = n
                    print(len(n))
            else:
                boxes, _ = detect.find_objects_yolo("detection-yolo/weights/detect/medium-train+val/weights/best.pt", 
                                                row["image_dir"], batch_size=1)
            connected_boxes = detect.connect_boxes(boxes, conf, min_view=0, )
            if torch.cuda.memory_reserved() // (1024**2) > 3500:
                torch.cuda.empty_cache()
            self.precomputed.at[index, "spikes"] = json.dumps(connected_boxes, default=converter)
        self.precomputed.to_csv(precompute_file, sep=",", index=False)

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
    
    @staticmethod
    def extend_box(box, img, padding):
        box = np.array(box).astype(np.int32)
        box[0:2] = np.maximum([0, 0], box[0:2] - padding)
        box[2:4] = np.minimum(box[2:4] + padding, (img.shape[1], img.shape[0]))
        return img[box[1]:box[3], box[0]:box[2]]

    def precompute_image_dataset(self, base_folder: str, padding = 0, automatic_inferred=True, min_views = 6):
        # Compute an image dataset of the individual spikes
        import cv2
        import tqdm

        if os.path.exists(base_folder):
            input("The base folder exists. Press enter to overwrite...")
        os.makedirs(base_folder, exist_ok=True)
        csv_path = os.path.join(base_folder, "vol_mapping.csv")
        table = pd.DataFrame(columns=["img_name", "volume"])
        
        camkeys = [f"cam_{i:02}.png" for i in range(1, 13)]
        for folder in tqdm.tqdm(self.spikescans["image_dir"].unique()):
            scans: pd.DataFrame = self.spikescans.loc[self.spikescans["image_dir"] == folder]
            precomp_json = json.loads(self.precomputed.loc[self.precomputed["image_dir"] == folder, "spikes"].iloc[0])

            export = {}
            for _, scan in scans.iterrows():
                export[scan["plant_id"]] = {"volume": scan["spikevolume"], "boxes": {}}
                if automatic_inferred:
                    # Create branch of images which were automatically selected. The ground truth here is the box which was selected first
                    max_iou = 0
                    max_id = -1
                    for id, box in precomp_json[scan["label_selectedon"]]:
                        iou = self.iou(scan[scan["label_selectedon"]], box)
                        if iou > max_iou:
                            max_iou = iou
                            max_id = id
                    if max_iou < 0.6:
                        max_id = None
                        print(f"For the scan {scan["range_lot"]}_{scan["row_lot"]}, label {scan["label"]} no matching bounding box was found")

                for i, cam_name in enumerate(camkeys):
                    img_name = f"{scan["range_lot"]}_{scan["row_lot"]}_{scan["plant_id"]}_{i + 1}_a.jpg"

                    if not automatic_inferred:
                        # Write manual images (i.e. boxes which were selected by hand)
                        if scan[cam_name]:
                            export[scan["plant_id"]]["boxes"][cam_name] = (img_name, scan[cam_name])
                    else:
                        if max_id:
                            c = 0
                            for id, box in precomp_json[cam_name]:
                                if max_id == id:
                                    export[scan["plant_id"]]["boxes"][cam_name] = (img_name, box)
                                    break # Ignore the case where multiple boxes appear on one image for now
                                    c += 1
            # Load all images
            all_imgs = {}
            for img_name in camkeys:
                img = cv2.imread(os.path.join(folder, img_name))
                all_imgs[img_name] = img

            # Do the actual export
            for a in export.values():
                if len(a["boxes"]) < min_views:
                    continue
                
                for cam_name, box in a["boxes"].items():
                    new_row = {"img_name": box[0], "volume": a["volume"]}
                    out_path = os.path.join(base_folder, box[0])
                    img = FIPDataset.extend_box(box[1], all_imgs[cam_name], padding)
                    cv2.imwrite(out_path, img)
                    table.loc[len(table)] = new_row

            table.to_csv(csv_path, index=False)

    def generate_unlabeled_spikes(self, base_folder: str, max_images: int = None, padding: int = 0):
        folders = self.spikescans["image_dir"].unique()
        camkeys = [f"cam_{i:02}.png" for i in range(1, 13)]

        import cv2
        all_images = []
        for folder in folders:
            for cam in camkeys:
                all_images.append((folder, cam))
        random.seed(10)
        random.shuffle(all_images)
        
        exported = 0
        for folder, cam_name in all_images:
            precomp_json = json.loads(self.precomputed.loc[self.precomputed["image_dir"] == folder, "spikes"].iloc[0])
            img = cv2.imread(os.path.join(folder, cam_name))

            for _, box in precomp_json[cam_name]:
                box = FIPDataset.extend_box(box, img, padding)
                cv2.imwrite(os.path.join(base_folder, f"{exported}.jpg"), box)
                exported += 1
                if max_images and exported >= max_images:
                    return

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
            "precompute_file": r"F:\FIP-data\csv\precomputed.csv",
        }

    data = FIPDataset(config["csv_folder"], config["img_folder"], config["ply_folder"], config["precompute_file"], config["annotation_file"])
    #data.precompute_image_dataset(r"F:\Boxes-ds\spike_dataset_manual_0_pad_min6", 0, False)
    #data.precompute_image_dataset(r"F:\Boxes-ds\spike_dataset_manual_20_pad_min6", 20, False)
    data.generate_unlabeled_spikes(r"F:\Boxes-ds\unlabeled_spikes", 100000, 20)
    #data.precompute_boxes(config["precompute_file"])