import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

import json
import numpy as np
import re
import pandas as pd

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
                    
                    new_rows.append(new_row)

            df = pd.DataFrame(new_rows)
            before = len(df)
            df = df.dropna(subset=camkeys, how='all')
            print(f"Ignoring {before - len(df)} of {before} spikes because no label is set")

            self.spikescans = pd.merge(self.spikescans, df, how='inner', on=["image_dir", "label"])

    def precompute(self, precompute_file):
        # Precompute bounding boxes
        self.precomputed = pd.DataFrame()
        self.precomputed["image_dir"] = self.spikescans["image_dir"].unique()
        from fip_detection import detect
        import tqdm
        with open("assets/fip_poses_configuration.json") as f:
            conf = json.load(f)
        def converter(obj):
            if isinstance(obj, np.int64):
                return int(obj)
            elif isinstance(obj, np.float32):
                return float(obj)
            
        for index, row in tqdm.tqdm(self.precomputed.iterrows()):
            boxes, _ = detect.find_objects("detection-yolo/weights/detect/medium-train+val/weights/best.pt", 
                                            conf,
                                            row["image_dir"], batch_size=1, min_view=0)
            self.precomputed.at[index, "spikes"] = json.dumps(boxes, default=converter)
        self.precomputed.to_csv(precompute_file, sep=",", index=False)

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
