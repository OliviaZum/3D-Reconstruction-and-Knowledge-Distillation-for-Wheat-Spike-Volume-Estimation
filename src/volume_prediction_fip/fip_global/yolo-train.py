"""
Contains the functions used for converting datasets to YOLO,
label the GWHD spikes using SAM and train the models. There is no validation 
code, but validation should run automatically at the end of training, and 
one can always manually validate a model on some split using .val method,
E.g. 
seg_model.val(data=r"F:\SPIKE_segm\YOLO_conv\dataset.yaml", split="train", imgsz=1024, batch=3)
to evaluate seg_model on the train split of SPIKE.
For inference, this is an example:
#r: List[Results] = det_model(r"F:\gwhd_2021\gwhd_yolo\test\images\7f16a775a5d8cdb7e16fb1cacb3248552558634bb6dba7c58626f7558388d2be.png",
#              imgsz=640, max_det=800, iou=0.5, conf=0.25)
"""

import os
from typing import List
from ultralytics import YOLO
from ultralytics.engine.results import Results
import torch
import numpy as np
from ultralytics import SAM
from ultralytics.data.converter import convert_coco
from pathlib import Path
import shutil

# Converts GWHD the YOLO detection format
def convert_gwhd_yolo(base: str, base_out: str):
    import os
    import csv
    import cv2
    import tqdm

    def parse_boxes_string(boxes_string):
        boxes = boxes_string.split(';')
        boxes_tuples = []
        for box in boxes:
            if box.strip():
                if (box == "no_box"):
                    break
                x1, y1, x2, y2 = map(int, box.split())
                boxes_tuples.append((x1, y1, x2, y2))
        return boxes_tuples

    def read_bbox_file(csvreader):
        for row in csvreader:
            image_name = row['image_name']
            boxes_string = row['BoxesString']
            boxes = parse_boxes_string(boxes_string)

            yield image_name, boxes

    splits = ["val", "test", "train"]

    os.makedirs(base_out)

    for split in splits:
        csv_file = os.path.join(base, f"competition_{split}.csv")
        img_folder = os.path.join(base, "images")

        img_out_dir = os.path.join(base_out, split, "images")
        labels_out_dir = os.path.join(base_out, split, "labels")
        os.makedirs(img_out_dir)
        os.makedirs(labels_out_dir)

        with open(csv_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for image_name, boxes in tqdm.tqdm(read_bbox_file(reader), split):
                
                img_path = os.path.join(img_folder, image_name)
                if not os.path.exists(img_path):
                    print(f"Image or mask not found for {image_name}")
                    continue

                img = cv2.imread(img_path) # This is always 1024x1024 apparently
                height, width, _ = img.shape

                cv2.imwrite(os.path.join(img_out_dir, image_name), img)
                with open(os.path.join(labels_out_dir, image_name.rsplit(".")[0] + ".txt"), "w") as f:
                    for x1, y1, x2, y2 in boxes:
                        f.write(f"0 {((x2 + x1) * 0.5)/ width} {((y1 + y2) * 0.5) / height} {(x2 - x1) / width} {(y2 - y1) / height}\n")

def get_boxes_autosegment(fname):
            with open(fname) as f:
                s = f.readlines()
            boxes = []
            for w in s:
                u = w.split(" ")
                boxes.append(list(map(lambda x: float(x), u[1:])))
            if len(boxes) == 0:
                boxes = np.zeros((0, 4))
            else:
                boxes = np.array(boxes)
                boxes *= 1024
                boxes[:, 0:2] -= boxes[:, 2:] / 2
                boxes[:, 2:] += boxes[:, 0:2]
                boxes = boxes.astype(np.int32)
            return boxes

# Create instance segmentation masks for gwhd using SAM
def auto_segment_gwhd(input_path_base: str = r"F:\gwhd_2021\gwhd_yolo", output_path_base: str = r"F:\gwhd_2021\gwhd_yolo_automask",
                        visualize_only=False):
    subdirs = ["train", "test", "val"]
    model = SAM("sam2_s.pt")

    for dir in subdirs:
        input_path = os.path.join(input_path_base, dir)
        output_path = os.path.join(output_path_base, dir)
        files = os.listdir(os.path.join(input_path, "labels"))

        for file in files:
            label_path = os.path.join(input_path, "labels", file)
            img_path = os.path.join(input_path, "images", os.path.splitext(file)[0] + ".png")
            current_output_path = os.path.join(output_path, "labels", file)
            boxes = get_boxes_autosegment(label_path)
            if boxes.shape[0] == 0:
                if visualize_only:
                    continue
                with open(current_output_path, "w") as f:
                    pass
            else:
                r = model(img_path, bboxes=boxes)
                if visualize_only:
                    print(f"Image: {img_path}")
                    #r[0].show(boxes=False)
                    r[0].plot(boxes=False, show=True)
                    input("Press enter for next")
                    continue
                segments = r[0].masks.xyn
                with open(current_output_path, "w") as f:
                    for i in range(len(segments)):
                        s = segments[i]
                        if len(s) == 0:
                            continue
                        segment = map(str, segments[i].reshape(-1).tolist())
                        f.write(f"0 " + " ".join(segment) + "\n")
                torch.cuda.empty_cache()


# converts the spike dataset to YOLO format (coco conversion)
def convert_SPIKE_seg_yolo(dir_in=Path(r"F:\SPIKE_segm\SPIKE_main"), dir_out=Path(r"F:\SPIKE_segm\YOLO_conv")):
    # Using spike seg from https://figshare.com/articles/journal_contribution/SPIKE_segmentation_dataset/22347037?file=39760753
    convert_coco(dir_in / "annotations", save_dir=dir_out / "tmp", use_segments=True) # Does not really work here, anyhow

    target_dirs = ["test", "train", "val"]
    for target in target_dirs:
        (dir_out / target / "images").mkdir(parents=True, exist_ok=True)
        (dir_out / target / "labels").mkdir(parents=True, exist_ok=True)

        src_labels = dir_out / "tmp" / "labels" / f"{target}_coco"
        for file in src_labels.iterdir():
            shutil.move(str(file), str(dir_out / target / "labels"))
        
        src_images = dir_in / target
        for file in src_images.iterdir():
            shutil.copy(str(file), str(dir_out / target / "images"))

        dataset_yaml = f"""path: {dir_out.as_posix()}
train: train/
val: val/
test: test/

names:
    0: spike
"""
    (dir_out / "dataset.yaml").write_text(dataset_yaml)

def train_instance_segmentation_SPIKE(dataset=Path(r"F:\SPIKE_segm\tmp_test\dataset.yaml")):
    seg_model = YOLO("yolo11m-seg")
    seg_model.train(data=dataset, epochs=50, batch=1, patience=20, amp=False, val=True, show_boxes=False, single_cls=True, workers=1, 
                close_mosaic=0, imgsz=640, scale=0.5, mosaic=False)
    
def train_instance_segmentation_GWHDSAM(dataset=Path(r"F:\gwhd_2021\gwhd_yolo_automask\dataset.yaml")):
    seg_model = YOLO("yolo11m-seg")
    seg_model.train(data=dataset, epochs=15, batch=2, patience=5, amp=False, val=True, show_boxes=False, single_cls=True, workers=1, 
                close_mosaic=2, imgsz=640, scale=0.5)
    
def train_object_detection_GWHD(dataset=Path(r"F:\gwhd_2021\gwhd_yolo\dataset.yaml")):
    det_model = YOLO("yolo11m")
    det_model.train(data=dataset, epochs=150, patience=20, batch=3, imgsz=640, val=True, scale=0.5)


if __name__ == "__main__":
    os.environ['WANDB_DISABLED'] = 'true'

