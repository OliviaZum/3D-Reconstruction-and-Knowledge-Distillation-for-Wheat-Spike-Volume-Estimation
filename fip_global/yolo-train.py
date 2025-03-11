import os
from ultralytics import YOLO
from ultralytics.engine.results import Results
import sys
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
import torch
import numpy as np
from ultralytics import SAM

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

# Create instance segmentation masks for gwhd using SAM
def auto_segment_gwhd(input_path_base: str, output_path_base: str):
    subdirs = ["train", "test", "val"]
    model = SAM("sam2_s.pt")

    for dir in subdirs:
        input_path = os.path.join(input_path_base, dir)
        output_path = os.path.join(output_path_base, dir)
        def get_boxes(fname):
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
        files = os.listdir(os.path.join(input_path, "labels"))

        for file in files:
            label_path = os.path.join(input_path, "labels", file)
            img_path = os.path.join(input_path, "images", os.path.splitext(file)[0] + ".png")
            current_output_path = os.path.join(output_path, "labels", file)
            boxes = get_boxes(label_path)
            if boxes.shape[0] == 0:
                with open(current_output_path, "w") as f:
                    pass
            else:
                r = model(img_path, bboxes=boxes)
                segments = r[0].masks.xyn
                with open(current_output_path, "w") as f:
                    for i in range(len(segments)):
                        s = segments[i]
                        if len(s) == 0:
                            continue
                        segment = map(str, segments[i].reshape(-1).tolist())
                        f.write(f"0 " + " ".join(segment) + "\n")
                torch.cuda.empty_cache()

if __name__ == "__main__":

    # TODO: clean this up
    convert_gwhd_yolo("F:/gwhd_2021/gwhd_2021", "F:/gwhd_2021/gwhd_yolo")
    auto_segment_gwhd(r"F:\gwhd_2021\gwhd_yolo\val", r"F:\gwhd_2021\gwhd_yolo_automask")

    model = YOLO("yolo11m.pt")
    model.train(data="F:\\gwhd_2021\\gwhd_yolo\\dataset.yaml", epochs=150, batch=3, patience=20, amp=False, val=True, show_labels=False, single_cls=True, multi_scale=True)
    seg_model = YOLO("yolo11m-seg")
    seg_model.train(data="F:\\gwhd_2021\\gwhd_yolo_automask\\dataset.yaml", epochs=50, batch=2, patience=20, amp=False, val=True, show_boxes=False, single_cls=True, workers=3, 
                close_mosaic=0, imgsz=500, scale=0.5)