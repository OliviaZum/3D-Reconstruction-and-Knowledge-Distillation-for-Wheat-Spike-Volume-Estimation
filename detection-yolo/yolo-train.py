from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG, RANK
import torch
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.loss import v8DetectionLoss
import torch.nn.functional as F

def convert_gwhd_yolo(base, base_out):
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

if __name__ == "__main__":
    #convert_gwhd_yolo("F:/gwhd_2021/gwhd_2021", "F:/gwhd_2021/gwhd_yolo")

    model = YOLO("C:/Users/Admin/Desktop/master_thesis/volume_prediction_fip/detection-yolo/weights/detect/medium-train+val/weights/best.pt", task='detect')

    #model.val(data="F:\\gwhd_2021\\gwhd_yolo\\dataset.yaml", batch=12, show_labels=False, split="test")

    #model.train(data="F:\\gwhd_2021\\gwhd_yolo\\dataset.yaml", epochs=150, batch=3, patience=20, amp=False, val=True, show_labels=False, close_mosaic=0)

    """
    
    
    model = YOLO("C:/Users/Admin/Desktop/master_thesis/FIPS-wheat-volume/segmentation-yolo/runs/segment/train-small/weights/best.pt", task='segment')

    model.val(data="F:\\SPIKE_segm\\YOLO_conv\\dataset.yaml", split='train')

    results: list[Results] = model("F:/FIP-data/2023/WW034/debayered/2023_06_08_13_11_Lot1/FPWW0340091_FIP2_20230608_122303/cam_03.png", imgsz=2000, device='cpu', max_det=500, conf=0.1)
    for result in results:

        result.plot(labels=False, probs=False, boxes=True, show=True, conf=True, color_mode='instance')
    
    """

    results: list[Results] = model("F:/FIP-data/2023/WW034/debayered/2023_06_08_13_11_Lot1/FPWW0340091_FIP2_20230608_122303/cam_03.png", imgsz=4000, max_det=800, iou=0.5)
    for result in results:
        result.plot(labels=False, probs=False, boxes=True, show=False, conf=True, color_mode='instance', save=True, filename="output.jpg")