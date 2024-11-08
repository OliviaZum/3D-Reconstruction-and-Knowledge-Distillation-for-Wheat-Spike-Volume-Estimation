import os
from ultralytics import YOLO
from ultralytics.engine.results import Results
import sys
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
import numpy as np

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

def convert_labelme_yolo(base, base_out):
    import cv2

    for p in os.listdir(base):
        if os.path.isfile(os.path.join(base, p)) and os.path.splitext(p)[1] == ".json":
            m = LabelMeImage(os.path.join(base, p))

            outer_box = np.array([10000, 10000, 0, 0])
            with open(os.path.join(base_out, os.path.splitext(p)[0] + ".txt"), "w") as f:
                imsize = np.array([m.data["imageWidth"], m.data["imageHeight"]])
                mapping = labels2numgeneralmapping()
                for category, box in m.boxes:
                    box = np.array(box)
                    outer_box[0:2] = np.minimum(outer_box[0:2], box[0:2])
                    outer_box[2:4] = np.maximum(outer_box[2:4], box[2:4])
                outer_box += np.array([-100, -100, 100, 100])
                outer_box = np.clip(outer_box, 0, [imsize[0], imsize[1], imsize[0], imsize[1]])
                boxsize = np.array([outer_box[2] - outer_box[0], outer_box[3] - outer_box[1]])
                offset = outer_box[0:2]
                for category, box in m.boxes:
                    box = np.array(box)
                    box[0:2] -= offset
                    box[2:4] -= offset
                    box[2:4] = box[2:4] - box[0:2]
                    box[0:2] += box[2:4] / 2
                    box[0:2] /= boxsize
                    box[2:4] /= boxsize
                    yolo_category = mapping[category] -1 # Yolo can only deal with id's from 0 to n for whatever reason
                    f.write(f"{yolo_category} {box[0]} {box[1]} {box[2]} {box[3]}\n")
            imname = os.path.splitext(p)[0] + ".png"
            img = cv2.imread(os.path.join(base, imname))
            cv2.imwrite(os.path.join(base_out, imname), img[outer_box[1]:outer_box[3], outer_box[0]:outer_box[2]])

if __name__ == "__main__":
    #convert_labelme_yolo(r"F:\Fip-Labels\LabelMe", r"F:\Fip-Labels\yolo")
    #exit()

    #convert_gwhd_yolo("F:/gwhd_2021/gwhd_2021", "F:/gwhd_2021/gwhd_yolo")

    #model = YOLO("C:/Users/Admin/Desktop/master_thesis/volume_prediction_fip/detection-yolo/weights/detect/medium-train+val/weights/best.pt", task='detect')
    model = YOLO(r"C:\Users\Admin\Desktop\master_thesis\volume_prediction_fip\detection-yolo\runs\detect\train2\weights\last.pt", task='detect')
    model.val(data=r"F:\FIP-coco-instance\yolo_spike\dataset.yaml", batch=12, show_labels=False, imgsz=800)

    #model = YOLO(r"C:\Users\Admin\Desktop\master_thesis\volume_prediction_fip\detection-yolo\runs\detect\train4\weights\best.pt")
    #model.train(data=r"F:\Fip-Labels\yolo\dataset.yaml", epochs=50, batch=1, amp=False, hsv_h=0, hsv_v=0, hsv_s=0, mosaic=0, multi_scale=True)

    #model.val(data="F:\\gwhd_2021\\gwhd_yolo\\dataset.yaml", batch=12, show_labels=False, split="test")

    #model.train(data="F:\\gwhd_2021\\gwhd_yolo\\dataset.yaml", epochs=150, batch=3, patience=20, amp=False, val=True, show_labels=False, single_cls=True, multi_scale=True)

    """
    
    
    model = YOLO("C:/Users/Admin/Desktop/master_thesis/FIPS-wheat-volume/segmentation-yolo/runs/segment/train-small/weights/best.pt", task='segment')

    model.val(data="F:\\SPIKE_segm\\YOLO_conv\\dataset.yaml", split='train')

    results: list[Results] = model("F:/FIP-data/2023/WW034/debayered/2023_06_08_13_11_Lot1/FPWW0340091_FIP2_20230608_122303/cam_03.png", imgsz=2000, device='cpu', max_det=500, conf=0.1)
    for result in results:

        result.plot(labels=False, probs=False, boxes=True, show=True, conf=True, color_mode='instance')
    
    """

    """
    results: list[Results] = model("F:/FIP-data/2023/WW034/debayered/2023_06_08_13_11_Lot1/FPWW0340091_FIP2_20230608_122303/cam_03.png", imgsz=4000, max_det=800, iou=0.5)
    for result in results:
        result.plot(labels=False, probs=False, boxes=True, show=False, conf=True, color_mode='instance', save=True, filename="output.jpg")
    """