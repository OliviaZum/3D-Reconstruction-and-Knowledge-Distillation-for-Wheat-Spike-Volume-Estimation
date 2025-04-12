"""
The tool used to annotate the spikes with labels.
Very much tailored to the specific use case, so probably not really reusable for anything else.
"""

import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox
import json
from volume_prediction_fip.fip_dataset import fip_dataset_manager
import pandas as pd
import os
import sys

class SaveDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Store references to label_dict, label_to_bbox_id, and image_dict from parent
        self.parent = parent
        self.result = None
        
        self.setup_ui()

    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Checkbox for "Fully Verified"
        self.verified_checkbox = QtWidgets.QCheckBox("Fully Verified")
        layout.addWidget(self.verified_checkbox)

        # Text field for "Comments"
        layout.addWidget(QtWidgets.QLabel("Comments:"))
        self.comments_field = QtWidgets.QLineEdit(self)
        layout.addWidget(self.comments_field)

        # List of labels and occurrences
        layout.addWidget(QtWidgets.QLabel("Label Occurrences:"))
        self.labels_list = QtWidgets.QListWidget(self)
        layout.addWidget(self.labels_list)

        # Save button
        self.save_button = QtWidgets.QPushButton("Save")
        self.save_button.clicked.connect(self.save_data)
        layout.addWidget(self.save_button)

        self.setWindowTitle("Save")
        self.setGeometry(100, 100, 400, 300)

    def showEvent(self, event):
        super().showEvent(event)
        self.populate_label_list()

    def populate_label_list(self):
        """ Populate the list with labels and their counts """
        self.labels_list.clear()
        for label in self.parent.label_dict:
            count = self.get_label_count(label)
            im = self.parent.label_to_setonimage.get(label)
            self.labels_list.addItem(f"{label}: {count} ({im})")

    def get_label_count(self, label):
        """ Count how many images each label appears in """
        count = 0

        # Check if label is present in label_to_bbox_id
        if label in self.parent.label_to_bbox_id:
            for image_name, boxes in self.parent.image_dict.items():
                for bbox_id, _ in boxes:
                    if bbox_id == self.parent.label_to_bbox_id[label]:
                        count += 1
        return count

    def save_data(self):
        """ Collect data and save it as JSON """
        result = {
            "verified": self.verified_checkbox.isChecked(),
            "comments": self.comments_field.text(),
        }

        # Iterate through image_dict and add the labels with bounding boxes
        for image_name, boxes in self.parent.image_dict.items():
            image_data = {}
            for bbox_id, box in boxes:
                # Find the label that corresponds to this bounding box
                for label, id_value in self.parent.label_to_bbox_id.items():
                    if id_value == bbox_id:
                        image_data[label] = box
            result[image_name] = image_data
        result["label_selectedon"] = self.parent.label_to_setonimage

        # Save to JSON
        json_output = json.dumps(result, indent=4)
        self.result = json_output
        print(json_output)  # Output the result, or save it to a file

        # Close the dialog after saving
        self.accept()

class ImageAnnotator(QtWidgets.QMainWindow):
    def __init__(self, image_folder, image_dict, label_dict, save_file=None):
        super().__init__()
        self.image_folder = image_folder
        self.image_dict = image_dict
        self.unique_id = 0
        for imbox in self.image_dict.values():
            for id, _ in imbox:
                if id > self.unique_id:
                    self.unique_id = id
        self.unique_id += 1
        self.label_dict = label_dict

        self.image_list = list(self.image_dict.keys())
        self.current_image_index = 0
        self.zoom_level = 1.0
        self.image = None
        self.bbox_to_label = {}
        self.label_to_bbox_id = {}
        self.label_to_setonimage = {} # On which Image was a label set globably?
        self.labeled_bounding_box_colors = {
            "green": (0, 255, 0),
            "red": (0, 0, 255),
            "yellowgreen": (0, 255, 170),
            "black": (255, 0, 255),
            "silver": (190, 190, 190),
            "violet": (190, 70, 120),
            "yellow": (0, 255, 255),
            "brown": (75, 75, 100),
            "white": (255, 255, 255),
            "blue": (255, 0, 0),
        }
        self.save_dialog = SaveDialog(self)
        if save_file:
            # Assign the bounding boxes in the predictions which have an iou of at least min_iou. (Can practically lead to dataloss if predictions change)
            # One could update this to insert a new bounding box if no matching one is found
            min_iou = 0.8
            for img in self.image_dict:
                for label, labelbox in save_file[img].items():
                    max_iou = 0
                    max_box = None
                    for box_id, box in self.image_dict[img]:
                        w = self.iou(labelbox, box)
                        if max_iou < w:
                            max_box = (box_id, box)
                            max_iou = w
                    if max_iou >= min_iou:
                        if label in self.label_to_bbox_id:
                            if self.label_to_bbox_id[label] == max_box[0]:
                                continue
                            self.swap_ids(max_box[0], label, img)
                        else:
                            self.label_to_bbox_id[label] = max_box[0]

            if "label_selectedon" in save_file:
                self.label_to_setonimage = save_file["label_selectedon"]
            else:
                print("Save file does not contain info about on which image a label was selected")
            
            self.save_dialog.verified_checkbox.setChecked(save_file["verified"])
            self.save_dialog.comments_field.setText(save_file["comments"])

        # Create central widget
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)

        # Create layout
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)

        # Create scene and view
        self.scene = QtWidgets.QGraphicsScene()
        self.view = QtWidgets.QGraphicsView(self.scene)
        self.view.setRenderHint(QtGui.QPainter.Antialiasing)
        self.view.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

        # Add view to layout
        self.layout.addWidget(self.view)

        # Add toolbar
        self.toolbar = QtWidgets.QToolBar(self)
        self.addToolBar(self.toolbar)
        self.save_action = self.toolbar.addAction("Save")
        self.save_action.triggered.connect(self.save)

        # Load the first image
        self.show_bounding_box = True
        self.show_labeled_only = False
        self.load_image()

        # Connect mouse wheel event for zooming
        self.is_panning = False
        self.last_mouse_pos = None
        self.view.wheelEvent = self.wheelEvent
        self.view.mousePressEvent = self.mousePressEvent
        self.view.mouseMoveEvent = self.mouseMoveEvent
        self.view.mouseReleaseEvent = self.mouseReleaseEvent

        # Connect key events
        self.view.keyPressEvent = self.keyPressEvent

    @staticmethod
    def iou(box1, box2):
        # box1 and box2 should be in (x1, y1, x2, y2) format
        x1, y1, x2, y2 = box1
        x1_b, y1_b, x2_b, y2_b = box2

        # Calculate the intersection area
        inter_x1 = max(x1, x1_b)
        inter_y1 = max(y1, y1_b)
        inter_x2 = min(x2, x2_b)
        inter_y2 = min(y2, y2_b)

        # Check if there is no overlap
        if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

        # Calculate the area of both the prediction and ground-truth boxes
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_b - x1_b) * (y2_b - y1_b)

        # Calculate the union area
        union_area = box1_area + box2_area - inter_area

        # Compute the IoU
        return inter_area / union_area

    def load_image(self):
        img_name = self.image_list[self.current_image_index]
        img_path = f"{self.image_folder}/{img_name}"
        self.image = cv2.imread(img_path)

        if self.image is None:
            QMessageBox.critical(self, "Error", f"Could not load image {img_name}")
            return

        self.display_image()

    def display_image(self):
        if self.image is None:
            return

        img_copy = self.image.copy()

        # Draw bounding boxes from image_dict
        if self.show_bounding_box:
            img_name = self.image_list[self.current_image_index]
            inverse_dict = {x: y for y, x in self.label_to_bbox_id.items()}
            for id, (x1, y1, x2, y2) in self.image_dict[img_name]:
                if self.show_labeled_only and id not in inverse_dict:
                    continue
                if id in inverse_dict:
                    color = self.labeled_bounding_box_colors[inverse_dict[id]]
                    text = f"ID: {id}, Label: {inverse_dict[id]}"
                else:
                    color = (0, 147, 255)
                    text = f"ID: {id}"
                self.draw_bbox(img_copy, (x1, y1, x2, y2), color, text)

        # Convert image to Qt format
        height, width, channel = img_copy.shape
        bytes_per_line = 3 * width
        q_img = QtGui.QImage(img_copy.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_BGR888)

        # Set image to scene
        self.scene.clear()
        self.scene.addPixmap(QtGui.QPixmap.fromImage(q_img))
        self.view.setScene(self.scene)

        return img_copy

    def draw_bbox(self, img, bbox, color, text):
        """Draw a bounding box on the image."""
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def save(self):
        self.save_dialog.exec_()
        if self.save_dialog.result:
            self.close()

    def wheelEvent(self, event):
        """Zoom in/out with the mouse wheel."""
        zoom_factor = 1.1
        if event.angleDelta().y() > 0:
            self.zoom_level *= zoom_factor  # Zoom in
        else:
            self.zoom_level /= zoom_factor  # Zoom out

        # Apply the zoom
        self.view.resetTransform()
        self.view.scale(self.zoom_level, self.zoom_level)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MiddleButton:
            self.is_panning = True
            self.last_mouse_pos = event.pos()
        elif event.button() == QtCore.Qt.LeftButton:
            pos = self.view.mapToScene(event.pos())
            found = False
            for id, (x1, y1, x2, y2) in self.image_dict[self.image_list[self.current_image_index]]:
                if x1 <= pos.x() <= x2 and y1 <= pos.y() <= y2:  # Check if clicked within bounding box
                    self.show_context_menu(event.pos(), id) 
                    found = True
            if not found:
                self.show_context_menu(event.pos(), None)


    def mouseMoveEvent(self, event):
        if self.is_panning and self.last_mouse_pos is not None:
            # Calculate the distance moved
            delta = event.pos() - self.last_mouse_pos
            self.view.horizontalScrollBar().setValue(self.view.horizontalScrollBar().value() - delta.x())
            self.view.verticalScrollBar().setValue(self.view.verticalScrollBar().value() - delta.y())
            self.last_mouse_pos = event.pos()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.MiddleButton:
            self.is_panning = False
            self.last_mouse_pos = None

    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() == QtCore.Qt.Key_Left:
            self.show_prev_image()
        elif event.key() == QtCore.Qt.Key_Right:
            self.show_next_image()
        elif event.key() == QtCore.Qt.Key.Key_S:
            self.show_bounding_box = not self.show_bounding_box
            self.display_image()
        elif event.key() == QtCore.Qt.Key.Key_P:
            self.show_labeled_only = not self.show_labeled_only
            self.display_image()
        elif event.key() == QtCore.Qt.Key.Key_1:
            img_copy = self.display_image()
            #cv2.imwrite(r"C:\Users\Admin\Desktop\master_thesis\volume_prediction_fip\local_stuff\training-results-detection\sample_preds_fip\export.png", img_copy)
            print("Written")

    def swap_ids(self, bbox_id, label, image_name):
        id_pre = self.label_to_bbox_id.get(label)
        if id_pre is None:
            return
        imgdict = self.image_dict[image_name]
        for i in range(len(imgdict)):
            id, _ = imgdict[i]
            if id == bbox_id: # If bbox_id is None, the current id will be changed, but no new box is assigned
                imgdict[i] = (id_pre, imgdict[i][1])
            elif id == id_pre:
                imgdict[i] = (self.unique_id, imgdict[i][1])
                self.unique_id += 1

    def update_label(self, bbox_id, label, new):
        if new: # Add or reset label for a particular id
            if bbox_id is None:
                if label in self.label_to_bbox_id:
                    self.label_to_bbox_id.pop(label)
                    self.label_to_setonimage.pop(label)
            else:
                if bbox_id in self.label_to_bbox_id.values():
                    oldlabel = [x for x, v in self.label_to_bbox_id.items() if v == bbox_id][0]
                    del self.label_to_bbox_id[oldlabel]
                    #self.label_to_bbox_id = {key: value for key, value in self.label_to_bbox_id.items() if value != bbox_id}
                    del self.label_to_setonimage[oldlabel]
                self.label_to_bbox_id[label] = bbox_id
                self.label_to_setonimage[label] = self.image_list[self.current_image_index]
        else: # Change the id of a box such that it matches the id of a label (and the old one doesn't anymore)
            self.swap_ids(bbox_id, label, self.image_list[self.current_image_index])

        self.display_image()

    def show_context_menu(self, position, bbox_id):
        context_menu = QtWidgets.QMenu(self)

        context_menu.addAction("Set new label" if bbox_id else "Reset label globably").setDisabled(True)
        for label in self.label_dict.keys():
            action = context_menu.addAction(f"{label} {'(Used)' if label in self.label_to_bbox_id else ''}")
            action.triggered.connect(lambda checked, color=label, id=bbox_id, new=True: self.update_label(id, color, new))
        context_menu.addAction("Update label on this image" if bbox_id else "Reset label on this image").setDisabled(True)
        for label in self.label_to_bbox_id.keys():
            action = context_menu.addAction(label)
            action.triggered.connect(lambda checked, color=label, id=bbox_id, new=False: self.update_label(id, color, new))

        context_menu.exec_(self.view.mapToGlobal(position))

    def show_next_image(self):
        """Show the next image."""
        if self.current_image_index < len(self.image_list) - 1:
            self.current_image_index += 1
            self.load_image()

    def show_prev_image(self):
        """Show the previous image."""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_image()


#"green","red","yellowgreen","black","silver","violet","yellow","brown","white","blue"

if __name__ == "__main__":
    config = {
        "annotation_file": r"F:\FIP-data\csv\labeled_spikes.csv",
        "csv_folder": r"F:\FIP-data\csv",
        "img_folder": r"F:\FIP-data\images",
        "ply_folder": r"F:\FIP-data\wheat-scans",
        "precompute_file": r"F:\FIP-data\csv\precomputed_new_setup.json",
        "show": lambda folder, data: True
    }

    data = fip_dataset_manager.FIPDataset(config["csv_folder"], config["img_folder"], config["ply_folder"], config["precompute_file"])

    if os.path.exists(config["annotation_file"]):
        annotations = pd.read_csv(config["annotation_file"])
    else:
        annotations = pd.DataFrame(columns=["image_dir", "labeled_spike"])
    app = QtWidgets.QApplication(sys.argv)

    for image_folder in data.spikescans["image_dir"].unique():
        rows = data.spikescans.loc[data.spikescans["image_dir"] == image_folder]
        current = annotations.loc[annotations["image_dir"] == image_folder, "labeled_spike"]
        if len(current) > 0:
            current = json.loads(current.iloc[0])
        else:
            current = None
        if not config["show"](image_folder, current):
            continue

        if image_folder not in data.precomputed:
            print("Somehow this key does not exist in the precompute file")
            continue
        # Annotation tool was written on an old format of storing boxes. For simplicity the new format is backconverted here
        # (instead of rewritting large parts)
        image_dict = {}
        for box in data.precomputed[image_folder]:
            image_dict.setdefault(box["image"], [])
            image_dict[box["image"]].append((box["cluster"], box["box"]))

        # Should probably solve this more clean at some point, but for now we impose the constraint that only one box has the same id here
        for img in image_dict:
            maxv = 0
            for id, box in image_dict[img]:
                if id > maxv:
                    maxv = id
            maxv += 1
            used_ids = set()
            for i, (id, box) in enumerate(image_dict[img]):
                if id in used_ids:
                    image_dict[img][i] = (maxv, box)
                    used_ids.add(maxv)
                    maxv += 1
                else:
                    used_ids.add(id)

        label_dict = {x: None for x in rows["label"]}

        annotator = ImageAnnotator(image_folder, image_dict, label_dict, current)
        first_row = rows.iloc[0]
        scan_info = f"Folder: {image_folder} Year: {first_row["year"]} Range: {first_row["range_lot"]} Row: {first_row["row_lot"]}"
        annotator.setWindowTitle(f"Folder: {image_folder} Year: {first_row["year"]} Range: {first_row["range_lot"]} Row: {first_row["row_lot"]}")
        print(scan_info)
        print(rows[["label", "plant_id", "ply"]])
        print("------------------\n")
        annotator.show()
        app.exec_()

        if annotator.save_dialog.result:
            if current:
                annotations.loc[annotations["image_dir"] == image_folder, "labeled_spike"] = annotator.save_dialog.result
            else:
                annotations = pd.concat((annotations, pd.DataFrame({"image_dir": image_folder, "labeled_spike": [annotator.save_dialog.result]})))
            annotations.to_csv(config["annotation_file"], index=False)
