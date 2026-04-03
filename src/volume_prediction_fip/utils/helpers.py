import os
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Returns a random rotation matrix. Axis of rotation is choosen randomly if not fixed, the angle has to be defined.
def random_rotation_matrix(angle: float, fixed_rotation_axis = None) -> np.ndarray:
    # Generate a random unit vector (axis of rotation)
    if fixed_rotation_axis is not None:
        axis = fixed_rotation_axis / np.linalg.norm(fixed_rotation_axis)
    else:
        axis = np.random.normal(size=3)
        axis /= np.linalg.norm(axis)

    # Compute the components of the rotation matrix using Rodrigues' rotation formula
    K = np.array([[    0, -axis[2],  axis[1]],
                  [ axis[2],     0, -axis[0]],
                  [-axis[1],  axis[0],     0]])
    
    rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    
    return rotation_matrix

# Returns all ply files in some directory
def get_ply_files(ply_dir_base):
    ply_files = []
    for root, _, files in os.walk(ply_dir_base):
        for file in files:
            if file.endswith('.ply'):
                full_path = os.path.join(root, file)
                ply_files.append(full_path)
    return ply_files

# Returns parameters on how to crop a box if it does not fit on a patch (use for put_image_on_patch)
# The idea here is that if we have multiple similar observations which all do not fit properly there should
# at least be some randomness in placing such that different parts of the spikes are visible. Spikes rarely do not 
# fit, and if they do not it is usually not of by much.
def get_crop_params(box_size, img):
    h, w = img.shape[:2]
    top = np.random.randint(0, max(h - box_size, 1))
    left = np.random.randint(0, max(w - box_size, 1))
    return left, top

# Creates a patch of size box_size x box_size and puts img centered on it.
# If img is to large to fit it is cropped as defined by the crop_params (see get_crop_params)
# The returned box indicates where on the patch img was put.
def put_image_on_patch(box_size, img, crop_params):
    if len(img.shape) != 3:
        pshape = (box_size, box_size)
    else:
        pshape = (box_size, box_size, 3)
    patch = np.zeros(pshape, dtype=img.dtype)
    
    left, top = crop_params
    h, w = img.shape[:2]
    img = img[top:top + min(h, box_size), left:left + min(w, box_size)]
    h, w = img.shape[:2]

    start_y = (box_size - h) // 2
    start_x = (box_size - w) // 2
    patch[start_y:start_y + h, start_x:start_x + w] = img
    box = np.array([-start_x + left, -start_y + top, -start_x + w + left, -start_y + h + left])
    return patch, box

# Returns a cropped box of an image with some customizable padding.
def extend_image_box(box, img, padding, return_box = False):
    box = np.array(box).astype(np.int32)
    box[0:2] = np.maximum([0, 0], box[0:2] - padding)
    box[2:4] = np.minimum(box[2:4] + padding, (img.shape[1], img.shape[0]))
    sub_img = img[box[1]:box[3], box[0]:box[2]]
    if return_box:
        return sub_img, box
    else:
        return sub_img
    
# Construct a rotation + translation + scale matrix given basic transformations and a second set of additional transformations to add
def construct_pose_scale(rotation_base: np.ndarray,
                      translate_base: np.ndarray,
                      scale_base: np.ndarray,
                      rotation_add: np.ndarray | None,
                      scale_add: float | None,
                      translate_add: np.ndarray | None) -> np.ndarray:
    if scale_add is None:
        scale_mat = scale_base
    else:
        scale_mat = scale_add * scale_base
    
    if rotation_add is None:
        rotation_add = np.eye(3)
    R = rotation_add @ scale_mat @ rotation_base
    if translate_add is None:
        t = translate_base
    else:
        t = translate_base + translate_add
    Rt = np.eye(4)
    Rt[0:3, 0:3] = R
    Rt[0:3, 3] = t
    return Rt

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

def get_package_folder():
    return Path(__file__).parent.parent.parent.parent

def get_assets_path():
    return get_package_folder() / "assets"

def get_exp_path():
    return get_package_folder() / "local_stuff_experiments"

def get_segmentation_model():
    return YOLO(get_assets_path() / "model-weights" / "yolo-medium-segment.pt")

def get_detection_model():
    #return YOLO(get_assets_path() / "model-weights" / "yolo-medium-detect-mAp50-0766.pt")
    return YOLO(get_assets_path() / "model-weights" / "best_yolo11l_40ep.pt")

def get_pose_file_path():
    return get_assets_path() / "poses"

def get_volume_model_path():
    return get_assets_path() / "model-weights" / "self-distill-regulated_transformer.pth"