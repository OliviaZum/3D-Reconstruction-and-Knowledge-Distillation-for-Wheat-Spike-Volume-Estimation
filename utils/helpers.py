import os
import numpy as np
import pyrender

def get_fip_campose(config, camera: str):
    e = config[camera]["extrinsics"]
    R = np.array(e["rotation"]).T
    c = e["center"]

    P = np.zeros((4, 4))
    P[0:3, 0:3] = R
    P[0:3, 3] = c
    P[3, 3] = 1

    Pnew = P @ np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])

    return Pnew

def get_fip_camintrinsics(config, camera: str, viewport_size: tuple):
    e = config[camera]["intrinsics"]
    w = e["width"]
    h = e["height"]
    f = e["focal_length"]
    distX = (viewport_size[0] / w)
    distY = (viewport_size[1] / h)
    fx = distX * f
    fy = distY * f
    pc = e["principal_point"] 
    cx = distX * pc[0]
    cy = distY * pc[1]

    return pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)

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

def get_ply_files(ply_dir_base):
    ply_files = []
    for root, _, files in os.walk(ply_dir_base):
        for file in files:
            if file.endswith('.ply'):
                full_path = os.path.join(root, file)
                ply_files.append(full_path)
    return ply_files

def get_crop_params(box_size, img):
    h, w = img.shape[:2]
    top = np.random.randint(0, max(h - box_size, 1))
    left = np.random.randint(0, max(w - box_size, 1))
    return left, top

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

def extend_image_box(box, img, padding, return_box = False):
    box = np.array(box).astype(np.int32)
    box[0:2] = np.maximum([0, 0], box[0:2] - padding)
    box[2:4] = np.minimum(box[2:4] + padding, (img.shape[1], img.shape[0]))
    sub_img = img[box[1]:box[3], box[0]:box[2]]
    if return_box:
        return sub_img, box
    else:
        return sub_img