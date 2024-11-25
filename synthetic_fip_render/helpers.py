import os
from typing import Tuple
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
                      rotation_add: np.ndarray,
                      scale_add: float,
                      translate_add) -> np.ndarray:
    scale_mat = scale_add * scale_base
    R = rotation_add @ scale_mat @ rotation_base
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