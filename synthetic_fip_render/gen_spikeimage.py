import os
from typing import Literal
import trimesh
import pyrender
import numpy as np
import helpers
import cv2
import pandas as pd

class ArtificialImageGenerator(pyrender.Scene):
    def __init__(self, viewport_size: tuple) -> None:
        super().__init__(bg_color=np.zeros(3))
        self.renderer = pyrender.OffscreenRenderer(viewport_width=viewport_size[0], viewport_height=viewport_size[1])
        self.offscreencamera = pyrender.PerspectiveCamera(0.5 * np.pi, 0.05, 1000)
        pose = np.eye(4)
        pose[2, 3] = +3
        self.add(self.offscreencamera, pose=pose)
        self.active_spike = None
        self.sun = None

        """
        def create_camera_mesh():
            cone = trimesh.creation.cone(radius=0.1, height=0.5)
            cone.apply_translation([0, 0, 0.25])
            cone.apply_transform([
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ])
            return pyrender.Mesh.from_trimesh(cone)
        m = create_camera_mesh()
        self.add(m, pose=pose)
        """

    def set_spike(self, file_name: str):
        if self.active_spike is not None:
            self.remove_node(self.active_spike)
        mesh = trimesh.load(file_name)
        mesh.apply_translation(-mesh.center_mass)
        cm = pyrender.Mesh.from_trimesh(mesh, smooth=True)
        self.active_spike = self.add(cm, pose=np.eye(4))

    def random_light(self):
        if self.sun is not None:
            self.remove_node(self.sun)
        directional_light = pyrender.DirectionalLight(color=np.ones(3), intensity=np.random.uniform(3, 8))
        r = np.eye(4)
        rot = helpers.random_rotation_matrix(np.random.uniform(-0.7 * np.pi, 0.7 * np.pi), None)
        r[0:3, 0:3] = rot
        self.sun = self.add(directional_light, pose=r)
        self.ambient_light = np.array(np.random.uniform(0.1, 0.3) * np.ones(3))

    @staticmethod
    def pose_generator(rotation: Literal["best", "worst", "random"], distance_shift: Literal["none", "mild", "strong"], trans_base = np.zeros(3)):
        if rotation == "worst":
            rot_axis = np.array([0, 0, 1])
            rot_base = np.array([
                [1, 0, 0],
                [0, 0, 1],
                [0, -1, 0]
            ])
        elif rotation == "best":
            rot_axis = np.array([0, 1, 0])
            rot_base = np.eye(3)
        elif rotation == "random":
            rot_axis = None
            rot_base = np.eye(3)
        else:
            assert False, f"{rotation} parameter is invalid"

        scale_base = np.eye(3) * 0.04
        if distance_shift == "none":
            trans_shift = np.zeros(3)
        elif distance_shift == "mild":
            trans_shift = np.array([0, 0, -np.random.uniform(0, 1)])
        elif distance_shift == "strong":
            trans_shift = np.array([0, 0, -np.random.uniform(0, 5)])
        else:
            assert False, f"{distance_shift} parameter is invalid"

        angle = np.random.uniform(0, 2 * np.pi)
        rand_rot = helpers.random_rotation_matrix(angle, rot_axis)
        
        return helpers.construct_pose_scale(
            rot_base,
            trans_base,
            scale_base,
            rand_rot,
            1.0,
            trans_shift
        )
    
    def generate(self, base_dir_in: str, dir_out: str, n_images_per_plant: int, get_pose, num_plants: int = None, cut_spike: bool=False):
        os.makedirs(dir_out, exist_ok=False)

        ply_files = helpers.get_ply_files(base_dir_in)
        img_names = []
        img_volumes = []
        for p_num, ply_file in enumerate(ply_files):
            folder, ply_file = os.path.split(ply_file)
            plant_id = os.path.splitext(ply_file)[0]
            vol_file = f"{plant_id}.txt"

            with open(os.path.join(folder, vol_file)) as f:
                volume = float(f.read())
                if volume <= 0:
                    continue
            
            self.set_spike(os.path.join(folder, ply_file))

            for i in range(n_images_per_plant):
                self.random_light()
                self.set_pose(self.active_spike, get_pose())
                img, _ = self.renderer.render(self)
                img_name = f"{plant_id}_{i}_b.jpg"
                img_names.append(img_name)
                img_volumes.append(volume)
                if cut_spike:
                    mask = img == 0
                    mask = ~np.all(mask, axis=2)
                    coords = np.nonzero(mask)
                    if len(coords[0]) == 0:
                        print(f"{plant_id} has an empty mask on img {i}")
                    else:
                        y_min, x_min = coords[0].min().item(), coords[1].min().item()
                        y_max, x_max = coords[0].max().item(), coords[1].max().item()
                        img = img[y_min:y_max + 1, x_min:x_max + 1, :]

                cv2.imwrite(os.path.join(dir_out, img_name), img)
                
                    
            if num_plants is not None and p_num >= num_plants:
                break
            
        df = pd.DataFrame()
        df["img_name"] = img_names
        df["volume"] = img_volumes
        df.to_csv(os.path.join(dir_out, "vol_mapping.csv"), index=False)


np.random.seed(0)
a = ArtificialImageGenerator((224, 224))

try:
    a.generate(r"F:\FIP-data\wheat-scans", r"F:\Boxes-ds\artifical\bestpose_mildshift_12_cut", 12, lambda: ArtificialImageGenerator.pose_generator("best", "mild"), cut_spike=True)
except:
    traceback.print_exception()
exit(0)

import traceback
try:
    a.generate(r"F:\FIP-data\wheat-scans", r"F:\Boxes-ds\artifical\bestpose_constant_far_12", 12, lambda: ArtificialImageGenerator.pose_generator("best", "none", trans_base=np.array([0, 0, -5])))
except:
    traceback.print_exception()

try:
    a.generate(r"F:\FIP-data\wheat-scans", r"F:\Boxes-ds\artifical\randompose_strongshift_cut_12", 12, lambda: ArtificialImageGenerator.pose_generator("random", "strong"), cut_spike=True)
except:
    traceback.print_exception()

try:
    a.generate(r"F:\FIP-data\wheat-scans", r"F:\Boxes-ds\artifical\bestpose_noshift_6", 6, lambda: ArtificialImageGenerator.pose_generator("best", "none"))
except:
    traceback.print_exception()
try:
    a.generate(r"F:\FIP-data\wheat-scans", r"F:\Boxes-ds\artifical\bestpose_noshift_12", 12, lambda: ArtificialImageGenerator.pose_generator("best", "none"))
except:
    traceback.print_exception()

try:
    a.generate(r"F:\FIP-data\wheat-scans", r"F:\Boxes-ds\artifical\worstpose_noshift_12", 12, lambda: ArtificialImageGenerator.pose_generator("worst", "none"))
except:
    traceback.print_exception()

try:
    a.generate(r"F:\FIP-data\wheat-scans", r"F:\Boxes-ds\artifical\randompose_noshift_12", 12, lambda: ArtificialImageGenerator.pose_generator("random", "none"))
except:
    traceback.print_exception()

try:
    a.generate(r"F:\FIP-data\wheat-scans", r"F:\Boxes-ds\artifical\bestpose_strongshift_12", 12, lambda: ArtificialImageGenerator.pose_generator("best", "strong"))
except:
    traceback.print_exception()
try:
    a.generate(r"F:\FIP-data\wheat-scans", r"F:\Boxes-ds\artifical\bestpose_strongshift_cut_12", 12, lambda: ArtificialImageGenerator.pose_generator("best", "strong"), cut_spike=True)
except:
    traceback.print_exception()