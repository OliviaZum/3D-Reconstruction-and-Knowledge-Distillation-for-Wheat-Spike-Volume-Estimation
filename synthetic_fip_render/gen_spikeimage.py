import os
from typing import Literal
import trimesh
import pyrender
import numpy as np
import helpers
import cv2
import pandas as pd
import traceback

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
    
    def generate(self, base_dir_in: str, dir_out: str, n_images_per_plant: int, get_pose, num_plants: int = None, cut_spike: bool=False, ignore_weird_samples = True):
        os.makedirs(dir_out, exist_ok=False)

        ply_files = helpers.get_ply_files(base_dir_in)
        # First three have a total different scale and produce black images. Last one appears twice (different spike, same name)
        weird_stuff = set(["2_10_8", "10_9_5", "11_8_9", "6_9_1"])
        img_names = []
        img_volumes = []
        for p_num, ply_file in enumerate(ply_files):
            folder, ply_file = os.path.split(ply_file)
            plant_id = os.path.splitext(ply_file)[0]
            if ignore_weird_samples and plant_id in weird_stuff:
                continue
            vol_file = f"{plant_id}.txt"

            with open(os.path.join(folder, vol_file)) as f:
                volume = float(f.read())
                if volume <= 0:
                    continue
            
            self.set_spike(os.path.join(folder, ply_file))
            plant_changed = True

            for i in range(n_images_per_plant):
                self.random_light()
                self.set_pose(self.active_spike, get_pose(plant_changed))
                plant_changed = False
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

class PoseGenerator:
    def __init__(self) -> None:
        self.constrained_rot_base = np.eye(3)
        self.fip_like_dist_base = 0

    def get(self, rotation: Literal["best", "worst", "random", "constrained"], distance_shift: Literal["none", "mild", "strong", "fiplike"], trans_base = np.zeros(3), plant_changed = False):
        if rotation == "worst":
            rot_axis = np.array([0, 0, 1])
            rot_base = np.array([
                [1, 0, 0],
                [0, 0, 1],
                [0, -1, 0]
            ])
            angle = np.random.uniform(0, 2 * np.pi)
        elif rotation == "best":
            rot_axis = np.array([0, 1, 0])
            rot_base = np.eye(3)
            angle = np.random.uniform(0, 2 * np.pi)
        elif rotation == "random":
            rot_axis = None
            rot_base = np.eye(3)
            angle = np.random.uniform(0, 2 * np.pi)
        elif rotation == "constrained":
            rot_axis = None
            if plant_changed:
                rot_base = helpers.random_rotation_matrix(np.random.uniform(0, 2 * np.pi))
                self.constrained_rot_base = rot_base
            else:
                rot_base = self.constrained_rot_base
            angle = np.random.uniform(0, 0.15 * np.pi)
        else:
            assert False, f"{rotation} parameter is invalid"

        # Roughly set to match the size of images from the fip
        scale_base = np.eye(3) * 0.0398
        if distance_shift == "none":
            trans_shift = np.zeros(3)
        elif distance_shift == "mild":
            trans_shift = np.array([0, 0, -np.random.uniform(0, 1)])
        elif distance_shift == "strong":
            trans_shift = np.array([0, 0, -np.random.uniform(0, 5)])
        elif distance_shift == "fiplike":
            if (trans_base != 0).any():
                assert False, "Do not use fiplike distance with translation"
            trans_base = np.array([0, 0, 3])
            if plant_changed:
                self.fip_like_dist_base = -np.clip(np.random.normal(3, 0.3), 2, 4)
            img_shift = - np.random.uniform(0, 0.5)
            trans_shift = np.array([0, 0, self.fip_like_dist_base + img_shift])
        else:
            assert False, f"{distance_shift} parameter is invalid"

        rand_rot = helpers.random_rotation_matrix(angle, rot_axis)
        
        return helpers.construct_pose_scale(
            rot_base,
            trans_base,
            scale_base,
            rand_rot,
            1.0,
            trans_shift
        )


np.random.seed(0)
size = 224
a = ArtificialImageGenerator((size, size))

global_pose = PoseGenerator()

a.generate(r"F:\FIP-data\wheat-scans", r"F:\Boxes-ds\artifical_ext_random_fiplike", 24, lambda plant_changed: global_pose.get("random", "fiplike", plant_changed=plant_changed), cut_spike=False)

#a.generate(r"F:\FIP-data\wheat-scans", r"F:\Boxes-ds\artifical\constrained_noshift_12", 12, lambda plant_changed: global_pose.get("constrained", "none", plant_changed=plant_changed), cut_spike=False)

generate_artifical_sets = False
if not generate_artifical_sets:
    exit(0)

try:
    a.generate(r"F:\FIP-data\wheat-scans", r"F:\Boxes-ds\artifical\randompose_mildshift_12_cut", 12, lambda _: global_pose.get("best", "none"), cut_spike=True)
except:
    traceback.print_exception()

try:
    a.generate(r"F:\FIP-data\wheat-scans", r"F:\Boxes-ds\artifical\bestpose_constant_far_12", 12, lambda _: global_pose.get("best", "none", trans_base=np.array([0, 0, -5])))
except:
    traceback.print_exception()

try:
    a.generate(r"F:\FIP-data\wheat-scans", r"F:\Boxes-ds\artifical\randompose_strongshift_cut_12", 12, lambda _: global_pose.get("random", "strong"), cut_spike=True)
except:
    traceback.print_exception()

try:
    a.generate(r"F:\FIP-data\wheat-scans", r"F:\Boxes-ds\artifical\bestpose_noshift_6", 6, lambda _: global_pose.get("best", "none"))
except:
    traceback.print_exception()
try:
    a.generate(r"F:\FIP-data\wheat-scans", r"F:\Boxes-ds\artifical\bestpose_noshift_12", 12, lambda _: global_pose.get("best", "none"))
except:
    traceback.print_exception()

try:
    a.generate(r"F:\FIP-data\wheat-scans", r"F:\Boxes-ds\artifical\worstpose_noshift_12", 12, lambda _: global_pose.get("worst", "none"))
except:
    traceback.print_exception()

try:
    a.generate(r"F:\FIP-data\wheat-scans", r"F:\Boxes-ds\artifical\randompose_noshift_12", 12, lambda _: global_pose.get("random", "none"))
except:
    traceback.print_exception()

try:
    a.generate(r"F:\FIP-data\wheat-scans", r"F:\Boxes-ds\artifical\bestpose_strongshift_12", 12, lambda _: global_pose.get("best", "strong"))
except:
    traceback.print_exception()
try:
    a.generate(r"F:\FIP-data\wheat-scans", r"F:\Boxes-ds\artifical\bestpose_strongshift_cut_12", 12, lambda _: global_pose.get("best", "strong"), cut_spike=True)
except:
    traceback.print_exception()