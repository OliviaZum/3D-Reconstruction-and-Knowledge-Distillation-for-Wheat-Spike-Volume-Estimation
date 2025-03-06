import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

import trimesh
import pyrender
import numpy as np
import json
import random
import os
import tqdm
import uuid
from utils.helpers import get_fip_camintrinsics, get_fip_campose, random_rotation_matrix, construct_pose_scale, get_ply_files

class FIPScene(pyrender.Scene):
    def __init__(self, conf_path: str, viewport_size: tuple, realistic = False, until_borders = True,
                    nodes=None, bg_color=(0, 0, 0), name=None):
        self.realistic = realistic
        self.until_borders = until_borders
        if realistic:
            ambient_light=[0.1, 0.1, 0.1]
        else:
            ambient_light=[1.0, 1.0, 1.0]
        super().__init__(nodes, bg_color, ambient_light, name)

        with open(conf_path) as f:
            self.conf = json.load(f)

        self.base_viewport_size = (4096, 3000)
        self.viewport_size = viewport_size
        self.viewport_normalization_factor = np.array(self.base_viewport_size) / np.array(viewport_size)
        self.renderer = pyrender.OffscreenRenderer(viewport_width=viewport_size[0], viewport_height=viewport_size[1])
        self.init_cameras()

        if self.realistic:
            self.sun = None
            self.random_light()

        self.scene_id = uuid.uuid4()


    def make_image(self, camera_name: str):
        self.main_camera_node = self.camnodes[camera_name]
        if self.realistic:
            return self.renderer.render(self)
        else:
            return self.renderer.render(self, flags=pyrender.RenderFlags.FLAT)
    
    def camnames(self):
        for n in self.camnodes.keys():
            yield n

    def random_light(self):
        if self.sun is not None:
            self.remove_node(self.sun)
        directional_light = pyrender.DirectionalLight(color=np.ones(3), intensity=np.random.uniform(3, 8))
        r = np.eye(4)
        rot = random_rotation_matrix(np.random.uniform(-0.7 * np.pi, 0.7 * np.pi), None)
        r[0:3, 0:3] = rot
        self.sun = self.add(directional_light, pose=r)
        self.ambient_light = np.array(np.random.uniform(0.1, 0.3) * np.ones(3))

    def init_cameras(self):
        self.camnodes = {}
        for n in self.conf.keys():
            cam = get_fip_camintrinsics(self.conf, n, self.viewport_size)
            node = self.add(cam, pose=get_fip_campose(self.conf, n))
            self.camnodes[n] = node

    def set_single_spike(self, ply_file):
        mesh = trimesh.load(ply_file)
        mesh.apply_translation(-mesh.center_mass)
        cm = pyrender.Mesh.from_trimesh(mesh)
        Rt = self.get_random_pose()
        return self.add(cm, pose=Rt)

    def get_random_pose(self):
        # This generation rules are very much handcrafted. 
        scale_base = np.eye(3) * 0.001 # Volume is given in mm^3, but here unit is meters, scale down by 10^3
        rot_base = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        move_base = np.array([
            0, 0, 2.9
        ])
        if self.until_borders:
            random_move = np.array([0.7, 0.6, 0.1]) * np.random.uniform(-1, 1, 3)
        else:
            random_move = np.array([0.2, 0.2, 0.1]) * np.random.uniform(-1, 1, 3)
        random_rot = random_rotation_matrix(np.random.uniform(-np.pi/2, np.pi/2, 1).item())
        Rt = construct_pose_scale(rot_base, move_base, scale_base, random_rot, None, random_move)

        return Rt

    def generate_spikes(self, ply_dir_base, n_instance, n_duplicates):
        ply_files = get_ply_files(ply_dir_base)

        id = 0
        self.color_to_id = {}
        color_set = set()
        with tqdm.tqdm(total=n_instance * n_duplicates, desc="Adding spikes") as pbar:
            for i in range(n_instance):
                select = ply_files[random.randint(0, len(ply_files) - 1)]
                mesh = trimesh.load(select)
                mesh.apply_translation(-mesh.center_mass)                
                
                for j in range(n_duplicates):
                    color = None
                    while color is None or tuple(color) in color_set:
                        color = np.random.uniform(0, 255, 3).astype(np.int32)
                    color_set.add(tuple(color))
                    self.color_to_id[tuple(color)] = id
                    color = color.astype(np.float32) / 255
                    cm = pyrender.Mesh.from_trimesh(mesh, smooth=False, material=pyrender.MetallicRoughnessMaterial(baseColorFactor=color))
                    Rt = self.get_random_pose()
                    self.add(cm, pose=Rt)
                    id += 1
                    pbar.update(1)

    def draw_camera_meshes(self):
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

        for n in self.conf.keys():
            m = create_camera_mesh()
            cp = get_fip_campose(self.conf, n)
            self.add(m, pose=cp)

    def normalize_viewport_size(self, p: np.ndarray):
        return (p * self.viewport_normalization_factor)

    # Returns instances as list of x, y points
    def get_instance_pixels(self, camera_name, normalize_viewport_size = True):
        img, _ = self.make_image(camera_name)
        id_to_pixel: dict[list] = {}

        h, w, _ = img.shape
        for j in range(h):
            for i in range(w):
                id = self.color_to_id.get(tuple(img[j, i, :]), None)
                if id:
                    id_to_pixel.setdefault(id, [])
                    r = (i, j)
                    if normalize_viewport_size:
                        r = self.normalize_viewport_size(np.array(r))
                        r = tuple(r)
                    id_to_pixel[id].append(r)
        
        return id_to_pixel
    
    # Returns format x_min, y_min, x_max, y_max
    def get_instance_bbox(self, camera_name, normalize_viewport_size = True):
        id_to_pixel = self.get_instance_pixels(camera_name, False)
        id_to_bbox = {}

        for id, pixlist in id_to_pixel.items():
            xymin = np.array([10000, 10000])
            xymax = np.array([-10000, -10000])
            for x, y in pixlist:
                c = np.array([x, y])
                xymin = np.minimum(c, xymin)
                xymax = np.maximum(c, xymax)
            if normalize_viewport_size:
                xymin = self.normalize_viewport_size(xymin)
                xymax = self.normalize_viewport_size(xymax)
            id_to_bbox[id] = (*xymin, *xymax)

        return id_to_bbox
    
    def get_all_bounding_boxes(self):
        r = {}
        for n in self.camnames():
            r[n] = self.get_instance_bbox(n)
        return r
    
def save_scene_data(scene: FIPScene, data, dir: str):
    path = os.path.join(dir, f"{str(scene.scene_id)}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def load_scene_data(file: str):
    with open(file) as f:
        data = json.load(f)
    return data

def load_scene_data_folder(dir: str):
    files = os.listdir(dir)
    for file in files:
        splitf = os.path.splitext(file)
        if splitf[1] == ".json":
            data = load_scene_data(os.path.join(dir, file))
            yield splitf[0], data
    
if __name__ == "__main__":
    a = FIPScene("assets/poses/2024_07_04_14_04_Lot3.json", "F:/wheat-scans-simplyfied-fast", (400, 300), 5, 30)
    a.draw_camera_meshes()
    pyrender.Viewer(a)