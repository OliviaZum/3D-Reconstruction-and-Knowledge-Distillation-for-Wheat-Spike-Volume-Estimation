import trimesh
import pyrender
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
import random
import os
import tqdm

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

    #scene.set_pose(cam_node, pose=Pnew)
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

def random_rotation_matrix(max_angle: float) -> np.ndarray:
    # Generate a random unit vector (axis of rotation)
    axis = np.random.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.random.uniform(-max_angle, max_angle)
    
    # Compute the components of the rotation matrix using Rodrigues' rotation formula
    K = np.array([[    0, -axis[2],  axis[1]],
                  [ axis[2],     0, -axis[0]],
                  [-axis[1],  axis[0],     0]])
    
    rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    
    return rotation_matrix

class FIPScene(pyrender.Scene):
    def __init__(self, conf_path: str, meshes_base_dir: str, viewport_size: tuple, n_instance: int, n_duplicates: int,
                    nodes=None, bg_color=(0, 0, 0), ambient_light=[1.0, 1.0, 1.0], name=None):
        super().__init__(nodes, bg_color, ambient_light, name)

        with open(conf_path) as f:
            self.conf = json.load(f)

        self.renderer = pyrender.OffscreenRenderer(viewport_width=viewport_size[0], viewport_height=viewport_size[1])
        self.init_cameras(viewport_size)
        self.generate_spikes(meshes_base_dir, n_instance, n_duplicates)


    def make_image(self, camera_name: str):
        self.main_camera_node = self.camnodes[camera_name]
        return self.renderer.render(self, flags=pyrender.RenderFlags.FLAT)
    
    def camnames(self):
        for n in self.camnodes.keys():
            yield n

    def init_cameras(self, viewport_size):
        self.camnodes = {}
        for n in self.conf.keys():
            cam = get_fip_camintrinsics(self.conf, n, viewport_size)
            node = self.add(cam, pose=get_fip_campose(self.conf, n))
            self.camnodes[n] = node

    def generate_spikes(self, ply_dir_base, n_instance, n_duplicates):
        ply_files = []
        for root, _, files in os.walk(ply_dir_base):
            for file in files:
                if file.endswith('.ply'):
                    full_path = os.path.join(root, file)
                    ply_files.append(full_path)

        def r():
            return (random.random() - 0.5) * 2

        id = 0
        self.color_to_id = {}
        color_set = set()
        with tqdm.tqdm(total=n_instance * n_duplicates, desc="Adding spikes") as pbar:
            for i in range(n_instance):
                select = ply_files[random.randint(0, len(ply_files) - 1)]
                mesh = trimesh.load(select)
                mesh.apply_translation(-mesh.center_mass)

                # This generation rules are very much handcrafted. Technically data "should" originate from configuration file (especially pose of center along x, y)
                # Practically this is a lot easier and works for now.
                scale_base = np.eye(3) * 0.002
                rot_base = np.array([
                    [1, 0, 0],
                    [0, 0, -1],
                    [0, 1, 0]
                ])
                move_base = np.array([
                    0.53, 0.96, -1.2 + 8
                ])
                
                for j in range(n_duplicates):
                    color = None
                    while color is None or tuple(color) in color_set:
                        color = np.random.uniform(0, 255, 3).astype(np.int32)
                    color_set.add(tuple(color))
                    self.color_to_id[tuple(color)] = id
                    color = color.astype(np.float32) / 255
                    cm = pyrender.Mesh.from_trimesh(mesh, smooth=False, material=pyrender.MetallicRoughnessMaterial(baseColorFactor=color))
                    random_move = np.array([
                        2 * r(), 2 * r(), 0.3 * r()
                    ])
                    random_rot = random_rotation_matrix(np.pi / 4)
                    random_scale = np.eye(3) * (r() * 0.3 + 1)
                    R = random_rot @ random_scale @ scale_base @ rot_base
                    t = random_move + move_base
                    Rt = np.eye(4)
                    Rt[0:3, 0:3] = R
                    Rt[0:3, 3] = t

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


a = FIPScene("C:/Users/Admin/Desktop/master_thesis/volume_prediction_fip/assets/fip_poses_configuration.json", "F:/wheat-scans", (1000, 750), 1, 160)
for n in a.camnames():
    img, _ = a.make_image(n)

    plt.imshow(img)
    plt.show()