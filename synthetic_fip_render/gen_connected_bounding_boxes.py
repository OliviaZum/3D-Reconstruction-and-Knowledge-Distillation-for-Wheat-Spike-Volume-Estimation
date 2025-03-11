"""
Generates a set of artificial fip scans with paired bounding box ground truth.
"""

import gen_scene
import os
import cv2

if __name__ == "__main__":
    n_iters = 1
    save_imgs = True
    out_dir = "F:/wheat-scans-simplyfied-fast/bboxes_ground_truth/simple"
    ply_dir = "F:/wheat-scans-simplyfied-fast/" # Load from a few spikes with reduced resolution to increase speed
    config = "assets/poses/2023_06_08_13_11_Lot1.json"

    for i in range(n_iters):
        a = gen_scene.FIPScene(config, (800, 600))
        a.generate_spikes(ply_dir, 5, 100)
        bboxes = a.get_all_bounding_boxes()
        
        gen_scene.save_scene_data(a, bboxes, out_dir)
        if save_imgs:
            path = os.path.join(out_dir, str(a.scene_id))
            os.makedirs(path)
            for n in a.camnames():
                img, _ = a.make_image(n)
                cv2.imwrite(os.path.join(path, n), img)
            

