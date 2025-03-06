import gen_scene
import os
import cv2

n_iters = 1
save_imgs = True
base_dir = "F:/wheat-scans-simplyfied-fast/bboxes_ground_truth/simple"
for i in range(n_iters):
    a = gen_scene.FIPScene("assets/fip_poses_configuration.json", (400, 300))
    a.generate_spikes("F:/wheat-scans-simplyfied-fast/", 5, 50)
    bboxes = a.get_all_bounding_boxes()
    
    gen_scene.save_scene_data(a, bboxes, base_dir)
    if save_imgs:
        path = os.path.join(base_dir, str(a.scene_id))
        os.makedirs(path)
        for n in a.camnames():
            img, _ = a.make_image(n)
            cv2.imwrite(os.path.join(path, n), img)
            

