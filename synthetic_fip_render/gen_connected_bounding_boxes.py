import gen_scene
import os
import cv2

n_iters = 20
save_imgs = False
base_dir = "/home/jannis/Schreibtisch/vol_pred_fip_data/bboxes_ground_truth/val"
for i in range(n_iters):
    a = gen_scene.FIPScene("/home/jannis/Schreibtisch/volume_prediction_fip/assets/fip_poses_configuration.json", "/home/jannis/Schreibtisch/vol_pred_fip_data/wheat-scans-simplyfied-fast/", (400, 300), 5, 320)
    bboxes = a.get_all_bounding_boxes()
    
    gen_scene.save_scene_data(a, bboxes, base_dir)
    if save_imgs:
        path = os.path.join(base_dir, str(a.scene_id))
        os.makedirs(path)
        for n in a.camnames():
            img, _ = a.make_image(n)
            cv2.imwrite(os.path.join(path, n), img)
            

