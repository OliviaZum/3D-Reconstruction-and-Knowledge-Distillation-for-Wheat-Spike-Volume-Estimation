import experiment_utils
import utils_3d
import numpy as np

def watch_ply_files():
    train, _ = experiment_utils.get_real_dataset()
    for i in range(len(train)):
        _, b, _ = train.__getitem__(i)
        utils_3d.visualize_point_cloud_single(b[:, 0:3].numpy(), b[:, 6].numpy())

watch_ply_files()