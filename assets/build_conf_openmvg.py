import json
import numpy as np
import cv2
import random

# Takes a openmvg scene json with extrinsics and converts to simpler to read configuration
def convert_openmvg_to_simplified(path_in, path_out):
    output = {}

    with open(path_in) as f:
        data = json.load(f)

    intrinsics_dict = {intrinsic['key']: intrinsic['value']['ptr_wrapper']['data'] for intrinsic in data['intrinsics']}
    extrinsics_dict = {extrinsic['key']: extrinsic['value'] for extrinsic in data['extrinsics']}

    for view in data['views']:
        img_name = view['value']['ptr_wrapper']['data']['filename']
        id_intrinsic = view['value']['ptr_wrapper']['data']['id_intrinsic']
        id_pose = view['value']['ptr_wrapper']['data']['id_pose']

        intrinsics_data = intrinsics_dict.get(id_intrinsic)
        extrinsics_data = extrinsics_dict.get(id_pose)
        if extrinsics_data is None:
            continue

        output[img_name] = {
            "intrinsics": intrinsics_data,
            "extrinsics": {
                "rotation": extrinsics_data['rotation'],
                "center": extrinsics_data['center']
            }
        }

    with open(path_out, "w") as f:
        json.dump(output, f, indent=4)
