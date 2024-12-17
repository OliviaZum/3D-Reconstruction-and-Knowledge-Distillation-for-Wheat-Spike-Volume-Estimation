import numpy as np
import yaml
import json
import sys

# untested for now
def convert_fip_calibration_to_simplified(extrinsics_path: str, intrinsics_path_base: str, out_path: str):
    """
    extrinsics_path: The path to a file containing intrinsics for all fip cameras (FIP format)
    intrinsics_path_base: The bath to a folder with the beginning of the name of the intrinsics files. (E.g. Path/To/Folder/2024_04_08_intrinsics_, 
        when files are named 2024_04_08_intrinsics_cam_01.yaml to 2024_04_08_intrinsics_cam_13.yaml)
    out_path: Where to write the file containing the whole configuration
    """
    config = {}
    camkeys = set([f"cam_{i:02}" for i in range(1, 13)]) # ignore 13 camera
    with open(extrinsics_path) as f:
        extrinsics_str = f.readlines()
    for line in extrinsics_str:
        if line.lstrip().startswith("#"):
            continue
        else:
            w = line.split()
            if len(w) == 0 or w[0] not in camkeys:
                continue
            rotation = np.array(list(map(lambda x: float(x), w[7:]))).reshape((3, 3), order="C")
            position = np.array(list(map(lambda x: float(x), w[1:4])))
            config[w[0]] = {
                "extrinsics": {
                    "rotation": (rotation).tolist(),
                    "center": (position).tolist()
                }
            }
    if len(camkeys.intersection(config.keys())) != len(camkeys):
        raise Exception("extrinsics file does not contain extrinsics for all cameras")
    
    target_width = 4096
    target_height = 3000
    for cam in camkeys:
        path = f"{intrinsics_path_base}{cam}.yaml"

        with open(path, 'r') as file:
            data = yaml.safe_load(file)
        
        # Rescale to match the width of the actual full images for simplicity
        width_scale = target_width / data["width"]
        height_scale = target_height / data["height"]
        focal_scale = (width_scale + height_scale) / 2 # -> shouldn't matter as the ratio should be the same
        
        config[cam]["intrinsics"] = {
                "width": target_width,
                "height": target_height,
                "focal_length": (data["fx"] + data["fy"]) * 0.5 * focal_scale,
                "principal_point": [
                    data["cx"] * width_scale,
                    data["cy"] * height_scale
                ],
                "disto_k3": [
                    data["k1"],
                    data["k2"],
                    data["k3"]
                ]
            }
    
    config = {f"{x}.png": y for x, y in config.items()}
    with open(out_path, "w") as f:
        json.dump(config, f, indent=4)
        
if __name__ == "__main__":
    convert_fip_calibration_to_simplified(sys.argv[1], sys.argv[2], sys.argv[3])