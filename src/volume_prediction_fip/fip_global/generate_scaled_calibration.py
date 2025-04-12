"""
Generates a calibration for a fip scan using openMVG. extrinsics are scaled roughly correctly
using the metrically correct calibration files of the FIP. (Those are to imprecise for most applications,
thats why OpenMVG is used in the first place)

The calibration file is saved as poses_scaled in the scan folder. Some files of the build are kept in the build folder
(the reconstruction report, the unscaled poses, the sfm data file)

If it crashes, usually means that there is an image for which this did not work. Try on another scan.
Expects to be run in the scan directory. 
Requires installation of OpenMVG binaries, tested on OpenMVG 2.1, commit 0fd4fc9dc856317900e9c2f2f7c1472d97e32de2.
"""

import os
import shutil
import subprocess
import argparse
import json
import numpy as np
import tempfile

# Takes an openmvg scene json with extrinsics and converts to simpler to read configuration
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

def read_fip_extrinsics(extrinsics_path):
    config = {}
    camkeys = set([f"cam_{i:02}" for i in range(1, 13)]) # ignore 13th camera
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
        raise Exception(f"extrinsics file {extrinsics_path} does not contain extrinsics for all cameras")
    return config

# Scales a configuration obtained by sfm using the correctly scaled extrinsics of the calibration.
# (Note: By definition this is somewhat approximate. The reason the calibration is done by Sfm in the first place
# is because the fip calibration is not precise enough for most applications. The scaling will hence also only be approx precise.)
def scale_configuration(path_in, fip_extrinsics_in, path_out):
    u = lambda x: f"cam_{(x + 1):02d}"

    fip_calibrations = []
    for file in os.listdir(fip_extrinsics_in):
        if os.path.splitext(file)[1] == ".txt":
            fip_calibrations.append(read_fip_extrinsics(os.path.join(fip_extrinsics_in, file)))

    metric_distance = np.zeros((len(fip_calibrations[0]), len(fip_calibrations[0]), len(fip_calibrations)))

    for pos_set in range(len(fip_calibrations)):
        for i in range(len(fip_calibrations[0])):
            for j in range(len(fip_calibrations[0])):
                p1 = np.array(fip_calibrations[pos_set][u(i)]["extrinsics"]["center"])
                p2 = np.array(fip_calibrations[pos_set][u(j)]["extrinsics"]["center"])
                dist = np.linalg.norm(p1 - p2)
                metric_distance[i, j, pos_set] = dist

    min_values = np.min(metric_distance, axis=2)
    max_values = np.max(metric_distance, axis=2)
    max_def = np.abs(max_values - min_values)
    mean_distance = metric_distance.mean(axis=2)
    mean_distance[mean_distance == 0] = 0.0001
    max_error = np.max(max_def / mean_distance)
    print(f"The maximum error given this set of possible fip calibrations is {max_error} (relative size)")

    with open(path_in) as f:
        sfm_calibration = json.load(f)
    sfm_positions = []
    for i in range(0, 12):
        sfm_positions.append(np.array(sfm_calibration[u(i) + ".png"]["extrinsics"]["center"]))
    scales = []
    for i in range(0, 12):
        for j in range(0, 12):
            if i == j:
                continue
            s = metric_distance[i, j] / np.linalg.norm(sfm_positions[i] - sfm_positions[j])
            scales.append(s)
    scale = np.array(scales).mean()
    diff = np.array(scales).max() - np.array(scales).min()
    print(f"Maximum error when scaling (difference between smallest and largest scale factor): {diff}")
    print(f"Scale factor found: {scale}")

    sfm_positions = np.stack(sfm_positions, axis=0)
    sfm_positions = sfm_positions * scale

    for i in range(0, 12):
        sfm_calibration[u(i) + ".png"]["extrinsics"]["center"] = sfm_positions[i].tolist()

    with open(path_out, "w") as f:
        json.dump(sfm_calibration, f, indent=4)

def calibrate(openmvg_path, fip_extrinsics_folder):
    with tempfile.TemporaryDirectory() as tempdir:
        os.makedirs("./build", exist_ok=False)
        os.makedirs('./build/images', exist_ok=True)

        for item in os.listdir('.'):
            if os.path.splitext(item)[1] in [".jpg", ".png"]:
                shutil.copy(item, './build/images')

        subprocess.run([
            os.path.join(openmvg_path, 'openMVG_main_SfMInit_ImageListing'),
            '-i', './build/images', '-o', 'build', '-f', '10200', '-g', '0'
        ], check=True)

        subprocess.run([
            os.path.join(openmvg_path, 'openMVG_main_ComputeFeatures'),
            '-i', 'build/sfm_data.json', '-o', 'build'
        ], check=True)

        subprocess.run([
            os.path.join(openmvg_path, 'openMVG_main_PairGenerator'),
            '-i', 'build/sfm_data.json', '-o', 'build/pairs.txt', '-m', 'EXHAUSTIVE'
        ], check=True)

        subprocess.run([
            os.path.join(openmvg_path, 'openMVG_main_ComputeMatches'),
            '-i', 'build/sfm_data.json', '-o', 'build/matches.putative.bin', '-p', 'build/pairs.txt'
        ], check=True)

        subprocess.run([
            os.path.join(openmvg_path, 'openMVG_main_GeometricFilter'),
            '-i', 'build/sfm_data.json', '-m', 'build/matches.putative.bin', '-o', 'build/matches.geom.bin'
        ], check=True)

        subprocess.run([
            os.path.join(openmvg_path, 'openMVG_main_SfM'),
            '-i', 'build/sfm_data.json', '-o', 'build', '-M', 'build/matches.geom.bin',
            '--sfm_engine', 'INCREMENTAL'
        ], check=True)

        subprocess.run([
            os.path.join(openmvg_path, 'openMVG_main_ConvertSfM_DataFormat'),
            '-i', 'build/sfm_data.bin', '-o', 'build/sfm_data_conf.json', '-V', '-I', '-E'
        ], check=True)

        convert_openmvg_to_simplified("build/sfm_data_conf.json", "build/simple_poses.json")
        scale_configuration("build/simple_poses.json", fip_extrinsics_folder, "poses_scaled.json")

        # Clean up the build folder
        for item in os.listdir('./build'):
            if item not in ["sfm_data.bin", "SfMReconstruction_Report.html", "simple_poses.json", "sfm_data_conf.json"]:
                item_path = os.path.join('./build', item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)

def main():
    parser = argparse.ArgumentParser(description="Tries to create a calibration file via an openmvg Sfm. Will run on the images in the current folder.")
    parser.add_argument(
        "-o", "--openmvg_path",
        required=True,
        help="Path to the OpenMVG binaries."
    )
    parser.add_argument(
        "-e", "--fip_extrinsics",
        required=True,
        help="A folder containing possible extrinsics of the fip (scale will be averaged). This is used to scale the calibration obtained by Sfm correctly."
    )
    args = parser.parse_args()


    calibrate(args.openmvg_path, args.fip_extrinsics)

if __name__ == "__main__":
    main()