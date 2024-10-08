import numpy as np
import json

def cross_product_matrix(x):
    X = np.array([[0, -x[2], x[1]],
                  [x[2], 0, -x[0]],
                  [-x[1], x[0], 0]])
    return X

# From: https://github.com/libmv/libmv/blob/master/src/libmv/multiview/fundamental.cc#L289
def fundamentalFromRT(R1, t1, K1, R2, t2, K2):
    R = R2 @ R1.T
    t = t2 - R @ t1
    Tx = cross_product_matrix(t)
    E =  Tx @ R
    return np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)

def load_k(intrinsics):
    f = intrinsics["focal_length"]
    pp = np.array(intrinsics["principal_point"])
    return np.array([
        [f, 0, pp[0]],
        [0, f, pp[1]],
        [0, 0, 1]
    ])

# Build a fundamental matrix for 2 cameras using a pose configuration file
def build_fundamental(poses_conf, cam1: str, cam2: str):
    cam1_data = poses_conf[cam1]
    cam2_data = poses_conf[cam2]

    R1 = np.array(cam1_data["extrinsics"]["rotation"])
    t1 = np.array(cam1_data["extrinsics"]["center"])
    K1 = load_k(cam1_data["intrinsics"])

    R2 = np.array(cam2_data["extrinsics"]["rotation"])
    t2 = np.array(cam2_data["extrinsics"]["center"])
    K2 = load_k(cam2_data["intrinsics"])

    t1 = -R1 @ t1
    t2 = -R2 @ t2

    # Satisfies: x2.T @ F @ x1 = 0
    return fundamentalFromRT(R1, t1, K1, R2, t2, K2)

# Assumes all points normalized homogenous
# Checks if the epipolar line defined by a point on cam1 intersects with a line defined by 2 points on cam2
# Untested currently
def check_epipolar_intersect(F: np.ndarray, point_cam1: np.ndarray, point1_cam2: np.ndarray, point2_cam2: np.ndarray):
    epi_line = F @ point_cam1
    cam2_line = np.cross(point1_cam2, point2_cam2)

    intersection = np.cross(epi_line, cam2_line)
    if np.abs(intersection[2]) < 0.001:
        return False

    v_1to2 = point2_cam2 - point1_cam2
    intersection /= intersection[2]
    zeroed_intersection = intersection - point1_cam2
    l = zeroed_intersection[0] / v_1to2[0]
    # Should hold always, delete later (intersection lies on the line from p1 to p2, just as v_1to2, vectors should be collinear) 
    assert np.abs(l - zeroed_intersection[1] / v_1to2[1]) < 0.1 

    return l >= 0 and l <= 1


def test_compute_fundamental(json_file):
    with open(json_file, 'r') as f:
        poses_conf = json.load(f)

    ps = {}
    ps["cam_01.png"] = (2914, 564, 1)
    ps["cam_02.png"] = (2733, 856, 1)
    ps["cam_03.png"] = (2931, 876, 1)
    ps["cam_04.png"] = (2966, 689, 1)
    ps["cam_05.png"] = (3032, 976, 1)
    ps["cam_06.png"] = (2796, 1080, 1)

    for i in range(2, 7):
        im1 = "cam_01.png"
        im2 = f"cam_0{i}.png"

        E = build_fundamental(poses_conf, im1, im2)

        print(f"Round {i}")
        print(np.dot(ps[im2], E @ ps[im1]))

        """
        img1 = cv2.imread(im1)
        img2 = cv2.imread(im2)

        for y in range(img2.shape[0]):
            for x in range(img2.shape[1]):
                w = np.dot(np.array([x, y, 1]), E @ ps[im1])
                if np.abs(w) < 0.001:
                    img2 = cv2.circle(img2, (x, y), 1, (0, 255, 0), -1)

        cv2.imshow("Cam 02 - Epipolar Lines", img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """

test_compute_fundamental('poses_conf.json')
