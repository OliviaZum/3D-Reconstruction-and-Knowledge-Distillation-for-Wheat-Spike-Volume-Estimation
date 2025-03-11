import numpy as np
import json
import warnings


# Assumes all points normalized homogenous
# Checks if the epipolar line defined by a point on cam1 intersects with a line defined by 2 points on cam2
# Points in x, y format
def check_epipolar_intersect(F: np.ndarray, point_cam1: np.ndarray, point1_cam2: np.ndarray, point2_cam2: np.ndarray):
    eps = 0.0001
    epi_line = F @ point_cam1
    cam2_line = np.cross(point1_cam2, point2_cam2)

    intersection = np.cross(epi_line, cam2_line)
    if np.abs(intersection[2]) < eps:
        return False

    v_1to2 = point2_cam2 - point1_cam2
    intersection /= intersection[2]
    zeroed_intersection = intersection - point1_cam2
    #l = np.linalg.norm(zeroed_intersection[0:2]) / np.linalg.norm(v_1to2[0:2])
    if v_1to2[0] < eps:
        if v_1to2[1] < eps:
            # Probably means that this is an invalid bounding box. Simply ignore.
            warnings.warn("check_epipolar_intersect got a 0 length line as input")
            return False
        else:
            l = zeroed_intersection[1] / v_1to2[1]
    else:
        l = zeroed_intersection[0] / v_1to2[0]
        
    # intersection lies on the line from p1 to p2, just as v_1to2, vectors should be collinear
    assert  np.linalg.norm(l * v_1to2[0:2] - zeroed_intersection[0:2]) < 0.001

    return l >= 0 and l <= 1

# Check if an epipolar line intersects a box on a second image
def check_epipolar_intersect_bbox(F: np.ndarray, point_cam1: np.ndarray, bbox_cam2: tuple):
    xmin, ymin, xmax, ymax = bbox_cam2
    p = np.array([
        [xmin, ymin, 1],
        [xmin, ymax, 1],
        [xmax, ymax, 1],
        [xmax, ymin, 1]
        ])
    hit = (check_epipolar_intersect(F, point_cam1, p[0], p[1]) or
         check_epipolar_intersect(F, point_cam1, p[0], p[3]) or
         check_epipolar_intersect(F, point_cam1, p[1], p[2]))
    return hit