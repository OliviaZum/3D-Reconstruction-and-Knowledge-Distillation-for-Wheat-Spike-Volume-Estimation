"""
Legacycode for detecting labels. 

(Does not work well)
"""

import json
import cv2
import os
import numpy as np
from skimage.util import view_as_windows
from matplotlib import pyplot as plt

class LabelMeImage:
    def __init__(self, json_path, read_img=False):
        with open(json_path, 'r') as file:
            self.data = json.load(file)
        
        self.boxes = []
        for shape in self.data['shapes']:
            label = shape['label']
            points = np.array(shape['points']).flatten()
            self.boxes.append((label, points))
        
        self.image_path = os.path.join(os.path.dirname(json_path), self.data['imagePath'])
        if read_img:
            self.image = cv2.imread(self.image_path)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        else:
            self.image = None

    def roi_box(self, box):
        x1, y1, x2, y2 = box.astype(np.int32)
        return self.image[y1:y2, x1:x2]

def histogrambased_detection(label, image):
    def l(values, a, img):
        bluemask = np.zeros(img.shape[0:2], dtype=np.bool_)
        for v in values:
            v = np.array(v)
            low = np.clip(v - a, 0, 255).reshape((1, 1, -1))
            high = np.clip(v + a, 0, 255).reshape((1, 1, -1))
            bluemask = np.logical_or(bluemask, np.all(np.logical_and(img >= low, img <= high), axis=2))
        return bluemask

    if label == 'blue':
        mask = l([[62, 155, 255], [30, 56, 134], [39, 73, 180], [52, 85, 204], [27, 43, 97], [24, 36, 73]], 5, image)
    elif label == 'white':
        mask = l([255, 255, 255], 0, image)
    elif label == 'brown':
        mask = l([[44, 33, 34], [60, 46, 52]], 5, image)
    elif label == 'yellow' or label == 'yellowgreen':
        yellow = l([[253, 248, 177], [255, 252, 221], [246, 183, 81], [156, 115, 60], [228, 157, 76], [140, 120, 87], [255, 253, 228], [109, 99, 74], [240, 205, 112]], 7, image)
        yellowgreen = l([[52, 118, 95], [54, 123, 102], [40, 54, 47], [62, 129, 104]], 5, image)
        green_ext = cv2.dilate(yellowgreen.astype(np.float32), np.ones((20, 20), np.uint8), iterations=1)
        if label == 'yellow':
            mask = np.logical_and(np.logical_not(green_ext), yellow)
        else:
            mask = np.logical_and(green_ext, yellow)
    elif label == 'green':
        mask = l([[44, 151, 185], [40, 135, 163], [48, 149, 186], [54, 199, 238], [35, 111, 132], [49, 166, 208]], 5, image)
    elif label == 'violet':
        mask = l([[35, 40, 89], [31, 35, 72], [37, 45, 98], [111, 126, 255], [150, 151, 255]], 5, image)
    elif label == 'red':
        mask = l([[177, 49, 50], [196, 53, 60], [204, 56, 59], [152, 44, 48], [177, 62, 69], [252, 67, 72], [255, 116, 114]], 15, image)
    elif label == 'silver':
        mask = l([[92, 118, 160], [86, 110, 151], [55, 66, 96], [41, 53, 81], [115, 150, 208], [106, 130, 180], [73, 89, 123], [57, 73, 97], [38, 47, 66]], 5, image)
    
    #plt.imshow(black.astype(np.float32), interpolation='nearest', cmap='gray')
    #plt.show()

    stride = 50
    wsize = 100
    windows = view_as_windows(mask, (wsize, wsize), step=(stride, stride))
    #subv = np.zeros(windows.shape[0:2])
    subv = windows.sum(axis=3)
    subv = subv.sum(axis=2)
    maxi = np.argmax(subv)
    y, x = np.unravel_index(maxi, subv.shape)
    value = subv[y, x]
    y_start = max(0, y - 1)
    y_end = min(subv.shape[0], y + 2)
    x_start = max(0, x - 1)
    x_end = min(subv.shape[1], x + 2)
    subv[y_start:y_end, x_start:x_end] = 0
    part = np.partition(subv.flatten(), -2)
    confidence = (value - part[-2]) / (wsize * wsize)
    
    print(confidence)
    image = cv2.rectangle(image, (x * stride, y * stride), (x * stride + wsize, y * stride + wsize),
                    color=[255, 0, 0], thickness=15)
    plt.imshow(image)
    plt.show()
    

    return (y, x), confidence

