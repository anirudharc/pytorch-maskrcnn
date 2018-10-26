import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from scipy.misc import toimage
import cv2

import coco
import utils
import model as modellib
import visualize

import torch

# Dataset directory
DATA_DIR = "/media/bighdd7/arayasam/dataset/charades/" 
OPFLOW_PATH = "/media/bighdd7/arayasam/dataset/charades/Charades_v1_flow"
RGB_PATH = "/media/bighdd7/arayasam/dataset/charades/Charades_v1_rgb" #06OKE/"

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights file
# Download this file and place in the root of your
# project (See README file for details)
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.pth")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    # GPU_COUNT = 0 for CPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object.
model = modellib.MaskRCNN(model_dir=MODEL_DIR, config=config)
if config.GPU_COUNT:
    model = model.cuda()

# Load weights trained on MS-COCO
model.load_state_dict(torch.load(COCO_MODEL_PATH))

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# Directory traversal
for root, dirs, files in os.walk(DATA_DIR, topdown=False):
    print("######################################")
    if len(dirs) == 0:
        print(root)
        print(files[0:10])
    print("######################################")
    # for name in files:
    #     print(os.path.join(root, name))
    
    # for name in dirs:
    #     print(os.path.join(root, name))

# Input and OD for RGB frame
path = RGB_PATH + "/0DBQD/0DBQD-000001.jpg"
image = cv2.imread(path)

cv2.imwrite("./result/input.png", image)
results = model.detect([image])
r = results[0]
print(r['class_ids'])

# Input and process optical flow frame
opflow = cv2.imread(OPFLOW_PATH + "/ESAIY/ESAIY-000142y.jpg")
inverted_opflow = cv2.bitwise_not(opflow)
retval, threshold = cv2.threshold(inverted_opflow, 128, 255, cv2.THRESH_BINARY)

cv2.imwrite("./result/opflow.png", opflow)
cv2.imwrite("./result/inverted_opflow.png", inverted_opflow)
cv2.imwrite("./result/thresholed_opflow.png", threshold)

# Apply mask on thresholded inverted opflow
inverted_diff = inverted_opflow
masked_opflow = opflow
threshold_diff = threshold
for idx in range(r['masks'].shape[2]):

    name = './result/' + str(idx) + '_my.png'
    cv2.imwrite(name, r['masks'][:,:,idx] * 255)
    
    mask = r['masks'][:,:,idx]

    for c in range(3):
            threshold_diff[:, :, c] = np.where(mask == 1,
                                    threshold_diff[:, :, c] * 0,
                                    threshold_diff[:, :, c])

    for c in range(3):
            inverted_diff[:, :, c] = np.where(mask == 1,
                                    inverted_diff[:, :, c] * 0,
                                    inverted_diff[:, :, c])
    
    for c in range(3):
            masked_opflow[:, :, c] = np.where(mask == 1,
                                    masked_opflow[:, :, c] * 0,
                                    masked_opflow[:, :, c])

cv2.imwrite("./result/masked_invereted_threshold.png", threshold_diff)
cv2.imwrite("./result/masked_invereted_opflow.png", inverted_diff)
cv2.imwrite("./result/masked_opflow.png", masked_opflow)

# Visualize result
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])


#Break down frames into parts
height, width = threshold.shape[:2]
h_gap = height//3
w_gap = width//4
print(height, h_gap, width, w_gap)

count = 1
for start_row in range(0, height, h_gap):
    end_row = start_row + h_gap
    for start_col in range(0, width, w_gap):
        end_col = start_col + w_gap
        name = './result/part_' + str(count) + '.png'
        count += 1

        print("Part Average " + name  + ":")
        # print(start_row, end_row, start_col, end_col)

        part_avg_color_per_row = np.average(threshold_diff[start_row:end_row, start_col:end_col], axis=0)
        part_avg_color = np.average(part_avg_color_per_row, axis=0)

        inv_part_avg_color_per_row = np.average(threshold[start_row:end_row, start_col:end_col], axis=0)
        inv_part_avg_color = np.average(inv_part_avg_color_per_row, axis=0)

        print("Inverted avg:", end =" ")
        print(inv_part_avg_color)
        print("Greyscale avg:", end =" ")
        print(part_avg_color)
        print()

        if (start_row + h_gap) <= height and (start_col + w_gap) <= width:  
            cv2.imwrite(name, threshold_diff[start_row:end_row, start_col:end_col])


# # Load a random image from the images folder
# file_names = next(os.walk(IMAGE_DIR))[2]
# # image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
# image = cv2.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

# # Run detection
# results = model.detect([image])
# cv2.imwrite("./result/input.png", image)
# cv2.imwrite("./result/input_snippet.png", image[10:100, 10:100])

# r = results[0]
# # # print(type(r['rois'][0]))
# # print(r['masks'].shape[2])
# # # print(r['rois'].shape)
# # # print(r['rois'][0].shape)
# # # print(r['rois'])

# diff = image.copy()
# for idx in range(r['masks'].shape[2]):

#     name = './result/' + str(idx) + '_my.png'
#     cv2.imwrite(name, r['masks'][:,:,idx] * 255)
    
#     mask = r['masks'][:,:,idx]
#     for c in range(3):
#             diff[:, :, c] = np.where(mask == 1,
#                                     diff[:, :, c] * 0,
#                                     diff[:, :, c])

# cv2.imwrite("./result/masked.png", diff)

# avg_color_per_row = np.average(image, axis=0)
# avg_color = np.average(avg_color_per_row, axis=0)

# diff_avg_color_per_row = np.average(diff, axis=0)
# diff_avg_color = np.average(diff_avg_color_per_row, axis=0)

# print(avg_color, diff_avg_color)
# print(r['class_ids'])

# #Break down frames into parts
# height, width = diff.shape[:2]
# print(height, width)
# h_gap = height//3
# w_gap = width//4

# count = 1
# for start_row in range(0, height, h_gap):
#     for start_col in range(0, width, w_gap):
#         name = './result/part_' + str(count) + '.png'
#         count += 1
#         print(name)
#         print(start_row, start_row + h_gap, start_col, start_col + w_gap)
#         if (start_row + h_gap) <= height and (start_col + w_gap) <= width:
            
#             cv2.imwrite(name, image[start_row:start_row+h_gap, start_col:start_col+w_gap])

# # Visualize result
# visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
#                             class_names, r['scores'])
# plt.show()