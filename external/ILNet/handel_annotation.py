import cv2
import os
import numpy as np

train_dir = "/mnt/data/danhnt/Seg/work_seg/Datasets/dokiem/test.txt"
base_dir = "/mnt/data/danhnt/Seg/work_seg/Datasets/dokiem/"

train_list = []


# read the train list
with open(train_dir, 'r') as f:
    for line in f.readlines():
        if line not in train_list:
            train_list.append(line.replace("\n", ""))

for i in train_list:
    path_img = base_dir + "masks/" + i + ".png"
    print(path_img)
    img = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img,15,255, cv2.THRESH_BINARY)[1]

    cv2.imwrite(base_dir + "mask/" + i + ".png",img)
    
