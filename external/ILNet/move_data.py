import cv2
import os
import numpy as np

base_dir = "/mnt/data/danhnt/Seg/ILNet/datasets/sky2coco/val2017/"

img_list = os.listdir(base_dir)

f = open("test.txt","a")
for i in img_list:
    i = i.split(".")[0]
    f.write(str(i) + '\n')

print("Done")



    
