import sys
from turtle import width
sys.path.append('core')
import argparse
import os
import cv2
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from config.parser import parse_args

import datasets
from raft import RAFT
from utils.flow_viz import flow_to_image
from utils.utils import load_ckpt
from utils import frame_utils

from PIL import Image

import glob

import time

from GME.StabTest import StabTest
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker

from tools.do_kiem_1_copy import run_optical_flow, stabilize_image, tensor_to_flow_numpy, run_detect_object

from draft import foe_based_moving_object_detection

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

img_check_mag = None

clicked_point = None


def cal_mag_median(fx, fy):
    # Lấy median flow của toàn frame (ước lượng background motion)
    fx_med = np.median(fx)
    fy_med = np.median(fy)

    # Tính vector flow tương đối
    fx_rel = fx - fx_med
    fy_rel = fy - fy_med
    mag_rel = np.sqrt(fx_rel**2 + fy_rel**2)  

    return mag_rel

# # --- Hàm vẽ optical flow dưới dạng HSV (màu sắc trực quan) ---
# def flow_to_rgb(flow):
#     global img_check_mag, clicked_point   # <-- khai báo global
#     h, w = flow.shape[:2]
#     fx, fy = flow[..., 0], flow[..., 1]
#     # mag, ang = cv2.cartToPolar(fx, fy)
    
#     mag = cal_mag_median(fx, fy)
    
#     rgb = None
#     # hsv = np.zeros((h, w, 3), dtype=np.uint8)
#     # hsv[..., 0] = ang * 180 / np.pi / 2
#     # hsv[..., 1] = 255
#     # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
#     # rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#     # cv2.namedWindow("rgb", cv2.namedWindow)
#     # cv2.imshow("rgb", rgb)
#     # cv2.waitKey()
    
#     # img_check_mag = cv2.imread("/mnt/data/linhnq11/Data/moving_object_detection/IR_bia_TANK/IR_bia_TANK0001/10718.jpg")
    
#     # Tạo một cửa sổ hiển thị ảnh
#     cv2.namedWindow("CHECK_MAG", cv2.WINDOW_NORMAL)
#     cv2.imshow("CHECK_MAG", img_check_mag)

#     # Gắn hàm callback cho cửa sổ
#     cv2.setMouseCallback("CHECK_MAG", mouse_event)

#     print("Hãy click vào ảnh để lấy giá trị fx, nhấn ESC để thoát...")

#     last_point = None
#     while True:
#         key = cv2.waitKey(20) & 0xFF
#         # ESC để thoát
#         if key == 27:
#             print("ESC pressed, closing CHECK_MAG")
#             break

#         # Nếu clicked_point thay đổi (người dùng click mới)
#         if clicked_point is not None and clicked_point != last_point:
#             x, y = clicked_point
#             last_point = clicked_point.copy()  # cập nhật điểm vừa in
#             # Lấy giá trị fx tại điểm click
#             if 0 <= x-1 < w and 0 <= y-1 < h:  # tránh out-of-bounds
#                 print(f"fx fy at clicked_point ({x}, {y}): {fx[y-1, x-1]}, {fy[y-1, x-1]}, mag : {mag[y-1, x-1]}")
#             else:
#                 print(f"Clicked point ({x}, {y}) ngoài kích thước ảnh!")

#     cv2.destroyWindow("CHECK_MAG")

#     # # Chờ người dùng nhấn phím bất kỳ
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
    
#     return rgb, mag

def demo(model, args):
    global img_check_mag
    with torch.no_grad():
        name_img = "test"

        imfile1 = "/mnt/data/linhnq11/Data/moving_object_detection/PESMOD/Pexels-Grisha-snow/images/frame0013.jpg"
        imfile2 = "/mnt/data/linhnq11/Data/moving_object_detection/PESMOD/Pexels-Grisha-snow/images/frame0016.jpg"   
        
        multi_size = 1
        
        image1 = cv2.imread(imfile1)
        h, w = image1.shape[:2]
        # image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        
        image2 = cv2.imread(imfile2)
        h, w = image2.shape[:2]
        frame_disp = image2.copy()
        # image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        
        # image1 = cv2.resize(image1, (960, 540))
        # image2 = cv2.resize(image2, (960, 540))
        
        height, width, _ = image1.shape

        
        flow = run_optical_flow(args, model, image2, image1)
        
        flow = flow[0][0].permute(1,2,0).cpu().numpy()
        
        # ✅ Code mới (tối ưu, không cần ghi/đọc file nữa)
        flows = tensor_to_flow_numpy(flow)
        
        print("flows : ", flows)
        
        list_bbox = run_detect_object(args, flows, [height, width])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name',
                        default= "config/eval/spring-L.json", type=str)
    parser.add_argument('--model', help='checkpoint path', type=str,
                        default= "SEA-RAFT-models/Tartan-C-T-TSKH-spring540x960-M.pth")
    parser.add_argument('--output', help='path to save output', type=str, default='demo_own')
    parser.add_argument('--iters', help='number of iterations', default=4, type=int)
    # parser.add_argument('--scale', help='input scale', type=int, default=0)
    parser.add_argument('--threshold_flow', help='threshold for visualizing flow', type=float, default=1.0)
 
    args = parse_args(parser)
    
    model = RAFT(args)
    load_ckpt(model, args.model)
    model = model.cuda()
    model.eval()

    demo(model, args)

if __name__ == '__main__':
    main()
    