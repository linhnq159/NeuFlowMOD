import sys
sys.path.append('..')
import torch, einops
import matplotlib.pyplot as plt
from torchvision.utils import flow_to_image
import torch.nn.functional as F
import os
import glob
import numpy as np
import cv2
import time
import argparse
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker

from skimage import morphology
import struct

parser = argparse.ArgumentParser()
parser.add_argument('--input_img', help="input image path")
parser.add_argument('--input_flow', help="input flow path")
parser.add_argument('--output', help="output path")


# args for bytetrack
parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
parser.add_argument("--track_buffer", type=int, default=5, help="the frames for keep lost tracks") # Cang nho xoa cang nhanh
parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking") # Cang be cang de matching
parser.add_argument(
    "--aspect_ratio_thresh", type=float, default=1.6,
    help="threshold for filtering out boxes of which aspect ratio are above the given value."
) # Mo rong bao nhieu lan
parser.add_argument('--min_box_area', type=float, default=0.1, help='filter out tiny boxes')
parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

threshold = 0.8

# img_size = [720, 1280]
img_size = [1080, 1920]
# img_size = [600, 1080]
# img_size = [960, 1280]

def clean_mask(mask, min_size=100, closing_iter=2):
    """
    mask: binary uint8 (0/255) or bool
    min_size: remove components smaller than this (pixels)
    closing_iter: number of morphological closing iterations
    """
    # ensure binary boolean
    if mask.dtype != bool:
        mask_bool = mask.astype(bool)
    else:
        mask_bool = mask

    # morphological closing to fill small holes and connect fragments
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    closed = cv2.morphologyEx(mask_bool.astype('uint8')*255, cv2.MORPH_CLOSE, kernel, iterations=closing_iter)
    closed = closed.astype(bool)

    # remove small objects
    cleaned = morphology.remove_small_objects(closed, min_size=min_size)

    # fill small holes
    cleaned = morphology.remove_small_holes(cleaned, area_threshold=min_size)

    return (cleaned.astype('uint8') * 255)

# --- Hàm đọc file .flo ---
def read_flo_file(filename):
    with open(filename, 'rb') as f:
        magic = struct.unpack('f', f.read(4))[0]
        if magic != 202021.25:
            raise ValueError('Magic number incorrect. Not a .flo file')
        w = struct.unpack('i', f.read(4))[0]
        h = struct.unpack('i', f.read(4))[0]
        data = np.fromfile(f, np.float32, count=2*w*h)
        flow = np.resize(data, (h, w, 2))
        return flow
    
# --- Hàm vẽ optical flow dưới dạng HSV (màu sắc trực quan) ---
def flow_to_rgb(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[..., 0], flow[..., 1]
    mag, ang = cv2.cartToPolar(fx, fy)
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb, mag    

# --- Xử lý phát hiện vật chuyển động ---
def detect_moving_objects(flow, mag_threshold=1.0, min_area=100, max_size=100):
    _, mag = flow_to_rgb(flow)
    mask = (mag > mag_threshold).astype(np.uint8) * 255
    
    pos_limit = 10
    
    cv2.namedWindow("mask before clean", cv2.WINDOW_NORMAL)
    cv2.imshow("mask before clean", mask)
    cv2.waitKey(1)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        print("x, y, w, h : ", x, y, w, h)
        
        # if w*h > min_area and w < max_size and h < max_size and x < pos_limit and y < pos_limit and x < img_size[1] - pos_limit and y < img_size[0] - pos_limit:  # bỏ qua các bbox quá nhỏ
        if w*h > min_area and w < max_size and h < max_size :  # bỏ qua các bbox quá nhỏ
            print("add old x, y, w, h : ", x, y, w, h)
            boxes.append([x  - w/2,y - h/2,x + w * 1.5, y + h * 1.5 ,1])
        
        if w*h > min_area and w < max_size and h < max_size and x < pos_limit and y < pos_limit and x < (img_size[1] - pos_limit) and y < (img_size[0] - pos_limit):  # bỏ qua các bbox quá nhỏ
        # if w*h > min_area and w < max_size and h < max_size :  # bỏ qua các bbox quá nhỏ
            print("add x, y, w, h : ", x, y, w, h)
            # boxes.append([x  - w/2,y - h/2,x + w * 1.5, y + h * 1.5 ,1])

    if len(boxes) < 1 :
        boxes.append([0,0,0,0,0])

    return boxes


pflows_1 = torch.Tensor()
pflows_2 = torch.Tensor()
pflows_0 = torch.Tensor()

os.makedirs(args.output, exist_ok= True)

# init Tracker
tracker = BYTETracker(args)

## Load Flow File 
# pflows = ['../RAFT/output/parkour/00000.flo',
#         '../RAFT/output/parkour/00001.flo',
#         '../RAFT/output/parkour/00002.flo']

# folder_flow = "../RAFT/output/Elliot-road/"
# folder_flow = "../RAFT/output/Wolfgang/"
folder_flow = args.input_flow

list_flow = sorted(glob.glob(folder_flow + "/*.flo"))

# folder_img = "../ZBS/demo/PESMOD/Pexels-Elliot-road/images"
# folder_img = "../ZBS/demo/PESMOD/Pexels-Wolfgang/images"
folder_img = args.input_img
list_img = sorted(glob.glob(folder_img + "/*.jpg")) + sorted(glob.glob(folder_img + "/*.png"))

frame_skip = 3 # 3

for idx in range(len(list_flow)):
    bbox_img = cv2.resize(cv2.imread(list_img[(idx - 1) * frame_skip]), (img_size[1], img_size[0]))  # đọc ảnh gốc để vẽ bbox chính xác
    
    list_bbox = []
        
    flows = read_flo_file(list_flow[idx])   
    print("type(flows) : ", type(flows))
    print("flows.shape : ", flows.shape)

    time_start = time.time()
    
    # --- Phát hiện đối tượng ---
    list_bbox = detect_moving_objects(flows, mag_threshold=2, min_area=36, max_size=100)

    time_end = time.time()
    print("Time per frame: ", (time_end - time_start))
        
    # tracking
    img_size = (img_size[0], img_size[1])
    tensor_bbox = np.array(list_bbox)
    tensor_bbox = torch.from_numpy(tensor_bbox).float()
    # print(list_bbox)
    print("tensor_bbox : ", tensor_bbox)
    out_track = tracker.update(tensor_bbox, img_size, img_size)
    print("out_track : ", out_track)

    for i in out_track :
        bbox = i.tlwh 
        x, y, w, h = bbox
        
        x = x + w/4
        y = y + h/4
        w = w/2
        h = h/2
        
        cv2.rectangle(bbox_img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 1)
        cv2.putText(bbox_img, str(int(w)) + "x" + str(int(h)), (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, color = (0, 255, 0), fontScale = 1,  thickness = 1)

    # Hiển thị bằng OpenCV
    cv2.namedWindow("demo_contour", cv2.WINDOW_NORMAL)
    cv2.imshow("demo_contour", bbox_img)
    cv2.waitKey(1)
    
    cv2.imwrite(os.path.join(args.output, str(idx).zfill(4) + "_bb.png"), bbox_img)

    # plt.figure(figsize=(8,8))
    # plt.title(f'{model_name} - only time {t_idx+1}')
    # plt.imshow(mask_t.detach().numpy(), interpolation='none', vmax=3)
    # plt.axis('off')
    # plt.show()