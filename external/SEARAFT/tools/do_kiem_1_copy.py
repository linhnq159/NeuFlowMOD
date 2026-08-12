import sys
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
import math

from config.parser import parse_args

import datasets
from raft import RAFT
from core.utils.flow_viz import flow_to_image
from core.utils.utils import load_ckpt
from core.utils import frame_utils

from PIL import Image

import glob

import time

from GME.StabTest import StabTest
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEBUG = False

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def create_color_bar(height, width, color_map):
    """
    Create a color bar image using a specified color map.

    :param height: The height of the color bar.
    :param width: The width of the color bar.
    :param color_map: The OpenCV colormap to use.
    :return: A color bar image.
    """
    # Generate a linear gradient
    gradient = np.linspace(0, 255, width, dtype=np.uint8)
    gradient = np.repeat(gradient[np.newaxis, :], height, axis=0)

    # Apply the colormap
    color_bar = cv2.applyColorMap(gradient, color_map)

    return color_bar

def add_color_bar_to_image(image, color_bar, orientation='vertical'):
    """
    Add a color bar to an image.

    :param image: The original image.
    :param color_bar: The color bar to add.
    :param orientation: 'vertical' or 'horizontal'.
    :return: Combined image with the color bar.
    """
    if orientation == 'vertical':
        return cv2.vconcat([image, color_bar])
    else:
        return cv2.hconcat([image, color_bar])

def vis_heatmap(name, image, heatmap):
    # theta = 0.01
    # print(heatmap.max(), heatmap.min(), heatmap.mean())
    heatmap = heatmap[:, :, 0]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    # heatmap = heatmap > 0.01
    heatmap = (heatmap * 255).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = image * 0.3 + colored_heatmap * 0.7
    # Create a color bar
    height, width = image.shape[:2]
    color_bar = create_color_bar(50, width, cv2.COLORMAP_JET)  # Adjust the height and colormap as needed
    # Add the color bar to the image
    overlay = overlay.astype(np.uint8)
    combined_image = add_color_bar_to_image(overlay, color_bar, 'vertical')
    cv2.imwrite(name, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))

def get_heatmap(info, args):
    raw_b = info[:, 2:]
    log_b = torch.zeros_like(raw_b)
    weight = info[:, :2].softmax(dim=1)              
    log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=args.var_max)
    log_b[:, 1] = torch.clamp(raw_b[:, 1], min=args.var_min, max=0)
    heatmap = (log_b * weight).sum(dim=1, keepdim=True)
    return heatmap

def forward_flow(args, model, image1, image2):
    # add image1 to device
    image1 = image1.to(DEVICE)
    image2 = image2.to(DEVICE)
    
    # print("image1 : ", image1)
    
    output = model(image1, image2, iters=args.iters, test_mode=True)
    flow_final = output['flow'][-1]
    info_final = output['info'][-1]
    # x = input()
    return flow_final, info_final

def calc_flow(args, model, image1, image2):
    # print("args.scale : ", args.scale)
    img1 = F.interpolate(image1, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    img2 = F.interpolate(image2, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    H, W = img1.shape[2:]
    # print("Just for checking H, W: ", H, W)
    flow, info = forward_flow(args, model, img1, img2)
    
    # print("flow.shape : ", flow.shape)
    # print("flow : ", flow[0])
    
    flow_down = F.interpolate(flow, scale_factor=0.5 ** args.scale, mode='bilinear', align_corners=False) * (0.5 ** args.scale)
    info_down = F.interpolate(info, scale_factor=0.5 ** args.scale, mode='area')
    
    # print("flow_down.shape : ", flow_down.shape)
    # print("flow_down : ", flow_down[0])
    
    return flow_down, info_down
    
# --- Hàm chuyển flow tensor sang numpy trực tiếp ---
def tensor_to_flow_numpy(flow_tensor):
    """
    Chuyển optical flow từ torch.Tensor sang numpy array (H, W, 2)
    mà không cần ghi/đọc file .flo.
    """
    
    # print("type(flow_tensor) : ", type(flow_tensor))
    
    # Nếu flow có shape [1, 2, H, W]
    if isinstance(flow_tensor, torch.Tensor):
        if flow_tensor.ndim == 4:
            flow_tensor = flow_tensor[0]  # Bỏ batch dimension
        # Chuyển [2, H, W] -> [H, W, 2]
        flow_np = flow_tensor.permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
        return flow_np
    
    # Nếu flow đã là numpy thì giữ nguyên
    elif isinstance(flow_tensor, np.ndarray):
        if flow_tensor.ndim == 3 and flow_tensor.shape[0] == 2:
            flow_np = np.transpose(flow_tensor, (1, 2, 0))
        else:
            flow_np = flow_tensor
        return flow_np.astype(np.float32)
    
    else:
        raise TypeError("flow_tensor phải là torch.Tensor hoặc numpy.ndarray")

    
def stabilize_image(img_ref, img_target, method='affine'):
    # Chuyển ảnh sang grayscale
    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    gray_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)

    # feature detection and description (using ORB)
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(gray_ref, None)
    kp2, des2 = orb.detectAndCompute(gray_target, None)

    # feature matching
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:200]  # best 200 points

    # Lấy tọa độ điểm tương ứng
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    # Tính ma trận biến đổi (affine hoặc homography)
    if method == 'affine':
        M, _ = cv2.estimateAffinePartial2D(pts2, pts1)
        if M is None:
            return _identity_stabilize(img_ref, img_target, method)
        stabilized = cv2.warpAffine(img_target, M, (img_ref.shape[1], img_ref.shape[0]))
    else:
        M, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
        if M is None:
            return _identity_stabilize(img_ref, img_target, method)
        stabilized = cv2.warpPerspective(img_target, M, (img_ref.shape[1], img_ref.shape[0]))

    # print("M : ", M)

    return stabilized, M

def stabilize_image_fast(img_ref, img_target,
                         method='orb',             # 'orb' or 'lk'
                         downscale=1.0,            # scale for feature detection (<=1.0) #default 0.5
                         max_features=1000,
                         n_matches=150,
                         use_flann=True,
                         lk_params=None,
                         orb_obj=None,
                         matcher_obj=None):
    """
    Return: (stabilized_img, M) or (None, None) on failure.
    - method 'orb': feature-based (ORB + FLANN/ BF + RANSAC)
    - method 'lk' : optical flow (fast for adjacent frames)
    - downscale: detect features on smaller image to speed up; coords scaled back.
    - orb_obj, matcher_obj: reuse across calls for speed (recommended for video)
    """

    h, w = img_ref.shape[:2]

    # Convert to gray
    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    gray_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)

    if method == 'lk':
        # Lucas-Kanade optical flow path (very fast for consecutive frames)
        if lk_params is None:
            lk_params = dict(winSize=(21,21), maxLevel=3,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        # detect good corners on reference (faster than ORB)
        pts_ref = cv2.goodFeaturesToTrack(gray_ref, maxCorners=500, qualityLevel=0.01, minDistance=7)
        if pts_ref is None:
            return None, None
        pts_target, st, err = cv2.calcOpticalFlowPyrLK(gray_ref, gray_target, pts_ref, None, **lk_params)
        if pts_target is None:
            return None, None
        # keep only good points
        st = st.reshape(-1)
        pts_ref_good = pts_ref[st==1].reshape(-1,2)
        pts_target_good = pts_target[st==1].reshape(-1,2)
        if len(pts_ref_good) < 6:
            return None, None
        # estimate affine
        M, inliers = cv2.estimateAffinePartial2D(pts_target_good, pts_ref_good, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if M is None:
            return None, None
        stabilized = cv2.warpAffine(img_target, M, (w, h), flags=cv2.INTER_LINEAR)
        return stabilized, M

    # ---------- ORB path ----------
    # Downscale for detection
    if downscale <= 0 or downscale > 1.0:
        downscale = 1.0
    if downscale < 1.0:
        small_ref = cv2.resize(gray_ref, (0,0), fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
        small_target = cv2.resize(gray_target, (0,0), fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
    else:
        small_ref = gray_ref
        small_target = gray_target

    # create ORB once if not provided
    if orb_obj is None:
        orb = cv2.ORB_create(nfeatures=max_features, scaleFactor=1.2, nlevels=8)
    else:
        orb = orb_obj

    kp1, des1 = orb.detectAndCompute(small_ref, None)
    kp2, des2 = orb.detectAndCompute(small_target, None)

    if des1 is None or des2 is None or len(kp1) < 6 or len(kp2) < 6:
        return None, None

    # Setup matcher
    if matcher_obj is not None:
        matcher = matcher_obj
        use_flann = False  # assume user supplied a BF matcher
    else:
        if use_flann:
            # FLANN for binary descriptors (LSH)
            index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                                table_number=6,
                                key_size=12,
                                multi_probe_level=1)
            search_params = dict(checks=32)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
            # Note: FLANN expects descriptors of type float32 for some OpenCV builds; convert if needed
            if des1.dtype != np.float32:
                des1_fl = np.asarray(des1, np.float32)
                des2_fl = np.asarray(des2, np.float32)
            else:
                des1_fl, des2_fl = des1, des2
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            des1_fl, des2_fl = des1, des2

    # knnMatch + ratio test (Lowe), often faster and more robust than crossCheck
    try:
        if use_flann:
            raw_matches = matcher.knnMatch(des1_fl, des2_fl, k=2)
        else:
            raw_matches = matcher.knnMatch(des1, des2, k=2)
    except Exception:
        # fallback to simple match
        raw_matches = matcher.match(des1, des2)
        raw_matches = [[m] for m in raw_matches]

    good = []
    # ratio test when possible
    for m_n in raw_matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good.append(m)
        else:
            # single match fallback
            good.append(m_n[0])

    if len(good) == 0:
        return None, None

    # sort and keep top n_matches
    good = sorted(good, key=lambda x: x.distance)[:n_matches]

    # prepare points and scale back to original size
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    if downscale < 1.0:
        pts1 /= downscale
        pts2 /= downscale

    if len(pts1) < 6:
        return None, None

    # reshape for OpenCV functions
    pts1 = pts1.reshape(-1,1,2)
    pts2 = pts2.reshape(-1,1,2)

    # estimate transform (affine partial is sufficient for stabilization)
    M, inliers = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=2000, confidence=0.99)
    if M is None:
        return None, None

    stabilized = cv2.warpAffine(img_target, M, (w, h), flags=cv2.INTER_LINEAR)
    return stabilized, M

def _identity_stabilize(img_ref, img_target, method):
    """
    Trả về ảnh target giữ nguyên, M là ma trận đơn vị (ảnh đứng yên).
    """
    h, w = img_ref.shape[:2]
    if method == 'affine':
        M = np.array([[1, 0, 0],
                      [0, 1, 0]], dtype=np.float32)
        stabilized = cv2.warpAffine(img_target, M, (w, h))
    else:
        M = np.eye(3, dtype=np.float32)
        stabilized = cv2.warpPerspective(img_target, M, (w, h))
    return stabilized, M

def run_optical_flow(args, model, image1, image2):
    image1 = torch.tensor(image1, dtype=torch.float32).permute(2, 0, 1)
    image2 = torch.tensor(image2, dtype=torch.float32).permute(2, 0, 1)

    image1 = image1[None].to(DEVICE)
    image2 = image2[None].to(DEVICE)

    time_t = time.time()
    flow = calc_flow(args, model, image1, image2)
    time_s = time.time()
    if DEBUG:
        print("Time inference : ", time_s - time_t)
    
    return flow


def cal_mag_median(fx, fy):
    # Lấy median flow của toàn frame (ước lượng background motion)
    fx_med = np.median(fx)
    fy_med = np.median(fy)

    # Tính vector flow tương đối
    fx_rel = fx - fx_med
    fy_rel = fy - fy_med
    mag_rel = np.sqrt(fx_rel**2 + fy_rel**2)  

    return mag_rel


# --- Hàm vẽ optical flow dưới dạng HSV (màu sắc trực quan) ---
def flow_to_rgb(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[..., 0], flow[..., 1]
    # mag, ang = cv2.cartToPolar(fx, fy)
    mag = cal_mag_median(fx, fy)
    rgb = None
    # hsv = np.zeros((h, w, 3), dtype=np.uint8)
    # hsv[..., 0] = ang * 180 / np.pi / 2
    # hsv[..., 1] = 255
    # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # cv2.namedWindow("rgb", cv2.namedWindow)
    # cv2.imshow("rgb", rgb)
    # cv2.waitKey()
    return rgb, mag

def smooth_mask_for_contours(mask,
                             blur_type='gaussian',   # 'gaussian' | 'median' | 'bilateral' | None
                             blur_ksize=5,
                             thresh=127,             # threshold value (0-255) or None to use Otsu
                             morph_open=3,           # kernel size for opening (remove small noise)
                             morph_close=7,          # kernel size for closing (fill holes)
                             min_area=100,           # remove small components smaller than this
                             return_binary=True):
    """
    Input:
      - mask: np.ndarray (H,W) or (H,W,1). dtype: uint8 (0..255) or float (0..1)
    Output:
      - processed binary mask (uint8 0/255) ready for cv2.findContours
    """
    # --- normalize to uint8 0..255 ---
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.dtype != np.uint8:
        # assume float 0..1 or 0..255
        mmax = mask.max()
        if mmax <= 1.0:
            mask_u8 = (mask * 255).astype(np.uint8)
        else:
            mask_u8 = mask.astype(np.uint8)
    else:
        mask_u8 = mask.copy()

    # --- blur / smooth ---
    if blur_type == 'gaussian':
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        blurred = cv2.GaussianBlur(mask_u8, (k,k), 0)
    elif blur_type == 'median':
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        blurred = cv2.medianBlur(mask_u8, k)
    elif blur_type == 'bilateral':
        # bilateral preserves edges better
        blurred = cv2.bilateralFilter(mask_u8, d=blur_ksize, sigmaColor=75, sigmaSpace=75)
    else:
        blurred = mask_u8

    # --- threshold ---
    if thresh is None:
        _, bw = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, bw = cv2.threshold(blurred, int(thresh), 255, cv2.THRESH_BINARY)

    # --- morphology: open then close (tweak order as needed) ---
    # opening: remove small specks; closing: fill small holes
    if morph_open and morph_open > 1:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open, morph_open))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k_open)
    if morph_close and morph_close > 1:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close, morph_close))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k_close)

    # --- remove small connected components ---
    if min_area and min_area > 0:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
        mask_filtered = np.zeros_like(bw)
        for i in range(1, num_labels):  # skip background 0
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                mask_filtered[labels == i] = 255
        bw = mask_filtered

    if return_binary:
        return bw
    else:
        return bw.astype(np.uint8)


# --- Xử lý phát hiện vật chuyển động ---
def detect_moving_objects(flow, img_size, mag_threshold=1.0, min_area=100, max_size=100):
    _, mag = flow_to_rgb(flow)
    # print("mag_threshold : ", mag_threshold)
    # x = input()
    
    # print("mag : ", mag)
    
    # for i in range(mag.shape[0]):
    #     for j in range(mag.shape[1]):
    #         print("i : {}, j : {}, mag : {}".format(i, j, mag[i][j]))
    
    # print(type(mag))
    # print("mag.shape : {}, max mag : {}, min mag : {}".format(mag.shape, np.max(mag), np.min(mag)))
    
    # mask = (mag > 5.0).astype(np.uint8) * 255
    
    # cv2.namedWindow("mask 1", cv2.WINDOW_NORMAL)
    # cv2.imshow("mask 1", mask)
    # cv2.waitKey(1)
    
    # mag = np.minimum(
    #     1.0, np.abs(np.log10(1+np.abs(mag / 2.0)))
    # )
    
    # print("mag : ", mag)
    
    # print("mean mag : ", np.mean(mag))
    
    # mag_threshold = 0.45
    
    # max_mag_threshold = 5.0
    
    max_mag_threshold = 1.1
    
    # mag_threshold = 1.0
    
    print("mag_threshold : ", mag_threshold)
    
    mask = ((mag > mag_threshold)).astype(np.uint8) * 255
    # mask = ((mag > mag_threshold) & (mag < max_mag_threshold)).astype(np.uint8) * 255
    # mask = mag.astype(np.uint8) * 255
    
    cv2.namedWindow("mask 2", cv2.WINDOW_NORMAL)
    cv2.imshow("mask 2", mask)
    cv2.waitKey(1)
    
    # mask = (mag > 2.0).astype(np.uint8) * 255
    
    pos_limit = 10
    
    # cv2.namedWindow("mask before clean", cv2.WINDOW_NORMAL)
    # cv2.imshow("mask before clean", mask)
    # cv2.imwrite("Mask.jpg", mask)
    # cv2.waitKey(1)
    
    time_t_contours = time.time()
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    if DEBUG:
        print("img_size : ", img_size)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if DEBUG:
            print("x, y, w, h : ", x, y, w, h)
        
        # boxes.append([x  - w/2,y - h/2,x + w * 1.5, y + h * 1.5 ,1])
        
        # # if w*h > min_area and w < max_size and h < max_size and x < pos_limit and y < pos_limit and x < img_size[1] - pos_limit and y < img_size[0] - pos_limit:  # bỏ qua các bbox quá nhỏ
        # if w*h >= min_area and w < max_size and h < max_size and w > 5 and h > 5 : # bỏ qua các bbox quá nhỏ
        # # if w*h >= min_area and w < max_size and h < max_size : # bỏ qua các bbox quá nhỏ
        #     # print("add old x, y, w, h : ", x, y, w, h)
        #     boxes.append([x  - w/2,y - h/2,x + w * 1.5, y + h * 1.5 ,1])
            
        # if w > max_size or y > max_size :
        #     x = input()
        
        # if w*h > min_area and w < max_size and h < max_size and x > pos_limit and y > pos_limit and x < (img_size[1] - pos_limit) and y < (img_size[0] - pos_limit):  # bỏ qua các bbox quá nhỏ
        if w >= math.sqrt(min_area) and h >= math.sqrt(min_area) and w < max_size and h < max_size and x > pos_limit and y > pos_limit and (x + w) < (img_size[1] - pos_limit) and (y +h) < (img_size[0] - pos_limit):  # bỏ qua các bbox quá nhỏ
            if DEBUG:
                print("add x, y, w, h : ", x, y, w, h)
            boxes.append([x  - w/2,y - h/2,x + w * 1.5, y + h * 1.5 ,1])

    time_s_contours = time.time()
    if DEBUG:
        print("Time find contours: ", time_s_contours - time_t_contours)

    # if len(boxes) < 1 :
    #     boxes.append([0,0,0,0,0])

    return boxes


def run_detect_object(args, flows, img_size):
    # --- Phát hiện đối tượng ---
    time_start = time.time()
    list_bbox = detect_moving_objects(flows, img_size, mag_threshold=args.threshold_flow, min_area=36, max_size=100)
    time_end = time.time()
    if DEBUG:
        print("Time run detect object: ", (time_end - time_start))
    
    # tensor_bbox = np.array(list_bbox)
    # tensor_bbox = torch.from_numpy(tensor_bbox).float()
    return list_bbox

def draw_bb(bbox_img, out_track, list_bbox):
    for i in out_track :
        bbox = i.tlwh 
        # print("bbox before : ", bbox)
        x, y, w, h = bbox
        
        x = x + w/4
        y = y + h/4
        w = w/2
        h = h/2
        
        # print("bbox : ", [int(x), int(y), int(w), int(h)])
        
        # if w > 100 or h > 100 :
        #     print("draw error w, h : ", w, h)
        #     x = input()

        # if x < 290 or x > 1630 :
        #     continue
        cv2.rectangle(bbox_img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 1)
        cv2.putText(bbox_img, str(int(w)) + "x" + str(int(h)), (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, color = (0, 255, 0), fontScale = 1,  thickness = 1)
    
    # print("list_bbox : ", list_bbox)
    
    for bbox_ in list_bbox:
        x1, y1, x2, y2, score = bbox_
        x = x1
        y = y1
        w = x2 - x1
        h = y2 - y1
         
        # print("bbox moving object : ", [int(x), int(y), int(w), int(h)])
        
        cv2.rectangle(bbox_img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 1)
    
    return bbox_img


def run_bytetrack(tracker, list_bbox, height, width, matrix_stable):
    img_size = (height, width)
    
    time_t_tracker = time.time()
    out_track = tracker.update(list_bbox, img_size, img_size, matrix_stable)
    time_s_tracker = time.time()
    if DEBUG:
        print("Time run ByteTrack : ", time_s_tracker - time_t_tracker)
    return out_track

def save_log(f, out_track, frame_id, width, height):
    for i in out_track:
        bbox = i.tlwh 
        x, y, w, h = bbox
        
        x = x + w/4
        y = y + h/4
        w = w/2
        h = h/2
        
        # if w < 0 or h < 0 :
        #     print("Error w, h < 0 : ", w, h)
        #     m = input()
            
        limit = 20
            
        if x + w < limit or y + h < limit or x > width - limit or y > height - limit :
            continue      
    
        track_id = i.track_id
        score = i.score
        line = f"{frame_id},{track_id},{int(round(x))},{int(round(y))},{int(round(w))},{int(round(h))},{score:.2f},-1,-1,-1\n"
        f.write(line)
    
def demo(model, args):
    os.makedirs(args.output, exist_ok=True)
    
    # Build GME
    stab_test = StabTest()
    
    tracker = BYTETracker(args)
    
    
    
    frame_skip = 3

    with torch.no_grad():
        # print("args.input_path : ", args.input_path)
        
        images = glob.glob(os.path.join(args.input_path, '*.png')) + \
                 glob.glob(os.path.join(args.input_path, '*.jpg'))
        
        images = sorted(images)
        
        # print("images : ", images)

        height, width, _ = cv2.imread(images[0]).shape

        idx = -1
        
        file_img_old = ""
        
        name_img = args.input_path.split("/")[-2]
        
        folder_log = os.path.join(args.output, "log", name_img)
        os.makedirs(folder_log, exist_ok=True)

        folder_results = os.path.join(args.output, "results", name_img)
        os.makedirs(folder_results, exist_ok=True)

        folder_flows = os.path.join(args.output, "flows", name_img)
        os.makedirs(folder_flows, exist_ok=True)

        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            idx += 1
            if DEBUG:
                print("frame_id : ", idx)
            
            if idx % frame_skip == 0:
                image1_orin = cv2.imread(imfile1)
                # frame_disp = image1.copy()
                image1_orin = cv2.cvtColor(image1_orin, cv2.COLOR_BGR2RGB)
            
            # print("imfile1 : {}, imfile2 : {}".format(imfile1, imfile2))
            
            num_frame = os.path.basename(imfile2).split(".")[0]
            # num_frame = os.path.basename(imfile2).split(".")[0]
            
            log_path = os.path.join(folder_log, num_frame + ".txt")
            
            # print("log_path : ", log_path)
            
            f = open(log_path, "w+")
            
            if idx != 0 and idx % frame_skip == 0:
                imfile1 = file_img_old
                
            # print("imfile1 : {}, imfile2 : {}".format(imfile1, imfile2))
            
            image1 = cv2.imread(imfile1)
            # frame_disp = image1.copy()
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            
            image2 = cv2.imread(imfile2)
            frame_disp = image2.copy()
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
            
            image2_orin = image2.copy()
            
            # print("num_frame : ", num_frame)
            
            time_t_stable = time.time()
            image1_stable, matrix_stable = stabilize_image(image2, image1, method='homography')
            # image1_stable, matrix_stable = stabilize_image_fast(image2, image1, method='orb')
            
            time_s_stable = time.time()
            if DEBUG:
                print("Time stable image : ", time_s_stable - time_t_stable)
            
            if idx % frame_skip == 0 :
                _, matrix_stable = stabilize_image(image2_orin, image1_orin, method='homography')
                # _, matrix_stable = stabilize_image_fast(image2_orin, image1_orin, method='orb')
            
            # print("matrix_stable : ", matrix_stable)
            
            # x = input()
            
            print("Processing frame: ", idx + 1)
            if idx % frame_skip != 0:
                # cv2.imwrite(os.path.join(folder_results, num_frame + ".jpg"), frame_disp)
                # f.close()
                # continue
                list_bbox = [[0, 0, 0, 0, 0]]
                out_track = tracker.update_frame_skip(matrix_stable, use_multi=True)
            else:
                if idx == 0 :
                    h, w = image1.shape[:2]
                    stab_test.init(image1)
                
                file_img_old = imfile2
                
                # print("file_img_old : ", file_img_old)
                
                # print("image2.shape : ", image2.shape)
                # print("image1_stable.shape : ", image1_stable.shape)
                
                # cv2.imwrite("image2.jpg", image2)
                # cv2.imwrite("image1_stable.jpg", image1_stable)
                
                # cv2.namedWindow("image1_stable", cv2.WINDOW_NORMAL)
                # cv2.imshow("image1_stable", image1_stable)
                # cv2.namedWindow("image2", cv2.WINDOW_NORMAL)
                # cv2.imshow("image2", image2)
                # cv2.waitKey()
                
                flow = run_optical_flow(args, model, image2, image1_stable)
                
                # print("flow[0].shape : ", flow[0].shape)
                # print("flow[1].shape : ", flow[1].shape)
                # x = input()
                
                flow_save = flow[0][0].permute(1,2,0).cpu().numpy()

                flow_vis = flow_to_image(flow_save, convert_to_bgr=True)
                cv2.imwrite(f"{folder_flows}/{num_frame}_flow_final.jpg", flow_vis)

                if DEBUG:
                    print("flow[0][0] : " , flow[0][0].shape)
                # x = input()

                flow = flow[0][0].permute(1,2,0).cpu().numpy()
                
                # print("flow.shape : ", flow.shape)
                
                # x = input()

                # ✅ Code mới (tối ưu, không cần ghi/đọc file nữa)
                flows = tensor_to_flow_numpy(flow)
                
                list_bbox = run_detect_object(args, flows, [height, width])
                    
                out_track = run_bytetrack(tracker, list_bbox, height, width, matrix_stable)
            
            save_log(f, out_track, idx + 2, width, height)

            bbox_img = draw_bb(frame_disp, out_track, list_bbox)

            # Hiển thị bằng OpenCV
            cv2.namedWindow("demo_contour", cv2.WINDOW_NORMAL)
            cv2.imshow("demo_contour", bbox_img)
            cv2.waitKey(1)
            
            if DEBUG:
                print("file : ", os.path.join(folder_results, num_frame + ".jpg"))
            
            cv2.imwrite(os.path.join(folder_results, num_frame + ".jpg"), bbox_img)
            f.close()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--model', help='checkpoint path', type=str, required=True)
    parser.add_argument('--input_path', help='path to image folder', type=str, required=True)
    parser.add_argument('--output', help='path to save output', type=str, default='demo_own')
    parser.add_argument('--iters', help='number of iterations', default=12, type=int)
    parser.add_argument('--scale', help='input scale', type=int, default=0)
    parser.add_argument('--threshold_flow', help='threshold for visualizing flow', type=float, default=1.0)
 
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

    args = parse_args(parser)
    
    model = RAFT(args)
    load_ckpt(model, args.model)
    model = model.cuda()
    model.eval()

    demo(model, args)

if __name__ == '__main__':
    main()