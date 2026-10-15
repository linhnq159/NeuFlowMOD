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

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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

def crop_img(image):
    height, width = image.shape[:2]
    
    print(height, width)

    crop_width = 960
    crop_height = 540

    center_x = width // 2
    center_y = height // 2

    x_start = max(center_x - crop_width // 2, 0)
    y_start = max(center_y - crop_height // 2, 0)

    x_end = x_start + crop_width
    y_end = y_start + crop_height

    x_end = min(x_end, width)
    y_end = min(y_end, height)

    cropped_image = image[y_start:y_end, x_start:x_end]

    return cropped_image

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
    output = model(image1, image2, iters=args.iters, test_mode=True)
    flow_final = output['flow'][-1]
    info_final = output['info'][-1]
    return flow_final, info_final

def calc_flow(args, model, image1, image2):
    print("image1 : ", image1.shape)
    img1 = F.interpolate(image1, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    img2 = F.interpolate(image2, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    H, W = img1.shape[2:]
    flow, info = forward_flow(args, model, img1, img2)
    flow_down = F.interpolate(flow, scale_factor=0.5 ** args.scale, mode='bilinear', align_corners=False) * (0.5 ** args.scale)
    info_down = F.interpolate(info, scale_factor=0.5 ** args.scale, mode='area')
    return flow_down, info_down

@torch.no_grad()
def demo_data(name, args, model, image1, image2, flow_gt):
    path = f"demo/{name}/"
    os.system(f"mkdir -p {path}")
    H, W = image1.shape[2:]
    cv2.imwrite(f"{path}image1.jpg", cv2.cvtColor(image1[0].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{path}image2.jpg", cv2.cvtColor(image2[0].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR))
    flow_gt_vis = flow_to_image(flow_gt[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True)
    cv2.imwrite(f"{path}gt.jpg", flow_gt_vis)
    flow, info = calc_flow(args, model, image1, image2)
    flow_vis = flow_to_image(flow[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True)
    cv2.imwrite(f"{path}flow_final.jpg", flow_vis)
    diff = flow_gt - flow
    diff_vis = flow_to_image(diff[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True)
    cv2.imwrite(f"{path}error_final.jpg", diff_vis)
    heatmap = get_heatmap(info, args)
    vis_heatmap(f"{path}heatmap_final.jpg", image1[0].permute(1, 2, 0).cpu().numpy(), heatmap[0].permute(1, 2, 0).cpu().numpy())
    epe = torch.sum((flow - flow_gt)**2, dim=1).sqrt()
    print(f"EPE: {epe.mean().cpu().item()}")

@torch.no_grad()
def demo_chairs(model, args, device=torch.device('cuda')):
    dataset = datasets.FlyingChairs(split='training')
    image1, image2, flow_gt, _ = dataset[1345]
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)
    demo_data('chairs', args, model, image1, image2, flow_gt)

def demo_sintel(model, args, device=torch.device('cuda')):
    dstype = 'final'
    dataset = datasets.MpiSintel(split='training', dstype=dstype)
    image1, image2, flow_gt, _ = dataset[100]
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)
    demo_data('sintel', args, model, image1, image2, flow_gt)

@torch.no_grad()
def demo_spring(model, args, device=torch.device('cuda'), split='train'):
    dataset = datasets.SpringFlowDataset(split=split)
    idx = 19198
    if split == 'train' or split == 'val':
        image1, image2, flow_gt, _ = dataset[idx]
    else:
        image1, image2,  _ = dataset[idx]
        h, w = image1.shape[1:]
        flow_gt = torch.zeros((2, h, w))

    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)
    demo_data('spring', args, model, image1, image2, flow_gt)

@torch.no_grad()
def demo_tartanair(model, args, device=torch.device('cuda')):
    dataset = datasets.TartanAir()
    image1, image2, flow_gt, _ = dataset[1070]
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)
    demo_data('tartanair', args, model, image1, image2, flow_gt)

@torch.no_grad()
def demo_infinigen(model, args, device=torch.device('cuda')):
    dataset = datasets.Infinigen()
    image1, image2, flow_gt, _ = dataset[1000]
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)
    demo_data('infinigen', args, model, image1, image2, flow_gt)

@torch.no_grad()
def demo_hd1k(model, args, device=torch.device('cuda')):
    dataset = datasets.HD1K()
    print(len(dataset))
    image1, image2, flow_gt, _ = dataset[0]
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)
    demo_data('hd1k', args, model, image1, image2, flow_gt)

@torch.no_grad()
def demo_middlebury(model, args, device=torch.device('cuda')):
    dataset = datasets.Middlebury()
    image1, image2, flow_gt, _ = dataset[3]
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)
    demo_data('middlebury', args, model, image1, image2, flow_gt)

@torch.no_grad()
def demo_custom(model, args, device=torch.device('cuda')):
    image1 = cv2.imread('../custom_images/0011.png')
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.imread('../custom_images/0012.png')
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    image1 = torch.tensor(image1, dtype=torch.float32).permute(2, 0, 1)
    image2 = torch.tensor(image2, dtype=torch.float32).permute(2, 0, 1)
    H, W = image1.shape[1:]
    flow_gt = torch.zeros([2, H, W], device=device)
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)
    demo_data('custom_downsample', args, model, image1, image2, flow_gt)
    
def stabilize_image(img_ref, img_target, method='affine'):
    # Chuyển ảnh sang grayscale
    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    gray_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)

    # Dò đặc trưng và mô tả (dùng ORB)
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(gray_ref, None)
    kp2, des2 = orb.detectAndCompute(gray_target, None)

    # Ghép đặc trưng
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:200]  # lấy 200 điểm tốt nhất

    # Lấy tọa độ điểm tương ứng
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    # Tính ma trận biến đổi (affine hoặc homography)
    if method == 'affine':
        M, _ = cv2.estimateAffinePartial2D(pts2, pts1)
        stabilized = cv2.warpAffine(img_target, M, (img_ref.shape[1], img_ref.shape[0]))
    else:
        M, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
        stabilized = cv2.warpPerspective(img_target, M, (img_ref.shape[1], img_ref.shape[0]))

    print("M : ", M)

    return stabilized, M
    
    
def demo(model, args):
    os.makedirs(args.output, exist_ok=True)
    
    # Build GME
    stab_test = StabTest()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.input_path, '*.png')) + \
                 glob.glob(os.path.join(args.input_path, '*.jpg'))
        
        images = sorted(images)
        idx = -1
        
        file_img_old = ""
        
        delta_x, delta_y = 0, 0
        
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            idx += 1
            if idx % 3 != 0:
                continue

            if idx != 0 :
                imfile1 = file_img_old
            
            image1 = cv2.imread(imfile1)
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            
            if idx == 0 :
                h, w = image1.shape[:2]
                stab_test.init(image1)
            
            # image1 = crop_img(image1)
            image2 = cv2.imread(imfile2)
            
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
            # image2 = crop_img(image2)

            image2, matrix_stable = stabilize_image(image1, image2, method='homography')

            # print("image : ", imfile1, imfile2)
            
            file_img_old = imfile2
            
            image1 = torch.tensor(image1, dtype=torch.float32).permute(2, 0, 1)
            image2 = torch.tensor(image2, dtype=torch.float32).permute(2, 0, 1)
            
            image1 = image1[None].to(DEVICE)
            image2 = image2[None].to(DEVICE)
            
            time_t = time.time()
            flow = calc_flow(args, model, image1, image2)
            
            time_s = time.time()
            
            print("Time inference : ", time_s - time_t)
            
            print("flow.shape : ", flow[1].shape)
            
            flow = flow[0][0].permute(1,2,0).cpu().numpy()
            
            # flow = frame_utils.filter_flow(flow, T_mag=3.0, min_area=25, max_area=10000)
            
            name = os.path.basename(imfile2).split('.')[0]
            
            output_file = f"{args.output}/{name}.flo"
            
            frame_utils.writeFlow(output_file, flow)

            print("flow_to_image.shape : ", flow.shape)

            flow_vis = flow_to_image(flow, convert_to_bgr=True)
            cv2.imwrite(f"{args.output}/{name}_flow_final.jpg", flow_vis)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--model', help='checkpoint path', type=str, required=True)
    parser.add_argument('--input_path', help='path to image folder', type=str, required=True)
    parser.add_argument('--output', help='path to save output', type=str, default='demo_own')
    parser.add_argument('--iters', help='number of iterations', default=12, type=int)
    parser.add_argument('--scale', help='input scale', type=int, default=0)
 
    args = parse_args(parser)
    model = RAFT(args)
    load_ckpt(model, args.model)
    model = model.cuda()
    model.eval()

    demo(model, args)

if __name__ == '__main__':
    main()