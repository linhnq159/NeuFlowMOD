import sys

from pathlib import Path

# Add segmentation for postprocessing
repo_segment = Path("/data/linhnq11/segmentation/ILNet-master")
sys.path.append(str(repo_segment))

import argparse
import os
import cv2
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as data

from config.parser import parse_args

sys.path.append('core')

import datasets
from raft import RAFT
from core.utils.flow_viz import flow_to_image
from core.utils.utils import load_ckpt
from core.utils import frame_utils

from model.LW_IRSTNet_V1 import LW_net
from model.LW_Opticalflow import LW_OpticalFLow

from PIL import Image

import glob

import time

from GME.StabTest import StabTest
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Demo:
    def __init__(self, args):
        self.args = args
        
        self.model = RAFT(args)
        load_ckpt(self.model, self.args.model)
        self.model = self.model.to(DEVICE)
        self.model.eval()

        self.use_mot = args.use_mot
        self.use_stable_img = args.use_stable_img

        if self.args.mode_segment == 'flow':
            self.model_seg = LW_OpticalFLow()
            checkpoint = torch.load(self.args.model_seg_path)
            self.model_seg.load_state_dict(checkpoint['model'])
            # self.model.load_state_dict(torch.load(model_path))
            self.model_seg.to(DEVICE)
            self.model_seg.eval() 
        elif self.args.mode_segment == 'image':
            self.model_seg = LW_net()
            checkpoint = torch.load(self.args.model_seg_path)
            self.model_seg.load_state_dict(checkpoint['model'])
            # self.model.load_state_dict(torch.load(model_path))
            self.model_seg.to(DEVICE)
            self.model_seg.eval() 
        # else:
        #     raise RuntimeError("Not have mode segment")

        ##### Create tracker and stable image
        # Build GME
        self.stab_test = StabTest()
        
        self.tracker = BYTETracker(self.args)

        ### Define score segment
        self.score_thresh_segment = 1.9

        ### normalize optical flow
        self.flow_normalize = transforms.Normalize(
                mean = [-0.0631835013628006, 0.006084870081394911], 
                std = [9.645421028137207, 10.885346412658691])


    def load_image(self, imfile):
        img = np.array(Image.open(imfile)).astype(np.uint8)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img[None].to(DEVICE)

    def create_color_bar(self, height, width, color_map):
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

    def add_color_bar_to_image(self, image, color_bar, orientation='vertical'):
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

    def vis_heatmap(self, name, image, heatmap):
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

    def get_heatmap(self, info, args):
        raw_b = info[:, 2:]
        log_b = torch.zeros_like(raw_b)
        weight = info[:, :2].softmax(dim=1)              
        log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=args.var_max)
        log_b[:, 1] = torch.clamp(raw_b[:, 1], min=args.var_min, max=0)
        heatmap = (log_b * weight).sum(dim=1, keepdim=True)
        return heatmap

    def forward_flow(self, image1, image2):
        output = self.model(image1, image2, iters=self.args.iters, test_mode=True)
        # flow_final = output['flow'][-1]
        # info_final = output['info'][-1]
        return output, output

    def calc_flow(self, image1, image2):
        img1 = F.interpolate(image1, scale_factor=2 ** self.args.scale, mode='bilinear', align_corners=False)
        img2 = F.interpolate(image2, scale_factor=2 ** self.args.scale, mode='bilinear', align_corners=False)
        H, W = img1.shape[2:]
        # print("Just for checking H, W: ", H, W)
        flow, info = self.forward_flow(img1, img2)
        flow_down = F.interpolate(flow, scale_factor=0.5 ** self.args.scale, mode='bilinear', align_corners=False) * (0.5 ** self.args.scale)
        info_down = F.interpolate(info, scale_factor=0.5 ** self.args.scale, mode='area')
        return flow_down, info_down
        
    # --- Hàm chuyển flow tensor sang numpy trực tiếp ---
    def tensor_to_flow_numpy(self, flow_tensor):
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

        
    def stabilize_image(self, img_ref, img_target, method='affine'):
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

        # print("M : ", M)

        return stabilized, M

    def run_optical_flow(self, image1, image2):
        if self.use_stable_img :
            time_t_stable = time.time()
            image2, _ = self.stabilize_image(image1, image2, method='homography')
            time_s_stable = time.time()
        # print("Time stable image : ", time_s_stable - time_t_stable)
        
        image1 = torch.tensor(image1, dtype=torch.float32).permute(2, 0, 1)
        image2 = torch.tensor(image2, dtype=torch.float32).permute(2, 0, 1)

        image1 = image1[None].to(DEVICE)
        image2 = image2[None].to(DEVICE)

        time_t = time.time()
        flow = self.calc_flow(image1, image2)
        time_s = time.time()
        print("Time inference : ", time_s - time_t)
        
        return flow


    # --- Hàm vẽ optical flow dưới dạng HSV (màu sắc trực quan) ---
    def flow_to_rgb(self, flow):
        h, w = flow.shape[:2]
        fx, fy = flow[..., 0], flow[..., 1]
        mag, ang = cv2.cartToPolar(fx, fy)
        rgb = None
        # hsv = np.zeros((h, w, 3), dtype=np.uint8)
        # hsv[..., 0] = ang * 180 / np.pi / 2
        # hsv[..., 1] = 255
        # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return rgb, mag    

    def detect_moving_objects_from_segment(self, output_seg, min_area=100, max_size=100):
        output = output_seg.squeeze(0).permute(1, 2, 0).cpu().numpy()
        # print(np.sum(np.array(output)))

        output = output > self.score_thresh_segment

        output = output.astype(np.uint8) * 255

        # output = cv2.resize(output, [960, 540])

        contours, _ = cv2.findContours(output, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # print("x, y, w, h : ", x, y, w, h)
            
            # if w*h > min_area and w < max_size and h < max_size and x < pos_limit and y < pos_limit and x < img_size[1] - pos_limit and y < img_size[0] - pos_limit:  # bỏ qua các bbox quá nhỏ
            if w*h > min_area and w < max_size and h < max_size :  # bỏ qua các bbox quá nhỏ
                # print("add old x, y, w, h : ", x, y, w, h)
                boxes.append([x  - w/2,y - h/2,x + w * 1.5, y + h * 1.5 ,1])
            
            # if w*h > min_area and w < max_size and h < max_size and x < pos_limit and y < pos_limit and x < (img_size[1] - pos_limit) and y < (img_size[0] - pos_limit):  # bỏ qua các bbox quá nhỏ
            # if w*h > min_area and w < max_size and h < max_size :  # bỏ qua các bbox quá nhỏ
                # print("add x, y, w, h : ", x, y, w, h)
                # boxes.append([x  - w/2,y - h/2,x + w * 1.5, y + h * 1.5 ,1])

        if len(boxes) < 1 :
            boxes.append([0,0,0,0,0])

        print("boxes : ", boxes)

        tensor_bbox = np.array(boxes)
        tensor_bbox = torch.from_numpy(tensor_bbox).float()

        return tensor_bbox

    # --- Xử lý phát hiện vật chuyển động ---
    def detect_moving_objects(self, flow, mag_threshold=1.0, min_area=100, max_size=100):
        _, mag = self.flow_to_rgb(flow)
        
        mask = (mag > mag_threshold).astype(np.uint8) * 255
        
        pos_limit = 10
        
        # cv2.namedWindow("mask before clean", cv2.WINDOW_NORMAL)
        # cv2.imshow("mask before clean", mask)
        # cv2.waitKey(1)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # print("x, y, w, h : ", x, y, w, h)
            
            # if w*h > min_area and w < max_size and h < max_size and x < pos_limit and y < pos_limit and x < img_size[1] - pos_limit and y < img_size[0] - pos_limit:  # bỏ qua các bbox quá nhỏ
            if w*h > min_area and w < max_size and h < max_size :  # bỏ qua các bbox quá nhỏ
                # print("add old x, y, w, h : ", x, y, w, h)
                boxes.append([x  - w/2,y - h/2,x + w * 1.5, y + h * 1.5 ,1])
            
            # if w*h > min_area and w < max_size and h < max_size and x < pos_limit and y < pos_limit and x < (img_size[1] - pos_limit) and y < (img_size[0] - pos_limit):  # bỏ qua các bbox quá nhỏ
            # if w*h > min_area and w < max_size and h < max_size :  # bỏ qua các bbox quá nhỏ
                # print("add x, y, w, h : ", x, y, w, h)
                # boxes.append([x  - w/2,y - h/2,x + w * 1.5, y + h * 1.5 ,1])

        if len(boxes) < 1 :
            boxes.append([0,0,0,0,0])

        return boxes

    def run_detect_object(self, flows):
        # --- Phát hiện đối tượng ---
        time_start = time.time()
        list_bbox = self.detect_moving_objects(flows, mag_threshold=2, min_area=36, max_size=100)
        time_end = time.time()
        print("Time run detect object: ", (time_end - time_start))
        
        tensor_bbox = np.array(list_bbox)
        tensor_bbox = torch.from_numpy(tensor_bbox).float()
        return tensor_bbox

    def draw_bb(self, bbox_img, out_track, tensor_bbox):
        if self.use_mot:
            for i in out_track :
                bbox = i.tlwh 
                # print("bbox before : ", bbox)
                x, y, w, h = bbox
                
                x = x + w/4
                y = y + h/4
                w = w/2
                h = h/2
                
                # print("bbox : ", [int(x), int(y), int(w), int(h)])
                
                cv2.rectangle(bbox_img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 1)
                cv2.putText(bbox_img, str(int(w)) + "x" + str(int(h)), (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, color = (0, 255, 0), fontScale = 1,  thickness = 1)
        else:
            for bbox_ in tensor_bbox :
                bbox = bbox_
                # print("bbox before : ", bbox)
                x, y, x2, y2, _ = bbox
            
                w = x2 - x
                h = y2 - y

                x = x + w/4
                y = y + h/4
                w = w/2
                h = h/2
                
                # print("bbox : ", [int(x), int(y), int(w), int(h)])
                
                cv2.rectangle(bbox_img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 1)
                cv2.putText(bbox_img, str(int(w)) + "x" + str(int(h)), (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, color = (0, 255, 0), fontScale = 1,  thickness = 1)

        return bbox_img


    def run_bytetrack(self, tensor_bbox, height, width):
        img_size = (height, width)
        
        time_t_tracker = time.time()
        out_track = self.tracker.update(tensor_bbox, img_size, img_size)
        time_s_tracker = time.time()
        # print("Time run ByteTrack : ", time_s_tracker - time_t_tracker)
        return out_track

    def save_log(self, f, out_track, tensor_bbox, frame_id):
        if self.use_mot:
            for i in out_track:
                bbox = i.tlwh 
                x, y, w, h = bbox
                
                x = x + w/4
                y = y + h/4
                w = w/2
                h = h/2
                
                if w < 0 or h < 0 :
                    print("Error w, h < 0 : ", w, h)
                    m = input()
                    
                track_id = i.track_id
                score = i.score
                line = f"{frame_id},{track_id},{int(round(x))},{int(round(y))},{int(round(w))},{int(round(h))},{score:.2f},-1,-1,-1\n"
                f.write(line)
        else:
            for i, tensor_bbox_ in enumerate(tensor_bbox):
                bbox = tensor_bbox_
                print("tensor_bbox_ : ", tensor_bbox_)
                x, y, x2, y2, score = bbox
                
                x = x
                y = y
                w = x2 - x
                h = y2 - y

                x = x + w/4
                y = y + h/4
                w = w/2
                h = h/2
                
                if w < 0 or h < 0 :
                    print("Error w, h < 0 : ", w, h)
                    m = input()
                    
                track_id = i
                line = f"{frame_id},{track_id},{int(round(float(x)))},{int(round(float(y)))},{int(round(float(w)))},{int(round(float(h)))},{score:.2f},-1,-1,-1\n"
                f.write(line)
        
    def demo(self):
        os.makedirs(self.args.output, exist_ok=True)

        video_file_name = self.args.output.split("/")[-1]

        images = glob.glob(os.path.join(self.args.input_path, '*.png')) + \
                    glob.glob(os.path.join(self.args.input_path, '*.jpg')) + \
                    glob.glob(os.path.join(self.args.input_path, '*.PNG'))
            
        images = sorted(images)
        images1 = cv2.imread(images[0])    


        size = (images1.shape[1], images1.shape[0])
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(os.path.join(self.args.output, video_file_name), fourcc, 15, size)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.883037269115448, 0.8753958344459534, 0.8630934357643127], [0.29727908968925476, 0.29396122694015503, 0.2957575023174286]),
        ])

        with torch.no_grad():
            print("args.input_path : ", self.args.input_path)
            
            images = glob.glob(os.path.join(self.args.input_path, '*.png')) + \
                    glob.glob(os.path.join(self.args.input_path, '*.jpg')) + \
                    glob.glob(os.path.join(self.args.input_path, '*.PNG'))
            
            images = sorted(images)
            
            print("images : ", images)

            height, width, _ = cv2.imread(images[0]).shape

            idx = -1
            
            file_img_old = ""
            
            name_img = self.args.input_path.split("/")[-2]
            
            folder_log = os.path.join(self.args.output, "log", name_img)
            os.makedirs(folder_log, exist_ok=True)

            folder_results = os.path.join(self.args.output, "results", name_img)
            print("folder_results : ", folder_results)
            os.makedirs(folder_results, exist_ok=True)

            folder_flows = os.path.join(self.args.output, "flows", name_img)
            os.makedirs(folder_flows, exist_ok=True)

            frame_skip = 3

            for imfile1, imfile2 in zip(images[:-frame_skip], images[frame_skip:]):
                
                num_frame = os.path.basename(imfile2).split(".")[0]
                
                log_path = os.path.join(folder_log, num_frame + ".txt")

                print("log_path : ", log_path)
                
                f = open(log_path, "w+")

                # frameId = int(os.path.basename(imfile2)[5:9])
                frameId = int(os.path.basename(imfile2).split(".")[0])

                print("frameId : ", frameId)
                
                idx += 1
                print("Processing frame: ", frameId)
                # if idx % 3 != 0:
                #     f.close()
                #     continue

                # if idx != 0 :
                #     imfile1 = file_img_old
                
                image1 = cv2.imread(imfile1)
                # frame_disp = image1.copy()
                image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
                
                if idx == 0 :
                    h, w = image1.shape[:2]
                    self.stab_test.init(image1)
                
                image2 = cv2.imread(imfile2)

                frame_disp = image2.copy()
                
                image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

                print("imfile1 : {}, imfile2 : {}".format(imfile1, imfile2))
                
                # file_img_old = imfile2
                
                flow = self.run_optical_flow(image2, image1)
                
                name = os.path.basename(imfile2).split('.')[0]
                
                flow_save = flow[0][0].permute(1,2,0).cpu().numpy()

                flow_vis = flow_to_image(flow_save, convert_to_bgr=True)
                
                if self.args.mode_segment == "flow":
                    print("len(flow) : ", len(flow))
                    print("flow.shape : ", flow[0].shape)
                    output_seg = self.model_seg(self.flow_normalize(flow[0]))
                    tensor_bbox = self.detect_moving_objects_from_segment(output_seg, min_area= 10, max_size= 100)
                elif self.args.mode_segment == "image":
                    output_seg = self.model_seg(transform(flow_vis).unsqueeze(0).to(DEVICE))
                    # print("output_seg : ", output_seg)
                    tensor_bbox = self.detect_moving_objects_from_segment(output_seg, min_area= 10, max_size= 100)
                else:
                    flow = flow[0][0].permute(1,2,0).cpu().numpy()
                    # ✅ Code mới (tối ưu, không cần ghi/đọc file nữa)
                    flows = self.tensor_to_flow_numpy(flow)
                    tensor_bbox = self.run_detect_object(flows)

                for tensor_bbox_ in tensor_bbox :
                    cv2.rectangle(flow_vis, (int(tensor_bbox_[0]), int(tensor_bbox_[1])), (int(tensor_bbox_[2]), int(tensor_bbox_[3])), (0, 255, 0), thickness=2)
                    cv2.imwrite(f"{folder_flows}/{name}_flow_final.jpg", flow_vis)

                print("tensor_bbox : ", tensor_bbox)

                if self.use_mot:
                    out_track = self.run_bytetrack(tensor_bbox, height, width)
                else:
                    out_track = []
                
                print("out_track : ", out_track)

                self.save_log(f, out_track, tensor_bbox, frameId)

                bbox_img = self.draw_bb(frame_disp, out_track, tensor_bbox)

                # # Hiển thị bằng OpenCV
                # cv2.namedWindow("demo_contour", cv2.WINDOW_NORMAL)
                # cv2.imshow("demo_contour", bbox_img)
                # cv2.waitKey(1)
                
                cv2.imwrite(os.path.join(folder_results, str(frameId).zfill(4) + "_bb.png"), bbox_img)
                f.close()
                video_writer.write(frame_disp)
            # cv2.destroyAllWindows()
        video_writer.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--model', help='checkpoint path', type=str, required=True)
    parser.add_argument('--input_path', help='path to image folder', type=str, required=True)
    parser.add_argument('--output', help='path to save output', type=str, default='demo_own')
    parser.add_argument('--iters', help='number of iterations', default=12, type=int)
    parser.add_argument('--scale', help='input scale', type=int, default=0)
    parser.add_argument('--use_mot', action='store_true')
    parser.add_argument('--use_stable_img', action='store_true')
 
    # model segmentation
    parser.add_argument('--model_seg_path', help = "path model segmentation ILNet", type = str, default = "")
    parser.add_argument('--mode_segment', help = "image: mode gen flow; flow : mode input optical flow", type = str, default= "flow")

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
    
    demo_ = Demo(args)
    demo_.demo()

    # demo(model, args, model_seg)

if __name__ == '__main__':
    main()