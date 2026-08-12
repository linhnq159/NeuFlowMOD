import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
from PIL import Image

import sys

from pathlib import Path
# Add segmentation for postprocessing
repo_segment = Path("/data/linhnq11/segmentation/ILNet-master")
sys.path.append(str(repo_segment))

from model.metrics import IoUMetric, nIoUMetric, ROCMetric
from model.metrics import cal_tp_pos_fp_neg
from model.ilnet import ILNet_S, ILNet_M, ILNet_L
from model.ilnet_V1 import ILNetV1_S
from model.LW_IRSTNet import LW_IRST_ablation, LWNet_L
from model.ISTDnet import MiM
from model.LW_IRSTNet_V1 import LW_net
from model.LW_Opticalflow import LW_OpticalFLow
from core.utils.flow_viz import flow_to_image
import skimage

from argparse import ArgumentParser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = ArgumentParser(description='Implement of MY net')

    # parser.add_argument('--img_path', type=int, default='', help='image path: .png')
    # parser.add_argument('--mask_path', type=int, default=1, help='mask path: .png')
    parser.add_argument('--img_path', type=str, default='', help='image path: .png')
    parser.add_argument('--mask_path', type=str, default='', help='image path: .png')
    parser.add_argument('--mode', type=str, default='LW_V1', help='mode: L, M, S')
    parser.add_argument('--checkpoint', type=str, default='', help='checkpoint: .pth')
    parser.add_argument('--output_path', type=str, default='work_seg/run/output_img')
    parser.add_argument('--output_path_mask', type=str, default='work_seg/run/output_mask')
    parser.add_argument('--output_path_merge', type=str, default='work_seg/run/output_merge')
    parser.add_argument('--output_txt', type=str, default='work_seg/run/output_txt')
    parser.add_argument('--imshow', action='store_true')

    args = parser.parse_args()
    return args


class Demo:

    def __init__(self, model_path,  img_path, mask_path, img_size=[640,512]) :
        self.img_path = img_path
        self.mask_path = mask_path

        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.883037269115448, 0.8753958344459534, 0.8630934357643127], [0.29727908968925476, 0.29396122694015503, 0.2957575023174286]),
        ])
        self.flow_normalize = transforms.Normalize(
                    mean = [-0.0631835013628006, 0.006084870081394911], 
                    std = [9.645421028137207, 10.885346412658691])
        self.score_thresh = 1.0
        assert args.mode in ['L', 'M', 'S','SV1', 'LW', 'LW_V1','MiM', 'LW_Opticalflow']
        if args.mode == 'L':
            self.model = ILNet_L()
        elif args.mode == 'M':
            self.model = ILNet_M()
        elif args.mode == 'S':
            self.model = ILNet_S()
        elif args.mode == "SV1":
            self.model = ILNetV1_S()
        elif args.mode == "LW":
            self.model = LW_IRST_ablation(channel=(8, 32, 64), dilations=(2,4,8,16), kernel_size=(7,7,7,7), padding=(3,3,3,3))
        elif args.mode == "LW_Opticalflow" :
            self.model = LW_OpticalFLow() 
        elif args.mode == "LW_V1" :
            self.model = LW_net()
        elif args.mode == 'MiM':
            self.model = MiM([2]*3,[8, 16, 32, 64, 128])
        else:
            NameError

        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model'])
        # self.model.load_state_dict(torch.load(model_path))
        self.model.to(device)
        self.model.eval()

    def test(self):
        flow = np.load(img_path)
        img = flow.astype(np.float32)
        flow_vis = flow_to_image(flow, convert_to_bgr=True)

        size_orin = img.shape

        flow_tensor = torch.from_numpy(img.transpose(2, 0, 1))
        name_img = self.img_path.split("/")[-1].split(".")[0]
        #img = cv2.resize(img, (640, 512), interpolation=cv2.INTER_LINEAR)        

        # print(img.shape)
        flow_vis = cv2.resize(flow_vis, (size_orin[1], size_orin[0]) , interpolation=cv2.INTER_LINEAR)
        origin_img = flow_vis.copy()
        # # save image
        # cv2.imwrite('resize.jpg', img)

        flow_tensor = torch.nn.functional.interpolate(
            flow_tensor.unsqueeze(0),
            size = self.img_size,
            mode = 'nearest')

        # transform
        flow_tensor = self.flow_normalize(flow_tensor).to(device)

        with torch.no_grad():
            import time
            time_t = time.time()
            output = self.model(flow_tensor)
            time_s = time.time()
            print("Time inference : ", time_s - time_t)
            # output = torch.sigmoid(output)
            # output = output.squeeze(0).squeeze(0).cpu().numpy()            
            output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
            # print(np.sum(np.array(output)))

            output = output > self.score_thresh

            output = output.astype(np.uint8) * 255
            # output = outputs * 255
            # output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        os.makedirs(args.output_path_mask, exist_ok= True)
        os.makedirs(args.output_path_merge, exist_ok= True)
        os.makedirs(args.output_path, exist_ok= True)
        os.makedirs(args.output_txt, exist_ok= True)

        file_txt = os.path.join(args.output_txt, name_img + ".txt")

        f = open(file_txt, "w+")

        # print("sum out: ", sum(output))
        # save mask

        file_mask = os.path.join(args.output_path_mask, self.img_path.split('/')[-1].split(".")[0] + ".png")

        output = cv2.resize(output, [size_orin[1], size_orin[0]])

        # print("output : ", output.shape)

        # print("file_mask : ", file_mask)

        # print("output : ", output)

        cv2.imwrite(file_mask, output)

        # Fix the shape and data type issue
        if len(output.shape) == 3 and output.shape[2] == 1:
            # If it's a single channel, squeeze it
            output = output.squeeze(2)
        
        # Ensure output is in the correct format for PIL
        output = output.astype(np.uint8)

        # Create a PIL Image from the mask
        mask = Image.fromarray(output, mode='L')  # 'L' mode for grayscale
        # Convert mask to 3-channel for concatenation
        mask_3channel = cv2.cvtColor(np.array(mask), cv2.COLOR_GRAY2BGR)
        copy = Image.fromarray(flow_vis.astype('uint8')).convert('RGB')
        
        # Convert copy to numpy array and ensure it's 3-channel
        copy_array = np.array(copy)
        
        # Ensure both arrays have the same dimensions for hstack
        if len(mask_3channel.shape) == 2:
            mask_3channel = cv2.cvtColor(mask_3channel, cv2.COLOR_GRAY2BGR)
            
        mer = np.hstack((copy_array, mask_3channel))
        file_name = os.path.join(args.output_path, self.img_path.split('/')[-1].split(".")[0] + ".png")
        file_name_mask = os.path.join(args.output_path_merge, self.img_path.split('/')[-1].split(".")[0] + ".png")
        cv2.imwrite(file_name_mask, mer)

        # find contours
        contours, _ = cv2.findContours(output, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # print(len(contours))
        
        for contour in contours:
            # draw rectangle around contour on original image with ratio
            x, y, w, h = cv2.boundingRect(contour)
            # print(x,y,w,h)
            # x = int(x * fRate) + m_iPos
            # y = int(y * fRate)
            # w = int(w * fRate)
            # h = int(h * fRate)
            # print(x,y,w,h)

            cv2.rectangle(origin_img, (x, y), (x + w, y + h), (0, 255, 0), 1)

            f.write("0" + " " + str((x + w/2)/(self.img_size[0])) + " " + str((y + h/2) / (self.img_size[1])) + " " +  str(w/(self.img_size[0])) + " " + str(h / (self.img_size[1])) + "\n")

        # cv2.imwrite('test.jpg', origin_img)
        cv2.imwrite(file_name,origin_img)
        if args.imshow:
            cv2.imshow('test', origin_img)
        #cv2.waitKey(1)
        f.close()

if __name__ == '__main__':
    args = parse_args()
    img_size_infer = [960, 544]
    # img_size_infer = [640, 512]
    # check if input is folder
    if os.path.isdir(args.img_path):
        img_list = os.listdir(args.img_path)
        img_list = sorted(img_list)
        for img in img_list:
            if "npy" in img:
                img_path = os.path.join(args.img_path, img)
                mask_path = os.path.join(args.mask_path, img.split(".")[0] + ".png")
                demo = Demo(args.checkpoint, img_path, mask_path, img_size = img_size_infer)
                demo.test()
    else:
        demo = Demo(args.checkpoint, args.img_path, args.mask_path, img_size_infer)
        demo.test()