import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
from PIL import Image
from model.metrics import IoUMetric, nIoUMetric, ROCMetric
from model.metrics import cal_tp_pos_fp_neg
from model.ilnet import ILNet_S, ILNet_M, ILNet_L
from model.ilnet_V1 import ILNetV1_S
from model.LW_IRSTNet import LW_IRST_ablation, LWNet_L
from model.ISTDnet import MiM
from model.LW_IRSTNet_V1 import LW_net
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
        self.score_thresh = 0.0
        assert args.mode in ['L', 'M', 'S','SV1', 'LW', 'LW_V1','MiM']
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
        img = cv2.imread(self.img_path)
        name_img = self.img_path.split("/")[-1].split(".")[0]
        print("self.img_path : " , self.img_path)
        #img = cv2.resize(img, (640, 512), interpolation=cv2.INTER_LINEAR)

        # origin_img = img.copy()
        # w, h = 640, 512
        # origin_w, origin_h = img.shape[1], img.shape[0]
        # fRate = origin_h / h
        # # fRate = 1
        # m_iPos = (origin_w - fRate * w) / 2
        # m_iPos = int(m_iPos)
        # # m_iPos = 0
        # # # print(m_iPos)
        # crop = img[:, m_iPos: origin_w - m_iPos]
        # # print(crop.shape)
        # # resize keep ratio and padding to w,h
        # new_w, new_h = crop.shape[1], crop.shape[0]

        # if new_w/w > new_h/h:
        #     ow = w
        #     oh = int(new_h * w / new_w)
        # else:
        #     oh = h
        #     ow = int(new_w * h / new_h)

 
        # img = cv2.resize(crop, (ow, oh), interpolation=cv2.INTER_LINEAR)
        
        # #  copy make boder to w,h
        # img = cv2.copyMakeBorder(img, 0, h - oh, 0, w - ow, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        # print(img.shape)
        img = cv2.resize(img, self.img_size , interpolation=cv2.INTER_LINEAR)
        origin_img = img.copy()
        # # save image
        # cv2.imwrite('resize.jpg', img)

        print("img : ", img.shape)

        # transform
        # img = self.transform(img).unsqueeze(0).to(device)
        img = transforms.ToTensor()(img).unsqueeze(0).to(device)



        with torch.no_grad():
            output = self.model(img)
            print("output : ", output.shape)
            # output = torch.sigmoid(output)
            # output = output.squeeze(0).squeeze(0).cpu().numpy()
            # output = output > self.score_thresh
            output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
            print("output : ", output.shape)
            # print(np.sum(np.array(output)))
            output = output * 255
            print("output : ", output)

        os.makedirs(args.output_path_mask, exist_ok= True)
        os.makedirs(args.output_path, exist_ok= True)
        os.makedirs(args.output_txt, exist_ok= True)

        print("args.output_txt : ", args.output_txt)
        print("args.output_txt is dir : ", os.path.isdir(args.output_txt))

        file_txt = os.path.join(args.output_txt, name_img + ".txt")

        f = open(file_txt, "w+")

        # print("sum out: ", sum(output))
        # save mask
        # cv2.imwrite('mask1.jpg', output)
        print("output : ", output.shape)
        
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
        copy = Image.fromarray(origin_img.astype('uint8')).convert('RGB')
        
        # Convert copy to numpy array and ensure it's 3-channel
        copy_array = np.array(copy)
        
        # Ensure both arrays have the same dimensions for hstack
        if len(mask_3channel.shape) == 2:
            mask_3channel = cv2.cvtColor(mask_3channel, cv2.COLOR_GRAY2BGR)
            
        mer = np.hstack((copy_array, mask_3channel))
        file_name = os.path.join(args.output_path, self.img_path.split('/')[-1])
        file_name_mask = os.path.join(args.output_path_mask, self.img_path.split('/')[-1])
        cv2.imwrite(file_name_mask, mer)

        # cv2.imwrite('test.jpg', origin_img)
        cv2.imwrite(file_name, origin_img)
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
            if "png" in img or "jpg" in img :
                img_path = os.path.join(args.img_path, img)
                mask_path = os.path.join(args.mask_path, img)
                demo = Demo(args.checkpoint, img_path, mask_path, img_size = img_size_infer)
                demo.test()
    else:
        demo = Demo(args.checkpoint, args.img_path, args.mask_path, img_size_infer)
        demo.test()





            





