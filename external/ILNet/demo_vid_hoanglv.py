import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import os

from model.metrics import IoUMetric, nIoUMetric, ROCMetric
from model.metrics import cal_tp_pos_fp_neg
from model.ilnet import ILNet_S, ILNet_M, ILNet_L
from model.ilnet_V1 import ILNetV1_S
from model.LW_IRSTNet import LW_IRST_ablation
from model.LW_IRSTNet_V1 import LW_net
from argparse import ArgumentParser
from PIL import Image, ImageOps, ImageFilter
import time
from tqdm import tqdm
import skimage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = ArgumentParser(description='Implement of MY net')


parser.add_argument('--vid_path', type=str, default='', help='video path', required=True)
parser.add_argument('--mode', type=str, default='LW_V1', help='mode: L, M, S')
parser.add_argument('--checkpoint', type=str, help='checkpoint path', default="/mnt/data/danhnt/Seg/work_seg/checkpoint/weights/LW_V1_ep292.pth")
parser.add_argument('--work_dir', type=str, default='../work_seg/result_infer_vid')
parser.add_argument('--img_size', type=list, default=[640,512], help='image size')
parser.add_argument('--thresh_score', type=float, default=0.85, help='threshold of score')
parser.add_argument('--show_result', type=bool, default=False)
parser.add_argument('--output_file', help='output MOT detection labels')
args = parser.parse_args()


class Demo:

    def __init__(self, args):
        self.args = args
        self.vid_path = args.vid_path
        self.output_file = args.output_file
        self.mode = args.mode
        self.checkpoint = args.checkpoint
        self.img_size = [640,512]

        self.model = self._load_model()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.45563722, 0.45559042, 0.45564058], [0.2213023,  0.22129012, 0.22129919]),
        ])
        self.score_thresh = args.thresh_score

    def _load_model(self):
        if self.mode == 'L':
            model = ILNet_L()
        elif self.mode == 'M':
            model = ILNet_M()
        elif self.mode == 'S':
            model = ILNet_S()
        elif self.mode == 'SV1':
            model = ILNetV1_S()
        elif self.mode == 'LW':
            model = LW_IRST_ablation(channel=(8, 32, 64), dilations=(2,4,8,16), kernel_size=(7,7,7,7), padding=(3,3,3,3))
        elif args.mode == "LW_V1" :
            model = LW_net()

        else:
            raise ValueError('mode should be L, M, S')

        checkpoint = torch.load(self.checkpoint)
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        model.eval()
        return model

    def _preprocess(self, img):
        self.img = img
        # get shape
        h  = img.shape[0]
        w = img.shape[1]

        # resize with ratio and padding
        if h > w:
            if h > 512:
                ratio = self.args.img_size[1] / h
            else:
                ratio = 1
            img = cv2.resize(img, (int(w*ratio), self.args.img_size))
            img = np.pad(img, ((0, 0), (0, self.args.img_size - img.shape[1]), (0, 0)), 'constant', constant_values=0)
        else:
            if w > 640:
                ratio = self.args.img_size[0] / w
            else:
                ratio = 1
            img = cv2.resize(img, (self.args.img_size, int(h*ratio)))
            img = np.pad(img, ((0, self.args.img_size - img.shape[0]), (0, 0), (0, 0)), 'constant', constant_values=0)

        return img

    def transform_val(self,img):
        w, h = img.shape[1], img.shape
        if (w/self.img_size[0]) > (h/self.img_size[1]):
            ow = self.img_size[0]
            oh = int(h * self.img_size[0] / w)

        else:
            oh = self.img_size[1]
            ow = int(w * self.img_size[1] / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        # padding
        padh = max(self.img_size[1] - oh, 0)
        padw = max(self.img_size[0] - ow, 0)
        img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)

        return img
        



    def _get_mask(self, img):
        with torch.no_grad():
            output = self.model(img).squeeze(0).permute(1, 2, 0).cpu()
            predict = (output.detach().numpy() > self.score_thresh).astype(np.uint8)

        return predict*255



    def save_video(self):
        # cap = cv2.VideoCapture(self.vid_path, 1400)
        # set frame id to 1000
        cap = cv2.VideoCapture(self.vid_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_path = self.vid_path.split('/')[-1]
        vid_name = os.path.join(self.args.work_dir, 'result')
        if not os.path.isdir(vid_name):
            os.makedirs(vid_name, exist_ok=True)
        # out = cv2.VideoWriter(vid_name + '/' + out_path, fourcc, fps, (1920, 1080))
        # create folder log to save mask 
        # if not os.path.isdir(vid_name + '/log' + '/'+ out_path):
            # os.makedirs(vid_name + '/log' + '/'+ out_path, exist_ok=True)
        # img_save = './extr/images/'
        # mask_save = './extr/masks/'
        frameID = 0
        # f = open("log.txt","a")
        k = 0
        with open(self.output_file, 'w') as f:
            while True:

                ret, frame = cap.read()
                if not ret:
                    break
                # frame_id = cap.get(1)
                

                # replace all pixel of mask  have y > 465 and y < 631 and x < 980 and x > 988 by another region

                #crop = frame[0 : 100, 960 -150 - 400: 960 + 150 - 400]
                #frame[0 : 100, 960 -150: 960 +150] = crop

                #crop = frame[465 - 200 :631 - 200, 975:990]
                #frame[465 :631, 975:990] = crop


                #crop = frame[540: 555, 868 - 300 : 1100 - 300]
                #frame[540: 555, 868 : 1100] = crop
                # frame = frame[512 - 256: 512+256, 960 - 320: 960+ 320]
                
                
                


                # median blur
                # frame = cv2.medianBlur(frame, 7)
                

                # frame = skimage.exposure.adjust_gamma(frame, gamma=2.5)
                
                # increase brightness
                #frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=0.0)
                copy = frame.copy()  

                

                # enhance contrast
                # frame = skimage.exposure.contrast_stretch(frame)

                w, h = 640, 512
                origin_w, origin_h = frame.shape[1], frame.shape[0]
                fRate = origin_h / h
                m_iPos = (origin_w - fRate * w) / 2
                m_iPos = int(m_iPos)
                # m_iPos = 320
                # print(m_iPos)
                crop = frame[:, m_iPos: origin_w - m_iPos]
                crop_x, crop_y = crop.shape[1], crop.shape[0]
                # print(crop.shape)
                # resize keep ratio and padding to w,h
                # new_w, new_h = crop.shape[1], crop.shape[0]

                # if new_w/w > new_h/h:
                #     ow = w
                #     oh = int(new_h * w / new_w)
                # else:
                #     oh = h
                #     ow = int(new_w * h / new_h)

                # frame = cv2.resize(crop, (ow, oh), interpolation=cv2.INTER_LINEAR)
                # frame = cv2.copyMakeBorder(frame, 0, h - oh, 0, w - ow, cv2.BORDER_CONSTANT, value=[0, 0, 0])


                frame = cv2.resize(crop, (640, 512), interpolation=cv2.INTER_CUBIC)
                
                
                # if (frameID < 1000) and (frameID %100 ==0):
                #     k = k+1
                #     cv2.imwrite("sample" + str(k) + '.png', frame)
                
                # frame = self.transform_val(frame)
                # frame = cv2.resize(frame, (self.args.frame_size, self.args.img_size))
                frame_tensor = self.transform(frame).unsqueeze(0).to(device)
                # get mark and hstack with frame
                mask = self._get_mask(frame_tensor)
                # convert mask to 3 channel and hstack with frame   
                
                # copy = Image.fromarray(frame.astype('uint8')).convert('RGB')
                # mask = Image.fromarray(mask.astype('uint8')).convert('RGB')
                # frame = np.hstack((np.array(copy), np.array(mask)))
                # mask = mask[0, :, :]
                # mask = np.array(mask)
                # mask = cv2.resize(mask, dsize= (h,  w), interpolation=cv2.INTER_NEAREST)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # write contour to file split by comma
                mask = np.dstack((mask, mask, mask))
                # resize mask to frame original size
                mask = cv2.resize(mask, (crop_x, crop_y), interpolation=cv2.INTER_NEAREST)
                # padding to original size
                mask = cv2.copyMakeBorder(mask, 0, origin_h - crop_y, m_iPos, origin_w - m_iPos - crop_x, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                # save mask to log folder
                # cv2.imwrite(vid_name + '/log/' + out_path + '/' + str(int(frameID)) + '.png', mask)
                # cv2.drawContours(copy,contours, -1, (0,0,255), 1)
                # draw bounding box

                # merge all box to 1 if distance center < 20
                for i in range(len(contours)):
                    x, y, w, h = cv2.boundingRect(contours[i])
                    x = int(x * fRate) + m_iPos
                    y = int(y * fRate)
                    w = int(w * fRate)
                    h = int(h * fRate)
                    if len(contours) > 1:
                        for j in range(len(contours)):
                            # print(j)
                            if j != i:
                                x1, y1, w1, h1 = cv2.boundingRect(contours[j])
                                x1 = int(x1 * fRate) + m_iPos
                                y1 = int(y1 * fRate)
                                w1 = int(w1 * fRate)
                                h1 = int(h1 * fRate)
                                if abs((x + w/2) - (x1 + w1/2)) < 10 and abs((y + h/2) - (y1 + h1/2)) < 10:
                                    x = min(x, x1)
                                    y = min(y, y1)
                                    w = max(x + w, x1 + w1) - x
                                    h = max(y + h, y1 + h1) - y
                                    # contours[j] = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
                    f.write(f"{frameID+1},-1,{x},{y},{w},{h},1,0,1\n")
                    print(f"{frameID+1},-1,{x},{y},{w},{h},1,0,1")
                    # cv2.rectangle(copy, (x-3, y-3), (x + w+3, y + h+3), (0, 255, 0), 1)
            
                # for contour in contours:
                #     x, y, w, h = cv2.boundingRect(contour)
                #     x = int(x * fRate) + m_iPos
                #     y = int(y * fRate)
                #     w = int(w * fRate)
                #     h = int(h * fRate)
                #     cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 255, 0), 2)


                # tem = ""
                # for co in coords:
                #     tem += str(co[0]) + " " + str(co[1]) + ";"
                # f.write(str(frameID) + ": " + tem + "\n")
                # imwrite frame_id to frame 
                # cv2.putText(copy,'Frame Id: ' + str(int(frameID)), (2,17), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)
                # out.write(copy)
                
                # print('fps: ', fps)
                # cv2.imshow('frame', copy)
                # out.write(copy)
                # cv2.imshow('mask', mask)
                # mask = np.array(mask)

                # write mask  and frame after 10 frame
                # if (frame_id )% 150 == 0:
                #     # replace all pixel of mask  have y > 329 to 0
                #     mask[130:165,35:135] = 0
                #     mask[300:335,270:351] = 0
                #     # mask[361:,:] = 0
                #     cv2.imwrite(img_save + 'A' + str(int(frame_id)) + '.png', frame)
                #     cv2.imwrite(mask_save + 'A' + str(int(frame_id)) + '.png', mask)
                frameID += 1
                # print("frameID: ", frameID)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
            cap.release()
            # out.release()
            # f.close()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    demo = Demo(args)
    start = time.time()
    demo.save_video()
    end = time.time()
    print('Time: ', end - start)
    

    
