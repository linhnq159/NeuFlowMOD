import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
import math

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
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = ArgumentParser(description='Implement of MY net')


parser.add_argument('--vid_path', type=str, default='', help='video path', required=True)
parser.add_argument('--mode', type=str, default='LW_V1', help='mode: L, M, S')
parser.add_argument('--checkpoint', type=str, help='checkpoint path', default="/mnt/data/danhnt/Seg/work_seg/checkpoint/weights/LW_V1_ep292.pth")
parser.add_argument('--work_dir', type=str, default='../work_seg/result_infer_vid')
parser.add_argument('--img_size', type=list, default=[640,360], help='image size')
parser.add_argument('--thresh_score', type=float, default=1.9, help='threshold of score')
parser.add_argument('--show_result', type=bool, default=False)

# tracking args
parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
parser.add_argument(
    "--aspect_ratio_thresh", type=float, default=1.6,
    help="threshold for filtering out boxes of which aspect ratio are above the given value."
)
parser.add_argument('--min_box_area', type=float, default=0.1, help='filter out tiny boxes')
parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

args = parser.parse_args()


class Demo:

    def __init__(self, args):
        self.args = args
        self.vid_path = args.vid_path
        self.mode = args.mode
        self.checkpoint = args.checkpoint
        self.img_size = [640,512]

        self.model = self._load_model()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.45563722, 0.45559042, 0.45564058], [0.2213023,  0.22129012, 0.22129919]),
            # transforms.Normalize([0.5186578 , 0.52004126 ,0.51559308], [0.03202766, 0.03567145, 0.04088691]),
            
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
        # model = torch.nn.DataParallel(model)
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
            output = self.model(img)
            # predict = (output.detach().numpy()).astype(np.uint8)

        return output

    def save_video(self):
        cap = cv2.VideoCapture(self.vid_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # init Tracker
        tracker = BYTETracker(args)
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_path = self.vid_path.split('/')[-1]
        vid_name = os.path.join(self.args.work_dir, 'result')
        if not os.path.isdir(vid_name):
            os.makedirs(vid_name, exist_ok=True)
        out = cv2.VideoWriter(vid_name + '/' + out_path, fourcc, fps, (640, 360))
        # create folder log to save mask 
        # if not os.path.isdir(vid_name + '/log' + '/'+ out_path):
            # os.makedirs(vid_name + '/log' + '/'+ out_path, exist_ok=True)
        # img_save = './extr/images/'
        # mask_save = './extr/masks/'
        frameID = 0
        k = 0
        count = 0
        while True:

            ret, frame = cap.read()
            if not ret:
                break
            copy = frame.copy()  

            w, h = 640, 512
            origin_w, origin_h = frame.shape[1], frame.shape[0]
            # fRate = origin_h / h
            # m_iPos = (origin_w - fRate * w) / 2
            # m_iPos = int(m_iPos)
            # crop = frame[:, m_iPos: origin_w - m_iPos]
            # crop_x, crop_y = crop.shape[1], crop.shape[0]

            # frame = cv2.resize(crop, (640, 512), interpolation=cv2.INTER_CUBIC)
            
            # if (frameID < 1000) and (frameID %100 ==0):
            #     k = k+1
            #     cv2.imwrite("sample" + str(k) + '.png', frame)

            # resize from 1280x720 to 640x512 keep aspect ratio and padding
            dh = (512 - origin_h /(origin_h/512))/2 # padding

            frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_CUBIC)
            frame = cv2.copyMakeBorder(frame, int(dh), int(dh), 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

            
            frame_tensor = self.transform(frame).unsqueeze(0).to(device)
            # get mark and hstack with frame
            mask = self._get_mask(frame_tensor)
            output_pred = mask.clone()
            mask = mask.squeeze(0).permute(1, 2, 0).cpu()
            mask = (mask.detach().numpy() > args.thresh_score).astype(np.uint8)
            mask = mask * 255
            # convert mask to 3 channel and hstack with frame   
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            fRate = origin_w / 640
            m_iPos = 0
            # # write contour to file split by comma
            # mask = np.dstack((mask, mask, mask))
            # # resize mask to frame original size
            # mask = cv2.resize(mask, (crop_x, crop_y), interpolation=cv2.INTER_NEAREST)
            # # padding to original size
            # mask = cv2.copyMakeBorder(mask, 0, origin_h - crop_y, m_iPos, origin_w - m_iPos - crop_x, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            
            contours = list(contours)
            print("Number of contours: ", len(contours))
            list_bbox = []

            # draw contours on frame
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)

            # imshow
            cv2.putText(frame,'Num Obj: ' + str(int(len(contours))), (2,17), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 1)
            cv2.imshow('frame', frame)
            out.write(frame)
            frameID += 1
            # print("frameID: ", frameID)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # # merge all box to 1 if distance center < 20
            for i,contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                if w*h < 10:
                    continue
                if len(contours) > 1:
                    #  check if x, y of contour i is 0,0
                    if x == 0 and y == 0:
                        continue
                    for j in range(i+1, len(contours)):
                        x1, y1, w1, h1 = cv2.boundingRect(contours[j])

                        c_x = x + w//2
                        c_y = y + h//2
                        c_x1 = x1 + w1//2
                        c_y1 = y1 + h1//2

                        if math.sqrt((c_x - c_x1)**2 + (c_y - c_y1)**2) < 25:
                            w = max(x + w, x1 + w1) - min(x, x1)
                            h = max(y + h, y1 + h1) - min(y, y1)
                            x = min(x, x1)
                            y = min(y, y1)

                            contours[i] = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
                            contours[j] = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])

                            #  remove box j
                            # contours.pop(j)
                
                x = x - 2
                y = y - 2
                w = w + 5
                h = h + 5


                x = int(x * fRate) + m_iPos
                y = int(y * fRate)
                w = int(w * fRate)
                h = int(h * fRate)

                temp_mask = output_pred.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
                temp_mask = temp_mask[y:y+h, x:x+w]
                zero_mask = temp_mask > 0
                temp_mask = temp_mask*zero_mask

                # calculate confidence score of object
                total = np.sum(temp_mask)
                score = total / np.sum(zero_mask)

                list_bbox.append([x - w/2, y - h/2, x+w*1.5, y+h*1.5, 1])
                x_ = int(x * fRate) + m_iPos
                y_ = int(y * fRate)
                w_ = int(w * fRate)
                h_ = int(h * fRate)
                # convert to tensor
                cv2.rectangle(copy, (int(x_), int(y_)), (int(x_ + w_), int(y_ + h_)), (255, 0, 0), 1)
                
            if len(list_bbox) < 1:
                list_bbox.append([0,0,0,0,0])

            # tracking
            img_size = (frame.shape[1], frame.shape[0])
            tensor_bbox = np.array(list_bbox)
            tensor_bbox = torch.from_numpy(tensor_bbox).float()
            print("list_bbox : ", list_bbox)
            out_track = tracker.update(tensor_bbox, img_size, img_size)

            print("out_track : ", out_track)
            
            # draw bounding box
            for i in out_track:
                bbox = i.tlwh 
                print("bbox : ", bbox)
                x, y, w, h = bbox
                x = x + w/4
                y = y + h/4
                w = w/2
                h = h/2
                x = int(x * fRate) + m_iPos
                y = int(y * fRate)
                w = int(w * fRate)
                h = int(h * fRate)
                # print("x, y, w, h: ", x, y, w, h)
                cv2.rectangle(copy, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 1)
                cv2.putText(copy,str(w) + 'x' + str(h), (x-2,y-2), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 1)
                count  = count + 1
            


            # imwrite frame_id to frame 
            cv2.putText(copy,'Frame Id: ' + str(int(frameID)), (2,17), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)
            # out.write(copy)
            

        print("total target: ", count)
        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    demo = Demo(args)
    start = time.time()
    demo.save_video()
    end = time.time()
    print('Time: ', end - start)
    

    
