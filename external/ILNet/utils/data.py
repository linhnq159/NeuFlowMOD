import os
import random
import numpy as np
import cv2
import torch.nn as nn
from torchvision import transforms
import torch.utils.data as Data
import torch
import os.path as osp
from PIL import Image, ImageOps, ImageFilter


class CustomDataset(nn.Module):
    def __init__(self, data_path, mode='train', args=None):
        super(CustomDataset, self).__init__()
        
        self.mode = mode
        self.args = args
        self.dataset = []
        self.len_list = []
        # print(args)
        if mode == 'train':
            self.data_path = data_path + ''
            self.list_folder = os.listdir(self.data_path)
            for folder in self.list_folder:
                self.dataset.append(singleFolder(self.data_path, folder, mode, self.args))
        elif mode == 'test':
            self.data_path = data_path + ''
            self.list_folder = os.listdir(self.data_path)
            for folder in self.list_folder:
                self.dataset.append(singleFolder(self.data_path, folder, mode, self.args))

        idx = -1
        for i,data in enumerate(self.dataset):
            idx += len(data)
            self.len_list.append(idx)

        # print(self.len_list)

    def __len__(self):

        return self.len_list[-1]
    
    def __getitem__(self, idx):
        for i, data in enumerate(self.dataset):
            # print(idx, self.len_list)
            if idx <= self.len_list[i] and i == 0:
                return data.__getitem__(idx)
            elif idx <= self.len_list[i] :
                return data.__getitem__(idx - self.len_list[i-1] - 1)
        return None

class CustomDatasetOpticalFlow(nn.Module):
    def __init__(self, data_path, mode='train', args=None):
        super(CustomDatasetOpticalFlow, self).__init__()
        
        self.mode = mode
        self.args = args
        self.dataset = []
        self.len_list = []
        # print(args)
        print("data_path : ", data_path)
        if mode == 'train':
            self.data_path = data_path + ''
            self.list_folder = os.listdir(self.data_path)
            for folder in self.list_folder:
                print("folder : ", folder)
                self.dataset.append(singleFolderOpticalFlow(self.data_path, folder, mode, self.args))
        elif mode == 'test':
            self.data_path = data_path + ''
            self.list_folder = os.listdir(self.data_path)
            for folder in self.list_folder:
                self.dataset.append(singleFolderOpticalFlow(self.data_path, folder, mode, self.args))

        idx = -1
        for i,data in enumerate(self.dataset):
            idx += len(data)
            self.len_list.append(idx)

        # print(self.len_list)

    def __len__(self):

        return self.len_list[-1]
    
    def __getitem__(self, idx):
        for i, data in enumerate(self.dataset):
            # print(idx, self.len_list)
            if idx <= self.len_list[i] and i == 0:
                return data.__getitem__(idx)
            elif idx <= self.len_list[i] :
                return data.__getitem__(idx - self.len_list[i-1] - 1)
        return None

class CustomDatasetTCCLA(nn.Module):
    def __init__(self, data_path, mode='train', args=None):
        super(CustomDatasetTCCLA, self).__init__()
        
        self.mode = mode
        self.args = args
        self.dataset = []
        self.len_list = []
        # print(args)
        if mode == 'train':
            self.data_path = data_path + ''
            self.list_folder = os.listdir(self.data_path)
            for folder in self.list_folder:
                self.dataset.append(singleFolderTCCLA(self.data_path, folder, mode, self.args))
        elif mode == 'test':
            self.data_path = data_path + ''
            self.list_folder = os.listdir(self.data_path)
            for folder in self.list_folder:
                self.dataset.append(singleFolderTCCLA(self.data_path, folder, mode, self.args))

        idx = -1
        for i,data in enumerate(self.dataset):
            idx += len(data)
            self.len_list.append(idx)

        # print(self.len_list)

    def __len__(self):

        return self.len_list[-1]
    
    def __getitem__(self, idx):
        for i, data in enumerate(self.dataset):
            # print(idx, self.len_list)
            if idx <= self.len_list[i] and i == 0:
                return data.__getitem__(idx)
            elif idx <= self.len_list[i] :
                return data.__getitem__(idx - self.len_list[i-1] - 1)
        return None   

class singleFolderOpticalFlow(nn.Module):
    def __init__(self, data_path, folder_name="", mode='train',args=None):
        super(singleFolderOpticalFlow, self).__init__()
        self.base_dir = folder_name
        self.mode = mode
        self.args = args
        self.list_black_target = []

        # print(args)
        # print(data_path)
        # self.img_size = args.img_size
        # self.img_size = [640,512]
        self.img_size = [960,544]
        if mode == 'train':
            self.imgs_dir = data_path + '/' + folder_name + '/npy'
            self.label_dir = data_path + '/' + folder_name + '/masks'
        elif mode == 'test':
            self.imgs_dir = data_path +  '/' + folder_name + '/npy'
            self.label_dir = data_path +  '/' + folder_name +'/masks'
            # self.data_train = data_path +  '/' + folder_name + '/images'
            # self.data_label = data_path +  '/' + folder_name +'/masks'

        self.list_img = []
        if self.mode == 'train':
            try:
                print("data_path + '/' + folder_name + '/trainval.txt' : ", data_path + '/' + folder_name + '/trainval.txt')
                with open(data_path + '/' + folder_name + '/trainval.txt', 'r') as f:
                    for line in f:
                        self.list_img.append(line.strip())
            except:
                return
        elif self.mode == 'test':
            try:
                with open(data_path  + '/' + folder_name + '/test.txt', 'r') as f:
                    for line in f:
                        self.list_img.append(line.strip())
            except:
                return

        # sort list_img to make sure the order of images and labels are the same
        self.list_img.sort()
        # print("aaaaaaaaaaaaaaaaaaaa", self.list_img)

        # self.transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     #transforms.Normalize([0.45563722, 0.45559042, 0.45564058], [0.2213023,  0.22129012, 0.22129919]),
        #     transforms.Normalize([0.883037269115448, 0.8753958344459534], [0.29727908968925476, 0.29396122694015503])
        # ])

        # self.flow_normalize = transforms.Normalize(
        #             mean = [-0.0631835013628006, 0.006084870081394911], 
        #             std = [9.645421028137207, 10.885346412658691])
        self.flow_normalize = transforms.Normalize(
                    mean = [-0.08003402501344681, 0.05379108712077141], 
                    std = [9.475221633911133, 10.464458465576172])
        # print("self.flow_normalize : " , self.flow_normalize)

    def __len__(self):
        return len(self.list_img)
    
    def __getitem__(self, i):

        name = self.list_img[i]
        img_path = osp.join(self.imgs_dir, name + '.npy')
    
        label_path = osp.join(self.label_dir, name + '.png')

        # img = np.load(img_path)
        # mask = Image.open(label_path).convert('L')
        # img_ch0 = Image.fromarray(img[:, :, 0], mode = "F")
        # img_ch1 = Image.fromarray(img[:, :, 1], mode = "F")

        # if self.mode == 'train':
        #     img_ch0, img_ch1, mask = self._sync_transform(img_ch0, img_ch1, mask)
        # else:
        #     img_ch0, img_ch1, mask = self._val_sync_transform(img_ch0, img_ch1, mask)

        # print("img_ch0 : ", np.array(img_ch0))

        # img_ch0_tensor = transforms.ToTensor()(img_ch0)
        # img_ch1_tensor = transforms.ToTensor()(img_ch1)

        # print("img_ch0_tensor : ", img_ch0_tensor)

        # img = torch.cat([img_ch0_tensor, img_ch1_tensor], dim=0)

        # print("img : ", img)
        # print("img : ", img.shape)

        # # img = transforms.ToTensor()(img)
        # img  = self.transform(img)
        # mask = transforms.ToTensor()(mask)

        # print("img_path : ", img_path)
        # print("label_path : ", label_path)

        img = np.load(img_path).astype(np.float32)
        flow_tensor = torch.from_numpy(img.transpose(2, 0, 1)) # (2, H, W)
        # print("flow_tensor.shape : ", flow_tensor.shape)
        # print("flow_tensor before normalize : ", flow_tensor)

        flow_tensor = self.flow_normalize(flow_tensor)
        mask = Image.open(label_path).convert('L')

        # # Thay bằng đường dẫn thực tế
        # label_path = '/data/linhnq11/MOD/CPP/mod/Data/Segment/OpticalFlowInput/test/masks/frame_2.png'
        # mask = Image.open(label_path).convert('L')
        # print("flow_tensor : ", flow_tensor)

        # print("flow_tensor : ", flow_tensor[0])
        # print("flow_tensor[1] : ", flow_tensor[1])
        # print("flow_tensor : ", flow_tensor)
        # print("flow min/max:", flow_tensor.min().item(), flow_tensor.max().item())
        # print("flow mean/std:", flow_tensor.mean().item(), flow_tensor.std().item())
        # print("mask : ", mask)

        if self.mode == 'train':
            flow_tensor, mask = self._sync_transform_flow(flow_tensor, mask)
        else:
            flow_tensor, mask = self._val_sync_transform_flow(flow_tensor, mask)

        # x1 = input()

        return flow_tensor, mask

    def _sync_transform_flow(self, flow_tensor, mask):
        # flow_tensor: (2, H, W), tensor float32
        # mask: PIL Image (L)

        w, h = mask.size
        ref_area = self.img_size[0] * self.img_size[1]

        # 1. Horizontal flip
        if random.random() < 0.5:
            flow_tensor = torch.flip(flow_tensor, dims=[2])  # flip width
            flow_tensor[0] = -flow_tensor[0]  # đảo hướng u (x-direction)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # 2. Random scale (chỉ resize mask, flow dùng interpolate)
        if random.random() > 0.5:
            area = w * h
            if area > ref_area:
                ratio = random.uniform(0.7, 1.3)
            else:
                ratio = random.uniform(1.0, 1.5)
            new_w, new_h = int(w * ratio), int(h * ratio)
            mask = mask.resize((new_w, new_h), Image.NEAREST)
            w, h = new_w, new_h

            # Resize flow tensor (nearest cho flow)
            flow_tensor = torch.nn.functional.interpolate(
                flow_tensor.unsqueeze(0), size=(new_h, new_w), mode='nearest'
            ).squeeze(0)

        # 3. Resize & crop (giữ nguyên logic như _sync_transform)
        img_size = self.img_size
        if w * h > img_size[0] * img_size[1]:
            short_edge = min(img_size)
            scale = random.randint(int(short_edge / 1.5), int(short_edge * 1.2))
            if h > w:
                ow, oh = scale, int(scale * h / w)
            else:
                oh, ow = scale, int(scale * w / h)
        else:
            ow, oh = w, h

        mask = mask.resize((ow, oh), Image.NEAREST)
        flow_tensor = torch.nn.functional.interpolate(
            flow_tensor.unsqueeze(0), size=(oh, ow), mode='nearest'
        ).squeeze(0)

        # Pad
        padh = max(img_size[1] - oh, 0)
        padw = max(img_size[0] - ow, 0)
        padh_x = random.randint(0, padh)
        padw_x = random.randint(0, padw)
        flow_tensor = torch.nn.functional.pad(
            flow_tensor, (padw_x, padw - padw_x, padh_x, padh - padh_x), mode='constant', value=0
        )
        mask = ImageOps.expand(mask, border=(padw_x, padh_x, padw - padw_x, padh - padh_x), fill=0)

        # Final crop
        w, h = ow + padw, oh + padh
        x1 = random.randint(0, max(0, w - img_size[0]))
        y1 = random.randint(0, max(0, h - img_size[1]))
        flow_tensor = flow_tensor[:, y1:y1+img_size[1], x1:x1+img_size[0]]
        mask = mask.crop((x1, y1, x1 + img_size[0], y1 + img_size[1]))

        # 4. Final crop mask → convert tensor
        mask_tensor = transforms.ToTensor()(mask)

        return flow_tensor, mask_tensor

    def _val_sync_transform_flow(self, flow_tensor, mask):
        # Tương tự nhưng không random (giữ nguyên logic _val_sync_transform)
        w, h = mask.size
        img_size = self.img_size

        if w * h > img_size[0] * img_size[1]:
            if w / img_size[0] > h / img_size[1]:
                ow = img_size[0]
                oh = int(h * img_size[0] / w)
            else:
                oh = img_size[1]
                ow = int(w * img_size[1] / h)
        else:
            ow, oh = w, h

        mask = mask.resize((ow, oh), Image.NEAREST)
        flow_tensor = torch.nn.functional.interpolate(
            flow_tensor.unsqueeze(0), size=(oh, ow), mode='nearest'
        ).squeeze(0)

        padh = max(img_size[1] - oh, 0)
        padw = max(img_size[0] - ow, 0)
        rand_x = random.randint(0, padw)
        rand_y = random.randint(0, padh)
        flow_tensor = torch.nn.functional.pad(
            flow_tensor, (rand_x, padw - rand_x, rand_y, padh - rand_y), mode='constant', value=0
        )
        mask = ImageOps.expand(mask, border=(rand_x, rand_y, padw - rand_x, padh - rand_y), fill=0)

        mask_tensor = transforms.ToTensor()(mask)
        return flow_tensor, mask_tensor

    def _sync_transform(self, img_ch0, img_ch1, mask):
        w, h = mask.size
        ref_area = self.img_size[0] * self.img_size[1]

        # Horizontal flip
        if random.random() < 0.5:
            img_ch0 = img_ch0.transpose(Image.FLIP_LEFT_RIGHT)
            img_ch1 = img_ch1.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # Random scale (only if larger than reference)
        if random.random() > 0.5:
            area = w * h
            if area > ref_area:
                ratio = random.uniform(0.7, 1.3)
            else:
                ratio = random.uniform(1.0, 1.5)
            new_w, new_h = int(w * ratio), int(h * ratio)
            img_ch0 = img_ch0.resize((new_w, new_h), Image.BILINEAR)
            img_ch1 = img_ch1.resize((new_w, new_h), Image.BILINEAR)
            mask = mask.resize((new_w, new_h), Image.NEAREST)
            w, h = new_w, new_h

        # Resize to fit near self.img_size (keep aspect ratio)
        img_size = self.img_size
        if w * h > img_size[0] * img_size[1]:
            short_edge = min(img_size)
            scale = random.randint(int(short_edge / 1.5), int(short_edge * 1.2))
            if h > w:
                ow, oh = scale, int(scale * h / w)
            else:
                oh, ow = scale, int(scale * w / h)
        else:
            ow, oh = w, h

        img_ch0 = img_ch0.resize((ow, oh), Image.BILINEAR)
        img_ch1 = img_ch1.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        # Pad to at least img_size
        padh = max(img_size[1] - oh, 0)
        padw = max(img_size[0] - ow, 0)
        padh_x = random.randint(0, padh)
        padw_x = random.randint(0, padw)
        img_ch0 = ImageOps.expand(img_ch0, border=(padw_x, padh_x, padw - padw_x, padh - padh_x), fill=0)
        img_ch1 = ImageOps.expand(img_ch1, border=(padw_x, padh_x, padw - padw_x, padh - padh_x), fill=0)
        mask = ImageOps.expand(mask, border=(padw_x, padh_x, padw - padw_x, padh - padh_x), fill=0)

        # Final crop to img_size
        w, h = img_ch0.size
        x1 = random.randint(0, max(0, w - img_size[0]))
        y1 = random.randint(0, max(0, h - img_size[1]))
        img_ch0 = img_ch0.crop((x1, y1, x1 + img_size[0], y1 + img_size[1]))
        img_ch1 = img_ch1.crop((x1, y1, x1 + img_size[0], y1 + img_size[1]))
        mask = mask.crop((x1, y1, x1 + img_size[0], y1 + img_size[1]))

        # # Blur
        # if random.random() < 0.5:
        #     radius = random.random()
        #     img_ch0 = img_ch0.filter(ImageFilter.GaussianBlur(radius=radius))
        #     img_ch1 = img_ch1.filter(ImageFilter.GaussianBlur(radius=radius))

        # Apply validation transforms (no random augment)
        img_ch0, img_ch1, mask = self._val_sync_transform(img_ch0, img_ch1, mask)

        return img_ch0, img_ch1, mask
        
    def _val_sync_transform(self, img_ch0, img_ch1, mask):
        w, h = mask.size
        img_size = self.img_size

        # Resize to fit within img_size (preserve aspect ratio)
        if w * h > img_size[0] * img_size[1]:
            if w / img_size[0] > h / img_size[1]:
                ow = img_size[0]
                oh = int(h * img_size[0] / w)
            else:
                oh = img_size[1]
                ow = int(w * img_size[1] / h)
        else:
            ow, oh = w, h

        img_ch0 = img_ch0.resize((ow, oh), Image.BILINEAR)
        img_ch1 = img_ch1.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        # Pad to img_size
        padh = max(img_size[1] - oh, 0)
        padw = max(img_size[0] - ow, 0)
        rand_x = random.randint(0, padw)
        rand_y = random.randint(0, padh)
        img_ch0 = ImageOps.expand(img_ch0, border=(rand_x, rand_y, padw - rand_x, padh - rand_y), fill=0)
        img_ch1 = ImageOps.expand(img_ch1, border=(rand_x, rand_y, padw - rand_x, padh - rand_y), fill=0)
        mask = ImageOps.expand(mask, border=(rand_x, rand_y, padw - rand_x, padh - rand_y), fill=0)

        return img_ch0, img_ch1, mask
 

class singleFolderTCCLA(nn.Module):
    def __init__(self, data_path, folder_name="", mode='train',args=None):
        super(singleFolderTCCLA, self).__init__()
        self.base_dir = folder_name
        self.mode = mode
        self.args = args
        self.list_black_target = []

        # print(args)
        print(data_path)
        # self.img_size = args.img_size
        self.img_size = [640,512]
        if mode == 'train':
            self.imgs_dir = data_path + '/' + folder_name + '/images'
            self.label_dir = data_path + '/' + folder_name + '/masks'
        elif mode == 'test':
            self.imgs_dir = data_path +  '/' + folder_name + '/images'
            self.label_dir = data_path +  '/' + folder_name +'/masks'
            # self.data_train = data_path +  '/' + folder_name + '/images'
            # self.data_label = data_path +  '/' + folder_name +'/masks'

        self.list_img = []
        if self.mode == 'train':
            try:
                with open(data_path + '/' + folder_name + '/trainval.txt', 'r') as f:
                    for line in f:
                        self.list_img.append(line.strip())
            except:
                return
        elif self.mode == 'test':
            try:
                with open(data_path  + '/' + folder_name + '/test.txt', 'r') as f:
                    for line in f:
                        self.list_img.append(line.strip())
            except:
                return

        # sort list_img to make sure the order of images and labels are the same
        self.list_img.sort()
        # print("aaaaaaaaaaaaaaaaaaaa", self.list_img)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize([0.45563722, 0.45559042, 0.45564058], [0.2213023,  0.22129012, 0.22129919]),
            transforms.Normalize([0.883037269115448, 0.8753958344459534, 0.8630934357643127], [0.29727908968925476, 0.29396122694015503, 0.2957575023174286])
        ])

    def __len__(self):
        return len(self.list_img)
    
    def __getitem__(self, i):

        try:
            name = self.list_img[i]
            img_path = osp.join(self.imgs_dir, name + '.png')

            if "SIRST" in self.base_dir:
                label_path = osp.join(self.label_dir, name + '_pixels0.png')
            else:
                label_path = osp.join(self.label_dir, name + '.png')

            img = Image.open(img_path).convert('RGB')
            mask = Image.open(label_path).convert('RGB')
        except:
            new_idx = random.randint(0, len(self.list_img) - 1)
            name = self.list_img[new_idx]
            img_path = osp.join(self.imgs_dir, name + '.jpg')

            if "open" in self.base_dir:
                label_path = osp.join(self.label_dir, name + '_pixels0.png')
            else:
                label_path = osp.join(self.label_dir, name + '.png')

            img = Image.open(img_path).convert('RGB')
            mask = Image.open(label_path).convert('RGB')
        
        if self.mode == 'train':

            img, mask = self._sync_transform(img, mask)

        else:
            img, mask = self._val_sync_transform(img, mask)
        
        img  = self.transform(img)
        # img = transforms.ToTensor()(img)
        mask = transforms.ToTensor()(mask)
        # mask = self.transform(mask)
 
        return img, mask
        
    def _sync_transform(self, img, mask):
        w,h = img.size
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() > 0.5:
            #  random ratio from (0.7 to 1.3)
            if w*h > self.img_size[0]* self.img_size[1]:
                ratio = random.uniform(0.7, 1.3)
            else:
                ratio = random.uniform(1.0, 1.5)

            # resize the image
            img = img.resize((int(w*ratio), int(h*ratio)), Image.BILINEAR)
            mask = mask.resize((int(w*ratio), int(h*ratio)), Image.NEAREST)

##########################################################################################
        w,h = img.size
        img_size = self.img_size

        if w*h > self.img_size[0]* self.img_size[1]:
            short_edg = min(img_size)
            scale = random.randint(int(short_edg / 1.5), int(short_edg * 1.2))
            w, h = img.size
            if h > w:
                ow = scale
                oh = int(scale * h / w)
            else:
                oh = scale
                ow = int(scale * w / h)


            img = img.resize((ow, oh), Image.BILINEAR)
            mask = mask.resize((ow, oh), Image.NEAREST)
            # pad crop
            if ow < img_size[0] or oh < img_size[1]:
                padh = max(img_size[1] - oh, 0)
                padw = max(img_size[0] - ow, 0)
                padh_x = random.randint(0, padh)
                padw_x = random.randint(0, padw)
                img = ImageOps.expand(img, border=(padw_x, padh_x, padw-padw_x, padh-padh_x), fill=0)
                mask = ImageOps.expand(mask, border=(padw_x, padh_x, padw-padw_x, padh-padh_x), fill=0)
            w, h = img.size
            x1 = random.randint(0, w - img_size[0])
            y1 = random.randint(0, h - img_size[1])
            img = img.crop((x1, y1, x1 + img_size[0], y1 + img_size[1]))
            mask = mask.crop((x1, y1, x1 + img_size[0], y1 + img_size[1]))
        else:
            padh = max(img_size[1] - h, 0)
            padw = max(img_size[0] - w, 0)
            padh_x = random.randint(0, padh)
            padw_x = random.randint(0, padw)
            img = ImageOps.expand(img, border=(padw_x, padh_x, padw-padw_x, padh-padh_x), fill=0)
            mask = ImageOps.expand(mask, border=(padw_x, padh_x, padw-padw_x, padh-padh_x), fill=0)
            w, h = img.size
            x1 = random.randint(0, w - img_size[0])
            y1 = random.randint(0, h - img_size[1])
            img = img.crop((x1, y1, x1 + img_size[0], y1 + img_size[1]))
            mask = mask.crop((x1, y1, x1 + img_size[0], y1 + img_size[1]))

        if random.random() < 0.5:
            radius = random.random()
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))

        img, mask = self._val_sync_transform(img, mask)

        return img, mask

    def _val_sync_transform(self, img, mask):
        w,h = img.size
        img_size = self.img_size

        if w*h > self.img_size[0]* self.img_size[1]:
            w, h = img.size
            if (w/self.img_size[0]) > (h/self.img_size[1]):
                ow = self.img_size[0]
                oh = int(h * self.img_size[0] / w)

            else:
                oh = self.img_size[1]
                ow = int(w * self.img_size[1] / h)
            img = img.resize((ow, oh), Image.BILINEAR)
            mask = mask.resize((ow, oh), Image.NEAREST)
            # padding
            padh = max(self.img_size[1] - oh, 0)
            padw = max(self.img_size[0] - ow, 0)
            rand_x = random.randint(0, padw)
            rand_y = random.randint(0, padh)
            img = ImageOps.expand(img, border=(rand_x, rand_y, padw - rand_x, padh - rand_y), fill=0)
            mask = ImageOps.expand(mask, border=(rand_x, rand_y, padw - rand_x, padh - rand_y), fill=0)
        else:
            padh = max(self.img_size[1] - h, 0)
            padw = max(self.img_size[0] - w, 0)
            padh_x = random.randint(0, padh)
            padw_x = random.randint(0, padw)
            img = ImageOps.expand(img, border=(padw_x, padh_x, padw-padw_x, padh-padh_x), fill=0)
            mask = ImageOps.expand(mask, border=(padw_x, padh_x, padw-padw_x, padh-padh_x), fill=0)
        

        return img, mask

class singleFolder(nn.Module):
    def __init__(self, data_path, folder_name="", mode='train',args=None):
        super(singleFolder, self).__init__()
        self.base_dir = folder_name
        self.mode = mode
        self.args = args
        self.list_black_target = []

        # print(args)
        print(data_path)
        # self.img_size = args.img_size
        # self.img_size = [640,512]
        self.img_size = [960,544]
        if mode == 'train':
            self.imgs_dir = data_path + '/' + folder_name + '/images'
            self.label_dir = data_path + '/' + folder_name + '/masks1'
        elif mode == 'test':
            self.imgs_dir = data_path +  '/' + folder_name + '/images'
            self.label_dir = data_path +  '/' + folder_name +'/masks1'
            # self.data_train = data_path +  '/' + folder_name + '/images'
            # self.data_label = data_path +  '/' + folder_name +'/masks'

        self.list_img = []
        if self.mode == 'train':
            try:
                with open(data_path + '/' + folder_name + '/trainval.txt', 'r') as f:
                    for line in f:
                        self.list_img.append(line.strip())
            except:
                return
        elif self.mode == 'test':
            try:
                with open(data_path  + '/' + folder_name + '/test.txt', 'r') as f:
                    for line in f:
                        self.list_img.append(line.strip())
            except:
                return

        # sort list_img to make sure the order of images and labels are the same
        self.list_img.sort()
        # print("aaaaaaaaaaaaaaaaaaaa", self.list_img)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize([0.45563722, 0.45559042, 0.45564058], [0.2213023,  0.22129012, 0.22129919]),
            transforms.Normalize([0.883037269115448, 0.8753958344459534, 0.8630934357643127], [0.29727908968925476, 0.29396122694015503, 0.2957575023174286])
        ])

    def __len__(self):
        return len(self.list_img)
    
    def __getitem__(self, i):

        try:
            name = self.list_img[i]
            img_path = osp.join(self.imgs_dir, name + '.png')

            if "SIRST" in self.base_dir:
                label_path = osp.join(self.label_dir, name + '_pixels0.png')
            else:
                label_path = osp.join(self.label_dir, name + '.png')

            img = Image.open(img_path).convert('RGB')
            mask = Image.open(label_path).convert('L')
        except:
            new_idx = random.randint(0, len(self.list_img) - 1)
            name = self.list_img[new_idx]
            img_path = osp.join(self.imgs_dir, name + '.jpg')

            if "open" in self.base_dir:
                label_path = osp.join(self.label_dir, name + '_pixels0.png')
            else:
                label_path = osp.join(self.label_dir, name + '.png')

            img = Image.open(img_path).convert('RGB')
            mask = Image.open(label_path).convert('L')
        
        if self.mode == 'train':

            img, mask = self._sync_transform(img, mask)

        else:
            img, mask = self._val_sync_transform(img, mask)
        
        img  = self.transform(img)
        mask = transforms.ToTensor()(mask)

        return img, mask
        
    def _sync_transform(self, img, mask):
        w,h = img.size
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() > 0.5:
            #  random ratio from (0.7 to 1.3)
            if w*h > self.img_size[0]* self.img_size[1]:
                ratio = random.uniform(0.7, 1.3)
            else:
                ratio = random.uniform(1.0, 1.5)

            # resize the image
            img = img.resize((int(w*ratio), int(h*ratio)), Image.BILINEAR)
            mask = mask.resize((int(w*ratio), int(h*ratio)), Image.NEAREST)

##########################################################################################
        w,h = img.size
        img_size = self.img_size

        if w*h > self.img_size[0]* self.img_size[1]:
            short_edg = min(img_size)
            scale = random.randint(int(short_edg / 1.5), int(short_edg * 1.2))
            w, h = img.size
            if h > w:
                ow = scale
                oh = int(scale * h / w)
            else:
                oh = scale
                ow = int(scale * w / h)


            img = img.resize((ow, oh), Image.BILINEAR)
            mask = mask.resize((ow, oh), Image.NEAREST)
            # pad crop
            if ow < img_size[0] or oh < img_size[1]:
                padh = max(img_size[1] - oh, 0)
                padw = max(img_size[0] - ow, 0)
                padh_x = random.randint(0, padh)
                padw_x = random.randint(0, padw)
                img = ImageOps.expand(img, border=(padw_x, padh_x, padw-padw_x, padh-padh_x), fill=0)
                mask = ImageOps.expand(mask, border=(padw_x, padh_x, padw-padw_x, padh-padh_x), fill=0)
            w, h = img.size
            x1 = random.randint(0, w - img_size[0])
            y1 = random.randint(0, h - img_size[1])
            img = img.crop((x1, y1, x1 + img_size[0], y1 + img_size[1]))
            mask = mask.crop((x1, y1, x1 + img_size[0], y1 + img_size[1]))
        else:
            padh = max(img_size[1] - h, 0)
            padw = max(img_size[0] - w, 0)
            padh_x = random.randint(0, padh)
            padw_x = random.randint(0, padw)
            img = ImageOps.expand(img, border=(padw_x, padh_x, padw-padw_x, padh-padh_x), fill=0)
            mask = ImageOps.expand(mask, border=(padw_x, padh_x, padw-padw_x, padh-padh_x), fill=0)
            w, h = img.size
            x1 = random.randint(0, w - img_size[0])
            y1 = random.randint(0, h - img_size[1])
            img = img.crop((x1, y1, x1 + img_size[0], y1 + img_size[1]))
            mask = mask.crop((x1, y1, x1 + img_size[0], y1 + img_size[1]))

        if random.random() < 0.5:
            radius = random.random()
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))

        img, mask = self._val_sync_transform(img, mask)

        return img, mask

    def _val_sync_transform(self, img, mask):
        w,h = img.size
        img_size = self.img_size

        if w*h > self.img_size[0]* self.img_size[1]:
            w, h = img.size
            if (w/self.img_size[0]) > (h/self.img_size[1]):
                ow = self.img_size[0]
                oh = int(h * self.img_size[0] / w)

            else:
                oh = self.img_size[1]
                ow = int(w * self.img_size[1] / h)
            img = img.resize((ow, oh), Image.BILINEAR)
            mask = mask.resize((ow, oh), Image.NEAREST)
            # padding
            padh = max(self.img_size[1] - oh, 0)
            padw = max(self.img_size[0] - ow, 0)
            rand_x = random.randint(0, padw)
            rand_y = random.randint(0, padh)
            img = ImageOps.expand(img, border=(rand_x, rand_y, padw - rand_x, padh - rand_y), fill=0)
            mask = ImageOps.expand(mask, border=(rand_x, rand_y, padw - rand_x, padh - rand_y), fill=0)
        else:
            padh = max(self.img_size[1] - h, 0)
            padw = max(self.img_size[0] - w, 0)
            padh_x = random.randint(0, padh)
            padw_x = random.randint(0, padw)
            img = ImageOps.expand(img, border=(padw_x, padh_x, padw-padw_x, padh-padh_x), fill=0)
            mask = ImageOps.expand(mask, border=(padw_x, padh_x, padw-padw_x, padh-padh_x), fill=0)
        

        return img, mask

# test:

if __name__ == '__main__':
    import cv2

    data_path = '/data/linhnq11/Data'
    data_save_dir = "/data/linhnq11/segmentation/ILNet-master/work_seg/run/output_img"
    os.makedirs(data_save_dir, exist_ok= True)
    dataset = CustomDatasetTCCLA(data_path, mode='train')

    for i in range(len(dataset)):
        img, mask = dataset[i]
        print(img.shape)
        img = img.permute(1,2,0).cpu().numpy()
        mask = mask.permute(1,2,0).cpu().numpy()
        # img = np.array(img)
        # mask = np.array(mask)

        img = img * 255
        mask = mask * 255        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
        # cv2.imshow("img",img)
        # cv2.imshow("mask",mask)



        print("mask : ", mask.shape)
        print("img : ", img.shape)

        # mask = Image.fromarray(img.astype('uint8')).convert('RGB')
        # copy = Image.fromarray(mask.astype('uint8')).convert('RGB')
        # mer = np.hstack((np.array(copy), np.array(mask)))
        
        print(f"{data_save_dir}/mask_{i}.jpg")
        print("mask : ", mask.shape)
        print("img : ", img.shape)
        file_mask = os.path.join(data_save_dir,  str(i) + "_mask.png")
        file_img = os.path.join(data_save_dir, str(i) + "_img.png")
        cv2.imwrite(file_mask, mask)
        cv2.imwrite(file_img, img)
        # cv2.waitKey(0)
        



        

        


        
