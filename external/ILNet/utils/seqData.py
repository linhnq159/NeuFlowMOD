import os
import random
import numpy as np
import cv2
import torch.nn as nn
from torchvision import transforms
import torch.utils.data as Data
import torch
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
            self.data_path = data_path + '/train'
            self.list_folder = os.listdir(self.data_path)
            for folder in self.list_folder:
                self.dataset.append(singleFolder(self.data_path, folder, mode, self.args))
        elif mode == 'test':
            self.data_path = data_path + '/test'
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
    
class singleFolder(nn.Module):
    def __init__(self, data_path, folder_name="", mode='train',args=None):
        super(singleFolder, self).__init__()

        self.mode = mode
        self.args = args
        # print(args)
        self.img_size = args.img_size
        if mode == 'train':
            self.data_train = data_path + '/' + folder_name + '/images'
            self.data_label = data_path + '/' + folder_name + '/masks'
        elif mode == 'test':
            self.data_train = data_path +  '/' + folder_name + '/images'
            self.data_label = data_path +  '/' + folder_name +'/masks'
            # self.data_train = data_path +  '/' + folder_name + '/images'
            # self.data_label = data_path +  '/' + folder_name +'/masks'

        self.list_img = []
        if self.mode == 'train':
            with open(data_path + '/' + folder_name + '/images.txt', 'r') as f:
                for line in f:
                    self.list_img.append(line.strip())
        elif self.mode == 'test':
            with open(data_path  + '/' + folder_name + '/images.txt', 'r') as f:
                for line in f:
                    self.list_img.append(line.strip())

        # sort list_img to make sure the order of images and labels are the same
        self.list_img.sort()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.5],[0.5])
            
        ])

    def __len__(self):
        return len(self.list_img)
    
    def __getitem__(self, idx):

        img = Image.open(self.data_train + '/' + self.list_img[idx])

        if self.mode == 'train':

            mask = Image.open(self.data_label + '/' + self.list_img[idx].replace('.jpg', '.png'))
        else:

            mask = Image.open(self.data_label + '/' + self.list_img[idx].replace('jpg', 'png'))


        
        if self.mode == 'train':

            img, mask = self.__trainSynch(img, mask)

        else:
            img, mask = self.__valSynch(img, mask)
        
        
        
        img  = self.transform(img)
        mask = transforms.ToTensor()(mask)
 
        return img, mask
        
       
    def __trainSynch(self, img, mask):
        # random to augment
        if random.random() < 0.5:
            
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            
            
        
        img_size = img.size # [256, 330]
        ratio = random.uniform(0.9, 1.2)
        w, h = img.size
        if w > h:
            new_w = int(ratio * w)
            new_h = int(new_w * h / w)
        else:
            new_h = int(ratio * h)
            new_w = int(new_h * w / h)

        
            img = img.resize((new_w, new_h))
            mask = mask.resize((new_w, new_h))

        



        if ratio < 1:
            # padd to img_size
            padh = img_size[1] - new_h if new_h < img_size[1] else 0
            padw = img_size[0] - new_w if new_w < img_size[0] else 0

            
            img = ImageOps.expand(img, (0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, (0, 0, padw, padh), fill=0)
            

        else:
            # crop to img_size
            x = random.randint(0, new_w - w)
            y = random.randint(0, new_h - h)

            
            img = img.crop((x, y, x + img_size[0], y + img_size[1]))
            mask = mask.crop((x, y, x + img_size[0], y + img_size[1]))

            

        if random.random() < 0.5:
            radius = random.random()
            
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        
        img, mask = self.__valSynch(img, mask)
        return img, mask
    
    def __valSynch(self, img, mask):
        # resize keeping ratio and padding
        img_size = self.img_size
        w, h = img.size
        ratio = min(img_size[0] / w, img_size[1] / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        # img1 = img1.resize((new_w, new_h))
        # # img2 = img2.resize((new_w, new_h))
        # img2 = cv2.resize(img2, (new_w, new_h))
        # mask = mask.resize((new_w, new_h))

        
        img = img.resize((new_w, new_h))
        mask = mask.resize((new_w, new_h))


        padw = img_size[0] - new_w
        padh = img_size[1] - new_h

        x,y = random.randint(0, padw), random.randint(0, padh)

        # expand random x,y
        # img1 = ImageOps.expand(img1, (x, y, padw - x, padh - y), fill=0)
        # # img2 = ImageOps.expand(img2, (x, y, padw - x, padh - y), fill=0)
        # img2 = cv2.copyMakeBorder(img2, y, padh - y, x, padw - x, cv2.BORDER_CONSTANT, value=0)
        # mask = ImageOps.expand(mask, (x, y, padw - x, padh - y), fill=0)

        
        img = ImageOps.expand(img, (x, y, padw - x, padh - y), fill=0)
        mask = ImageOps.expand(mask, (x, y, padw - x, padh - y), fill=0)
       


        return img, mask

        
