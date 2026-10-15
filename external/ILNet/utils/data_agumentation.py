import os.path

import torch
import torch.utils.data as Data
import torchvision.transforms as transforms

from PIL import Image, ImageOps, ImageFilter
import os.path as osp
import sys
from skimage import measure
from skimage.measure import regionprops
import random
import bisect
import cv2
import numpy as np

class Get_Dataset(Data.Dataset):
    def __init__(self, args, mode='train', scale = 1.5):
        
        self.scale = scale
        self._irstd1k = Handel_Dataset(args, mode, scale, base_dir = '../work_seg/Datasets/data_custom/IRSTD-1k')
        self._add_data = Handel_Dataset(args, mode, scale, base_dir = '../work_seg/Datasets/data_custom/add_data')
        self.mavic1 = Handel_Dataset(args,mode,self.scale, base_dir = '../work_seg/Datasets/data_custom/Mavic1')
        self.mavic2 = Handel_Dataset(args,mode,self.scale, base_dir = '../work_seg/Datasets/data_custom/Mavic2')
        self.sirst_v2 = Handel_Dataset(args,mode,self.scale, base_dir = '../work_seg/Datasets/data_custom/sirst_v2')
        self.custom = Handel_Dataset(args,mode,self.scale, base_dir = '../work_seg/Datasets/data_custom/Custom')
        self.light = Handel_Dataset(args,mode,self.scale, base_dir = '../work_seg/Datasets/data_custom/Light-IRST')
        self.sirst = Handel_Dataset(args,mode,self.scale, base_dir = '../work_seg/Datasets/data_custom/SIRST')
        self.sisd = Handel_Dataset(args,mode,self.scale, base_dir = '../work_seg/Datasets/data_custom/SISD')
        self.tin = Handel_Dataset(args,mode,self.scale, base_dir = '../work_seg/Datasets/data_custom/TIN')
        # self.new_data = Handel_Dataset(args,mode,self.scale,base_dir='../work_seg/Datasets/data_custom/data_new')

        self.names = self._irstd1k.names + self._add_data.names + self.mavic1.names + self.mavic2.names + self.sirst_v2.names + self.custom.names + self.light.names + self.sirst.names + self.sisd.names + self.tin.names 
        self.mode = mode

    def __getitem__(self, i):
        if i < self._irstd1k.__len__():
            return self._irstd1k.__getitem__(i)
        elif i  < (self._irstd1k.__len__() + self._add_data.__len__()) and i >= self._irstd1k.__len__():
            return self._add_data.__getitem__(i - self._irstd1k.__len__())
        elif i  < (self._irstd1k.__len__() + self._add_data.__len__() + self.mavic1.__len__()) and i >= (self._irstd1k.__len__() +self._add_data.__len__()):
            return self.mavic1.__getitem__(i- self._irstd1k.__len__() - self._add_data.__len__() )
        elif i  < (self._irstd1k.__len__() + self._add_data.__len__() + self.mavic1.__len__() + self.mavic2.__len__()) and i >= (self._irstd1k.__len__() +self._add_data.__len__() + self.mavic1.__len__()):
            return self.mavic2.__getitem__(i- self._irstd1k.__len__() - self._add_data.__len__() - self.mavic1.__len__())
        elif i  < (self._irstd1k.__len__() + self._add_data.__len__() + self.mavic1.__len__() + self.mavic2.__len__() + self.sirst_v2.__len__()) and i >= (self._irstd1k.__len__() +self._add_data.__len__() + self.mavic1.__len__() + self.mavic2.__len__()):
            return self.sirst_v2.__getitem__(i- self._irstd1k.__len__() - self._add_data.__len__() - self.mavic1.__len__() - self.mavic2.__len__())
        elif i  < (self._irstd1k.__len__() + self._add_data.__len__() + self.mavic1.__len__() + self.mavic2.__len__() + self.sirst_v2.__len__() + self.custom.__len__()) and i >= (self._irstd1k.__len__() +self._add_data.__len__() + self.mavic1.__len__() + self.mavic2.__len__() + self.sirst_v2.__len__()):
            return self.custom.__getitem__(i- self._irstd1k.__len__() - self._add_data.__len__() - self.mavic1.__len__() - self.mavic2.__len__() - self.sirst_v2.__len__())
        elif i  < (self._irstd1k.__len__() + self._add_data.__len__() + self.mavic1.__len__() + self.mavic2.__len__() + self.sirst_v2.__len__() + self.custom.__len__() + self.light.__len__()) and i >= (self._irstd1k.__len__() +self._add_data.__len__() + self.mavic1.__len__() + self.mavic2.__len__() + self.sirst_v2.__len__() + self.custom.__len__()):
            return self.light.__getitem__(i- self._irstd1k.__len__() - self._add_data.__len__() - self.mavic1.__len__() - self.mavic2.__len__() - self.sirst_v2.__len__() - self.custom.__len__())
        elif i  < (self._irstd1k.__len__() + self._add_data.__len__() + self.mavic1.__len__() + self.mavic2.__len__() + self.sirst_v2.__len__() + self.custom.__len__() + self.light.__len__() + self.sirst.__len__()) and i >= (self._irstd1k.__len__() +self._add_data.__len__() + self.mavic1.__len__() + self.mavic2.__len__() + self.sirst_v2.__len__() + self.custom.__len__() + self.light.__len__()):
            return self.sirst.__getitem__(i- self._irstd1k.__len__() - self._add_data.__len__() - self.mavic1.__len__() - self.mavic2.__len__() - self.sirst_v2.__len__() - self.custom.__len__() - self.light.__len__())
        elif i  < (self._irstd1k.__len__() + self._add_data.__len__() + self.mavic1.__len__() + self.mavic2.__len__() + self.sirst_v2.__len__() + self.custom.__len__() + self.light.__len__() + self.sirst.__len__() + self.sisd.__len__()) and i >= (self._irstd1k.__len__() +self._add_data.__len__() + self.mavic1.__len__() + self.mavic2.__len__() + self.sirst_v2.__len__() + self.custom.__len__() + self.light.__len__() + self.sirst.__len__()):
            return self.sisd.__getitem__(i- self._irstd1k.__len__() - self._add_data.__len__() - self.mavic1.__len__() - self.mavic2.__len__() - self.sirst_v2.__len__() - self.custom.__len__() - self.light.__len__() - self.sirst.__len__())
        elif i  < (self._irstd1k.__len__() + self._add_data.__len__() + self.mavic1.__len__() + self.mavic2.__len__() + self.sirst_v2.__len__() + self.custom.__len__() + self.light.__len__() + self.sirst.__len__() + self.sisd.__len__() + self.tin.__len__()) and i >= (self._irstd1k.__len__() +self._add_data.__len__() + self.mavic1.__len__() + self.mavic2.__len__() + self.sirst_v2.__len__() + self.custom.__len__() + self.light.__len__() + self.sirst.__len__() + self.sisd.__len__()):
            return self.tin.__getitem__(i- self._irstd1k.__len__() - self._add_data.__len__() - self.mavic1.__len__() - self.mavic2.__len__() - self.sirst_v2.__len__() - self.custom.__len__() - self.light.__len__() - self.sirst.__len__() - self.sisd.__len__())
        # elif i  < (self._irstd1k.__len__() + self._add_data.__len__() + self.mavic1.__len__() + self.mavic2.__len__() + self.sirst_v2.__len__() + self.custom.__len__() + self.light.__len__() + self.sirst.__len__() + self.sisd.__len__() + self.tin.__len__()) and i >= (self._irstd1k.__len__() +self._add_data.__len__() + self.mavic1.__len__() + self.mavic2.__len__() + self.sirst_v2.__len__() + self.custom.__len__() + self.light.__len__() + self.sirst.__len__() + self.sisd.__len__() + self.tin.__len__()):
        #     return self.new_data.__getitem__(i- self._irstd1k.__len__() - self._add_data.__len__() - self.mavic1.__len__() - self.mavic2.__len__() - self.sirst_v2.__len__() - self.custom.__len__() - self.light.__len__() - self.sirst.__len__() - self.sisd.__len__() - self.tin.__len__())
    def __len__(self):
        return len(self.names)

class Handel_Dataset(Data.Dataset):

    def __init__(self, args, mode='train', scale = 1.5,  base_dir = '../work_seg/Datasets/data_custom/Mavic1'):

 
        self.base_dir = base_dir
        if 'Light-IRST' in self.base_dir:
            self.a = 12
            self.u = 50
        else:
            self.a = 60
            self.u = 120
        self.total_add_noise = 0
        self.mode = mode
        if mode == 'train':
            txtfile = 'trainval.txt'
        elif mode == 'val':
            # txtfile = 'test.txt'
            txtfile = 'test.txt'
        # elif mode == 'test':
        #     txtfile = 'val.txt'
        self.scale = scale
        self.list_dir = osp.join(base_dir, txtfile)
        self.imgs_dir = osp.join(base_dir, 'images')
        self.label_dir = osp.join(base_dir, 'masks')
        # self.noise_dir = osp.join(base_dir,'Noise')

        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]

        self.mode = mode
        self.img_size = args.img_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        #   transforms.Normalize([0.46294058, 0.46289157, 0.4629441 ], [0.21915354, 0.21914229, 0.21915013]),
        transforms.Normalize([0.45563722, 0.45559042, 0.45564058], [0.2213023,  0.22129012, 0.22129919]),
        ])

    def __getitem__(self, i):
        try:
            name = self.names[i]
            img_path = osp.join(self.imgs_dir, name + '.png')

            if "SIRST" in self.base_dir:
                label_path = osp.join(self.label_dir, name + '_pixels0.png')
            else:
                label_path = osp.join(self.label_dir, name + '.png')

            img = Image.open(img_path).convert('RGB')
            mask = Image.open(label_path).convert('L')
        except:
            new_idx = random.randint(0, len(self.names) - 1)
            name = self.names[new_idx]
            img_path = osp.join(self.imgs_dir, name + '.png')

            if "open" in self.base_dir:
                label_path = osp.join(self.label_dir, name + '_pixels0.png')
            else:
                label_path = osp.join(self.label_dir, name + '.png')

            img = Image.open(img_path).convert('RGB')
            mask = Image.open(label_path).convert('L')

        # import numpy as np
        # # enhance feature
        # img = np.array(img)
        # # cv2.imwrite("before.png",img)
        # b,g,r = cv2.split(img)
        # b_blur = cv2.medianBlur(b,3)
        # g_edge = cv2.Canny(g,self.a, self.u)
        # g_edge = cv2.bitwise_or(g,g_edge)
        # # cv2.imwrite("r.png",g_edge)
        # img = cv2.merge((b_blur, g_edge, r))
        # # cv2.imwrite("after.png",img)
        # # cv2.imwrite("mask.png",np.array(mask))
        # img = Image.fromarray(img)

        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            raise ValueError("Unkown self.mode")
        
        # get random number from 0 to 1
        random_num = random.random()
       
        # if random_num < 0.25 and self.mode == 'train':
        #     self.total_add_noise += 1
        #     # print("add noise", self.total_add_noise)
        #     w,h = img.size
        #     noise_path = osp.join(self.noise_dir, name + '.png')
        #     noise = Image.open(noise_path)
        #     noise = noise.resize((w,h), Image.NEAREST)
        #     # add noise to image
        #     img = Image.blend(img, noise, random_num)

        #     # # save image
        #     # img.save('add_noise.png')
        #     # import sys
        #     # sys.exit(0)


        # if random_num > 0.25 and random_num < 0.5 and self.mode == 'train':
        #     # agumentation target from mask
        #     # find all object in mask
        #     import numpy as np
        #     mask_copy = np.array(mask)

        #     label_image = measure.label(mask_copy)
        #     regions = regionprops(label_image)
        #     for region in regions:
        #         # take regions with large enough areas
                
        #         minr, minc, maxr, maxc = region.bbox
        #         # crop mask and image
        #         extend_region = random.randint(10, 30)
        #         crop_img = img.crop((minc-extend_region, minr-extend_region, maxc+extend_region, maxr+extend_region))
        #         crop_mask = mask.crop((minc-extend_region, minr-extend_region, maxc+extend_region, maxr+extend_region))
        #         # new position to paste crop image( ignore the position of crop image)
        #         minr = random.randint(0, self.img_size[1] - (maxr - minr + 2*extend_region))
        #         minc = random.randint(0, self.img_size[0] - (maxc - minc + 2*extend_region))
        #         # paste crop image to original image
        #         rotated_degree = random.randint(0, 3)*90
        #         flipped_img = crop_img.transpose(Image.FLIP_LEFT_RIGHT).rotate(rotated_degree)
        #         img.paste(flipped_img, (minc, minr))
        #         flipped_mask = crop_mask.transpose(Image.FLIP_LEFT_RIGHT).rotate(rotated_degree)
        #         mask.paste(flipped_mask, (minc, minr))


        img, mask = self.transform(img), transforms.ToTensor()(mask)
        return img, mask

    def __len__(self):
        return len(self.names)

    def _sync_transform(self, img, mask):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        img_size = self.img_size # [640,512]
        # random scale (short edge)
        short_edg = min(img_size)
        scale = random.randint(int(short_edg / 2), int(short_edg * self.scale))
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
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))


        # import sys
        # sys.exit()
        return img, mask

    def _val_sync_transform(self, img, mask):
        # resize img to [self.img_size[0], self.img_size[1]] with ratio and padding
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
        img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)


        return img, mask

    def _testval_sync_transform(self, img, mask):
        img = img.resize(self.img_size, Image.BILINEAR)
        mask = mask.resize(self.img_size, Image.NEAREST)
        return img, mask

# test code
if __name__ == '__main__':
   from torch.utils.data import DataLoader
   import matplotlib.pyplot as plt
   import numpy as np
   import argparse

   parser = argparse.ArgumentParser()
   parser.add_argument('--img_size', type=tuple, default=(640, 512))
   args = parser.parse_args()

   dataset = Get_Dataset(args, mode='train')
   dataset.__getitem__(5000)
   dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

#    for i, (img, mask) in enumerate(dataloader):
#        # print(img.size(), mask.size())
#        continue


