import os.path

import torch
import torch.utils.data as Data
import torchvision.transforms as transforms

from PIL import Image, ImageOps, ImageFilter
import os.path as osp
import sys
import random
import numpy as np


class CustomDataset(Data.Dataset):
    def __init__(self, args, mode='train', scale = 1.5):
        
        self.scale = scale

        self._sirst = SirstDataset(args, mode, self.scale)
        self._irstd1k = IRSTD1K_Dataset(args, mode, self.scale)
        self._light_irst = Light_IRST_Dataset(args, mode, self.scale)
        # self._MDvsFA_cGAN = MD_FA_cGAN_Dataset(args, mode, self.scale)
        self._MDvsFA_cGAN =  Val_Dataset(args,mode,self.scale, base_dir = '../work_seg/Datasets/data_custom/add_data')
        self._custom = Custom_dataset(args, mode, self.scale)
        self.mavic1 = Val_Dataset(args,mode,self.scale, base_dir = '../work_seg/Datasets/data_custom/Mavic1')
        self.mavic2 = Val_Dataset(args,mode,self.scale, base_dir = '../work_seg/Datasets/data_custom/Mavic2')
        self.sirst_v2 = Val_Dataset(args,mode,self.scale, base_dir = '../work_seg/Datasets/data_custom/sirst_v2')

        self.names = self._sirst.names + self._irstd1k.names + self._light_irst.names + self._MDvsFA_cGAN.names + self._custom.names + self.mavic1.names + self.mavic2.names + self.sirst_v2.names
        self.mode = mode

    def __getitem__(self, i):
        if i < len(self._sirst.names):
            return self._sirst.__getitem__(i)
        elif i >= len(self._sirst.names) and i < len(self._sirst.names) + len(self._irstd1k.names):
            return self._irstd1k.__getitem__(i - len(self._sirst.names))
        elif i >= len(self._sirst.names) + len(self._irstd1k.names) and i < len(self._sirst.names) + len(self._irstd1k.names) + len(self._light_irst.names):
            return self._light_irst.__getitem__(i - len(self._sirst.names) - len(self._irstd1k.names))
        elif i >= len(self._sirst.names) + len(self._irstd1k.names) + len(self._light_irst.names) and i < len(self._sirst.names) + len(self._irstd1k.names) + len(self._light_irst.names) + len(self._MDvsFA_cGAN.names):
            return self._MDvsFA_cGAN.__getitem__(i - len(self._sirst.names) - len(self._irstd1k.names) - len(self._light_irst.names))
        elif i >=  len(self._sirst.names) + len(self._irstd1k.names) + len(self._light_irst.names) and i < len(self._sirst.names) + len(self._irstd1k.names) + len(self._light_irst.names) + len(self._MDvsFA_cGAN.names) + len(self._custom.names):
            return self._custom.__getitem__(i - len(self._sirst.names) - len(self._irstd1k.names) - len(self._light_irst.names) - len(self._MDvsFA_cGAN.names))
        elif i >= len(self._sirst.names) + len(self._irstd1k.names) + len(self._light_irst.names) and i < len(self._sirst.names) + len(self._irstd1k.names) + len(self._light_irst.names) + len(self._MDvsFA_cGAN.names) + len(self._custom.names) + len(self.mavic1.names):
            return self.mavic1.__getitem__(i - len(self._sirst.names) - len(self._irstd1k.names) - len(self._light_irst.names) - len(self._MDvsFA_cGAN.names) - len(self._custom.names))
        elif i >= len(self._sirst.names) + len(self._irstd1k.names) + len(self._light_irst.names) and i < len(self._sirst.names) + len(self._irstd1k.names) + len(self._light_irst.names) + len(self._MDvsFA_cGAN.names) + len(self._custom.names) + len(self.mavic1.names) + len(self.mavic2.names):
            return self.mavic2.__getitem__(i - len(self._sirst.names) - len(self._irstd1k.names) - len(self._light_irst.names) - len(self._MDvsFA_cGAN.names) - len(self._custom.names) - len(self.mavic1.names))
        
        else:
            return self.sirst_v2.__getitem__(i - len(self._sirst.names) - len(self._irstd1k.names) - len(self._light_irst.names) - len(self._MDvsFA_cGAN.names) - len(self._custom.names) - len(self.mavic1.names) - len(self.mavic2.names))
    def __len__(self):
        return len(self.names)

class SirstDataset(Data.Dataset):

    def __init__(self, args, mode='train', scale = 1.5):

        base_dir = '../work_seg/Datasets/data_custom/SIRST'

        if mode == 'train':
            txtfile = 'trainval.txt'
        elif mode == 'val':
            txtfile = 'test.txt'
        self.scale = scale
        self.list_dir = osp.join(base_dir, 'idx_427', txtfile)
        self.imgs_dir = osp.join(base_dir, 'images')
        self.label_dir = osp.join(base_dir, 'masks')

        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]

        # self.list_dir = osp.join(base_dir, 'idx_320', txtfile)
        # with open(self.list_dir, 'r') as f:
        #     self.names += [line.strip() for line in f.readlines()]

        self.mode = mode
        self.img_size = args.img_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.45563722, 0.45559042, 0.45564058], [0.2213023,  0.22129012, 0.22129919]),
            transforms.Normalize([0.45563722, 0.45559042, 0.45564058], [0.2213023,  0.22129012, 0.22129919]),
        ])

    def __getitem__(self, i):

        name = self.names[i]
        img_path = osp.join(self.imgs_dir, name + '.png')
        label_path = osp.join(self.label_dir, name + '_pixels0.png')

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path)

        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._testval_sync_transform(img, mask)
        else:
            raise ValueError("Unkown self.mode")

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


class IRSTD1K_Dataset(Data.Dataset):

    def __init__(self, args, mode='train', scale = 1.5):

        base_dir = '/mnt/data/danhnt/Seg/work_seg/Datasets/data_custom/IRSTD-1k/'

        if mode == 'train':
            txtfile = 'trainval.txt'
        elif mode == 'val':
            txtfile = 'test.txt'
        self.scale = scale
        self.list_dir = osp.join(base_dir, txtfile)
        self.imgs_dir = osp.join(base_dir, 'images')
        self.label_dir = osp.join(base_dir, 'masks')

        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]

        self.mode = mode
        self.img_size = args.img_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.46294058, 0.46289157, 0.4629441 ], [0.21915354, 0.21914229, 0.21915013]),
            # transforms.Normalize([0.45563722, 0.45559042, 0.45564058], [0.2213023,  0.22129012, 0.22129919]),
            transforms.Normalize([0.28450727, 0.28450724, 0.28450724], [0.22880708, 0.22880709, 0.22880709])
        ])

    def __getitem__(self, i):

        name = self.names[i]
        img_path = osp.join(self.imgs_dir, name + '.png')
        label_path = osp.join(self.label_dir, name + '.png')

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path)

        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            raise ValueError("Unkown self.mode")

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


class Light_IRST_Dataset(Data.Dataset):

    def __init__(self, args, mode='train', scale = 1.5):

        base_dir = '../work_seg/Datasets/data_custom/Light-IRST'

        if mode == 'train':
            txtfile = 'trainval.txt'
        elif mode == 'val':
            txtfile = 'test.txt'
        self.scale = scale
        self.list_dir = osp.join(base_dir, '50_50', txtfile)
        self.imgs_dir = osp.join(base_dir, 'images')
        self.label_dir = osp.join(base_dir, 'masks')

        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]

        self.names = [name for name in self.names if os.path.isfile(os.path.join(self.imgs_dir, name + '.png')) and os.path.isfile(os.path.join(self.label_dir, name + '.png'))]

        self.mode = mode
        self.img_size = args.img_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.46294058, 0.46289157, 0.4629441 ], [0.21915354, 0.21914229, 0.21915013]),
            transforms.Normalize([0.45563722, 0.45559042, 0.45564058], [0.2213023,  0.22129012, 0.22129919]),
        ])

    def __getitem__(self, i):
        try:
            name = self.names[i]
            img_path = osp.join(self.imgs_dir, name + '.png')
            label_path = osp.join(self.label_dir, name + '.png')

            img = Image.open(img_path).convert('RGB')
            mask = Image.open(label_path)
        except:
            new_idx = random.randint(0, len(self.names) - 1)
            name = self.names[new_idx]
            img_path = osp.join(self.imgs_dir, name + '.png')
            label_path = osp.join(self.label_dir, name + '.png')

            img = Image.open(img_path).convert('RGB')
            mask = Image.open(label_path)

        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._testval_sync_transform(img, mask)
        else:
            raise ValueError("Unkown self.mode")

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


class MD_FA_cGAN_Dataset(Data.Dataset):

    def __init__(self, args, mode='train', scale = 1.5):

        # base_dir = '../work_seg/Datasets/data_custom/MDvsFA_cGAN'
        base_dir = '../work_seg/Datasets/data_custom/MDvsFA_cGAN'

        if mode == 'train':
            txtfile = 'train.txt'
        elif mode == 'val':
            txtfile = 'test.txt'
        self.scale = scale
        self.list_dir = osp.join(base_dir, txtfile)
        self.imgs_dir = osp.join(base_dir, 'images')
        self.label_dir = osp.join(base_dir, 'masks')

        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]

        self.mode = mode
        self.img_size = args.img_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        #    transforms.Normalize([0.46294058, 0.46289157, 0.4629441 ], [0.21915354, 0.21914229, 0.21915013]),
        transforms.Normalize([0.45563722, 0.45559042, 0.45564058], [0.2213023,  0.22129012, 0.22129919]),
        ])

    def __getitem__(self, i):

        name = self.names[i]
        img_path = osp.join(self.imgs_dir, name)
        label_path = osp.join(self.label_dir, name.replace('img', 'lbs'))

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path).convert('L')

        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._testval_sync_transform(img, mask)
        else:
            raise ValueError("Unkown self.mode")

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



class Custom_dataset(Data.Dataset):
    def __init__(self, args, mode='train', scale = 1.5):
        base_dir = '../work_seg/Datasets/data_custom/Custom'
        if mode == 'train':
            txtfile = 'trainval.txt'
        elif mode == 'val':
            txtfile = 'test.txt'
        self.list_dir = osp.join(base_dir, txtfile)
        self.imgs_dir = osp.join(base_dir, 'images')
        self.label_dir = osp.join(base_dir, 'masks')
        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]
        self.mode = mode
        self.img_size = args.img_size
        self.scale = scale
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        #    transforms.Normalize([0.46294058, 0.46289157, 0.4629441 ], [0.21915354, 0.21914229, 0.21915013]),
        #transforms.Normalize([0.45563722, 0.45559042, 0.45564058], [0.2213023,  0.22129012, 0.22129919]),
        ])

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.imgs_dir, name + '.png')
        label_path = osp.join(self.label_dir, name + '.png')
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path)
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._testval_sync_transform(img, mask)
        else:
            raise ValueError("Unkown self.mode")
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

class Val_Dataset(Data.Dataset):

    def __init__(self, args, mode='train', scale = 1.5, base_dir = '../work_seg/Datasets/data_custom/Mavic1'):

        self.base_dir = base_dir

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
        self.label_dir = osp.join(base_dir, 'mask')

        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]

        self.mode = mode
        self.img_size = args.img_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        #   transforms.Normalize([0.5, 0.5, 0.5 ], [0.5, 0.5, 0.5]),
        transforms.Normalize([0.45563722, 0.45559042, 0.45564058], [0.2213023,  0.22129012, 0.22129919]),
        ])

    def __getitem__(self, i):

        name = self.names[i]
        if "Mavic1" in self.base_dir:
            img_path = osp.join(self.imgs_dir, name + '.jpg')
        elif "Mavic2" in self.base_dir:
            img_path = osp.join(self.imgs_dir, name + '.PNG')
        # elif "open" in self.base_dir:
        #     img_path = osp.join(self.imgs_dir, name + '_pixels0.png')
        else :
            img_path = osp.join(self.imgs_dir, name + '.png')

        if "open" in self.base_dir:
            label_path = osp.join(self.label_dir, name + '_pixels0.png')
        else:
            label_path = osp.join(self.label_dir, name + '.png')
            
        if "bia" in self.base_dir:
            img_path = osp.join(self.imgs_dir, name + '.jpg')
            label_path = osp.join(self.label_dir, name + '.txt')
            
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path)
        # original_size = img.size
        # img = img.resize((640, 512))
        # bbox_annotations = []
        # with open(label_path, 'r') as file:
        #     for line in file:
                
        #         bbox = list(map(float, line.strip().split()))
        #         bbox_annotations.append(bbox)
        
        # mask = np.zeros((512, 640), dtype=np.uint8)
        
        # check = 0
        # if len(bbox_annotations) > 0:
        #     for bbox in bbox_annotations:
        #         class_id, x_center, y_center, width, height = bbox
                
               
        #         x_center = int(x_center * original_size[0])
        #         y_center = int(y_center * original_size[1])
        #         width = int(width * original_size[0])
        #         height = int(height * original_size[1])

               
        #         x1 = int(x_center - width / 2)
        #         y1 = int(y_center - height / 2)
        #         x2 = int(x_center + width / 2)
        #         y2 = int(y_center + height / 2)

                
        #         # target = np.ones((abs(y2-y1),abs(x2-x1)), dtype=np.uint8)
        #         mask[y1:y2, x1:x2] = 255
        #         # print(x1,x2,y1,y2)
        #         # check = 1
                
        
        # mask = Image.fromarray(mask)
        # # if check ==1:
        # #     mask.save(img_path.split('/')[-1])
        

        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._testval_sync_transform(img, mask)
        else:
            raise ValueError("Unkown self.mode")

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

class GS_Dataset(Data.Dataset):

    def __init__(self, args, mode='train', scale = 1.5, base_dir = '../work_seg/Datasets/RGB_Drone/'):

        self.base_dir = base_dir

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

        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]

        self.mode = mode
        self.img_size = args.img_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            #transforms.Normalize([0.5186578 , 0.52004126 ,0.51559308], [0.03202766, 0.03567145, 0.04088691]),
        ])

    def __getitem__(self, i):

        name = self.names[i]

        img_path = osp.join(self.imgs_dir, name + '.jpg')
        label_path = osp.join(self.label_dir, name + '.png')

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path)

        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            raise ValueError("Unkown self.mode")

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
