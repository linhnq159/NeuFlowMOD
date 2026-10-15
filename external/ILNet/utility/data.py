import torch
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import random
# To:
import sys
import os
from argparse import ArgumentParser
# In /data/linhnq11/segmentation/ILNet-master/utility/data.py
# Replace the import statement with:
import sys
import os

# Get the absolute path of the project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
utils_path = os.path.join(project_root, 'utils')

# Add to Python path
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

# Import the CustomDataset
from data import CustomDataset
#from .augmentation import Cutout

from utility.args import Args


Args.add_argument("--dataThreads", type=int, help="Number of CPU threads for dataloaders.")
Args.add_argument("--dataDir", type=str, help="main directory to store datasets")
Args.add_argument("--batchSize", type=int, help="batch size")
Args.add_argument("--dataset", type=str, help="Dataset")
Args.add_argument("--imageSize", type=int, help="resize images after augmentation. 0 for no resizing")
Args.add_argument("--flip", type=bool, help="flip horizontally")
Args.add_argument("--crop", type=bool, help="crop 32x32 padding 4")
Args.add_argument("--cut", type=bool, help="cutout")
Args.add_argument("--cutoutProp", type=float, help="Probability for cutout augmenation.")
Args.add_argument("--randAugment", type=bool, help="")
Args.add_argument("--randAugment_magnitude", type=int, help="")
Args.add_argument("--mixup", type=bool, help="")
Args.add_argument("--mixupProp", type=float, help="")
Args.add_argument("--normalize", type=str, help="std or min_max")

class ImageNet(torchvision.datasets.ImageFolder):
    def __init__(self, root, train = True, download = None, transform = []):
        self.train = train
        if train:
            dir = os.path.join(root, "ImageNet", 'train')
            transform.transforms = [
                transforms.RandomResizedCrop(Args.imageSize if Args.imageSize is not None else 224, scale=(0.08, 1.0), ratio=(3. / 4, 4. / 3.))
                ] + transform.transforms
        else:
            dir = os.path.join(root, "ImageNet", 'val')
            transform.transforms = [
                CenterCrop(),
                ] + transform.transforms

        super(ImageNet, self).__init__(dir, transform = transform)

# name, numClasses
availableDatasets = {
    "ImageNet": [ImageNet, 1000],
    "CIFAR10" : [torchvision.datasets.CIFAR10 , 10],
    "CIFAR100": [torchvision.datasets.CIFAR100, 100],
}
dataSetStatistics = {
    "ImageNet": [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    "CIFAR10" : [[0.4913999140262604, 0.48215872049331665, 0.4465313255786896], [0.24703197181224823, 0.243484228849411, 0.26158687472343445]],
    "CIFAR100": [[0.5070753693580627, 0.4865487813949585, 0.44091784954071045], [0.2673334777355194, 0.25643861293792725, 0.2761503756046295]],
}

def worker_init_fn(id):
    """ manually seed each workers np random generator. see https://github.com/pytorch/pytorch/issues/5059"""
    uint64_seed = torch.initial_seed()
    np.random.seed([uint64_seed >> 32, uint64_seed & 0xffff_ffff])


class NormalizeMinMax(torch.nn.Module):
    """Normalize a x\in[0,1] tensor to x\in[-1,1]"""
    def __init__(self, inplace = False):
        super().__init__()
        self.min = -1
        self.max = 1
        self.inplace = inplace

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.inplace:
            tensor = tensor.clone()

        tensor.mul_(self.max-self.min).add_(self.min)
        return tensor

class CenterCrop(torch.nn.Module):
    def __init__(self, crop_fraction = 0.875):
        super().__init__()
        self.crop_fraction = crop_fraction

    def forward(self, img):
        size = int(min(img.size) * self.crop_fraction)
        return transforms.functional.center_crop(img, size)


class DataLoader:
    def __init__(self):
         
        self.datasetName = Args.dataset
        #try:
        #    self.dataset, self.numClasses = availableDatasets[self.datasetName]
        #except KeyError:
        #    raise NameError(f"Dataset {self.datasetName} not found. Available datasets are: {', '.join(availableDatasets.keys())}")

        if Args.normalize not in ["std", "min_max"]:
            raise RuntimeError(f"Arguments 'normalize' must be 'std' or 'min_max' but is '{Args.normalize}'")

        if Args.normalize == "std":
            if self.datasetName in dataSetStatistics:
                mean, std = dataSetStatistics[self.datasetName]
            else:
                print("Calculating Mean and Std...")
                mean, std = self._get_statistics()
                print(f"Mean = {list(map(float,mean))}, Std = {list(map(float,std))}")




        transform_list = []
        if Args.mixup:
            raise NotImplementedError #@TODO add mixup
        if Args.randAugment:
            transform_list.append(transforms.RandAugment(magnitude = Args.randAugment_magnitude))
        if Args.flip:
            transform_list.append(transforms.RandomHorizontalFlip())
        if Args.cut:
            transform_list.append(Cutout(mask_val=mean))
        if Args.crop and self.datasetName in ["CIFAR10","CIFAR100"]:
            transform_list.append(transforms.RandomCrop(size=(32, 32), padding=4))

        if Args.imageSize != 0:
            transform_list += [
                transforms.Resize(Args.imageSize, interpolation = transforms.InterpolationMode.BICUBIC),
            ]

        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean, std) if Args.normalize == "std" else NormalizeMinMax(),
        ]
        train_transform = transforms.Compose(transform_list)

        if Args.imageSize != 0:
            test_transform = [
                transforms.Resize(Args.imageSize, interpolation = transforms.InterpolationMode.BICUBIC),
            ]
        else:
            test_transform = []

        test_transform = transforms.Compose(test_transform + [
            transforms.ToTensor(),
            transforms.Normalize(mean, std) if Args.normalize == "std" else NormalizeMinMax(),
        ])

        train_set = self.dataset(root=Args.dataDir, train=True, download=True, transform=train_transform)
        test_set  = self.dataset(root=Args.dataDir, train=False, download=True, transform=test_transform)
        
        if self.datasetName == "ImageNet":
            assert len(train_set) == 1281167, "train data is incomplete!"
            assert len(test_set) == 50000, "test data is incomplete!"
        
        # to get random order of train data the seed has to be set manually. the seed value must be constant among all processes. 
        seed = torch.tensor(random.getrandbits(32)).cuda(torch.distributed.get_rank() % torch.cuda.device_count())
        torch.distributed.broadcast(seed, src = 0)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True , seed = seed.item())
        test_sampler  = torch.utils.data.distributed.DistributedSampler(test_set , shuffle=False)
        
        #calc BS per worker
        batch_size = Args.batchSize // torch.distributed.get_world_size() + int(Args.batchSize % torch.distributed.get_world_size() > torch.distributed.get_rank())
        self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=Args.dataThreads, worker_init_fn=worker_init_fn, sampler = train_sampler)
        self.test  = torch.utils.data.DataLoader(test_set , batch_size=batch_size, num_workers=Args.dataThreads, worker_init_fn=worker_init_fn, sampler = test_sampler)

    def _get_statistics(self):
#        data = self.dataset(root=Args.dataDir, train=True, download=True, transform=transforms.ToTensor())
        self.args = self.parse_args()
        data = CustomDataset(data_path = '/data/linhnq11/MOD/CPP/mod/Data/Segment', mode='train', args = self.args)
        firstMoment = torch.zeros(3)
        secondMoment = torch.zeros(3)
        N = 0

        for inputs, _ in torch.utils.data.DataLoader(data, batch_size=1, num_workers=Args.dataThreads):
            print("inputs : ", inputs)
            #print("inputs.shape[0] : ", inputs.shape[0])
            N += inputs.shape[0]
            for i in range(3):
                firstMoment[i] += inputs[:,i,:,:].mean()*inputs.shape[0]
                secondMoment[i] += (inputs[:,i,:,:]**2).mean()*inputs.shape[0]

        print("N : ", N)
        print("firstMoment : ", firstMoment)
        print("secondMoment : ", secondMoment)
        firstMoment .div_(N)
        secondMoment.div_(N)
        return firstMoment, torch.sqrt(secondMoment-firstMoment**2)

    def parse_args(self):
        parser = ArgumentParser(description='Implement of MY model')

        parser.add_argument('--work_dir', type=str, default='work_seg/checkpoint', help='save weights folder')

        parser.add_argument('--img_size', type= list, default=[960,540], help='input image size')
        parser.add_argument('--batch_size', type=int, default=8, help='batch_size for training')
        parser.add_argument('--epochs', type=int, default=600, help='number of epochs')
        parser.add_argument('--warm_up_epochs', type=int, default=10, help='warm up epochs')
        parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
        parser.add_argument('--optim', type= str, default = "adan", help= "Optimizer Algrothirms: msam, adan,...")
        parser.add_argument('--lr_sch', type= str, default = "ilnet", help= "Learning rate schedulers: ilnet, cos")

        parser.add_argument('--dataset', type=str, default='sirst', help='datasets: sirst or IRSTD-1k')
        parser.add_argument('--mode', type=str, default='L', help='mode: L, M, S')
        parser.add_argument('--loss', type=str, default='CUSTOM')

        parser.add_argument('--resume', default='', help='resume from checkpoint')
        parser.add_argument('--start_epoch', default=1, type=int, help='start epoch')

        parser.add_argument('--amp', default=True, help='Use torch.cuda.amp for mixed precision training')

        parser.add_argument('--teacher', default='', help='teacher model path')
        parser.add_argument('--alpha', default=0.5, type=float, help='alpha for distillation loss')
        parser.add_argument('--T', default=10, type=float, help='temperature for distillation loss')
        parser.add_argument('--device', type = str, default='cuda:0', help='device')
        parser.add_argument('--eval_freq', type=int, default=1)
        parser.add_argument('--data_path', type = str)
        parser.add_argument('--gain', type=float, default=0.4, help='Gain loss')
        parser.add_argument('--multi_gpu', action='store_true', help='use multi gpu')
        parser.add_argument('--transfer', type = str)

        args = parser.parse_args()
        return args


if __name__ == '__main__':
    import cv2

    data_path = '/data/linhnq11/MOD/CPP/mod/Data/Segment/Test/images'
    Args.dataDir = data_path
    Args.dataset = "custom"
    Args.normalize = "std"
    Args.dataThreads = 24


    dataset_ = DataLoader()
