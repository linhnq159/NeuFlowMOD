import torch
import torchvision.transforms as transforms
import cv2
import torch.utils.data as Data
import utils.data_custom as custom

from tqdm import tqdm
from argparse import ArgumentParser
from PIL import Image
from model.metrics import IoUMetric, nIoUMetric, PD_FA, ROCMetric2
#from utils.data import SirstDataset, IRSTD1K_Dataset
from model.ilnet import ILNet_S, ILNet_M, ILNet_L
from model.ilnet_V1 import ILNetV1_S
from model.LW_IRSTNet import LW_IRST_ablation
from model.LW_IRSTNet_V1 import LW_net
from utils.data_agumentation import Get_Dataset
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = ArgumentParser(description='Implement of MY net')

    parser.add_argument('--img_size', type= list, default=[427,342], help='input image size')
    parser.add_argument('--batch_size', type=int, default=10, help='batch_size for testing')
    parser.add_argument('--dataset', type=str, default='sirst', help='datasets: sirst or IRSTD-1k')
    parser.add_argument('--mode', type=str, default='L', help='mode: L, M, S')
    parser.add_argument('--checkpoint', type=str, default=r'.pth', help='checkpoint: .pth')

    args = parser.parse_args()
    return args


class Val:

    def __init__(self, args, load_path: str):
        self.args = args

        # datasets
        if args.dataset == 'sirst':
            self.val_set = SirstDataset(args, mode='val')
        elif args.dataset == 'IRSTD-1k':
            self.val_set = custom.IRSTD1K_Dataset(args, mode='val')
        elif args.dataset == "custom":
            self.val_set = custom.CustomDataset(args, mode='val')
        elif args.dataset == "real":
            self.val_set = custom.Val_Dataset(args, mode = 'val', base_dir = '../work_seg/Datasets/data_infer/real_data_val')
        elif args.dataset == 'noise_data':
            self.val_set = Get_Dataset(args, mode='val')
        else:
            NameError
        self.val_data_loader = Data.DataLoader(self.val_set, batch_size= args.batch_size, num_workers=8, pin_memory=True, drop_last=False)

        assert args.mode in ['L', 'M', 'S', 'SV1', 'LW','LW_V1']
        if args.mode == 'L':
            self.model = ILNet_L()
            self.model = self.model.cuda()
            import torchsummary
            torchsummary.summary(self.model, (3, 640, 512))
        elif args.mode == 'M':
            self.model = ILNet_M()
        elif args.mode == 'S':
            self.model = ILNet_S()
            self.model = self.model.cuda()
        elif args.mode == 'SV1':
            self.model = ILNetV1_S()
            self.model = self.model.cuda()
        elif args.mode == 'LW_V1':
            self.model = LW_net()
            self.model = self.model.cuda()
        elif args.mode == 'LW':
            self.model = LW_IRST_ablation(channel=(8, 32, 64), dilations=(2,4,8,16), kernel_size=(7,7,7,7), padding=(3,3,3,3))
            self.model = self.model.cuda()
        else:
            NameError


        # ignored_list = []
        # for i in range(0, 5):
        #     for j in [1,4]:
        #         ignored_list.append('ipof_modules.{}.conv_up.{}.weight'.format(i, j))
        #         ignored_list.append('ipof_modules.{}.conv_up.{}.bias'.format(i, j))

        checkpoint = torch.load(load_path)
        state_dict = checkpoint['model']
        # transfer weights from the old model to the new model, ignore the weights of layer in ignored_list
        # new_state_dict = self.model.state_dict()
        # for k in state_dict.keys():
        #     if k in new_state_dict.keys() and k not in ignored_list:
        #         new_state_dict[k] = state_dict[k]
        self.model.load_state_dict(state_dict)
        
        print(device)
        self.model.to(device)
        self.model.eval()

        # import torchsummary
        # torchsummary.summary(self.model, (3, 512, 640))
        dump_in = torch.rand(1, 3, 512, 640).to(device)
        dump_out = self.model(dump_in)
        print('Model loaded successfully')

        self.iou_metric = IoUMetric()
        self.nIoU_metric = nIoUMetric(1, score_thresh=0.5)
        self.PD_FA = PD_FA(args.img_size)
        self.ROC = ROCMetric2(1, bins=10)

    def test_model(self):
        self.iou_metric.reset()
        self.nIoU_metric.reset()
        self.PD_FA.reset()

        self.model.eval()
        print(next(self.model.parameters()).device)
        tbar = tqdm(self.val_data_loader)
        for i, (data, labels) in enumerate(tbar):
            with torch.no_grad():
                output = self.model(data.cuda())
                output = output.cpu()

            self.iou_metric.update(output, labels)
            self.nIoU_metric.update(output, labels)
            self.PD_FA.update(output, labels)
            # output2 = output.squeeze(0).permute(1, 2, 0)
            # labels2 = labels.squeeze(0)
            # self.ROC.update(output2, labels2)

            _, IoU = self.iou_metric.get()
            _, nIoU = self.nIoU_metric.get()
            Fa, Pd = self.PD_FA.get(len(self.val_set))
            # ture_positive_rate, false_positive_rate, recall, precision = self.ROC.get()
            ture_positive_rate, false_positive_rate, recall, precision = 0,0,0,0

            # 进度条描述信息
            tbar.set_description('IoU:%f, nIoU:%f, Fa:%.10f, Pd:%.10f'
                                 % (IoU, nIoU, Fa, Pd))
        return IoU, nIoU, Fa, Pd, ture_positive_rate, false_positive_rate

    def validation(self, epoch):

        self.iou_metric.reset()
        self.nIoU_metric.reset()
        self.PD_FA.reset()

        eval_losses = []
        self.model.cuda()
        self.model.eval()
        tbar = tqdm(self.val_data_loader)
        for i, (data, labels) in enumerate(tbar):
            with torch.no_grad():
                output = self.model(data.cuda())
                output = output.cpu()

            # loss = self.criterion(output, labels)
            
            # eval_losses.append(loss.item())

            self.iou_metric.update(output, labels)
            self.nIoU_metric.update(output, labels)
            self.PD_FA.update(output, labels)

            _, IoU = self.iou_metric.get()
            _, nIoU = self.nIoU_metric.get()
            Fa, Pd = self.PD_FA.get(len(self.val_set))

            tbar.set_description('  Epoch:%3d, eval loss:%f, IoU:%f, nIoU:%f, Fa:%.8f, Pd:%.5f'
                                 % (epoch, 0, IoU, nIoU, Fa, Pd))
if __name__ == "__main__":
    args = parse_args()

    value = Val(args, load_path=args.checkpoint)
    value.validation(0)
    # IoU, nIoU, Fa, Pd, ture_positive_rate, false_positive_rate = value.test_model()
    # print('IoU:{},\n nIoU:{},\n Fa:{},\n Pd:{},\n TPR:{},\n FPR:{}'.format(IoU, nIoU, Fa, Pd, ture_positive_rate, false_positive_rate))
