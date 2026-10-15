import torch
import torchvision.transforms as transforms
import cv2
import torch.utils.data as Data
import utils.data_custom as custom
import onnxruntime as ort

from tqdm import tqdm
from argparse import ArgumentParser
from PIL import Image
from model.metrics import IoUMetric, nIoUMetric, PD_FA, ROCMetric2
from utils.data import SirstDataset, IRSTD1K_Dataset
from model.ilnet import ILNet_S, ILNet_M, ILNet_L
from model.ilnet_V1 import ILNetV1_S
from model.LW_IRSTNet import LW_IRST_ablation
from metric import Detect
from model.LW_IRSTNet_V1 import LW_net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = ArgumentParser(description='Implement of MY net')

    parser.add_argument('--img_size', type= list, default=[640,512], help='input image size')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size for testing')
    parser.add_argument('--dataset', type=str, default='sirst', help='datasets: sirst or IRSTD-1k')
    parser.add_argument('--mode', type=str, default='L', help='mode: L, M, S')
    parser.add_argument('--checkpoint', type=str, default=r'.pth', help='checkpoint: .pth')
    parser.add_argument('--data_path', type=str,  help='data path')

    args = parser.parse_args()
    return args


class Val:

    def __init__(self, args, load_path: str):
        self.args = args

        # datasets
        if args.dataset == 'sirst':
            self.val_set = SirstDataset(args, mode='val')
        elif args.dataset == 'IRSTD-1k':
            self.val_set = IRSTD1K_Dataset(args, mode='val')
        elif args.dataset == "custom":
            self.val_set = custom.CustomDataset(args, mode='val')
        elif args.dataset == "real":
            self.val_set = custom.Val_Dataset(args, mode = 'val', base_dir = args.data_path)
        else:
            NameError
        self.val_data_loader = Data.DataLoader(self.val_set, batch_size=args.batch_size)

        assert args.mode in ['L', 'M', 'S', 'SV1', 'LW','LW_V1']
        if args.mode == 'L':
            self.model = ILNet_L()
            self.model = self.model.cuda()
            # import torchsummary
            # torchsummary.summary(self.model, (3, 640, 512))
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


        checkpoint = torch.load(load_path)
        state_dict = checkpoint['model']
        self.model.load_state_dict(state_dict)

        # self.load_network(load_path)

        # import torchsummary
        # torchsummary.summary(self.model, (3, 512, 640))
        # dump_in = torch.rand(1, 3, 512, 640).to(device)
        # dump_out = self.model(dump_in)
        # print('Model loaded successfully')

        self.Detect = Detect(isDanger=False)
        self.iou_metric = IoUMetric()
        self.nIoU_metric = nIoUMetric(1, score_thresh=0.5)
        self.PD_FA = PD_FA(args.img_size)
        self.ROC = ROCMetric2(1, bins=10)

    def load_network(self, model):
        device = ort.get_device()
        cuda = True if device == 'GPU' else False
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            so = ort.SessionOptions()
            so.log_severity_level = 3

            self.model = ort.InferenceSession(model, providers=providers, sess_options=so)
            self.output_details = [i.name for i in self.model.get_outputs()]
            self.input_details = [i.name for i in self.model.get_inputs()]

            self.is_inititated = True
        except Exception as e:
            raise Exception(f"Cannot load model {model}: {e}")        

    def test_model(self):
        self.Detect.reset()
        self.iou_metric.reset()
        self.nIoU_metric.reset()
        self.PD_FA.reset()

        self.model.to(device)
        self.model.eval()
        print(next(self.model.parameters()).device)
        tbar = tqdm(self.val_data_loader)
        for i, (data, labels) in enumerate(tbar):
            
            # ort_inputs = {self.input_details[0]: data.detach().numpy()}
            # ort_outs = self.model.run(None, ort_inputs)

            # output = ort_outs[0]            
            with torch.no_grad():
                output = self.model(data.cuda())
                output = output.cpu()
                # output = output.squeeze(0).permute(1, 2, 0).cpu()
                # labels = labels.squeeze(0).permute(1, 2, 0).cpu()

            self.Detect.update(output, labels)
            self.iou_metric.update(output, labels)
            self.nIoU_metric.update(output, labels)
            self.PD_FA.update(output, labels)
            output2 = output.squeeze(0).permute(1, 2, 0)
            labels2 = labels.squeeze(0)
            self.ROC.update(output2, labels2)

            result, total_pixel, miss, false_detect = self.Detect.get()
            _, IoU = self.iou_metric.get()
            _, nIoU = self.nIoU_metric.get()
            Fa, Pd = self.PD_FA.get(len(self.val_set))
            ture_positive_rate, false_positive_rate, recall, precision = self.ROC.get()
            # IoU, nIoU, Fa, Pd, ture_positive_rate, false_positive_rate = 0,0,0,0,0,0
            tbar.set_description('IoU:%f, nIoU:%f, Fa:%.10f, Pd:%.10f, True_detect:%f, miss_detect:%f, false_detect:%f' % (IoU, nIoU, Fa, Pd, result, miss, false_detect))
        self.Detect.get_max_min()
        
        return IoU, nIoU, Fa, Pd, ture_positive_rate, false_positive_rate, result, total_pixel




if __name__ == "__main__":
    args = parse_args()

    value = Val(args, load_path=args.checkpoint)
    IoU, nIoU, Fa, Pd, ture_positive_rate, false_positive_rate,result, total_pixel = value.test_model()
    print('IoU:{},\n nIoU:{},\n Fa:{},\n Pd:{},\n TPR:{},\n FPR:{}, \n True_detect:{}, miss_detect:{}, false_detect:{}'.format(IoU, nIoU, Fa, Pd, ture_positive_rate, false_positive_rate, result, total_pixel))
    print("total_pixel: ", total_pixel)
