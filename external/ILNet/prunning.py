import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from tensorboardX import SummaryWriter
from argparse import ArgumentParser
from tqdm import tqdm
import os
import os.path as ops
import numpy as np
import time

from utils.Adan import Adan
from utils.data import SirstDataset, IRSTD1K_Dataset
import utils.data_custom as DC
from utils.lr_scheduler import adjust_learning_rate, create_lr_scheduler
from model.loss import SoftIoULoss, criterion, dice_critetion
from model.unified_focal_loss_pytorch import AsymmetricUnifiedFocalLoss
from model.metrics import IoUMetric, nIoUMetric, PD_FA
from model.ilnet import ILNet_S, ILNet_M, ILNet_L
from model.ilnet_V1 import ILNetV1_S, ILNetV1_L
from model.ilnet_Batch import ILNet_S_Batch
from apex.contrib.sparsity import ASP
from apex import amp
from apex.opt import LazyPruner
import torch.onnx


# torch.backends.cudnn.benchmark = True


def parse_args():
    parser = ArgumentParser(description='Implement of MY model')

    parser.add_argument('--img_size', type= list, default=[640,512], help='input image size')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size for training')
    parser.add_argument('--epochs', type=int, default=600, help='number of epochs')
    parser.add_argument('--warm_up_epochs', type=int, default=10, help='warm up epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')

    parser.add_argument('--dataset', type=str, default='sirst', help='datasets: sirst or IRSTD-1k')
    parser.add_argument('--mode', type=str, default='L', help='mode: L, M, S')
    parser.add_argument('--loss', type=str, default='BCE')

    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=1, type=int, help='start epoch')

    parser.add_argument('--amp', default=True, help='Use torch.cuda.amp for mixed precision training')

    parser.add_argument('--teacher', default='', help='teacher model path')
    parser.add_argument('--alpha', default=0.5, type=float, help='alpha for distillation loss')
    parser.add_argument('--T', default=10, type=float, help='temperature for distillation loss')
    parser.add_argument('--device', type = str, default='cuda:0', help='device')
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--transfer', type = str)
    
    args = parser.parse_args()
    return args


class Trainer(object):

    def __init__(self, args):
        self.args = args

        # datasets
        if args.dataset == 'sirst':
            self.train_set = SirstDataset(args, mode='train')
            self.val_set = SirstDataset(args, mode='val')
        elif args.dataset == 'IRSTD-1k':
            self.train_set = DC.IRSTD1K_Dataset(args, mode='train')
            self.val_set = DC.IRSTD1K_Dataset(args, mode='val')
        elif args.dataset == 'custom':
            self.train_set = DC.CustomDataset(args, mode='train')
            self.val_set = DC.CustomDataset(args, mode='val')
        else:
            NameError

        self.train_data_loader = Data.DataLoader(self.train_set, batch_size=args.batch_size, shuffle=True,
                                                 num_workers=20, pin_memory=True, drop_last=True)
        self.val_data_loader = Data.DataLoader(self.val_set, batch_size=args.batch_size, num_workers=20, pin_memory=True, drop_last=True)

        assert args.mode in ['L', 'M', 'S', 'SV1', 'LV1', 'B']
        if args.mode == 'L':
            self.model = ILNet_L()
        elif args.mode == 'M':
            self.model = ILNet_M()
        elif args.mode == 'S':
            self.model = ILNet_S()
        elif args.mode == 'SV1':
            self.model = ILNetV1_S()
        elif args.mode == 'LV1':
            self.model = ILNetV1_L()
        elif args.mode == 'B':
            self.model = ILNet_S_Batch()
        else:
            NameError

        self.device = torch.device(args.device)

        self.model = self.model.to(self.device)
        self.scaler = torch.cuda.amp.GradScaler() if args.amp else None

        if args.teacher != '':
            self.teacher = ILNet_L()
            self.teacher.load_state_dict(torch.load(args.teacher)['model'])
            self.teacher = self.teacher.to(self.device)
            # self.teacher.eval()

        if args.loss.upper() == 'BCE':
            self.criterion = criterion
        elif args.loss.upper() == 'SOFTIOULOSS':
            self.criterion = SoftIoULoss()
        elif args.loss.upper() == 'ASIMMUNIFOCAL':
            self.criterion = AsymmetricUnifiedFocalLoss().forward_general
        elif args.loss.upper() =="DICE" :
            self.criterion = dice_critetion
        else:
            self.criterion = criterion 
        

        # self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        # self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=0)
        self.optimizer = Adan(self.model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

        self.lr_scheduler = create_lr_scheduler(self.optimizer, len(self.train_data_loader), args.epochs, warmup=True,
                                                warmup_epochs=args.warm_up_epochs)
        if args.resume:
            checkpoint = torch.load(args.resume)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.amp:
                self.scaler.load_state_dict(checkpoint["scaler"])

        if args.transfer:
            # ignored_list = []
            # for i in range(0, 5):
            #     for j in [1,4]:
            #         ignored_list.append('ipof_modules.{}.conv_up.{}.weight'.format(i, j))
            #         ignored_list.append('ipof_modules.{}.conv_up.{}.bias'.format(i, j))

            checkpoint = torch.load(args.transfer)
            state_dict = checkpoint['model']
            # # transfer weights from the old model to the new model, ignore the weights of layer in ignored_list
            # new_state_dict = self.model.state_dict()
            # for k in state_dict.keys():
            #     if k in new_state_dict.keys() and k not in ignored_list:
            #         new_state_dict[k] = state_dict[k]
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device)
            
        # ASP.prune_trained_model(self.model, self.optimizer)
        pruner = LazyPruner(self.model, param_names=["weight"], sparsity=0.05)
            


        self.iou_metric = IoUMetric()
        self.nIoU_metric = nIoUMetric(1, score_thresh=0.5)
        self.best_iou = -1
        self.best_nIoU = 0
        self.best_PD = 0
        self.best_FA = 1
        self.PD_FA = PD_FA(args.img_size)

        if args.resume:
            folder_name = os.path.abspath(
                os.path.dirname(os.path.abspath(os.path.dirname(args.resume) + os.path.sep + "."))
                + os.path.sep + ".")
        else:
            folder_name = '%s_bs%s_lr%s' % (time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())),
                                            args.batch_size, args.learning_rate)
            folder_name = args.mode + '_' + args.loss + '_' + folder_name

        if self.train_set.__class__.__name__ == 'SirstDataset':
            self.save_folder = ops.join('results_sirst/', folder_name)  # sirst
            self.save_pth = ops.join(self.save_folder, 'checkpoint')
            if not ops.exists('results_sirst'):
                os.mkdir('results_sirst')
            if not ops.exists(self.save_folder):
                os.mkdir(self.save_folder)
            if not ops.exists(self.save_pth):
                os.mkdir(self.save_pth)
        if self.train_set.__class__.__name__ == 'IRSTD1K_Dataset':
            self.save_folder2 = ops.join('results_IRSTD-1k/', folder_name)  # IRSTD-1k
            self.save_pth2 = ops.join(self.save_folder2, 'checkpoint')
            if not ops.exists('results_IRSTD-1k'):
                os.mkdir('results_IRSTD-1k')
            if not ops.exists(self.save_folder2):
                os.mkdir(self.save_folder2)
            if not ops.exists(self.save_pth2):
                os.mkdir(self.save_pth2)
        if self.train_set.__class__.__name__ == 'CustomDataset':
            self.save_folder3 = ops.join('results_custom/', folder_name)
            self.save_pth3 = ops.join(self.save_folder3, 'checkpoint')
            if not ops.exists('results_custom'):
                os.mkdir('results_custom')
            if not ops.exists(self.save_folder3):
                os.mkdir(self.save_folder3)
            if not ops.exists(self.save_pth3):
                os.mkdir(self.save_pth3)

        # # save args
        # with open(ops.join(self.save_folder2, 'config.txt'), 'w') as f:
        #     f.write(str(args))

        # Tensorboard SummaryWriter
        if self.train_set.__class__.__name__ == 'SirstDataset':
            self.writer = SummaryWriter(log_dir=self.save_folder)
            self.writer.add_text(folder_name, 'Args:%s, ' % args)
            # self.writer.add_graph(self.model, next(iter(self.train_data_loader))[0].to(self.device))
        if self.train_set.__class__.__name__ == 'IRSTD1K_Dataset':
            self.writer = SummaryWriter(log_dir=self.save_folder2)
            self.writer.add_text(folder_name, 'Args:%s, ' % args)
            # self.writer.add_graph(self.model, next(iter(self.train_data_loader))[0].to(self.device))
        if self.train_set.__class__.__name__ == 'CustomDataset':
            self.writer = SummaryWriter(log_dir=self.save_folder3)
            self.writer.add_text(folder_name, 'Args:%s, ' % args)
            # self.writer.add_graph(self.model, next(iter(self.train_data_loader))[0].to(self.device))

        if self.train_set.__class__.__name__ == 'SirstDataset':
            print('folder: %s' % self.save_folder)
        if self.train_set.__class__.__name__ == 'IRSTD1K_Dataset':
            print('folder: %s' % self.save_folder2)
        if self.train_set.__class__.__name__ == 'CustomDataset':
            print('folder: %s' % self.save_folder3)
        print('Args: %s' % args)

    def training(self, epoch):

        losses = []
        self.model.train()
        

        tbar = tqdm(self.train_data_loader)
        for i, (data, labels) in enumerate(tbar):
            data, labels = data.to(self.device), labels.to(self.device)

            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                output = self.model(data)
                loss = self.criterion(output, labels)

                # for i in range(len(output)):
                #     print(output[i].shape)
                # import sys
                # sys.exit()
                if self.args.teacher != '':
                    teacher_output = self.teacher(data)

                    distill_loss = 0
                    # print(len(output))

                    ##########################################################################################
                    # output = torch.concat(list(output), dim=1)
                    # teacher_output = torch.concat(list(teacher_output), dim=1)

                    # B, C, H, W = output.shape

                    # distill_loss += 1 -  F.cosine_similarity(output, teacher_output).mean() 

                    ##########################################################################################
                    for j in range(len(output)):
                        B, C, H, W = output[j].shape
                        distill_loss += 1 -  F.cosine_similarity(output[j], teacher_output[j],dim = 2).sum() /(B*C*H*W) 
                        # distill_loss += (1 -  F.cosine_similarity(output[j], teacher_output[j],dim = 2).mean())
                        # distill_loss += F.mse_loss(output[j],teacher_output[j], reduction="sum") / (2*B*C*H*W)
                    distill_loss = distill_loss/len(teacher_output)

                    ##########################################################################################
                        # print(B,C,H,W)
                    #     # flaten the output and teacher_output to [B, C*H*W] [B, C, H, W]
                    #     output_flat = output[j].view(B, -1)
                    #     teacher_output_flat = teacher_output[j].view(B, -1)

                    #     distill_loss += 1 - F.cosine_similarity(output_flat, teacher_output_flat,dim=0).mean() # sum()/B

                    ##########################################################################################



                        # distill_loss += F.mse_loss(output[i].detach(),teacher_output[i].detach(), size_average=True, reduction="sum")/(B*H*W)
                    # print(len(teacher_output))

                    
                    loss_BCE = loss
                    loss = self.args.alpha * loss + (1 - self.args.alpha) * distill_loss
                else:
                    distill_loss = 0.0
                    loss_BCE = loss

            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            self.lr_scheduler.step()

            pruner.step()

            losses.append(loss.item())
            tbar.set_description('Epoch:%3d, lr:%f, Total loss:%f, loss_KD:%f, loss_BCE:%f' 
                                 % (epoch, trainer.optimizer.param_groups[0]['lr'], np.mean(losses), distill_loss,loss_BCE ))
        # adjust_learning_rate(self.optimizer, epoch, args.epochs, args.learning_rate,
        #                      args.warm_up_epochs, 1e-6)

        self.writer.add_scalar('Losses/train loss', np.mean(losses), epoch)
        self.writer.add_scalar('Learning rate/', trainer.optimizer.param_groups[0]['lr'], epoch)

    def validation(self, epoch):

        self.iou_metric.reset()
        self.nIoU_metric.reset()
        self.PD_FA.reset()

        eval_losses = []
        self.model.eval()
        tbar = tqdm(self.val_data_loader)
        for i, (data, labels) in enumerate(tbar):
            with torch.no_grad():
                output = self.model(data.to(self.device))
                output = output.cpu()

            loss = self.criterion(output, labels)
            eval_losses.append(loss.item())

            self.iou_metric.update(output, labels)
            self.nIoU_metric.update(output, labels)
            self.PD_FA.update(output, labels)

            _, IoU = self.iou_metric.get()
            _, nIoU = self.nIoU_metric.get()
            Fa, Pd = self.PD_FA.get(len(self.val_set))

            tbar.set_description('  Epoch:%3d, eval loss:%f, IoU:%f, nIoU:%f, Fa:%.8f, Pd:%.5f'
                                 % (epoch, np.mean(eval_losses), IoU, nIoU, Fa, Pd))

        pkl_name = 'Epoch-%3d_IoU-%.4f_nIoU-%.4f_Fa:%.8f_Pd:%.5f.pth' % (epoch, IoU, nIoU, Fa, Pd)
        save_file = {"model": self.model.state_dict(),
                     "optimizer": self.optimizer.state_dict(),
                     "lr_scheduler": self.lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = self.scaler.state_dict()

        if self.train_set.__class__.__name__ == 'SirstDataset':
            save_pth = self.save_pth
        if self.train_set.__class__.__name__ == 'IRSTD1K_Dataset':
            save_pth = self.save_pth2
        if self.train_set.__class__.__name__ == 'CustomDataset':
            save_pth = self.save_pth3
        if IoU > self.best_iou:
            # save weight file with best IoU into weight folder
            temp_name = args.mode + '_best.pth'
            torch.save(save_file, ops.join("/mnt/data/danhnt/Seg/ILNet-master/weights/", temp_name))
            # print('Save best weight file with IoU: ')

            torch.save(save_file, ops.join(save_pth, pkl_name))
            self.best_iou = IoU
        if nIoU > self.best_nIoU:
            torch.save(save_file, ops.join(save_pth, pkl_name))
            self.best_nIoU = nIoU
        if Pd > self.best_PD:
            torch.save(save_file, ops.join(save_pth, pkl_name))
            self.best_PD = Pd
        if Fa < self.best_FA:
            torch.save(save_file, ops.join(save_pth, pkl_name))
            self.best_FA = Fa     

        self.writer.add_scalar('Losses/eval_loss', np.mean(eval_losses), epoch)
        self.writer.add_scalar('Eval/IoU', IoU, epoch)
        self.writer.add_scalar('Eval/nIoU', nIoU, epoch)
        self.writer.add_scalar('Best/IoU', self.best_iou, epoch)
        self.writer.add_scalar('Best/nIoU', self.best_nIoU, epoch)
        self.writer.add_scalar('Eval/Pd', Pd, epoch)
        self.writer.add_scalar('Eval/Fa', Fa, epoch)
        self.writer.add_scalar('Best/Pd', self.best_PD, epoch)
        self.writer.add_scalar('Best/Fa', self.best_FA, epoch)

  

if __name__ == '__main__':
    args = parse_args()

    trainer = Trainer(args)

    for epoch in range(args.start_epoch, args.epochs + 1):
        if epoch % args.eval_freq == 0:
            trainer.validation(epoch)
        # trainer.validation(epoch)

    # ecport onnx
    dummy_input = torch.randn(2, 3, 512, 640).to(trainer.device)
    torch.onnx.export(trainer.model, dummy_input, "ilnet_S_prunning.onnx",
                opset_version=17, verbose=False, 
                do_constant_folding=True,
                input_names=['input'], output_names=['output']
                )
    
    print('Best IoU: %.5f, best nIoU: %.5f, Best Pd: %.5f, best Fa: %.5f' %
          (trainer.best_iou, trainer.best_nIoU, trainer.best_PD, trainer.best_FA))
