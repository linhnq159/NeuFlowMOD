import torch.onnx
from model.ilnet import ILNet_S, ILNet_M, ILNet_L
import argparse
from model.ilnet import RSU,IPOF
from model.ilnet_V1 import ILNetV1_S
from model.ilnet_Batch import ILNet_S_Batch, ILNet_M_Batch, ILNet_L_Batch
from model.LW_IRSTNet import LW_IRST_ablation
from model.LW_IRSTNet_V1 import LW_net
from model.LW_Opticalflow import LW_OpticalFLow
from model.LW_KAN import LW_net_Kan
import os
# export model to onnx

parser = argparse.ArgumentParser(description='ILNet')
parser.add_argument('--model', type=str, default='LW_V1', help='S, M, L')
parser.add_argument('--weight', type=str, default='ILNet_S.pth', help='model weights')
parser.add_argument('--output', type=str, default='LW_Net_V1.onnx', help='output onnx model')
parser.add_argument('--work_dir', type=str, default='./work_seg/onnx/', help='save weights folder')
parser.add_argument('--opset', type=int, default=12, help='if model contained Layernorm, the opset must be 12')
args = parser.parse_args()

if args.model == 'S':
    model = ILNet_S()
    # model = ILNet_S_Batch()
elif args.model == 'M':
    model = ILNet_M()
    # model = ILNet_M_Batch()
elif args.model == 'L':
    model = ILNet_L()
    # model = ILNet_L_Batch()
elif args.model =="SV1" :
    model = ILNetV1_S()
elif args.model == "A":
    model = ILNet_S_Batch()
elif args.model == "LW_Opticalflow" :
    model = LW_OpticalFLow() 
elif args.model == "LW_V1":
    model = LW_net()
elif args.model == "LW":
    model = LW_IRST_ablation(channel=(8, 32, 64), dilations=(2,4,8,16), kernel_size=(7,7,7,7), padding=(3,3,3,3))
elif args.model == "KAN":
    model = LW_net_Kan()


if os.path.isfile(args.weight):
    checkpoint = torch.load(args.weight)
    model.load_state_dict(checkpoint['model'])
else:
    print("Cannot load weight from checkpoints!")
    print("Weights of model will be randomize init")
model.eval()




print('Exporting model to onnx...')

#dummy_input = torch.randn(1, 3, 512, 640) # (batch_size, channels, height, width)
dummy_input = torch.randn(1, 2, 544, 960)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)
torch.onnx.export(model, dummy_input, args.work_dir + args.output, 
                opset_version=12, verbose=False,  # ILNet must use oppset_version=12
                do_constant_folding=True,
                input_names=['input'], output_names=['output']
                )

print('Done!')

# simplyfied version onnx export

import torch
import onnx
import onnxsim

onnx_model = onnx.load(args.work_dir +args.output)
model_simp, check = onnxsim.simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp,args.work_dir +  args.output.replace('.onnx', '_sim.onnx'))
print('Done!')
