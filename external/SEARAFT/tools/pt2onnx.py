import numpy as np 

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

import torch.nn as nn
import torch.nn.init as init
import os

from config.parser import parse_args
import argparse
from core.raft import RAFT
from core.utils.flow_viz import flow_to_image
from core.utils.utils import load_ckpt
from core.utils import frame_utils

use_fp16 = False

file_name_folder = "SEARAFT_960_544"

image_width = 960 ## 960  ## 768
image_height = 544  ## 544 ## 432 must divible 16

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def fuse_conv_and_bn(conv, bn):
        """Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/."""
        fusedconv = (
            torch.nn.Conv2d(
                conv.in_channels,
                conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                dilation=conv.dilation,
                groups=conv.groups,
                bias=True,
            )
            .requires_grad_(False)
            .to(conv.weight.device)
        )

        # Prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

        # Prepare spatial bias
        b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

        return fusedconv

def build_model(device = "cpu"):
    # print(f"building  model...")
    # cfg = get_cfg()
    # cfg.model = "/data/linhnq11/OpticalFlow/FlowFormerPlusPlus-main/checkpoints/things.pth"
    # print("torch.load(cfg.model) : ", torch.load(cfg.model).keys())
    # # model = torch.nn.DataParallel(build_flowformer(cfg))
    # print("cfg : ", cfg)
    # model = nn.DataParallel(build_flowformer(cfg))
    # model.load_state_dict(torch.load(cfg.model))
    # model.cuda()
    # model.eval()

    device = torch.device(device)

    model = NeuFlow().to(device)

    checkpoint = torch.load('neuflow_things.pth', map_location=device)

    model.load_state_dict(checkpoint['model'], strict=True)

    for m in model.modules():
        if type(m) is ConvBlock:
            m.conv1 = fuse_conv_and_bn(m.conv1, m.norm1)  # update conv
            m.conv2 = fuse_conv_and_bn(m.conv2, m.norm2)  # update conv
            delattr(m, "norm1")  # remove batchnorm
            delattr(m, "norm2")  # remove batchnorm
            m.forward = m.forward_fuse  # update forward

    model.eval()

    model.init_bhwd(1, image_height, image_width, device, use_fp16)

    return model

# Build model

device = "cpu"

use_gpu = False

if use_gpu and torch.cuda.is_available():
    map_location = None
    dtype = torch.cuda.FloatTensor
    device == "cuda"
else:
    dtype = torch.FloatTensor
    device == "cpu"

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
parser.add_argument('--model', help='checkpoint path', type=str, required=True)
parser.add_argument('--output', help='path to save output', type=str, default='demo_own')
parser.add_argument('--iters', help='number of iterations', default=8, type=int)
parser.add_argument('--scale', help='input scale', type=int, default=0)
args = parse_args(parser)

model = RAFT(args)
load_ckpt(model, args.model)
model = model.to(DEVICE)
model.eval()

# if 'model' in model_path:
#     model.load_state_dict(torch.load(model_path, map_location = lambda storage, loc : storage.cpu()))
# else:
#     model.load_state_dict(torch.load(model_path, map_location = lambda storage, loc : storage.cpu())["state_dict"])

print("device : ", device)

model.to(device)

if device == "cuda" and torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

input1_size = [(3,image_height, image_width)]
input2_size = [(3,image_height, image_width)]

input1 = [torch.rand(*input1_size).type(dtype) for in_size_input1 in input1_size]
input2 = [torch.rand(*input2_size).type(dtype) for in_size_input2 in input2_size]

input1 = torch.stack(input1).to(device)
input2 = torch.stack(input2).to(device)

torch_out = torch.onnx.export(model.module if hasattr(model, "module") else model,
                            (input1, input2),
                            f"onnx_tensorrt/{file_name_folder}.onnx",
                            input_names = ["image1", "image2"],
                            output_names = ["flow_final"],
                            verbose = True,
                            opset_version = 16,
                            export_params = True)

import onnx
import onnxsim
onnx_model = onnx.load(f"onnx_tensorrt/{file_name_folder}.onnx")
onnx.checker.check_model(onnx_model)
onnx_model, check = onnxsim.simplify(onnx_model)
onnx.save(onnx_model, f"onnx_tensorrt/{file_name_folder}_ssim.onnx")