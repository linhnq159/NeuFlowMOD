import numpy as np 

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

import torch.nn as nn
import torch.nn.init as init
import os

from NeuFlow.neuflow import NeuFlow
from NeuFlow.backbone_v7 import ConvBlock

use_fp16 = False

file_name_folder = "NeuFlowV2_320"

image_width = 320 ## 960  ## 768
image_height = 320  ## 544 ## 432 must divible 16

class TRTLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)


        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine


        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)


    def forward(self, x):
        dims = tuple(range(x.dim() - len(self.normalized_shape), x.dim()))

        mean = x.mean(dim=dims, keepdim=True)
        diff = x - mean
        var = (diff * diff).mean(dim=dims, keepdim=True)

        # Thay Div(Sqrt()) bằng Mul(Rsqrt())
        x = diff * torch.rsqrt(var + self.eps)

        if self.elementwise_affine:
            x = x * self.weight + self.bias
        return x

def replace_layernorm(module):
    for name, child in module.named_children():

        if isinstance(child, nn.LayerNorm):
            new_layer = TRTLayerNorm(
                normalized_shape=child.normalized_shape,
                eps=child.eps,
                elementwise_affine=child.elementwise_affine
            )


            if child.elementwise_affine:
                with torch.no_grad():
                    new_layer.weight.copy_(child.weight)
                    new_layer.bias.copy_(child.bias)


            setattr(module, name, new_layer)


            print(f"Replace: {name}")

        else:
            replace_layernorm(child)

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

    checkpoint = torch.load('/data/linhnq11/OpticalFlow/NeuFlow/experiment/07072026_resume_1/epoch_020_step_025641.pth', map_location=device)

    model.load_state_dict(checkpoint['model'], strict=True)

    # for m in model.modules():
    #     if type(m) is ConvBlock:
    #         m.conv1 = fuse_conv_and_bn(m.conv1, m.norm1)  # update conv
    #         m.conv2 = fuse_conv_and_bn(m.conv2, m.norm2)  # update conv
    #         delattr(m, "norm1")  # remove batchnorm
    #         delattr(m, "norm2")  # remove batchnorm
    #         m.forward = m.forward_fuse  # update forward

    replace_layernorm(model)

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

model = build_model(device)


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
                            (input1, input2, 1, 10),
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