import numpy as np 

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

from configs.submissions import get_cfg
from core.FlowFormer import build_flowformer

import torch.nn as nn
import torch.nn.init as init
import os

def build_model():
    print(f"building  model...")
    cfg = get_cfg()
    cfg.model = "checkpoints/things.pth"
    print("torch.load(cfg.model) : ", torch.load(cfg.model).keys())
    # model = torch.nn.DataParallel(build_flowformer(cfg))
    print("cfg : ", cfg)
    model = nn.DataParallel(build_flowformer(cfg))
    model.load_state_dict(torch.load(cfg.model))
    model.cuda()
    model.eval()

    return model

model_path = "checkpoints/things.pth"
folder_onnx = "onnx_tensorrt"
os.makedirs(folder_onnx, exist_ok = True)
file_name_folder = (model_path.split("/")[-1]).split(".")[0]

batch_size = 1

# Build model
model = build_model()

# Initialize 
map_location = lambda storge, loc : storge

device = "cuda"

use_gpu = False

if use_gpu and torch.cuda.is_available():
    map_location = None
    dtype = torch.cuda.FloatTensor
    device == "cuda"
else:
    dtype = torch.FloatTensor
    device == "cpu"

# if 'model' in model_path:
#     model.load_state_dict(torch.load(model_path, map_location = lambda storage, loc : storage.cpu()))
# else:
#     model.load_state_dict(torch.load(model_path, map_location = lambda storage, loc : storage.cpu())["state_dict"])

model.to(device)

if device == "cuda" and torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

input1_size = [(3,544, 960)]
input2_size = [(3,544, 960)]

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