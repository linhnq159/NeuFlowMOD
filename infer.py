import torch
from glob import glob
import os
import numpy as np
import cv2
from NeuFlow.neuflow import NeuFlow
from NeuFlow.backbone_v7 import ConvBlock
from data_utils import flow_viz


image_width = 1920 ## 960  ## 768
image_height = 1088 ## 544  ## 432 must divible 16
use_fp16 = False

def get_cuda_image(image_path):
    image = cv2.imread(image_path)

    image = cv2.resize(image, (image_width, image_height))

    if use_fp16:
        image = torch.from_numpy(image).permute(2, 0, 1).half()
    else:
        image = torch.from_numpy(image).permute(2, 0, 1).float()
    return image[None].cuda()


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

# folder_input = "/data/linhnq11/Data/U3/Video_co_giam_chan/img/35_8/*.jpg"
folder_input = "/data/linhnq11/Data/U3/Video_co_giam_chan/img/35_9"
# folder_input = "/data/linhnq11/Data/U3/M3OT/M3OT/1/ir/train/1-01T/img1"
name_folder = folder_input.split("/")[-1]

image_path_list = sorted(glob(f'{folder_input}/*'))
# image_path_list = sorted(glob('/data/linhnq11/OpticalFlow/NeuFlow/test_images/*.jpg'))

vis_path = f'test_results/{name_folder}'

device = torch.device('cuda')

model = NeuFlow().to(device)

checkpoint = torch.load('neuflow_things.pth', map_location='cuda')
# checkpoint = torch.load('experiment/sintel/step_012000.pth', map_location = 'cuda')

model.load_state_dict(checkpoint['model'], strict=True)

for m in model.modules():
    if type(m) is ConvBlock:
        m.conv1 = fuse_conv_and_bn(m.conv1, m.norm1)  # update conv
        m.conv2 = fuse_conv_and_bn(m.conv2, m.norm2)  # update conv
        delattr(m, "norm1")  # remove batchnorm
        delattr(m, "norm2")  # remove batchnorm
        m.forward = m.forward_fuse  # update forward

model.eval()
if use_fp16:
    model.half()

model.init_bhwd(1, image_height, image_width, 'cuda', use_fp16)

if not os.path.exists(vis_path):
    os.makedirs(vis_path)

frame_skip = 3

for image_path_0, image_path_1 in zip(image_path_list[:-frame_skip], image_path_list[frame_skip:]):

    print(f"{image_path_0},{image_path_1}")

    image_0 = get_cuda_image(image_path_0)
    image_1 = get_cuda_image(image_path_1)

    file_name = os.path.basename(image_path_0)

    with torch.no_grad():
        # print("len(model(image_0, image_1)) : ", model(image_0, image_1))

        flow = model(image_0, image_1, iters_s8 = 12)[-1][0]

        flow = flow.permute(1,2,0).cpu().numpy()
        
        flow = flow_viz.flow_to_image(flow)

        image_0 = cv2.resize(cv2.imread(image_path_0), (image_width, image_height))

        cv2.imwrite(os.path.join(vis_path, file_name), np.vstack([image_0, flow]))
