import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiScaleCNNRepBlock(nn.Module):
        def __init__(
            self,
            channels,
            kernel_size=3,
            padding=1,
            dilation=1,
            median_kernel=3,
            bias=False,
            relu=True,
            dropout_prob=0.0,
            k=1,
            groups=1,
        ):
            # print(kernel_size // median_kernel)

            super().__init__()
            if relu:
                activation = nn.ReLU
            else:
                activation = nn.PReLU

            self.deploy = False
            self.in_channels = channels
            self.kernel_size = kernel_size
            self.median_kernel = median_kernel
            self.groups = groups

            self.k = k

            self.conv_1x1_list = nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=bias,
                    groups=groups,
                    dilation=dilation,
                ),
                nn.BatchNorm2d(channels)
            )
            self.conv_1xC_list = []
            self.conv_Cx1_list = []
            self.conv_CxC_list = []

            if kernel_size - median_kernel < 2:
                print("kernel_size %.0f should be larger than median_kernel %.0f to use intermediate branch" % (kernel_size, median_kernel))
            else:
                self.conv_mxm_list = []
                self.conv_1xm_list = []
                self.conv_mx1_list = []

            for i in range(k):
                self.conv_1xC_list.append(nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=(1, kernel_size),
                        stride=1,
                        padding=(0, padding),
                        dilation=dilation,
                        groups=groups,
                        bias=bias,
                    ),
                    nn.BatchNorm2d(channels),
                ))

                self.conv_Cx1_list.append(nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=(kernel_size, 1),
                        stride=1,
                        padding=(padding, 0),
                        dilation=dilation,
                        groups=groups,
                        bias=bias,
                    ),
                    nn.BatchNorm2d(channels),
                ))

                self.conv_CxC_list.append(nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding,
                        dilation=dilation,
                        groups=groups,
                        bias=bias,
                    ),
                    nn.BatchNorm2d(channels),
                ))

                if kernel_size - median_kernel >= 2:
                    new_padding = dilation * (median_kernel - 1) // 2
                    self.conv_mxm_list.append(nn.Sequential(
                        nn.Conv2d(
                            channels,
                            channels,
                            kernel_size=median_kernel,
                            stride=1,
                            padding=new_padding,
                            dilation=dilation,
                            groups=groups,
                            bias=bias,
                        ),
                        nn.BatchNorm2d(channels),
                    ))

                    self.conv_1xm_list.append(nn.Sequential(
                        nn.Conv2d(
                            channels,
                            channels,
                            kernel_size=(1, median_kernel),
                            stride=1,
                            padding=(0, new_padding),
                            dilation=dilation,
                            groups=groups,
                            bias=bias,
                        ),
                        nn.BatchNorm2d(channels),
                    ))

                    self.conv_mx1_list.append(nn.Sequential(
                        nn.Conv2d(
                            channels,
                            channels,
                            kernel_size=(median_kernel, 1),
                            stride=1,
                            padding=(new_padding, 0),
                            dilation=dilation,
                            groups=groups,
                            bias=bias,
                        ),
                        nn.BatchNorm2d(channels),
                    ))



            self.prev_act = activation()

            self.out_activation = activation()

            self.conv_1xC_list = nn.ModuleList(self.conv_1xC_list)
            self.conv_Cx1_list = nn.ModuleList(self.conv_Cx1_list)
            self.conv_CxC_list = nn.ModuleList(self.conv_CxC_list)
            if kernel_size - median_kernel >= 2:
                self.conv_mxm_list = nn.ModuleList(self.conv_mxm_list)
                self.conv_1xm_list = nn.ModuleList(self.conv_1xm_list)
                self.conv_mx1_list = nn.ModuleList(self.conv_mx1_list)

        def forward(self, x):
            main = x
            if not self.deploy:
                ext = self.conv_1x1_list(x)
                for i in range(self.k):
                    ext1 = self.conv_1xC_list[i](x)
                    ext2 = self.conv_Cx1_list[i](x)
                    ext3 = self.conv_CxC_list[i](x)
                    # print(ext1.size(), ext2.size(), ext3.size())
                    if self.kernel_size - self.median_kernel >= 2:
                        ext5 = self.conv_mxm_list[i](x)
                        ext6 = self.conv_1xm_list[i](x)
                        ext7 = self.conv_mx1_list[i](x)
                        # print(ext5.size(), ext6.size(), ext7.size())
                        ext = ext +  ext1 + ext2 + ext3  + ext5 + ext6 + ext7
                    
                    else:
                        ext = ext +  ext1 + ext2 + ext3
                
            else:
                ext = self.conv_CxC_deploy(x)

            ext = self.prev_act(ext)
            out = main + ext
            out = self.out_activation(out)
            return out
        
        def get_equivalent_kernel_bias(self):
            
            kernel, bias = self._fuse_bn_tensor(self.conv_1x1_list)
            kernel = self._pad_1x1_to_CxC_tensor(kernel, self.kernel_size)

            for i in range(self.k):
                kernelCxC, biasCxC = self._fuse_bn_tensor(self.conv_CxC_list[i])
                kernel = kernel + kernelCxC
                bias = bias + biasCxC
                
                kernelCx1, biasCx1 = self._fuse_bn_tensor(self.conv_Cx1_list[i])
                kernelCx1 = self._pad_Cx1_to_CxC_tensor(kernelCx1, self.kernel_size)
                kernel = kernel + kernelCx1
                bias = bias + biasCx1

                kernel1xC, bias1xC = self._fuse_bn_tensor(self.conv_1xC_list[i])
                kernel1xC = self._pad_1xC_to_CxC_tensor(kernel1xC, self.kernel_size)
                kernel = kernel + kernel1xC
                bias = bias + bias1xC


                if self.kernel_size - self.median_kernel >= 2:
                        
                        kernel_mxm, bias_mxm = self._fuse_bn_tensor(self.conv_mxm_list[i])
                        kernel_mxm = self._pad_mxm_to_CxC_tensor(kernel_mxm, self.kernel_size)
                        kernel += kernel_mxm
                        bias += bias_mxm

                        kernel_1xm, bias_1xm = self._fuse_bn_tensor(self.conv_1xm_list[i])
                        kernel_1xm = self._pad_1xm_to_CxC_tensor(kernel_1xm, self.kernel_size)
                        kernel += kernel_1xm
                        bias += bias_1xm

                        kernel_mx1, bias_mx1 = self._fuse_bn_tensor(self.conv_mx1_list[i])
                        kernel_mx1 = self._pad_mx1_to_CxC_tensor(kernel_mx1, self.kernel_size)
                        kernel += kernel_mx1
                        bias += bias_mx1

                    

            return kernel, bias
        
        def _pad_1x1_to_CxC_tensor(self, kernel1x1,target_size):
            """
            Pad a 1x1 kernel to the specified target size CxC.

            Args:
            - kernel (torch.Tensor): The 1x1 kernel tensor of shape (1, 1).
            - target_size (int): The desired target size of the kernel.

            Returns:
            - torch.Tensor: The padded kernel of shape (target_size, target_size).
            """
            if kernel1x1 is None:
                return 0
            else:
                pad_amount = (target_size - 1) // 2
                return F.pad(kernel1x1, [pad_amount, pad_amount, pad_amount, pad_amount])
            
        def _pad_1xC_to_CxC_tensor(self, kernel1xC,target_size):
            """
            Pad a 1xC kernel to the specified target size CxC.

            Args:
            - kernel (torch.Tensor): The 1xC kernel tensor of shape (1, C).
            - target_size (int): The desired target size of the kernel.

            Returns:
            - torch.Tensor: The padded kernel of shape (target_size, C).
            """
            if kernel1xC is None:
                return 0
            else:
                pad_amount = (target_size - 1) // 2
                return F.pad(kernel1xC, [0, 0, pad_amount, pad_amount])
            
        def _pad_Cx1_to_CxC_tensor(self, kernelCx1,target_size):
            """
            Pad a Cx1 kernel to the specified target size CxC.

            Args:
            - kernel (torch.Tensor): The Cx1 kernel tensor of shape (C, 1).
            - target_size (int): The desired target size of the kernel.

            Returns:
            - torch.Tensor: The padded kernel of shape (C, target_size).
            """
            if kernelCx1 is None:
                return 0
            else:
                pad_amount = (target_size - 1) // 2
                return F.pad(kernelCx1, [pad_amount, pad_amount, 0, 0])
            
        def _pad_mxm_to_CxC_tensor(self, kernel_mxm,target_size):
            """
            Pad a mxm kernel to the specified target size CxC.

            Args:
            - kernel (torch.Tensor): The mxm kernel tensor of shape (mxm).
            - target_size (int): The desired target size of the kernel.

            Returns:
            - torch.Tensor: The padded kernel of shape (target_size, target_size).
            """
            if kernel_mxm is None:
                return 0
            else:
                pad_amount = (target_size - self.median_kernel) // 2
                return F.pad(kernel_mxm, [pad_amount, pad_amount, pad_amount, pad_amount])
            
        def _pad_1xm_to_CxC_tensor(self, kernel_1xm,target_size):
            """
            Pad a 1xm kernel to the specified target size CxC.

            Args:
            - kernel (torch.Tensor): The 1xm kernel tensor of shape (1, mxm).
            - target_size (int): The desired target size of the kernel.

            Returns:
            - torch.Tensor: The padded kernel of shape (target_size, target_size).
            """
            if kernel_1xm is None:
                return 0
            else:
                pad_left_right = (target_size - self.median_kernel) // 2
                pad_bottom_top = (target_size - 1) // 2
                return F.pad(kernel_1xm, [pad_left_right, pad_left_right, pad_bottom_top, pad_bottom_top])
            
        def _pad_mx1_to_CxC_tensor(self, kernel_mx1,target_size):
            """
            Pad a mx1 kernel to the specified target size CxC.

            Args:
            - kernel (torch.Tensor): The mx1 kernel tensor of shape (mxm, 1).
            - target_size (int): The desired target size of the kernel.

            Returns:
            - torch.Tensor: The padded kernel of shape (target_size, target_size).
            """
            if kernel_mx1 is None:
                return 0
            else:
                pad_left_right = (target_size - 1) // 2
                pad_bottom_top = (target_size - self.median_kernel) // 2
                return F.pad(kernel_mx1, [pad_left_right, pad_left_right, pad_bottom_top, pad_bottom_top])
            
            
        def _fuse_bn_tensor(self, branch):
            if branch is None:
                return 0, 0
            if isinstance(branch, nn.Sequential):
                kernel = branch[0].weight
                running_mean = branch[1].running_mean
                running_var = branch[1].running_var
                gamma = branch[1].weight
                beta = branch[1].bias
                eps = branch[1].eps
            else:
                assert isinstance(branch, nn.BatchNorm2d)
                if not hasattr(self, "id_tensor"):
                    print("id_tensor")
                    input_dim = self.in_channels // self.groups
                    kernel_value = np.zeros(
                        (self.in_channels, input_dim, 3, 3)
                    )
                    for i in range(self.in_channels):
                        kernel_value[i, i % input_dim, 1, 1] = 1
                    self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
                kernel = self.id_tensor
                running_mean = branch.running_mean
                running_var = branch.running_var
                gamma = branch.weight
                beta = branch.bias
                eps = branch.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return kernel * t, beta - running_mean * gamma / std
        
        def switch_to_deploy(self):
            kernel, bias = self.get_equivalent_kernel_bias()

            # get in_channels, out_channels, kernel_size, stride, padding, bias of conv_CxC

            for module in self.conv_CxC_list[0]:
                if isinstance(module, nn.Conv2d):
                    in_channels = module.in_channels
                    out_channels = module.out_channels
                    kernel_size = module.kernel_size
                    stride = module.stride
                    padding = module.padding
                    groups = module.groups
                    dilation = module.dilation
                    break
            self.conv_CxC_deploy = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=True, groups=groups, dilation=dilation
            )
            # print(self.conv_CxC_deploy)
            self.conv_CxC_deploy.weight.data = kernel
            self.conv_CxC_deploy.bias.data = bias
            self.__delattr__("conv_Cx1_list")
            self.__delattr__("conv_1xC_list")
            self.__delattr__("conv_1x1_list")
            if self.kernel_size - self.median_kernel >= 2:
                self.__delattr__("conv_mxm_list")
                self.__delattr__("conv_1xm_list")
                self.__delattr__("conv_mx1_list")
            self.deploy = True

            


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, relu=True):
        super().__init__()
        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        inter_channels = (out_channels + in_channels) // 4
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.BatchNorm2d(inter_channels),
            activation(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, kernel_size=3, stride=2, padding=1, bias=bias),
            nn.BatchNorm2d(inter_channels),
            activation(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(inter_channels, out_channels - in_channels, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.BatchNorm2d(out_channels - in_channels),
            activation(),
        )

        

        self.ext = nn.MaxPool2d(2)

    def forward(self, x):
        
        ext = self.ext(x)
        main = self.conv1(x)
        main = self.conv2(main)
        main = self.conv3(main)
        return torch.cat([main, ext], 1)
    
class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False, relu=True):
        super().__init__()
        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels*2, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias),
            # nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            activation(),
        )


    def forward(self, x1,x2):
        x = torch.cat([x1, x2], 1)
        # x = x1 + x2
        return self.main(x)
    
class Dw_MultiscaleBottleNeck(nn.Module):
    def __init__(self, in_channels, kernel_size=3, bias=False, relu=True, median_kernel=3, dropout_prob=0.0):
        super().__init__()
        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        inter_channels = in_channels // 4
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.BatchNorm2d(inter_channels),
            activation(),
        )

        self.MutiScaleBlock = MultiScaleCNNRepBlock(inter_channels, kernel_size=kernel_size, padding=padding, bias=bias, relu=relu,groups= inter_channels, median_kernel=median_kernel)

        self.conv3 = nn.Sequential(
            nn.Conv2d(inter_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.BatchNorm2d(in_channels),
            activation(),
        )

        self.dropout = nn.Dropout(dropout_prob)

        self.out_act = activation()

    def forward(self, x):
        ext = x
        main = self.conv1(x)
        main = self.MutiScaleBlock(main)
        main = self.conv3(main)
        
        main = self.dropout(main)
        return self.out_act(main + ext)
    
class Reg_MultiscaleBottleNeck(nn.Module):
    def __init__(self, in_channels, kernel_size=3, bias=False, relu=True, median_kernel=3, dropout_prob=0.0):
        super().__init__()
        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        inter_channels = in_channels // 4
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.BatchNorm2d(inter_channels),
            activation(),
        )

        self.MutiScaleBlock = MultiScaleCNNRepBlock(inter_channels, kernel_size=kernel_size, padding=padding, bias=bias, relu=relu, median_kernel=median_kernel)

        self.conv3 = nn.Sequential(
            nn.Conv2d(inter_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.BatchNorm2d(in_channels),
            activation(),
        )

        self.dropout = nn.Dropout(dropout_prob)

        self.out_act = activation()

    def forward(self, x):
        ext = x
        main = self.conv1(x)
        main = self.MutiScaleBlock(main)
        main = self.conv3(main)
        
        main = self.dropout(main)
        return self.out_act(main + ext)
    
class Dil_MultiScaleBottleNeck(nn.Module):
    def __init__(self, in_channels, kernel_size=3, bias=False, relu=True, median_kernel=3, dropout_prob=0.0 ,dilation=1):
        super().__init__()
        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        inter_channels = in_channels // 4
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.BatchNorm2d(inter_channels),
            activation(),
        )

        padding = (dilation * (kernel_size - 1)) // 2
        self.MutiScaleBlock = MultiScaleCNNRepBlock(inter_channels, kernel_size=kernel_size, padding=padding, bias=bias, relu=relu, median_kernel=median_kernel, dilation=dilation)

        self.conv3 = nn.Sequential(
            nn.Conv2d(inter_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.BatchNorm2d(in_channels),
            activation(),
        )

        self.dropout = nn.Dropout(dropout_prob)

        self.out_act = activation()

    def forward(self, x):
        ext = x
        main = self.conv1(x)
        main = self.MutiScaleBlock(main)
        main = self.conv3(main)
        
        main = self.dropout(main)
        return self.out_act(main + ext)
    
class init_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False, relu=True):
        super().__init__()
        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            activation(),
        )

    def forward(self, x):
        return self.main(x)
    
class InitBlock(nn.Module):
    def __init__(self, in_dim, out_dim, relu=True, bias=False):
        super(InitBlock, self).__init__()
        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        self.main_branch = nn.Conv2d(
            in_dim, out_dim - 3, kernel_size=3, stride=2, padding=1, bias=bias
        )
        self.ext_branch = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.bn = nn.BatchNorm2d(out_dim)
        self.activation = activation()

    def forward(self, x):
        main = self.main_branch(x)
        ext = self.ext_branch(x)
        out = torch.cat([main, ext], 1)
        out = self.bn(out)
        out = self.activation(out)
        return out
    
class CommonBlock(nn.Module):
    def __init__(self, channel = 3, kernel_size=3, bias=False, relu=True, dilation=[2,4], median_kernel=3):
        super().__init__()
        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU
        self.Dw = Dw_MultiscaleBottleNeck(channel, kernel_size=kernel_size, bias=bias, relu=relu, median_kernel=median_kernel, dropout_prob=0.1)
        self.Dil = Dil_MultiScaleBottleNeck(channel, kernel_size=kernel_size, bias=bias, relu=relu, median_kernel=median_kernel, dropout_prob=0.1, dilation=dilation[0])
        self.Reg = Reg_MultiscaleBottleNeck(channel, kernel_size=kernel_size, bias=bias, relu=relu, median_kernel=median_kernel, dropout_prob=0.1)
        self.Dil_1 = Dil_MultiScaleBottleNeck(channel, kernel_size=kernel_size, bias=bias, relu=relu, median_kernel=median_kernel, dropout_prob=0.1, dilation=dilation[1])

    def forward(self, x):
        x = self.Dw(x)
        x = self.Dil(x)
        x = self.Reg(x)
        x = self.Dil_1(x)
        return x
    

    
class Net(nn.Module):
    def __init__(self, channels = [8,16,32], dilation = [2,4,6,8], kernel_size = [7,5,3], bias=False,  median_kernel=[5,3,3], padding=[3,2,1], endcoder_relu = False, decoder_relu = True, num_classes=1):
        super().__init__()
        self.init_block = init_block(3, channels[0], kernel_size=kernel_size[0], padding=padding[0], bias=bias, relu= endcoder_relu)
        # encoder block1
        self.Reg_1 = Reg_MultiscaleBottleNeck(channels[0], kernel_size=kernel_size[0], bias=bias, relu=endcoder_relu, median_kernel=median_kernel[0], dropout_prob=0.01)
        self.Down_1 = DownSample(channels[0], channels[1], bias=bias, relu=endcoder_relu)

        # encoder block2
        self.Reg_2 = Reg_MultiscaleBottleNeck(channels[1], kernel_size=kernel_size[1], bias=bias, relu=endcoder_relu, median_kernel=median_kernel[1], dropout_prob=0.01)
        self.Dw_2 = Dw_MultiscaleBottleNeck(channels[1], kernel_size=kernel_size[1], bias=bias, relu=endcoder_relu, median_kernel=median_kernel[1], dropout_prob=0.01)
        self.Down_2 = DownSample(channels[1], channels[2], bias=bias, relu=endcoder_relu)

        # encoder block3
        self.Reg_3 = Reg_MultiscaleBottleNeck(channels[2], kernel_size=kernel_size[2], bias=bias, relu=endcoder_relu, median_kernel=median_kernel[2], dropout_prob=0.01)
        self.Dw_3 = Dw_MultiscaleBottleNeck(channels[2], kernel_size=kernel_size[2], bias=bias, relu=endcoder_relu, median_kernel=median_kernel[2], dropout_prob=0.01)
        self.Dil_3 = Dil_MultiScaleBottleNeck(channels[2], kernel_size=kernel_size[2], bias=bias, relu=endcoder_relu, median_kernel=median_kernel[2], dropout_prob=0.01, dilation=dilation[0])
        # self.Down_3 = DownSample(channels[2], channels[2], bias=bias, relu=endcoder_relu)

        # common block
        self.com_1 = CommonBlock(channel=channels[2], kernel_size=kernel_size[0], bias=bias, relu=endcoder_relu, dilation=[dilation[0],dilation[1]], median_kernel=median_kernel[0])
        self.com_2 = CommonBlock(channel=channels[2], kernel_size=kernel_size[1], bias=bias, relu=endcoder_relu, dilation=[dilation[1],dilation[2]], median_kernel=median_kernel[1])
        self.com_3 = CommonBlock(channel=channels[2], kernel_size=kernel_size[2], bias=bias, relu=endcoder_relu, dilation=[dilation[2],dilation[3]], median_kernel=median_kernel[2])

        # decoder block3
        self.up_4 = UpSample(channels[2], channels[1], bias=bias, relu=decoder_relu)
        self.Reg_4 = Reg_MultiscaleBottleNeck(channels[1], kernel_size=kernel_size[2], bias=bias, relu=decoder_relu, median_kernel=median_kernel[2], dropout_prob=0.01)
        self.Dw_4 = Dw_MultiscaleBottleNeck(channels[1], kernel_size=kernel_size[2], bias=bias, relu=decoder_relu, median_kernel=median_kernel[2], dropout_prob=0.01)
        self.Dil_4 = Dil_MultiScaleBottleNeck(channels[1], kernel_size=kernel_size[2], bias=bias, relu=decoder_relu, median_kernel=median_kernel[2], dropout_prob=0.01, dilation=dilation[2])
        self.conv_4 = nn.Conv2d(channels[1], num_classes, kernel_size=1, stride=1, padding=0, bias=bias)

        # decoder block2
        self.up_5 = UpSample(channels[1], channels[0], bias=bias, relu=decoder_relu)
        self.Reg_5 = Reg_MultiscaleBottleNeck(channels[0], kernel_size=kernel_size[1], bias=bias, relu=decoder_relu, median_kernel=median_kernel[1], dropout_prob=0.01)
        self.Dw_5 = Dw_MultiscaleBottleNeck(channels[0], kernel_size=kernel_size[1], bias=bias, relu=decoder_relu, median_kernel=median_kernel[1], dropout_prob=0.01)
        self.conv_5 = nn.Conv2d(channels[0], num_classes, kernel_size=1, stride=1, padding=0, bias=bias)

        # decoder block1
        self.up_6 = UpSample(channels[0], channels[0]//2, bias=bias, relu=decoder_relu)
        self.Reg_6 = Reg_MultiscaleBottleNeck(channels[0]//2, kernel_size=kernel_size[0], bias=bias, relu=decoder_relu, median_kernel=median_kernel[0], dropout_prob=0.01)
        self.conv_6 = nn.Conv2d(channels[0]//2, num_classes, kernel_size=1, stride=1, padding=0, bias=bias)

        # representation block
        self.out_conv = nn.Conv2d(num_classes*3, num_classes, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn = nn.BatchNorm2d(num_classes)

    def forward(self, x):
        decoder_output = []
        B, C, H, W = x.size()

        # init block
        x1 = self.init_block(x)
        x1 = self.Reg_1(x1)

        # encoder block1
        x2 = self.Down_1(x1)
        x2 = self.Reg_2(x2)
        x2 = self.Dw_2(x2)

        x3 = self.Down_2(x2)
        x3 = self.Reg_3(x3)
        x3 = self.Dw_3(x3)
        x3 = self.Dil_3(x3)


        # common block
        x4 = self.com_1(x3)
        x4 = self.com_2(x3)
        x4 = self.com_3(x3)

        # decoder block3
        x5 = self.up_4(x4, x3)
        x5 = self.Reg_4(x5)
        x5 = self.Dw_4(x5)
        x5 = self.Dil_4(x5)
        decoder_output.append(F.interpolate(self.conv_4(x5), (H, W), mode='bilinear', align_corners=True))

        # decoder block2
        x6 = self.up_5(x5, x2)
        x6 = self.Reg_5(x6)
        x6 = self.Dw_5(x6)
        decoder_output.append(F.interpolate(self.conv_5(x6), (H, W), mode='bilinear', align_corners=True))

        # decoder block1
        x7 = self.up_6(x6, x1)
        x7 = self.Reg_6(x7)
        # decoder_output.append(F.interpolate(self.conv_6(x7), (H, W), mode='bilinear', align_corners=True))
        decoder_output.append(self.conv_6(x7))

        # representation block
        out = torch.cat(decoder_output, dim=1)
        out = self.out_conv(out)

        if self.training:
            return [out] + decoder_output
        else:
            return self.bn(out) 


def Our_S():
    print("dev")
    return Net(channels=[8,16,32])

def Our_M():
    return Net(channels=[16,32,64])

def Our_L():
    print("dev mode is running")
    return Net(channels=[32,64,128])
    
# test module 
if __name__ == '__main__':
    model = Our_M(1)
    # switch to deploy
    dummy_input = torch.randn(1, 3, 512, 512)
    model.eval()
    
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            nn.init.uniform_(module.running_mean, 0, 0.1)
            nn.init.uniform_(module.running_var, 0, 0.1)
            nn.init.uniform_(module.weight, 0, 0.1)
            nn.init.uniform_(module.bias, 0, 0.1)    
    
    from thop import profile, clever_format
    import time
    
    FLOPs, params = profile(model, inputs=(dummy_input, ))
    FLOPs, params = clever_format([FLOPs, params], "%.3f")
    start = time.perf_counter()
    output = model(dummy_input)
    end = time.perf_counter()
    print("FLOPs: ", FLOPs)
    print("params: ", params)
    print("Time: ", end - start)
    print("=================================")
    
    for m in model.modules():
        if isinstance(m, MultiScaleCNNRepBlock):
            
            m.switch_to_deploy()

    
    FLOPs, params = profile(model, inputs=(dummy_input, ))
    FLOPs, params = clever_format([FLOPs, params], "%.3f")
    start = time.perf_counter()
    output1 = model(dummy_input)
    end = time.perf_counter()
    print("Diff:")
    print(((output - output1)**2).sum())

    torch.onnx.export(model, (dummy_input,), 'test.onnx',
                opset_version=12, verbose=False,
                do_constant_folding=True,
                input_names=['input'], output_names=['output']
                )
    import torch
    import onnx
    import onnxsim
    
    onnx_model = onnx.load('test.onnx')
    model_simp, check = onnxsim.simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp,'Ours_new_M.onnx')
    print('Done!')
    print("FLOPs: ", FLOPs)
    print("params: ", params)
    print("Time: ", end - start)
    print("=================================")
