import torch.nn as nn
import torch
# from torchviz import make_dot,make_dot_from_trace
import torch.nn.functional as F


class dw_conv(nn.Module):
    def __init__(self, in_dim, out_dim, relu=True):
        super(dw_conv, self).__init__()
        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU
        self.dw_conv_k3 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, groups=in_dim, bias=False),
            nn.BatchNorm2d(out_dim),
            activation())
    def forward(self, x):
        x = self.dw_conv_k3(x)
        return x

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Initial Downsampling
class InitialBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=False,
                 relu=True,
                 pool_channels=None):
        super().__init__()

        self.pool_channels = in_channels if pool_channels is None else pool_channels
        if not 0 < self.pool_channels <= in_channels or self.pool_channels >= out_channels:
            raise ValueError('Require 0 < pool_channels <= in_channels and pool_channels < out_channels.')
        self.pool_flow_only = self.pool_channels != in_channels

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        self.main_branch = nn.Conv2d(
            in_channels,
            out_channels - self.pool_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=bias)

        # Extension branch
        self.ext_branch = nn.MaxPool2d(3, stride=2, padding=1)

        print("out_channels : ", out_channels)
        # Initialize batch normalization to be used after concatenation
        self.batch_norm = nn.BatchNorm2d(out_channels)

        # PReLU layer to apply after concatenating the branches
        self.out_activation = activation()
        self.se = SEBlock(out_channels)

    def forward(self, x): # Flow, optionally followed by current-frame RGB.
        main = self.main_branch(x) # 1 6 H/2 W/2

        ext = self.ext_branch(x[:, :self.pool_channels] if self.pool_flow_only else x)

        # Concatenate branches
        out = torch.cat((main, ext), 1) # 1 8 H/2 W/2

        # Apply batch normalization
        out = self.batch_norm(out)

        out = self.out_activation(out)

        # Add SEBlock
        out = self.se(out)
        return out

# Regular/asymmetric/depthwise/dilated conv.
class RegularBottleneck(nn.Module):
    def __init__(self,
                 channels,
                 internal_ratio=4,
                 kernel_size=3,
                 padding=0,
                 dilation=1,
                 asymmetric=False,
                 depthwise=False,
                 dilated=False,
                 regular=False,
                 dropout_prob=0.0,
                 bias=False,
                 relu=True):
        super().__init__()

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        if asymmetric:
            internal_channels = channels // internal_ratio
            # 1x1 projection convolution
            self.ext_conv1 = nn.Sequential(
                nn.Conv2d(
                    channels,
                    internal_channels,
                    kernel_size=1,
                    stride=1,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())

            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(kernel_size, 1),
                    stride=1,
                    padding=(padding, 0),
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation(),
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(1, kernel_size),
                    stride=1,
                    padding=(0, padding),
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())
            # 1x1 expansion convolution
            self.ext_conv3 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    channels,
                    kernel_size=1,
                    stride=1,
                    bias=bias), nn.BatchNorm2d(channels), activation())

        elif depthwise:
            internal_channels = channels * 2
            # 1x1 projection convolution
            self.ext_conv1 = nn.Sequential(
                nn.Conv2d(
                    channels,
                    internal_channels,
                    kernel_size=1,
                    stride=1,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())

            self.ext_conv2 = dw_conv(internal_channels,internal_channels) #深度可分离卷积

            # 1x1 expansion convolution
            self.ext_conv3 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    channels,
                    kernel_size=1,
                    stride=1,
                    bias=bias), nn.BatchNorm2d(channels), activation())

        elif dilated:
            internal_channels = channels // internal_ratio
            # 1x1 projection convolution
            self.ext_conv1 = nn.Sequential(
                nn.Conv2d(
                    channels,
                    internal_channels,
                    kernel_size=1,
                    stride=1,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())

            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())

            # 1x1 expansion convolution
            self.ext_conv3 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    channels,
                    kernel_size=1,
                    stride=1,
                    bias=bias), nn.BatchNorm2d(channels), activation())

        elif regular:
            internal_channels = channels // internal_ratio
            # 1x1 projection convolution
            self.ext_conv1 = nn.Sequential(
                nn.Conv2d(
                    channels,
                    internal_channels,
                    kernel_size=1,
                    stride=1,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())

            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())

            # 1x1 expansion convolution
            self.ext_conv3 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    channels,
                    kernel_size=1,
                    stride=1,
                    bias=bias), nn.BatchNorm2d(channels), activation())

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after adding the branches
        self.out_activation = activation()

        ### Add SE Block
        self.se = SEBlock(channels)

    def forward(self, x):
        # Main branch shortcut
        main = x

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Add main and extension branches
        out = main + ext

        out = self.out_activation(out)

        ## Add SEBlock
        out = self.se(out)

        return out

# Middle Downsampling
class DownsamplingBottleneck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 internal_ratio=4,
                 dropout_prob=0.0,
                 bias=False,
                 relu=True):
        super().__init__()

        internal_channels = in_channels // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # Main branch - max pooling followed by feature map (channels) padding
        self.main_max1 = nn.MaxPool2d(
            2,
            stride=2)

        # 2x2 projection convolution with stride 2
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                internal_channels,
                kernel_size=2,
                stride=2,
                bias=bias), nn.BatchNorm2d(internal_channels), activation())

        # Convolution
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                internal_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias), nn.BatchNorm2d(internal_channels), activation())

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(out_channels), activation())

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after concatenating the branches
        self.out_activation = activation()
        self.se = SEBlock(out_channels)

    def forward(self, x):
        # Main branch shortcut

        main = self.main_max1(x)

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Main branch channel padding
        n, ch_ext, h, w = ext.size()
        ch_main = main.size()[1]
        padding = torch.zeros(n, ch_ext - ch_main, h, w)

        # Before concatenating, check if main is on the CPU or GPU and
        # convert padding accordingly
        if main.is_cuda:
            padding = padding.to(ext.get_device())
        # Concatenate
        main = torch.cat((main, padding), 1)

        # Add main and extension branches
        out = main + ext

        out = self.out_activation(out)

        ### Add SEBlock
        out = self.se(out)

        return out


# ✅ Multi-scale context aggregation (PSP-style)
class PSPModule(nn.Module):
    def __init__(self, in_channels, bins=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        out_channels = in_channels // len(bins)
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for bin in bins
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + out_channels * len(bins), in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
    def forward(self, x):
        h, w = x.size()[2:]
        out = [x]
        for stage in self.stages:
            out.append(F.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=True))
        out = torch.cat(out, 1)
        return self.bottleneck(out)


# ✅ Boundary-aware decoder
class BoundaryAwareDecoder(nn.Module):
    def __init__(self, in_channels_high, in_channels_low, out_channels):
        super(BoundaryAwareDecoder, self).__init__()
        self.high_conv = nn.Sequential(
            nn.Conv2d(in_channels_high, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        self.low_conv = nn.Sequential(
            nn.Conv2d(in_channels_low, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels * 2, 1, 1),
            nn.Sigmoid())

    def forward(self, high, low):
        high = self.high_conv(high)
        low = self.low_conv(low)
        low = F.interpolate(low, size=high.size()[2:], mode='bilinear', align_corners=True)
        # Attention-based fusion
        cat = torch.cat([high, low], dim=1)
        att = self.attention(cat)
        fused = self.fuse(torch.cat([high * att, low * (1 - att)], dim=1))
        return fused

# Lightweight Infrared small segmentation
class LW_MOD(nn.Module):
    def __init__(self, n_classes=1, encoder_relu=False, decoder_relu=True, channel=(8, 32, 64), dilations=(2,4,8,16), kernel_size=(3,5,7,9), padding=(1,2,3,4), in_channels=2):
        super().__init__()

        # Stage 1 - Encoder
        if in_channels not in (2, 5):
            raise ValueError('The motion head expects 2 flow or 5 flow+RGB channels.')
        self.initial_block = InitialBlock(in_channels, channel[0], relu=encoder_relu, pool_channels=2)

        # Stage 2 - Encoder
        self.downsample1_0 = DownsamplingBottleneck(
            channel[0],
            channel[1],
            dropout_prob=0.01,
            relu=encoder_relu)
        self.regular1_1 = RegularBottleneck(
            channel[1], padding=1, regular=True, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_2 = RegularBottleneck(
            channel[1], padding=1, regular=True, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_3 = RegularBottleneck(
            channel[1], padding=1, regular=True, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_4 = RegularBottleneck(
            channel[1], padding=1, regular=True, dropout_prob=0.01, relu=encoder_relu)

        # Stage 3 - Encoder
        self.downsample2_0 = DownsamplingBottleneck(
            channel[1],
            channel[2],
            dropout_prob=0.1,
            relu=encoder_relu)

        #DAAA Module1
        self.Depthwise2_1 = RegularBottleneck(
            channel[2], padding=1, depthwise=True, dropout_prob=0.1, relu=encoder_relu)
        self.Atrous2_2 = RegularBottleneck(
            channel[2], dilation=dilations[0], padding=dilations[0], dilated=True, dropout_prob=0.1, relu=encoder_relu)
        self.Asymmetric2_3 = RegularBottleneck(
            channel[2],
            kernel_size=kernel_size[0],
            padding=padding[0],
            asymmetric=True,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.Atrous2_4 = RegularBottleneck(
            channel[2], dilation=dilations[1], padding=dilations[1], dilated=True, dropout_prob=0.1, relu=encoder_relu)

        # DAAA Module2
        self.Depthwise2_5 = RegularBottleneck(
            channel[2], padding=1, depthwise=True, dropout_prob=0.1, relu=encoder_relu)
        self.Atrous2_6 = RegularBottleneck(
            channel[2], dilation=dilations[2], padding=dilations[2], dilated=True, dropout_prob=0.1, relu=encoder_relu)
        self.Asymmetric2_7 = RegularBottleneck(
            channel[2],
            kernel_size=kernel_size[1],
            padding=padding[1],
            asymmetric=True,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.Atrous2_8 = RegularBottleneck(
            channel[2], dilation=dilations[3], padding=dilations[3], dilated=True, dropout_prob=0.1, relu=encoder_relu)

        # DAAA Module3
        self.Depthwise3_1 = RegularBottleneck(
            channel[2], padding=1, depthwise=True, dropout_prob=0.1, relu=encoder_relu)
        self.Atrous3_2 = RegularBottleneck(
            channel[2], dilation=dilations[0], padding=dilations[0], dilated=True, dropout_prob=0.1, relu=encoder_relu)
        self.Asymmetric3_3 = RegularBottleneck(
            channel[2],
            kernel_size=kernel_size[2],
            padding=padding[2],
            asymmetric=True,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.Atrous3_4 = RegularBottleneck(
            channel[2], dilation=dilations[1], padding=dilations[1], dilated=True, dropout_prob=0.1, relu=encoder_relu)

        # DAAA Module4
        self.Depthwise3_5 = RegularBottleneck(
            channel[2], padding=1, depthwise=True, dropout_prob=0.1, relu=encoder_relu)
        self.Atrous3_6 = RegularBottleneck(
            channel[2], dilation=dilations[2], padding=dilations[2], dilated=True, dropout_prob=0.1, relu=encoder_relu)
        self.Asymmetric3_7 = RegularBottleneck(
            channel[2],
            kernel_size=kernel_size[3],
            padding=padding[3],
            asymmetric=True,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.Atrous3_8 = RegularBottleneck(
            channel[2], dilation=dilations[3], padding=dilations[3], dilated=True, dropout_prob=0.1, relu=encoder_relu)

        # ✅ PSP module sau encoder
        self.psp = PSPModule(channel[2], bins=(1, 2, 3, 6))

        # Stage 4 - Decoder
        self.transposed4_conv = nn.ConvTranspose2d(
            channel[2],
            channel[1],
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)
        self.regular4_1 = RegularBottleneck(
            channel[1], padding=1, regular=True, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_2 = RegularBottleneck(
            channel[1], padding=1, regular=True, dropout_prob=0.1, relu=decoder_relu)

        self.out1 = nn.Conv2d(32,1,kernel_size=3, padding=1)

        # Stage 5 - Decoder
        self.transposed5_conv = nn.ConvTranspose2d(
            channel[1],
            channel[0],
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)
        self.boundary_decoder = BoundaryAwareDecoder(channel[0], channel[0], channel[0])  # high: x5, low: x1
        self.regular5_1 = RegularBottleneck(
            channel[0], padding=1, regular=True, dropout_prob=0.1, relu=decoder_relu)
        
        self.out2 = nn.Conv2d(8,1,kernel_size=3, padding=1)

        # Stage 6 - Decoder
        self.transposed6_conv = nn.ConvTranspose2d(
            channel[0],
            n_classes,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)

        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                64,
                32,
                kernel_size=1,
                stride=1,
                bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(
                16,
                8,
                kernel_size=1,
                stride=1,
                bias=False), nn.BatchNorm2d(8), nn.ReLU())

        self.conv1 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(32, 8, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv3 = nn.Conv2d(8, n_classes, kernel_size=1, stride=1, padding=0, bias=False)
        self.out_cnn = nn.Conv2d(3, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)

        # # Edit for TCCLA
        # self.conv3 = nn.Conv2d(8, 3, kernel_size=1, stride=1, padding=0, bias=False)
        # self.out_cnn = nn.Conv2d(3, 3, kernel_size=1)
        # self.bn = nn.BatchNorm2d(3)

    def forward(self, x):
        # print(x.size())
        B,C,H,W = x.size()
        decode_outputs = []
        # Stage 1-Encoder
        input_size = x.size()         # 1 3 256 256
        # print("input_size : ", input_size)
        x1 = self.initial_block(x)    # 1 8 128 128
        # Stage 2-Encoder
        stage1_input_size = x1.size() # 1 8 128 128
        x2 = self.downsample1_0(x1)   # 1 32 64 64
        x2 = self.regular1_1(x2)      # 1 32 64 64
        x2 = self.regular1_2(x2)      # 1 32 64 64
        x2 = self.regular1_3(x2)      # 1 32 64 64
        x2 = self.regular1_4(x2)      # 1 32 64 64

        # Stage3.1 -Encoder
        stage2_input_size = x2.size() # 1 32 64 64
        x3 = self.downsample2_0(x2)   # 1 64 32 32
        #DAAA Module1
        x3 = self.Depthwise2_1(x3)    # 1 64 32 32
        x3 = self.Atrous2_2(x3)       # 1 64 32 32
        x3 = self.Asymmetric2_3(x3)   # 1 64 32 32
        x3 = self.Atrous2_4(x3)       # 1 64 32 32
        #DAAA Module2
        x3 = self.Depthwise2_5(x3)    # 1 64 32 32
        x3 = self.Atrous2_6(x3)       # 1 64 32 32
        x3 = self.Asymmetric2_7(x3)   # 1 64 32 32
        x3 = self.Atrous2_8(x3)       # 1 64 32 32

        # Stage3.2 -Encoder
        #DAAA Module3
        x3 = self.Depthwise3_1(x3)    # 1 64 32 32
        x3 = self.Atrous3_2(x3)       # 1 64 32 32
        x3 = self.Asymmetric3_3(x3)   # 1 64 32 32
        x3 = self.Atrous3_4(x3)       # 1 64 32 32
        #DAAA Module4
        x3 = self.Depthwise3_5(x3)    # 1 64 32 32
        x3 = self.Atrous3_6(x3)       # 1 64 32 32
        x3 = self.Asymmetric3_7(x3)   # 1 64 32 32
        x3 = self.Atrous3_8(x3)       # 1 64 32 32

        # Stage4 -Decoder
        x4 = self.transposed4_conv(x3, output_size=stage2_input_size)  # 1 32 64 64
        # x4 = F.interpolate(x3, size=(64, 64), mode='bilinear', align_corners=True)  # 1 64 64 64
        # x4 = self.conv1(x4)

        # sum
        x4 = x4 + x2                                                   # 1 32 64 64
        # concat
        # x4 = torch.cat([x4, x2], dim=1)
        # x4 = self.ext_conv1(x4)
        x4 = self.regular4_1(x4)                                       # 1 32 64 64
        x4 = self.regular4_2(x4)                                       # 1 32 64 64

        decode_outputs.append(F.interpolate(self.out1(x4), size=[H, W], mode='bilinear', align_corners=False))

        # Stage5 -Decoder
        x5 = self.transposed5_conv(x4, output_size=stage1_input_size)  # 1 8 128 128
        # x5 = F.interpolate(x4, size=(128, 128), mode='bilinear', align_corners=True)  # 1 32 128 128
        # x5 = self.conv2(x5)

        # sum
        x5 = x5 + x1                                                   # 1 8 128 128
        # concat
        # x5 = torch.cat([x5, x1], dim=1)
        # x5 = self.ext_conv2(x5)
        x5 = self.regular5_1(x5)                                       # 1 8 128 128
        
        decode_outputs.append(F.interpolate(self.out2(x5), size=[H, W], mode='bilinear', align_corners=False))

        # Stage6 -Decoder
        x6 = self.transposed6_conv(x5, output_size=input_size)   # 1 1  256 256
        # x6 = self.conv3(x5)  # 1 16 128 128  1*1conv.
        # x6 = F.interpolate(x6, size=(256, 256), mode='bilinear', align_corners=True)
        decode_outputs.append(x6)

        out = self.out_cnn(torch.concat(decode_outputs, dim=1))
        
        if self.training :
            return [out] + decode_outputs
            # return [out]

        else:
            # for segment
            out = self.bn(out)
            return out 



def LW_OpticalFLow(in_channels=2):
    print("Using LW_V1")
    return LW_MOD(channel=(8, 32, 64), dilations=(2,4,8,16), kernel_size=(7,7,7,7), padding=(3,3,3,3), in_channels=in_channels)

if __name__ == '__main__':
    from thop import profile, clever_format
    import time
    import matplotlib.pyplot as plt
    inputs = torch.randn((10, 3, 512, 640))
    
    model = LW_IRST_ablation(channel=(8, 32, 64), dilations=(2,4,8,16), kernel_size=(7,7,7,7), padding=(3,3,3,3)) # kernel_size/padding = 5/2 7/3 9/4
    macs, params = profile(model, inputs=(inputs, ))
    macs, params = clever_format([macs, params])
    # model = model.eval()

    # import torch.onnx
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # torch.onnx.export(model, inputs, 'LWnet.onnx', 
    #             opset_version=17, verbose=False, 
    #             do_constant_folding=True,
    #             input_names=['input'], output_names=['output']
    #             )
    start = time.perf_counter()
    out = model(inputs)
    end = time.perf_counter()
    # FLOPs, params = profile(model, inputs=(inputs,))
    # macs, params = clever_format([macs, params])
    running_FPS = 1 / (end - start)
    print("macs: ", macs)
    print("params: ", params)
    print('running_FPS:', running_FPS)
