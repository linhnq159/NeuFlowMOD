import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.nn.functional as F
from mmcv.runner import (BaseModule)
from mmcv.utils import to_2tuple
from functools import partial

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

# Initial Downsampling
class InitialBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=False,
                 relu=True):
        super().__init__()

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        self.main_branch = nn.Conv2d(
            in_channels,
            out_channels - 3,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=bias)

        # Extension branch
        self.ext_branch = nn.MaxPool2d(3, stride=2, padding=1)

        # Initialize batch normalization to be used after concatenation
        self.batch_norm = nn.BatchNorm2d(out_channels)

        # PReLU layer to apply after concatenating the branches
        self.out_activation = activation()

    def forward(self, x):
        main = self.main_branch(x) #1 8 128 128
        ext = self.ext_branch(x) #1 3 128 128

        # Concatenate branches
        out = torch.cat((main, ext), 1) # 1 16 128 128

        # Apply batch normalization
        out = self.batch_norm(out)

        return self.out_activation(out)

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

        return self.out_activation(out)

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

        return self.out_activation(out)


#################################### MHATTN ####################################

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class Mlp(BaseModule):
    def __init__(self, 
                 in_features, 
                 hidden_features=None, 
                 out_features=None, 
                 act_layer=nn.GELU, 
                 drop=0.,
                 init_cfg=None):

        super().__init__(init_cfg)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)


    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x + self.dwconv(x, H, W))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(BaseModule):
    def __init__(self, 
                 dim, 
                 ca_num_heads=4, 
                 sa_num_heads=8,
                 qkv_bias=False, 
                 qk_scale=None, 
                 attn_drop=0., 
                 proj_drop=0.,
                 ca_attention=1,
                 expand_ratio=2,
                 init_cfg=None):
        super().__init__(init_cfg)
        
        self.ca_attention = ca_attention
        self.dim = dim
        self.ca_num_heads = ca_num_heads
        self.sa_num_heads = sa_num_heads

        assert dim % ca_num_heads == 0, f"dim {dim} should be divided by num_heads {ca_num_heads}."
        assert dim % sa_num_heads == 0, f"dim {dim} should be divided by num_heads {sa_num_heads}."

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.split_groups = self.dim//ca_num_heads
        
        if ca_attention == 1:
            self.v = nn.Linear(dim, dim, bias=qkv_bias)
            self.s = nn.Linear(dim, dim, bias=qkv_bias)
            for i in range(self.ca_num_heads):
                local_conv = nn.Conv2d(dim//self.ca_num_heads, dim//self.ca_num_heads, kernel_size=(3+i*2), padding=(1+i), stride=1, groups=dim//self.ca_num_heads)
                setattr(self, f"local_conv_{i + 1}", local_conv)

            self.proj0 = nn.Conv2d(dim, dim*expand_ratio, kernel_size=1, padding=0, stride=1, groups=self.split_groups)
            self.bn = nn.BatchNorm2d(dim*expand_ratio)
            self.proj1 = nn.Conv2d(dim*expand_ratio, dim, kernel_size=1, padding=0, stride=1)

        else:

            head_dim = dim // sa_num_heads
            self.scale = qk_scale or head_dim ** -0.5
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.attn_drop = nn.Dropout(attn_drop)
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
            self.dw_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)
        

    def forward(self, x, H, W):
        B, N, C = x.shape
        if self.ca_attention == 1:
            v = self.v(x)
            s = self.s(x).reshape(B, H, W, self.ca_num_heads, C//self.ca_num_heads).permute(3, 0, 4, 1, 2)
            for i in range(self.ca_num_heads):
                local_conv = getattr(self, f"local_conv_{i + 1}")
                s_i= s[i]
                s_i = local_conv(s_i).reshape(B, self.split_groups, -1, H, W)
                if i == 0:
                    s_out = s_i
                else:
                    s_out = torch.cat([s_out,s_i],2)


            s_out = s_out.reshape(B, C, H, W)
            s_out = self.proj1(self.act(self.bn(self.proj0(s_out)))).reshape(B, C, N).permute(0, 2, 1)
            x = s_out * v

        else:
            q = self.q(x).reshape(B, N, self.sa_num_heads, C // self.sa_num_heads).permute(0, 2, 1, 3)
            kv = self.kv(x).reshape(B, -1, 2, self.sa_num_heads, C // self.sa_num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C) + \
                self.dw_conv(v.transpose(1, 2).reshape(B, N, C).transpose(1, 2).view(B,C, H, W)).view(B, C, N).transpose(1, 2)
        
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class ATTN_Block(BaseModule):

    def __init__(self, 
                 dim, 
                 ca_num_heads, 
                 sa_num_heads,
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop=0., 
                 attn_drop=0.,
                 drop_path=0., 
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm,
                 ca_attention=1,
                 expand_ratio=2,
                 init_cfg=None):
        super().__init__()

        self.init_cfg = init_cfg

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            ca_num_heads=ca_num_heads, sa_num_heads=sa_num_heads, qkv_bias=qkv_bias, 
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, ca_attention=ca_attention, 
            expand_ratio=expand_ratio,init_cfg=None)
            
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, 
            hidden_features=mlp_hidden_dim, 
            act_layer=act_layer, 
            drop=drop,
            init_cfg=None)


    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm2(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(BaseModule):

    def __init__(self, 
                 patch_size=3, 
                 stride=2, 
                 in_chans=3, 
                 embed_dim=768,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        patch_size = to_2tuple(patch_size)

        self.stride = stride
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)


    def forward(self, x):

        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W




# Lightweight Infrared small segmentation
class LW_IRST_ablation(nn.Module):
    def __init__(self, n_classes=1, encoder_relu=False, decoder_relu=True, channel=(8, 32, 64), dilations=(2,4,8,16), kernel_size=(3,5,7,9), padding=(1,2,3,4),norm_layer = partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()

        # Stage 1 - Encoder
        self.initial_block = InitialBlock(3, channel[0], relu=encoder_relu)

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


        # MAT
        self.patch_embed_1 = OverlapPatchEmbed(patch_size=3, stride=1, in_chans=channel[1], embed_dim=channel[1])

        self.MHA_1 = ATTN_Block(dim=channel[1], ca_num_heads=4, sa_num_heads=16, ca_attention=1, expand_ratio=2, norm_layer= norm_layer)

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

        # MSA 2
        self.patch_embed_2 = OverlapPatchEmbed(patch_size=3, stride=1, in_chans=channel[0], embed_dim=channel[0])
        self.MHA_2 = ATTN_Block(dim=channel[0], ca_num_heads=4, sa_num_heads=8, ca_attention=1, expand_ratio=2, norm_layer= norm_layer)

        # Stage 5 - Decoder
        self.transposed5_conv = nn.ConvTranspose2d(
            channel[1],
            channel[0],
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)
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
    def forward(self, x):
        B = x.size(0)
        decode_outputs = []
        # Stage 1-Encoder
        input_size = x.size()         # 1 3 256 256
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

        #MSA 1
        path_embed_1, H, W = self.patch_embed_1(x2)
        MHA_1 = self.MHA_1(path_embed_1, H, W)
        MHA_1 = MHA_1.reshape(B,H,W,-1).permute(0,3,1,2).contiguous()
        # sum
        x4 = x4 + MHA_1                                                # 1 32 64 64

        # concat
        # x4 = torch.cat([x4, x2], dim=1)
        # x4 = self.ext_conv1(x4)
        x4 = self.regular4_1(x4)                                       # 1 32 64 64
        x4 = self.regular4_2(x4)                                       # 1 32 64 64

        decode_outputs.append(F.interpolate(self.out1(x4), size=[512, 640], mode='bilinear', align_corners=False))

 

        # Stage5 -Decoder
        x5 = self.transposed5_conv(x4, output_size=stage1_input_size)  # 1 8 128 128
        # x5 = F.interpolate(x4, size=(128, 128), mode='bilinear', align_corners=True)  # 1 32 128 128
        # x5 = self.conv2(x5)

        #MSA 2
        path_embed_2, H, W = self.patch_embed_2(x1)
        MHA_2 = self.MHA_2(path_embed_2, H, W)
        MHA_2 = MHA_2.reshape(B,H,W,-1).permute(0,3,1,2).contiguous()
        # sum
        x5 = x5 + MHA_2                                                   # 1 8 128 128

        # concat
        # x5 = torch.cat([x5, x1], dim=1)
        # x5 = self.ext_conv2(x5)
        x5 = self.regular5_1(x5)                                       # 1 8 128 128
        
        decode_outputs.append(F.interpolate(self.out2(x5), size=[512, 640], mode='bilinear', align_corners=False))

        # Stage6 -Decoder
        x6 = self.transposed6_conv(x5, output_size=input_size)         # 1 1  256 256
        # x6 = self.conv3(x5)  # 1 16 128 128  1*1conv.
        # x6 = F.interpolate(x6, size=(256, 256), mode='bilinear', align_corners=True)
        decode_outputs.append(x6)

        out = self.out_cnn(torch.concat(decode_outputs, dim=1))
        
        if self.training :
            return [out] + decode_outputs

        else:
            out = self.bn(out)
            return out



def LW_custom():
    return LW_IRST_ablation(channel=(8, 32, 64), dilations=(2,4,8,16), kernel_size=(7,7,7,7), padding=(3,3,3,3))

if __name__ == '__main__':
    from thop import profile, clever_format
    import time
    import matplotlib.pyplot as plt
    inputs = torch.randn((1, 3, 512, 640))
    
    model = LW_IRST_ablation(channel=(8, 32, 64), dilations=(2,4,8,16), kernel_size=(7,7,7,7), padding=(3,3,3,3)) # kernel_size/padding = 5/2 7/3 9/4
    model = model.eval()
    macs, params = profile(model, inputs=(inputs, ))
    macs, params = clever_format([macs, params])
    # model = model.eval()

    # import torch.onnx
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # torch.onnx.export(model, inputs, 'LWnet.onnx', 
    #             # operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
    #             opset_version=12, verbose=False, 
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
