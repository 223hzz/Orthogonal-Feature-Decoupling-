# --------------------------------------------------------
# Dual Cross Attention
# Copyright (c) 2023 Gorkem Can Ates
# Licensed under The MIT License [see LICENSE for details]
# Written by Gorkem Can Ates (gca45@miami.edu)
# --------------------------------------------------------


import torch
import torch.nn as nn
import einops
import matplotlib.pyplot as plt
from nets.dual_DCA.main_block import *


def params(module):
    return sum(p.numel() for p in module.parameters())


class UpsampleConv(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=(3, 3),
                 padding=(1, 1),
                 norm_type=None,
                 activation=False,
                 scale=(2, 2),
                 conv='conv') -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale,
                              mode='bilinear',
                              align_corners=True)
        if conv == 'conv':
            self.conv = conv_block(in_features=in_features,
                                   out_features=out_features,
                                   kernel_size=(1, 1),
                                   padding=(0, 0),
                                   norm_type=norm_type,
                                   activation=activation)
        elif conv == 'depthwise':
            self.conv = depthwise_conv_block(in_features=in_features,
                                             out_features=out_features,
                                             kernel_size=kernel_size,
                                             padding=padding,
                                             norm_type=norm_type,
                                             activation=activation)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class depthwise_projection(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 groups,
                 kernel_size=(1, 1),
                 padding=(0, 0),
                 norm_type=None,
                 activation=False,
                 pointwise=False) -> None:
        super().__init__()

        self.proj = depthwise_conv_block(in_features=in_features,
                                         out_features=out_features,
                                         kernel_size=kernel_size,
                                         padding=padding,
                                         groups=groups,
                                         pointwise=pointwise,
                                         norm_type=norm_type,
                                         activation=activation)

    def forward(self, x):
        P = int(x.shape[1] ** 0.5)#假设输入特征的形状为 (B, H*W, C)，则 P 表示空间维度 H 和 W 的平方根，即 P = int(x.shape[1] ** 0.5),用途为计算 H 和 W： 通过开方运算，得到 P，它表示 H 和 W 的值，（假设 H 和 W 相等）
        x = einops.rearrange(x, 'B (H W) C-> B C H W', H=P)
        x = self.proj(x)
        x = einops.rearrange(x, 'B C H W -> B (H W) C')
        return x
# 深度卷积 (Depthwise Convolution):
# 传统的卷积操作是对所有通道同时进行卷积。
# 而深度卷积对每个输入通道独立地应用一个卷积核，也就是说，一个卷积核只负责一个通道，各个通道之间没有信息交流。
# 这样做的好处是大大减少了参数量和计算量。
# 逐点卷积 (Pointwise Convolution):
# 在深度卷积之后，每个通道的特征图都只包含了该通道的信息。
# 为了融合不同通道之间的信息，使用1x1的卷积核对所有通道的特征图进行线性组合。
# 由于卷积核大小只有1x1，所以这一步的计算量相对较小。


class conv_projection(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.proj = conv_block(in_features=in_features,
                               out_features=out_features,
                               kernel_size=(1, 1),
                               padding=(0, 0),
                               norm_type=None,
                               activation=False)

    def forward(self, x):
        P = int(x.shape[1] ** 0.5)
        x = einops.rearrange(x, 'B (H W) C-> B C H W', H=P)
        x = self.proj(x)
        x = einops.rearrange(x, 'B C H W -> B (H W) C')
        return x


class PatchEmbedding(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 size,
                 patch=28,
                 proj='conv'
                 ) -> None:
        super().__init__()
        self.proj = proj
        if self.proj == 'conv':
            self.projection = nn.Conv2d(in_channels=in_features,
                                        out_channels=out_features,
                                        kernel_size=size // patch_size,
                                        stride=size // patch_size,
                                        padding=(0, 0),
                                        )

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class PoolEmbedding(nn.Module):
    def __init__(self,
                 pooling,
                 patch,
                 ) -> None:
        super().__init__()
        self.projection = pooling(output_size=(patch, patch))

    def forward(self, x):
        x = self.projection(x)
        x = einops.rearrange(x, 'B C H W -> B (H W) C')
        return x


class Layernorm(nn.Module):
    def __init__(self, features, eps=1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(features, eps=eps)

    def forward(self, x):
        H = x.shape[2]
        x = einops.rearrange(x, 'B C H W -> B (H W) C')
        x = self.norm(x)
        x = einops.rearrange(x, 'B (H W) C-> B C H W', H=H)
        return x


class ScaleDotProduct(nn.Module):#计算缩放点积注意力，得到注意力权重 att
    def __init__(self) -> None:
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2, x3, scale):
        x2 = x2.transpose(-2, -1)
        # print("x2.shape:",x2.shape)
        # print("x1.shape:",x1.shape)
        # x12 = torch.einsum('bhcw, bhwk -> bhck', x1, x2) * scale
        x12 = (x1 @ x2) * scale
        att = self.softmax(x12)
        # x123 = torch.einsum('bhcw, bhwk -> bhck', att, x3)
        x123 = att @ x3
        return x123