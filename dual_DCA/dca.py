# --------------------------------------------------------
# Dual Cross Attention
# Copyright (c) 2023 Gorkem Can Ates
# Licensed under The MIT License [see LICENSE for details]
# Written by Gorkem Can Ates (gca45@miami.edu)
# --------------------------------------------------------


import torch
import torch.nn as nn
import einops
from nets.dual_DCA.main_block import *
from nets.dual_DCA.dca_uilt import *


class ChannelAttention(nn.Module):
    def __init__(self, in_features, out_features, n_heads=1) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.q_map = depthwise_projection(in_features=out_features,
                                          out_features=out_features,
                                          groups=out_features)
        self.k_map = depthwise_projection(in_features=in_features,
                                          out_features=in_features,
                                          groups=in_features)
        self.v_map = depthwise_projection(in_features=in_features,
                                          out_features=in_features,
                                          groups=in_features)

        self.projection = depthwise_projection(in_features=out_features,
                                               out_features=out_features,
                                               groups=out_features)
        self.sdp = ScaleDotProduct() #使用 sdp 对象计算缩放点积注意力，得到注意力权重 att

    def forward(self, x):
        q, k, v = x[0], x[1], x[2]
        q = self.q_map(q)
        k = self.k_map(k)
        v = self.v_map(v)
        b, hw, c_q = q.shape
        c = k.shape[2]
        scale = c ** -0.5
        q = q.reshape(b, hw, self.n_heads, c_q // self.n_heads).permute(0, 2, 1, 3).transpose(2, 3)
        k = k.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3).transpose(2, 3)
        v = v.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3).transpose(2, 3)
        att = self.sdp(q, k, v, scale).permute(0, 3, 1, 2).flatten(2)
        att = self.projection(att)
        return att


class SpatialAttention(nn.Module):
    def __init__(self, in_features, out_features, n_heads=4) -> None:
        super().__init__()
        self.n_heads = n_heads

        self.q_map = depthwise_projection(in_features=in_features,
                                          out_features=in_features,
                                          groups=in_features)
        self.k_map = depthwise_projection(in_features=in_features,
                                          out_features=in_features,
                                          groups=in_features)
        self.v_map = depthwise_projection(in_features=out_features,
                                          out_features=out_features,
                                          groups=out_features)

        self.projection = depthwise_projection(in_features=out_features,
                                               out_features=out_features,
                                               groups=out_features)
        self.sdp = ScaleDotProduct()

    def forward(self, x):
        q, k, v = x[0], x[1], x[2]
        q = self.q_map(q)
        k = self.k_map(k)
        v = self.v_map(v)
        b, hw, c = q.shape
        c_v = v.shape[2]
        scale = (c // self.n_heads) ** -0.5
        q = q.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3)
        k = k.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3)
        v = v.reshape(b, hw, self.n_heads, c_v // self.n_heads).permute(0, 2, 1, 3)
        att = self.sdp(q, k, v, scale).transpose(1, 2).flatten(2)
        x = self.projection(att)
        return x


class CCSABlock(nn.Module):
    def __init__(self,
                 features,
                 channel_head,#每个阶段通道注意力机制中注意力头的数量
                 spatial_head,#每个阶段空间注意力机制中注意力头的数量
                 spatial_att=True,#是否使用空间注意力机制
                 channel_att=True) -> None:#是否使用通道注意力机制
        super().__init__()
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        if self.channel_att:
            self.channel_norm = nn.ModuleList([nn.LayerNorm(in_features,
                                                            eps=1e-6)
                                               for in_features in features])

            self.c_attention = nn.ModuleList([ChannelAttention(
                in_features=sum(features),
                out_features=feature,
                n_heads=head,
            ) for feature, head in zip(features, channel_head)])
            #假设 features = [64, 128, 256] 且 channel_head = [4, 8, 16]， 那么这行代码相当于创建了三个 ChannelAttention 模块:
            # ChannelAttention(in_features=448, out_features=64, n_heads=4)
            # ChannelAttention(in_features=448, out_features=128, n_heads=8)
            # ChannelAttention(in_features=448, out_features=256, n_heads=16)
        if self.spatial_att:
            self.spatial_norm = nn.ModuleList([nn.LayerNorm(in_features,
                                                            eps=1e-6)
                                               for in_features in features])

            self.s_attention = nn.ModuleList([SpatialAttention(
                in_features=sum(features),
                out_features=feature,
                n_heads=head,
            )
                for feature, head in zip(features, spatial_head)])

    def forward(self, x):
        if self.channel_att:
            x_ca = self.channel_attention(x)
            x = self.m_sum(x, x_ca)
        if self.spatial_att:
            x_sa = self.spatial_attention(x)
            x = self.m_sum(x, x_sa)
        return x

    def channel_attention(self, x):
        x_c = self.m_apply(x, self.channel_norm)#将所有输入的图片图片统一进行归一化
        x_cin = self.cat(*x_c)#拼接所有图片
        x_in = [[q, x_cin, x_cin] for q in x_c]
        #假设 x_c = [q1, q2, q3]， 那么这行代码会生成以下列表：
        # x_in = [
        #     [q1, x_cin, x_cin],
        #     [q2, x_cin, x_cin],
        #     [q3, x_cin, x_cin]]
        x_att = self.m_apply(x_in, self.c_attention)
        return x_att

    def spatial_attention(self, x):
        x_c = self.m_apply(x, self.spatial_norm)
        x_cin = self.cat(*x_c)
        x_in = [[x_cin, x_cin, v] for v in x_c]
        x_att = self.m_apply(x_in, self.s_attention)
        return x_att

    def m_apply(self, x, module):
        return [module[i](j) for i, j in enumerate(x)]

    # x = [x1, x2, x3]是一个包含三个元素的列表。
    # module = nn.ModuleList([module1, module2, module3])
    # 是一个包含三个模块的nn.ModuleList对象。
    # 那么m_apply(x, module)的执行过程如下：
    # 遍历x，分别取出x1, x2, x3。
    # 将x1输入到module1中计算，得到结果y1 = module1(x1)。
    # 将x2输入到module2中计算，得到结果y2 = module2(x2)。
    # 将x3输入到module3中计算，得到结果y3 = module3(x3)。
    # 返回结果列表[y1, y2, y3]

    def m_sum(self, x, y):
        return [xi + xj for xi, xj in zip(x, y)]

    def cat(self, *args):
        return torch.cat((args), dim=2)
    #这个 cat 函数的作用是将多个张量沿着维度 2（通道维度）拼接在一起

class DCA(nn.Module):
    def __init__(self,
                 features,
                 strides,#一个列表，包含了每个阶段的上采样步长
                 patch=28,
                 channel_att=True,
                 spatial_att=True,
                 n=1,
                 channel_head=[1, 1, 1, 1],
                 spatial_head=[4, 4, 4, 4],
                 ):
        super().__init__()
        self.n = n
        self.features = features
        self.spatial_head = spatial_head
        self.channel_head = channel_head
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        self.patch = patch
        self.patch_avg = nn.ModuleList([PoolEmbedding(
            pooling=nn.AdaptiveAvgPool2d,
            patch=patch,
        )
            for _ in features])#调整每个图片的大小为固定尺寸HW
        self.avg_map = nn.ModuleList([depthwise_projection(in_features=feature,
                                                           out_features=feature,
                                                           kernel_size=(1, 1),
                                                           padding=(0, 0),
                                                           groups=feature
                                                           )
                                      for feature in features])

        self.attention = nn.ModuleList([
            CCSABlock(features=features,
                      channel_head=channel_head,
                      spatial_head=spatial_head,
                      channel_att=channel_att,
                      spatial_att=spatial_att)
            for _ in range(n)])

        self.upconvs = nn.ModuleList([UpsampleConv(in_features=feature,
                                                   out_features=feature,
                                                   kernel_size=(1, 1),
                                                   padding=(0, 0),
                                                   norm_type=None,
                                                   activation=False,
                                                   scale=stride,
                                                   conv='conv')
                                      for feature, stride in zip(features, strides)])
        self.bn_relu = nn.ModuleList([nn.Sequential(
            nn.BatchNorm2d(feature),
            nn.ReLU()
        )
            for feature in features])

    def forward(self, raw):
        x = self.m_apply(raw, self.patch_avg)
        x = self.m_apply(x, self.avg_map)
        for block in self.attention:
            x = block(x)
        x = [self.reshape(i) for i in x]
        x = self.m_apply(x, self.upconvs)
        x_out = self.m_sum(x, raw)
        x_out = self.m_apply(x_out, self.bn_relu)
        return (*x_out,)

    def m_apply(self, x, module):
        return [module[i](j) for i, j in enumerate(x)]

    def m_sum(self, x, y):
        return [xi + xj for xi, xj in zip(x, y)]

    def reshape(self, x):
        return einops.rearrange(x, 'B (H W) C-> B C H W', H=self.patch)



