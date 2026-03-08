from functools import partial

import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
#重构unetup类
# class unetUp(nn.Module):
#     def __init__(self, dim, depth, i_layer, upsample):
#         super(unetUp, self).__init__()
#         self.vsslayer_block = create_vsslayer_up(dim, depth, i_layer, upsample)
#
#     def forward(self, inputs1, inputs2):#也就意味着可以随意交换顺序
#         outputs = inputs1+inputs2  # 叠加操作dim=1代表按行排堆砌，两张图输入的尺寸必须一样，两张图通道数相加。
#         outputs =  self.vsslayer_block(outputs.permute(0, 2, 3, 1))
#         return outputs.permute(0, 3, 1, 2)
#
# inputs1 = torch.randn(1,512, 28, 28)
# inputs2 = torch.randn(1,512, 28, 28)
# unet_up = unetUp(dim=512, depth=2,i_layer=0,upsample=None)
# output = unet_up(inputs1, inputs2)
# print(output.shape)


from nets.mamaba_net.IntmdSequential import IntermediateSequential
#重构自注意力机制类
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

#
# class SelfAttention(nn.Module):
#     def __init__(
#         self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
#     ):
#         super().__init__()
#         self.num_heads = heads
#         head_dim = dim // heads
#         self.scale = qk_scale or head_dim ** -0.5
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(dropout_rate)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(dropout_rate)
#
#     def forward(self, x):
#         B,C,H,W  = x.shape
#         N=H*W
#         qkv = (
#             self.qkv(x)
#             .reshape(B, N, 3, self.num_heads, C // self.num_heads)
#             .permute(2, 0, 3, 1, 4)
#         )
#         q, k, v = (
#             qkv[0],
#             qkv[1],
#             qkv[2],
#         )  # make torchscript happy (cannot use tensor as tuple)
#         #这段代码先计算了查询（query）和键（key）的点积，然后乘以一个缩放因子self.scale。缩放因子通常是1 / sqrt(d_k)，其中d_k是键的维度，这样做是为了防止点积过大导致softmax函数的梯度过小。接着，应用softmax函数来规范化这些分数，使得每一行的分数之和为1，这样每一行就代表了对应键上的概率分布。
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x.reshape(B, H, W, C)


class DilateAttention(nn.Module):
    "Implementation of Dilate-attention"
    def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=3, dilation=[1,2,3,4]):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size=kernel_size
        self.unfold = nn.Unfold(kernel_size, dilation, dilation*(kernel_size-1)//2, 1)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self,q,k,v):
        #B, C//3, H, W
        B,d,H,W = q.shape
        q = q.reshape([B, d//self.head_dim, self.head_dim, 1 ,H*W]).permute(0, 1, 4, 3, 2)  # B,h,N,1,d
        k = self.unfold(k).reshape([B, d//self.head_dim, self.head_dim, self.kernel_size*self.kernel_size, H*W]).permute(0, 1, 4, 2, 3)  #B,h,N,d,k*k
        attn = (q @ k) * self.scale  # B,h,N,1,k*k
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        v = self.unfold(v).reshape([B, d//self.head_dim, self.head_dim, self.kernel_size*self.kernel_size, H*W]).permute(0, 1, 4, 3, 2)  # B,h,N,k*k,d
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, d)
        return x


class MultiDilatelocalAttention(nn.Module):
    "Implementation of Dilate-attention"

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0.,proj_drop=0., kernel_size=3, #dilation=[1, 2, 3]
                  dilation=[1,2,3,4]):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.scale = qk_scale or head_dim ** -0.5
        self.num_dilation = len(dilation)
        assert num_heads % self.num_dilation == 0, f"num_heads{num_heads} must be the times of num_dilation{self.num_dilation}!!"
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.dilate_attention = nn.ModuleList(
            [DilateAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
             for i in range(self.num_dilation)])
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()# B, C, H, W
        qkv = self.qkv(x.clone()).reshape(B, 3, self.num_dilation, C//self.num_dilation, H, W).permute(2, 1, 0, 3, 4, 5)
        #num_dilation,3,B,C//num_dilation,H,W
        x = x.reshape(B, self.num_dilation, C//self.num_dilation, H, W).permute(1, 0, 3, 4, 2 )
        # num_dilation, B, H, W, C//num_dilation
        for i in range(self.num_dilation):
            x[i] = self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2])# B, H, W,C//num_dilation
        x = x.permute(1, 2, 3, 0, 4).reshape(B, H, W, C)
        # print(x.shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        # print(x.shape)
        return x

class Residual(nn.Module):#定义残差结构
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNormDrop(nn.Module):
    def __init__(self, dim, drop_path, fn):
        super().__init__()
        norm_with_eps = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_with_eps(dim)
        self.dropout = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))

class PreNorm(nn.Module):
    def __init__(self, dim, drop_path,fn):
        super().__init__()
        norm_with_eps = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_with_eps(dim)
        self.fn = fn
        self.dropout = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x):
        # print(x.shape)
        return self.dropout(self.fn(self.norm(x)))


# class medium_layer(nn.Module):
#     def __init__(self,dim1,dim2,fn):
#         super().__init__()
#         self.downsample = Down_wt(dim1,dim2)
#         self.fn = fn
#     def forward(self,x):
#         x = x.permute(0, 2, 3, 1)
#         self.fn(x)


class DilateBlock(nn.Module):
    "Implementation of Dilate-attention block"
    def __init__(self, dim, num_heads,mlp_ratio=4., qkv_bias=True,qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1,act_layer=nn.GELU,
                 #norm_layer=nn.LayerNorm,
                 kernel_size=3,
                 # dilation=[1, 2, 3],
                 dilation=[1,2]
                 ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.kernel_size = kernel_size
        self.dilation = dilation
        # self.norm1 = norm_layer(dim)
        #self.attn = MultiDilatelocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                                        attn_drop=attn_drop, kernel_size=kernel_size, dilation=dilation)

        # self.drop_path = DropPath(
        #     drop_path) if drop_path > 0. else nn.Identity()

        # self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
        #                act_layer=act_layer, drop=drop)
        # depths = [2, 2, 6, 2]
        depths = [2, 2]
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        layers = []
        for i_layer in range(len(depths)):
            for  j_layer in range(depths[i_layer]):
                    drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])]
                    drop_path=  drop_path[j_layer] if isinstance(drop_path, list) else drop_path
                    # print(drop_path)
                    layers.extend(
                        [   #第一个Residual代表着做selfattention的多头注意力机制的运算
                            Residual(# Residual代表残差结构：fn（x）+x
                                PreNormDrop( #  PreNormDrop等于dropout(SelfAttentio(self.layernorm(x)))
                                    dim,
                                    drop_path,
                                    MultiDilatelocalAttention(self.dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                                              qk_scale=qk_scale,
                                                              attn_drop=attn_drop, kernel_size=kernel_size,
                                                              dilation=dilation),
                                )
                            ),
                            # drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                            Residual(
                                PreNorm(self.dim,
                                        drop_path,
                                        Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop))# self.mlp(self.layernorm(x))
                            ),

                        ]
                    )
        self.net = IntermediateSequential(*layers)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x = x.permute(0, 2, 3, 1)
        # x = x + self.drop_path(self.attn(self.norm1(x)))
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        # x = x.permute(0, 3, 1, 2)
        #
        # return x
        # x = x.permute(0, 2, 3, 1)#B, C, H, W
        return self.net(x)



# import torch

# # #
# # 设置设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #
# batch_size = 1
# height = 14
# width = 14
# channels = 1024
# num_heads = 8
# input_tensor = torch.randn(batch_size, channels, height, width).to(device)  # 移动输入张量到 GPU
#
# # 创建 DilateBlock 模块并移动到 GPU
# dilate_block = DilateBlock(dim=channels, num_heads=num_heads).to(device)
#
#
# # def bytes_to_gib(bytes):
# #     return bytes / (1024**3)
# #
# # from torch.profiler import profile, record_function, ProfilerActivity
# # # 使用 PyTorch Profiler 分析显存占用
# # with profile(activities=[
# #         ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
# #     with record_function("model_forward"):
# # 前向传播
#         #intermediate_outputs = dilate_block(input_tensor)
#         # for layer_name, layer_output in intermediate_outputs.items():
#         #     print(f"Layer '{layer_name}' output shape:", layer_output.shape)  # 打印每一层的输出结果形状
# output_0,output_1,output_2,output_3 = dilate_block(input_tensor)
# print(output_0.shape)


# 分析结果
# key_averages = prof.key_averages()
# max_cuda_memory_usage = max(key_averages, key=lambda x: x.cuda_memory_usage).cuda_memory_usage
# max_cuda_memory_usage_gib = bytes_to_gib(max_cuda_memory_usage)
# print(f"Max CUDA memory usage: {max_cuda_memory_usage_gib:.2f} GiB")
#
# # 打印显存占用表格 (可选)
# print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))