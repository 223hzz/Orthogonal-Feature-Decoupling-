import torch
from torch import nn
from typing import Union, List, Tuple
import math
from torch.cuda.amp import autocast
from mamba_ssm import Mamba

# 引入依赖
from dynamic_network_architectures.building_blocks.residual import BasicBlockD
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op as get_matching_conv_op
from dynamic_network_architectures.building_blocks.helper import (
    get_matching_instancenorm,
    maybe_convert_scalar_to_list
)


# --------------------------------------------------------
# 核心 UMambaBot 类 (保持不变)
# --------------------------------------------------------
class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        return out


class UpsampleLayer(nn.Module):
    def __init__(self, conv_op, input_channels, output_channels, pool_op_kernel_size, mode='nearest'):
        super().__init__()
        self.conv = conv_op(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        x = self.conv(x)
        return x


class BasicResBlock(nn.Module):
    def __init__(self, conv_op, input_channels, output_channels, norm_op, norm_op_kwargs,
                 kernel_size=3, padding=1, stride=1, use_1x1conv=False,
                 nonlin=nn.LeakyReLU, nonlin_kwargs={'inplace': True}):
        super().__init__()
        self.conv1 = conv_op(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = norm_op(output_channels, **norm_op_kwargs)
        self.act1 = nonlin(**nonlin_kwargs)
        self.conv2 = conv_op(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = norm_op(output_channels, **norm_op_kwargs)
        self.act2 = nonlin(**nonlin_kwargs)
        if use_1x1conv or stride != 1 or input_channels != output_channels:
            self.conv3 = conv_op(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)


class UNetResEncoder(nn.Module):
    def __init__(self, input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides, n_blocks_per_stage,
                 conv_bias=False, norm_op=None, norm_op_kwargs=None, nonlin=None, nonlin_kwargs=None,
                 return_skips=False, stem_channels=None, pool_type='conv'):
        super().__init__()
        if isinstance(kernel_sizes, int): kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int): features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int): n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int): strides = [strides] * n_stages

        self.conv_pad_sizes = []
        for krnl in kernel_sizes:
            self.conv_pad_sizes.append([i // 2 for i in krnl] if isinstance(krnl, (list, tuple)) else krnl // 2)

        stem_channels = features_per_stage[0]
        self.stem = nn.Sequential(
            BasicResBlock(conv_op, input_channels, stem_channels, norm_op, norm_op_kwargs, kernel_sizes[0],
                          self.conv_pad_sizes[0], stride=1, nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
                          use_1x1conv=True),
            *[BasicBlockD(conv_op, stem_channels, stem_channels, kernel_sizes[0], 1, conv_bias, norm_op, norm_op_kwargs,
                          nonlin, nonlin_kwargs) for _ in range(n_blocks_per_stage[0] - 1)]
        )
        input_channels = stem_channels
        stages = []
        for s in range(n_stages):
            stages.append(nn.Sequential(
                BasicResBlock(conv_op, input_channels, features_per_stage[s], norm_op, norm_op_kwargs, kernel_sizes[s],
                              self.conv_pad_sizes[s], strides[s], use_1x1conv=True, nonlin=nonlin,
                              nonlin_kwargs=nonlin_kwargs),
                *[BasicBlockD(conv_op, features_per_stage[s], features_per_stage[s], kernel_sizes[s], 1, conv_bias,
                              norm_op, norm_op_kwargs, nonlin, nonlin_kwargs) for _ in range(n_blocks_per_stage[s] - 1)]
            ))
            input_channels = features_per_stage[s]
        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        if self.stem is not None: x = self.stem(x)
        ret = []
        for s in self.stages:
            x = s(x)
            ret.append(x)
        return ret if self.return_skips else ret[-1]


class UNetResDecoder(nn.Module):
    def __init__(self, encoder, num_classes, n_conv_per_stage, deep_supervision, nonlin_first=False):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int): n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        stages, upsample_layers, seg_layers = [], [], []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_upsampling = encoder.strides[-s]
            upsample_layers.append(
                UpsampleLayer(encoder.conv_op, input_features_below, input_features_skip, stride_for_upsampling))
            stages.append(nn.Sequential(
                BasicResBlock(encoder.conv_op,
                              2 * input_features_skip if s < n_stages_encoder - 1 else input_features_skip,
                              input_features_skip, encoder.norm_op, encoder.norm_op_kwargs,
                              kernel_size=encoder.kernel_sizes[-(s + 1)], padding=encoder.conv_pad_sizes[-(s + 1)],
                              stride=1, use_1x1conv=True, nonlin=encoder.nonlin, nonlin_kwargs=encoder.nonlin_kwargs),
                *[BasicBlockD(encoder.conv_op, input_features_skip, input_features_skip, encoder.kernel_sizes[-(s + 1)],
                              1, encoder.conv_bias, encoder.norm_op, encoder.norm_op_kwargs, encoder.nonlin,
                              encoder.nonlin_kwargs) for _ in range(n_conv_per_stage[s - 1] - 1)]
            ))
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))
        self.stages, self.upsample_layers, self.seg_layers = nn.ModuleList(stages), nn.ModuleList(
            upsample_layers), nn.ModuleList(seg_layers)

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.upsample_layers[s](lres_input)
            if s < (len(self.stages) - 1): x = torch.cat((x, skips[-(s + 2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x
        seg_outputs = seg_outputs[::-1]
        return seg_outputs if self.deep_supervision else seg_outputs[0]


class UMambaBot(nn.Module):
    def __init__(self, input_channels, n_stages, features_per_stage, kernel_sizes, strides, n_conv_per_stage,
                 num_classes, n_conv_per_stage_decoder, conv_op=nn.Conv2d, conv_bias=False, norm_op=None,
                 norm_op_kwargs=None, deep_supervision=False, stem_channels=None):
        super().__init__()
        if norm_op is None: norm_op = get_matching_instancenorm(conv_op)
        if norm_op_kwargs is None: norm_op_kwargs = {'eps': 1e-5, 'affine': True}

        # Mamba 替换配置
        n_blocks_per_stage = [x for x in n_conv_per_stage]
        n_decoder = [x for x in n_conv_per_stage_decoder]
        for s in range(math.ceil(n_stages / 2), n_stages): n_blocks_per_stage[s] = 1
        for s in range(math.ceil((n_stages - 1) / 2 + 0.5), n_stages - 1): n_decoder[s] = 1

        self.encoder = UNetResEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                      n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, nn.LeakyReLU,
                                      {'inplace': True}, return_skips=True, stem_channels=stem_channels)
        self.mamba_layer = MambaLayer(dim=features_per_stage[-1])
        self.decoder = UNetResDecoder(self.encoder, num_classes, n_decoder, deep_supervision)
        self.apply(InitWeights_He(1e-2))

    def forward(self, x):
        skips = self.encoder(x)
        skips[-1] = self.mamba_layer(skips[-1])
        return self.decoder(skips)


# ==============================================================================
# 5. 【重点！】傻瓜式调用助手函数 (复制这个！)
# ==============================================================================

def get_umamba_model(input_channels: int,
                     num_classes: int,
                     dim: int = 2,  # 设置为 3 就是 3D模型，2 就是 2D模型
                     ):
    """
    只需输入通道数和类别数，自动配置 nnU-Net 黄金标准参数。
    """

    # 1. 设置卷积类型 (2D 或 3D)
    if dim == 2:
        conv_op = nn.Conv2d
        kernel_size = [3, 3]
        strides_base = [2, 2]  # 每次下采样2倍
    elif dim == 3:
        conv_op = nn.Conv3d
        kernel_size = [3, 3, 3]
        strides_base = [2, 2, 2]
    else:
        raise ValueError("dim must be 2 or 3")

    # 2. 定义 nnU-Net 标准 6 层架构
    n_stages = 6

    # 特征数: 从32开始，每层翻倍，直到512封顶
    # [32, 64, 128, 256, 512, 512]
    features_per_stage = [min(32 * (2 ** i), 512) for i in range(n_stages)]

    # Stride (步长): 第一层通常不降采样，后面每层降采样
    strides = [[1] * dim] + [strides_base] * (n_stages - 1)

    # Kernel sizes: 全部使用 3x3 (或 3x3x3)
    kernel_sizes = [kernel_size] * n_stages

    # 每个阶段的卷积块数量: 编码器每层2块，解码器每层2块
    n_conv_per_stage = [2] * n_stages
    n_conv_per_stage_decoder = [2] * (n_stages - 1)

    # 3. 实例化模型
    model = UMambaBot(
        input_channels=input_channels,
        n_stages=n_stages,
        features_per_stage=features_per_stage,
        kernel_sizes=kernel_sizes,
        strides=strides,
        n_conv_per_stage=n_conv_per_stage,
        num_classes=num_classes,
        n_conv_per_stage_decoder=n_conv_per_stage_decoder,
        conv_op=conv_op,
        deep_supervision=False
    )

    return model

#
# # ==============================================================================
# # 使用演示
# # ==============================================================================
#
# if __name__ == "__main__":
#     # 场景 1: 我想跑一个 2D 分割，输入是灰度图(1通道)，分割2类(背景+前景)
#     model_2d = get_umamba_model(input_channels=1, num_classes=2, dim=2)
#     print("2D 模型构建成功！")
#
#     # 场景 2: 我想跑一个 3D 分割，输入是CT(1通道)，分割4个器官(4类)
#     model_3d = get_umamba_model(input_channels=1, num_classes=4, dim=3)
#     print("3D 模型构建成功！")
#
#     # 简单测试一下
#     input_tensor = torch.randn(1, 1, 128, 128, 128).cuda()
#     model_3d.cuda()
#     output = model_3d(input_tensor)
#     # 因为开启了 deep_supervision，输出是一个列表
#     print("输出形状:", output[0].shape)  # 应该是 [1, 4, 128, 128, 128]





#
# # 测试性能
import torch
import time
from thop import profile


# 确保你的模型类引入是对的，这里假设是 A2FPN
# from your_file import A2FPN

# ================= 配置区域 =================
INPUT_SIZE = (1, 3, 256, 256)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ===========================================

def measure_model():
    print(f"正在测试模型效率 (极简模式：无预热、单次运行)...")
    print(f"测试设备: {device}")

    # 1. 实例化模型
    # model = A2FPN(class_num=2).to(device)  <-- 你的模型类名
    # 这里为了演示，我用你的变量名，请确保类名正确
    model =  get_umamba_model(input_channels=3, num_classes=2, dim=2).to(device)
    model.eval()

    # 2. 创建虚拟输入
    input_tensor = torch.randn(*INPUT_SIZE).to(device)

    # ---------------------------------------------------
    # 3. 计算 FLOPs 和 Params
    # ---------------------------------------------------
    print("\n[1/2] 正在计算 FLOPs 和 Params ...")
    try:
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        print(f" >> Params (参数量): {params / 1e6:.4f} M")
        print(f" >> FLOPs  (计算量): {flops / 1e9:.4f} G")
    except Exception as e:
        print(f"thop计算出错: {e}")

    # ===================================================
    # 【关键修正】重新实例化模型！
    # 必须这样做，否则 thop 留下的钩子会导致后面测速报错
    # ===================================================
    print(" >> 重置模型以清除 thop 钩子...")
    model =get_umamba_model(input_channels=3, num_classes=2, dim=2).to(device)
    model.eval()

    # ---------------------------------------------------
    # 4. 计算 Inference Time (绝对单次，无预热)
    # ---------------------------------------------------
    print("\n[2/2] 正在计算推理速度 (单次, 无预热) ...")

    # 这里的预热已经被我删掉了，直接开始

    # === 正式开始计时 (只运行 1 次) ===
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()

    with torch.no_grad():
        _ = model(input_tensor)  # 只跑一次

    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()

    # 计算结果
    total_time_seconds = end_time - start_time
    latency_ms = total_time_seconds * 1000
    fps = 1.0 / total_time_seconds if total_time_seconds > 0 else 0

    print(f" >> Latency (延迟): {latency_ms:.2f} ms")
    print(f" >> FPS (帧率): {fps:.2f}")


if __name__ == '__main__':
    measure_model()