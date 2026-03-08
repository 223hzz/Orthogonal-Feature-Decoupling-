# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.segformer.backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5


class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        super(SegFormerHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            c1=embedding_dim * 4,
            c2=embedding_dim,
            k=1,
        )

        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x


class SegFormer(nn.Module):
    def __init__(self, num_classes=21, phi='b2', pretrained=False):
        super(SegFormer, self).__init__()
        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
        }[phi]
        self.backbone = {
            'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
        }[phi](pretrained)
        self.embedding_dim = {
            'b0': 256, 'b1': 256, 'b2': 768,
            'b3': 768, 'b4': 768, 'b5': 768,
        }[phi]
        self.decode_head = SegFormerHead(num_classes, self.in_channels, self.embedding_dim)

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)

        x = self.backbone.forward(inputs)
        x = self.decode_head.forward(x)

        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x


#
# #测试性能
# import torch
# import time
# from thop import profile
#
# # 假设你的 BANet 类就在这个文件里，或者已经 import 进来了
# # from your_model_file import BANet
#
# # ================= 配置区域 =================
# INPUT_SIZE = (1, 3, 256, 256)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#
# # ===========================================
#
# def measure_model():
#     print(f"正在测试 BANet 模型效率 (单次运行模式)...")
#     print(f"测试设备: {device}")
#
#     # 1. 实例化模型
#     model = SegFormer(num_classes=2).to(device)
#     model.eval()
#
#     # 2. 创建虚拟输入
#     input_tensor = torch.randn(*INPUT_SIZE).to(device)
#
#     # ---------------------------------------------------
#     # 3. 计算 FLOPs 和 Params (这一步通常还是很快的)
#     # ---------------------------------------------------
#     print("\n[1/2] 正在计算 FLOPs 和 参数量 ...")
#     try:
#         flops, params = profile(model, inputs=(input_tensor,), verbose=False)
#         print(f" >> Params (参数量): {params / 1e6:.4f} M")
#         print(f" >> FLOPs  (计算量): {flops / 1e9:.4f} G")
#     except Exception as e:
#         print(f"thop计算出错: {e}")
#
#     # ---------------------------------------------------
#     # 4. 计算 Inference Time (只跑一次)
#     # ---------------------------------------------------
#     print("\n[2/2] 正在计算推理速度 (单次运行) ...")
#
#     # 【注意】预热 (Warm up) 还是建议保留
#     # 如果完全没有预热，第一次运行会包含 GPU 初始化和内存分配的时间，
#     # 导致测出来的时间比实际慢很多。如果你连预热都不想要，把下面两行注释掉即可。
#     with torch.no_grad():
#         _ = model(input_tensor)
#
#         # === 正式开始计时 (只运行 1 次) ===
#     torch.cuda.synchronize() if device.type == 'cuda' else None
#     start_time = time.time()
#
#     with torch.no_grad():
#         _ = model(input_tensor)  # 只跑一次
#
#     torch.cuda.synchronize() if device.type == 'cuda' else None
#     end_time = time.time()
#
#     # 计算结果
#     total_time_seconds = end_time - start_time
#     latency_ms = total_time_seconds * 1000
#     fps = 1.0 / total_time_seconds
#
#     print(f" >> Latency (延迟): {latency_ms:.2f} ms")
#     print(f" >> FPS (帧率): {fps:.2f}")
#
#
# if __name__ == '__main__':
#     measure_model()