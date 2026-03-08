import torch.nn as nn
from einops.layers.torch import Rearrange
import torch
from nets.hiformer.encoder import All2Cross
from nets.hiformer.decoder import ConvUpsample, SegmentationHead


class HiFormer(nn.Module):
    def __init__(self, config, img_size=256, in_chans=3, n_classes=9):
        super().__init__()
        self.img_size = img_size
        self.patch_size = [4,16]
        self.n_classes = n_classes
        self.All2Cross = All2Cross(config=config, img_size=img_size, in_chans=in_chans)

        self.ConvUp_s = ConvUpsample(in_chans=384, out_chans=[128, 128], upsample=True)
        self.ConvUp_l = ConvUpsample(in_chans=96, upsample=False)

        self.segmentation_head = SegmentationHead(
            in_channels=16,
            out_channels=n_classes,
            kernel_size=3,
        )

        self.conv_pred = nn.Sequential(
            nn.Conv2d(
                128, 16,
                kernel_size=1, stride=1,
                padding=0, bias=True),
            # nn.GroupNorm(8, 16),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        xs = self.All2Cross(x)
        embeddings = [x[:, 1:] for x in xs]
        reshaped_embed = []
        for i, embed in enumerate(embeddings):
            embed = Rearrange('b (h w) d -> b d h w', h=(self.img_size // self.patch_size[i]),
                              w=(self.img_size // self.patch_size[i]))(embed)
            embed = self.ConvUp_l(embed) if i == 0 else self.ConvUp_s(embed)

            reshaped_embed.append(embed)

        C = reshaped_embed[0] + reshaped_embed[1]
        C = self.conv_pred(C)

        out = self.segmentation_head(C)

        return out


class Config:
    def __init__(self,
                 image_size=256,
                 patch_size=4,
                 num_classes=9,
                 cnn_backbone="resnet50",
                 cnn_pyramid_fm=[256, 512, 1024],
                 swin_pyramid_fm=[96, 192, 384],
                 resnet_pretrained=False,
                 depth=[[1, 2, 0]],
                 num_heads=(6, 12),
                 mlp_ratio=(2.0, 2.0, 1.0),
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.0,
                 qkv_bias=True,
                 qk_scale=None,
                 cross_pos_embed=True):
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.cnn_backbone = cnn_backbone
        self.cnn_pyramid_fm = cnn_pyramid_fm
        self.swin_pyramid_fm = swin_pyramid_fm
        self.resnet_pretrained = resnet_pretrained

        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.cross_pos_embed = cross_pos_embed

#
# config = Config(image_size=256)
# model = HiFormer(config=config, img_size=config.image_size, in_chans=3, n_classes=2)
# inputs = torch.randn(1,3,256,256)
# outputs = model(inputs)
# print(outputs.shape)


#
# # 测试性能
# import torch
# import time
# from thop import profile
# from nets.hiformer.hiformer import Config,HiFormer
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
#     config = Config(image_size=256)
#     model = HiFormer(config=config, img_size=config.image_size, in_chans=3, n_classes=2).to(device)
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
# if __name__ == '__main__':
#     measure_model()







#
#
# # 记得导入你的模型类只测fps
# # from your_model_file import BANet
# import torch
# import time
# def measure_fps_clean():
#     # ================= 配置 =================
#     INPUT_SIZE = (1, 3, 256, 256)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     # =======================================
#
#     print(f"正在进行纯净版 FPS 测试...")
#     print(f"测试设备: {device}")
#
#     # 1. 【关键】重新实例化模型
#     # 这一步是为了确保模型上没有 thop 留下的脏 hook
#     config = Config(image_size=256)
#     model = HiFormer(config=config, img_size=config.image_size, in_chans=3, n_classes=2).to(device)
#     model.eval()
#
#     # 2. 准备数据
#     input_tensor = torch.randn(*INPUT_SIZE).to(device)
#
#     # 3. 预热 (Warm up)
#     # 即使你只想测一次，预热也是必须的，否则显卡初始化时间会被算进去
#     print(" >> 正在预热 GPU (Warm up)...")
#     with torch.no_grad():
#         for _ in range(50):
#             _ = model(input_tensor)
#
#     # 4. 正式测速
#     print(" >> 开始计时 (运行 100 次取平均)...")
#     test_rounds = 100
#
#     torch.cuda.synchronize() if device.type == 'cuda' else None
#     start_time = time.time()
#
#     with torch.no_grad():
#         for _ in range(test_rounds):
#             _ = model(input_tensor)
#
#     torch.cuda.synchronize() if device.type == 'cuda' else None
#     end_time = time.time()
#
#     # 5. 计算结果
#     avg_time_sec = (end_time - start_time) / test_rounds
#     latency_ms = avg_time_sec * 1000
#     fps = 1.0 / avg_time_sec
#
#     print(f"--------------------------------")
#     print(f" Final Results:")
#     print(f" Latency: {latency_ms:.2f} ms")
#     print(f" FPS    : {fps:.2f}")
#     print(f"--------------------------------")
#
#
# if __name__ == '__main__':
#     measure_fps_clean()