import torch
import torch.nn as nn
from nets.dual_DCA.main_block import conv_block, double_conv_block_a, double_conv_block, Upconv, params
from nets.dual_DCA.dca import DCA
# from torchsummary import summary

class dca_Unet(nn.Module):
    def __init__(self,
                 attention=True,
                 n=1,
                 in_features=3,
                 out_features=2,
                 k=0.5,
                 input_size=(1024, 1024),#改动这个
                 patch_size=8,
                 spatial_att=True,
                 channel_att=True,
                 spatial_head_dim=[4, 4, 4, 4],
                 channel_head_dim=[1, 1, 1, 1],
                 device='cuda',
                 ) -> None:
        super().__init__()
        # if device == 'cuda':
        #     torch.cuda.set_enabled_lms(True)

        self.attention = attention
        patch = input_size[0] // patch_size

        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.relu = nn.ReLU()
        norm2 = None
        self.conv1 = double_conv_block_a(in_features=in_features,
                                         out_features1=int(64 * k),
                                         out_features2=int(64 * k),
                                         norm1='bn',
                                         norm2=norm2,
                                         act1=True,
                                         act2=False)
        self.norm1 = nn.BatchNorm2d(int(64 * k))
        self.conv2 = double_conv_block_a(in_features=int(64 * k),
                                         out_features1=int(128 * k),
                                         out_features2=int(128 * k),
                                         norm1='bn',
                                         norm2=norm2,
                                         act1=True,
                                         act2=False)
        self.norm2 = nn.BatchNorm2d(int(128 * k))

        self.conv3 = double_conv_block_a(in_features=int(128 * k),
                                         out_features1=int(256 * k),
                                         out_features2=int(256 * k),
                                         norm1='bn',
                                         norm2=norm2,
                                         act1=True,
                                         act2=False)
        self.norm3 = nn.BatchNorm2d(int(256 * k))

        self.conv4 = double_conv_block_a(in_features=int(256 * k),
                                         out_features1=int(512 * k),
                                         out_features2=int(512 * k),
                                         norm1='bn',
                                         norm2=norm2,
                                         act1=True,
                                         act2=False)
        self.norm4 = nn.BatchNorm2d(int(512 * k))

        self.conv5 = double_conv_block(in_features=int(512 * k),
                                       out_features1=int(1024 * k),
                                       out_features2=int(1024 * k),
                                       norm_type='bn')

        if self.attention:
            self.DCA = DCA(n=n,
                           features=[int(64 * k), int(128 * k), int(256 * k), int(512 * k)],
                           strides=[patch_size, patch_size // 2, patch_size // 4, patch_size // 8],
                           patch=patch,
                           spatial_att=spatial_att,
                           channel_att=channel_att,
                           spatial_head=spatial_head_dim,
                           channel_head=channel_head_dim,
                           )

        self.up1 = Upconv(in_features=int(1024 * k),
                          out_features=int(512 * k),
                          norm_type='bn')

        self.upconv1 = double_conv_block(in_features=int(512 * k + 512 * k),
                                         out_features1=int(512 * k),
                                         out_features2=int(512 * k),
                                         norm_type='bn')

        self.up2 = Upconv(in_features=int(512 * k),
                          out_features=int(256 * k),
                          norm_type='bn')

        self.upconv2 = double_conv_block(in_features=int(256 * k + 256 * k),
                                         out_features1=int(256 * k),
                                         out_features2=int(256 * k),
                                         norm_type='bn')

        self.up3 = Upconv(in_features=int(256 * k),
                          out_features=int(128 * k),
                          norm_type='bn')

        self.upconv3 = double_conv_block(in_features=int(128 * k + 128 * k),
                                         out_features1=int(128 * k),
                                         out_features2=int(128 * k),
                                         norm_type='bn')

        self.up4 = Upconv(in_features=int(128 * k),
                          out_features=int(64 * k),
                          norm_type='bn')

        self.upconv4 = double_conv_block(in_features=int(64 * k + 64 * k),
                                         out_features1=int(64 * k),
                                         out_features2=int(64 * k),
                                         norm_type='bn')

        self.out = conv_block(in_features=int(64 * k),
                              out_features=out_features,
                              norm_type=None,
                              activation=False,
                              kernel_size=(1, 1),
                              padding=(0, 0))

        # self.initialize_weights()

    def forward(self, x):
        x1 = self.conv1(x)
        x1_n = self.norm1(x1)
        x1_a = self.relu(x1_n)
        # print("x1.shape:",x1.shape)        #[1, 32, 256, 256]
        x2 = self.maxpool(x1_a)
        # print("x2.0.shape:", x2.shape)     #[1, 32, 128, 128]
        x2 = self.conv2(x2)
        x2_n = self.norm2(x2)
        x2_a = self.relu(x2_n)
        # print("x2.1.shape:", x2.shape)     #[1, 64, 128, 128]
        x3 = self.maxpool(x2_a)
        # print("x3.shape:", x3.shape)       #[1, 64, 64, 64]
        x3 = self.conv3(x3)
        x3_n = self.norm3(x3)
        x3_a = self.relu(x3_n)
        # print("x3.shape:", x3.shape)        #[1, 128, 64, 64]
        x4 = self.maxpool(x3_a)
        # print(x4.shape)                       #[1, 128, 32, 32]
        x4 = self.conv4(x4)
        # print(x4.shape)                      #[1, 256, 32, 32]
        x4_n = self.norm4(x4)
        x4_a = self.relu(x4_n)
        x5 = self.maxpool(x4_a)
        # print(x5.shape)                         #[1, 256, 16, 16]
        x = self.conv5(x5)
        # print(x.shape)                         #[1, 512, 16, 16]
        if self.attention:
            x1, x2, x3, x4 = self.DCA([x1, x2, x3, x4])
        x = self.up1(x)
        x = torch.cat((x, x4), dim=1)
        x = self.upconv1(x)
        x = self.up2(x)
        x = torch.cat((x, x3), dim=1)
        x = self.upconv2(x)
        x = self.up3(x)
        x = torch.cat((x, x2), dim=1)
        x = self.upconv3(x)
        x = self.up4(x)
        x = torch.cat((x, x1), dim=1)
        x = self.upconv4(x)
        x = self.out(x)
        return x



# import torch
# from fvcore.nn import FlopCountAnalysis
# from fvcore.nn import flop_count_table
#
# model = dca_Unet()
# model.eval()
# flops = FlopCountAnalysis(model, torch.rand(1, 3, 256, 256))
# print(flop_count_table(flops))

# #
# if __name__ == '__main__':
#     model = Unet()
#     in1 = torch.rand(1, 3, 256, 256)
#     out = model(in1)
#     total_params = sum(p.numel() for p in model.parameters())
#     print(total_params)
#     print(out.shape)



# if __name__ == '__main__':
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model =  Unet().to(device)
#     device = torch.device('cuda')
#     model.to(device)
#     dump_input = torch.ones(1,3,256,256).to(device)
#     output = model(dump_input)
#     gpu_memory_allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
#     gpu_memory_cached = torch.cuda.memory_cached(device) / (1024 * 1024)
#
#     print(f"已分配 GPU 内存: {gpu_memory_allocated:.2f} MB")
#     print(f"缓存的 GPU 内存: {gpu_memory_cached:.2f} MB")



#
# if __name__ == "__main__":
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     m = Unet().to(device)
#     summary(m,(3, 256, 256))








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
#     model = dca_Unet(out_features=2,  input_size=(256, 256)).to(device)
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