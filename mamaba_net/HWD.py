import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward


class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x

#
# # 输入 N C H W,  输出 N C H W
# if __name__ == '__main__':
#     block = Down_wt(3, 64)  # 输入通道数，输出通道数
#     input = torch.rand(3, 3, 64, 64)
#     output = block(input)
#     print(output.shape)







#
# #测试模块1
# import cv2
# import torch
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# import torch.nn as nn
# from pytorch_wavelets import DWTForward
#
#
# class StridedConvolution(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, is_relu=True):
#         super(StridedConvolution, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)
#         self.relu = nn.ReLU(inplace=True)
#         self.is_relu = is_relu
#
#     def forward(self, x):
#         x = self.conv(x)
#         if self.is_relu:
#             x = self.relu(x)
#         return x
#
#
# class HWDownsampling(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super(HWDownsampling, self).__init__()
#         self.wt = DWTForward(J=1, wave='haar', mode='zero')
#         self.conv_bn_relu = nn.Sequential(
#             nn.Conv2d(in_channel * 4, out_channel, kernel_size=1, stride=1),
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU(inplace=True),
#         )
#
#     def forward(self, x):
#         yL, yH = self.wt(x)
#         y_HL = yH[0][:, :, 0, ::]
#         y_LH = yH[0][:, :, 1, ::]
#         y_HH = yH[0][:, :, 2, ::]
#         x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
#         x = self.conv_bn_relu(x)
#         return x
#
#
# class DeeperCNN(nn.Module):
#     def __init__(self):
#         super(DeeperCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
#         self.batch_norm1 = nn.BatchNorm2d(16)
#         self.relu = nn.ReLU()
#         self.pool1_1= nn.MaxPool2d(kernel_size=2, stride=2)
#         self.pool1_2 = nn.AvgPool2d(kernel_size=2, stride=2)
#         self.pool1_3 = HWDownsampling(16, 16)
#         self.pool1_4 = StridedConvolution(16, 16, is_relu=True)
#
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
#         self.batch_norm2 = nn.BatchNorm2d(32)
#         self.pool2_1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.pool2_2 = nn.AvgPool2d(kernel_size=2, stride=2)
#         self.pool2_3 = HWDownsampling(32, 32)
#         self.pool2_4 = StridedConvolution(32, 32, is_relu=True)
#
#         self.conv6 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
#
#     def forward(self, x):
#         x1 = self.pool1_1(self.relu(self.batch_norm1(self.conv1(x))))
#         x2 = self.pool1_2(self.relu(self.batch_norm1(self.conv1(x))))
#         x3 = self.pool1_3(self.relu(self.batch_norm1(self.conv1(x))))
#         x4 = self.pool1_4(self.relu(self.batch_norm1(self.conv1(x))))
#         print(x1.shape)
#         y1 = self.pool2_1(self.relu(self.batch_norm2(self.conv2(x1))))
#         y2 = self.pool2_2(self.relu(self.batch_norm2(self.conv2(x2))))
#         y3 = self.pool2_3(self.relu(self.batch_norm2(self.conv2(x3))))
#         y4 = self.pool2_4(self.relu(self.batch_norm2(self.conv2(x4))))
#         print(y1.shape)
#         y1 = self.conv6(y1)
#         y2 = self.conv6(y2)
#         y3 = self.conv6(y3)
#         y4 = self.conv6(y4)
#         print(x1.shape)
#         print(y1.shape)
#         return  y1, y2, y3, y4
#
#
# image_path = r'D:\py.SETR-transformer\setr-pytorch-main\setr\Birmingham_img_0_0.png'
# image = cv2.imread(image_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# transform = transforms.Compose([transforms.ToTensor()])
# input_image = transform(image).unsqueeze(0)
# import numpy as np
#
# model = DeeperCNN()
#
# y1, y2, y3, y4 = model(input_image)
#
# plt.figure(figsize=(16, 8))  # 调整画布大小
# input_image = input_image.squeeze(0).permute(1, 2, 0).numpy()
# # 左侧绘制 x1、x2、x3、x4 图像
# plt.subplot(2, 5, 1)
# plt.imshow(input_image)
# plt.title('Input Image')
#
# for i, tensor in enumerate([y1, y2, y3, y4], start=1):
#     # 将张量转换为 numpy 数组，并压缩批处理维度（如果存在）
#     image_np = tensor.squeeze().detach().numpy()
#     # 将 image_np 重塑为三维张量以匹配 y1 的形状
#     image_np = image_np.reshape((256, 256, 1))
#     # 归一化处理和限制值范围
#     image_np = image_np / image_np.max()
#     image_np = np.clip(image_np, 0, 1)
#     # 绘制图像
#     plt.subplot(2, 5, i + 1)
#     plt.imshow(image_np.squeeze())  # 显示灰度图像
#     plt.title(f'y{i}')
# plt.tight_layout()  # 调整子图之间的间距
# plt.show()


#
#测试模块2
# import torch
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# import torch.nn as nn
# from pytorch_wavelets import DWTForward
# from 局部放大显示功能 import on_EVENT_LBUTTONDOWN
# import cv2
# class StridedConvolution(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, is_relu=True):
#         super(StridedConvolution, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)
#         self.relu = nn.ReLU(inplace=True)
#         self.is_relu = is_relu
#
#     def forward(self, x):
#         x = self.conv(x)
#         if self.is_relu:
#             x = self.relu(x)
#         return x
#
#
# class HWDownsampling(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super(HWDownsampling, self).__init__()
#         self.wt = DWTForward(J=1, wave='haar', mode='zero')
#         self.conv_bn_relu = nn.Sequential(
#             nn.Conv2d(in_channel * 4, out_channel, kernel_size=1, stride=1),
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU(inplace=True),
#         )
#
#     def forward(self, x):
#         yL, yH = self.wt(x)
#         y_HL = yH[0][:, :, 0, ::]
#         y_LH = yH[0][:, :, 1, ::]
#         y_HH = yH[0][:, :, 2, ::]
#         x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
#         x = self.conv_bn_relu(x)
#         return x
#
#
# class DeeperCNN(nn.Module):
#     def __init__(self):
#         super(DeeperCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
#         self.batch_norm1 = nn.BatchNorm2d(16)
#         self.relu = nn.ReLU()
#         # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         # self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
#         self.pool1 = HWDownsampling(16, 16)
#         # self.pool1 = StridedConvolution(16, 16, is_relu=True)
#
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
#         self.batch_norm2 = nn.BatchNorm2d(32)
#         # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         # self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
#         self.pool2 = HWDownsampling(32, 32)
#         # self.pool2 = StridedConvolution(32, 32, is_relu=True)
#
#         self.conv6 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
#
#     def forward(self, x):
#         x = self.pool1(self.relu(self.batch_norm1(self.conv1(x))))
#         print(x.shape)
#         x = self.pool2(self.relu(self.batch_norm2(self.conv2(x))))
#         print(x.shape)
#         x = self.conv6(x)
#         return x
#
#
# image_path = r'D:\py.SETR-transformer\setr-pytorch-main\setr\Birmingham_img_0_0.png'
# image = cv2.imread(image_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# transform = transforms.Compose([transforms.ToTensor()])
# input_image = transform(image).unsqueeze(0)
# import numpy as np
#
# model = DeeperCNN()
# output = model(input_image)
# print("Output shape:", output.shape)
#
# input_image = input_image.squeeze(0).permute(1, 2, 0).numpy()
# output_image = output.squeeze(0).permute(1, 2, 0).detach().numpy()
# output_image = output_image / output_image.max()
# output_image = np.clip(output_image, 0, 1)
#
# plt.subplot(1, 2, 1)
# plt.imshow(input_image)
# plt.title('Input Image')
#
# plt.subplot(1, 2, 2)
# plt.imshow(output_image)
# plt.title('Output Image')
#
# plt.show()


