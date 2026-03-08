import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary
# class PoolingAttention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
#                  pool_ratios=[1, 2, 3, 4]):
#         super().__init__()
#         assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
#
#         self.dim = dim
#         self.num_heads = num_heads
#         self.num_elements = np.array([t * t for t in pool_ratios]).sum()
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#
#         self.q = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias))
#         self.kv = nn.Sequential(nn.Linear(dim, dim * 2, bias=qkv_bias))
#
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#         self.pool_ratios = pool_ratios
#         self.pools = nn.ModuleList()
#
#         self.norm = nn.LayerNorm(dim)
#
#     def forward(self, x, H = 7, W = 7, d_convs=nn.ModuleList(
#             [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512) for temp
#              in [1, 2, 3, 4]])):
#         B, N, C = x.shape
#
#         q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
#         pools = []
#         x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
#         for (pool_ratio, l) in zip(self.pool_ratios, d_convs):
#             pool = F.adaptive_avg_pool2d(x_, (round(H / pool_ratio), round(W / pool_ratio)))
#             # print("pool.shape:",pool.shape)
#             pool = pool + l(pool)  # fix backward bug in higher torch versions when training
#             pools.append(pool.view(B, C, -1))
#         pools = torch.cat(pools, dim=2)
#         print("pool.shape1:", pools.shape)
#         pools = self.norm(pools.permute(0, 2, 1))
#         print("pool.shape2:", pools.shape)
#         kv = self.kv(pools).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         print("kv.shape:",kv.shape)
#         k, v = kv[0], kv[1]
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         print("attn.shape:",attn.shape)
#         attn = attn.softmax(dim=-1)
#         x = (attn @ v)
#         x = x.transpose(1, 2).contiguous().reshape(B, N, C)
#
#         x = self.proj(x)
#
#         return x
#
# model = PoolingAttention(512)
# inputs = torch.randn(1,49,512)
# outputs = model(inputs)
# print(outputs.shape)


#
# class GlobalAttention(nn.Module):
#     "Implementation of self-attention"
#
#     def __init__(self, dim, num_heads=12, qkv_bias=False,
#                   attn_drop=0., proj_drop=0.,cpe_per_block=True):
#         super().__init__()
#         self.num_heads = num_heads
#         self.pool_ratios = [1, 2, 3]
#         self.v = nn.Linear(dim, dim, bias=qkv_bias)
#         self.qk = nn.Sequential(nn.Linear(dim, dim * 2, bias=qkv_bias))
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.cpe_per_block = cpe_per_block
#         self.d_convs = nn.ModuleList(
#             [nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim) for temp
#              in self.pool_ratios])
#     def get_index(self, seq_len):
#         index = np.pi / 2 * torch.arange(1, seq_len + 1).reshape(1, -1, 1)
#         # 生成一个形状为 (1, seq_len, 1) 的张量，其中每个元素的值为 $\pi / 2$ 乘以相应的索引值
#         return nn.Parameter(index, requires_grad=False)
#
#     def forward(self, x):
#         B, H, W, C = x.shape
#         # qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads,
#         #                           C // self.num_heads).permute(2, 0, 3, 1, 4)
#         v = self.v(x).reshape(B*self.num_heads, H*W ,
#                                   C // self.num_heads)
#         pools = []
#         for (pool_ratio, l) in zip(self.pool_ratios,self.d_convs):
#             pool = F.adaptive_avg_pool2d(x.reshape((B, C, H, W)), (round(H / pool_ratio), round(W / pool_ratio)))
#             # print("pool.shape:",pool.shape)
#             pool = pool + l(pool)  # fix backward bug in higher torch versions when training
#             pools.append(pool.view(B, C, -1))
#         pools = torch.cat(pools, dim=2).permute(0, 2, 1)
#         qk = self.qk(pools).reshape(B*self.num_heads, -1, 2, C // self.num_heads).permute(2, 0, 1, 3)
#         q, k = qk[0], qk[1]
#         q = F.relu(q)
#         k = F.relu(k)
#         # print("q.shape[1]",q.shape[1])
#         tgt_len = q.shape[1]
#         if self.cpe_per_block:
#             weight_index = self.get_index(tgt_len).to(q)
#             # print("weight_index.sahpe",weight_index.shape)
#         # attn = (q @ k.transpose(-2, -1)) * self.scale
#         # attn = attn.softmax(dim=-1)
#         # attn = self.attn_drop(attn)
#         #
#         # x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
#         # (N * h, L, 2 * d)
#         q_ = torch.cat(
#             [q * torch.sin(weight_index[:, :tgt_len, :] / tgt_len), q * torch.cos(weight_index[:, :tgt_len, :] / tgt_len)], dim=-1)
#         # (N * h, S, 2 * d)
#         k_ = torch.cat(
#             [k * torch.sin(weight_index[:, :tgt_len, :] / tgt_len), k * torch.cos(weight_index[:, :tgt_len, :] / tgt_len)], dim=-1)
#         # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, 2 * d, d)
#         kv_ = k.transpose(-2, -1) @ v
#         # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
#         z_ = 1 / torch.clamp_min(torch.einsum('nld,nd->nl', q_, torch.sum(k_, axis=1)), 1e-6)
#         # (N * h, L, 2 * d) (N * h, d, 2 * d) (N * h, L) -> (N * h, L, d)
#         attn_output = torch.einsum('nld,ndm,nl->nlm', q_, kv_, z_).contiguous().view(B, H, W, -1)
#         x = self.proj(attn_output)
#         x = self.proj_drop(x)
#         return x
#
# model = GlobalAttention(576)
# inputs = torch.randn(1, 8, 8, 576)
# outputs = model(inputs)
# print(outputs.shape)






#
# import os
# import rasterio
# from rasterio.windows import Window
#
#
# def crop_images(input_folder, output_folder, window_size=256, stride=256):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     for root, dirs, files in os.walk(input_folder):
#         for file in files:
#             if file.endswith('.tif') or file.endswith('.tiff'):  # 假设影像为TIFF格式，根据实际情况修改
#                 img_path = os.path.join(root, file)
#                 label_path = img_path.replace('data', 'label')
#                 if not os.path.exists(label_path):
#                     # 尝试更换文件后缀名为.tif
#                     label_path = label_path.replace('.tiff', '.tif')
#                 # 获取对应的label路径
#                 output_img_folder = root.replace(input_folder, output_folder)
#                 output_label_folder = output_img_folder.replace('data', 'label')
#
#                 if not os.path.exists(output_img_folder):
#                     os.makedirs(output_img_folder)
#                 if not os.path.exists(output_label_folder):
#                     os.makedirs(output_label_folder)
#
#                 with rasterio.open(img_path) as src_img:
#                     with rasterio.open(label_path) as src_label:
#                         width = src_img.width
#                         height = src_img.height
#
#                         for y in range(0, height - window_size + 1, stride):
#                             for x in range(0, width - window_size + 1, stride):
#                                 window = Window(x, y, window_size, window_size)
#
#                                 img_crop = src_img.read(window=window)
#                                 label_crop = src_label.read(window=window)
#
#                                 # 如果影像中含有目标并且不含nodata空值，则保存裁剪后的影像和label
#                                 if label_crop.any() and not (img_crop == 255).any():
#                                     output_img_path = os.path.join(output_img_folder,
#                                                                    f"{file.split('.')[0]}_{x}_{y}.tif")
#                                     output_label_path = os.path.join(output_label_folder,
#                                                                      f"{file.split('.')[0]}_{x}_{y}.tif")
#
#                                     profile = src_img.profile
#                                     profile['width'], profile['height'] = window_size, window_size
#
#                                     with rasterio.open(output_img_path, 'w', **profile) as dst_img:
#                                         dst_img.write(img_crop)
#
#                                     profile_label = src_label.profile
#                                     profile_label['width'], profile_label['height'] = window_size, window_size
#
#                                     with rasterio.open(output_label_path, 'w', **profile_label) as dst_label:
#                                         dst_label.write(label_crop)
#
#
# # 调用函数对数据集进行裁剪
# train_folder = './马萨诸塞州数据集/train'
# val_folder = './马萨诸塞州数据集/valid'
# test_folder = './马萨诸塞州数据集/test'
#
# crop_images(train_folder, 'train_cropped', window_size=256, stride=256)
# crop_images(val_folder, 'val_cropped', window_size=256, stride=256)
# crop_images(test_folder, 'test_cropped', window_size=256, stride=256)
