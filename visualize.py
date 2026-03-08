import torch
import numpy as np
import matplotlib.pyplot as plt


#
# # 选择要可视化的神经元索引
# unit_idx = 0
#
# # 选择两个特征通道
# feature1_idx = 0
# feature2_idx = 1
#
# # 从输入数据中提取两个特征通道
# feature1 = outputs[0, :, :, feature1_idx].detach().numpy()
# feature2 = outputs[0, :, :, feature2_idx].detach().numpy()
#
# # 创建网格数据
# x1 = np.linspace(np.min(feature1), np.max(feature1), 50)
# x2 = np.linspace(np.min(feature2), np.max(feature2), 50)
# X1, X2 = np.meshgrid(x1, x2)
#
# # 将网格数据转换为输入格式
# inputs_grid = np.zeros((50, 50, inputs.shape[3]))
# inputs_grid[:, :, feature1_idx] = X1
# inputs_grid[:, :, feature2_idx] = X2
#
# # 将网格数据转换为张量
# inputs_grid = torch.from_numpy(inputs_grid).float()
#
# # 绘制热力图
# plt.imshow(outputs.reshape(X1.shape), cmap='viridis')
# plt.colorbar()
# plt.xlabel(f'Feature {feature1_idx}')
# plt.ylabel(f'Feature {feature2_idx}')
# plt.title(f'Unit Function {unit_idx}')
# plt.show()





