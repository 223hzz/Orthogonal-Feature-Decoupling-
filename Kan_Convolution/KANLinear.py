import torch
import torch.nn.functional as F
import math

"""
项目讲解链接：https://www.bilibili.com/video/BV1Nw4m1D75E/?spm_id_from=333.999.0.0&vd_source=6528929ff3772e61e9f5baf9b8ab1e64
1.内存效率提升：原始实现需要扩展所有中间变量来执行不同的激活函数，而此代码中将计算重新制定为使用不同的基函数激活输入，
  然后线性组合它们。这种重新制定可以显著降低内存成本，并将计算变得更加高效。

2.正则化方法的改变：原始实现中使用的L1正则化需要对张量进行非线性操作，与重新制定的计算不兼容。
  因此，此代码中将L1正则化改为对权重的L1正则化，这更符合神经网络中常见的正则化方法，并且与重新制定的计算兼容。

3.激活函数缩放选项：原始实现中包括了每个激活函数的可学习缩放，但这个库提供了一个选项来禁用这个特性。
  禁用缩放可以使模型更加高效，但可能会影响结果。

4.参数初始化的改变：为了解决在MNIST数据集上的性能问题，此代码修改了参数的初始化方式，使用kaiming初始化。
"""

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,  # 网格大小，默认为 5
        spline_order=3, # 分段多项式的阶数，默认为 3
        scale_noise=0.1,  # 缩放噪声，默认为 0.1
        scale_base=1.0,   # 基础缩放，默认为 1.0
        scale_spline=1.0,    # 分段多项式的缩放，默认为 1.0
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,  # 基础激活函数，默认为 SiLU（Sigmoid Linear Unit）
        grid_eps=0.02,
        grid_range=[-1, 1],  # 网格范围，默认为 [-1, 1]，将输入数据归一化到一个特定范围是常见的预处理步骤。 [-1, 1] 是常用的归一化范围之一，例如使用 torchvision.transforms.Normalize 进行图像预处理时，默认会将像素值归一化到 [-1, 1]
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size # 设置网格大小和分段多项式的阶数
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size   # 计算网格步长
        grid = ( # 生成网格
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0] 
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)  # 将网格作为缓冲区注册
        #grid 张量最终存储的就是所有网格节点的位置信息，每个节点对应一个样条函数的连接点。
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features)) # 初始化基础权重和分段多项式权重，负责基础的线性变换
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )#负责更灵活的非线性变换，为每个网格点上的一个向量，表示该节点对应的样条函数的系数
        if enable_standalone_scale_spline:  # 如果启用独立的分段多项式缩放，则初始化分段多项式缩放参数
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )
        self.scale_noise = scale_noise # 保存缩放噪声、基础缩放、分段多项式的缩放、是否启用独立的分段多项式缩放、基础激活函数和网格范围的容差
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()  # 重置参数

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)# 使用 Kaiming 均匀初始化基础权重
        with torch.no_grad():
            noise = (# 生成缩放噪声
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )#引入噪声的概念使spline初始化函数更精细化
            self.spline_weight.data.copy_( # 计算分段多项式权重
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(#计算过程
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:  # 如果启用独立的分段多项式缩放，则使用 Kaiming 均匀初始化分段多项式缩放参数
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        """
        计算给定输入张量的 B-样条基函数。

        参数:
        x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。

        返回:
        torch.Tensor: B-样条基函数张量，形状为 (batch_size, in_features, grid_size + spline_order)。
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = ( # 形状为 (in_features, grid_size + 2 * spline_order + 1)
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        """
        计算插值给定点的曲线的系数。

        参数:
        x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
        y (torch.Tensor): 输出张量，形状为 (batch_size, in_features, out_features)。
        返回:
        torch.Tensor: 系数张量，形状为 (out_features, in_features, grid_size + spline_order)。
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        # 计算 B-样条基函数
        A = self.b_splines(x).transpose(
            0, 1 # 形状为 (in_features, batch_size, grid_size + spline_order)
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features) # 形状为 (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(   # 使用最小二乘法求解线性方程组
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)  # 形状为 (in_features, grid_size + spline_order, out_features)
        result = solution.permute( # 调整结果的维度顺序
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        """
        获取缩放后的分段多项式权重。

        返回:
        torch.Tensor: 缩放后的分段多项式权重张量，形状与 self.spline_weight 相同。
        """
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor): # 将输入数据通过模型的各个层，经过线性变换和激活函数处理，最终得到模型的输出结果
        """
        前向传播函数。

        参数:
        x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。

        返回:
        torch.Tensor: 输出张量，形状为 (batch_size, out_features)。
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight) # 计算基础线性层的输出
        #output = input @ weight.T
        #将输入张量 input 与权重矩阵 weight 的转置 (weight.T) 进行矩阵乘法
        spline_output = F.linear( # 计算分段多项式线性层的输出
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        #B-样条基函数张量，形状为 (batch_size, in_features, grid_size + spline_order)
        #b_splines函数计算的是B样条基函数(basisfunction)，它定义了一组用于构建样条曲线的“基本单元”。
        #spline_weight存储的是每个B样条基函数的系数(coefficients)，通过调整这些系数，可以控制每个基函数的形状和贡献，从而得到最终的样条曲线。
        return base_output + spline_output  # 返回基础线性层输出和分段多项式线性层输出的和

    @torch.no_grad()
    # 更新网格。
    # 参数:
    # x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
    # margin (float): 网格边缘空白的大小。默认为 0.01。
    # 根据输入数据 x 的分布情况来动态更新模型的网格,使得模型能够更好地适应输入数据的分布特点，从而提高模型的表达能力和泛化能力。
    def update_grid(self, x: torch.Tensor, margin=0.01): 
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)  # 计算 B-样条基函数
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)  # 调整维度顺序为 (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)  # 调整维度顺序为 (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0] # 对每个通道单独排序以收集数据分布
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)   # 更新网格和分段多项式权重
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        # 计算正则化损失，用于约束模型的参数，防止过拟合
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        """
        计算正则化损失。

        这是对原始 L1 正则化的简单模拟，因为原始方法需要从扩展的（batch, in_features, out_features）中间张量计算绝对值和熵，
        而这个中间张量被 F.linear 函数隐藏起来，如果我们想要一个内存高效的实现。

        现在的 L1 正则化是计算分段多项式权重的平均绝对值。作者的实现也包括这一项，除了基于样本的正则化。

        参数:
        regularize_activation (float): 正则化激活项的权重，默认为 1.0。
        regularize_entropy (float): 正则化熵项的权重，默认为 1.0。

        返回:
        torch.Tensor: 正则化损失。
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module): # 封装了一个KAN神经网络模型，可以用于对数据进行拟合和预测。
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        """
        初始化 KAN 模型。

        参数:
            layers_hidden (list): 包含每个隐藏层输入特征数量的列表。
            grid_size (int): 网格大小，默认为 5。
            spline_order (int): 分段多项式的阶数，默认为 3。
            scale_noise (float): 缩放噪声，默认为 0.1。
            scale_base (float): 基础缩放，默认为 1.0。
            scale_spline (float): 分段多项式的缩放，默认为 1.0。
            base_activation (torch.nn.Module): 基础激活函数，默认为 SiLU。
            grid_eps (float): 网格调整参数，默认为 0.02。
            grid_range (list): 网格范围，默认为 [-1, 1]。
        """
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            # layers_hidden = [128, 256, 512]
            # 表示一个包含三个隐藏层的神经网络，节点数量分别为128、256和512。
            # zip(layers_hidden, layers_hidden[1:]): 这个函数将layers_hidden与其自身错开一位的切片layers_hidden[1:]打包成一个个元组。
            # 例如，对于上面的layers_hidden，zip()函数会生成以下元组列表：[(128, 256), (256, 512)]
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False): # 调用每个KANLinear层的forward方法，对输入数据进行前向传播计算输出。
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
            update_grid (bool): 是否更新网格。默认为 False。

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, out_features)。
        """
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):#计算正则化损失的方法，用于约束模型的参数，防止过拟合。
        """
        计算正则化损失。

        参数:
            regularize_activation (float): 正则化激活项的权重，默认为 1.0。
            regularize_entropy (float): 正则化熵项的权重，默认为 1.0。

        返回:
            torch.Tensor: 正则化损失。
        """
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )




# #可视化
# import torch
# import matplotlib.pyplot as plt
# #
# # 初始化 KANLinear 层
# kan_linear = KANLinear(in_features=1, out_features=1, grid_size=5, spline_order=3, grid_range=[-1, 1])
# #
# # 生成输入数据
# x = torch.linspace(-1, 1, 100).unsqueeze(1)
#
# # 计算 B 样条基函数
# bases = kan_linear.b_splines(x)
#
# # 绘制8条B样条基函数
# plt.figure(figsize=(10, 7))
# # 获取当前的Axes对象，以便设置刻度参数
# ax = plt.gca()
#
# for i in range(bases.shape[2]):
#     plt.plot(x.numpy(), bases[:, 0, i].numpy(), label=f"Basis {i}",linewidth=2)
# plt.title("B-Spline Bases", fontsize=18)
# plt.xlabel("Input", fontsize=14)
# plt.ylabel("Basis Value", fontsize=14)
# # 刻度数字清晰一点：设置刻度标签字体大小
# ax.tick_params(axis='x', labelsize=14) # X轴刻度标签字体大小
# ax.tick_params(axis='y', labelsize=14) # Y轴刻度标签字体大小
#
# plt.legend(fontsize=12)
#
# plt.grid(True)
# plt.savefig("my_plot8.png", dpi=300, bbox_inches='tight')
# plt.show()




# #绘制基础函数图像
# plt.figure(figsize=(10, 7))
# ax = plt.gca()
# plt.plot(x.numpy(), kan_linear.base_activation(x).numpy())
# plt.title("Base Activation Function (SiLU)", fontsize=18)
# plt.xlabel("Input", fontsize=16)
# plt.ylabel("Activation Value", fontsize=20)
# # 刻度数字清晰一点：设置刻度标签字体大小
# ax.tick_params(axis='x', labelsize=14) # X轴刻度标签字体大小
# ax.tick_params(axis='y', labelsize=14) # Y轴刻度标签字体大小
# plt.legend(fontsize=12)
# plt.grid(True)
# plt.savefig("my_plot9.png", dpi=600, bbox_inches='tight')
# plt.show()




import torch
import matplotlib.pyplot as plt

# 初始化 KANLinear 层
kan_linear = KANLinear(in_features=1, out_features=1, grid_size=5, spline_order=3, grid_range=[-1, 1])

# 随机生成 spline_weight
torch.manual_seed(0)  # 设置随机种子，保证结果可复现
kan_linear.spline_weight.data.normal_(0, 1)

# 生成输入数据
x = torch.linspace(-1, 1, 100).unsqueeze(1)

#
# #  8条B样条基函数通过线性组合的方式，与spline_weight 作为系数，构成的一条样条函数
# bases = kan_linear.b_splines(x)
#
# # 计算样条函数  # 这里改正了错误
# spline_function = torch.sum(kan_linear.scaled_spline_weight.squeeze(0) * bases, dim=2).squeeze()
#
# # 绘制样条函数
# plt.figure(figsize=(10, 7))
# ax = plt.gca()
# plt.plot(x.numpy(), spline_function.detach().numpy())
# plt.title("Spline Function", fontsize=18)
# plt.xlabel("Input", fontsize=16)
# plt.ylabel("Spline Function Value", fontsize=20)
# # 刻度数字清晰一点：设置刻度标签字体大小
# ax.tick_params(axis='x', labelsize=14) # X轴刻度标签字体大小
# ax.tick_params(axis='y', labelsize=14) # Y轴刻度标签字体大小
# plt.legend(fontsize=12)
# plt.grid(True)
# plt.savefig("my_plot10.png", dpi=300, bbox_inches='tight')
# plt.show()



output = kan_linear(x)  # 直接调用 forward() 函数计算最终激活函数值
plt.figure(figsize=(10, 7))
ax = plt.gca()
plt.plot(x.numpy(), output.detach().numpy())
plt.title("Final Activation Function", fontsize=18)
ax.tick_params(axis='x', labelsize=14) # X轴刻度标签字体大小
ax.tick_params(axis='y', labelsize=14) # Y轴刻度标签字体大小
plt.xlabel("Input", fontsize=16)
plt.ylabel("Spline Function Value", fontsize=20)
plt.legend(fontsize=12)
plt.grid(True)
plt.savefig("my_plot11.png", dpi=300, bbox_inches='tight')
plt.show()

#可视化