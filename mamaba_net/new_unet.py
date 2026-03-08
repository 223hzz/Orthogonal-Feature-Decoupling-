import torch
import torch.nn as nn
from nets.mamaba_net.new_ResNet import resnet50
from nets.mamaba_net.HybridSETR import hybridsert
from nets.mamaba_net.mamba_unet import create_vsslayer_up,PatchExpand2D,Final_PatchExpand2D
from nets.mamaba_net.DySampl_upsample import DySample
from functools import partial
from timm.models.layers import DropPath, trunc_normal_
from nets.mamaba_net.DilateAttention import Mlp
class Chang_C(nn.Module):
    def __init__(self,dim_in,dim_out):
        super(Chang_C, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.layernorm= nn.LayerNorm(dim_out, eps=1e-6)
        self.relu = nn.ReLU()
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
        x = self.linear(x)
        x = self.layernorm(x)
        x = self.relu(x)
        return x

# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x


# def drop_path(x, drop_prob: float = 0., training: bool = False):
#     """
#     Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
#     This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
#     the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
#     See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
#     changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
#     'survival rate' as the argument.
#     """
#     if drop_prob == 0. or not training:
#         return x
#     keep_prob = 1 - drop_prob
#     shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
#     random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
#     random_tensor.floor_()  # binarize
#     output = x.div(keep_prob) * random_tensor
#     return output
#
#
# class DropPath(nn.Module):
#     """
#     Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
#     """
#     def __init__(self, drop_prob=None):
#         super(DropPath, self).__init__()
#         self.drop_prob = drop_prob
#
#     def forward(self, x):
#         return drop_path(x, self.drop_prob, self.training)

# class Final_PatchExpand2D(nn.Module):
#     def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.dim = dim
#         self.dim_scale = dim_scale
#         self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
#         self.norm = norm_layer(self.dim // dim_scale)
#
#     def forward(self, x):
#         B, C ,H, W, = x.shape
#         x = x.permute(0, 2, 3, 1)
#         x = self.expand(x)
#         print(x.shape)
#         x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
#         x = self.norm(x)
#         return x.permute(0, 3, 1, 2)

# class PatchExpand2D(nn.Module):  # 高度和宽度会扩展为两倍，而通道数变为1/2
#     def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.dim = dim * 2
#         self.dim_scale = dim_scale
#         self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
#         self.norm = norm_layer(self.dim // dim_scale)
#
#     def forward(self, x):
#         B, H, W, C  = x.shape  # 排列顺序中通道数放在最后了
#         x = self.expand(x)
#
#         x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
#                       c=C // self.dim_scale)
#         x = self.norm(x)
#
#         return x

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int,if_convert:bool=False):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            #nn.UpsamplingBilinear2d(scale_factor=2),
            # if if_convert:
            #     F_g=DySample(F_g)
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=False)
        if if_convert:
            self.W_g_convert = DySample(F_g)
            self.if_convert = True
        else:
            self.if_convert = False
    def forward(self, g, x):
        if self.if_convert:
            g = self.W_g_convert(g)
        else:
            g = g.permute(0, 3, 1, 2)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x*psi


# class unetUp(nn.Module):
#     def __init__(self, in_size, out_size):
#         super(unetUp, self).__init__()
#         self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
#         # self.up = nn.UpsamplingBilinear2d(scale_factor=2)  # 自己线叠加一次，将尺寸变为2倍
#         self.relu = nn.ReLU(inplace=True)
#         # 其中inputs1的属性包含in_size和out_size
#
#     def forward(self, inputs1, inputs2):#也就意味着可以随意交换顺序
#         # outputs = torch.cat([inputs1, inputs2], 1)  # 叠加操作dim=1代表按行排堆砌，两张图输入的尺寸必须一样，两张图通道数相加。
#         # outputs = self.conv1(outputs)  # outputs的in_size等于41行的1024和out_size等于512，
#         # # outputs的in_size等于43行1024，输出256
#         # # outputs的in_size等于45行512，输出128
#         # # outputs的in_size等于48行192，输出64
#         # outputs = self.relu(outputs)
#         # outputs = self.conv2(outputs)
#         # outputs = self.relu(outputs)
#         inputs2_permuted = inputs2.permute(0, 2, 3, 1)
#         outputs =
#         return outputs

class unetUp(nn.Module):
    def  __init__(self,i,dim,drop_path_ratio=0.1,depth=4):
        super(unetUp, self).__init__()
        # self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        # # self.up = nn.UpsamplingBilinear2d(scale_factor=2)  # 自己线叠加一次，将尺寸变为2倍
        # self.relu = nn.ReLU(inplace=True)
        # # 其中inputs1的属性包含in_size和out_size
        norm_with_eps = partial(nn.LayerNorm, eps=1e-6)
        self.layer_norm = norm_with_eps(dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        self.drop_path_ratio = dpr[i]
        self.drop_path = DropPath(self.drop_path_ratio) if self.drop_path_ratio > 0. else nn.Identity()
        self.mlp = Mlp(dim)

    def forward(self, inputs1, inputs2):#也就意味着可以随意交换顺序
        # outputs = torch.cat([inputs1, inputs2], 1)  # 叠加操作dim=1代表按行排堆砌，两张图输入的尺寸必须一样，两张图通道数相加。
        # outputs = self.conv1(outputs)  # outputs的in_size等于41行的1024和out_size等于512，
        # # outputs的in_size等于43行1024，输出256
        # # outputs的in_size等于45行512，输出128
        # # outputs的in_size等于48行192，输出64
        # outputs = self.relu(outputs)
        # outputs = self.conv2(outputs)
        # outputs = self.relu(outputs)
        inputs2_permuted = inputs2.permute(0, 2, 3, 1)#这个是将(batch_size, channels, height, width)->(batch_size, height, width, channels)
        outputs = self.drop_path(self.mlp(self.layer_norm(inputs2_permuted+inputs1)))
        return outputs


class new_Unet(nn.Module):
    def __init__(self, num_classes=21, pretrained=False, backbone='resnet50',input_shape=[224,224]):
        super(new_Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg = VGG16(pretrained=pretrained)  # 不加载预训练权值
            in_filters = [192, 384, 768, 1024]
            at_filters = [64,128,256,512,512]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained=pretrained)
            self.img_dimgs = input_shape[0]
            self.HybridSETR = hybridsert(self.img_dimgs)
            #in_filters = [192, 384, 768, 2560]
            # in_filters = [576, 640, 768, 1024]
            at_filters = [64,256,512,1024,2048]

        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]
        #out_filters = [64, 128, 256, 1024]
        #self.up = nn.UpsamplingBilinear2d(scale_factor=2) #定义尺寸变化函数将自己线叠加一次，将尺寸变为2倍
        # upsampling
        # 28,28,512
        self.enxpand4 = PatchExpand2D(at_filters[3],2)#将传入的7*7*2048变为14*14*1024
        self.attention4 = Attention_block(at_filters[4],at_filters[3], out_filters[3],True)#新增代码：传入参数为2048*7*7与1024*14*14，输出等于1024*14*14
        self.up_concat4 = unetUp(0,at_filters[3])#第1层，传入通道数为1024
        self.mamba_block4 = create_vsslayer_up(dim=out_filters[3], depth=3,i_layer=0,upsample=PatchExpand2D)
        # self.up_concat4 = unetUp(in_filters[3],
        #                          out_filters[3])  # 加入使用的是“resnet50”网络，则传入的参数是2048+512=2560，512.等于in_size=2560,out_size=512,进行初始化
        # 28,28,256
        self.expand3 = Final_PatchExpand2D(at_filters[2])#将传入的7*7*2048变为28*28*512
        self.attention3 = Attention_block(at_filters[2],at_filters[2], out_filters[2],False)#新增代码：传入参数512*28*28与28*28*512，输出等于512*28*28
        self.up_concat3 = unetUp(1,at_filters[2])#第2层，传入通道为512
        self.mamba_block3 = create_vsslayer_up(dim=out_filters[2], depth=4, i_layer=1, upsample=PatchExpand2D)


        # self.attention3 = Attention_block(at_filters[2],at_filters[2], out_filters[2])#新增代码：传入参数为28*28*512与28*28*512，输出等于28*28*256

        # self.up_concat3 = unetUp(in_filters[2], out_filters[2])  # 传入参数为512+256=768，256，输入为768输出为256
        # # 56,56,128
        #这一步开始要开始改变大小
        self.expand2 = PatchExpand2D(at_filters[1],8)#将传入的7*7*2048变为56*56*256
        self.attention2 = Attention_block(at_filters[1],at_filters[1], out_filters[1],False)#新增代码，传入参数为28*28*256与256*56*56，输出的等于256*56*56
        self.up_concat2 = unetUp(2, at_filters[1])  #第3层，传入通道为256
        self.mamba_block2 = create_vsslayer_up(dim=out_filters[1],depth=6, i_layer=2,upsample=PatchExpand2D)

        #56,56,64
        #112，112，64
        self.expand1 = PatchExpand2D(out_filters[1],16)#将传入的7*7*2048变为112*112*128
        self.change_c = Chang_C(out_filters[1],out_filters[0])#用在up2与decoder_output3上面
        self.attention1 = Attention_block(at_filters[0], at_filters[0], int(out_filters[0]/2),False)#新增代码，传入参数为512+64通道也就是self.up_concat输出得到的通道数=56*56*128，at_filters[0]为64*56*56，输出为64*56*56
        self.up_concat1 = unetUp(3, at_filters[0])  #第4层，传入通道为64
        self.mamba_block1 = create_vsslayer_up(dim=int(out_filters[0]/2),depth=3, i_layer=2,upsample=PatchExpand2D)
        # self.up = nn.UpsamplingBilinear2d(scale_factor=2)  # 自己线叠加一次，将尺寸变为2倍
        # if backbone == 'resnet50':
        #     self.up_conv = Final_PatchExpand2D(dim=out_filters[0], dim_scale=4)
        # else:
        #     self.up_conv = None
        #224，224，64
        self.final_conv = nn.Conv2d(int(out_filters[0]/2), num_classes, 1)
        self.backbone = backbone

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)
            # 64.256.512.1024.2048]
            [decoder_output0, decoder_output1,decoder_output2,decoder_output3]=self.HybridSETR.forward(inputs)
            #全部都是

        decoder_output0_process = self.enxpand4(decoder_output0)#生成14*14*1024
        at4 = self.attention4(feat5,feat4)#输出的是1024*14*14
        up4 = self.up_concat4(decoder_output0_process,at4)#传入的是14*14*1024和1024*14*14，注意此处位置顺序不能搞错，输出的也是14*14*1024通道数排在最后
        up4 = self.mamba_block4(up4)#输入的是14*14*1024输出的是28*28*512，通道数排最后

        # up4 = self.up_concat4(decoder_output0,at5
        #                       )  #512*28*28，512*28*28，in_size=torch.cat([512, 512], 1)=1024=in_filters[3]，up4输出的得到512*28*28

        # at4 = self.attention3(up4,feat3)#新增代码，输入up4=512*28*28，feat3=512*28*28

        decoder_output1_process = self.expand3(decoder_output1)#生成28*28*512
        at3 = self.attention3(up4,feat3)#up4必须放在前头
        up3 = self.up_concat3(decoder_output1_process,at3) #decoder_output1_process为28*28*512，at4为512*28*28,注意此处位置顺序不能搞错，输出的也是28*28*512通道数排在最后
        up3 = self.mamba_block3(up3)#输入的是28*28*512,输出的是56*56*256

        # #开始要进行尺寸的调整输入的时256*28*28与256*56*56
        # up3=self.up(up3)#将up3的倍数变维256*56*56
        # decoder_output2=self.up(decoder_output2)#变维56*56*512
        #
        # at3 = self.attention2(up3, feat2)  #新增代码：输入的时256*56*56与256*56*56，输出的时128*56*56

        decoder_output2_process = self.expand2(decoder_output2)#生成56*56*256
        at2 = self.attention2(up3,feat2)#up3必须放前头
        up2 = self.up_concat2(decoder_output2_process,at2) # decoder_output2_process为56*56*256，at2为256*56*56，注意此处位置顺序不能搞错，输出的也是56*56*256通道数排在最后
        up2 = self.mamba_block2(up2)#输入的是56*56*256，输出的是112*112*128


        decoder_output3_process=self.change_c(self.expand1(decoder_output3))#生成112*112*128又经变维变成112*112*64
        up2 = self.change_c(up2)
        at1 = self.attention1(up2, feat1) #up2必须放前头
        up1 = self.up_concat1(decoder_output3_process,at1)# decoder_output3_process维112*112*64，at1为64*112*112，注意此处位置顺序不能搞错，输出的也是112*112*64通道数排在最后
        up1 = self.mamba_block1(up1)#输入的是112*112*64,输出的是224*224*32
        # if self.up_conv != None:
        #     up1 = self.up_conv(up1)  # 64*224*224
        up1 = up1.permute(0, 3, 1, 2)
        final = self.final_conv(up1)

        return final
    #没有使用冻结训练
    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                # 循环获取参数
                param.requires_grad = False  # 把层结构冻住不做训练，指定为False、
                # requires_grad是Pytorch中通用数据结构Tensor的一个属性，用于说明当前量是否需要在计算中保留对应的梯度信息，权重和偏置值将不会更新。
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True  # 解冻后训练所有层
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True


#
# inputs = torch.randn(1,3,224,224)
# model = new_Unet(num_classes=2)
# output = model(inputs)
# print(output.shape)
# 检查 GPU 可用性
# if torch.cuda.is_available():
#     device = torch.device("cuda:0")  # 选择 GPU 设备
# else:
#     device = torch.device("cpu")  # 使用 CPU
#
# # 创建输入数据和模型
# inputs = torch.randn(1, 3, 224, 224).to(device)
# model = new_Unet(num_classes=2).to(device)
#
# # 执行模型推理
# output = model(inputs)
# print(output.shape)
# # gpu_memory_allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
# gpu_memory_cached = torch.cuda.memory_cached(device) / (1024 * 1024)
#
# print(f"已分配 GPU 内存: {gpu_memory_allocated:.2f} MB")
# print(f"缓存的 GPU 内存: {gpu_memory_cached:.2f} MB")
# print(output.shape)







# import torch
# from ptflops import get_model_complexity_info
# import re
#
# device = torch.device("cuda:0")
# #Model thats already available
# net = new_Unet(num_classes=2).to(device)
# macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
# print_per_layer_stat=True, verbose=True).to(device)
# # Extract the numerical value
# flops = eval(re.findall(r'([\d.]+)', macs)[0])*2
# # Extract the unit
# flops_unit = re.findall(r'([A-Za-z]+)', macs)[0][0]
#
# print('Computational complexity: {:<8}'.format(macs))
# print('Computational complexity: {} {}Flops'.format(flops, flops_unit))
# print('Number of parameters: {:<8}'.format(params))