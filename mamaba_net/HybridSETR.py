from datetime import time

import torch
import torch.nn as nn
#from ResNet import ResNetV2Model
from nets.mamaba_net.new_ResNet import resnet50
# from Transformer import TransformerModel
from nets.mamaba_net.PositionalEncoding import LearnedPositionalEncoding
from nets.mamaba_net.HWD import Down_wt
from nets.mamaba_net.DilateAttention import DilateBlock
from nets.mamaba_net.cosFormer import CosformerAttention

# import numpy as np
class HybridSegmentationTransformer(nn.Module):
    def __init__(
        self,
        img_dim,#输入图像的大小
        patch_dim,
        # num_classes,#分类数
        # num_channels,#输入的通道
        # embedding_dim,#转化为向量的维度=（patch_dim）²*3
        num_heads,#多头注意力机制的头数
        num_layers,#Transformer模型中的编码器（encoder）和解码器（decoder）堆叠层的数量
        #hidden_dim,#是hidden_dim很可能是指Transformer模型中前馈网络（Feed-Forward Network, FFN）的隐藏层的维度。在典型的Transformer模型中，每个Transformer层由两部分构成：一个多头自注意力模块和一个前馈网络。前馈网络通常包含两个线性层，其中第一个线性层将维度从模型的embedding_dim扩展到hidden_dim，并且通常跟随一个非线性激活函数，如ReLU；第二个线性层则通常将维度从hidden_dim再次降低回embedding_dim。
        # #include_conv5=False,#它决定了是否在模型的backbone网络中包含第五层卷积层（即Conv5）
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        #positional_encoding_type="learned",
        backbone='resnet50',
    ):
        super(HybridSegmentationTransformer, self).__init__()
        #self.img_dim = img_dim
        self.img_dim=img_dim
        self.patch_dim=patch_dim
        #self.embedding_dim = embedding_dim
        # self.embedding_dim = embedding_dim
        #self.num_heads = num_heads
        self.num_heads=num_heads
        #self.num_channels = num_channels
        #self.num_channels=2048
        #self.include_conv5 = include_conv5
        self.backbone = backbone
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.num_layers= num_layers
        # self.hidden_dim=hidden_dim
        #self.patch_dim = patch_dim
        #self.num_classes = num_classes
        self.backbone_model, self.flatten_dim = self.configure_backbone()#返回这个ResNet50模型，flatten_dim=2048
        assert self.flatten_dim % num_heads == 0
        # self.projection_encoding = nn.Linear(self.flatten_dim, embedding_dim)#将2048个向量转化为向量的维度512
        #self.decoder_dim = int(img_dim / 16.0) ** 2#解码器维度用于可学习的位置编码之中，在标准的ResNet中，初始特征图的空间尺寸（高度和宽度）会在前四个阶段（每个阶段包含步长为2的下采样）中被依次缩减为原来的1/16
        #if self.include_conv5:
        self.decoder_dim = int(img_dim / self.patch_dim) ** 2
        self.downsample = Down_wt(self.flatten_dim,self.flatten_dim*2)

        # if positional_encoding_type == "learned":#位置编采用“learned”类型,位置编码
        self.position_encoding = LearnedPositionalEncoding(
                self.decoder_dim, self.flatten_dim, self.decoder_dim
            )
        # elif positional_encoding_type == "fixed":
        #     self.position_encoding = FixedPositionalEncoding(
        #         self.embedding_dim,
        #     )
        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        # self.transformer = TransformerModel(
        #     # embedding_dim,#512
        #     num_layers,#堆叠24层
        #     num_heads,#8
        #     hidden_dim,
        #     self.dropout_rate,
        #     self.attn_dropout_rate,
        # )\
        self.transformer = DilateBlock(
        dim=self.flatten_dim,
        num_heads=8
        )
        # self.pre_head_ln = nn.LayerNorm(self.flatten_dim)
        self.CosformerAttention = CosformerAttention(embed_dim=self.flatten_dim*2,
            num_heads= self.num_heads)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.x_Con = nn.Conv2d(
        #     in_channels=self.flatten_dim,
        #     out_channels=self.flatten_dim,
        #     kernel_size=1,
        #     stride=1,
        #     padding=self._get_padding('VALID', (1, 1), ),
        # )
        # self.x_norm = nn.BatchNorm2d(self.flatten_dim)
        # self.relu = nn.ReLU()
    # def process_image(self, image):
    #     input =  self.downsample(image.permute(0, 3, 1, 2))
    #     convert_image_image = convert_image(input)
    #     output = self.CosformerAttention(convert_image_image,depths=8)
    #     return output

    def process_all_outputs(self,intmd_encoder_outputs):
        processed_outputs = []
        for output in intmd_encoder_outputs:
            input = self.downsample(output.permute(0, 3, 1, 2))
            # print("input.shape",input.shape)
            output = self.CosformerAttention(input, depths=8)
            output = self.view_ouput(output)
            processed_outputs.append(output)
        return processed_outputs
    def encode(self, x):
        # apply bit backbone
        x = self.backbone_model(x)[3] #接收来自resnet50的结果返回了feat5
        #[batch_size, channels, height, width]
        # self.img_dim=x.size(2)#获取height
        # self.flatten_dim=x.size(1)#获取channel
        # self.projection_encoding = nn.Linear(self.flatten_dim, self.embedding_dim)
        x = x.view(x.size(0), -1, self.flatten_dim)
        #[batch_size, num_features = channels * height * width]->[batch_size,num_features/flatten_dim,flatten_dim]
        #这里patch_size没有派上用场

        # x = self.projection_encoding(x)
        x = self.position_encoding(x)
        x = self.pe_dropout(x)
        # print(x.shape)

        # apply transformer
        x = self._reshape_output(x)
        # x, intmd_x = self.transformer(x)#进入transformer model
        intmd_x = self.transformer(x)
        # x = self.pre_head_ln(x)
        # return x, intmd_x # intermediate_outputs[]是一个list里面储存的是TransformerModel的每一层的输出结果
        return intmd_x
    #def decode(self, x, intmd_x, intmd_layers=None):
    # def decode(self, x):
    #     x = self._reshape_output(x)#图像还原
    #     x = self.x_Con(x)
    #     x = self.x_norm(x)
    #     x = self.relu(x)
    #     #
    #     # x = nn.Conv2d(
    #     #     in_channels=self.embedding_dim,
    #     #     out_channels=self.num_classes,
    #     #     kernel_size=1,
    #     #     stride=1,
    #     #     padding=self._get_padding('VALID', (1, 1),),
    #     # )(x)
    #     #x = nn.Upsample(scale_factor=self.patch_dim, mode='bilinear')(x)

        # return x

    #def forward(self, x, auxillary_ou tput_layers=None):
    def forward(self, x):
        # x = x.to(self.device)#新增代码
        # encoder_output, intmd_encoder_outputs = self.encode(x)
        intmd_encoder_outputs = self.encode(x)
        # print("The type of intmd_encoder_outputs is:", type(intmd_encoder_outputs))
        # for key, value in intmd_encoder_outputs.items():
        #     print(f"{key}: {value.shape}")
        # decoder_output0 = self.decode(
        # #    encoder_output, intmd_encoder_outputs, auxillary_output_layers
        #     encoder_output
        # )
        # print(decoder_output0.shape)
        #auxillary_output_layers= len(intmd_encoder_outputs)#47
        #if auxillary_output_layers is not None:
        intmd_encoder_outputs = self.process_all_outputs(intmd_encoder_outputs)
        # 示例用法
        # processed_outputs = process_all_outputs(intmd_encoder_outputs)
        # auxillary_output_layers = [2,4,6]
        # auxillary_outputs = {}
        # all_keys = []
        # for i in auxillary_output_layers:
        #     key = str(i)
        #     # print(key)
        #     all_keys.append(key)
        #     auxillary_outputs[key] = intmd_encoder_outputs[key]
        #     #return decoder_output, auxillary_outputs
        # # all_keys.reverse()
        # decoder_outputs = []
        # decoder_output0 =self.view_ouput(self.process_image(decoder_output0))
        # decoder_outputs.append(decoder_output0)
        # for key in reversed(all_keys):
        #     temp_x = auxillary_outputs[key]
        #     decoder_output = self.decode(temp_x)
        #     decoder_output = self.view_ouput(self.process_image(decoder_output))
        #     decoder_outputs.append(decoder_output)
            # print("Decoder output shape:", decoder_output.shape)

            # 在循环结束后返回结果
        #输出结果的形状属于测试代码
        # sizes = [output.shape for output in intmd_encoder_outputs]
        # print("Decoder Outputs Sizes:")
        # for size in sizes:
        #     print(size)#输出最后的结果为torch.Size([1, 7, 7, 2048])

        return intmd_encoder_outputs
        # print(all_keys)
        # temp_x= auxillary_outputs[all_keys[1]]
        # decoder_output1 =self.decode(temp_x)
        #
        # temp_x = auxillary_outputs[all_keys[2]]
        # decoder_output2 = self.decode(temp_x)
        #
        # temp_x = auxillary_outputs[all_keys[3]]
        # decoder_output3 = self.decode(temp_x)
        #
        # return [decoder_output0, decoder_output1,decoder_output2,decoder_output3]

    # def _get_padding(self, padding_type, kernel_size):
    #     assert padding_type in ['SAME', 'VALID']
    #     if padding_type == 'SAME':
    #         _list = [(k - 1) // 2 for k in kernel_size]
    #         return tuple(_list)
    #     return tuple(0 for _ in kernel_size)
    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            self.flatten_dim,
        )
        # x = x.permute(0, 3, 1, 2).contiguous()
        return x
    #
    def view_ouput(self, x):
        N, B, C = x.shape
        new_dim = int(N ** (1 / 2))  # 使用 int() 将浮点数转换为整数

        # 确保 N 是 new_dim 的平方，否则 view 操作可能会失败
        if N != new_dim * new_dim:
            raise ValueError(f"N ({N}) is not a perfect square for new_dim ({new_dim})")

            # 现在可以使用整数元组调用 view
        x = x.view(
            B,
            new_dim,
            new_dim,
            C
        )
        return x
    def configure_backbone(self):
        """
        Current support offered for all BiT models
        KNOWN_MODELS in https://github.com/google-research/big_transfer/blob/master/bit_pytorch/models.py

        expects model name of style 'r{depth}x{width}'
        where depth in [50, 101, 152]
        where width in [1,2,3,4]
        """
        backbone = self.backbone

        #splits = backbone.split('x')#以x为分割
        #model_name = splits[0]#r50
        #width_factor = int(splits[1])#1
        #这个机制的目的是为了确保模型的基础架构可以正确地被设置和初始化，从而在后续的训练或者推理中不会出现问题。
        # if model_name in ['r50', 'r101'] and width_factor in [2, 4]:
        #     return ValueError(
        #         "Invalid Configuration of models -- expect 50x1, 50x3, 101x1, 101x3"
        #     )
        # elif model_name == 'r152' and width_factor in [1, 3]:
        #     return ValueError(
        #         "Invalid Configuration of models -- expect 152x2, 152x4"
        #     )

        # block_units_dict = {
        #     'r50': [3, 4, 6, 3],
        #     'r101': [3, 4, 23, 3],
        #     'r152': [3, 8, 36, 3],
        # }
        # block_units = block_units_dict.get(model_name, [3, 4, 6, 3])#get方法试图根据 model_name 变量来获取字典中的值。如果 model_name 在字典中不存在，则默认返回 [3, 4, 6, 3]
        # model = resnet50(
        #     block_units, width_factor, head_size=self.num_classes
        # )
        #
        #if self.num_channels == 3:
        flatten_dim = 1024
        model=resnet50(pretrained=False)
        #at_filters = [64, 256, 512, 1024, 2048]

        return model, flatten_dim

def hybridsert (img_dimgs):
    img_dim = img_dimgs
    patch_dim= 16#这个要和resnet50中图片的尺寸大小对应，比如传入的是（14，14）那么img_dim /patch_dim =》224/16=14
    num_heads = 8
    num_layers = 4
    model = HybridSegmentationTransformer(img_dim,patch_dim,num_heads,num_layers)
    return model




#
# device = torch.device("cuda:0")  # 选择 GPU 设备
#
# import torch
# input_tensor = torch.randn(1, 3, 224, 224).to(device)
# model =  hybridsert(224).to(device)
# output = model(input_tensor)#真正近期取得是（1，1024，14，14）出来的是torch.Size([1, 7, 7, 2048])
# gpu_memory_allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
# gpu_memory_cached = torch.cuda.memory_cached(device) / (1024 * 1024)
# print(f"已分配 GPU 内存: {gpu_memory_allocated:.2f} MB")
# print(f"缓存的 GPU 内存: {gpu_memory_cached:.2f} MB")
# # print(output.shape)








# 参数两计算方法1只能计算parameters
# def model_structure(model):
#     blank = ' '
#     print('-' * 90)
#     print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
#           + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
#           + ' ' * 3 + 'number' + ' ' * 3 + '|')
#     print('-' * 90)
#     num_para = 0
#     type_size = 1  # 如果是浮点数就是4
#
#     for index, (key, w_variable) in enumerate(model.named_parameters()):
#         if len(key) <= 30:
#             key = key + (30 - len(key)) * blank
#         shape = str(w_variable.shape)
#         if len(shape) <= 40:
#             shape = shape + (40 - len(shape)) * blank
#         each_para = 1
#         for k in w_variable.shape:
#             each_para *= k
#         num_para += each_para
#         str_num = str(each_para)
#         if len(str_num) <= 10:
#             str_num = str_num + (10 - len(str_num)) * blank
#
#         print('| {} | {} | {} |'.format(key, shape, str_num))
#     print('-' * 90)
#     print('The total number of parameters: ' + str(num_para))
#     print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
#     print('-' * 90)
#
# model_structure(hybridsert(224))





#计算flops
# import torch
# from ptflops import get_model_complexity_info
# import re
# device = torch.device("cuda:0")
# #Model thats already available
# net = hybridsert(224).to(device)
# macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
# print_per_layer_stat=True, verbose=True)
# # Extract the numerical value
# flops = eval(re.findall(r'([\d.]+)', macs)[0])*2
# # Extract the unit
# flops_unit = re.findall(r'([A-Za-z]+)', macs)[0][0]
#
# print('Computational complexity: {:<8}'.format(macs))
# print('Computational complexity: {} {}Flops'.format(flops, flops_unit))
# print('Number of parameters: {:<8}'.format(params))


#
# import thop
# if __name__ == '__main__':
#     model =  hybridsert(224)
#     device = torch.device('cuda')
#     model.to(device)
#     dump_input = torch.ones(1,3,224,224).to(device)
#     output = model(dump_input)
#     gpu_memory_allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
#     gpu_memory_cached = torch.cuda.memory_cached(device) / (1024 * 1024)
#
#     print(f"已分配 GPU 内存: {gpu_memory_allocated:.2f} MB")
#     print(f"缓存的 GPU 内存: {gpu_memory_cached:.2f} MB")
#     dump_input = torch.ones(1,3,224,224).to(device)
#     flops, params = thop.profile(model, inputs=(dump_input,))
#     print(f"FLOPs: {flops / 1e9} G")  # 打印计算量（以十亿次浮点运算为单位）
#     print(f"Params: {params / 1e6} M")  # 打印参数量（以百万为单位）
#     # Warn-up
#     for _ in range(5):
#         start = time.time()
#         outputs = model(dump_input)
#         torch.cuda.synchronize()
#         end = time.time()
#         print('Time:{}ms'.format((end-start)*1000))
#
#     with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
#         outputs = model(dump_input)
#     print(prof.table())