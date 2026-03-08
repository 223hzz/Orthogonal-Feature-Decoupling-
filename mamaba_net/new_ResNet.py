import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
# from torchsummary import summary
import torch
from nets.mamaba_net.HWD import Down_wt
from nets.mamaba_net.GemPooling import GeMPooling
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
#此是为了膨胀卷积替换时卷积保证输出的尺寸大小不变所以根据公式算出padding=dilation膨胀卷积系数

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups#群组卷积没啥用
        # 利用1x1卷积下降通道数
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 =  nn.BatchNorm2d(width)
        # 利用3x3卷积进行特征提取
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 =nn.BatchNorm2d(width)
        # 利用1x1卷积上升通道数
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        # -----------------------------------------------------------#
        #   假设输入图像为600,600,3
        #   当我们使用resnet50的时候
        # -----------------------------------------------------------#
        # if replace_conv is None:
        #     replace_conv = [False, True, True]
        self.dilation_rate = 1
        self.inplanes = 64
        super(ResNet, self).__init__()
        # 600,600,3 -> 300,300,64
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)
        self.Down_wt1 = Down_wt(3,64)
        # 300,300,64 -> 150,150,64
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)  # change
        self.gempooling = GeMPooling(feature_size=64,pool_size=2, init_norm=2.0)
        # self.Down_wt2 = Down_wt(64,256)
        # 150,150,64 -> 150,150,25
        self.layer1 = self._make_layer(block, 64, layers[0])  # 重复该层3次layers【0】=3
        # 150,150,256 -> 75,75,512
        # self.Down_wt2 = Down_wt(64,256)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,replace_conv=False)
        # 75,75,512 -> 38,38,1024
        # self.Down_wt3 = Down_wt(256,512)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,replace_conv=False)
        # 38,38,1024 -> 19,19,2048
        # self.Down_wt4 = Down_wt(512,1024)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,replace_conv=False)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # 对层开始初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1,replace_conv=False):
        # 这里的block等于传入的layers[0]之类的代表的是传入层结构该重复执行几次
        downsample = None
        # 设置下采样函数，捷径分支上的输出
        previous_dilation_rate = self.dilation_rate
        if replace_conv:
            self.dilation_rate *= stride
            stride = 1#使它的stride等于1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
            # 主干上的输出
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,dilation=previous_dilation_rate))
        # inplanes为输入的通道数，planes为卷积核的个数
        self.inplanes = planes * block.expansion
        # 重复执行
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,dilation=self.dilation_rate))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        #
        # x = self.conv1(x)
        # x = self.bn1(x)
        # feat1 = self.relu(x)
        feat1 = self.Down_wt1(x)
        # print("feat1.shape:",feat1.shape)
        # x = self.Down_wt2(feat1)
       # x = self.maxpool(feat1)
        x = self.gempooling(feat1)
        # print("x.shape:",x.shape)
        feat2 = self.layer1(x)
        # feat2 = self.layer1(x) + self.Down_wt2(feat1)
        # print(feat2.shape)
        feat3 = self.layer2(feat2)
        # feat3 = self.layer2(feat2) + self.Down_wt3(feat2)
        # print(feat3.shape)
        feat4 = self.layer3(feat3)
        # feat4 = self.layer3(feat3) + self.Down_wt4(feat3)
        # print(feat4.shape)
        feat5 = self.layer4(feat4)
        # print(feat5.shape)
        return [feat1, feat2, feat3, feat4, feat5]
        # 64，256，512，1024，2048



def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(
            model_zoo.load_url('https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth', model_dir='model_data'),
            strict=False)
    # load_url等于load_state_dict_from_url函数，都是去取权重文件，之后再通过load_state_dict进行加载
    del model.avgpool
    del model.fc
    return model

#
# input_tensor = torch.randn(1, 3, 224, 224)
# model = resnet50(pretrained=False)
# output = model(input_tensor)
# # if __name__ == "__main__":
# #     input_shape = [224, 224]
# #     num_classes = 2
# #
# #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #     model = resnet50().to(device)
# #     summary(model, (3, input_shape[0], input_shape[1]))
