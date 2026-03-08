import torch
import torch.nn as nn
import torch.nn.functional as F


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        # normalizer = torch.as_tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)
#
#
# class Attention_block(nn.Module):
#     def __init__(self, F_g, F_l, F_int):
#         super(Attention_block, self).__init__()
#         self.W_g = nn.Sequential(
#             DySample(F_g),
#             nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(F_int)
#         )
#         self.W_x = nn.Sequential(
#             nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(F_int)
#         )
#         self.psi = nn.Sequential(
#             nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, g, x):
#         g1 = self.W_g(g)
#         x1 = self.W_x(x)
#         psi = self.relu(g1 + x1)
#         psi = self.psi(psi)
#         return x*psi
#
# model = Attention_block(2048,1024,512)
# feat5 = torch.randn(1, 2048, 7, 7)
# feat4 = torch.randn(1, 1024, 14, 14)
# output = model(feat5,feat4)
# print(output.shape)

#
# class Attention_block(nn.Module):
#     def __init__(self, F_g, F_l, F_int,if_convert:bool=False):
#         super(Attention_block, self).__init__()
#         self.W_g = nn.Sequential(
#             #nn.UpsamplingBilinear2d(scale_factor=2),
#             # if if_convert:
#             #     F_g=DySample(F_g)
#             nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(F_int)
#         )
#         self.W_x = nn.Sequential(
#             nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(F_int)
#         )
#         self.psi = nn.Sequential(
#             nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )
#         self.relu = nn.ReLU(inplace=True)
#         if if_convert:
#             self.W_g_convert = DySample(F_g)
#             self.if_convert = True
#         else:
#             self.if_convert = False
#     def forward(self, g, x):
#         if self.if_convert:
#             g = self.W_g_convert(g)
#         else:
#             g = g.permute(0, 3, 1, 2)
#         g1 = self.W_g(g)
#         x1 = self.W_x(x)
#         psi = self.relu(g1 + x1)
#         psi = self.psi(psi)
#         return x*psi
#
# out_filters = [64, 128, 256, 512]
# model = Attention_block(out_filters[0],out_filters[0],int(out_filters[0]/2),if_convert=False)
# decoder_output1= torch.randn(1, 28, 28, 64)
# feat3 = torch.randn(1, 64, 28, 28)
# output = model(decoder_output1,feat3)
# print(output.shape)

# if __name__ == '__main__':
#     x = torch.rand(2, 2048, 7, 7)
#     dys = DySample(2048)
#     print(dys(x).shape) #输出结果为torch.Size([2, 2048, 14, 14])