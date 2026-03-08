import torch
import torch.nn as nn
import torch.nn.functional as F


class GeMPooling(nn.Module):
    def __init__(self,feature_size,pool_size=7, init_norm=3.0, eps=1e-6, normalize=False, **kwargs):
        super(GeMPooling, self).__init__(**kwargs)
        self.feature_size = feature_size  # Final layer channel size, the pow calc at -1 axis
        self.pool_size = pool_size
        self.init_norm = init_norm
        self.p = torch.nn.Parameter(torch.ones(self.feature_size) * self.init_norm, requires_grad=True)
        self.p.data.fill_(init_norm)
        self.normalize = normalize
        self.avg_pooling = nn.AvgPool2d((self.pool_size, self.pool_size))
        self.eps = eps

    def forward(self, features):
        # filter invalid value: set minimum to 1e-6
        # features-> (B, H, W, C)
        x = features * 0.02  # 对输入张量进行缩放
        # filter invalid value: set minimum to 1e-6
        # features-> (B, H, W, C)
        x = x.permute((0, 2, 3, 1))
        x = x.clamp(min=self.eps).pow(self.p)
        x = x.permute((0, 3, 1, 2))
        x = self.avg_pooling(x)
        # x = torch.squeeze(x)
        x = x.permute((0, 2, 3, 1))
        # print("x.shape:",x.shape)
        x = torch.pow(x, (1.0 / self.p))
        # unit vector
        if self.normalize:
            x = F.normalize(x, dim=-1, p=2)
        return x.permute((0, 3, 1, 2))


if __name__ == '__main__':
    # x = torch.randn(8, 64, 64, 150)
    x = torch.randn(2, 150,64, 64)
    gem = GeMPooling(feature_size=150, pool_size=2,
                     #stride=1,
                     init_norm=2.0)

    print("input : ", x.shape)
    print("=========================")
    print(gem(x).shape)