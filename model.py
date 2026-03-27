import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def transdim(X, ext=5):  # 扩充 sin,cos 维度，使得相邻点变得区别大
    A = X
    for i in range(ext):
        T = (2.0**i) * math.pi
        A = torch.cat([A, torch.sin(T * X), torch.cos(T * X)], dim=-1)
    return A


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, weights):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, weights[0]))
        for i in range(len(weights) - 1):
            self.layers.append(nn.Linear(weights[i], weights[i + 1]))
        self.layers.append(nn.Linear(weights[-1], out_dim))
        with torch.no_grad():  # 给初值，保证尽量都不是负数
            nn.init.constant_(self.layers[-1].bias[:3], 0.0)
            nn.init.constant_(self.layers[-1].bias[3], 0.1)

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        x = self.layers[-1](x)
        return x


def Calc_Light(O, D, Model, n=64, near=2.0, far=6.0):
    # O: [m,3], D:[m,3]
    m = O.shape[0]

    # 1. 生成采样点（分区间随机）
    t_val = torch.linspace(near, far, n + 1)
    L = t_val[:-1]
    R = t_val[1:]
    dis = L + torch.randn([m, n]) * (R - L)  # [m,n]
    points = O[..., None, :] + dis[..., None] * D[..., None, :]  # [m,n,3]

    # 2.用模型计算取点信息
    logits = Model(transdim(points))
    cols = torch.sigmoid(logits[..., :3])  # [m,n,3]，颜色必须是 [0,1]
    dens = F.relu(logits[..., 3])  # [m,n] ，密度必须是非负

    # print(dis[..., 0].shape, (dis[..., 1:] - dis[..., :-1]).shape)

    # 3.带入公式
    deltas = (far - near) / (n + 1)
    alphas = 1 - torch.exp(-deltas * dens)  # [m,n]
    preT = torch.cumprod(1.0 - alphas + 1e-10, dim=-1)  # [m,n]
    preT = torch.cat([torch.ones([m, 1]), preT[..., :-1]], dim=-1)
    weights = preT * alphas  # [m,n]
    res = torch.sum(weights[..., None] * cols, dim=1)  # [m,3]

    # 4.填充背景颜色（白色）
    sumw = torch.sum(weights, dim=-1)  # [m]
    res = res + (1 - sumw[..., None])

    return res
