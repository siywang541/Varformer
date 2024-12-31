import math
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        # print('mu, var', mu.mean(), var.mean())
        # d.append([mu.mean(), var.mean()])
        y = (x - mu) / (var + eps).sqrt()
        weight, bias, y = weight.contiguous(), bias.contiguous(), y.contiguous()  # avoid cuda error
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        # y, var, weight = ctx.saved_variables
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6, requires_grad=True):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels), requires_grad=requires_grad))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels), requires_grad=requires_grad))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class LKA(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super().__init__()
        self.conv0 = nn.Conv2d(inp_dim, inp_dim, 5, padding=2, groups=inp_dim)
        self.conv_spatial = nn.Conv2d(inp_dim, inp_dim, 7, stride=1, padding=9, groups=inp_dim, dilation=3)
        self.conv1 = nn.Conv2d(inp_dim, out_dim, 1)

    def forward(self, x):
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return attn


class IMAConv(nn.Module):
    ''' Mutual Affine Convolution (MAConv) layer '''
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True, split=4, n_curve=3):
        super(IMAConv, self).__init__()
        assert split >= 2, 'Num of splits should be larger than one'

        self.num_split = split
        splits = [1 / split] * split
        self.in_split, self.in_split_rest, self.out_split = [], [], []
        self.n_curve = n_curve
        self.relu = nn.ReLU(inplace=False)

        for i in range(self.num_split):
            in_split = round(in_channel * splits[i]) if i < self.num_split - 1 else in_channel - sum(self.in_split)
            in_split_rest = in_channel - in_split
            out_split = round(out_channel * splits[i]) if i < self.num_split - 1 else in_channel - sum(self.out_split)

            self.in_split.append(in_split)
            self.in_split_rest.append(in_split_rest)
            self.out_split.append(out_split)

            setattr(self, 'predictA{}'.format(i), nn.Sequential(*[
                nn.Conv2d(in_split_rest, in_split, 5, stride=1, padding=2),nn.ReLU(inplace=True),
                nn.Conv2d(in_split, in_split, 3, stride=1, padding=1),nn.ReLU(inplace=True),
                nn.Conv2d(in_split, n_curve, 1, stride=1, padding=0),
                nn.Sigmoid()
            ]))
            setattr(self, 'conv{}'.format(i), nn.Conv2d(in_channels=in_split, out_channels=out_split, 
                                                        kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))

    def forward(self, input):
        input = torch.split(input, self.in_split, dim=1)
        output = []

        for i in range(self.num_split):
            a = getattr(self, 'predictA{}'.format(i))(torch.cat(input[:i] + input[i + 1:], 1))
            x = self.relu(input[i]) - self.relu(input[i]-1)
            for j in range(self.n_curve):
                x = x + a[:,j:j+1]*x*(1-x)
            output.append(getattr(self, 'conv{}'.format(i))(x))

        return torch.cat(output, 1)      

class AIEM(nn.Module): 
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., split_group=4, n_curve=3):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca1 = LKA(dw_channel, dw_channel//2)
        self.sca2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.IMAC = IMAConv(in_channel=ffn_channel // 2, out_channel=ffn_channel // 2, kernel_size=3, stride=1, padding=1, bias=True, split=split_group, n_curve=n_curve)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.sg(x)
        x = (self.sca2(x) * self.sca1(x)) * self.sg(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.IMAC(x)
        x = self.conv5(x)


        x = self.dropout2(x)

        return y + x * self.gamma

class EnhanceLayers(nn.Module):
    def __init__(self, embed_dim=256, n_layers=4, split_group=4, n_curve=3):
        super().__init__()
        self.blks = nn.ModuleList()
        for i in range(n_layers):
            layer = AIEM(embed_dim, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., split_group=split_group, n_curve=n_curve)
            self.blks.append(layer)
    
    def forward(self, x):
        for m in self.blks:
            x = m(x)
        return x
