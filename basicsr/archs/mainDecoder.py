import math
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange
from ops.dcn import ModulatedDeformConvPack, modulated_deform_conv
from utils import get_root_logger
from distutils.version import LooseVersion
import sys


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


class TransAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(TransAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class IKBAFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, att, selfk, selfg, selfb, selfw):
        B, nset, H, W = att.shape
        KK = selfk
        selfc = x.shape[1]

        att = att.reshape(B, nset, H * W).transpose(-2, -1)

        ctx.selfk, ctx.selfg, ctx.selfc, ctx.KK, ctx.nset = selfk, selfg, selfc, KK, nset
        ctx.x, ctx.att, ctx.selfb, ctx.selfw = x, att, selfb, selfw

        bias = att @ selfb
        attk = att @ selfw

        attk_h = attk[:,:,:attk.shape[-1]//2]
        attk_w = attk[:,:,attk.shape[-1]//2:]

        uf_h = torch.nn.functional.unfold(x, kernel_size=(selfk, 1), padding=(selfk // 2, 0))

        # for unfold att / less memory cost
        uf_h = uf_h.reshape(B, selfg, selfc // selfg * KK, H * W).permute(0, 3, 1, 2)  # B, H * W, selfg, selfc // selfg * KK, 1
        attk_h = attk_h.reshape(B, H * W, selfg, selfc // selfg, selfc // selfg * KK)  # B, H * W, selfg, selfc // selfg, selfc // selfg * KK

        x_h = attk_h @ uf_h.unsqueeze(-1)  # B, H * W, selfg, selfc // selfg
        del attk, attk_h, uf_h
        x_h = x_h.squeeze(-1).reshape(B, H * W, selfc)
        x_h = x_h.transpose(-1, -2).reshape(B, selfc, H, W)

        uf_w = torch.nn.functional.unfold(x_h, kernel_size=(1, selfk), padding=(0, selfk // 2))
        uf_w = uf_w.reshape(B, selfg, selfc // selfg * KK, H * W).permute(0, 3, 1, 2)
        attk_w = attk_w.reshape(B, H * W, selfg, selfc // selfg, selfc // selfg * KK)

        x_w = attk_w @ uf_w.unsqueeze(-1)  # B, H * W, selfg, selfc // selfg
        del attk_w, uf_w
        x_w = x_w.squeeze(-1).reshape(B, H * W, selfc) + bias
        x_w = x_w.transpose(-1, -2).reshape(B, selfc, H, W)

        return x_w

    @staticmethod
    def backward(ctx, grad_output):
        x, att, selfb, selfw = ctx.x, ctx.att, ctx.selfb, ctx.selfw
        selfk, selfg, selfc, KK, nset = ctx.selfk, ctx.selfg, ctx.selfc, ctx.KK, ctx.nset

        B, selfc, H, W = grad_output.size()

        dbias = grad_output.reshape(B, selfc, H * W).transpose(-1, -2)

        dselfb = att.transpose(-2, -1) @ dbias
        datt = dbias @ selfb.transpose(-2, -1)

        attk = att @ selfw

        attk_h = attk[:, :, :attk.shape[-1] // 2]
        attk_w = attk[:, :, attk.shape[-1] // 2:]

        uf_h = torch.nn.functional.unfold(x, kernel_size=(selfk, 1), padding=(selfk // 2, 0))

        # for unfold att / less memory cost
        uf_h = uf_h.reshape(B, selfg, selfc // selfg * KK, H * W).permute(0, 3, 1,
                                                                          2)  # B, H * W, selfg, selfc // selfg * KK, 1
        attk_h = attk_h.reshape(B, H * W, selfg, selfc // selfg,
                                selfc // selfg * KK)  # B, H * W, selfg, selfc // selfg, selfc // selfg * KK

        x_h = attk_h @ uf_h.unsqueeze(-1)  # B, H * W, selfg, selfc // selfg
        x_h = x_h.squeeze(-1).reshape(B, H * W, selfc)
        x_h = x_h.transpose(-1, -2).reshape(B, selfc, H, W)

        uf_w = torch.nn.functional.unfold(x_h, kernel_size=(1, selfk), padding=(0, selfk // 2))
        uf_w = uf_w.reshape(B, selfg, selfc // selfg * KK, H * W).permute(0, 3, 1, 2)
        attk_w = attk_w.reshape(B, H * W, selfg, selfc // selfg, selfc // selfg * KK)

        dx = dbias.view(B, H * W, selfg, selfc // selfg, 1)

        dattk_w = dx @ uf_w.view(B, H * W, selfg, 1, selfc // selfg * KK)
        duf_w = attk_w.transpose(-2, -1) @ dx
        del attk, attk_w, uf_w

        duf_w = duf_w.permute(0, 2, 3, 4, 1).view(B, -1, H * W)
        dx_h = F.fold(duf_w, output_size=(H, W), kernel_size=(1, selfk), padding=(0, selfk // 2))

        dx_h = dx_h.reshape(B, selfc, H * W).view(B, H * W, selfg, selfc // selfg, 1)
        dattk_h = dx_h @ uf_h.view(B, H * W, selfg, 1, selfc // selfg * KK)
        duf_h = attk_h.transpose(-2, -1) @ dx_h

        dattk = torch.cat((dattk_h, dattk_w), -1)
        dattk = dattk.view(B, H * W, -1)
        datt += dattk @ selfw.transpose(-2, -1)
        dselfw = att.transpose(-2, -1) @ dattk

        duf_h = duf_h.permute(0, 2, 3, 4, 1).view(B, -1, H * W)
        dx = F.fold(duf_h, output_size=(H, W), kernel_size=(selfk, 1), padding=(selfk // 2, 0))

        datt = datt.transpose(-1, -2).view(B, nset, H, W)

        return dx, datt, None, None, dselfb, dselfw


class HFF(nn.Module):
    def __init__(self, dim=256, bias=False, gc=4, nset=128, k=3):
        super(HFF, self).__init__()
        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,
                      groups=dim, bias=bias),
        )

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # Simplified Channel Attention
        self.lka = LKA(dim)


        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,
                      groups=dim, bias=bias),
        )

        c = dim
        self.k, self.c = k, c
        self.nset = nset

        self.g = c // gc
        self.w = nn.Parameter(torch.zeros(1, nset, c * c // self.g * self.k * 2))
        self.b = nn.Parameter(torch.zeros(1, nset, c))
        self.init_p(self.w, self.b)
        interc = min(dim, 32)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=interc, kernel_size=3, padding=1, stride=1, groups=interc,
                      bias=True),
            SimpleGate(),
            nn.Conv2d(interc // 2, self.nset, 1, padding=0, stride=1),
        )
        self.conv211 = nn.Conv2d(in_channels=dim, out_channels=self.nset, kernel_size=1)
        self.attgamma = nn.Parameter(torch.zeros((1, self.nset, 1, 1)) + 1e-2, requires_grad=True)
        self.ga1 = nn.Parameter(torch.zeros((1, c, 1, 1)) + 1e-2, requires_grad=True)

    def init_p(self, weight, bias=None):
        init.kaiming_uniform_(weight, a=math.sqrt(5))
        if bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(bias, -bound, bound)

    def IKBA(self, x, att, selfk, selfg, selfb, selfw):
        return IKBAFunction.apply(x, att, selfk, selfg, selfb, selfw)

    def forward(self, x):
        sca = self.sca(x)
        x1 = self.dwconv(x)
        lka = self.lka(x)

        att = self.conv2(x) * self.attgamma + self.conv211(x)
        uf = self.conv1(x)
        x2 = self.IKBA(uf, att, self.k, self.g, self.b, self.w) * self.ga1 + uf

        x = x1 * x2
        x = x * sca * lka
        x = self.project_out(x)
        return x


class HIMB(nn.Module):
    def __init__(self, dim=256, num_heads=8, bias=True):
        super(HIMB, self).__init__()
        self.norm1 = LayerNorm2d(dim)
        self.norm2 = LayerNorm2d(dim)

        self.attn = HFF(dim, bias)
        self.ffn = TransAttention(dim, num_heads, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

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


class AIEM1(nn.Module): 
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
        self.sca1 = LKA(dw_channel//2, dw_channel//2)
        self.sca2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel//2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
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
        x = self.sg(x)
        x = self.sca2(x) * x + self.sca1(x) * x
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.IMAC(x)
        x = self.conv5(x)


        x = self.dropout2(x)

        return y + x * self.gamma


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

 

class TWAM(nn.Module):
    def __init__(self, c, num_heads):
        super(TWAM, self).__init__()
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.num_heads = num_heads
        self.norm_d = LayerNorm2d(c)
        self.norm_g = LayerNorm2d(c)
        self.d_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.g_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.d_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.g_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

    def forward(self, x_d, x_g):
        b,c,h,w = x_d.shape

        Q_d = self.d_proj1(self.norm_d(x_d))  # B, C, H, W
        Q_g_T = self.g_proj1(self.norm_g(x_g)) # B, C, H, W

        V_d = self.d_proj2(x_d)  # B, C, H, W
        V_g = self.g_proj2(x_g) # B, C, H, W

        Q_d = rearrange(Q_d, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        Q_g_T = rearrange(Q_g_T, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        V_d = rearrange(V_d, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        V_g = rearrange(V_g, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        Q_d = torch.nn.functional.normalize(Q_d, dim=-1)
        Q_g_T = torch.nn.functional.normalize(Q_g_T, dim=-1)

        # (B, head, c, hw) x (B, head, hw, c) -> (B, head, c, c)
        attention = (Q_d @ Q_g_T.transpose(-2,-1)) * self.scale

        F_g2d = torch.matmul(torch.softmax(attention, dim=-1), V_g)  # B, head, c, hw
        F_d2g = torch.matmul(torch.softmax(attention, dim=-1), V_d)  # B, head, c, hw

        # scale
        F_g2d = rearrange(F_g2d, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        F_d2g = rearrange(F_d2g, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return x_d + F_g2d * self.beta, x_g + F_d2g * self.gamma




class TWAM2(nn.Module): 
    def __init__(self, c,bf_g_c, num_heads):
        super(TWAM2, self).__init__()
        super().__init__()
        self.bf_g = nn.Conv2d(bf_g_c, c, kernel_size=1, stride=1, padding=0)

        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.num_heads = num_heads
        self.norm_d = LayerNorm2d(c)
        self.norm_g = LayerNorm2d(c)
        self.d_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.g_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.d_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.g_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

    def forward(self, x_d, x_g):
        b,c,h,w = x_d.shape
        x_g = self.bf_g(x_g)
        Q_d = self.d_proj1(self.norm_d(x_d))  # B, C, H, W
        Q_g_T = self.g_proj1(self.norm_g(x_g)) # B, C, H, W

        V_d = self.d_proj2(x_d)  # B, C, H, W
        V_g = self.g_proj2(x_g) # B, C, H, W

        Q_d = rearrange(Q_d, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        Q_g_T = rearrange(Q_g_T, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        V_d = rearrange(V_d, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        V_g = rearrange(V_g, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        Q_d = torch.nn.functional.normalize(Q_d, dim=-1)
        Q_g_T = torch.nn.functional.normalize(Q_g_T, dim=-1)

        # (B, head, c, hw) x (B, head, hw, c) -> (B, head, c, c)
        attention = (Q_d @ Q_g_T.transpose(-2,-1)) * self.scale

        F_g2d = torch.matmul(torch.softmax(attention, dim=-1), V_g)  # B, head, c, hw
        F_d2g = torch.matmul(torch.softmax(attention, dim=-1), V_d)  # B, head, c, hw

        # scale
        F_g2d = rearrange(F_g2d, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        F_d2g = rearrange(F_d2g, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return x_d + F_g2d * self.beta, x_g + F_d2g * self.gamma



class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.
    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.
    Ref:
        Delving Deep into Deformable Alignment in Video Super-Resolution.
    """

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 50:
            logger = get_root_logger()
            # logger.warning(f'Offset abs mean is {offset_absmean}, larger than 50.')

        if LooseVersion(torchvision.__version__) >= LooseVersion('0.9.0'):
            return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                                 self.dilation, mask)
        else:
            return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding,
                                         self.dilation, self.groups, self.deformable_groups)

class TextureWarpingModule(nn.Module):

    def __init__(self, channel, cond_channels, deformable_groups, previous_offset_channel=0):
        super(TextureWarpingModule, self).__init__()
        self.offset_conv1 = nn.Sequential(
            nn.Conv2d(channel + cond_channels, channel, kernel_size=1),
            nn.GroupNorm(num_groups=32, num_channels=channel, eps=1e-6, affine=True), nn.SiLU(inplace=True),
            nn.Conv2d(channel, channel, groups=channel, kernel_size=7, padding=3),
            nn.GroupNorm(num_groups=32, num_channels=channel, eps=1e-6, affine=True), nn.SiLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=1))

        self.offset_conv2 = nn.Sequential(
            nn.Conv2d(channel + previous_offset_channel, channel, 3, 1, 1),
            nn.GroupNorm(num_groups=32, num_channels=channel, eps=1e-6, affine=True), nn.SiLU(inplace=True))
        self.dcn = DCNv2Pack(channel, channel, 3, padding=1, deformable_groups=deformable_groups)

    def forward(self, x_main, inpfeat, previous_offset=None):
        _, _, h, w = inpfeat.shape
        offset = self.offset_conv1(torch.cat([inpfeat, x_main], dim=1))
        if previous_offset is None:
            offset = self.offset_conv2(offset)
        else:
            offset = self.offset_conv2(torch.cat([offset, previous_offset], dim=1))
        warp_feat = self.dcn(x_main, offset)
        return warp_feat, offset


class ResnetBlock(nn.Module):

    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=channels_in, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=(3, 3), stride=(1, 1), padding=1)

        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=channels_out, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=(3, 3), stride=(1, 1), padding=1)

        self.act = nn.SiLU(inplace=True)
        if channels_in != channels_out:
            self.residual_func = nn.Conv2d(channels_in, channels_out, kernel_size=1)
        else:
            self.residual_func = nn.Identity()

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)
        return x + self.residual_func(residual)


