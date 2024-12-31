import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2) # BCHW -> BNC
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += self.embed_dim * H * W
        return flops
class PatchUnembed(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)
    return x

import torch
from timm.layers import trunc_normal_
from torch import nn


class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # Calculate flops for one window attention module
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * self.dim * 3
        # attn = (q @ k.transpose(-2, -1))
        flops += N * self.num_heads * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += N * self.num_heads * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops
    
from timm.layers import to_2tuple, DropPath
from torch import nn
import torch

class MLP(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class SwinTransformerBlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        attn_mask = self.calculate_mask(self.input_resolution) if self.shift_size > 0 else None

        self.register_buffer('attn_mask', attn_mask)

    def calculate_mask(self, input_resolution):
        H, W = input_resolution
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * self.dim * self.dim * H * W * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops
            
class BasicSwinLayer(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops
    
class ResidualSwinTransformerBlock(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super(ResidualSwinTransformerBlock, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicSwinLayer(dim=dim,
                                             input_resolution=input_resolution,
                                             depth=depth,
                                             num_heads=num_heads,
                                             window_size=window_size,
                                             mlp_ratio=mlp_ratio,
                                             qkv_bias=qkv_bias, qk_scale=qk_scale,
                                             drop=drop, attn_drop=attn_drop,
                                             drop_path=drop_path,
                                             norm_layer=norm_layer,
                                             downsample=downsample,
                                             use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None
        )

        self.patch_unembed = PatchUnembed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None
        )

    def forward(self, x, x_size):
        return self.patch_embed(
            self.conv(
                self.patch_unembed(
                    self.residual_group(x, x_size), x_size
                )
            )
        ) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += self.dim * self.dim * H * W * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()
        return flops

class RACMoudle(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size):
        super(RACMoudle, self).__init__()
        self.conv1x1 = nn.Conv2d(dim*2, dim*2, kernel_size=1)
        self.swin_blocks = nn.Sequential(*[ResidualSwinTransformerBlock(dim*2,input_resolution,depth, num_heads, window_size) for _ in range(2)])
        self.softmax = nn.Softmax(dim=1)
    def forward(self, z, zq):
        # Concatenate features along the channel dimension
        concatenated = torch.cat((z, zq), dim=1)  # Shape: (1, 64, 16, 16)

        # Reshape for MultiheadAttention
        B, C, H, W = concatenated.shape
        concatenated = concatenated.view(B, C, H * W)  # Shape: (1, 64, 256)
        concatenated = concatenated.permute(0, 2, 1)
        # Pass through Residual Swin Transformer Blocks
        for b in self.swin_blocks:
            concatenated = b(concatenated, x_size=(H,W))        
        output = concatenated


        output = output.permute(0, 2, 1)
        # Reshape back to original shape
        output = output.view(B, C, H, W)  # Shape: (1, 64, 16, 16)

        # 1x1 convolution
        output = self.conv1x1(output)

        # Softmax activation
        softmax_output = self.softmax(output.mean(dim=[2, 3]))  # Take the mean over spatial dimensions

        # Split the concatenated features to obtain two weight matrices
        w_z, w_zq = softmax_output.chunk(2,dim=1)

        # Matrix multiplication with original features
        weighted_z = w_z.unsqueeze(-1).unsqueeze(-1) * z
        weighted_zq = w_zq.unsqueeze(-1).unsqueeze(-1) * zq

        # Element-wise addition
        final_output = weighted_z + weighted_zq

        return final_output





class RACMoudle3(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size):
        super(RACMoudle3, self).__init__()
        self.conv1x1 = nn.Conv2d(dim*2, 2, kernel_size=1)
        self.swin_blocks = nn.Sequential(*[ResidualSwinTransformerBlock(dim*2,input_resolution,depth, num_heads, window_size) for _ in range(2)])
        self.softmax = nn.Softmax(dim=1)
    def forward(self, z, zq):
        # Concatenate features along the channel dimension
        concatenated = torch.cat((z, zq), dim=1)  # Shape: (1, 64, 16, 16)

        # Reshape for MultiheadAttention
        B, C, H, W = concatenated.shape
        concatenated = concatenated.view(B, C, H * W)  # Shape: (1, 64, 256)
        concatenated = concatenated.permute(0, 2, 1)
        # Pass through Residual Swin Transformer Blocks
        for b in self.swin_blocks:
            concatenated = b(concatenated, x_size=(H,W))        
        output = concatenated


        output = output.permute(0, 2, 1)
        # Reshape back to original shape
        output = output.view(B, C, H, W)  # Shape: (1, 64, 16, 16)

        # 1x1 convolution
        output = self.conv1x1(output)


        # Softmax activation
        softmax_output = self.softmax(output)  # Take the mean over spatial dimensions
        # Split the concatenated features to obtain two weight matrices
        w_z, w_zq = softmax_output.chunk(2,dim=1)

        # Matrix multiplication with original features
        weighted_z = torch.einsum('nhw,nchw->nchw', w_z.squeeze(1), z)
        weighted_zq = torch.einsum('nhw,nchw->nchw', w_zq.squeeze(1), zq)

        # Element-wise addition
        final_output = weighted_z + weighted_zq

        return final_output, weighted_z



# AdaCode
class WPMoudle3(nn.Module):
    def __init__(self, dim_in, dim_out, input_resolution, depth, num_heads, window_size):
        super(WPMoudle3, self).__init__()
        self.conv1x1 = nn.Conv2d(dim_in, dim_out, kernel_size=1)
        self.swin_blocks = nn.Sequential(*[ResidualSwinTransformerBlock(dim_in,input_resolution,depth, num_heads, window_size) for _ in range(2)])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, q_x, v_x):
        # Concatenate features along the channel dimension

        # Reshape for MultiheadAttention
        B, C, H, W = q_x.shape
        q_x = q_x.view(B, C, H * W)  # Shape: (1, 64, 256)
        q_x = q_x.permute(0, 2, 1)
        # Pass through Residual Swin Transformer Blocks
        for b in self.swin_blocks:
            q_x = b(q_x, x_size=(H,W))        
        output = q_x


        output = output.permute(0, 2, 1)
        # Reshape back to original shape
        output = output.view(B, C, H, W)  # Shape: (1, 64, 16, 16)

        # 1x1 convolution
        output = self.conv1x1(output)
        output = self.global_pool(output).squeeze(-1).squeeze(-1).unsqueeze(1)
        # Softmax activation



        # Matrix multiplication with original features

        final_output = torch.einsum('nkr,nrchw->nkchw', output, v_x)
        final_output = final_output.squeeze(1)



        return final_output









class Upsample2x(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))
class Upsample4x(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=4, mode='nearest'))
class Upsample8x(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=8, mode='nearest'))

class Upsample16x(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=16, mode='nearest'))
def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

def InNormalize(in_channels):
    return torch.nn.InstanceNorm2d(num_features=in_channels)

# AdaCode
class WPMoudle4(nn.Module):
    def __init__(self, dim_in, dim_out, out_dim, up_r,input_resolution, depth, num_heads, window_size):
        super(WPMoudle4, self).__init__()
        self.conv1x1 = nn.Conv2d(dim_in, dim_out, kernel_size=1)
        self.swin_blocks = nn.Sequential(*[ResidualSwinTransformerBlock(dim_in,input_resolution,depth, num_heads, window_size) for _ in range(2)])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.norm_3= InNormalize(320)
        self.conv_out = nn.Conv2d(320, out_dim, kernel_size=1)
        self.up_r = up_r
        if self.up_r == 2:
            self.upsample=Upsample2x(out_dim)
        if self.up_r == 4:
            self.upsample=Upsample4x(out_dim)
        if self.up_r == 8:
            self.upsample=Upsample8x(out_dim)
        if self.up_r == 16:
            self.upsample=Upsample16x(out_dim)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, q_x, v_x):
        # Concatenate features along the channel dimension

        # Reshape for MultiheadAttention
        B, C, H, W = q_x.shape
        q_x = q_x.view(B, C, H * W)  # Shape: (1, 64, 256)
        q_x = q_x.permute(0, 2, 1)
        # Pass through Residual Swin Transformer Blocks
        for b in self.swin_blocks:
            q_x = b(q_x, x_size=(H,W))        
        output = q_x


        output = output.permute(0, 2, 1)
        # Reshape back to original shape
        output = output.view(B, C, H, W)  # Shape: (1, 64, 16, 16)

        # 1x1 convolution
        output = self.conv1x1(output)
        output = self.global_pool(output).squeeze(-1).squeeze(-1)
        # Softmax activation
        # softmax_output = self.softmax(output.mean(dim=[2, 3]))  # Take the mean over spatial dimensions

        final_output = torch.einsum('nr,nrchw->nrchw', output, v_x)
        B,R,C,H,W = final_output.shape
        final_output = final_output.reshape( B,R*C,H,W)

        final_output = self.norm_3(final_output)
        final_output = self.conv_out(final_output)
        if self.up_r != 1:
            final_output = self.upsample(final_output)


        return final_output


# AdaCode  
class WPMoudle4_norm(nn.Module):
    def __init__(self, dim_in, dim_out, out_dim, up_r,input_resolution, depth, num_heads, window_size):
        super(WPMoudle4_norm, self).__init__()
        
        self.swin_blocks = nn.Sequential(*[ResidualSwinTransformerBlock(dim_in,input_resolution,depth, num_heads, window_size) for _ in range(2)])
        self.conv1x1 = nn.Conv2d(dim_in, dim_out, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.norm_3= InNormalize(320)
        self.conv_out = nn.Conv2d(320, out_dim, kernel_size=1)
        self.up_r = up_r
        if self.up_r == 2:
            self.upsample=Upsample2x(out_dim)
        if self.up_r == 4:
            self.upsample=Upsample4x(out_dim)
        if self.up_r == 8:
            self.upsample=Upsample8x(out_dim)
        if self.up_r == 16:
            self.upsample=Upsample16x(out_dim)


    def forward(self, q_x, v_x):  # enc_feat_dict_var  prompt_hat 
        # Concatenate features along the channel dimension
        # concatenated = torch.cat((z, zq), dim=1)  # Shape: (1, 64, 16, 16)

        # Reshape for MultiheadAttention
        B, C, H, W = q_x.shape
        q_x = q_x.view(B, C, H * W)  # Shape: (1, 64, 256)
        q_x = q_x.permute(0, 2, 1)
        # Pass through Residual Swin Transformer Blocks
        for b in self.swin_blocks:
            q_x = b(q_x, x_size=(H,W))        
        output = q_x


        output = output.permute(0, 2, 1)
        # Reshape back to original shape
        output = output.view(B, C, H, W)  # Shape: (1, 64, 16, 16)

        # 1x1 convolution
        output = self.conv1x1(output)
        output = self.global_pool(output).squeeze(-1).squeeze(-1)
        output = F.softmax(output, dim=1)

        final_output = torch.einsum('nr,nrchw->nrchw', output, v_x)
        B,R,C,H,W = final_output.shape
        final_output = final_output.reshape( B,R*C,H,W)

        final_output = self.norm_3(final_output)
        final_output = self.conv_out(final_output)
        if self.up_r != 1:
            final_output = self.upsample(final_output)


        return final_output


class RACMoudle2(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size):
        super(RACMoudle2, self).__init__()
        self.conv1x1 = nn.Conv2d(dim*2, 2, kernel_size=1)
        self.swin_blocks = nn.Sequential(*[ResidualSwinTransformerBlock(dim*2,input_resolution,depth, num_heads, window_size) for _ in range(2)])
        self.softmax = nn.Softmax(dim=1)
    def forward(self, z, zq):
        # Concatenate features along the channel dimension
        concatenated = torch.cat((z, zq), dim=1)  # Shape: (1, 64, 16, 16)

        # Reshape for MultiheadAttention
        B, C, H, W = concatenated.shape
        concatenated = concatenated.view(B, C, H * W)  # Shape: (1, 64, 256)
        concatenated = concatenated.permute(0, 2, 1)
        # Pass through Residual Swin Transformer Blocks
        for b in self.swin_blocks:
            concatenated = b(concatenated, x_size=(H,W))        
        output = concatenated


        output = output.permute(0, 2, 1)
        # Reshape back to original shape
        output = output.view(B, C, H, W)  # Shape: (1, 64, 16, 16)

        # 1x1 convolution
        output = self.conv1x1(output)


        # Softmax activation
        softmax_output = self.softmax(output)  # Take the mean over spatial dimensions
        # Split the concatenated features to obtain two weight matrices
        w_z, w_zq = softmax_output.chunk(2,dim=1)


        # Matrix multiplication with original features
        weighted_z = torch.einsum('nhw,nchw->nchw', w_z.squeeze(1), z)
        weighted_zq = torch.einsum('nhw,nchw->nchw', w_zq.squeeze(1), zq)


        # Element-wise addition
        final_output = weighted_z + weighted_zq

        return final_output

