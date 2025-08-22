import torch.utils.checkpoint as checkpoint

import torch
import torch.nn.functional as F
import torch.nn as nn
import scipy
from math import ceil, floor
from timm.models.layers import DropPath, trunc_normal_, to_2tuple
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count_table, parameter_count
from functools import partial


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor):
        '''
        x: (b c h w)
        '''
        x = x.permute(0, 2, 3, 1).contiguous()  # (b h w c)
        x = self.norm(x)  # (b h w c)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, 3, 2, 1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, 3, 1, 1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, 2, 1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.BatchNorm2d(embed_dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).permute(0, 2, 3, 1)  # (b h w c)
        return x


class DWConv2d(nn.Module):

    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        '''
        x: (b c h w)
        '''
        x = self.conv(x)  # (b c h w)
        return x


class GRNwithNHWC(nn.Module):
    """ GRN (Global Response Normalization) layer
    Originally proposed in ConvNeXt V2 (https://arxiv.org/abs/2301.00808)
    This implementation is more efficient than the original (https://github.com/facebookresearch/ConvNeXt-V2)
    We assume the inputs to this layer are (N, H, W, C)
    """

    def __init__(self, dim, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        if self.use_bias:
            self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        if self.use_bias:
            return (self.gamma * Nx + 1) * x + self.beta
        else:
            return (self.gamma * Nx + 1) * x


class FeedForwardNetwork(nn.Module):
    def __init__(
            self,
            embed_dim,
            ffn_dim,
            groups=1,
            activation_fn='custom',
            dropout=0.0,
            activation_dropout=0.0,
            layernorm_eps=1e-6,
            subln=False,
            subconv=True,
            input_dim=None
    ):
        super().__init__()

        if input_dim != None:
            self.input_dim = input_dim
        else:
            self.input_dim = embed_dim

        self.embed_dim = embed_dim

        if activation_fn == 'custom':
            self.activation_fn = nn.Sequential(
                nn.GELU(),
                GRNwithNHWC(ffn_dim, use_bias=True))
        else:
            self.activation_fn = F.gelu

        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Conv2d(self.input_dim, ffn_dim, kernel_size=1, stride=1, padding=0, groups=groups)
        self.fc2 = nn.Conv2d(ffn_dim, self.embed_dim, kernel_size=1, stride=1, padding=0)
        self.ffn_layernorm = LayerNorm2d(ffn_dim) if subln else None
        self.dwconv = DWConv2d(ffn_dim, 3, 1, 1) if subconv else None

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        if self.ffn_layernorm is not None:
            self.ffn_layernorm.reset_parameters()

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        residual = x
        if self.dwconv is not None:
            x = self.dwconv(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = x + residual
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x


class EFG(nn.Module):
    def __init__(self, dim, groups, mode, bessel_num=None, input_param=None):
        super().__init__()
        self.dim = dim

        self.filter = None
        self.param = None

        self.groups = groups
        self.mode = mode
        self.bessel_num = bessel_num

        if self.mode == 'Bessel':
            self.param = nn.Parameter(
                torch.randn((dim // groups * 2, self.bessel_num)), requires_grad=True)
        self.reset(input_param)

    def reset(self, input_param):
        if self.mode == 'SPA':
            filter = [self._CircleFilter(input_param['pad_H'], input_param['pad_W'], r, lamb).unsqueeze(0) for r, lamb in
                      zip(input_param['r'], input_param['lamb'])]
            filter = torch.stack(filter, dim=0)
            self.filter = nn.Parameter(filter, requires_grad=False)
        elif self.mode == 'GF':
            self.filter = nn.Parameter(
                torch.zeros(self.head, self.dim // self.head, input_param['pad_H'], input_param['pad_W'], 2),
                requires_grad=False)
        elif self.mode == 'Bessel':
            filter = [self._BesselFilter(input_param['pad_H'], input_param['pad_W'], b).unsqueeze(0) for b in
                      range(self.bessel_num)]
            filter = torch.concat(filter, dim=0)
            self.filter = nn.Parameter(filter, requires_grad=False)
        else:
            self.filter = None
            raise Exception("Error: Unknown model type.")

    def _CircleFilter(self, H, W, r, lamb):
        """
            --image size (H,W)
            --r : radius
        """
        X, Y = torch.meshgrid(torch.fft.fftfreq(H), torch.fft.rfftfreq(W))
        circle = torch.sqrt(X ** 2 + Y ** 2)

        lp_F = (circle < r).clone().to(torch.float32)
        hp_F = (circle > r).clone().to(torch.float32)

        combined_Filter = lp_F * lamb + hp_F * (1 - lamb)  # (H, W)
        combined_Filter[~(circle < r) & ~(circle > r)] = 1 / 3  # cutoff

        return combined_Filter

    def _BesselFilter(self, H, W, b):
        """
            --image size (H,W)
            --r : radius
        """
        X, Y = torch.meshgrid(torch.fft.fftfreq(H), torch.fft.rfftfreq(W))

        circle = torch.sqrt(X ** 2 + Y ** 2).numpy() * 16
        circle = scipy.special.spherical_jn(b, circle)

        circle = torch.tensor(circle).to(torch.float32)

        return circle

    def forward(self):
        assert self.filter is not None

        if self.param is None:
            return self.filter
        else:
            output = torch.complex(
                *torch.split(torch.einsum("hn, nij -> hij", self.param, self.filter).unsqueeze(0).unsqueeze(0),
                             self.dim // self.groups, dim=2))
            output.imag *= -1
            return output


class IFG(nn.Module):
    def __init__(self, dim, pe_dim, ffn_dim, groups, input_params=None):
        super().__init__()

        self.dim = dim * 2
        self.groups = groups

        self.ffn = FeedForwardNetwork(self.dim // self.groups, self.dim // self.groups * 3, input_dim=3, groups=3)
        self.reset(input_params)

    def reset(self, input_params):
        X, Y = torch.meshgrid(torch.fft.fftfreq(input_params['pad_H']), torch.fft.rfftfreq(input_params['pad_W']))

        fe = torch.stack([X, Y, X ** 2 + Y ** 2], dim=0).reshape(1, 3, X.shape[-2], X.shape[-1])
        self.fe = nn.Parameter(fe, requires_grad=False)

    def forward(self):
        # pre-calculate the positional encoding in inference
        _, _, H, W = self.fe.shape
        w = self.ffn(self.fe)

        output = torch.complex(
            *torch.split(w.view(1, 1, self.dim // self.groups, H, W), self.dim // self.groups // 2, dim=2))
        output.imag *= -1
        return output


# ================== This function decides which conv implementation (the native or iGEMM) to use
#   Note that iGEMM large-kernel conv impl will be used if
#       -   you attempt to do so (attempt_to_use_large_impl=True), and
#       -   it has been installed (follow https://github.com/AILab-CVC/UniRepLKNet), and
#       -   the conv layer is depth-wise, stride = 1, non-dilated, kernel_size > 5, and padding == kernel_size // 2
def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
               attempt_use_lk_impl=True):
    kernel_size = to_2tuple(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = to_2tuple(padding)
    need_large_impl = kernel_size[0] == kernel_size[1] and kernel_size[0] > 5 and padding == (
        kernel_size[0] // 2, kernel_size[1] // 2)

    if attempt_use_lk_impl and need_large_impl:
        print('---------------- trying to import iGEMM implementation for large-kernel conv')
        try:
            from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
            print('---------------- found iGEMM implementation ')
        except:
            DepthWiseConv2dImplicitGEMM = None
            print(
                '---------------- found no iGEMM. use original conv. follow https://github.com/AILab-CVC/UniRepLKNet to install it.')
        if DepthWiseConv2dImplicitGEMM is not None and need_large_impl and in_channels == out_channels \
                and out_channels == groups and stride == 1 and dilation == 1:
            print(f'===== iGEMM Efficient Conv Impl, channels {in_channels}, kernel size {kernel_size} =====')
            return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


def get_bn(dim, use_sync_bn=False):
    if use_sync_bn:
        return nn.SyncBatchNorm(dim)
    else:
        return nn.BatchNorm2d(dim)


def fuse_bn(conv, bn):
    conv_bias = 0 if conv.bias is None else conv.bias
    std = (bn.running_var + bn.eps).sqrt()
    return conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1), bn.bias + (
            conv_bias - bn.running_mean) * bn.weight / std


def convert_dilated_to_nondilated(kernel, dilate_rate):
    identity_kernel = torch.ones((1, 1, 1, 1)).to(kernel.device)
    if kernel.size(1) == 1:
        #   This is a DW kernel
        dilated = F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
        return dilated
    else:
        #   This is a dense or group-wise (but not DW) kernel
        slices = []
        for i in range(kernel.size(1)):
            dilated = F.conv_transpose2d(kernel[:, i:i + 1, :, :], identity_kernel, stride=dilate_rate)
            slices.append(dilated)
        return torch.cat(slices, dim=1)


def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
    large_k = large_kernel.size(2)
    dilated_k = dilated_kernel.size(2)
    equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    rows_to_pad = large_k // 2 - equivalent_kernel_size // 2
    merged_kernel = large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)
    return merged_kernel


class DilatedReparamBlock(nn.Module):
    """
    Dilated Reparam Block proposed in UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet)
    We assume the inputs to this block are (N, C, H, W)
    """

    def __init__(self, channels, kernel_size, deploy, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv2d(channels, channels, kernel_size, stride=1,
                                    padding=kernel_size // 2, dilation=1, groups=channels, bias=deploy,
                                    attempt_use_lk_impl=attempt_use_lk_impl)
        self.attempt_use_lk_impl = attempt_use_lk_impl

        #   Default settings. We did not tune them carefully. Different settings may work better.
        if kernel_size == 17:
            self.kernel_sizes = [5, 9, 3, 3, 3]
            self.dilates = [1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 5, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [5, 5, 3, 3]
            self.dilates = [1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3]
            self.dilates = [1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        if not deploy:
            self.origin_bn = get_bn(channels, use_sync_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                                 nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels,
                                           bias=False))
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), get_bn(channels, use_sync_bn=use_sync_bn))

    def forward(self, x):
        if not hasattr(self, 'origin_bn'):  # deploy mode
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            out = out + bn(conv(x))
        return out

    def merge_dilated_branches(self):
        if hasattr(self, 'origin_bn'):
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
                bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
                branch_k, branch_b = fuse_bn(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
            merged_conv = get_conv2d(origin_k.size(0), origin_k.size(0), origin_k.size(2), stride=1,
                                     padding=origin_k.size(2) // 2, dilation=1, groups=origin_k.size(0), bias=True,
                                     attempt_use_lk_impl=self.attempt_use_lk_impl)
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__('origin_bn')
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__('dil_conv_k{}_{}'.format(k, r))
                self.__delattr__('dil_bn_k{}_{}'.format(k, r))


class MGFMixer(nn.Module):
    def __init__(self, dim, groups, pe_dim, pe_ffn_dim,
                use_ifg=True, use_efg=True, use_cfg=True,
                input_param=dict()):
        super().__init__()

        self.dim = dim
        self.groups = groups
        self.use_ifg = use_ifg
        self.use_efg = use_efg
        self.use_cfg = use_cfg
        
        self.kernel_size = 13
        self.kernel_shape = (1, 1, self.kernel_size, self.kernel_size)
        self.padding = [int((k - 1) / 2) for k in self.kernel_shape[2:]]
        self.signal_padding = [r(p) for p in self.padding[::-1] for r in (floor, ceil)]


        # shape test
        test_img = torch.randn((1, 1, input_param['H'], input_param['W']))
        test_img, signal_size = self.fft_process(test_img)

        input_param['pad_H'] = test_img.shape[2]
        input_param['pad_W'] = test_img.shape[3]

        self.crop_slices = [slice(None), slice(None)] + [slice(0, (signal_size[i] - self.kernel_shape[i] + 1), 1) \
                                                         for i in range(2, test_img.ndim)]
        # define filter generator
        
        if use_ifg:
            self.IFG = IFG(dim, pe_dim, pe_ffn_dim, groups, input_param)
            self.bn1 = get_bn(dim)
            
        if use_efg: 
            self.EFG = EFG(dim, 1, mode='Bessel', bessel_num=8, input_param=input_param)
            self.bn2 = get_bn(dim)
            
        if use_cfg:
            self.DW_conv = DilatedReparamBlock(dim, 13, deploy=False)

        # # Perform fourier convolution -- FFT, matrix multiply, then IFFT
        # signal_fr = rfftn(signal.float(), dim=tuple(range(2, signal.ndim)))
        # kernel_fr = rfftn(padded_kernel.float(), dim=tuple(range(2, signal.ndim)))

    def reset(self, input_param):
        # shape test
        test_img = torch.randn((1, 1, input_param['H'], input_param['W']))
        test_img = self.fft_process(test_img)

        input_param['pad_H'] = test_img.shape[2]
        input_param['pad_W'] = test_img.shape[3]

        self.fg.reset(input_param)
        self.bg.reset(input_param)

    def fft_process(self, x):
        x = F.pad(x, self.signal_padding, mode='constant')
        signal_size = x.shape
        # original signal size without padding to even
        if x.size(-1) % 2 != 0:
            x = F.pad(x, [0, 1])
        return x, signal_size

    def train_forward(self, x):
        B, C, H, W = x.shape

        x1 = x2 = x3 = 0
        
        if self.use_cfg:
            x1 = self.DW_conv(x)

        if self.use_ifg or self.use_efg:
            x, _ = self.fft_process(x)
            _, _, H_pad, W_pad = x.shape

            x = torch.fft.rfft2(x, dim=(-2, -1))
            _, _, H_rfft, W_rfft = x.shape
            
            if self.use_ifg:
                w_a = self.IFG()
                x = x.view(B, self.groups, self.dim // self.groups, H_rfft, W_rfft)
                x2 = x * w_a

                x2 = torch.fft.irfft2(x2, dim=(-2, -1)).real.view(B, C, H_pad, W_pad)
                x2 = x2[self.crop_slices].contiguous()
                x2 = self.bn1(x2)

            if self.use_efg:
                w_b = self.EFG()
                x = x.view(B, 1, -1, H_rfft, W_rfft)
                x3 = x * w_b
                x3 = torch.fft.irfft2(x3, dim=(-2, -1)).real.view(B, C, H_pad, W_pad)
                x3 = x3[self.crop_slices].contiguous()
                x3 = self.bn2(x3)

        return x1 + x2 + x3

    def eval_forward(self, x):
        raise NotImplementedError("Raise after the paper is accept")

    def forward(self, x):
        return self.train_forward(x)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block proposed in SENet (https://arxiv.org/abs/1709.01507)
    We assume the inputs to this layer are (N, C, H, W)
    """

    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons,
                              kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels,
                            kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels
        self.nonlinear = nn.GELU()

    def forward(self, inputs):
        x = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x = self.down(x)
        x = self.nonlinear(x)
        x = self.up(x)
        x = F.sigmoid(x)
        return inputs * x.view(-1, self.input_channels, 1, 1)


class MGFBlock(nn.Module):
    def __init__(self, dim, groups, ffn_dim, pe_dim, pe_ffn_dim,
                 use_ifg=True, use_efg=True, use_cfg=True,
                 drop_path=0., layerscale=False, layer_init_values=1e-5,
                 input_param=None):
        super().__init__()
        self.layerscale = layerscale
        self.dim = dim

        self.drop_path = DropPath(drop_path)

        self.pos = DWConv2d(dim, 3, 1, 1)

        self.mixer_layer_norm = get_bn(self.dim)
        self.final_layer_norm = get_bn(self.dim)

        self.mixer = MGFMixer(dim, groups, pe_dim, pe_ffn_dim,
                              use_ifg, use_efg, use_cfg, input_param)
        self.ffn = FeedForwardNetwork(dim, ffn_dim)

        self.se = SEBlock(dim, dim // 4)

        if layerscale:
            self.gamma = nn.Parameter(layer_init_values * torch.ones(1, dim, 1, 1), requires_grad=True)

    def training_forward(self, x):
        x = self.mixer(x)
        x = self.mixer_layer_norm(x)
        x = self.se(x)
        x = self.ffn(x)
        x = self.final_layer_norm(x)
        return x

    def eval_forward(self, x):
        x = self.mixer.eval_forward(x)
        x = self.se(x)
        x = self.ffn(x)
        return x

    def forward(
            self,
            x: torch.Tensor,
    ):
        x = x + self.pos(x)
        if self.layerscale:
            x = x + self.drop_path(self.gamma * self.training_forward(x))
        else:
            x = x + self.drop_path(self.training_forward(x))
        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(dim, out_dim, 3, 2, 1)
        self.norm = LayerNorm2d(out_dim)

    def forward(self, x):
        '''
        x: B H W C
        '''
        x = self.reduction(x)  # (b oc oh ow)
        x = self.norm(x)
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, out_dim, groups, ffn_dim, pe_dim, pe_ffn_dim, depth,
                use_ifg=True, use_efg=True, use_cfg=True,
                 drop_path=0., norm_layer=nn.LayerNorm,
                 downsample: PatchMerging = None, use_checkpoint=False,
                 layerscale=False, layer_init_values=1e-5, input_param=None):

        super().__init__()
        self.dim = dim
        self.groups = groups
        self.ffn_dim = ffn_dim
        self.pe_dim = pe_dim
        self.pe_ffn_dim = pe_ffn_dim

        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            MGFBlock(dim, groups, ffn_dim, pe_dim, pe_ffn_dim, use_ifg, use_efg, use_cfg,
                     drop_path[i] if isinstance(drop_path, list) else drop_path, layerscale, layer_init_values,
                     input_param=input_param)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        b, h, w, d = x.size()
        for blk in self.blocks:
            if self.use_checkpoint:
                tmp_blk = partial(blk)
                x = checkpoint.checkpoint(tmp_blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, in_chans=3, embed_dim=96):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(embed_dim))

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        return x


class MGFNet(nn.Module):
    def __init__(self, in_chans=3, num_classes=100,
                 embed_dims=[96, 192, 384, 768], mlp_ratios=[3, 3, 3, 3], groups=[1, 1, 1, 1],
                 pe_dims=[192, 192, 192, 192], depths=[2, 2, 6, 2],
                 use_ifg = [True, True, True, True],
                 use_efg = [True, True, True, True],
                 use_cfg = [True, True, True, True],
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True, use_checkpoints=[False, False, False, False], projection=1024,
                 layerscales=[False, False, False, False], layer_init_values=1e-6, image_size=[224, 224]):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dims[0]
        self.patch_norm = patch_norm
        self.num_features = embed_dims[-1]
        self.mlp_ratios = mlp_ratios

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dims[0])

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        image_size = [image_size[0] // 4, image_size[1] // 4]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=embed_dims[i_layer],
                out_dim=embed_dims[i_layer + 1] if (i_layer < self.num_layers - 1) else None,
                groups=groups[i_layer],
                depth=depths[i_layer],
                ffn_dim=int(mlp_ratios[i_layer] * embed_dims[i_layer]),
                pe_dim=pe_dims[i_layer],
                pe_ffn_dim=int(mlp_ratios[i_layer] * pe_dims[i_layer]),
                use_ifg=use_ifg[i_layer],
                use_efg=use_efg[i_layer],
                use_cfg=use_cfg[i_layer],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoints[i_layer],
                layerscale=layerscales[i_layer],
                layer_init_values=layer_init_values,
                input_param={'H': image_size[0], 'W': image_size[1]},
            )
            self.layers.append(layer)
            image_size = [image_size[0] // 2, image_size[1] // 2]

        self.proj = nn.Conv2d(self.num_features, projection, kernel_size=1, stride=1, padding=0)
        self.norm = nn.BatchNorm2d(projection)
        self.swish = MemoryEfficientSwish()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(projection, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            try:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            except:
                pass

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)

        for layer in self.layers:
            x = layer(x)

        x = self.proj(x)  # (b h w c)
        x = self.norm(x)
        x = self.swish(x)

        x = self.avgpool(x)  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


@register_model
def MGFNet_S(args=None):
    model = MGFNet(
        embed_dims=[96, 192, 384, 768],
        depths=[3, 3, 10, 3],
        groups=[8, 8, 8, 8],
        pe_dims=[192, 192, 192, 192],
        mlp_ratios=[4, 4, 4, 3],
        drop_path_rate=0.15,
        layerscales=[False, False, False, False],
        use_checkpoints=[True, True, True, True]
    )
    model.default_cfg = _cfg()
    return model

@register_model
def MGFNet_S_IFGONLY(args=None):
    model = MGFNet(
        embed_dims=[96, 192, 384, 768],
        depths=[3, 3, 10, 3],
        groups=[8, 8, 8, 8],
        pe_dims=[192, 192, 192, 192],
        mlp_ratios=[4, 4, 4, 3],
        drop_path_rate=0.15,
        layerscales=[False, False, False, False],
        use_checkpoints=[True, True, True, True],
        use_ifg = [True, True, True, True],
        use_efg = [False, False, False, False],
        use_cfg = [False, False, False, False],
    )
    model.default_cfg = _cfg()
    return model

@register_model
def MGFNet_S_EFGONLY(args=None):
    model = MGFNet(
        embed_dims=[96, 192, 384, 768],
        depths=[3, 3, 10, 3],
        groups=[8, 8, 8, 8],
        pe_dims=[192, 192, 192, 192],
        mlp_ratios=[4, 4, 4, 3],
        drop_path_rate=0.15,
        layerscales=[False, False, False, False],
        use_checkpoints=[True, True, True, True],
        use_ifg = [False, False, False, False],
        use_efg = [True, True, True, True],
        use_cfg = [False, False, False, False],
    )
    model.default_cfg = _cfg()
    return model

@register_model
def MGFNet_S_CFGONLY(args=None):
    model = MGFNet(
        embed_dims=[96, 192, 384, 768],
        depths=[3, 3, 10, 3],
        groups=[8, 8, 8, 8],
        pe_dims=[192, 192, 192, 192],
        mlp_ratios=[4, 4, 4, 3],
        drop_path_rate=0.15,
        layerscales=[False, False, False, False],
        use_checkpoints=[True, True, True, True],
        use_ifg = [False, False, False, False],
        use_efg = [False, False, False, False],
        use_cfg = [True, True, True, True],
    )
    model.default_cfg = _cfg()
    return model

@register_model
def MGFNet_B(args=None):
    model = MGFNet(
        embed_dims=[96, 192, 384, 768],
        depths=[3, 3, 27, 3],
        groups=[8, 8, 8, 8],
        pe_dims=[192, 192, 192, 192],
        mlp_ratios=[4, 4, 4, 4],
        drop_path_rate=0.4,
        layerscales=[False, False, True, True],
        layer_init_values=1e-6
    )
    model.default_cfg = _cfg()
    return model


@register_model
def MGFNet_L(args=None):
    model = MGFNet(
        embed_dims=[128, 256, 512, 1024],
        depths=[3, 3, 27, 3],
        groups=[8, 8, 8, 8],
        pe_dims=[192, 192, 192, 192],
        mlp_ratios=[4, 4, 4, 4],
        drop_path_rate=0.5,
        layerscales=[False, False, True, True],
        layer_init_values=1e-6
    )
    model.default_cfg = _cfg()
    return model


def fft_handler(inputs, outputs):
    from fvcore.nn.jit_handles import get_shape
    import math
    input_shapes = get_shape(inputs[0])
    return math.prod(input_shapes[:-1]) * (input_shapes[:-1] // 2 + 1) * 5 * \
        (math.floor(math.log(input_shapes[-1])) + math.floor(math.log(input_shapes[-2] // 2 + 1)))
