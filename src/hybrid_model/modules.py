import torch
import torch.nn as nn
import numpy as np
from src.quantize import DSQ, bn_fold
from typing import Tuple, Union, List


class QConv2d(nn.Conv2d):
    r"""
    2D Conv layer inserted with DSQ layer.
    Weight is quantized to bit_width bits.
    Bias keeps float during training.
    """
    def __init__(self,
                 training_q: bool,
                 if_bn: bool,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 bias: bool = True,
                 bit_width: int = 1,
                 clip_range: Union[Tuple[float, float], None] = (-2., 2.),
                 slope: float = 5,
                 learn_lower: bool = False,
                 learn_upper: bool = False,
                 learn_slope: bool = False,
                 dtype=torch.float32):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        padding = ((kernel_size[0] - 1) // 2, kernel_size[0] - 1 - (kernel_size[0] - 1) // 2)
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride, padding=padding, bias=bias)
        self.dtype = dtype
        self.w_dsq = DSQ(bit_width=bit_width, clip_range=clip_range, slope=slope,
                         learn_lower=learn_lower, learn_upper=learn_upper, learn_slope=learn_slope, dtype=dtype)
        self.bn = nn.BatchNorm2d(out_channels)
        self.training_q = training_q
        self.if_bn = if_bn
        if self.training_q:
            self.q_mode = 'training'
        else:
            self.q_mode = 'post-training'
        # quantized
        self.quantized = False
        self.register_buffer('scale', None)
        self.register_buffer('zero_point', None)
        self.register_buffer('bit_width', None)

    def set_param(self, param_name: str, value: float):
        self.w_dsq.set_param(param_name, value)
        return

    def clip_(self, param_name: str, bot: float, top: float):
        self.w_dsq.clip_(param_name, bot, top)

    def training_quantize(self, inputs: torch.Tensor = None):
        if inputs is None:
            # Assume the weight and BN have been folded. -> self.fold_bn() has been executed.
            trained_w, trained_bias = self.weight, self.bias

            lower = torch.min(trained_w).detach()
            upper = torch.max(trained_w).detach()
            scale = (upper - lower) / (2 ** self.w_dsq.bit_width - 1)
            offset = lower
            int_w = torch.round((trained_w - offset) / scale).clip(0, 2 ** self.w_dsq.bit_width - 1)

            self.weight.data = int_w.detach().data
            self.bias.data = trained_bias.detach().data
            self.bit_width = torch.tensor(self.w_dsq.bit_width, dtype=self.dtype, device=self.w_dsq.clip_lower.device)
            self.scale = scale.detach().data
            self.zero_point = - lower / self.scale
            self.quantized = True
            return

        # Training
        if self.if_bn:
            # update running mean, running var of bn.
            # TODO: Trace training:
            # with torch.profiler.record_function("Pseudo Conv2d forward"):

            with torch.no_grad():
                pseudo_output = self._conv_forward(inputs, self.weight, self.bias)
                pseudo_output = self.bn(pseudo_output)

            # TODO: Trace training:
            # with torch.profiler.record_function("Weight Quantization"):

            # fold w and b
            fold_w, bias = bn_fold(self, self.bn)
            lower = torch.min(fold_w).detach()
            upper = torch.max(fold_w).detach()

            # STE
            scale = (upper - lower) / (2 ** self.w_dsq.bit_width - 1)
            offset = lower
            int_w = torch.round((fold_w - offset) / scale).clip(0, 2 ** self.w_dsq.bit_width - 1)
            hard_w = int_w * scale + offset
            q_w = fold_w + (hard_w - fold_w).detach()
        else:
            lower = torch.min(self.weight).detach()
            upper = torch.max(self.weight).detach()
            q_w = self.w_dsq.forward(self.weight, lower=lower, upper=upper)
            bias = self.bias
        return q_w, bias

    def fold_bn(self):
        if self.if_bn:
            fold_w, bias = bn_fold(self, self.bn)
            self.weight.data = fold_w.detach().data
            self.bias.data = bias.detach().data
            self.bn.reset_running_stats()
            self.bn.reset_parameters()
        return

    def forward(self, inputs: torch.Tensor):
        if self.quantized:
            # If quantized, the weights are integers
            # de-quantize:
            deq_w = (self.weight - self.zero_point) * self.scale
            output = self._conv_forward(inputs, deq_w, self.bias)

        elif self.training_q:
            # TODO: Trace training:
            # with torch.profiler.record_function("Conv2d weight fake quantization"):

            q_w, bias = self.training_quantize(inputs)

            # TODO: Trace training:
            # with torch.profiler.record_function("Real Conv2d forward"):

            output = self._conv_forward(inputs, q_w, bias)
        else:
            output = self._conv_forward(inputs, self.weight, self.bias)
            if self.if_bn:
                output = self.bn(output)
        return output


class QReLU(nn.Module):
    def __init__(self, training_q: bool,
                 bit_width: int, clip_range, slope, learn_lower, learn_upper, learn_slope, dtype):
        super().__init__()
        self.tracking_activation = False
        self.dtype = dtype
        self.relu = nn.ReLU()
        self.training_q = training_q
        self.a_dsq = DSQ(bit_width=bit_width, clip_range=clip_range,
                         slope=slope, learn_lower=learn_lower,
                         learn_upper=learn_upper, learn_slope=learn_slope, dtype=dtype)
        self.register_buffer('activation_record', None)
        if self.training_q:
            self.q_mode = 'training'
        else:
            self.q_mode = 'post-training'
        # quantized
        self.quantized = False
        self.register_buffer('scale', None)
        self.register_buffer('zero_point', None)
        self.register_buffer('bit_width', None)

    def training_quantize(self, inputs: torch.Tensor = None):
        if inputs is None:
            self.quantized = True
            self.bit_width = torch.tensor(self.a_dsq.bit_width, dtype=self.dtype, device=self.a_dsq.clip_lower.device)
            self.scale = ((self.a_dsq.clip_upper - self.a_dsq.clip_lower) / (2**self.bit_width - 1)).detach().data
            self.zero_point = torch.round(- self.a_dsq.clip_lower.detach().data / self.scale)
            return
        else:
            return self.a_dsq(inputs)

    def post_quantize(self, inputs: torch.Tensor):
        q_output = torch.round(inputs / self.scale + self.zero_point)
        q_output = torch.clip(q_output, 0, 2 ** self.bit_width - 1)
        return q_output

    def anti_quantize(self, inputs: torch.Tensor):
        output = (inputs - self.zero_point) * self.scale
        return output

    def forward(self, inputs):
        output = self.relu(inputs)
        if self.quantized:
            # quantize
            q_output = self.post_quantize(output)
            # anti quantize
            output = self.anti_quantize(q_output)
        elif self.training_q:
            # TODO: Trace training:
            # with torch.profiler.record_function("ReLU DSQ"):

            output = self.training_quantize(output)
        if self.tracking_activation:
            self.activation_record = output.detach().clone()
        return output


class QResidualBlock(nn.Module):
    r"""
    Basic Residual block inserted with DSQ layer, including activation quantization and weight quantization.

    Args:
        quantize_w: whether to quantize the weight in this residual block during quantization-aware training.
        quantize_a: whether to quantize the activation(ReLU) in this residual block during quantization-aware training.
        w_bit_width: quantization bit width of weights.
        a_bit_width: quantization bit width of activation.
        q_w_ranges: ((w1_clip_lower, w1_clip_upper), (w2_clip_lower, w2_clip_upper), ...)
        q_a_ranges: ((a1_clip_lower, a1_clip_upper), (a2_clip_lower, a2_clip_upper), ...)
        w_slopes: (w1_slope, w2_slope)
        a_slopes: (a1_slope, a2_slope)
        learn_w_lower: whether learn lower bound of weight
        learn_w_upper: whether learn upper bound of weight
        learn_a_lower: whether learn lower bound of activation
        learn_a_upper: whether learn upper bound of activation
        learn_w_slope: whether learn slope of weight DSQ
        learn_a_slope: whether learn slope of weight DSQ
        kernel_size: kernel size.
        in_ch_num: input channels of the first layer of block.
        out_ch_num: output channels of th block.
        stride: sliding stride.
    """
    def __init__(self,
                 in_ch_num: int,
                 out_ch_num: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 quantize_w: Union[bool, Tuple[bool, bool], List[bool]],
                 quantize_a: Union[bool, Tuple[bool, bool], List[bool]],
                 w_bit_width: Union[int, list],
                 a_bit_width: Union[int, list],
                 q_w_ranges: Union[Tuple[float, float], Tuple[Tuple[float, float]]],
                 q_a_ranges: Union[Tuple[float, float], Tuple[Tuple[float, float]]],
                 w_slopes: Union[float, Tuple[float]],
                 a_slopes: Union[float, Tuple[float]],
                 learn_w_lower: bool,
                 learn_w_upper: bool,
                 learn_a_lower: bool,
                 learn_a_upper: bool,
                 learn_w_slope: bool,
                 learn_a_slope: bool,
                 bias: bool = True,
                 stride: int = 1,
                 dtype=torch.float32):
        super().__init__()

        self.stride = stride
        self.in_ch_num = in_ch_num
        self.out_ch_num = out_ch_num
        self.bias = bias
        self.dtype = dtype
        if self.stride != 1 or self.in_ch_num != self.out_ch_num:
            self.down_sample = True
            self.conv_num = 3
        else:
            self.down_sample = False
            self.conv_num = 2

        if isinstance(quantize_w, bool):
            quantize_w = tuple(quantize_w for i in range(self.conv_num))
        self.quantize_w = quantize_w
        if len(self.quantize_w) != self.conv_num:
            raise ValueError('expect specifying whether quantize {} conv layers, '
                             'but {} given.'.format(self.conv_num, len(self.quantize_w)))

        if isinstance(quantize_a, bool):
            quantize_a = (quantize_a, quantize_a)
        self.quantize_a = quantize_a
        if len(self.quantize_a) != 2:
            raise ValueError('expect specifying whether quantize {} activation layers, '
                             'but {} given.'.format(2, len(self.quantize_a)))

        if isinstance(w_bit_width, int):
            w_bit_width = [w_bit_width for _ in range(3)]
        if isinstance(a_bit_width, int):
            a_bit_width = [a_bit_width for _ in range(2)]
        if len(w_bit_width) != 3:
            raise ValueError(
                "Expect specifying the quantization bit-width of 3 conv layers: conv1, conv2, shortcut"
            )
        if len(a_bit_width) != 2:
            raise ValueError(
                "Expect specifying the quantization bit-width of 2 ReLU: relu1, relu2"
            )
        self.w_bit_width = w_bit_width
        self.a_bit_width = a_bit_width
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        self.padding = ((self.kernel_size[0] - 1) // 2, self.kernel_size[0] - 1 - (self.kernel_size[0] - 1) // 2)
        self._construct(q_w_ranges=q_w_ranges, q_a_ranges=q_a_ranges, w_slopes=w_slopes, a_slopes=a_slopes,
                        learn_w_lower=learn_w_lower, learn_w_upper=learn_w_upper, learn_a_lower=learn_a_lower,
                        learn_a_upper=learn_a_upper, learn_w_slope=learn_w_slope, learn_a_slope=learn_a_slope)

    @staticmethod
    def _check_ranges(q_ranges, if_quantize):
        q_blocks_num = sum(if_quantize)
        if q_ranges is None:
            if q_blocks_num != 0:
                raise ValueError('Expect q_ranges when blocks to be quantized specified.')
            q_ranges = tuple(None for i in range(len(if_quantize)))
        elif isinstance(q_ranges[0], (tuple, list)):
            if len(q_ranges) != q_blocks_num:
                raise ValueError('Expect {} pairs of range, but {} got.'.format(q_blocks_num, len(q_ranges)))
            padded_q_ranges = []
            idx = 0
            for block_idx in range(len(if_quantize)):
                block_range = None
                if if_quantize[block_idx]:
                    block_range = q_ranges[idx]
                    idx += 1
                padded_q_ranges.append(block_range)
            q_ranges = tuple(padded_q_ranges)
        elif isinstance(q_ranges[0], (float, int)):
            q_ranges = tuple(q_ranges for i in range(len(if_quantize)))
        else:
            raise ValueError('Wrong format.')
        return q_ranges

    @staticmethod
    def _check_slopes(slopes, if_quantize):
        q_blocks_num = sum(if_quantize)
        if slopes is None:
            if q_blocks_num != 0:
                raise ValueError('Expect slopes when blocks to be quantized specified.')
        elif isinstance(slopes, (tuple, list)):
            if len(slopes) != q_blocks_num:
                raise ValueError('Expect {} slopes, but {} got.'.format(q_blocks_num, len(slopes)))
            padded_slopes = []
            idx = 0
            for block_idx in range(len(if_quantize)):
                slope = None
                if if_quantize[block_idx]:
                    slope = slopes[idx]
                    idx += 1
                padded_slopes.append(slope)
            slopes = tuple(padded_slopes)
        elif isinstance(slopes, (float, int)):
            slopes = tuple(float(slopes) for i in range(len(if_quantize)))
        else:
            raise ValueError('Wrong format of slopes')
        return slopes

    def _construct(self, q_w_ranges, q_a_ranges, w_slopes, a_slopes, learn_w_lower, learn_w_upper,
                   learn_a_lower, learn_a_upper, learn_w_slope, learn_a_slope):
        self.q_w_ranges = self._check_ranges(q_w_ranges, self.quantize_w)
        self.w_slopes = self._check_slopes(w_slopes, self.quantize_w)

        self.q_a_ranges = self._check_ranges(q_a_ranges, self.quantize_a)
        self.a_slopes = self._check_slopes(a_slopes, self.quantize_a)
        self.learn_w_lower = learn_w_lower
        self.learn_w_upper = learn_w_upper
        self.learn_a_lower = learn_a_lower
        self.learn_a_upper = learn_a_upper

        self.conv1 = QConv2d(training_q=self.quantize_w[0], if_bn=True, in_channels=self.in_ch_num,
                             out_channels=self.out_ch_num, kernel_size=self.kernel_size, stride=self.stride,
                             bias=self.bias, bit_width=self.w_bit_width[0], clip_range=self.q_w_ranges[0],
                             slope=self.w_slopes[0], learn_lower=learn_w_lower,
                             learn_upper=learn_w_upper, learn_slope=learn_w_slope, dtype=self.dtype)
        self.relu1 = QReLU(training_q=self.quantize_a[0], bit_width=self.a_bit_width[0], clip_range=self.q_a_ranges[0],
                           slope=self.a_slopes[0], learn_lower=learn_a_lower,
                           learn_upper=learn_a_upper, learn_slope=learn_a_slope, dtype=self.dtype)

        self.conv2 = QConv2d(training_q=self.quantize_w[1], if_bn=True, in_channels=self.out_ch_num,
                             out_channels=self.out_ch_num, kernel_size=self.kernel_size, stride=1,
                             bias=self.bias, bit_width=self.w_bit_width[1], clip_range=self.q_w_ranges[1],
                             slope=self.w_slopes[1], learn_lower=learn_w_lower,
                             learn_upper=learn_w_upper, learn_slope=learn_w_slope, dtype=self.dtype)
        self.relu2 = QReLU(training_q=self.quantize_a[1], bit_width=self.a_bit_width[1], clip_range=self.q_a_ranges[1],
                           slope=self.a_slopes[1], learn_lower=learn_a_lower,
                           learn_upper=learn_a_upper, learn_slope=learn_a_slope, dtype=self.dtype)

        if self.down_sample:
            # down-sampling shortcut
            self.shortcut_conv = QConv2d(training_q=self.quantize_w[2], if_bn=True, in_channels=self.in_ch_num,
                                         out_channels=self.out_ch_num, kernel_size=1, stride=self.stride, bias=self.bias,
                                         bit_width=self.w_bit_width[2], clip_range=self.q_w_ranges[2],
                                         slope=self.w_slopes[2],
                                         learn_lower=learn_w_lower, learn_upper=learn_w_upper,
                                         learn_slope=learn_w_slope, dtype=self.dtype)

    def forward(self, x: torch.Tensor):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        if self.down_sample:
            out += self.shortcut_conv(x)
        else:
            out += x
        out = self.relu2(out)
        return out


class QBottleneck(nn.Module):
    r"""
    # TODO: to be checked.
    Bottleneck Residual block inserted with DSQ layer, including activation quantization and weight quantization.

    Args:
        quantize_w: whether to quantize the weight in this residual block during quantization-aware training.
        quantize_a: whether to quantize the activation(ReLU) in this residual block during quantization-aware training.
        w_bit_width: quantization bit width of weights.
        a_bit_width: quantization bit width of activation.
        q_w_ranges: ((w1_clip_lower, w1_clip_upper), (w2_clip_lower, w2_clip_upper), ...)
        q_a_ranges: ((a1_clip_lower, a1_clip_upper), (a2_clip_lower, a2_clip_upper), ...)
        w_slopes: (w1_slope, w2_slope)
        a_slopes: (a1_slope, a2_slope)
        learn_w_lower: whether learn lower bound of weight
        learn_w_upper: whether learn upper bound of weight
        learn_a_lower: whether learn lower bound of activation
        learn_a_upper: whether learn upper bound of activation
        learn_w_slope: whether learn slope of weight DSQ
        learn_a_slope: whether learn slope of weight DSQ
        kernel_size: kernel size.
        in_ch_num: input channels of the first layer of block.
        out_ch_num: output channels of th block. (The middle channel num is out_ch_num / 4)
        stride: sliding stride.
    """
    def __init__(
        self,
        in_ch_num: int,
        out_ch_num: int,
        kernel_size: Union[int, Tuple[int, int]],
        quantize_w: Union[bool, Tuple[bool, bool], List[bool]],
        quantize_a: Union[bool, Tuple[bool, bool], List[bool]],
        w_bit_width: Union[int, list],
        a_bit_width: Union[int, list],
        q_w_ranges: Union[Tuple[float, float], Tuple[Tuple[float, float]]],
        q_a_ranges: Union[Tuple[float, float], Tuple[Tuple[float, float]]],
        w_slopes: Union[float, Tuple[float]],
        a_slopes: Union[float, Tuple[float]],
        learn_w_lower: bool,
        learn_w_upper: bool,
        learn_a_lower: bool,
        learn_a_upper: bool,
        learn_w_slope: bool,
        learn_a_slope: bool,
        bias: bool = True,
        stride: int = 1,
        dtype=torch.float32
    ):
        super().__init__()

        self.stride = stride
        self.in_ch_num = in_ch_num
        self.out_ch_num = out_ch_num
        self.bias = bias
        self.dtype = dtype
        if self.stride != 1 or self.in_ch_num != self.out_ch_num:
            self.down_sample = True
            self.conv_num = 4
        else:
            self.down_sample = False
            self.conv_num = 3

        if isinstance(quantize_w, bool):
            quantize_w = tuple(quantize_w for i in range(self.conv_num))
        self.quantize_w = quantize_w
        if len(self.quantize_w) != self.conv_num:
            raise ValueError('expect specifying whether quantize {} conv layers, '
                             'but {} given.'.format(self.conv_num, len(self.quantize_w)))

        if isinstance(quantize_a, bool):
            quantize_a = (quantize_a, quantize_a, quantize_a)
        self.quantize_a = quantize_a
        if len(self.quantize_a) != 3:
            raise ValueError('expect specifying whether quantize {} activation layers, '
                             'but {} given.'.format(3, len(self.quantize_a)))

        if isinstance(w_bit_width, int):
            w_bit_width = [w_bit_width for _ in range(4)]
        if isinstance(a_bit_width, int):
            a_bit_width = [a_bit_width for _ in range(3)]
        if len(w_bit_width) != 4:
            raise ValueError(
                "Expect specifying the quantization bit-width of 4 conv layers: conv1, conv2, conv3, shortcut"
            )
        if len(a_bit_width) != 3:
            raise ValueError(
                "Expect specifying the quantization bit-width of 2 ReLU: relu1, relu2, relu3"
            )

        self.w_bit_width = w_bit_width
        self.a_bit_width = a_bit_width
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        self.padding = ((self.kernel_size[0] - 1) // 2, self.kernel_size[0] - 1 - (self.kernel_size[0] - 1) // 2)
        self._construct(q_w_ranges=q_w_ranges, q_a_ranges=q_a_ranges, w_slopes=w_slopes, a_slopes=a_slopes,
                        learn_w_lower=learn_w_lower, learn_w_upper=learn_w_upper, learn_a_lower=learn_a_lower,
                        learn_a_upper=learn_a_upper, learn_w_slope=learn_w_slope, learn_a_slope=learn_a_slope)

    @staticmethod
    def _check_ranges(q_ranges, if_quantize):
        q_blocks_num = sum(if_quantize)
        if q_ranges is None:
            if q_blocks_num != 0:
                raise ValueError('Expect q_ranges when blocks to be quantized specified.')
            q_ranges = tuple(None for i in range(len(if_quantize)))
        elif isinstance(q_ranges[0], (tuple, list)):
            if len(q_ranges) != q_blocks_num:
                raise ValueError('Expect {} pairs of range, but {} got.'.format(q_blocks_num, len(q_ranges)))
            padded_q_ranges = []
            idx = 0
            for block_idx in range(len(if_quantize)):
                block_range = None
                if if_quantize[block_idx]:
                    block_range = q_ranges[idx]
                    idx += 1
                padded_q_ranges.append(block_range)
            q_ranges = tuple(padded_q_ranges)
        elif isinstance(q_ranges[0], (float, int)):
            q_ranges = tuple(q_ranges for i in range(len(if_quantize)))
        else:
            raise ValueError('Wrong format.')
        return q_ranges

    @staticmethod
    def _check_slopes(slopes, if_quantize):
        q_blocks_num = sum(if_quantize)
        if slopes is None:
            if q_blocks_num != 0:
                raise ValueError('Expect slopes when blocks to be quantized specified.')
        elif isinstance(slopes, (tuple, list)):
            if len(slopes) != q_blocks_num:
                raise ValueError('Expect {} slopes, but {} got.'.format(q_blocks_num, len(slopes)))
            padded_slopes = []
            idx = 0
            for block_idx in range(len(if_quantize)):
                slope = None
                if if_quantize[block_idx]:
                    slope = slopes[idx]
                    idx += 1
                padded_slopes.append(slope)
            slopes = tuple(padded_slopes)
        elif isinstance(slopes, (float, int)):
            slopes = tuple(float(slopes) for i in range(len(if_quantize)))
        else:
            raise ValueError('Wrong format of slopes')
        return slopes

    def _construct(self, q_w_ranges, q_a_ranges, w_slopes, a_slopes, learn_w_lower, learn_w_upper,
                   learn_a_lower, learn_a_upper, learn_w_slope, learn_a_slope):
        self.q_w_ranges = self._check_ranges(q_w_ranges, self.quantize_w)
        self.w_slopes = self._check_slopes(w_slopes, self.quantize_w)

        self.q_a_ranges = self._check_ranges(q_a_ranges, self.quantize_a)
        self.a_slopes = self._check_slopes(a_slopes, self.quantize_a)
        self.learn_w_lower = learn_w_lower
        self.learn_w_upper = learn_w_upper
        self.learn_a_lower = learn_a_lower
        self.learn_a_upper = learn_a_upper

        mid_ch_num = int(self.out_ch_num / 4)

        self.conv1 = QConv2d(training_q=self.quantize_w[0], if_bn=True, in_channels=self.in_ch_num,
                             out_channels=mid_ch_num, kernel_size=(1, 1), stride=1,
                             bias=self.bias, bit_width=self.w_bit_width[0], clip_range=self.q_w_ranges[0],
                             slope=self.w_slopes[0], learn_lower=learn_w_lower,
                             learn_upper=learn_w_upper, learn_slope=learn_w_slope, dtype=self.dtype)
        self.relu1 = QReLU(training_q=self.quantize_a[0], bit_width=self.a_bit_width[0], clip_range=self.q_a_ranges[0],
                           slope=self.a_slopes[0], learn_lower=learn_a_lower,
                           learn_upper=learn_a_upper, learn_slope=learn_a_slope, dtype=self.dtype)
        # conv2 stride
        self.conv2 = QConv2d(training_q=self.quantize_w[1], if_bn=True, in_channels=mid_ch_num,
                             out_channels=mid_ch_num, kernel_size=self.kernel_size, stride=self.stride,
                             bias=self.bias, bit_width=self.w_bit_width[1], clip_range=self.q_w_ranges[1],
                             slope=self.w_slopes[1], learn_lower=learn_w_lower,
                             learn_upper=learn_w_upper, learn_slope=learn_w_slope, dtype=self.dtype)
        self.relu2 = QReLU(training_q=self.quantize_a[1], bit_width=self.a_bit_width[1], clip_range=self.q_a_ranges[1],
                           slope=self.a_slopes[1], learn_lower=learn_a_lower,
                           learn_upper=learn_a_upper, learn_slope=learn_a_slope, dtype=self.dtype)

        self.conv3 = QConv2d(training_q=self.quantize_w[2], if_bn=True, in_channels=mid_ch_num,
                             out_channels=self.out_ch_num, kernel_size=(1, 1), stride=1,
                             bias=self.bias, bit_width=self.w_bit_width[2], clip_range=self.q_w_ranges[2],
                             slope=self.w_slopes[2], learn_lower=learn_w_lower,
                             learn_upper=learn_w_upper, learn_slope=learn_w_slope, dtype=self.dtype)
        self.relu3 = QReLU(training_q=self.quantize_a[2], bit_width=self.a_bit_width[2], clip_range=self.q_a_ranges[2],
                           slope=self.a_slopes[2], learn_lower=learn_a_lower,
                           learn_upper=learn_a_upper, learn_slope=learn_a_slope, dtype=self.dtype)

        if self.down_sample:
            # down-sampling shortcut
            self.shortcut_conv = QConv2d(training_q=self.quantize_w[3], if_bn=True, in_channels=self.in_ch_num,
                                         out_channels=self.out_ch_num, kernel_size=(1, 1), stride=self.stride,
                                         bias=self.bias, bit_width=self.w_bit_width[3], clip_range=self.q_w_ranges[3],
                                         slope=self.w_slopes[3],
                                         learn_lower=learn_w_lower, learn_upper=learn_w_upper,
                                         learn_slope=learn_w_slope, dtype=self.dtype)

    def forward(self, x: torch.Tensor):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        if self.down_sample:
            out += self.shortcut_conv(x)
        else:
            out += x
        out = self.relu3(out)
        return out
