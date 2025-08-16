import torch
import numpy as np


class AICQ:
    def __init__(self):
        self.eps = 1e-6
        self.max_iterations = 1e5
        self.x_dict = {}
        self.relu_x_dict = {}

    @staticmethod
    def f(x, bit_width):
        r"""
        func: x / (3 * 2**(2*bit_width)) - exp(-x）
        """
        k = 1 / (3 * 2 ** (2 * bit_width))
        return k * x - np.exp(-x)

    @staticmethod
    def relu_f(x, bit_width):
        r"""
        func: x / (12 * 2**(2*bit_width)) - exp(-x）
        """
        k = 1 / (12 * 2 ** (2 * bit_width))
        return k * x - np.exp(-x)

    def newton(self, func, bit_width):
        r"""
        solve the equation: func = 0.
        :param func
        :param bit_width:
        :return:
        """

        def df(x, dx=1-8):
            return (func(x + dx, bit_width) - func(x, bit_width)) / dx

        xx = 0
        count = 0
        while abs(func(xx, bit_width)) > self.eps and count <= self.max_iterations:
            xx = xx - func(xx, bit_width) / df(xx)
            count += 1

        if count > self.max_iterations:
            return None
        else:
            return xx

    def calculate_median_range(self, m: torch.Tensor, bit_width: int):
        median = torch.quantile(m, q=0.5)
        b = torch.mean(torch.abs(m - median))
        print('b: ', b)
        if bit_width not in self.x_dict.keys():
            x = self.newton(func=self.f, bit_width=bit_width)
            self.x_dict[bit_width] = x
        else:
            x = self.x_dict[bit_width]
        alpha = b * x
        return median, alpha

    def calculate_relu_median_range(self, m: torch.Tensor, bit_width: int):
        median = torch.quantile(m, q=0.5)
        b = torch.mean(torch.abs(m - median))
        if bit_width not in self.relu_x_dict.keys():
            x = self.newton(func=self.relu_f, bit_width=bit_width)
            self.relu_x_dict[bit_width] = x
        else:
            x = self.relu_x_dict[bit_width]
        alpha = b * x
        return median, alpha

    @staticmethod
    def relu_median_alpha_to_offset_scale(median, alpha, bit_width):
        r""" scale = 0 or scale / 2 """
        scale = alpha / 2 ** bit_width
        offset = torch.tensor(0., dtype=alpha.dtype, device=alpha.device)
        return scale, offset

    @staticmethod
    def median_alpha_to_offset_scale(median, alpha, bit_width):
        r"""
        median: median of laplace distribution approximation.
        alpha: half clip length: (clip_upper - clip_lower) / 2.

        return:
            scale, offset: used in anti_quantize: w = w_q * scale + offset.
        """
        scale = alpha / 2**(bit_width - 1)
        offset = median - alpha + scale / 2
        return scale, offset

    @staticmethod
    def offset_scale_to_median_alpha(offset, scale, bit_width):
        alpha = scale * 2**(bit_width - 1)
        median = offset + scale * alpha - scale / 2
        return median, alpha

    def quantize_by_median_alpha(self, m: torch.Tensor, bit_width: int, median, alpha):
        assert bit_width > 0
        interval = alpha / 2 ** (bit_width - 1)
        q_matrix = torch.round((m - median + alpha - interval / 2) / interval)
        q_matrix = torch.clip(q_matrix, 0, 2 ** bit_width - 1)

        # bias correction
        scale, offset = self.median_alpha_to_offset_scale(median, alpha, bit_width)
        back_matrix = self.anti_quantize(q_matrix, scale, offset)
        mean = torch.mean(m)
        l2_norm = torch.norm(m - mean, p=2)
        q_mean = torch.mean(back_matrix)
        q_l2_norm = torch.norm(back_matrix - q_mean, p=2)
        zoom = l2_norm / q_l2_norm
        shift = mean - zoom * q_mean

        median = median * zoom + shift
        alpha = zoom * alpha
        scale, offset = self.median_alpha_to_offset_scale(median, alpha, bit_width)
        return q_matrix, scale, offset

    def bias_correction(self, m, q_m, scale, offset):
        mean = torch.mean(m)
        l2_norm = torch.norm(m - mean, p=2)
        back_matrix = self.anti_quantize(q_m, scale, offset)
        q_mean = torch.mean(back_matrix)
        q_l2_norm = torch.norm(back_matrix - q_mean, p=2)
        zoom = l2_norm / q_l2_norm
        shift = mean - zoom * q_mean
        scale = scale * zoom
        offset = offset * zoom + shift
        return scale, offset

    def naive_quantize(self, m, bit_width):
        m_max = torch.max(m)
        m_min = torch.min(m)
        interval = (m_max - m_min) / (2**bit_width)
        offset = m_min + interval / 2
        scale = interval
        zero_point = torch.round(- offset / scale)
        q_matrix = torch.round(m / scale) + zero_point
        q_matrix = torch.clip(q_matrix, 0, 2 ** bit_width - 1)
        scale, offset = self.bias_correction(m, q_matrix, scale, offset)
        return q_matrix, scale, offset

    def quantize(self, m: torch.Tensor, bit_width: int):
        r"""
        quantize the tensor according to Analytical Clipping for Integer Quantization.
            refer to: https://arxiv.org/pdf/1810.05723.pdf
        return the quantized tensor and corresponding scale and offset:
            w = w_q * scale + offset
            w_q = clip(round((w - offset) / scale), 0, 2**bit_width - 1)
        """

        # median, alpha = self.calculate_median_range(m, bit_width)
        # print('median: {} alpha: {}'.format(median, alpha))
        # q_matrix, scale, offset = self.quantize_by_median_alpha(m, bit_width, median, alpha)

        q_matrix, scale, offset = self.naive_quantize(m, bit_width)
        return q_matrix, scale, offset

    def anti_quantize(self, q_m: torch.tensor, scale, offset):
        m = q_m * scale + offset
        return m
