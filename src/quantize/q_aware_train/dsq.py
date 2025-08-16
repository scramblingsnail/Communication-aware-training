import torch
import torch.nn as nn
from typing import Tuple, Union


class DSQ(nn.Module):
    def __init__(
        self,
        bit_width: int,
        clip_range: Union[Tuple[float, float], None],
        slope: float,
        learn_lower: bool = True,
        learn_upper: bool = True,
        learn_slope: bool = False,
        dtype=torch.float32,
    ):
        super().__init__()
        self.bit_width = bit_width
        self.dtype = dtype
        if clip_range is None:
            clip_range = (0., 2.)
        if clip_range[1] <= clip_range[0]:
            raise ValueError('clip_upper must be larger than clip_lower.')
        if slope is None:
            slope = 5.
        self.clip_lower = nn.Parameter(torch.tensor(clip_range[0], dtype=self.dtype))
        self.clip_upper = nn.Parameter(torch.tensor(clip_range[1], dtype=self.dtype))
        self.slope = nn.Parameter(torch.tensor(slope, dtype=self.dtype))
        if not learn_lower:
            self.clip_lower.requires_grad = False
        if not learn_upper:
            self.clip_upper.requires_grad = False
        if not learn_slope:
            self.slope.requires_grad = False

    def set_param(self, param_name: str, value: float):
        if param_name not in ['clip_lower', 'clip_upper', 'slope']:
            raise ValueError("available params: 'clip_lower', 'clip_upper', 'slope'.")

        param = self.__getattr__(param_name)
        with torch.no_grad():
            if isinstance(param, nn.Parameter):
                param.data = torch.tensor(value, dtype=self.dtype, device=param.data.device)
            else:
                param = torch.tensor(value, dtype=self.dtype, device=param.device)
                self.__setattr__(param_name, param)

    def clip_(self, param_name: str, bot: float, top: float):
        if param_name not in ['clip_lower', 'clip_upper', 'slope']:
            raise ValueError("available params: 'clip_lower', 'clip_upper', 'slope'.")
        param = self.__getattr__(param_name)
        with torch.no_grad():
            if isinstance(param, nn.Parameter):
                torch.clip_(param.data, bot, top)
            elif isinstance(param, torch.Tensor):
                torch.clip_(param, bot, top)

    def soft_q(
        self,
        inputs: torch.Tensor,
        lower: torch.Tensor = None,
        upper: torch.Tensor = None,
        de_quantize: bool = False,
    ):
        lower = lower or self.clip_lower
        upper = upper or self.clip_upper
        scale = (upper - lower) / (2**self.bit_width - 1)
        offset = lower
        f_amp = 0.5 / torch.tanh(self.slope.mul(0.5))

        affine_inputs = (inputs - offset) / scale
        affine_floor = torch.floor(affine_inputs)
        soft_q_inputs = affine_floor + 0.5 + f_amp * torch.tanh(self.slope * (affine_inputs - affine_floor - 0.5))
        # clip
        clipped = torch.clip(soft_q_inputs, 0, 2**self.bit_width - 1)
        if de_quantize:
            # de-quantize
            outputs = clipped * scale + offset
            return outputs
        return clipped

    def hard_q(
        self,
        soft_inputs: torch.Tensor,
        lower: torch.Tensor = None,
        upper: torch.Tensor = None,
    ):
        r"""

        Args:
            soft_inputs (torch.Tensor): The inputs that have underwent soft quantization.
            lower (torch.Tensor): Specified lower.
            upper (torch.Tensor): Specified upper.
        """
        lower = lower or self.clip_lower
        upper = upper or self.clip_upper
        scale = (upper - lower) / (2**self.bit_width - 1)
        offset = lower
        hard_q_inputs = soft_inputs + (torch.round(soft_inputs) - soft_inputs).detach()
        # de-quantize
        outputs = hard_q_inputs * scale + offset
        return outputs

    def ste_forward(self, inputs: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor):
        scale = (upper - lower) / (2 ** self.bit_width - 1)
        offset = lower
        int_inputs = torch.round((inputs - offset) / scale).clip(0, 2 ** self.bit_width - 1)
        hard_q_inputs = int_inputs * scale + offset
        q_inputs = inputs + (hard_q_inputs - inputs).detach()
        return q_inputs

    def forward(self, inputs: torch.Tensor, lower: torch.Tensor = None, upper: torch.Tensor = None):
        r"""
        Args:
            inputs (torch.Tensor):
            lower (torch.Tensor): Specified lower.
            upper (torch.Tensor): Specified upper.


        Return:
            quantize -> dequantized val

        """
        if inputs is None:
            return None
        else:
            return self.hard_q(self.soft_q(inputs, lower=lower, upper=upper), lower, upper)
