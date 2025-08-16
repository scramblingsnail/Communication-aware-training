import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from .modules import QResidualBlock, QConv2d, DSQ, QReLU, QBottleneck
from ..quantize import AICQ, bn_fold
from ..channel_simulate.differentiable_flip import SymbolWiseDFlip, MetaSymbolWiseFlip
from ..channel_simulate.differentiable_serialize import DSerialize, DDeSerialize
from ..channel_simulate.link_simulation_torch.simulation import ChannelSimulator
from typing import List, Tuple, Union


class HybridCNN(nn.Module):
    r"""

    """
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        blocks_setting: list,
        labels_num: int,
        quantize_w_blocks: list,
        quantize_a_blocks: list,
        quantize_a: list,
        w_bit_width: int,
        a_bit_width: list,
        q_w_range,
        q_a_range,
        w_slope,
        a_slope,
        learn_w_lower,
        learn_w_upper,
        learn_a_lower,
        learn_a_upper,
        learn_w_slope,
        learn_a_slope,
        dtype,
        channel_simulator: ChannelSimulator,
        eval_multi_channel_configs: dict,
        ber_model_path: List[str] = None,
        bottleneck_type: bool = False,
        first_kernel_size: int = 3,
        first_stride: int = 1,
        first_pool_layer_setting: dict = None,
        init_wireless_p: float = 0.5,
        wireless_p_scale: float = 1.,
        flat_channel_aug_mode: bool = True,
        freq_selective_aug_mode: bool = False,
    ):
        super().__init__()
        feature_channels = blocks_setting[0][0]
        first_kernel_size = first_kernel_size or kernel_size
        self.dtype = dtype
        self.channel_simulator = channel_simulator
        self.conv0 = QConv2d(training_q=False, if_bn=True, in_channels=in_channels, out_channels=feature_channels,
                             kernel_size=first_kernel_size, stride=first_stride)
        self.relu0 = QReLU(training_q=False, bit_width=2, clip_range=(0., 1.), slope=10., learn_lower=False,
                           learn_upper=False, learn_slope=False, dtype=dtype)
        if first_pool_layer_setting:
            self.max_pool = nn.MaxPool2d(
                kernel_size=first_pool_layer_setting["kernel_size"],
                stride=first_pool_layer_setting["stride"],
                padding=first_pool_layer_setting["padding"],
                ceil_mode=False,
            )
        else:
            self.max_pool = None
        self.blocks_num = len(blocks_setting)
        self.ber = 0
        self.noisy = False
        self.evaluate_noisy = False
        self.channel_model_eval = False
        self.tracking_trained_wireless_param = False
        self.edge_end_block_idx = 0
        self.edge_end_a_bit_width = 1
        self.bottleneck_type = bottleneck_type
        if ber_model_path:
            if len(ber_model_path) == 1:
                channel_aimc_bit = eval_multi_channel_configs["aimc_bit"][0]
                channel_order = eval_multi_channel_configs["order"][0]
                self.set_channel_param(name="aimc_bit", val=channel_aimc_bit)
                self.set_channel_param(name="order", val=channel_order)
                self.flipper = SymbolWiseDFlip(
                    ber_model_path=ber_model_path[0],
                    init_wireless_p=init_wireless_p,
                    wireless_p_scale=wireless_p_scale,
                    flat_channel_aug_mode=flat_channel_aug_mode,
                    freq_selective_aug_mode=freq_selective_aug_mode,
                )
            else:
                self.flipper = MetaSymbolWiseFlip(
                    ber_model_paths=ber_model_path,
                    init_wireless_p=init_wireless_p,
                    wireless_p_scale=wireless_p_scale,
                    flat_channel_aug_mode=flat_channel_aug_mode,
                    freq_selective_aug_mode=freq_selective_aug_mode,
                    eval_channel_configs=eval_multi_channel_configs,
                )

        else:
            self.flipper = None

        if self.bottleneck_type:
            BlockClass = QBottleneck
        else:
            BlockClass = QResidualBlock

        for block_idx in range(self.blocks_num):
            if block_idx in quantize_w_blocks:
                q_w = True
            else:
                q_w = False
            if block_idx in quantize_a_blocks:
                q_a_idx = quantize_a_blocks.index(block_idx)
                q_a = quantize_a[q_a_idx]
                block_a_bit_width = a_bit_width[q_a_idx]
                self.edge_end_block_idx = block_idx
                self.edge_end_a_bit_width = block_a_bit_width[-1]
            else:
                q_a = False
                block_a_bit_width = -1
            block_name = 'res_block{}'.format(block_idx)
            block = BlockClass(in_ch_num=blocks_setting[block_idx][0], out_ch_num=blocks_setting[block_idx][1],
                                   stride=blocks_setting[block_idx][2], kernel_size=kernel_size,
                                   quantize_w=q_w, quantize_a=q_a, w_bit_width=w_bit_width,
                                   a_bit_width=block_a_bit_width, q_w_ranges=q_w_range, q_a_ranges=q_a_range,
                                   w_slopes=w_slope, a_slopes=a_slope, learn_w_lower=learn_w_lower,
                                   learn_w_upper=learn_w_upper, learn_a_lower=learn_a_lower,
                                   learn_a_upper=learn_a_upper, learn_w_slope=learn_w_slope,
                                   learn_a_slope=learn_a_slope, dtype=dtype)
            self.__setattr__(name=block_name, value=block)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(blocks_setting[-1][1], labels_num)
        self.post_q_machine = AICQ()
        self.serializer = DSerialize(bit_width=self.edge_end_a_bit_width)
        self.deserializer = DDeSerialize(bit_width=self.edge_end_a_bit_width)

    def reload_ber_model(self, ber_model_path: str):
        device = next(self.flipper.parameters()).device
        self.flipper.reload_ber_model(ber_model_path=ber_model_path, device=device)

    def map_inner_to(self, device: torch.device):
        if isinstance(self.flipper, SymbolWiseDFlip):
            self.flipper.ber_model = self.flipper.ber_model.to(device)
            self.flipper.ber_model.wireless_p_lower = self.flipper.ber_model.wireless_p_lower.to(device)
            self.flipper.ber_model.wireless_p_upper = self.flipper.ber_model.wireless_p_upper.to(device)
        elif isinstance(self.flipper, MetaSymbolWiseFlip):
            self.flipper.ber_models = self.flipper.ber_models.to(device)
            for ber_model in self.flipper.ber_models:
                ber_model.wireless_p_lower = ber_model.wireless_p_lower.to(device)
                ber_model.wireless_p_upper = ber_model.wireless_p_upper.to(device)

    def noisy_training_mode(self, val: bool = False):
        self.noisy = val

    def noisy_eval_mode(self, val: bool = False):
        self.evaluate_noisy = val

    def use_channel_model_for_eval(self, val: bool = False):
        self.channel_model_eval = val

    def track_trained_wireless_param(self, val: bool = False):
        self.tracking_trained_wireless_param = val

    def enable_activation_buffer(self, enable: bool = False):
        for module in self.modules():
            if isinstance(module, QReLU):
                module.tracking_activation = enable
        return

    def fold_fusion(self):
        for module in self.modules():
            if isinstance(module, QConv2d):
                module.fold_bn()
        return

    def training_quantize(self, verbose: bool = False):
        # training quantize
        for module in self.modules():
            if isinstance(module, (QConv2d, QReLU)):
                if module.q_mode == 'training':
                    module.training_quantize()
                    if verbose:
                        print('===================================================================================')
                        print(module)
                        print('\ttraining q scale: ', module.scale)
                        print('\ttraining q zero_point: ', module.zero_point)
                        print('\ttraining q bit width: ', module.bit_width)
        return

    def _post_quantize_w(self, module, bit_width, verbose: bool = False):
        if isinstance(module, QConv2d):
            if not module.quantized:
                q_w, scale, offset = self.post_q_machine.quantize(m=module.weight, bit_width=bit_width)
                # print('difference: ', torch.mean((module.weight - (q_w * scale + offset)).abs()))
                # print('raw max: {}; raw min: {}'.format(torch.max(module.weight), torch.min(module.weight)))
                # print('raw mean: {}: '.format(torch.mean(torch.abs(module.weight))))
                zero_point = torch.round(- offset / scale)
                module.quantized = True
                module.weight.data = q_w.detach().data
                module.scale = scale
                module.zero_point = zero_point
                module.bit_width = torch.tensor(bit_width, dtype=module.dtype, device=module.weight.device)
                if verbose:
                    print('===================================================================================')
                    print(module)
                    print('\tpost q w scale: ', module.scale)
                    print('\tpost q w zero_point: ', module.zero_point)
                    print('\tpost q w bit width: ', module.bit_width)
        return

    def post_quantize_w(self, post_q_indices, post_w_bit_width):
        # post-training quantize weight
        for idx in post_q_indices:
            if idx == -1:
                self._post_quantize_w(self.conv0, post_w_bit_width)
            else:
                block = self._modules['res_block{}'.format(idx)]
                for module in block.modules():
                    self._post_quantize_w(module, post_w_bit_width)
        return

    @staticmethod
    def _enable_activation_buffer(module):
        if isinstance(module, QReLU):
            if not module.quantized:
                module.tracking_activation = True
        return

    def enable_activation_buffers(self, post_q_indices):
        for idx in post_q_indices:
            if idx == -1:
                self._enable_activation_buffer(self.relu0)
            else:
                block = self._modules['res_block{}'.format(idx)]
                for module in block.modules():
                    self._enable_activation_buffer(module)
        return

    def load_activation_buffers(self):
        buffers = []
        for module in self.modules():
            if isinstance(module, QReLU) and module.tracking_activation:
                buffers.append(module.activation_record.clone())
        return buffers

    def quantize(
        self,
        post_q_w_indices: Union[List[int], Tuple[int]],
        post_q_a_indices: Union[List[int], Tuple[int]],
        post_w_bit_width,
        post_a_bit_width,
        calibration_loader,
        calibration_epochs,
        DDP_mode: bool = False,
        world_size: int = None,
    ):
        # fold all conv layer and bn
        self.fold_fusion()
        self.training_quantize()
        self.post_quantize_w(post_q_w_indices, post_w_bit_width)
        if len(post_q_a_indices) == 0:
            return

        self.enable_activation_buffers(post_q_a_indices)
        # approximate activation
        median_buffers = None
        alpha_buffers = None

        print('===================================================================================')
        print('Calibrating for activation post quantization ...')
        for epoch in range(calibration_epochs):
            for batch_idx, batch_data in enumerate(calibration_loader):
                batch_images, _ = batch_data
                batch_images = batch_images.to(next(self.parameters()).device)
                self.forward(batch_images)
                batch_buffers = self.load_activation_buffers()
                batch_median = []
                batch_alpha = []
                for buffer in batch_buffers:
                    median, alpha = self.post_q_machine.calculate_relu_median_range(m=buffer, bit_width=post_a_bit_width)
                    batch_median.append(median.unsqueeze(0))
                    batch_alpha.append(alpha.unsqueeze(0))

                # dim 0: activation num; dim 1: data_num
                batch_median = torch.stack(batch_median, dim=0)
                batch_alpha = torch.stack(batch_alpha, dim=0)
                if DDP_mode:
                    batch_median_list = [torch.zeros((1, 1), dtype=torch.float32) for _ in range(world_size)]
                    batch_alpha_list = [torch.zeros((1, 1), dtype=torch.float32) for _ in range(world_size)]
                    dist.all_gather(batch_median_list, batch_median)
                    dist.all_gather(batch_alpha_list, batch_alpha)
                    batch_median = torch.cat(batch_median_list, dim=0)
                    batch_alpha = torch.cat(batch_alpha_list, dim=0)

                # print(batch_buffers[0])
                print('-- epoch {} -- batch {} -- median: '.format(epoch, batch_idx), batch_median)
                print('-- epoch {} -- batch {} -- alpha: '.format(epoch, batch_idx), batch_alpha)
                if median_buffers is None:
                    median_buffers = batch_median
                    alpha_buffers = batch_alpha
                else:
                    median_buffers = torch.cat((median_buffers, batch_median), dim=1)
                    alpha_buffers = torch.cat((alpha_buffers, batch_alpha), dim=1)
                    # for idx in range(len(median_buffers)):
                    #     median_buffers[idx] = torch.cat((median_buffers[idx], batch_median[idx]), dim=0)
                    #     alpha_buffers[idx] = torch.cat((alpha_buffers[idx], batch_alpha[idx]), dim=0)
        # quantize according to buffers.
        a_idx = 0
        for module in self.modules():
            if isinstance(module, QReLU) and module.tracking_activation:
                median = torch.quantile(median_buffers[a_idx], q=0.5)
                alpha = torch.mean(alpha_buffers[a_idx])
                scale, offset = self.post_q_machine.relu_median_alpha_to_offset_scale(median=median, alpha=alpha,
                                                                                      bit_width=post_a_bit_width)
                zero_point = torch.round(- offset / scale)
                module.quantized = True
                module.scale = scale
                module.zero_point = zero_point
                module.bit_width = torch.tensor(post_a_bit_width, dtype=module.dtype)
                a_idx += 1
                print('===================================================================================')
                print(module)
                print('\tpost q a scale: ', module.scale)
                print('\tpost q a zero_point: ', module.zero_point)
                print('\tpost q a bit width: ', module.bit_width)
        return

    def set_w_q_param(self, param_name: str, value: float):
        if param_name not in ['clip_lower', 'clip_upper', 'slope']:
            raise ValueError("available params: 'clip_lower', 'clip_upper', 'slope'.")
        for module in self.modules():
            if isinstance(module, QConv2d) and module.training_q:
                module.set_param(param_name=param_name, value=value)
        return

    def clip_w_q_param(self, param_name: str, bot: float, top: float):
        if param_name not in ['clip_lower', 'clip_upper', 'slope']:
            raise ValueError("available params: 'clip_lower', 'clip_upper', 'slope'.")
        for module in self.modules():
            if isinstance(module, QConv2d) and module.training_q:
                module.clip_(param_name=param_name, bot=bot, top=top)
        return

    def clip_a_q_param(self, param_name: str, bot: float, top: float):
        if param_name not in ['clip_lower', 'clip_upper', 'slope']:
            raise ValueError("available params: 'clip_lower', 'clip_upper', 'slope'.")
        for module in self.modules():
            if isinstance(module, QReLU) and module.training_q:
                module.a_dsq.clip_(param_name=param_name, bot=bot, top=top)
        return

    def set_a_q_param(self, param_name: str, value: float):
        if param_name not in ['clip_lower', 'clip_upper', 'slope']:
            raise ValueError("available params: 'clip_lower', 'clip_upper', 'slope'.")
        for module in self.modules():
            if isinstance(module, QReLU) and module.training_q:
                module.a_dsq.set_param(param_name=param_name, value=value)
        return

    def get_a_q_param(self):
        lowers, uppers, slopes = [], [], []
        for module in self.modules():
            if isinstance(module, QReLU) and module.training_q:
                lowers.append(module.a_dsq.__getattr__('clip_lower').detach().data)
                uppers.append(module.a_dsq.__getattr__('clip_upper').detach().data)
                slopes.append(module.a_dsq.__getattr__('slope').detach().data)
        return lowers, uppers, slopes

    def get_w_q_param(self):
        lowers, uppers, slopes = [], [], []
        for module in self.modules():
            if isinstance(module, QConv2d) and module.training_q:
                lowers.append(module.w_dsq.__getattr__('clip_lower').detach().data)
                uppers.append(module.w_dsq.__getattr__('clip_upper').detach().data)
                slopes.append(module.w_dsq.__getattr__('slope').detach().data)
        return lowers, uppers, slopes

    def set_channel_param(self, name: str, val):
        valid_names = ["avg_snr", "aimc_bit", "order"]
        if name not in valid_names:
            raise ValueError("Only these params valid: {}".format(", ".join(valid_names)))
        self.channel_simulator.set_params(name, val)
        # print("Reset the {} to {}".format(name, self.channel_simulator.__getattribute__(name)))

    def symbol_wise_random_flip(
        self,
        raw_data: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        eval_wireless_params: torch.Tensor=None,
        trace_mode: bool = False,
    ):
        r"""
        Differentiable randomly flipping the feature bits symbol-wise

        Args:
            raw_data (torch.Tensor): The intermediate data containing discrete values to be transmitted.
            lower (torch.Tensor): The clipping lower bound of quantization.
            upper (torch.Tensor): The clipping upper bound of quantization.
            eval_wireless_params (Optional[torch.Tensor]): The externally specified eval wireless params.
            trace_mode (bool): If True, enable training trace mode.

        Returns:
            noisy_data.
        """
        if trace_mode:
            with torch.profiler.record_function("Differentiable serialization"):
                bits = self.serializer.forward(quantized_data=raw_data, lower=lower, upper=upper)
        else:
            bits = self.serializer.forward(quantized_data=raw_data, lower=lower, upper=upper)

        # bits = self.serializer.forward(quantized_data=raw_data, lower=lower, upper=upper)
        if self.flipper is None:
            raise ValueError("No valid ber model.")
        self.flipper: Union[SymbolWiseDFlip, MetaSymbolWiseFlip]
        if self.channel_model_eval and not self.training:
            if isinstance(self.flipper, SymbolWiseDFlip):
                # Use channel simulator to evaluate.
                if eval_wireless_params is not None:
                    eval_param = eval_wireless_params.detach().cpu().item()
                else:
                    eval_param = self.flipper.meaningful_wireless_param().detach().cpu().item()

                if self.tracking_trained_wireless_param:
                    self.set_channel_param(name="avg_snr", val=eval_param)

                print("SNR: ", self.channel_simulator.avg_snr)
                print(": AIMC bit", self.channel_simulator.aimc_bit)
                print(": OFDM_order", self.channel_simulator.order)
                print(": multi_path_num", self.channel_simulator.multi_path_num)
                print(": min_path_delay", self.channel_simulator.min_path_delay)
                print(": max_path_delay", self.channel_simulator.max_path_delay)
                print(": max_path_delay", self.channel_simulator.max_path_delay)

                # (batch_size, C, H, W, Bit-width)
                noisy_bits = self.channel_simulator.forward(batch_bits=bits)
            elif isinstance(self.flipper, MetaSymbolWiseFlip):
                if eval_wireless_params is not None:
                    eval_params = eval_wireless_params.detach().cpu().numpy()
                else:
                    eval_params = self.flipper.meaningful_wireless_param().detach().cpu().numpy()

                noisy_bits = []
                for model_idx, each_p in enumerate(eval_params):
                    if self.tracking_trained_wireless_param:
                        self.set_channel_param(name="avg_snr", val=each_p)
                    for cfg_name in self.flipper.eval_channel_configs.keys():
                        cfg_val_list = self.flipper.eval_channel_configs[cfg_name]
                        # E.g. aimc_bit, order.
                        self.set_channel_param(name=cfg_name, val=cfg_val_list[model_idx])
                        # print("Set the channel model: ", cfg_name, self.channel_simulator.__getattribute__(cfg_name))

                    # Channel simulation.
                    each_noisy_bits = self.channel_simulator.forward(batch_bits=bits)

                    noisy_bits.append(each_noisy_bits)
                # (models_num * batch_size, C, H, W, Bit-width)
                noisy_bits = torch.cat(noisy_bits, dim=0)
            else:
                raise ValueError("Invalid Flipper.")

        else:
            if trace_mode:
                with torch.profiler.record_function("Differentiable bit flipping"):
                    # (batch_size, C, H, W, Bit-width)
                    noisy_bits = self.flipper.forward(bits=bits, training_mode=self.training, trace_mode=trace_mode)
            else:
                # (batch_size, C, H, W, Bit-width)
                noisy_bits = self.flipper.forward(bits=bits, training_mode=self.training)

        # wrong_rate = (bits - noisy_bits).abs().sum() / torch.prod(torch.tensor(noisy_bits.size()))
        # print("Flip rate: ", wrong_rate)

        if trace_mode:
            with torch.profiler.record_function("Differentiable deserialize"):
                noisy_data = self.deserializer.forward(bits=noisy_bits, lower=lower, upper=upper)
        else:
            noisy_data = self.deserializer.forward(bits=noisy_bits, lower=lower, upper=upper)
        return noisy_data

    def noisy_forward(
        self,
        batch_input: torch.Tensor,
        eval_wireless_params: torch.Tensor = None,
        trace_mode: bool = False,
    ):
        # # TODO: Trace training:
        # with torch.profiler.record_function("Edge forward"):

        edge_data = self.conv0(batch_input)
        edge_data = self.relu0(edge_data)
        if self.max_pool is not None:
            edge_data = self.max_pool(edge_data)

        for block_idx in range(self.edge_end_block_idx + 1):
            edge_data = self._modules[f"res_block{block_idx}"](edge_data)
            # print("Relu bit-width: ", self._modules['res_block0'].relu2.a_dsq.bit_width)

        # TODO: Trace training:
        # with torch.profiler.record_function("Get upper lower"):

        if self.bottleneck_type:
            upper = self._modules[f"res_block{self.edge_end_block_idx}"].relu3.a_dsq.clip_upper.detach()
            lower = self._modules[f"res_block{self.edge_end_block_idx}"].relu3.a_dsq.clip_lower.detach()
        else:
            upper = self._modules[f"res_block{self.edge_end_block_idx}"].relu2.a_dsq.clip_upper.detach()
            lower = self._modules[f"res_block{self.edge_end_block_idx}"].relu2.a_dsq.clip_lower.detach()

        # TODO: Trace training:
        # with torch.profiler.record_function("Flipping forward"):

        noisy_data = self.symbol_wise_random_flip(
            raw_data=edge_data,
            lower=lower,
            upper=upper,
            eval_wireless_params=eval_wireless_params,
            trace_mode=trace_mode,
        )
        # TODO: Test no gradient for wireless param.
        if edge_data.size() != noisy_data.size():
            repeat_num = noisy_data.size()[0] // edge_data.size()[0]
            edge_data = torch.cat([edge_data for _ in range(repeat_num)], dim=0)

        batch_output = (- edge_data).detach() + edge_data + noisy_data

        # TODO: Trace training:
        # with torch.profiler.record_function("Cloud forward"):

        # rest inference.
        for block_idx in range(self.edge_end_block_idx + 1, self.blocks_num):
            batch_output = self._modules['res_block{}'.format(block_idx)](batch_output)
        batch_output = self.pooling(batch_output)
        batch_output = self.fc(batch_output.view(batch_output.size()[0], -1))
        return batch_output

    def ideal_forward(self, batch_input: torch.Tensor):
        batch_output = self.conv0(batch_input)
        batch_output = self.relu0(batch_output)
        if not self.max_pool is None:
            batch_output = self.max_pool(batch_output)

        for block_idx in range(self.blocks_num):
            batch_output = self._modules['res_block{}'.format(block_idx)](batch_output)
        batch_output = self.pooling(batch_output)
        batch_output = self.fc(batch_output.view(batch_output.size()[0], -1))
        return batch_output

    def forward(self, batch_input: torch.Tensor, wireless_params: torch.Tensor = None, trace_mode: bool=False):
        if (self.noisy and self.training) or (self.evaluate_noisy and not self.training):
            batch_output = self.noisy_forward(batch_input, eval_wireless_params=wireless_params, trace_mode=trace_mode)
        else:
            batch_output = self.ideal_forward(batch_input)
        return batch_output
