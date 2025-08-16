import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Function
from pathlib import Path
from src.channel_simulate.fitting import BERModel
from copy import deepcopy
from typing import List, Union, Tuple


class SGN(Function):
	@staticmethod
	def forward(ctx, inputs):
		return torch.sgn(inputs)

	@staticmethod
	def backward(ctx, grad_output):
		return grad_output


class MetaSymbolWiseFlip(nn.Module):
	r"""
	Multiple symbol-wise flip.
	"""
	def __init__(
		self,
		ber_model_paths: List[str],
		eval_channel_configs: dict,
		init_wireless_p: Union[float, List[float]],
		flat_channel_aug_mode: bool = True,
		freq_selective_aug_mode: bool = True,
		wireless_p_scale: float = 1.,
	):
		super().__init__()
		self.meta_size = len(ber_model_paths)
		self.eval_channel_configs = eval_channel_configs
		for key in eval_channel_configs.keys():
			if len(eval_channel_configs[key]) != self.meta_size:
				raise ValueError(f"Should give {self.meta_size} groups of channel model parameters.")
		self.ber_models = nn.ModuleList(
			[
				torch.load(model_path, map_location=torch.device("cpu")) for model_path in ber_model_paths
			]
		)
		self.wireless_p_scale = wireless_p_scale
		# init_param = torch.tensor([init_wireless_p * wireless_p_scale])
		init_param = torch.ones(self.meta_size).mul(init_wireless_p * wireless_p_scale)
		print("Init scaled wireless param: ", init_param)
		self.wireless_p = nn.Parameter(init_param)
		self.flat_channel_aug_mode = flat_channel_aug_mode
		self.freq_selective_aug_mode = freq_selective_aug_mode
		for ber_model in self.ber_models:
			for p in ber_model.parameters():
				p.requires_grad = False

	def reload_ber_model(self, ber_model_paths: List[str], device: torch.device):
		r""" Attention! Here we assume the new ber_model is the same type as the former ber_model """
		self.ber_models = nn.ModuleList(
			[
				torch.load(model_path, map_location=device) for model_path in ber_model_paths
			]
		)

		for ber_model in self.ber_models:
			for p in ber_model.parameters():
				p.requires_grad = False

	def get_wireless_param_str(self):
		w_p = self.meaningful_wireless_param()
		w_p = w_p.detach().cpu().numpy()
		w_p_str = "\t".join(["{:.4f}".format(each_p) for each_p in w_p])
		state_str = "Meaningful Wireless Parameter: {}".format(w_p_str)
		return state_str

	def set_meaningful_wireless_param(self, meaningful_param: torch.Tensor):
		r""" Meaningful param; E.g. SNR: dB"""
		assert meaningful_param.size() == self.wireless_p.data.size()
		meaningful_param = meaningful_param.to(self.wireless_p.device)

		lower_upper = [self._get_ber_model_lower_upper(idx) for idx in range(self.meta_size)]
		lower_upper = torch.stack(lower_upper, dim=0)
		wireless_p_lower = lower_upper[:, 0]
		wireless_p_upper = lower_upper[:, 1]
		norm_param = (meaningful_param - wireless_p_lower) / (wireless_p_upper - wireless_p_lower)
		param = norm_param.clip(0, 1).mul(self.wireless_p_scale)
		self.set_wireless_param(param)

		# for ber_model_idx in range(self.meta_size):
		# 	lower_upper = self._get_ber_model_lower_upper(model_idx=ber_model_idx)
		# 	wireless_p_lower = lower_upper[0]
		# 	wireless_p_upper = lower_upper[1]
		# 	norm_p = ()
		#
		# 	norm_param = (meaningful_param - wireless_p_lower) / (wireless_p_upper - wireless_p_lower)
		# 	param = norm_param.clip(0, 1).mul(self.wireless_p_scale)
		# self.set_wireless_param(param)

	def set_wireless_param(self, param: torch.Tensor):
		assert param.size() == self.wireless_p.data.size()
		if param.device != self.wireless_p.device:
			param = param.to(self.wireless_p.device)
		self.wireless_p.data = param

	def clamp_(self):
		with torch.no_grad():
			# TODO: Use normalized wireless param
			self.wireless_p.data = torch.clamp(self.wireless_p.data, 0, self.wireless_p_scale, )

	def _get_ber_model_lower_upper(self, model_idx: int) -> torch.Tensor:
		ber_model = self.ber_models[model_idx]
		if ber_model.wireless_p_lower.device != self.wireless_p.device:
			wireless_p_lower = ber_model.wireless_p_lower.to(self.wireless_p.device)
			wireless_p_upper = ber_model.wireless_p_upper.to(self.wireless_p.device)
		else:
			wireless_p_lower = ber_model.wireless_p_lower
			wireless_p_upper = ber_model.wireless_p_upper

		return torch.cat([wireless_p_lower, wireless_p_upper], dim=0)

	def meaningful_wireless_param(self) -> torch.Tensor:
		r"""

		Returns:
			params (torch.Tensor): -- Size: (ber_model_num, )

		"""
		lower_upper = [self._get_ber_model_lower_upper(model_idx=idx) for idx in range(len(self.ber_models))]
		lower_upper = torch.stack(lower_upper, dim=0)
		wireless_p_lower = lower_upper[:, 0]
		wireless_p_upper = lower_upper[:, 1]
		params = self.wireless_p.div(self.wireless_p_scale).clamp(0, 1.) * (wireless_p_upper - wireless_p_lower) + wireless_p_lower
		return params

	def wireless_loss(self):
		# TODO: Use normalized wireless param
		cost = self.wireless_p.sum()
		return cost

	def freq_selective_augmentation(self, ber_vec: torch.Tensor):
		with torch.no_grad():
			std = (0.5 - ber_vec) / 3
			noise = torch.randn(ber_vec.size(), device=ber_vec.device).mul(std).abs()
		new_vec = ber_vec + noise
		new_vec = new_vec.clip(0, 0.5)
		return new_vec

	def _get_ber(self, training_mode: bool = False) -> List[torch.Tensor]:
		r"""

		Args:
			training_mode (bool):

		Returns:
			ber_list (list): The BER of multiple models -- Size: (models_num, symbol_bits_num).
		"""
		meaningful_wireless_p = self.meaningful_wireless_param()
		ber_list = [
			self.ber_models[model_idx].forward(meaningful_wireless_p[model_idx]) for model_idx in range(len(self.ber_models))
		]
		return ber_list

	def _channel_augmentation(
		self,
		symbol_wise_ber: torch.Tensor,
		batch_size: int,
		sample_symbol_num: int,
		training_mode: bool = False,
	):
		r"""

		Return:
			ber_mask (torch.Tensor): -- Size: (batch_size, sample_symbol_num, symbol_bits_num)
		"""
		symbol_bits_num = symbol_wise_ber.size()[-1]

		flat_prob = torch.rand(1)
		freq_selective_prob = torch.rand(1)

		if training_mode and self.flat_channel_aug_mode and flat_prob < 0.5:
			symbol_wise_ber = symbol_wise_ber * torch.rand(1, device=self.wireless_p.device)

		if training_mode and self.freq_selective_aug_mode and freq_selective_prob < 0.5:
			symbol_wise_ber = self.freq_selective_augmentation(ber_vec=symbol_wise_ber)

		ber_mask = torch.broadcast_to(
			symbol_wise_ber[None, None, :],
			size=(batch_size, int(sample_symbol_num), symbol_bits_num),
		)
		return ber_mask

	def _channel_forward(self, flatten_bits: torch.Tensor, ber_mask: torch.Tensor):
		r"""

		Args:
			flatten_bits: (batch_size, -1)
			ber_mask: (batch_size,  sample_symbol_num, symbol_bits_num)

		Returns:
			flipped_bits: (batch_size, -1)
		"""
		batch_size = flatten_bits.size()[0]

		ber_mask = torch.reshape(ber_mask, (batch_size, -1))
		ber_mask = ber_mask[..., :flatten_bits.size()[-1]]
		samp = torch.rand(*ber_mask.size(), device=flatten_bits.device)
		sgn = torch.sgn(samp - ber_mask) - (samp - ber_mask).detach() + (samp - ber_mask)
		flipped_bits = (1 + (2 * flatten_bits - 1) * sgn) / 2
		return flipped_bits

	# def eval_forward(self, bits: torch.Tensor):
	# 	batch_size = bits.size()[0]
	# 	flatten_bits = torch.reshape(bits, (batch_size, -1))
	# 	# (models_num, symbol_bits_num)
	# 	valid_bers_list = self._get_ber(training_mode=False)
	# 	models_num = len(valid_bers_list)
	#
	# 	bits_list = []
	#
	# 	for model_idx in range(models_num):
	# 		each_flipped = self._channel_forward(flatten_bits=flatten_bits, symbol_wise_ber=valid_bers_list[model_idx])
	# 		bits_list.append(each_flipped)
	#
	# 	# (models_num, batch_size, -1)
	# 	flipped_bits = torch.stack(bits_list, dim=0)
	#
	# 	# Flipped or no flipped
	# 	output_size = torch.Size([models_num * batch_size, ]) + bits.size()[1:]
	# 	flipped_bits = torch.reshape(flipped_bits, output_size)
	# 	return flipped_bits


	def forward(self, bits: torch.Tensor, training_mode: bool = False, trace_mode: bool = False):
		r"""
		Randomly flip bits according to the symbol-wise ber given by the ber model.

		Args:
			bits (torch.Tensor): The serialized bits of a batch of intermediate data. (batch_size, C, H, W, Bit-width)
			training_mode (float): If training, the no_flip_prob is valid.
			trace_mode (float): If True, training trace mode.

		Returns:
			flipped_bits (torch.Tensor): The flipped bits with the size: (batch_size * ber_model_num, ...)

		"""
		batch_size = bits.size()[0]
		raw_size = bits.size()
		flatten_bits = torch.reshape(bits, (batch_size, -1))
		# (models_num, symbol_bits_num)
		if trace_mode:
			with torch.profiler.record_function("BER models forward"):
				valid_bers_list = self._get_ber(training_mode=training_mode)
		else:
			valid_bers_list = self._get_ber(training_mode=training_mode)

		flipped_bits = []

		# # Forward 3 times
		# for model_idx in range(self.meta_size):
		# 	symbol_bits_num = valid_bers_list[model_idx].size()[-1]
		# 	sample_symbol_num = np.ceil(flatten_bits.size()[-1] / symbol_bits_num)
		# 	ber_mask = self._channel_augmentation(
		# 		symbol_wise_ber=valid_bers_list[model_idx],
		# 		batch_size=batch_size,
		# 		sample_symbol_num=sample_symbol_num,
		# 		training_mode=training_mode,
		# 	)
		# 	each_flipped_bits = self._channel_forward(flatten_bits=flatten_bits, ber_mask=ber_mask)
		# 	each_flipped_bits = torch.reshape(each_flipped_bits, raw_size)
		# 	flipped_bits.append(each_flipped_bits)
		# flipped_bits = torch.cat(flipped_bits, dim=0)

		# Forward One times
		end = 0
		sub_batch_size = int(np.ceil(batch_size / self.meta_size))
		for model_idx in range(self.meta_size):
			if end < batch_size:
				sub_flatten_bits = flatten_bits[end: end + sub_batch_size]
				end += sub_batch_size
				symbol_bits_num = valid_bers_list[model_idx].size()[-1]
				sample_symbol_num = np.ceil(sub_flatten_bits.size()[-1] / symbol_bits_num)
				ber_mask = self._channel_augmentation(
					symbol_wise_ber=valid_bers_list[model_idx],
					batch_size=sub_flatten_bits.size()[0],
					sample_symbol_num=sample_symbol_num,
					training_mode=training_mode,
				)
				each_flipped_bits = self._channel_forward(flatten_bits=sub_flatten_bits, ber_mask=ber_mask)
				each_flipped_bits = torch.reshape(each_flipped_bits, sub_flatten_bits.size()[:1] + raw_size[1:])
				flipped_bits.append(each_flipped_bits)
		flipped_bits = torch.cat(flipped_bits, dim=0)
		return flipped_bits


class SymbolWiseDFlip(nn.Module):
	r"""
	Symbol-wise random flip.
	"""
	def __init__(
		self,
		ber_model_path: str,
		init_wireless_p: float,
		flat_channel_aug_mode: bool = True,
		wireless_p_scale: float = 1.,
		freq_selective_aug_mode: bool = False,
	):
		super().__init__()
		self.ber_model: BERModel = torch.load(ber_model_path, map_location=torch.device("cpu"))
		wireless_p_lower = self.ber_model.wireless_p_lower
		wireless_p_upper = self.ber_model.wireless_p_upper
		self.wireless_p_scale = wireless_p_scale
		init_param = torch.tensor([init_wireless_p * wireless_p_scale])
		print("Init scaled wireless param: ", init_param)

		# Raw wireless param
		# init_param = init_wireless_p * (wireless_p_upper - wireless_p_lower) + wireless_p_lower

		self.wireless_p = nn.Parameter(init_param)
		self.freq_selective_aug_mode = freq_selective_aug_mode
		if self.freq_selective_aug_mode:
			print("Add freq selective aug mode.")

		self.flat_channel_aug_mode = flat_channel_aug_mode
		for p in self.ber_model.parameters():
			p.requires_grad = False

	def reload_ber_model(self, ber_model_path: str, device: torch.device):
		r""" Attention! Here we assume the new ber_model is the same type as the former ber_model """
		self.ber_model = torch.load(ber_model_path, map_location=device)
		for p in self.ber_model.parameters():
			p.requires_grad = False

	def get_wireless_param_str(self):
		w_p = self.meaningful_wireless_param()
		w_p = w_p.detach().cpu().item()

		state_str = "Meaningful Wireless Parameter: {:.6f}".format(w_p)
		return state_str

	def set_meaningful_wireless_param(self, meaningful_param: torch.Tensor):
		r""" Meaningful param; E.g. SNR: dB"""
		assert meaningful_param.size() == self.wireless_p.data.size()
		meaningful_param = meaningful_param.to(self.wireless_p.device)
		wireless_p_lower = self.ber_model.wireless_p_lower.to(self.wireless_p.device)
		wireless_p_upper = self.ber_model.wireless_p_upper.to(self.wireless_p.device)
		norm_param = (meaningful_param - wireless_p_lower) / (wireless_p_upper - wireless_p_lower)
		param = norm_param.clip(0, 1) * self.wireless_p_scale
		self.set_wireless_param(param)

	def set_wireless_param(self, param: torch.Tensor):
		assert param.size() == self.wireless_p.data.size()
		if param.device != self.wireless_p.device:
			param = param.to(self.wireless_p.device)
		self.wireless_p.data = param

	def clamp_(self):
		with torch.no_grad():
			# TODO: Use normalized wireless param
			self.wireless_p.data = torch.clamp(
				self.wireless_p.data,
				0,
				self.wireless_p_scale,
			)

	def meaningful_wireless_param(self):
		# TODO: Use normalized wireless param
		if self.ber_model.wireless_p_lower.device != self.wireless_p.device:
			wireless_p_lower = self.ber_model.wireless_p_lower.to(self.wireless_p.device)
			wireless_p_upper = self.ber_model.wireless_p_upper.to(self.wireless_p.device)
		else:
			wireless_p_lower = self.ber_model.wireless_p_lower
			wireless_p_upper = self.ber_model.wireless_p_upper
		param = self.wireless_p.div(self.wireless_p_scale) * (wireless_p_upper - wireless_p_lower) + wireless_p_lower
		return param

	def wireless_loss(self):
		# TODO: Use normalized wireless param
		wireless_p = self.meaningful_wireless_param()

		# raw param
		# wireless_p = self.wireless_p

		clamped_p = torch.clamp(wireless_p, self.ber_model.wireless_p_lower, self.ber_model.wireless_p_upper)
		# Loss for average SNR
		cost = clamped_p.add(- self.ber_model.wireless_p_lower).div(self.ber_model.wireless_p_upper - self.ber_model.wireless_p_lower)

		# Loss for aimc precision
		# cost = torch.exp2(clamped_p) / torch.exp2(self.ber_model.wireless_p_upper)
		return cost

	def freq_selective_augmentation(self, ber_vec: torch.Tensor):
		mean = ber_vec.mean()

		# # Uniform
		# new_vec = torch.rand(ber_vec.size(), device=ber_vec.device)
		# new_vec = new_vec.div(new_vec.mean()).mul(mean)
		# new_vec = new_vec.clip(0, 0.5)

		with torch.no_grad():
			std = (0.5 - ber_vec) / 3
			noise = torch.randn(ber_vec.size(), device=ber_vec.device).mul(std).abs()
		new_vec = ber_vec + noise
		new_vec = new_vec.clip(0, 0.5)
		return new_vec

	def forward(self, bits: torch.Tensor, training_mode: bool = False):
		r"""
		Randomly flip bits according to the symbol-wise ber given by the ber model.

		Args:
			bits (torch.Tensor): The serialized bits of a batch of intermediate data. (batch_size, C, H, W, Bit-width)
			training_mode (float): If training, the no_flip_prob is valid.

		Returns:
			flipped_bits (torch.Tensor): The flipped bits with the same size of bits.

		"""
		flat_prob = torch.rand(1)
		freq_selective_prob = torch.rand(1)

		flatten_bits = torch.reshape(bits, (bits.size()[0], -1))
		meaningful_wireless_p = self.meaningful_wireless_param()
		valid_ber = self.ber_model(meaningful_wireless_p)

		# Flat fading augment
		if training_mode and self.flat_channel_aug_mode and flat_prob < 0.5:
			valid_ber = valid_ber * torch.rand(1, device=valid_ber.device)

		# Freq-selective augment
		if training_mode and self.freq_selective_aug_mode and freq_selective_prob < 0.5:
			valid_ber = self.freq_selective_augmentation(ber_vec=valid_ber)

		sample_symbol_num = np.ceil(flatten_bits.size()[-1] / valid_ber.size()[0])
		ber_mask = torch.broadcast_to(
			valid_ber.unsqueeze(0).unsqueeze(0),
			size=(flatten_bits.size()[0], int(sample_symbol_num), valid_ber.size()[0]),
		)
		ber_mask = torch.reshape(ber_mask, (flatten_bits.size()[0], -1))
		ber_mask = ber_mask[:, :flatten_bits.size()[-1]]
		samp = torch.rand(*flatten_bits.size(), device=bits.device)
		sgn = SGN.apply(samp - ber_mask)
		flipped_bits = (1 + (2*flatten_bits - 1) * sgn) / 2
		# Flipped or no flipped
		flipped_bits = torch.reshape(flipped_bits, bits.size())
		return flipped_bits


def check_flip():
	flipper = SymbolWiseDFlip(ber_model_path=r"D:\python_works\HybridQuantize\checkpoints\channel_models\fitted_model_snr_-5.0dB.pkl")
	test_bits = torch.randint(0, 2, size=(5, 3, 10, 10, 1))
	flipped_bits = flipper(test_bits)
	print("flipped: ", flipped_bits.size())
	wrong_num = (flipped_bits - test_bits).abs().sum()
	print("Wrong num: ", wrong_num)


if __name__ == "__main__":
	check_flip()
