import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Function


class Serialize(Function):
	r"""
	Serialize integers into corresponding bits, and compute the gradients of these integers in backward propagation.
	"""
	@staticmethod
	def forward(ctx, integers, bit_width):
		ctx.save_for_backward(integers, bit_width)
		return integers

	@staticmethod
	def backward(ctx, grad_output):
		integers, bit_width = ctx.saved_tensors
		return grad_output, None


def int_to_bin(integers: torch.Tensor, bit_width: int):
	r""" The LSB is at the head. """
	quotient = integers
	bits = torch.zeros((*integers.size(), bit_width), device=integers.device)
	for i in range(bit_width):
		next_quotient = torch.floor(0.5 * quotient)
		bit = quotient - 2 * next_quotient
		bits[..., i] = bit
		quotient = next_quotient
	return bits


def bin_to_int(bits: torch.Tensor, bit_width: int):
	r"""
	Transform binary stream to integers.

	Args:
		bits (torch.Tensor): The bit stream of shape (..., bit_width); The LSB is at the head.
		bit_width (int): The bit_width of integers.

	Returns:
		The transformed integers.
	"""
	assert bits.size()[-1] == bit_width
	bit_indices = torch.arange(0, bit_width, 1, device=bits.device)
	scale = torch.exp2(bit_indices)
	integers = torch.matmul(bits, scale)
	return integers


class DSerialize(nn.Module):
	def __init__(self, bit_width: int):
		super().__init__()
		self.bit_width = bit_width

	def forward(self, quantized_data: torch.tensor, lower: torch.Tensor, upper: torch.Tensor):
		r"""
		Quantize data into integers and serialize them into binary stream in a differentiable way.

		Args:
			quantized_data (torch.Tensor): Data that are quantized to discrete values.
			lower (torch.Tensor): The clipping lower bound when transforming values to integers.
			upper (torch.Tensor): The clipping upper bound when transforming values to integers.

		Returns:
			The serialized binary stream of data.

		"""
		split_num = 2 ** self.bit_width - 1
		interval = torch.abs(upper - lower) / split_num
		raw_integers = (quantized_data - lower) / interval

		hard_integers = torch.clip(torch.round(raw_integers), 0, 2 ** self.bit_width - 1)
		integers = (hard_integers - raw_integers).detach() + raw_integers
		with torch.no_grad():
			bits = int_to_bin(integers=integers, bit_width=self.bit_width)

		# bits = integers
		return bits


class DDeSerialize(nn.Module):
	def __init__(self, bit_width: int):
		super().__init__()
		self.bit_width = bit_width

	def forward(self, bits: torch.tensor, lower: torch.Tensor, upper: torch.Tensor):
		r"""
		DeSerialize bits into integers, then de-quantize these integers into discrete values in a differentiable way.

		Args:
			bits (torch.Tensor): Binary stream of intermediate data.
			lower (torch.Tensor): The clipping lower bound of quantization.
			upper (torch.Tensor): The clipping upper bound of quantization.

		Returns:
			The deserialized and de-quantized data.
		"""
		integers = bin_to_int(bits=bits, bit_width=self.bit_width)

		# integers = bits
		split_num = 2 ** self.bit_width - 1
		interval = torch.abs(upper - lower) / split_num
		dequantized_data = integers * interval + lower
		return dequantized_data


if __name__ == "__main__":
	# dd = DSerialize(bit_width=1)
	# rede = DDeSerialize(bit_width=1)
	# unit = 1.
	# lower = 0
	# upper = 1.5
	# random_data = torch.rand(10) * (upper - lower) + lower
	# s_bits = dd.forward(quantized_data=random_data, lower=lower, upper=upper)
	#
	# re_data = rede.forward(bits=s_bits, lower=lower, upper=upper)
	# print(random_data)
	# print(s_bits)
	#
	# print("re: ")
	# print(re_data)


	aa = torch.randint(0, 8, size=(2, 3))
	bb = int_to_bin(integers=aa, bit_width=3)
	cc = bin_to_int(bits=bb, bit_width=3)
	print(aa)
	print(bb)
	print(cc)

	loss = bb.sum()






