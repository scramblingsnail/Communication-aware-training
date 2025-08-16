import numpy as np


def _symbol_to_amp(symbol_bits: np.ndarray):
	r"""
	Transform the sub-symbols of a series of sub-carriers to corresponding amplitudes.

	Args:
		symbol_bits: shape: (symbol_num, sub-symbol num, sub-symbol bits); sub-symbol: the bits carried by a sub-carrier

	Return:
		The amplitude of each sub-carrier. shape: (symbol_num, sub-symbol num)
	"""
	bits_num = symbol_bits.shape[-1]
	sgn = 1 - symbol_bits[..., :1] * 2
	if bits_num == 1:
		return np.squeeze(sgn, axis=-1)

	bin_factors = np.power(2, np.arange(bits_num - 2, -0.5, -1))
	bin_factors = bin_factors[:, np.newaxis]
	amp_idx = np.matmul(symbol_bits[..., 1:], bin_factors)
	amp_idx = (amp_idx + 1) / np.power(2, bits_num - 1)
	amp_array = amp_idx * sgn
	print("amp array: ", amp_array.shape)
	return np.squeeze(amp_array, axis=-1)


def _amp_to_symbol(amp_array: np.ndarray, sub_carrier_bit_num: int):
	r"""
	Transform the amplitudes of a series of sub-carriers to corresponding sub-symbols.

	Assert the amplitudes has been equalized.

	Args:
		amp_array (np.ndarray): (symbol_num, sub-symbol num); Along the last axis: (Q1, I1, Q2, I2, ...)

	Return:
		The bits; (..., symbol_num, sub-symbol num, sub_carrier_bit_num)
	"""
	sgn_array = np.sign(amp_array)
	# -1 -> 1; +1 -> 0
	# (symbol_num, sub-symbol num)
	sgn_bits = (1 - sgn_array) / 2
	sgn_bits = sgn_bits.astype(int)

	if sub_carrier_bit_num == 1:
		return sgn_bits
	else:
		max_int = np.power(2, sub_carrier_bit_num - 1) - 1
		amp_interval = 1 / np.power(2, (sub_carrier_bit_num - 1))
		int_array = np.round(np.abs(amp_array) / amp_interval) - 1
		int_array = np.clip(int_array, 0, max_int).astype(int)

	mask = np.arange(sub_carrier_bit_num - 2, -0.5, -1) + 1
	mask = mask.astype(int)
	# *int_array.shape, int_bit_num
	mask = np.broadcast_to(mask, shape=(*int_array.shape, *mask.shape))
	# print("mask: \n", mask)
	# print("int array: \n", int_array)
	int_array = np.broadcast_to(int_array[..., np.newaxis], mask.shape)
	# print("broad int array: \n", int_array)
	int_bits = np.bitwise_and(int_array, mask)
	# print("and results: \n", int_bits)
	int_bits = np.not_equal(int_bits, 0).astype(int)
	# print("int: \n", int_bits)
	all_bits = np.concatenate([sgn_bits[..., np.newaxis], int_bits], axis=-1)

	# print("all bits: ", all_bits)
	return all_bits


def parallel_to_serial(parallel_amp: np.ndarray, sub_carrier_bit_num: int):
	r"""
	Transform the amplitudes to bits

	- parallel_amp: (..., sub_carrier_num * 2)
		E.g. (symbols_num, sub_carrier_num * 2), (group_num, symbols_num, sub_carrier_num * 2)

	Args:
		parallel_amp (np.ndarray): (..., sub_carrier_num * 2); The amplitude of all sub-carriers.
			[[sin(f1), cos(f1), sin(f2), cos(f2), ... ], ...]
		sub_carrier_bit_num (int): The amplitude of a sub-carrier is divided into (2**sub_carrier_bit_num)

	Return:
		The serialized bits: (symbols_num, symbol_bits_num);
			[[Q_f1_bit1, I_f1_bit1, Q_f1_bit2, I_f1_bit2, ...], ...]
			symbol_bits_num: sub-carrier_num * sub_carrier_bit_num * 2
	"""
	q_amp = parallel_amp[..., ::2]
	i_amp = parallel_amp[..., 1::2]
	# print("q_amp: \n", q_amp)
	# print("i_amp: \n", i_amp)
	# (symbol_num, sub-symbol num, sub_carrier_bit_num)
	q_bits = _amp_to_symbol(amp_array=q_amp, sub_carrier_bit_num=sub_carrier_bit_num)
	i_bits = _amp_to_symbol(amp_array=i_amp, sub_carrier_bit_num=sub_carrier_bit_num)
	# print("Unserial Q bits: \n", q_bits)
	# print("Unserial I bits: \n", i_bits)
	q_bits = np.reshape(q_bits, (*parallel_amp.shape[:-1], -1))
	i_bits = np.reshape(i_bits, (*parallel_amp.shape[:-1], -1))
	# print("Q bits: \n", q_bits)
	# print("I bits: \n", i_bits)
	symbol_bits = np.empty((*parallel_amp.shape[:-1], q_bits.shape[-1] * 2), dtype=float)
	# print(symbol_bits.shape)
	symbol_bits[..., ::2] = q_bits
	symbol_bits[..., 1::2] = i_bits
	return symbol_bits


def serial_to_parallel(serial_bits: np.ndarray, sub_carrier_num: int, sub_carrier_bit_num: int):
	r"""
	The bits are organized as [symbol of sub-carrier 1, symbol of sub-carrier 2, ...]

	- serial_bits: (... , symbol_bits_num);
		E.g. (symbols_num , symbol_bits_num), (group_num, symbols_num , symbol_bits_num)

	Each symbol of a pair of sub-carrier f1 is organized as:
		[Q_f1_bit1, I_f1_bit1, Q_f1_bit2, I_f1_bit2, ...]
	Take 4-QAM as an example:
		[Q_f1_bit1, I_f1_bit1]
	16-QAM:
		[Q_f1_bit1, I_f1_bit1, Q_f1_bit2, I_f1_bit2]

	The bits of a sub-carrier is corresponding to the amplitude,
	Take 4-QAM as an example:
		0: +1; 1: -1;
	16-QAM:
		00: +1/2; 01: +1; 10: -1/2; 11: -1;

	Args:
		serial_bits (np.ndarray): The bits to be modulated, and the bits num is sub_carrier_num * sub_carrier_bit_num * 2.
			shape: (symbol_num, symbol_bits_num)
		sub_carrier_num (int): Frequency number of sub-carriers.
		sub_carrier_bit_num (int): The amplitude of a sub-carrier is divided into (2**sub_carrier_bit_num), for example,
			4-QAM: 1-bit amplitude; 16-QAM: 2-bit amplitude

	Return:
		The amplitude of sub-carriers:
			[[sin(f1), cos(f1), sin(f2), cos(f2), ... ], ...]
			shape: [symbol_num, sub_carrier_num * 2]
	"""
	assert serial_bits.shape[-1] == sub_carrier_num * sub_carrier_bit_num * 2
	# symbol_num = serial_bits.shape[0]
	q_bits = serial_bits[..., ::2]
	i_bits = serial_bits[..., 1::2]
	q_amp = np.reshape(q_bits, (*serial_bits.shape[:-1], sub_carrier_num, sub_carrier_bit_num))
	i_amp = np.reshape(i_bits, (*serial_bits.shape[:-1], sub_carrier_num, sub_carrier_bit_num))
	q_amp = _symbol_to_amp(q_amp)
	i_amp = _symbol_to_amp(i_amp)
	parallel_amp = np.empty((*serial_bits.shape[:-1], sub_carrier_num * 2), dtype=float)
	parallel_amp[..., ::2] = q_amp
	parallel_amp[..., 1::2] = i_amp
	return parallel_amp


if __name__ == "__main__":
	# my_bits = np.array(
	# 	[
	# 		[0, 0, 0],
	# 		[0, 0, 1],
	# 		[0, 1, 0],
	# 		[0, 1, 1],
	# 		[1, 0, 0],
	# 		[1, 1, 1],
	# 	]
	# )
	# amps = _symbol_to_amp(my_bits)
	# print(amps)
	#
	# src_bits = np.random.randint(0, 2, size=60)
	# print(src_bits)
	# serial_to_parallel(serial_bits=src_bits, sub_carrier_num=15, sub_carrier_bit_num=2,)

	# amp_list = np.array([[-1, -0.75, 0.5, 0.25], [-0.25, 1, -1, 0.25]])
	# amp_list = amp_list[..., np.newaxis]
	# symbol_bits = _amp_to_symbol(amp_array=amp_list, sub_carrier_bit_num=3)
	# print(symbol_bits)

	sub_carrier_num = 2
	sub_carrier_bit_num = 3
	bit_list = np.random.randint(0, 2, size=(5, sub_carrier_bit_num * sub_carrier_num * 2))
	parallel_amps = serial_to_parallel(serial_bits=bit_list, sub_carrier_bit_num=sub_carrier_bit_num, sub_carrier_num=sub_carrier_num)
	decode_bits = parallel_to_serial(parallel_amp=parallel_amps, sub_carrier_bit_num=sub_carrier_bit_num)
	print(bit_list)
	print(decode_bits)

	# amp_list = np.array([[[-1.2, -0.6, 0.4, 0.2], [-0.6, 0.9, -1.2, 0.4]], [[-1.2, -0.6, 0.4, 0.2], [-0.6, 0.9, -1.2, 0.4]]])
	# serial_bits = parallel_to_serial(parallel_amp=amp_list, sub_carrier_bit_num=1)
	# re_amp_list = serial_to_parallel(serial_bits=serial_bits, sub_carrier_bit_num=1, sub_carrier_num=2)
	# print(serial_bits)
	# print(re_amp_list)
