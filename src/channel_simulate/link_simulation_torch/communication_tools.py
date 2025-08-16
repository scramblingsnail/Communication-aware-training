import torch


def _symbol_to_amp(symbol_bits: torch.Tensor):
	bits_num = symbol_bits.size()[-1]
	sgn = 1 - symbol_bits[..., :1] * 2
	if bits_num == 1:
		return sgn.squeeze(-1)

	bin_factors = torch.pow(2, torch.arange(bits_num - 2, -0.5, -1, device=symbol_bits.device)).unsqueeze(-1)
	amp_idx = torch.matmul(symbol_bits[..., 1:], bin_factors)
	amp_idx = amp_idx.add(1).div(2**(bits_num - 1))
	amp_array = amp_idx * sgn
	return amp_array.squeeze(-1)


def serial_to_parallel(
	serial_bits: torch.Tensor,
	sub_carrier_num: int,
	sub_carrier_bit_num: int,
):
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
		serial_bits (torch.Tensor): The bits to be modulated, and the bits num is sub_carrier_num * sub_carrier_bit_num * 2.
			shape: (symbol_num, symbol_bits_num)
		sub_carrier_num (int): Frequency number of sub-carriers.
		sub_carrier_bit_num (int): The amplitude of a sub-carrier is divided into (2**sub_carrier_bit_num), for example,
			4-QAM: 1-bit amplitude; 16-QAM: 2-bit amplitude

	Return:
		The amplitude of sub-carriers:
			[[sin(f1), cos(f1), sin(f2), cos(f2), ... ], ...]
			shape: [symbol_num, sub_carrier_num * 2]
	"""
	assert serial_bits.size()[-1] == sub_carrier_num * sub_carrier_bit_num * 2
	q_bits = serial_bits[..., ::2]
	i_bits = serial_bits[..., 1::2]
	raw_size = serial_bits.size()
	new_size = raw_size[:-1] + torch.Size((sub_carrier_num, sub_carrier_bit_num))
	q_amp = torch.reshape(q_bits, new_size)
	i_amp = torch.reshape(i_bits, new_size)
	q_amp = _symbol_to_amp(q_amp)
	i_amp = _symbol_to_amp(i_amp)
	amp_size = raw_size[:-1] + torch.Size((sub_carrier_num * 2, ))
	parallel_amp = torch.empty(amp_size, dtype=torch.float32, device=serial_bits.device)
	parallel_amp[..., ::2] = q_amp
	parallel_amp[..., 1::2] = i_amp
	return parallel_amp


def _amp_to_symbol(amp_array: torch.Tensor, sub_carrier_bit_num: int):
	sgn_array = torch.sign(amp_array)
	sgn_bits = (1 - sgn_array) / 2
	if sub_carrier_bit_num == 1:
		return sgn_bits

	max_int = 2**(sub_carrier_bit_num - 1) - 1
	amp_interval = 1 / 2**(sub_carrier_bit_num - 1)
	int_array = torch.round(amp_array.abs() / amp_interval) - 1
	int_array = torch.clip(int_array, 0, max_int)

	mask = torch.arange(sub_carrier_bit_num - 2, -0.5, -1, device=amp_array.device) + 1
	mask_size = int_array.size() + mask.size()
	mask = torch.broadcast_to(mask, mask_size)
	int_array = torch.broadcast_to(int_array[..., None], mask_size)
	int_bits = torch.bitwise_and(int_array.long(), mask.long())
	int_bits = torch.not_equal(int_bits, 0).float()
	all_bits = torch.cat([sgn_bits[..., None], int_bits], dim=-1)
	return all_bits


def parallel_to_serial(
	parallel_amp: torch.Tensor,
	sub_carrier_bit_num: int,
):
	r"""
	Transform the amplitudes to bits

	- parallel_amp: (..., sub_carrier_num * 2)
		E.g. (symbols_num, sub_carrier_num * 2), (group_num, symbols_num, sub_carrier_num * 2)

	Args:
		parallel_amp (torch.Tensor): (..., sub_carrier_num * 2); The amplitude of all sub-carriers.
			[[sin(f1), cos(f1), sin(f2), cos(f2), ... ], ...]
		sub_carrier_bit_num (int): The amplitude of a sub-carrier is divided into (2**sub_carrier_bit_num)

	Return:
		The serialized bits: (symbols_num, symbol_bits_num);
			[[Q_f1_bit1, I_f1_bit1, Q_f1_bit2, I_f1_bit2, ...], ...]
			symbol_bits_num: sub-carrier_num * sub_carrier_bit_num * 2
	"""
	q_amp = parallel_amp[..., ::2]
	i_amp = parallel_amp[..., 1::2]

	q_bits = _amp_to_symbol(amp_array=q_amp, sub_carrier_bit_num=sub_carrier_bit_num)
	i_bits = _amp_to_symbol(amp_array=i_amp, sub_carrier_bit_num=sub_carrier_bit_num)
	new_size = parallel_amp.size()[:-1] + torch.Size((-1, ))
	q_bits = torch.reshape(q_bits, new_size)
	i_bits = torch.reshape(i_bits, new_size)
	bits_size = parallel_amp.size()[:-1] + torch.Size((q_bits.size()[-1] * 2, ))
	symbol_bits = torch.empty(bits_size, dtype=torch.float32, device=parallel_amp.device)
	symbol_bits[..., ::2] = q_bits
	symbol_bits[..., 1::2] = i_bits
	return symbol_bits


if __name__ == "__main__":
	# amp_list = torch.tensor([[-1, -0.75, 0.5, 0.25], [-0.25, 1, -1, 0.25]])
	# amp_list = amp_list[..., None]
	# symbol_bits = _amp_to_symbol(amp_array=amp_list, sub_carrier_bit_num=3)
	# print(symbol_bits)

	sub_carrier_num = 2
	sub_carrier_bit_num = 3
	bit_list = torch.randint(0, 2, size=(5, sub_carrier_bit_num * sub_carrier_num * 2), dtype=torch.float)
	parallel_amps = serial_to_parallel(serial_bits=bit_list, sub_carrier_bit_num=sub_carrier_bit_num, sub_carrier_num=sub_carrier_num)
	print(parallel_amps.size())
	decode_bits = parallel_to_serial(parallel_amp=parallel_amps, sub_carrier_bit_num=sub_carrier_bit_num)
	print(bit_list)
	print(decode_bits)


	# amp_list = torch.tensor([[-1.2, -0.6, 0.4, 0.2], [-0.6, 0.9, -1.2, 0.4], [-0.5, 0.6, 0.4, -0.2], [0.6, -0.9, 1.2, 0.4]])
	# serial_bits = parallel_to_serial(parallel_amp=amp_list, sub_carrier_bit_num=2)
	# re_amp_list = serial_to_parallel(serial_bits=serial_bits, sub_carrier_bit_num=2, sub_carrier_num=2)
	# print(serial_bits)
	# print(re_amp_list)
