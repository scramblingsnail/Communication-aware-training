import os.path

import torch
import numpy as np
from abc import abstractmethod
from src.channel_simulate.link_simulation_torch.communication_tools import serial_to_parallel, parallel_to_serial
from src.channel_simulate.channel_model.fwgn_np import prepare_fadings


class ChannelSimulator:
	@abstractmethod
	def forward(self, batch_bits: torch.Tensor):
		r""" Transmit the bits over wireless channel and return the received bits. """


class ClarkeSimulator(ChannelSimulator):
	def __init__(
		self,
		training_symbol_interval: int,
		avg_snr: float,
		channel_fading_dir: str,
		symbol_period: float,
		samp_per_symbol: int,
		cyclic_prefix_num: int,
		aimc_bit: float,
		sub_carrier_num: int,
		modulation_order: int,
		multi_path_num: int = 4,
		min_path_delay: int = 0,
		max_path_delay: int = 5,
		min_path_power: float = -20,
		max_path_power: float = -5,
		path_delay_list: torch.Tensor = None,
		path_power_list: torch.Tensor = None,
		flat_mode: bool = True,
		train_symbol_snr: float = None,
	):
		r"""
		Args:
			training_symbol_interval (int): denote it as N, -> every N data symbols transmit a training symbol.
			avg_snr (float): average Power_signal / Power_noise; unit: dB.
			symbol_period (float): The duration of a symbol; unit: ms
			cyclic_prefix_num (int): cyclic prefix num
			aimc_bit (float): The precision of AIMC chips.
			sub_carrier_num (int): The number of sub-carriers that carrying valid data. E.g. 14
			modulation_order (int): The bit num of a sub-carrier. E.g. 1 for 4-QAM; 2 for 16-QAM; 3 for 64-QAM
			multi_path_num (int): The number of different paths when considering frequency-selective channels.
			max_path_delay (int): The maximum delay of paths. unit: T_symbol / samp_per_symbol
			path_delay_list (list): Specified path delays -- size (multi_path_num, ).
			path_power_list (list): Specified path powers -- size (multi_path_num, ).
		"""
		self.group_symbols = training_symbol_interval
		self.avg_snr = avg_snr
		self.fading_buffer = self._load_fadings(channel_fading_dir)
		self.fs = 1e3 / symbol_period
		self.samp_per_symbol = samp_per_symbol
		self.cyclic_prefix_num = cyclic_prefix_num
		self.aimc_bit = aimc_bit
		self.chip_std = 1 / ((2 ** self.aimc_bit - 1) * 4)
		self.freq_num = sub_carrier_num
		self.order = modulation_order
		self.multi_path_num = multi_path_num
		assert max_path_delay > min_path_delay
		self.min_path_delay = min_path_delay
		self.max_path_delay = max_path_delay
		self.min_path_power = min_path_power
		self.max_path_power = max_path_power
		self.path_delay_list = path_delay_list
		self.path_power_list = path_power_list
		self.flat_mode = flat_mode
		self.train_symbol_snr = train_symbol_snr
		if path_delay_list is not None:
			assert path_delay_list.size()[0] == path_power_list.size()[0] == multi_path_num
		assert multi_path_num > 1
		self.symbol_valid_bits = self.freq_num * self.order * 2
		self.modulation_matrix = self._modulation_matrix()
		self.demodulation_matrix = self._demodulation_matrix()
		self.training_symbol, self.training_symbol_amps = self._training_symbol()

	def _reset_aimc_chip_param(self, aimc_bit: float):
		self.aimc_bit = aimc_bit
		self.chip_std = 1 / ((2 ** self.aimc_bit - 1) * 4)

	def _reset_wireless_settings(self, order):
		self.order = order
		self.symbol_valid_bits = self.freq_num * self.order * 2
		self.training_symbol, self.training_symbol_amps = self._training_symbol()

	def set_params(self, p_name: str, p):
		if p_name == "avg_snr":
			self.avg_snr = p
		elif p_name == "aimc_bit":
			self._reset_aimc_chip_param(aimc_bit=p)
		elif p_name == "order":
			self._reset_wireless_settings(p)
		else:
			raise KeyError("Invalid param.")

	def _training_symbol(self):
		symbol = torch.randint(0, 2, size=(self.symbol_valid_bits, ))
		symbol_amps = serial_to_parallel(
			serial_bits=symbol.float(),
			sub_carrier_num=self.freq_num,
			sub_carrier_bit_num=self.order,
		)
		return symbol, symbol_amps

	def _amps_to_complex(self, amps: torch.Tensor):
		r"""
		Args:
			amps (torch.Tensor): The last dim: Q1, I1, Q2, I2, ..., Qn, In.

		"""
		return amps[..., 1::2] + 1j * amps[..., ::2]

	def _complex_to_amps(self, complex_tensor: torch.Tensor):
		sub_carrier_num = complex_tensor.size()[-1]
		amps = torch.empty(complex_tensor.size()[:-1] + torch.Size((sub_carrier_num * 2, )), dtype=torch.float, device=complex_tensor.device)
		amps[..., 1::2] = complex_tensor.real
		amps[..., ::2] = complex_tensor.imag
		return amps

	def _load_fadings(self, data_dir: str) -> torch.Tensor:
		i_fading_path = rf"{data_dir}/fading_i.npy"
		q_fading_path = rf"{data_dir}/fading_q.npy"

		i_fading = np.load(i_fading_path)
		q_fading = np.load(q_fading_path)
		fading = torch.from_numpy(i_fading).float() + 1j * torch.from_numpy(q_fading).float()
		return fading

	def sample_fading(self, fading_num: int = 1):
		r"""
		Sample fading_num fading sequence from the fading buffer.

		Return:
			(fading_num, image_symbols_num)
		"""
		num = self.fading_buffer.size()[0]
		indices = torch.multinomial(torch.arange(0, num, 1.), num_samples=fading_num, replacement=True).long()
		return self.fading_buffer[indices]

	def _modulation_matrix(self):
		matrix = np.zeros((self.freq_num * 2, self.samp_per_symbol))
		# 1 -> freq_num - 2; freq_num -> freq_num + 1
		freq_list = list(range(1, self.freq_num - 1)) + list(range(self.freq_num, self.freq_num + 2))
		for idx, each_freq in enumerate(freq_list):
			for each_t in range(1, self.samp_per_symbol + 1):
				sin = np.sin(each_freq * each_t * np.pi * 2 / self.samp_per_symbol)
				cos = np.cos(each_freq * each_t * np.pi * 2 / self.samp_per_symbol)
				matrix[2 * idx][each_t - 1] = sin
				matrix[2 * idx + 1][each_t - 1] = cos
		matrix = torch.from_numpy(matrix).float()
		return matrix

	def _demodulation_matrix(self):
		matrix = np.zeros((self.samp_per_symbol, self.freq_num * 2))
		freq_list = list(range(1, self.freq_num - 1)) + list(range(self.freq_num, self.freq_num + 2))
		for idx, each_freq in enumerate(freq_list):
			for each_t in range(1, self.samp_per_symbol + 1):
				sin = np.sin(each_freq * each_t * np.pi * 2 / self.samp_per_symbol)
				cos = np.cos(each_freq * each_t * np.pi * 2 / self.samp_per_symbol)
				matrix[each_t - 1][2 * idx] = sin
				matrix[each_t - 1][2 * idx + 1] = cos
		matrix = torch.from_numpy(matrix).float()
		return matrix

	def aimc_modulate(
		self,
		sub_carrier_amps: torch.Tensor,
	):
		r"""

		Args:
			sub_carrier_amps (torch.Tensor): (path_num, batch_size, group_num, group_symbols, sub_carrier_num * 2)
				The first symbol in a group is the train symbol

		Return:
			wave (torch.Tensor): (path_num, batch_size, group_num, group_symbols, samp_per_symbol + cyclic_prefix_num)
		"""
		if self.modulation_matrix.device != sub_carrier_amps.device:
			self.modulation_matrix = self.modulation_matrix.to(sub_carrier_amps.device)

		chip_noise = torch.randn(self.modulation_matrix.size(), device=self.modulation_matrix.device).mul(self.chip_std)
		noisy_matrix = self.modulation_matrix + chip_noise

		# modulate training symbol with no noise
		# (path_num, batch_size, group_num, 1, samp_per_symbol)
		train_wave = torch.matmul(sub_carrier_amps[..., :1, :], self.modulation_matrix)
		# (path_num, batch_size, group_num, group_symbols, samp_per_symbol)
		data_wave = torch.matmul(sub_carrier_amps[..., 1:, :], noisy_matrix)
		wave = torch.cat([train_wave, data_wave], dim=-2)
		cyclic_prefix = wave[..., -self.cyclic_prefix_num:]
		wave = torch.cat([cyclic_prefix, wave], dim=-1)
		return wave

	def ideal_channel(self, batch_i_signals: torch.Tensor, batch_q_signals: torch.Tensor):
		return batch_i_signals + batch_q_signals

	def flat_fading_channel(self, batch_signals: torch.Tensor):
		r"""
		Flat fading.

		Args:
			batch_signals (torch.Tensor): (path_num, batch_size, group_num, group_symbols, cyclic_prefix_num + samp_per_symbol)
				The signals modulated by the OFDM modulator. The first symbol in a group is the training symbol.

		Return:
			batch_noisy_signals (torch.Tensor): (batch_size, group_num, group_symbols, samp_per_symbol + cyclic_prefix_num)
		"""
		batch_signals = batch_signals.squeeze(0)

		# Add Gaussian noise
		# (batch_size, ) calculate avg power for each image.
		signal_power = batch_signals.square().mean(dim=list(range(1, len(batch_signals.size()))))
		noise_power = signal_power.div(10**(self.avg_snr / 10))
		noise_std = torch.sqrt(noise_power)[..., None, None, None].broadcast_to(batch_signals.size())
		noise = torch.randn(batch_signals.size(), device=batch_signals.device) * noise_std

		if self.train_symbol_snr is None:
			# TODO Training symbols no noise:
			noise[:, :, 0, :] = 0
		else:
			train_noise_power = signal_power.div(10**(self.train_symbol_snr / 10))
			train_noise_size = noise[:, :, 0, :].size()
			train_noise_std = torch.sqrt(train_noise_power)[..., None, None].broadcast_to(train_noise_size)
			train_noise = torch.randn(train_noise_size, device=batch_signals.device) * train_noise_std
			noise[:, :, 0, :] = train_noise

		batch_noisy_signals = batch_signals + noise
		return batch_noisy_signals

	def freq_selective_channel(self, batch_signals: torch.Tensor):
		r"""
		Frequency-selective fading.

		Args:
			batch_signals (torch.Tensor): (multi_path_num, batch_size, group_num, group_symbols, samp_per_symbol)
				The in-phase signals modulated by the OFDM modulator. The first symbol in a group is the training symbol.

		Return:
			batch_noisy_signals (torch.Tensor): (batch_size, group_num, 1 + group_symbols, samp_per_symbol)
		"""
		# Prepare path characteristics.
		assert batch_signals.size()[0] == self.multi_path_num
		batch_size = batch_signals.size()[1]
		samp_per_symbol = batch_signals.size()[-1]
		group_num = batch_signals.size()[2]
		group_symbols = batch_signals.size()[-2]
		samp_period_num = group_num * group_symbols

		if self.path_delay_list is None:
			path_delay_list = torch.randint(
				low=self.min_path_delay, high=self.max_path_delay, size=(self.multi_path_num, ), device=batch_signals.device,
			).long()
			path_delay_list[0] = 0
		else:
			if self.path_delay_list.device != batch_signals.device:
				self.path_delay_list = self.path_delay_list.to(batch_signals.device)
			path_delay_list = self.path_delay_list.long()

		# (multi_path_num, ); unit: symbol period.
		if self.path_power_list is None:
			path_power_list = torch.rand((self.multi_path_num, ), device=batch_signals.device)
			path_power_list = path_power_list * (self.max_path_power - self.min_path_power) + self.min_path_power
			path_power_list[0] = 0
		else:
			if self.path_power_list.device != batch_signals.device:
				self.path_power_list = self.path_power_list.to(batch_signals.device)
			path_power_list = self.path_power_list

		path_power_list = torch.pow(10, path_power_list / 10)
		# Total power is 1.
		path_power_list = path_power_list.div(path_power_list.sum())

		# (multi_path_num, )
		path_scale_list = path_power_list.sqrt()

		# (multi_path_num, batch_size, -1)
		flatten_signals = torch.reshape(batch_signals, (self.multi_path_num, batch_size, -1))

		# FIR: S1(t - delay1) * scale1 * fading_1 + S2(t - delay2) * scale2 * fading_2 + ...
		# (batch_size, -1)
		sum_signals = torch.zeros(flatten_signals.size()[1:], device=batch_signals.device)

		for path_idx in range(self.multi_path_num):
			delay_samp_num = path_delay_list[path_idx]
			# print("delay num: ", delay_samp_num)
			if delay_samp_num == 0:
				path_signal = flatten_signals[path_idx]
			else:
				# (batch_size, -1)
				path_signal = flatten_signals[path_idx]
				path_signal = torch.cat(
					[
						torch.zeros((batch_size, delay_samp_num), device=batch_signals.device),
						path_signal[:,  : -delay_samp_num],
					],
					dim=-1,
				)

			path_scale = path_scale_list[path_idx]
			sum_signals += path_signal * path_scale

			# print(f"Path {path_idx} -- delay: {delay} power: {torch.pow(path_scale, 2)}")

		# Add Gaussian noise
		# (batch_size, ) calculate avg power for each image.
		signal_power = sum_signals.square().mean(dim=list(range(1, len(sum_signals.size()))))
		noise_power = signal_power.div(10**(self.avg_snr / 10))
		noise_std = torch.sqrt(noise_power)[..., None].broadcast_to(sum_signals.size())
		noise = torch.randn(sum_signals.size(), device=sum_signals.device) * noise_std
		# (batch_size, group_num, group_symbols, samp_per_symbol)
		noise = noise.reshape(batch_signals.size()[1:])

		if self.train_symbol_snr is None:
			# TODO train symbols: no noise
			noise[:, :, 0, :] = 0
		else:
			train_noise_power = signal_power.div(10**(self.train_symbol_snr / 10))
			train_noise_size = noise[:, :, 0, :].size()
			train_noise_std = torch.sqrt(train_noise_power)[..., None, None].broadcast_to(train_noise_size)
			train_noise = torch.randn(train_noise_size, device=batch_signals.device) * train_noise_std
			noise[:, :, 0, :] = train_noise

		batch_noisy_signals = sum_signals.reshape(batch_signals.size()[1:]) + noise
		return batch_noisy_signals

	def aimc_demodulate(
		self,
		batch_signals: torch.Tensor,
	):
		r"""
		# TODO: 加入基于导频的信道估计

		Args:
			batch_signals (torch.Tensor): (batch_size, group_num, 1 + group_symbols, cyclic_prefix_num + samp_per_symbol)

		Return:
			demodulate_results (torch.Tensor): (batch_size, group_num, 1 + group_symbols, symbol_valid_bits)
		"""
		if self.demodulation_matrix.device != batch_signals.device:
			self.demodulation_matrix = self.demodulation_matrix.to(batch_signals.device)

		chip_noise = torch.randn(self.demodulation_matrix.size(), device=self.demodulation_matrix.device).mul(self.chip_std)
		noisy_matrix = self.demodulation_matrix + chip_noise

		# TODO: train symbol demodulate with no noise
		train_results = torch.matmul(batch_signals[:, :, :1, self.cyclic_prefix_num:], self.demodulation_matrix)
		data_results = torch.matmul(batch_signals[:, :, 1:, self.cyclic_prefix_num:], noisy_matrix)
		demodulate_results = torch.cat([train_results, data_results], dim=-2)
		demodulate_results = demodulate_results.div(self.samp_per_symbol / 2)
		return demodulate_results

	def channel_estimation_equalization(
		self,
		batch_results: torch.Tensor
	):
		r"""
		Args:
			batch_results (torch.Tensor): (batch_size, group_num, 1 + group_symbols, sub-carrier_num * 2)

		Return:
			batch_receive_bits (torch.Tensor): (batch_size, group_num, group_symbols, symbol_valid_bits)
		"""
		batch_training_amps = batch_results[:, :, :1, :]
		batch_data_amps = batch_results[:, :, 1:, :]
		# ideal_amps * fading = training_amps
		if self.training_symbol_amps.device != batch_results.device:
			self.training_symbol_amps = self.training_symbol_amps.to(batch_results.device)
		ideal_amps = self.training_symbol_amps[None, None, None, :].broadcast_to(batch_training_amps.size())

		# (batch_size, group_num, 1, sub-carrier_num)
		batch_training_complex = self._amps_to_complex(amps=batch_training_amps)
		ideal_training_complex = self._amps_to_complex(amps=ideal_amps)

		# (batch_size, group_num, 1, sub-carrier_num)
		calculate_fading = batch_training_complex / ideal_training_complex
		# print("Calculate fading: ", calculate_fading[0, 0, 0, :])

		# equalization:
		batch_data_complex = self._amps_to_complex(amps=batch_data_amps)
		batch_data_complex = batch_data_complex / calculate_fading
		batch_data_amps = self._complex_to_amps(complex_tensor=batch_data_complex)

		batch_receive_bits = parallel_to_serial(
			parallel_amp=batch_data_amps,
			sub_carrier_bit_num=self.order,
		)
		return batch_receive_bits

	def channel_equalization(
		self,
		batch_results: torch.Tensor,
		training_fading: torch.Tensor,
	):
		r"""
		Args:
			batch_results (torch.Tensor): (batch_size, group_num, group_symbols, symbol_valid_bits)
			training_fading (torch.Tensor): (batch_size, group_num, 1)

		"""
		i_fading = training_fading.real.unsqueeze(-1)
		q_fading = training_fading.imag.unsqueeze(-1)

		# Q component
		batch_results[..., ::2] = batch_results[..., ::2] / q_fading
		# I component
		batch_results[..., 1::2] = batch_results[..., 1::2] / i_fading
		return batch_results

	def _inputs_to_symbols(self, inputs: torch.Tensor):
		r"""
		Transform the input bits to the amplitudes of the transmitting sub-carriers.

		Args:
			inputs (torch.Tensor): size -- (batch_size, C, H, W, Bit-width)

		Return:
			sending_symbols: size -- (batch_size, group_num, 1 + group_symbols, symbol_bit_num)
		"""
		batch_size = inputs.size()[0]
		flatten = inputs.reshape((batch_size, -1))
		bits_num = flatten.size()[-1]
		symbols_num = int(np.ceil(bits_num / self.symbol_valid_bits))
		group_num = int(np.ceil(symbols_num / self.group_symbols))
		padded_symbols_num = group_num * self.group_symbols
		padded_bits_num = padded_symbols_num * self.symbol_valid_bits
		padded_bits = torch.cat(
			[
				flatten,
				torch.zeros((batch_size, padded_bits_num - bits_num), device=flatten.device),
			],
			dim=-1,
		)
		symbols = torch.reshape(padded_bits, (batch_size, group_num, self.group_symbols, self.symbol_valid_bits))

		# add training symbols
		if self.training_symbol.device != symbols.device:
			self.training_symbol = self.training_symbol.to(symbols.device)
		extra_symbols = torch.broadcast_to(
			self.training_symbol[None, None, None, :],
			size=(batch_size, group_num, 1, self.symbol_valid_bits),
		)
		sending_symbols = torch.cat((extra_symbols, symbols), dim=-2)
		return sending_symbols

	def _reshape(self, inputs: torch.Tensor, batch_fading: torch.Tensor):
		r"""
		# TODO: 每个Group加入导频

		Args:
			inputs (torch.Tensor): size -- (batch_size, C, H, W, Bit-width)
			batch_fading (torch.Tensor): size -- (image_symbols_num, )

		Return:
			parallel_amps: size -- (batch_size, group_num, group_symbols, sub_carrier_num * 2)
			training_fading: size -- (batch_size, group_num, 1)
			data_fading: size -- (batch_size, group_num, group_symbols)
		"""
		batch_size = inputs.size()[0]
		flatten = inputs.reshape((batch_size, -1))
		bits_num = flatten.size()[-1]
		symbols_num = int(np.ceil(bits_num / self.symbol_valid_bits))
		group_num = int(np.ceil(symbols_num / self.group_symbols))
		padded_symbols_num = group_num * self.group_symbols
		padded_bits_num = padded_symbols_num * self.symbol_valid_bits
		padded_bits = torch.cat(
			[
				flatten,
				torch.zeros((batch_size, padded_bits_num - bits_num), device=flatten.device),
			],
			dim=-1,
		)
		symbols = torch.reshape(padded_bits, (batch_size, group_num, self.group_symbols, self.symbol_valid_bits))

		parallel_amps = serial_to_parallel(
			serial_bits=symbols,
			sub_carrier_num=self.freq_num,
			sub_carrier_bit_num=self.order,
		)

		samp_num = group_num * (1 + self.group_symbols)
		if batch_fading.size()[0] < samp_num:
			raise ValueError(
				f"The length of fading factors must be longer than the data bits;"
				f"Expect at least {samp_num}; but {batch_fading.size()[0]} given."
			)
		batch_fading = torch.reshape(batch_fading[:samp_num], (group_num, self.group_symbols + 1))
		training_fading = batch_fading[:, :1].unsqueeze(0)
		training_fading = torch.broadcast_to(training_fading, torch.Size((batch_size, )) + training_fading.size()[1:])
		data_fading = batch_fading[:, 1:].unsqueeze(0)
		data_fading = torch.broadcast_to(data_fading, torch.Size((batch_size, )) + data_fading.size()[1:])
		return parallel_amps, training_fading, data_fading

	def _flat_fade_amps(self, sub_carrier_amps: torch.Tensor):
		r"""
		Args:
			sub_carrier_amps (torch.Tensor): (batch_size, group_num, group_symbols, sub_carrier_num * 2)
				The first symbol in a group is the train symbol.

		Return:
			fade_sub_carrier_amps (fade_path_num, batch_size, group_num, group_symbols, sub_carrier_num * 2):
		"""
		group_num = sub_carrier_amps.size()[1]
		group_symbols = sub_carrier_amps.size()[2]
		samp_period_num = group_num * group_symbols

		complex_amps = self._amps_to_complex(amps=sub_carrier_amps)

		# (fading_num, image_symbols_num)
		fading_t_seq = self.sample_fading().to(sub_carrier_amps.device)

		# TODO: Debug
		# constant_fading = torch.randn((1, )) + 1j * torch.randn((1, ))
		# print("Constant fading: ", constant_fading)
		# fading_t_seq = constant_fading[None, :].broadcast_to((1, samp_period_num)).to(sub_carrier_amps.device)

		if fading_t_seq.size()[1] < samp_period_num:
			raise ValueError(
				f"The length of fading factors must be longer than the data bits;"
				f"Expect at least {samp_period_num}; but {fading_t_seq.size()[1]} given."
			)

		fading_t_seq = fading_t_seq[..., :samp_period_num].reshape((1, group_num, group_symbols))
		fading_t_seq = fading_t_seq[..., None].broadcast_to(complex_amps.size())

		fade_complex_amps = complex_amps * fading_t_seq
		# (batch_size, group_num, 1 + group_symbols, sub_carrier_num * 2)
		fade_sub_carrier_amps = self._complex_to_amps(fade_complex_amps)
		# (fade_path_num, batch_size, group_num, 1 + group_symbols, sub_carrier_num * 2)
		fade_sub_carrier_amps = fade_sub_carrier_amps.unsqueeze(0)
		return fade_sub_carrier_amps

	def _freq_selective_fade_amps(self, sub_carrier_amps: torch.Tensor):
		r"""
		Args:
			sub_carrier_amps (torch.Tensor): (batch_size, group_num, group_symbols, sub_carrier_num * 2)
				The first symbol in a group is a training symbol.

		Return:
			fade_sub_carrier_amps (fade_path_num, batch_size, group_num, group_symbols, sub_carrier_num * 2):
		"""
		# Prepare path characteristics.
		group_num = sub_carrier_amps.size()[1]
		group_symbols = sub_carrier_amps.size()[-2]
		samp_period_num = group_num * group_symbols

		amp_size = sub_carrier_amps.size()
		multi_path_amps = sub_carrier_amps.unsqueeze(0).broadcast_to(
			torch.Size((self.multi_path_num, )) + amp_size
		)
		# (fade_path_num, batch_size, group_num, group_symbols, sub_carrier_num)
		complex_amps = self._amps_to_complex(multi_path_amps)

		# Multi-path fading
		# (fading_num, image_symbols_num)
		fading_t_seq = self.sample_fading(self.multi_path_num).to(sub_carrier_amps.device)
		# (fade_path_num, group_num, group_symbols)
		fading_t_seq = fading_t_seq[..., :samp_period_num].reshape((self.multi_path_num, group_num, group_symbols))
		fading_t_seq = fading_t_seq[:, None, :, :, None].broadcast_to(complex_amps.size())
		fade_complex_amps = complex_amps * fading_t_seq
		fade_sub_carrier_amps = self._complex_to_amps(fade_complex_amps)
		return fade_sub_carrier_amps

	def forward(self, batch_bits: torch.Tensor, plotting_mode: bool = False):
		r"""
		Args:
			batch_bits (torch.Tensor): size -- (batch_size, C, H, W, Bit-width)
			plotting_mode (bool): If true, return the sending data symbols and batch_receive bits
		"""
		data_size = batch_bits.size()
		each_data_num = torch.prod(torch.tensor(data_size[1:])).item()
		# (batch_size, group_num, 1 + group_symbols, symbol_valid_bits)
		sending_symbols = self._inputs_to_symbols(inputs=batch_bits)

		# Change to sub-carrier amplitude:
		parallel_amps = serial_to_parallel(
			serial_bits=sending_symbols,
			sub_carrier_num=self.freq_num,
			sub_carrier_bit_num=self.order,
		)

		if self.flat_mode:
			fade_amps = self._flat_fade_amps(sub_carrier_amps=parallel_amps)
		else:
			# TODO: Multi-path fading
			fade_amps = self._freq_selective_fade_amps(sub_carrier_amps=parallel_amps)

		# (fade_path_num, batch_size, group_num, group_symbols, samp_per_symbol)
		batch_transmit_signals = self.aimc_modulate(
			sub_carrier_amps=fade_amps,
		)

		if self.flat_mode:
			batch_receive_signals = self.flat_fading_channel(
				batch_signals=batch_transmit_signals,
			)
		else:
			batch_receive_signals = self.freq_selective_channel(
				batch_signals=batch_transmit_signals,
			)

		batch_results = self.aimc_demodulate(batch_receive_signals)
		# (batch_size, group_num, group_symbols, symbol_valid_bits)
		batch_receive_bits = self.channel_estimation_equalization(batch_results=batch_results)
		if plotting_mode:
			return sending_symbols[:, :, 1:, :], batch_receive_bits

		batch_receive_bits = batch_receive_bits.reshape((data_size[0], -1))
		batch_valid_bits = batch_receive_bits[:, :each_data_num]
		batch_valid_bits = batch_valid_bits.reshape(data_size)
		return batch_valid_bits


def simulate_ber_data(qam_order: int = 1, overwrite: bool = False):
	import matplotlib.pyplot as plt
	from pathlib import Path

	root = Path(__file__)
	for _ in range(4):
		root = root.parent

	data_dir = root / "data/channel"
	bers_path = data_dir / f"symbol_wise_ber_order{qam_order}.npy"
	snr_path = data_dir / f"snr_list_order{qam_order}.npy"
	if os.path.exists(bers_path) and os.path.exists(snr_path) and not overwrite:
		return

	ss = ClarkeSimulator(
		training_symbol_interval=32,
		aimc_bit=qam_order+1,
		channel_fading_dir=str(data_dir),
		avg_snr=10,
		modulation_order=qam_order,
		samp_per_symbol=32,
		cyclic_prefix_num=8,
		sub_carrier_num=14,
		symbol_period=2.5,
		multi_path_num=2,
		min_path_delay=10,
		max_path_delay=30,
		flat_mode=True,
		path_delay_list=None,
		path_power_list=None,
		train_symbol_snr=50.,
	)

	snr_ber_list = []
	snr_list = list(range(0, 31, 1))
	avg_ber_list = []
	batch_size = 1000
	for snr in snr_list:
		ss.avg_snr = snr
		ber_list = []
		avg_ber = 0
		for idx in range(batch_size):
			input_image = torch.randint(0, 2, size=(1, 256, 56, 56, 1)).float()
			# (batch_size, group_num, group_symbols, symbol_valid_bits_num)
			send_bits, receive_bits = ss.forward(input_image, plotting_mode=True)
			valid_symbol_bits = receive_bits.size()[-1]
			wrong = torch.not_equal(send_bits, receive_bits).long()
			wrong_num = torch.sum(wrong, dim=[0, 1, 2])
			symbols_num = torch.tensor(wrong.size()[:3], device=wrong.device)
			symbol_wise_bers = wrong_num / torch.prod(symbols_num)
			ber_list.append(symbol_wise_bers)

			each_avg_ber = symbol_wise_bers.sum() / valid_symbol_bits
			avg_ber += each_avg_ber / batch_size
		batch_bers = torch.stack(ber_list, dim=0).cpu().numpy()
		snr_ber_list.append(batch_bers)

		avg_ber_list.append(avg_ber.item())
		print(f"SNR: {snr} -- Avg BER: {avg_ber}")

	snr_bers = np.stack(snr_ber_list, axis=0)
	np.save(str(bers_path), snr_bers)

	snr_list = np.array(snr_list)
	np.save(str(snr_path), snr_list)

	plt.plot(snr_list, avg_ber_list)
	plt.semilogy()
	plt.savefig(str(root / fr"check_simulator_order{qam_order}.png"))



if __name__ == "__main__":
	simulate_ber_data(qam_order=3)
