import time
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft
from pathlib import Path
import torch

from src.channel_simulate.link_simulation_np.serial_parallel_np import serial_to_parallel, parallel_to_serial


root = Path(__file__)
for _ in range(4):
	root = root.parent


BLUE = np.array([19, 164, 192, 255]) / 255
FILL_BLUE = np.array([210, 233, 240, 255]) / 255
RED = np.array([234, 96, 142, 255]) / 255
FILL_RED = np.array([246, 208, 217, 255]) / 255
YELLOW = np.array([192, 164, 19, 255]) / 255



def plot_fwgn(fd_m: float, samp_ms: float, samp_num: int = int(3.6e4)):
	start = time.time()
	fs = 1e3 / samp_ms

	blue_color = np.array([19, 164, 192, 255]) / 255
	image_dir = root / r"images/fwgn"
	abs_dist_img_path = image_dir / r"abs_dist.svg"
	phase_dist_img_path = image_dir / r"phase_dist.svg"
	if not os.path.exists(str(image_dir)):
		os.makedirs(str(image_dir))

	all_h_abs = []
	all_h_angle = []
	all_h_db = []
	each_h_db = None
	each_doppler_coeff = None
	for idx in range(5):
		h_factors, n_fft, n_ifft, each_doppler_coeff = fwgn_model(fd_m=fd_m, fs=fs, samp_num=samp_num)
		end = time.time()
		print("time cost: ", end - start)
		h_abs = np.abs(h_factors)
		h_angle = np.angle(h_factors)
		each_power_mean = np.mean(np.power(h_abs, 2))
		print("each power: ", each_power_mean)
		each_h_db = 10 * np.log10(h_abs)
		print(each_h_db.shape)
		all_h_db.append(each_h_db)
		all_h_abs.append(h_abs)
		all_h_angle.append(h_angle)
	all_h_abs = np.concatenate(all_h_abs, axis=0)
	all_h_angle = np.concatenate(all_h_angle, axis=0)

	power_mean = np.mean(np.power(all_h_abs, 2))
	print("Power mean: ", power_mean)

	t_array = np.arange(samp_num) * samp_ms / 1000
	# plt.plot(each_doppler_coeff)
	# plt.show()

	abs_fig, abs_ax = plt.subplots(1, 1, figsize=(12, 6))
	abs_ax.hist(all_h_abs, bins=np.linspace(0, 4, 50), color=blue_color, density=True)
	abs_ax.tick_params(axis="both", which="major", direction="in")
	plt.savefig(str(abs_dist_img_path), dpi=300)

	phase_fig, phase_ax = plt.subplots(1, 1, figsize=(6, 6))
	phase_ax.hist(all_h_angle, bins=np.linspace(-3.2, 3.2, 50), color=blue_color, density=True)
	phase_ax.tick_params(axis="both", which="major", direction="in")
	plt.savefig(str(phase_dist_img_path), dpi=300)

	first_ch_samp_num = 116
	last_ch_samp_num = 115

	zoom_in_start_samp_1 = 0
	zoom_in_end_samp_1 = zoom_in_start_samp_1 + first_ch_samp_num

	print(f"Start samp t: {zoom_in_start_samp_1}; End samp t: {zoom_in_end_samp_1}")

	for wav_idx in range(5):
		zoom_in_path = image_dir / fr"fading_{wav_idx + 1}_zoom_in.svg"
		each_wav_hb = all_h_db[wav_idx]
		fading_fig, fading_ax = plt.subplots(1, 1, figsize=(12, 6))
		fading_path = image_dir / fr"fading_{wav_idx + 1}.svg"
		fading_ax.plot(t_array, each_wav_hb, linewidth=0.5, color=BLUE)
		fading_ax.tick_params(axis="both", which="major", direction="in")
		plt.savefig(str(fading_path), dpi=300)

		zoom_in_fig, zoom_in_ax = plt.subplots(2, 1, figsize=(12, 12))
		zoom_in_hb_1 = each_wav_hb[zoom_in_start_samp_1: zoom_in_end_samp_1]
		zoom_in_t_1 = t_array[zoom_in_start_samp_1: zoom_in_end_samp_1]
		zoom_in_ax[0].plot(zoom_in_t_1, zoom_in_hb_1, marker="o", markersize=3, linewidth=0.5)
		# zoom_in_ax.set_xticks(np.arange(zoom_in_start_samp, zoom_in_end_samp, 1e-2))
		zoom_in_ax[0].tick_params(axis="both", which="major", direction="in")
		max_hb_1, min_hb_1 = np.max(zoom_in_hb_1), np.min(zoom_in_hb_1)
		max_hb_1 += 1
		min_hb_1 -= 1
		zoom_in_ax[0].set_ylim(min_hb_1, max_hb_1)

		zoom_in_hb_2 = each_wav_hb[-last_ch_samp_num: ]
		zoom_in_t_2 = t_array[-last_ch_samp_num: ]
		zoom_in_ax[1].plot(zoom_in_t_2, zoom_in_hb_2, marker="o", markersize=3, linewidth=0.5)
		# zoom_in_ax.set_xticks(np.arange(zoom_in_start_samp, zoom_in_end_samp, 1e-2))
		zoom_in_ax[1].tick_params(axis="both", which="major", direction="in")
		max_hb_2, min_hb_2 = np.max(zoom_in_hb_2), np.min(zoom_in_hb_2)
		max_hb_2 += 1
		min_hb_2 -= 1
		zoom_in_ax[1].set_ylim(min_hb_2, max_hb_2)

		plt.savefig(str(zoom_in_path), dpi=300)


def nextpow2_int(num: int):
	factor = int(np.ceil(np.log2(num)))
	return np.power(2, factor)


def fwgn_model(fd_m: float, fs: float, samp_num: int = 3.6e4):
	r"""
	Args:
		fd_m: doppler frequency.
		fs: sampling frequency.
		samp_num: the number of sampling.
	"""
	n_fft = nextpow2_int(samp_num)
	n_ifft = int(np.ceil(n_fft * fs / (2 * fd_m)))
	# Gaussian
	g_i = np.random.randn(n_fft)
	g_q = np.random.randn(n_fft)
	cg_i = fft(g_i)
	cg_q = fft(g_q)
	doppler_coeff = doppler_spectrum(fd_m=fd_m, n_fft=n_fft)
	f_cg_i = cg_i * np.sqrt(doppler_coeff)
	f_cg_q = cg_q * np.sqrt(doppler_coeff)
	filtered_cg_i = np.concatenate(
		[
			f_cg_i[:n_fft//2],
			np.zeros(n_ifft - n_fft),
			f_cg_i[n_fft//2:]
		],
		axis=0
	)
	filtered_cg_q = np.concatenate(
		[
			f_cg_q[:n_fft//2],
			np.zeros(n_ifft - n_fft),
			f_cg_q[n_fft//2:],
		],
		axis=0
	)
	h_i = ifft(filtered_cg_i)
	h_q = ifft(filtered_cg_q)
	ray_envelope = np.sqrt(np.abs(h_i)**2 + np.abs(h_q)**2)
	ray_rms = np.sqrt(np.mean(np.power(ray_envelope, 2)) / 2)
	real = h_i.real[:samp_num]
	imag = - h_q.real[:samp_num]
	h_factors = (real + 1j * imag) / ray_rms
	return h_factors, n_fft, n_ifft, doppler_coeff


def doppler_spectrum(fd_m: float, n_fft: int):
	delta_f = 2 * fd_m / n_fft
	freq_array = np.zeros(n_fft // 2)
	spectrum = np.zeros(n_fft)
	spectrum[0] = 1.5 / (np.pi * fd_m)

	for i in range(1, n_fft // 2):
		freq_array[i] = i * delta_f
		spectrum[i] = 1.5 / (np.pi * fd_m * np.sqrt(1 - np.power(freq_array[i]/fd_m, 2)))
		spectrum[n_fft - i] = spectrum[i]
	n_fit_num = 3
	factors = np.polyfit(freq_array[-n_fit_num - 1: ], spectrum[n_fft // 2 - n_fit_num - 1: n_fft // 2], n_fit_num)
	spectrum[n_fft // 2] = np.polyval(factors, freq_array[-1] + delta_f)
	return spectrum


def prepare_fadings(
	data_bits_num: int,
	training_symbol_interval: int,
	f_doppler: float,
	symbol_ms: float,
	fading_num: int,
	overwrite: bool = False
):
	fading_data_dir = root / r"data/channel"
	if not os.path.exists(fading_data_dir):
		os.makedirs(fading_data_dir)

	fading_i_path = fading_data_dir / r"fading_i.npy"
	fading_q_path = fading_data_dir / r"fading_q.npy"

	if os.path.exists(fading_i_path) and os.path.exists(fading_q_path) and not overwrite:
		return

	fs = 1e3 / symbol_ms
	symbol_valid_bits = 28

	symbols_num = int(np.ceil(data_bits_num / symbol_valid_bits))
	symbol_group_num = int(np.ceil(symbols_num / training_symbol_interval))
	padded_symbols_num = symbol_group_num * training_symbol_interval

	fading_i_buffer = []
	fading_q_buffer = []

	# 1, training_interval, 1, training_interval, ...

	for idx in range(fading_num):
		fading_factors, _, _, _ = fwgn_model(fd_m=f_doppler, fs=fs, samp_num=padded_symbols_num + symbol_group_num)
		fading_i_buffer.append(fading_factors.real)
		fading_q_buffer.append(fading_factors.imag)

	fading_i_buffer = np.stack(fading_i_buffer, axis=0)
	fading_q_buffer = np.stack(fading_q_buffer, axis=0)
	print("fading i shape: ", fading_i_buffer.shape)
	print("fading q shape: ", fading_q_buffer.shape)
	np.save(fading_i_path, fading_i_buffer)
	np.save(fading_q_path, fading_q_buffer)


if __name__ == "__main__":
	image_bits_num = 256*56*56
	symbols_num = ((image_bits_num // 28) // 32) * 33
	print("Symbols num: ", symbols_num)

	plot_fwgn(fd_m=0.5, samp_ms=2.5, samp_num=symbols_num)
