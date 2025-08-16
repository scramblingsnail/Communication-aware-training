import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from typing import List


root = Path(__file__)
for i in range(3):
	root = root.parent


def load_channel_snr_data(ofdm_order: int):
	snr_path = root / f"data/channel/snr_list_order{ofdm_order}.npy"
	ber_path = root / f"data/channel/symbol_wise_ber_order{ofdm_order}.npy"
	snr_list = np.load(str(snr_path))
	snr_list = snr_list[..., np.newaxis]
	symbol_bers = np.load(str(ber_path))
	# (snr_num, data_num, valid_bits_num)
	avg_bers = np.mean(symbol_bers, axis=1)
	print(snr_list.shape)
	print(symbol_bers.shape)
	print(avg_bers.shape)
	return snr_list, avg_bers


class ChannelDataset(Dataset):
	def __init__(self, load_data_func, **kwargs):
		wireless_params, ber = load_data_func(**kwargs)
		self.wireless_params = torch.tensor(wireless_params, dtype=torch.float)
		self.ber = torch.tensor(ber, dtype=torch.float)

	def __getitem__(self, index: int):
		return self.wireless_params[index], self.ber[index]

	def __len__(self):
		return self.wireless_params.size()[0]


def build_mlp(layer_dims: List[int]):
	layers = []
	for idx in range(len(layer_dims) - 1):
		in_dim = layer_dims[idx]
		out_dim = layer_dims[idx + 1]
		layers.extend(
			[
				nn.Linear(in_dim, out_dim),
				nn.Tanh()
			]
		)
	layers[-1] = nn.Sigmoid()
	model = nn.Sequential(*layers)
	return model


class BERModel(nn.Module):
	def __init__(
		self,
		wireless_p_dim: int,
		wireless_p_lower: torch.Tensor,
		wireless_p_upper: torch.Tensor,
		symbol_bits: int,
		hidden_dims: List[int],
	):
		super().__init__()
		assert wireless_p_lower.size() == (wireless_p_dim, )
		assert wireless_p_upper.size() == (wireless_p_dim, )
		self.wireless_p_dim = wireless_p_dim
		self.wireless_p_lower = wireless_p_lower
		self.wireless_p_upper = wireless_p_upper
		self.symbol_bits = symbol_bits
		self.ber_net = build_mlp(layer_dims=[wireless_p_dim, *hidden_dims, symbol_bits])

	def forward(self, inputs):
		if self.wireless_p_upper.device != inputs.device:
			self.wireless_p_lower = self.wireless_p_lower.to(inputs.device)
			self.wireless_p_upper = self.wireless_p_upper.to(inputs.device)

		inputs = (inputs - self.wireless_p_lower) / (self.wireless_p_upper - self.wireless_p_lower)
		inputs = torch.clamp(inputs, 0, 1)
		return self.ber_net(inputs)


def fit_ber_curve(load_data_func=load_channel_snr_data, **kwargs):
	qam_order = kwargs.get("qam_order", 1)
	overwrite = kwargs.get("overwrite", False)
	channel_model_dir = root / "checkpoints/clarke_channel_models"
	if not os.path.exists(str(channel_model_dir)):
		os.makedirs(str(channel_model_dir))

	best_model_path = channel_model_dir / f"clarke_model_order{qam_order}.pkl"

	if os.path.exists(best_model_path) and not overwrite:
		return

	epochs = 5000
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	channel_dataset = ChannelDataset(load_data_func, **kwargs)
	train_loader = DataLoader(dataset=channel_dataset, batch_size=5, shuffle=True)

	input_size = channel_dataset.wireless_params.size()[-1]
	output_size = channel_dataset.ber.size()[-1]

	inputs_lower, _ = torch.min(channel_dataset.wireless_params, dim=0)
	print(inputs_lower)
	inputs_upper, _ = torch.max(channel_dataset.wireless_params, dim=0)
	print(inputs_upper)
	inputs_lower = inputs_lower.to(device)
	inputs_upper = inputs_upper.to(device)

	hidden_dims = [8, 16]
	model = BERModel(
		wireless_p_dim=input_size,
		wireless_p_lower=inputs_lower,
		wireless_p_upper=inputs_upper,
		symbol_bits=output_size,
		hidden_dims=hidden_dims,
	)
	model.to(device)

	criterion = nn.MSELoss()
	lr = 5e-3
	optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
	lr_schedule = ReduceLROnPlateau(
		optimizer=optimizer,
		mode="min",
		factor=0.5,
		patience=100,
		threshold=1e-4,
		min_lr=5e-4,
	)
	min_loss = 1e4

	for epoch_idx in range(epochs):
		avg_loss = 0
		for batch_input, batch_ber in train_loader:
			batch_input = batch_input.to(device)
			batch_ber = batch_ber.to(device)
			batch_output = model(batch_input)
			loss = criterion(batch_output, batch_ber)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			avg_loss += loss.detach().cpu().item()

		lr_schedule.step(avg_loss)
		current_lr = optimizer.param_groups[0]["lr"]
		print(f"Epoch {epoch_idx}; Average loss: {avg_loss}; Current lr: {current_lr}")

		if avg_loss < min_loss:
			torch.save(model, str(best_model_path))


def net_ber(ofdm_order: int):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model_path = root / f"checkpoints/clarke_channel_models/clarke_model_order{ofdm_order}.pkl"
	model = torch.load(model_path, map_location=device)
	inputs, valid_bers = load_channel_snr_data(ofdm_order=ofdm_order)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	input_num = inputs.shape[0]
	inputs = np.reshape(inputs, (-1, ))

	inserted_inputs = np.linspace(min(inputs), max(inputs), 1000)
	model_inputs = inserted_inputs[:, np.newaxis]
	model_inputs = torch.tensor(model_inputs, dtype=torch.float, device=device)
	fitted_valid_bers = model(model_inputs).detach().cpu().numpy()
	print(fitted_valid_bers.shape)
	print(valid_bers.shape)
	return inputs, valid_bers, inserted_inputs, fitted_valid_bers


def plot_snr_bers(ofdm_order: int, dummy_save: bool):
	blue = np.array([19, 164, 192, 255]) / 255
	green = np.array([193, 148, 255, 255]) / 255
	red = np.array([219, 93, 135, 255]) / 255
	d_color = (blue - red) / 10
	d_red_to_green = (green - red) / 5
	d_green_to_blue = (blue - green) / 5

	valid_symbol_bits = 28 * ofdm_order
	row_num = 6
	col_num = 5
	dpi = 300
	snr_list, varying_bers, inserted_snr, fitted_ber = net_ber(ofdm_order = ofdm_order)
	print("here")
	print(snr_list.shape)
	print(varying_bers.shape)
	print(inserted_snr.shape)
	print(fitted_ber.shape)

	marker_list = ["o", "D", "v", "s", "2"]

	if dummy_save:
		dummy_figs, dummy_axes = plt.subplots(row_num, col_num, figsize=(15, 18), dpi=dpi)
	else:
		figs, axes = plt.subplots(row_num, col_num, figsize=(15, 18), dpi=dpi)

	x_ticks = np.arange(snr_list[0], snr_list[-1] + 1, 5)
	for bit_idx in range(valid_symbol_bits):
		freq_idx = bit_idx // (2 * ofdm_order)
		symbol_bit_idx = bit_idx % (2 * ofdm_order)

		if bit_idx % 2 == 0:
			ax_idx = 2 * freq_idx
			freq_name = "Q"
		else:
			ax_idx = 2 * freq_idx + 1
			freq_name = "I"

		row_idx = ax_idx // col_num
		col_idx = ax_idx % col_num

		if bit_idx % 2 == 0:
			color = blue
		else:
			color = red

		if dummy_save:
			dummy_axes[row_idx][col_idx].plot(
				snr_list, varying_bers[:, bit_idx],
				marker=marker_list[symbol_bit_idx // 2], color=color, markersize=4,
				label=f"Bit {bit_idx} carried by freq {freq_idx}, {freq_name}",
			)
		else:
			axes[row_idx][col_idx].scatter(
				snr_list, varying_bers[:, bit_idx],
				marker=marker_list[symbol_bit_idx // 2], color=color, s=4,
			)

			axes[row_idx][col_idx].set_xticks(x_ticks)
			axes[row_idx][col_idx].plot(
				inserted_snr,
				fitted_ber[:, bit_idx],
				color=color,
				linewidth=0.6,
				label=f"Bit {bit_idx} carried by freq {freq_idx}, {freq_name}",
			)
			axes[row_idx][col_idx].tick_params(
				axis="both", which='major', length=3, width=0.75, colors='black', direction='in',
			)

	for r_idx in range(row_num):
		for c_idx in range(col_num):
			if dummy_save:
				dummy_axes[r_idx][c_idx].legend()
			else:
				axes[r_idx][c_idx].legend()

	# plt.show()
	image_dir = root / "images"
	if not os.path.exists(str(image_dir)):
		os.makedirs(str(image_dir))
	image_path = image_dir / f"net_bers_order{ofdm_order}.svg"
	dummy_image_path = image_dir / f"dummy_net_bers_order{ofdm_order}.svg"

	if dummy_save:
		plt.savefig(dummy_image_path, dpi=dpi)
	else:
		plt.savefig(image_path, dpi=dpi)
	# plt.show()


if __name__ == "__main__":
	# load_channel_snr_data()
	fit_ber_curve(ofdm_order=2)

	# for order in range(1, 4):
	# 	for dummy in [True, False]:
	# 		plot_snr_bers(ofdm_order=order, dummy_save=dummy)