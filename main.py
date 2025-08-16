from src.channel_simulate.channel_model.fwgn_np import prepare_fadings
from src.channel_simulate.link_simulation_torch.simulation import simulate_ber_data
from src.channel_simulate.fitting import fit_ber_curve
from dist_train_image_net import train_eval_imagenet


if __name__ == "__main__":
	prepare_fadings(
		data_bits_num=256*56*56,
		training_symbol_interval=32,
		f_doppler=0.5,
		symbol_ms=2.5,
		fading_num=5000,
		overwrite=False,
	)

	for qam_order in range(1, 4):
		simulate_ber_data(qam_order=qam_order, overwrite=False)
		fit_ber_curve(qam_order=qam_order, overwrite=False)

	train_eval_imagenet()
