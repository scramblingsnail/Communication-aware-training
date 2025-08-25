import torch
import random
import datetime
import logging
import yaml
import os
import time
import shutil
import torchvision
import torch.nn as nn
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer, SGD
from torch.optim.lr_scheduler import StepLR

from src.dataset.image_net_loader import get_dataloader
from src.channel_simulate.fitting import BERModel
from src.hybrid_model import HybridCNN
from src.channel_simulate.link_simulation_torch.simulation import ClarkeSimulator
from typing import Union, Tuple, List
from collections import OrderedDict



class Logger:
	def __init__(
		self,
		file_path: str
	):
		self.file_path = file_path

	def critical(self, log_str):
		t_str = datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
		with open(self.file_path, "a") as log_f:
			log_f.write(f"\n{t_str}\n{log_str}\n")


class ParamStep:
	def __init__(self, param_step_settings: dict):
		self.required_names = ("add_mul_mode", "step_size", "period")
		self._check_settings(param_step_settings)
		self.step_settings = param_step_settings
		self.count = 0

	def _check_settings(self, settings: dict):
		for key in settings.keys():
			each_setting = settings[key]
			if not isinstance(each_setting, dict):
				raise ValueError("dict type required for each param setting.")
			for name in self.required_names:
				if name not in each_setting.keys():
					raise ValueError(f"Require setting: '{name}'.")

	def _step(self, config: dict, verbose: bool):
		log_str = "Step hyper-params: "
		for p_name in self.step_settings.keys():
			p_setting = self.step_settings[p_name]
			if p_name not in config.keys():
				continue

			p = config[p_name]
			period = p_setting["period"]
			if self.count % period != 0:
				continue

			if p_setting["add_mul_mode"].upper() == "ADD":
				p = p + p_setting["step_size"]
			else:
				p = p * p_setting["step_size"]
			config[p_name] = p
			log_str += f"-- {p_name} to {p}\t"
		if verbose:
			print(log_str + "\n")
		return config

	def step(self, config: dict, verbose: bool=False):
		self.count += 1
		c = self._step(config, verbose)
		return c


def get_config(config_path):
	with open(config_path, 'r') as f:
		config = yaml.safe_load(f)
	return config


def update_config(config: dict, new_config: dict):
	config.update(new_config)
	if config['dataset'].upper() in ['MNIST', 'FASHIONMNIST']:
		config['in_channels'] = 1
	else:
		config['in_channels'] = 3

	if config["dataset"].upper() in ["MNIST", "FASHIONMNIST", "CIFAR-10"]:
		config['labels_num'] = 10
	elif config["dataset"].upper() in ["IMAGENET-100", "CIFAR-100"]:
		config["labels_num"] = 100
	elif config["dataset"].upper() == "IMAGENET":
		config["labels_num"] = 1000
	else:
		raise ValueError("Invalid dataset.")
	return config


def _coupling_checkpoint_dir(checkpoints_dir):
	return f"{checkpoints_dir}_noise"


def make_dirs(config: dict):
	checkpoints_dir = config["checkpoints_directory"]
	training_with_noise = config["noisy_training"]
	flat_aug_mode = config["flat_channel_aug_mode"]
	selective_aug_mode = config["freq_selective_aug_mode"]
	train_ber_model_orders = config["train_ber_model_orders"]
	# A same dir prefix corresponds to two checkpoint dir: ckpt_dir & {ckpt_dir}_noise
	if training_with_noise:
		checkpoints_dir = _coupling_checkpoint_dir(checkpoints_dir)
		if flat_aug_mode:
			checkpoints_dir = f"{checkpoints_dir}_flat"
		if selective_aug_mode:
			checkpoints_dir = f"{checkpoints_dir}_selective"

		# Training OFDM order
		mark_str = "order_" + "_".join([str(each) for each in train_ber_model_orders])
		checkpoints_dir = f"{checkpoints_dir}_{mark_str}"

	config["checkpoints_directory"] = checkpoints_dir
	if not os.path.exists(checkpoints_dir):
		os.makedirs(checkpoints_dir)

	ber_model_path = config["ber_model_path"]
	ber_models = []
	for order in train_ber_model_orders:
		ber_models.append(f"{ber_model_path}_order{order}.pkl")

	config["ber_model_path"] = ber_models


Eval_ACC1 = "Eval_ACC1"
Eval_ACC5 = "Eval_ACC5"
Eval_Loss = "Eval_Loss"
Train_ACC1 = "Train_ACC1"
Train_ACC5 = "Train_ACC5"
Train_Loss = "Train_Loss"
Wireless_Loss = "Wireless_Loss"
Iter_Time = "Iter_Time"


def init_my_model(config: dict):
	clarke_channel = ClarkeSimulator(
		training_symbol_interval=config["training_symbol_interval"],
		avg_snr=0,
		channel_fading_dir=config["channel_fading_dir"],
		symbol_period=config["symbol_period"],
		samp_per_symbol=config["samp_per_symbol"],
		cyclic_prefix_num=config["cyclic_prefix_num"],
		aimc_bit=config["init_aimc_bit"],
		sub_carrier_num=config["valid_sub_carrier_num"],
		modulation_order=config["init_modulation_order"],
		flat_mode=config["flat_mode"],
		multi_path_num=config["multi_path_num"],
		min_path_power=config["min_path_power"],
		max_path_power=config["max_path_power"],
		min_path_delay=config["min_path_delay"],
		max_path_delay=config["max_path_delay"],
		train_symbol_snr=config["train_symbol_snr"]
	)

	resnet = HybridCNN(
		in_channels=config['in_channels'], kernel_size=config['kernel_size'],
		first_stride=config.get("first_stride", 1),
		first_kernel_size=config.get("first_kernel_size", 3),
		first_pool_layer_setting=config.get("first_pool_layer_setting", None),
		bottleneck_type=config["bottleneck_type"],
		channel_simulator=clarke_channel,
		blocks_setting=config['blocks'], labels_num=config['labels_num'],
		quantize_w_blocks=config['quantize_w_blocks'], quantize_a_blocks=config['quantize_a_blocks'],
		quantize_a=config['quantize_a'], w_bit_width=config['w_bit_width'],
		a_bit_width=config['a_bit_width'], q_w_range=config['q_w_range'], q_a_range=config['q_a_range'],
		w_slope=config['w_slope'], a_slope=config['a_slope'], learn_w_lower=config['learn_w_lower'],
		learn_w_upper=config['learn_w_upper'], learn_a_lower=config['learn_a_lower'],
		learn_a_upper=config['learn_a_upper'], learn_w_slope=config['learn_w_slope'],
		learn_a_slope=config['learn_a_slope'], dtype=torch.float32, ber_model_path=config["ber_model_path"],
		init_wireless_p=config["init_wireless_p"], wireless_p_scale=config["wireless_p_scale"],
		flat_channel_aug_mode=config["flat_channel_aug_mode"],
		freq_selective_aug_mode=config["freq_selective_aug_mode"],
		eval_multi_channel_configs=config["eval_multi_channel_configs"],
	)
	if config.get("noisy_training", False):
		resnet.noisy_training_mode(True)
	else:
		resnet.noisy_training_mode(False)
	if config.get("noisy_evaluation", False):
		resnet.noisy_eval_mode(True)
	else:
		resnet.noisy_eval_mode(False)
	if config.get("use_channel_model_for_eval", False):
		resnet.use_channel_model_for_eval(True)
	if config.get("tracking_trained_wireless_param", False):
		resnet.track_trained_wireless_param(True)
	return resnet


def slope_step(config, model: Union[DDP, HybridCNN], current_epoch: int, gpu_id: int, train_logger):
	if current_epoch <= config['slope_warmup_epochs']:
		w_slope = config['warmup_slope'] + (config['w_slope'] - config['warmup_slope']) / config['slope_warmup_epochs'] * current_epoch
		a_slope = config['warmup_slope'] + (config['a_slope'] - config['warmup_slope']) / config['slope_warmup_epochs'] * current_epoch
		if not config['learn_w_slope']:
			if isinstance(model, DDP):
				model.module.set_w_q_param('slope', w_slope)
			else:
				model.set_w_q_param('slope', w_slope)
		if not config['learn_a_slope']:
			if isinstance(model, DDP):
				model.module.set_a_q_param('slope', a_slope)
			else:
				model.set_a_q_param('slope', a_slope)

		# if config["learn_a_upper"]:
		# 	u_bot, u_top = config["clip_a_upper_range"]
		# 	if isinstance(model, DDP):
		# 		model.module.clip_a_q_param("clip_upper", u_bot, u_top)
		# 	else:
		# 		model.clip_a_q_param("clip_upper", u_bot, u_top)

	if isinstance(model, DDP):
		lowers, uppers, slopes = model.module.get_a_q_param()
		w_lowers, w_uppers, w_slopes = model.module.get_w_q_param()
	else:
		lowers, uppers, slopes = model.get_a_q_param()
		w_lowers, w_uppers, w_slopes = model.get_w_q_param()

	if gpu_id == 0:
		upper_values = [val.detach().cpu().item() for val in uppers]
		slope_values = [val.detach().cpu().item() for val in slopes]
		upper_strs = ["{:.4f}".format(val) for val in upper_values]
		slope_strs = ["{:.4f}".format(val) for val in slope_values]
		log_str = "Epoch {}\tCurrent a upper: {};\tCurrent a slope: {};".format(
			current_epoch, ", ".join(upper_strs), ", ".join(slope_strs)
		)
		if w_lowers:
			w_upper_values = [val.detach().cpu().item() for val in w_uppers]
			w_lower_values = [val.detach().cpu().item() for val in w_lowers]
			w_slope_values = [val.detach().cpu().item() for val in w_slopes]

			w_upper_strs = ["{:.4f}".format(val) for val in w_upper_values]
			w_lower_strs = ["{:.4f}".format(val) for val in w_lower_values]
			w_slope_strs = ["{:.4f}".format(val) for val in w_slope_values]
			if config["learn_w_lower"]:
				log_str += "\tCurrent w lower: {}".format(", ".join(w_lower_strs))
			if config["learn_w_upper"]:
				log_str += "\tCurrent w upper: {}".format(", ".join(w_upper_strs))
			if config["learn_w_slope"]:
				log_str += "\tCurrent w slope: {}".format(", ".join(w_slope_strs))
			# w_log_str = "\nEpoch {}\tCurrent w upper: {};\tCurrent w lower: {}\tCurrent w slope: {};".format(
			# 	current_epoch, ", ".join(w_upper_strs), ", ".join(w_lower_strs), ", ".join(w_slope_strs)
			# )
			# log_str += w_log_str
		print(log_str)
		train_logger.critical(log_str)


class AverageMeter(object):
	def __init__(self, name, percent_mode: bool = False):
		self.name = name
		self.avg = 0
		self.sum = 0
		self.count = 0
		self.percent_mode = percent_mode
		self.reset()

	def reset(self):
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n: int = 1):
		self.sum += val * n
		self.count += n

	def all_reduce(self, to_item: bool = True):
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		if isinstance(self.sum, np.ndarray):
			val_array = list(self.sum)
			val_array.append(self.count)
			meta_val = torch.tensor(val_array, device=device).float()
			dist.all_reduce(meta_val, op=dist.ReduceOp.SUM, async_op=False)
			meta_val_np = meta_val.cpu().numpy()
			self.sum, self.count = meta_val_np[:-1], int(meta_val_np[-1])
		elif isinstance(self.sum, float):
			meta_val = torch.tensor([self.sum, self.count], device=device).float()
			dist.all_reduce(meta_val, op=dist.ReduceOp.SUM, async_op=False)
			self.sum, self.count = meta_val.tolist()
		else:
			raise ValueError

		if self.count > 0:
			self.avg = self.sum / self.count
		else:
			self.avg = 0
		if self.percent_mode:
			self.avg *= 100.
		if to_item:
			self.avg = np.mean(self.avg)


class StatusRecorder(object):
	r"""
	Attributes:
		meters: List[AverageMeter]
	"""
	def __init__(self, meters: List[AverageMeter], show_len: int, train_logger):
		self.meters = meters
		self.show_len = show_len
		self.train_logger = train_logger

	def show_status(self):
		global logger
		name_list = [meter.name for meter in self.meters]
		val_list = [meter.avg for meter in self.meters]
		show_names = [each.ljust(self.show_len) for each in name_list]
		name_str = "\t".join(show_names)
		show_values = []
		sep_list = ["=".ljust(self.show_len, "=") for i in range(len(name_list))]
		sep_str = "\t".join(sep_list)
		for val in val_list:
			val_str = "{}".format(val)
			val_str = val_str.ljust(self.show_len)
			show_values.append(val_str)
		val_str = "\t".join(show_values)
		log_str = f"{sep_str}\n{name_str}\n{val_str}\n{sep_str}"
		self.train_logger.critical(log_str)
		print(log_str)


def compute_acc(
	batch_output: torch.Tensor,
	batch_labels: torch.Tensor,
	topk_list: list,
	return_acc: bool = True
):
	r"""
	Compute the accuracy.

	Args:
		batch_output (torch.Tensor): (batch_size, label_num) or (model_num, batch_size, label_num)
		batch_labels (torch.Tensor): (batch_size, )
		topk_list (list): e.g. [1, 5]
		return_acc (bool): if True, return batch acc; else, return correct num.

	Returns:
		acc_list or correct_num_list.
	"""
	max_k = max(topk_list)
	# (model_num, batch_size, max_k) or (batch_size, max_k)
	batch_size = batch_labels.size()[0]
	_, top_ids = torch.topk(batch_output, k=max_k, dim=-1, largest=True, sorted=True)
	if len(top_ids.size()) == 2:
		expand_labels = torch.broadcast_to(batch_labels.unsqueeze(1), top_ids.size())
	else:
		expand_labels = torch.broadcast_to(batch_labels[None, :, None], top_ids.size())

	# (model_num, batch_size, max_k) or (batch_size, max_k)
	correct = torch.eq(top_ids, expand_labels)
	correct_num_list = []
	for top_k in topk_list:
		# (model_num, ) or (1, )
		correct_num = torch.sum(correct[..., :top_k], dim=[-2, -1])
		if len(correct_num.size()) == 0:
			correct_num = correct_num.detach().cpu().item()
		else:
			correct_num = correct_num.detach().cpu().numpy()
		correct_num_list.append(correct_num)

	if return_acc:
		acc_list = [num * 100. / batch_size for num in correct_num_list]
		return acc_list
	return correct_num_list


def _generate_random_ber_vec(model_ber_vec: torch.Tensor, ):
	valid_vec = torch.cat([model_ber_vec[:24], model_ber_vec[26:]], dim=0)
	vec_mean = valid_vec.mean()
	random_ber_vec = torch.rand(model_ber_vec.size(), device=model_ber_vec.device)
	random_ber_vec[24:26] = 0
	random_mean = random_ber_vec.mean()
	random_ber_vec = random_ber_vec.div(random_mean) * vec_mean

	while random_ber_vec.gt(1.).any():
		random_ber_vec = torch.rand(model_ber_vec.size(), device=model_ber_vec.device)
		random_ber_vec[24:26] = 0
		random_mean = random_ber_vec.mean()
		random_ber_vec = random_ber_vec.div(random_mean) * vec_mean

	valid_random = torch.cat([random_ber_vec[:24], random_ber_vec[26:]], dim=0)
	similarity = (valid_random - valid_vec).abs().mean()
	return random_ber_vec, similarity


def _filter_checkpoint(checkpoint: OrderedDict, target_model: nn.Module):
	names = list(checkpoint.keys())
	target_state_dict = target_model.state_dict()
	for p_name in names:
		items = p_name.split(".")
		if items[1] == "flipper" and items[-1] != "wireless_p":
			checkpoint[p_name] = target_state_dict[p_name]
	return checkpoint

def evaluate_worker(gpu_id: int, world_size: int, config: dict, val_logger):
	os.environ["MASTER_ADDR"] = "localhost"
	os.environ["MASTER_PORT"] = "19993"
	dist.init_process_group(
		backend="nccl",
		rank=gpu_id,
		world_size=world_size,
	)
	torch.cuda.set_device(gpu_id)
	device = torch.device(f"cuda:{gpu_id}")
	loss_func = nn.CrossEntropyLoss().to(device)

	model = init_my_model(config=config)
	model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
	model.cuda(gpu_id)
	model = DDP(model, device_ids=[gpu_id,], find_unused_parameters=True)
	# load state dict
	checkpoint_dir = config["checkpoints_directory"]
	checkpoint_path = f"{checkpoint_dir}/best.pth.tar"
	checkpoint = torch.load(checkpoint_path, map_location=device)

	model_checkpoint = _filter_checkpoint(checkpoint=checkpoint["model_state_dict"], target_model=model)

	model.load_state_dict(model_checkpoint)
	model.module.map_inner_to(device)

	each_batch_size = int(config["batch_size"] / world_size)
	num_workers = int(config["num_workers"] / world_size)
	_, val_loader, calibration_loader = get_dataloader(
		dataset_dir=config["dataset_directory"],
		batch_size=each_batch_size,
		class_calibration_num=config["each_class_calibration_num"],
		num_workers=num_workers,
		dist_mode=True,
		gpu_id=gpu_id,
	)

	# TODO: Check quantize.
	model.module.quantize(
		post_q_w_indices=config['post_quantize_w_layers'],
		post_q_a_indices=config['post_quantize_a_layers'],
		post_w_bit_width=config['post_w_bit_width'],
		post_a_bit_width=config['post_a_bit_width'],
		calibration_loader=calibration_loader,
		calibration_epochs=config['calibration_epochs'],
		DDP_mode=True,
	)

	# if not model.module.noisy:
	# 	# Set the wireless parameters to that of the corresponding noisy model.
	# 	coupling_config = config.copy()
	# 	coupling_config["noisy_training"] = True
	# 	coupling_model = init_my_model(config=coupling_config)
	# 	coupling_model.cuda(gpu_id)
	# 	coupling_model = DDP(coupling_model, device_ids=[gpu_id, ], find_unused_parameters=True)
	# 	coupling_ckpt_dir = _coupling_checkpoint_dir(checkpoint_dir)
	# 	coupling_ckpt_path = f"{coupling_ckpt_dir}/best.pth.tar"
	# 	coupling_ckpt = torch.load(coupling_ckpt_path)
	# 	print(coupling_ckpt_path)
	# 	coupling_model.load_state_dict(coupling_ckpt["model_state_dict"])
	# 	# TODO: multi params.
	# 	wireless_param = coupling_model.module.flipper.wireless_p.data.detach()
	# 	wireless_param = wireless_param.to(device)
	# 	model.module.flipper.set_wireless_param(wireless_param)

	loaded_wireless_p = model.module.flipper.meaningful_wireless_param().detach().cpu().numpy()

	if config.get("eval_wireless_param", None) is not None:
		eval_wireless_param = config["eval_wireless_param"]
		model.module.track_trained_wireless_param(False)
		model.module.set_channel_param(name="avg_snr", val=eval_wireless_param)
		loaded_wireless_p = model.module.channel_simulator.avg_snr

	if config.get("Manual_flip_pattern_mode", False):
		with torch.no_grad():
			wireless_p = model.module.flipper.meaningful_wireless_param()
			model_ber_vec = model.module.flipper.ber_model(wireless_p)
			manual_ber_vec = torch.zeros(model_ber_vec.size(), device=device)
		if gpu_id == 0:
			manual_ber_vec, similarity = _generate_random_ber_vec(model_ber_vec=model_ber_vec)
			print("BER vec: ", manual_ber_vec, model_ber_vec.device)
			print("BER Mean: ", model_ber_vec.mean())
			print("Similarity: ", similarity)
		dist.broadcast(manual_ber_vec, src=0)
		model.module.flipper.reload_dummy_model(manual_ber_vec)
		loaded_ber_vec = model.module.flipper.ber_model.random_ber_vec
		print("Manually set BER vec: ", loaded_ber_vec)

	eval_mode = model.module.evaluate_noisy
	if gpu_id == 0:
		print("======== Loaded Checkpoint (Epoch {}): {}".format(checkpoint["epoch"], checkpoint_path))
		print("======== Loaded wireless parameter: {}".format(loaded_wireless_p))
		print("======== Evaluating with noise: {}".format(eval_mode))

	acc1, acc5 = validate(
		gpu_id=gpu_id,
		val_loader=val_loader,
		model=model,
		loss_func=loss_func,
		device=device,
		val_logger=val_logger,
		return_item=False,
	)

	if isinstance(acc1, np.ndarray):
		acc1 = ",".join(["{:.4f}".format(val) for val in acc1])
		acc5 = ",".join(["{:.4f}".format(val) for val in acc5])

	eval_result_path = f"{checkpoint_dir}/eval_results.txt"
	if gpu_id == 0:
		with open(eval_result_path, "a") as eval_f:
			log_str = "Evaluate with Noise or not: {}\tAcc@top1: {}\tAcc@top5: {}".format(
				model.module.evaluate_noisy,
				acc1,
				acc5,
			)
			log_str += "\tWirelessParam: {:.4f}".format(loaded_wireless_p)
			log_str += "\tChannel flat mode: {}".format(config["flat_mode"])

			if config.get("eval_multi_channel_configs", None) is not None:
				eval_channel_config = config["eval_multi_channel_configs"]
				log_str += "\tAIMC bit: " + "-".join(list(map(str, eval_channel_config["aimc_bit"])))
				log_str += "\tOFDM order: " + "-".join(list(map(str, eval_channel_config["order"]))) + "\n"
			else:
				eval_orders = config["train_ber_model_orders"]
				eval_aimc_bits = [each_order + 1 for each_order in eval_orders]
				log_str += "\tAIMC bit: {}".format("-".join(list(map(str, eval_aimc_bits))))
				log_str += "\tOFDM order: {}\n".format("-".join(list(map(str, eval_orders))))
			eval_f.write(log_str)


def evaluate_main(world_size: int, config: dict, val_logger):
	seed = config.get("seed", None)
	if seed:
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)

	mp.spawn(evaluate_worker, nprocs=world_size, args=(world_size, config, val_logger))


def train(
	gpu_id: int,
	train_loader: DataLoader,
	model: Union[nn.Module, DDP],
	optimizer: Optimizer,
	loss_func: nn.Module,
	device: torch.device,
	config: dict,
	train_logger,
):
	global Train_ACC1, Train_ACC5, Train_Loss, Iter_Time

	train_acc1 = AverageMeter(name=Train_ACC1)
	train_acc5 = AverageMeter(name=Train_ACC5)
	train_loss = AverageMeter(name=Train_Loss)
	wireless_loss = AverageMeter(name=Wireless_Loss)
	iter_time = AverageMeter(name=Iter_Time)

	training_recorder = StatusRecorder(
		meters=[train_acc1, train_acc5, train_loss, wireless_loss, iter_time],
		show_len=8,
		train_logger=train_logger,
	)

	top_k_list = [1, 5]
	model.train()
	before = time.time()
	count = 0
	wireless_loss_ratio = config["wireless_loss_ratio"]
	u_bot, u_top = config["clip_a_upper_range"]
	slope_bot, slope_top = config["slope_a_range"]
	w_upper_bot, w_upper_top = config["clip_w_upper_range"]
	w_lower_bot, w_lower_top = config["clip_w_lower_range"]
	w_slope_bot, w_slope_top = config["slope_w_range"]
	trace_mode = False

	# TODO: Trace training
	# def trace_handler(prof):
	# 	dataset_name = config["dataset"]
	# 	log_dir = f"./train_logs/{dataset_name}"
	# 	if not os.path.exists(log_dir):
	# 		os.makedirs(log_dir)
	#
	# 	table_log_path = fr"{log_dir}/train_key_averages_table_{str(prof.step_num)}.txt"
	# 	key_averages_table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)
	# 	with open(table_log_path, "w") as table_f:
	# 		table_f.write(key_averages_table)
	# 	print(key_averages_table)
	# 	prof.export_chrome_trace(f"{log_dir}/train_trace_" + str(prof.step_num) + ".json")

	# trace_mode = True
	# with torch.profiler.profile(
	# 	activities=[
	# 		torch.profiler.ProfilerActivity.CPU,
	# 		torch.profiler.ProfilerActivity.CUDA,
	# 	],
	# 	schedule=torch.profiler.schedule(
	# 		wait=2,
	# 		warmup=2,
	# 		active=5,
	# 	),
	# 	with_stack=False,
	# 	on_trace_ready=trace_handler,
	#
	# ) as torch_prof:

	for batch_images, batch_labels in train_loader:
		# Profiling, disable it when normally training.
		# TODO: Trace training
		# torch_prof.step()

		# TODO: Trace training
		# with torch.profiler.record_function("Inputs loading & preprocessing"):

		count += 1
		batch_size = batch_labels.size()[0]
		batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)

		# TODO: Trace training
		# with torch.profiler.record_function("Forward"):

		batch_output = model.forward(batch_images, trace_mode=trace_mode)

		# TODO: Trace training
		# with torch.profiler.record_function("Loss calculation"):

		output_num = batch_output.size()[0]
		if output_num == batch_size:
			net_loss = loss_func(batch_output, batch_labels)
		else:
			# Multi channel models.
			channel_model_num = output_num // batch_size
			output_size = torch.Size((channel_model_num, batch_size)) + batch_output.size()[1:]
			batch_output = batch_output.reshape(output_size)
			net_loss = 0
			for m_idx in range(channel_model_num):
				each_loss = loss_func(batch_output[m_idx], batch_labels)
				net_loss = net_loss + each_loss / batch_output.size()[0]

		loss = net_loss
		# wireless loss
		communicate_loss = None
		if isinstance(model, DDP):
			if model.module.noisy:
				communicate_loss = model.module.flipper.wireless_loss()
				loss = loss + wireless_loss_ratio * communicate_loss
		else:
			if model.noisy:
				communicate_loss = model.flipper.wireless_loss()
				loss = loss + wireless_loss_ratio * communicate_loss

		# TODO: Trace training
		# with torch.profiler.record_function("Backward"):

		# BackPropagation
		optimizer.zero_grad()
		loss.backward()
		clip_grad_norm_(parameters=model.parameters(), max_norm=10.)

		# TODO: Trace training
		# with torch.profiler.record_function("Updating parameter"):

		optimizer.step()
		# clip upper
		if config["learn_a_upper"]:
			model.module.clip_a_q_param("clip_upper", u_bot, u_top)
		if config["learn_a_slope"]:
			model.module.clip_a_q_param("slope", slope_bot, slope_top)
		if config["learn_w_lower"]:
			model.module.clip_w_q_param("clip_lower", w_lower_bot, w_lower_top)
		if config["learn_w_upper"]:
			model.module.clip_w_q_param("clip_upper", w_upper_bot, w_upper_top)
		if config["learn_w_slope"]:
			model.module.clip_w_q_param("slope", w_slope_bot, w_slope_top)
		model.module.flipper.clamp_()

		# Performance
		now = time.time()
		batch_time = now - before
		before = now

		# TODO: Trace training
		# with torch.profiler.record_function("Accuracy calculation"):

		# float or 1d numpy array
		batch_acc1, batch_acc5 = compute_acc(
			batch_output=batch_output,
			batch_labels=batch_labels,
			topk_list=top_k_list,
			return_acc=True,
		)
		train_acc1.update(batch_acc1, batch_size)
		train_acc5.update(batch_acc5, batch_size)
		train_loss.update(net_loss.detach().cpu().item(), batch_size)
		if communicate_loss is not None:
			wireless_loss.update(communicate_loss.detach().cpu().item() * wireless_loss_ratio, batch_size)
		iter_time.update(batch_time, batch_size)

	train_acc1.all_reduce(to_item=False)
	train_acc5.all_reduce(to_item=False)
	iter_time.all_reduce()
	train_loss.all_reduce()
	wireless_loss.all_reduce()

	if gpu_id == 0:
		training_recorder.show_status()


def validate(
	gpu_id: int,
	val_loader: DataLoader,
	model: nn.Module,
	loss_func: nn.Module,
	device: torch.device,
	val_logger,
	return_item: bool = True,
	eval_wireless_params: torch.Tensor = None,
):
	global Eval_ACC1, Eval_ACC5, Eval_Loss

	eval_acc1 = AverageMeter(name=Eval_ACC1)
	eval_acc5 = AverageMeter(name=Eval_ACC5)
	eval_loss = AverageMeter(name=Eval_Loss)

	eval_recorder = StatusRecorder(
		meters=[eval_acc1, eval_acc5, eval_loss],
		show_len=8,
		train_logger=val_logger,
	)

	model.eval()
	top_k_list = [1, 5]
	with torch.no_grad():
		for batch_images, batch_labels in val_loader:
			batch_size = batch_labels.size()[0]
			batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
			batch_output = model.forward(batch_images, wireless_params=eval_wireless_params)

			output_num = batch_output.size()[0]
			if output_num == batch_size:
				loss = loss_func(batch_output, batch_labels)
			else:
				# Multi channel models.
				channel_model_num = output_num // batch_size
				output_size = torch.Size((channel_model_num, batch_size)) + batch_output.size()[1:]
				batch_output = batch_output.reshape(output_size)
				loss = 0
				for m_idx in range(channel_model_num):
					each_loss = loss_func(batch_output[m_idx], batch_labels)
					loss = loss + each_loss / batch_output.size()[0]

			batch_acc1, batch_acc5 = compute_acc(
				batch_output=batch_output,
				batch_labels=batch_labels,
				topk_list=top_k_list,
				return_acc=True,
			)

			eval_acc1.update(batch_acc1, batch_size)
			eval_acc5.update(batch_acc5, batch_size)
			eval_loss.update(loss.cpu().item(), batch_size)

	eval_acc1.all_reduce(to_item=return_item)
	eval_acc5.all_reduce(to_item=return_item)
	eval_loss.all_reduce()
	if gpu_id == 0:
		eval_recorder.show_status()
	return eval_acc1.avg, eval_acc5.avg


def save_checkpoint(state: dict, better: bool, checkpoint_path: str, best_path: str):
	torch.save(state, checkpoint_path)
	if better:
		shutil.copyfile(checkpoint_path, best_path)


def main_worker(gpu_id: int, world_size: int, config: dict, train_logger):
	global Train_ACC1, Train_ACC5, Train_Loss, Iter_Time
	global Eval_ACC1, Eval_ACC5, Eval_Loss
	best_acc1, best_acc5 = 0, 0
	best_init_acc1, best_init_acc5 = 0, 0
	best_wireless_status = ""

	os.environ["MASTER_ADDR"] = "localhost"
	os.environ["MASTER_PORT"] = "19993"
	dist.init_process_group(
		backend="nccl",
		rank=gpu_id,
		world_size=world_size,
	)
	torch.cuda.set_device(gpu_id)
	device = torch.device(f"cuda:{gpu_id}")

	model = init_my_model(config=config)

	model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
	model.cuda(gpu_id)
	model = DDP(model, device_ids=[gpu_id,], find_unused_parameters=True)
	model.module.map_inner_to(device)

	init_wireless_params = model.module.flipper.meaningful_wireless_param().detach()
	best_wireless_params = model.module.flipper.meaningful_wireless_param().detach()

	loss_func = nn.CrossEntropyLoss().to(device)
	optimizer = SGD(
		params=model.parameters(),
		lr=config["lr"],
		momentum=config["momentum"],
		weight_decay=config["weight_decay"],
	)
	scheduler = StepLR(
		optimizer=optimizer,
		step_size=30,
		gamma=0.1,
	)

	p_stepper = ParamStep(
		param_step_settings=config["hyper_param_schedule"],
	)

	start_epoch = 0
	if config["start_from_checkpoint"]:
		checkpoint_dir = config["checkpoints_directory"]
		checkpoint_path = f"{checkpoint_dir}/best.pth.tar"
		checkpoint = torch.load(checkpoint_path, map_location=device)
		model.load_state_dict(checkpoint["model_state_dict"])
		optimizer.load_state_dict(checkpoint["optimizer"])
		scheduler.load_state_dict(checkpoint["scheduler"])
		start_epoch = checkpoint["epoch"]

	each_batch_size = int(config["batch_size"] / world_size)
	num_workers = int(config["num_workers"] / world_size)

	train_loader, val_loader, _ = get_dataloader(
		dataset_dir=config["dataset_directory"],
		batch_size=each_batch_size,
		class_calibration_num=config["each_class_calibration_num"],
		num_workers=num_workers,
		dist_mode=True,
		gpu_id=gpu_id,
	)
	print("batch num: ", len(train_loader))

	epoch_num = config["epochs"]
	checkpoints_dir = config["checkpoints_directory"]
	checkpoint_path = f"{checkpoints_dir}/checkpoint.pth.tar"
	best_path = f"{checkpoints_dir}/best.pth.tar"

	if gpu_id == 0:
		if model.module.noisy:
			train_logger.critical("Training with noisy wireless channels")
		else:
			train_logger.critical("Training with no noise")

		if model.module.evaluate_noisy:
			train_logger.critical("Evaluating with noisy wireless channels")
		else:
			train_logger.critical("Evaluating with no noise")

	for epoch in range(start_epoch, epoch_num):
		if gpu_id == 0:
			print(f"Processing Epoch {epoch} ...")
		assert isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler)
		train_loader.sampler.set_epoch(epoch)
		slope_step(model=model, current_epoch=epoch, config=config, gpu_id=gpu_id, train_logger=train_logger)

		train(
			gpu_id=gpu_id,
			train_loader=train_loader,
			model=model,
			optimizer=optimizer,
			loss_func=loss_func,
			device=device,
			config=config,
			train_logger=train_logger,
		)

		wireless_p_status = model.module.flipper.get_wireless_param_str()
		if gpu_id == 0:
			print(wireless_p_status)
			train_logger.critical(wireless_p_status)

		if gpu_id == 0:
			print("\nEvaluating with the Initial wireless params ...")
		init_acc1, init_acc5 = validate(
			gpu_id=gpu_id,
			val_loader=val_loader,
			model=model,
			loss_func=loss_func,
			device=device,
			val_logger=train_logger,
			return_item=False,
			eval_wireless_params=init_wireless_params,
		)

		if gpu_id == 0:
			print("\nEvaluating with the Former Best wireless params ...")
		acc1, acc5 = validate(
			gpu_id=gpu_id,
			val_loader=val_loader,
			model=model,
			loss_func=loss_func,
			device=device,
			val_logger=train_logger,
			return_item=False,
			eval_wireless_params=best_wireless_params,
		)

		scheduler.step()
		config = p_stepper.step(config, verbose=gpu_id==0)

		if isinstance(acc1, np.ndarray):
			acc1_str = ",".join(["{:.4f}".format(val) for val in acc1])
			acc5_str = ",".join(["{:.4f}".format(val) for val in acc5])
			init_acc1_str = ",".join(["{:.4f}".format(val) for val in init_acc1])
			init_acc5_str = ",".join(["{:.4f}".format(val) for val in init_acc5])
			acc1 = np.mean(acc1)
			acc5 = np.mean(acc5)
			init_acc1 = np.mean(init_acc1)
			init_acc5 = np.mean(init_acc5)
		else:
			acc1_str = "{:.4f}".format(acc1)
			acc5_str = "{:.4f}".format(acc5)
			init_acc1_str = "{:.4f}".format(init_acc1)
			init_acc5_str = "{:.4f}".format(init_acc5)

		# save checkpoints
		better = init_acc1 > best_init_acc1 and acc1 > best_acc1
		if better:
			best_wireless_params = model.module.flipper.meaningful_wireless_param().detach()
			best_acc1, best_acc5 = acc1, acc5
			best_init_acc1, best_init_acc5 = init_acc1, init_acc5
			best_acc1_str, best_acc5_str = acc1_str, acc5_str
			best_init_acc1_str, best_init_acc5_str = init_acc1_str, init_acc5_str
			best_wireless_status = wireless_p_status
			best_str = "\nBest Acc@1: {};\tBest Acc@5: {}\nBest Init Acc@1: {};\tBest Init Acc@5: {}\nBest Wireless Param: {}\n".format(
				best_acc1_str,
				best_acc5_str,
				best_init_acc1_str,
				best_init_acc5_str,
				best_wireless_status,
			)

			if gpu_id == 0:
				print(best_str)

		if gpu_id == 0:
			save_checkpoint(
				state={
					"epoch": epoch + 1,
					"seed": config["seed"],
					"model_state_dict": model.state_dict(),
					"eval_acc1": acc1,
					"best_acc1": best_acc1,
					"eval_init_acc1": init_acc1,
					"best_init_acc1": best_init_acc1,
					"optimizer": optimizer.state_dict(),
					"scheduler": scheduler.state_dict(),
				},
				better=better,
				checkpoint_path=checkpoint_path,
				best_path=best_path,
			)
	# end
	if gpu_id == 0:
		log_str = "\nBest Acc@1: {:.2f};\tBest Acc@5: {:.2f}\tBest Wireless Param: {}".format(best_acc1, best_acc5, best_wireless_status)
		train_logger.critical(log_str)
	dist.destroy_process_group()


def train_main(world_size: int, config: dict, train_logger):
	seed = config.get("seed", None)
	if seed:
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)

	mp.spawn(main_worker, nprocs=world_size, args=(world_size, config, train_logger))


def a_train_task(train_config: dict):
	make_dirs(train_config)
	logger = Logger(file_path="{}/train_info.log".format(train_config["checkpoints_directory"]))

	setting_log_str = "======== Training Settings ========\n"

	for key in train_config.keys():
		val_set = train_config[key]
		each_line = "{}:\t{}\n".format(key, val_set)
		setting_log_str += each_line
	logger.critical(setting_log_str)

	print("Imagenet loading.", train_config["dataset"])
	if train_config["dataset"].upper() == "IMAGENET":
		imagenet_mean = torch.FloatTensor([0.485, 0.456, 0.406])
		imagenet_std = torch.FloatTensor([0.229, 0.224, 0.225])
		dataset_dir = train_config["dataset_directory"]
		torch.save(imagenet_mean, f"{dataset_dir}/train_mean.pt")
		torch.save(imagenet_std, f"{dataset_dir}/train_std.pt")

	gpu_num = torch.cuda.device_count()

	config_path = "resnet50_config.yaml"
	my_config = get_config(config_path)
	my_config = update_config(my_config, new_config=train_config)

	train_main(world_size=gpu_num, config=my_config, train_logger=logger)


def a_evaluate_task(eval_config: dict):
	make_dirs(eval_config)
	logger = Logger(file_path="{}/eval_info.log".format(eval_config["checkpoints_directory"]))
	gpu_num = torch.cuda.device_count()

	print("Imagenet loading.", eval_config["dataset"])
	if eval_config["dataset"].upper() == "IMAGENET":
		imagenet_mean = torch.FloatTensor([0.485, 0.456, 0.406])
		imagenet_std = torch.FloatTensor([0.229, 0.224, 0.225])
		dataset_dir = eval_config["dataset_directory"]
		torch.save(imagenet_mean, f"{dataset_dir}/train_mean.pt")
		torch.save(imagenet_std, f"{dataset_dir}/train_std.pt")

	config_path = "resnet50_config.yaml"
	my_config = get_config(config_path)
	my_config = update_config(my_config, new_config=eval_config)
	evaluate_main(world_size=gpu_num, config=my_config, val_logger=logger)


def train_eval_imagenet():
	my_train_config = get_config("train_config.yaml")
	ber_model_num = len(my_train_config["train_ber_model_orders"])
	my_train_config["wireless_loss_ratio"] = 0.05 / ber_model_num
	my_train_config["hyper_param_schedule"]["wireless_loss_ratio"]["step_size"] = 0.05 / ber_model_num
	my_train_config["wireless_p_scale"] = my_train_config["wireless_p_scale"] / ber_model_num
	a_train_task(train_config=my_train_config)

	for flat_mode in [True, False]:
		snr_list = list(range(-10, 31, 1))
		for snr in snr_list:
			each_config = {
				"flat_mode": flat_mode,
				"eval_wireless_param": snr, # It specifies the channel model settings during evaluation
			}
			my_train_config.update(each_config)
			a_evaluate_task(eval_config=my_train_config)


if __name__ == "__main__":
	train_eval_imagenet()
