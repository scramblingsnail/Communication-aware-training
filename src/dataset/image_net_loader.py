import os
import time
import shutil
import multiprocessing as mps
import math
from multiprocessing import Process
import numpy as np
import torch
import torch.utils.data as data
from torchvision.datasets import ImageFolder
from torchvision import transforms


def _calculate_mean_std(data_dir: str):
	img_dataset = ImageFolder(
		data_dir,
		transform=transforms.Compose(
			[
				transforms.ToTensor(),
			]
		)
	)
	img_dataloader = data.DataLoader(img_dataset, batch_size=1, shuffle=False, num_workers=16)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	img_sum = torch.zeros((3, ), device=device)
	img_square_sum = torch.zeros((3, ), device=device)
	pixel_num = 0
	count = 0
	for batch_img, batch_label in img_dataloader:
		# NCHW
		if (count + 1) % 10000 == 0:
			print(f"{count} finished.")
		batch_img = batch_img.to(device)
		img_h, img_w = batch_img.size()[2], batch_img.size()[3]
		# N
		pixel_num += img_h * img_w
		# calculate image sum
		each_sum = torch.sum(batch_img, [0, 2, 3])
		img_sum += each_sum
		# calculate square sum
		each_square_sum = torch.sum(torch.square(batch_img), dim=[0, 2, 3])
		img_square_sum += each_square_sum
		count += 1

	mean = torch.div(img_sum, pixel_num)
	square = img_square_sum / pixel_num - torch.square(mean)
	std = torch.sqrt(square)
	print("mean: ", mean)
	print("std: ", std)
	return mean.cpu(), std.cpu()


class ToFloatTensor:
	def __call__(self, tensor: torch.Tensor):
		"""
		Args:
			tensor (torch.Tesnor): a tensor.

		Returns:
			Tensor: float tensor.
		"""
		return tensor.float()

	def __repr__(self):
		return self.__class__.__name__ + '()'


def _process_dataset_mean_std(dataset_dir: str, train_dir: str = None, val_dir: str = None):
	train_dir = train_dir or f"{dataset_dir}/train"
	val_dir = val_dir or f"{dataset_dir}/val"
	train_mean, train_std = _calculate_mean_std(train_dir)
	# val_mean, val_std = _calculate_mean_std(val_dir)
	train_mean_path = f"{dataset_dir}/train_mean.pt"
	train_std_path = f"{dataset_dir}/train_std.pt"
	# val_mean_path = f"{dataset_dir}/val_mean.pt"
	# val_std_path = f"{dataset_dir}/val_std.pt"
	torch.save(train_mean, train_mean_path)
	torch.save(train_std, train_std_path)
	# torch.save(val_mean, val_mean_path)
	# torch.save(val_std, val_std_path)


def _make_calibration_dataset(calibration_dir: str, val_dir: str, class_calibration_num: int):
	os.makedirs(calibration_dir)
	class_dirs = os.listdir(val_dir)

	for class_dir in class_dirs:
		src_dir = f"{val_dir}/{class_dir}"
		target_class_dir = f"{calibration_dir}/{class_dir}"
		os.makedirs(target_class_dir)
		file_names = os.listdir(src_dir)
		for idx in range(class_calibration_num):
			name = file_names[idx]
			src_path = f"{src_dir}/{name}"
			target_path = f"{target_class_dir}/{name}"
			shutil.copyfile(src_path, target_path)
	print(f"Calibration dataset generated.")

def get_dataloader(
	dataset_dir: str,
	batch_size: int,
	num_workers: int,
	train_dir: str = None,
	val_dir: str = None,
	calibration_dir: str = None,
	class_calibration_num: int = None,
	train_mean: torch.Tensor = None,
	train_std: torch.Tensor = None,
	dist_mode: bool = False,
	gpu_id: int = 0,
):
	if not os.path.exists(dataset_dir):
		raise FileExistsError(f"Dataset directory do not exists: {dataset_dir}")

	train_dir = train_dir or f"{dataset_dir}/train"
	val_dir = val_dir or f"{dataset_dir}/val"
	calibration_dir = calibration_dir or f"{dataset_dir}/calibration"
	if not os.path.exists(calibration_dir):
		if class_calibration_num is None:
			raise ValueError("Require the number of calibration images (for each class).")
		if gpu_id == 0:
			_make_calibration_dataset(
				calibration_dir=calibration_dir,
				class_calibration_num=class_calibration_num,
				val_dir=val_dir,
			)

	train_mean_path = f"{dataset_dir}/train_mean.pt"
	train_std_path = f"{dataset_dir}/train_std.pt"
	path_exists = [os.path.exists(each) for each in [train_mean_path, train_std_path]]
	if False in path_exists and None in [train_mean, train_std]:
		if gpu_id == 0:
			_process_dataset_mean_std(
				dataset_dir=dataset_dir,
				train_dir=train_dir,
				val_dir=val_dir,
			)
		else:
			while False in path_exists:
				path_exists = [os.path.exists(each) for each in [train_mean_path, train_std_path]]
				time.sleep(1)

	if train_mean is None:
		train_mean = torch.load(train_mean_path).float()
	if train_std is None:
		train_std = torch.load(train_std_path).float()
	print(train_mean)
	print(train_std)
	normalize = transforms.Normalize(mean=train_mean, std=train_std)

	train_dataset = ImageFolder(
		train_dir,
		transform=transforms.Compose(
			[
				transforms.RandomResizedCrop(224),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				normalize,
				ToFloatTensor(),
			]
		)
	)
	val_dataset = ImageFolder(
		val_dir,
		transform=transforms.Compose(
			[
				transforms.Resize(256),
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				normalize,
				ToFloatTensor(),
			]
		)
	)

	calibration_dataset = ImageFolder(
		calibration_dir,
		transform=transforms.Compose(
			[
				transforms.Resize(256),
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				normalize,
				ToFloatTensor(),
			]
		)
	)

	if dist_mode:
		# drop_last = True to avoid the last batch's influence to BN layers.
		train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, drop_last=True)
		val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
		calibration_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
	else:
		train_sampler = None
		val_sampler = None
		calibration_sampler = None

	train_dataloader = data.DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=not dist_mode,
		num_workers=num_workers,
		pin_memory=True,
		sampler=train_sampler,
	)
	val_dataloader = data.DataLoader(
		val_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=True,
		sampler=val_sampler,
	)

	calibration_dataloader = data.DataLoader(
		calibration_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=True,
		sampler=calibration_sampler,
	)

	return train_dataloader, val_dataloader, calibration_dataloader


if __name__ == "__main__":
	# _load_images(
	# 	data_dir=r"/root/autodl-tmp/imagenet-100/train",
	# 	store_dir=r"/root/autodl-tmp/imagenet-100/train_tensors",
	# 	start_label=0,
	# 	end_label=5,
	# )

	# load_dataset(train_data_dir, val_data_dir)

	# for data_idx in range(1, 6):
	# 	sub_dataset_dir = fr"/root/autodl-tmp/imagenet-100-v{data_idx}"
	# 	_process_dataset_mean_std(sub_dataset_dir)

	# train_loader, val_loader = load_dataset(train_data_dir, val_data_dir)
	#
	# start_t = time.time()
	# for batch_data, batch_label in train_loader:
	# 	print(batch_data.size())
	# end_t = time.time()
	# t_cost = end_t - start_t
	# print("Time cost: ", t_cost)

	_make_calibration_dataset(
		calibration_dir=r"/root/autodl-tmp/imagenet/calibration",
		val_dir="/root/autodl-tmp/imagenet/val",
		class_calibration_num=10,
	)




