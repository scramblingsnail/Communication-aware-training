import torch
import os
import numpy as np
from pathlib import Path


def load_symbol_ber():
	root = Path(__file__)
	for i in range(3):
		root = root.parent

	ber_path = root / "experiment_data/symbol-wise/symbol_ber.txt"

	ber_array = np.loadtxt(str(ber_path))
	return ber_array


SYMBOL_BER_ARRAY = load_symbol_ber()


# from .modules import QConv2d, QReLU
# from .hybrid_cnn import HybridCNN


# def record_conv_layer(conv_layer: QConv2d, activation: QReLU):
#     conv_dict = {}
#     conv_dict['w_quantized'] = conv_layer.quantized
#     conv_dict['weight'] = conv_layer.weight.detach().data.numpy()
#     conv_dict['bias'] = conv_layer.bias.detach().data.numpy()
#     conv_dict['w_scale'] = conv_layer.scale.numpy() if conv_dict['w_quantized'] else None
#     conv_dict['w_zero_point'] = conv_layer.zero_point.numpy() if conv_dict['w_quantized'] else None
#
#     conv_dict['a_quantized'] = activation.quantized
#     conv_dict['a_scale'] = activation.scale.numpy() if conv_dict['a_quantized'] else None
#     conv_dict['a_zero_point'] = activation.zero_point.numpy() if conv_dict['a_quantized'] else None
#     return conv_dict
#
#
# def record_residual_block(record_dict: dict):
#     res_dict = {}
#
#     return
#
#
# def record_quantized_model(quantized_model: HybridCNN, config):
#     if not os.path.exists(config['edge_network_directory']):
#         os.mkdir(config['edge_network_directory'])
#     # quantized_model = torch.load(quantized_model_path)
#     save_path = os.path.join(config['edge_network_directory'], 'edge_cnn.h5')
#     data_save_path = os.path.join(config['edge_network_directory'], 'input_data.h5')
#     f = h5py.File(save_path, 'w')
#     conv0_layer = record_conv_layer(quantized_model.conv0, quantized_model.relu0)
#     f.create_dataset(name='conv0', data=conv0_layer)
#     for block_idx in range(quantized_model.blocks_num):
#         block = quantized_model.__getattr__('res_block{}'.format(block_idx))
#         block_dict = {}



# test_path = './test.h5'
# test_f = h5py.File(test_path, 'w')
#
# test_f.create_dataset('str1', data=[[False, True]])
# test_f.create_dataset('dict1', data=np.random.rand(2, 3, 4))
# test_f.close()
#
# read_f = h5py.File(test_path, 'r')
# for key in read_f.keys():
#     print(key)
#     print(read_f[key][0])
