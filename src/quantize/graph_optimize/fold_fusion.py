from torch.nn import Conv2d, BatchNorm2d
import torch


def bn_fold(conv_layer: Conv2d, bn_layer: BatchNorm2d):
    if conv_layer.out_channels != bn_layer.num_features:
        raise ValueError('num features of BN layer and out channels num of CONV layer must match.')
    mean = bn_layer.running_mean
    var = bn_layer.running_var
    eps = bn_layer.eps
    affine_w = bn_layer.weight
    affine_b = bn_layer.bias
    conv_w = conv_layer.weight
    conv_b = conv_layer.bias

    fold_scale = affine_w / torch.pow(var + eps, 0.5)
    fold_w = conv_w * fold_scale[:, None, None, None]
    fold_b = conv_b * fold_scale + affine_b - fold_scale * mean
    return fold_w, fold_b
