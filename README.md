# Hybrid Quantize

## 概述
本项目混合使用 quantization-aware training 与 post-training quantization, 实现对ResNet-20的任意层（权重/激活）进行任意比特的量化。所使用的quantization-aware training方法为[Soft DSQ](https://arxiv.org/pdf/1908.05033.pdf),用于 4-bit 以下的量化; post-training quantization方法为[AICQ](https://arxiv.org/pdf/1810.05723v1.pdf)，用于 4-bit 以上的量化。在进行训练后量化之前，先对网络进行BN-fold，以简化计算流程。


## 运行环境
Windows/Linux + CPU/GPU + python3.8

## 代码结构
>``` 
> HybridQuantize
>   -- checkpoints                  // 存储训练过程中的模型文件。（该目录自动生成）
>   -- experiment_data              // 存储与点乘实验相关的数据与模型
>       -- q_model                      // 存放量化后的模型文件
>       -- results                      // 存放实验数据
>       -- test_data                    // 存放测试集数据
>   -- images                       // 图片
>   -- src                          // 源代码目录
>       -- dataset                      // 数据集
>           -- __init__.py
>           -- data_loader.py           // 数据集下载、预处理、与加载。
>       -- experiment_tools         // 与点乘实验相关代码
>           -- lib                      // @赵懿晨提供的点乘setup代码
>           -- __init__.py      
>           -- hardware_data.py         // 量化网络的前向计算中插入采集点乘数据
>           -- mvm_core.py              // 定义点乘引擎基类、卷积点乘引擎等。
>       -- hybrid_model             // 支持混合量化的模型代码
>           -- __init__.py          
>           -- hybrid_cnn.py        // 定义支持混合量化的ResNet
>           -- modules.py           // 定义支持混合量化的基本模块：QConv2d, QReLU, QResidualBlock
>       -- quantize                 // 计算融合与量化算法代码
>           -- graph_optimize           // 计算图优化
>             -- __init__.py
>             -- fold_fusion.py             // 计算图融合，包括：BN-fold
>           -- post_train_q             // 训练后量化算法
>             -- __init__.py
>             -- aicq.py                    // AICQ算法
>           -- q_aware_train            // 量化训练算法
>             -- __init__.py
>             -- dsq.py                   // DSQ算法
>           __init__.py
>        __init__.py
>   -- calibrate.py                 // 校准脚本，用于校准点乘setup
>   -- config.yaml                  // 全局配置文件，定义训练、量化、实验的所有参数
>   -- experiment.py                // 点乘实验脚本
>   -- README.md                    // Markdown 文件
>   -- requirements.txt             // 依赖库说明
>   -- train.py                     // 训练脚本
>   -- utils.py                     // 训练与实验相关工具
>
>```

## 依赖库
使用的第三方库与版本见 [requirement](requirements.txt)

## 文章
本项目代码用于获得 “基于存算一体器件的端云协同推理” 课题中 Fig.XX 与 Fig.XXX 所需的实验与仿真数据。