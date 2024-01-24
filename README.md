# DAC-DETR 

This is the official implementation (PyTorch and [PaddlePaddle](https://github.com/huzhengdongcs/DAC-DETR/tree/main/paddle)) of the paper "[DAC-DETR: Divide the Attention Layers and Conquer](https://openreview.net/pdf?id=8JMexYVcXB)".

Authors: Zhengdong Hu, Yifan Sun, Jingdong Wang, Yi Yang

&#x1F4E7; &#x1F4E7; &#x1F4E7; Contact: huzhengdongcs@gmail.com

## News
[Sep. 22 2023] DAC-DETR: Divide the Attention Layers and Conquer, has been accepted at NeurIPS 2023 as a poster.


## Methods：

This paper reveals a characteristic of DEtection Transformer (DETR) that negatively impacts its training efficacy, i.e., the cross-attention and self-attention layers in DETR decoder have contrary impacts on the object queries (though both impacts are important). Specifically, we observe the cross-attention tends to gather multiple queries around the same object, while the self-attention disperses these queries far away. To improve the training efficacy, we propose a Divide-And-Conquer DETR (DAC-DETR) that divides the cross-attention out from this contrary for better conquering. During training, DAC-DETR employs an auxiliary decoder that focuses on learning the cross-attention layers. The auxiliary decoder, while sharing all the other parameters, has NO self-attention layers and employs one-to-many label assignment to improve the gathering effect. Experiments show that DAC-DETR brings remarkable improvement over popular DETRs. For example, under the 12 epochs training scheme on MS-COCO, DAC-DETR improves Deformable DETR (ResNet-50) by +3.4 AP and achieves 50.9 (ResNet-50) / 58.1 AP (Swin-Large) based on some popular methods (i.e., DINO and an IoU-related loss). 


<div align=center> <img width=80% height=80% src="https://github.com/huzhengdongcs/DAC-DETR/blob/main/figs/pipline.jpg"/></div>

## Analysis

We count the averaged number of queries that have large affinity with each object. 
Compared with the baseline, DAC-DETR 1) has more queries for each object, and 2) improves the quality of the closest queries.y-axis denotes “avg number of queries / object".

<div align=center> <img width=100% height=100% src="https://github.com/huzhengdongcs/DAC-DETR/blob/main/figs/analyse.jpg"/></div>

## Experiments

<div align=center> <img width=80% height=80% src="https://github.com/huzhengdongcs/DAC-DETR/blob/main/figs/exe1.jpg"/></div>
<div align=center> <img width=80% height=80% src="https://github.com/huzhengdongcs/DAC-DETR/blob/main/figs/exe2.jpg"/></div>

## To Do

Notes: You can first access the pytorch version of  'dac-detr + contrastive denoising (cdn)' and model from
1. [Baidu Netdisk.](https://pan.baidu.com/s/1aIBlrkYSR1Td8f8t-S3w1g?pwd=nun6) 
2. [Google Drive](https://drive.google.com/file/d/14pV5HLxTLmTgIoieFW2UQG2suanABqy2/view?usp=drive_link)
   
This model has been trained under 12 epochs, as detailed in the paper, and has achieved a result of 50.0. In addition to the code and model, we are also providing the training log.

Below are some key experimental details for your reference:

1. This code is implemented by PyTorch.
2. You need to ' sh env_run.sh ' to prepare the basic environment. (including pycocotools, tqdm, cython, scipy)
3. If you want to train the code, you can ' sh train.sh ', and use ' sh test.sh ' to evaluate the model.
4. Please note that our implementations are based on 8 A100 or 8 V100 GPUS.

We will release the whole code ASAP.

## Citing DAC-DETR
If you find DAC-DETR useful to your research, please consider citing:

```
@inproceedings{
hu2023dacdetr,
title={{DAC}-{DETR}: Divide the Attention Layers and Conquer},
author={Zhengdong Hu and Yifan Sun and Jingdong Wang and Yi Yang},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=8JMexYVcXB}
}
