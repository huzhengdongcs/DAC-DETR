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


## Installation

We use python=3.7.10, pytorch=1.8.0, cuda=11.1.

Clone the repo
```
git https://github.com/huzhengdongcs/DAC-DETR.git
cd DAC-DETR
```
Prepare environments

```
sh env_run.sh
```

## Data

mkdir ./data/
```
data/
  └── coco/
     ├── train2017/
     ├── val2017/
     └── annotations/
```

## Pretrain backbones
mkdir ./initmodel

You can download [Resnet50](https://drive.google.com/file/d/1vSEiR7td16dD7wNqZAb7I6txa69eln6i/view?usp=drive_link)  and [Swin_transformer](https://drive.google.com/file/d/1BSZHgF9cD9c4BqI47NwMxAxUxRphNKw6/view?usp=drive_link) and put them into ./initmodel

## Run

Please note that our implementations are based on 8 A100 or 8 V100 GPUS.

For example, you can run dac_cdn_ice with 12 epochs, Res50 by 
```
sh train.sh 
```
## Eval

The trained models are saved in output.

For example, you can test dac_cdn_ice with 12 epochs, Res50 by 

```
sh test.sh
```

## Models


| Name | Backbone | epochs | AP | Model| log|
|:---:|:---:|:---:|:---:|:---:|:---:|
| dac_cdn| Res50 | 12 |  50.0 | [Google](https://drive.google.com/drive/folders/1PRZLpr7i6L1VycyKjUTMlpPyn5Bex8We?usp=drive_link), [Baidu](https://pan.baidu.com/s/1rYQXtblTLHzD9q38YIcwfw?pwd=7c7g) | [Google](https://drive.google.com/drive/folders/1PRZLpr7i6L1VycyKjUTMlpPyn5Bex8We?usp=drive_link), [Baidu](https://pan.baidu.com/s/1rYQXtblTLHzD9q38YIcwfw?pwd=7c7g) | 
| dac_cdn| Res50 | 24 |  51.2 | [Google](https://drive.google.com/drive/folders/1Es-dilk_XCMQayFipCkGZZuW3RGdg2DY?usp=drive_link), [Baidu](https://pan.baidu.com/s/1_i5HHXI1JeNFr6j-mnXXkg?pwd=7ng5) | [Google](https://drive.google.com/drive/folders/1Es-dilk_XCMQayFipCkGZZuW3RGdg2DY?usp=drive_link), [Baidu](https://pan.baidu.com/s/1_i5HHXI1JeNFr6j-mnXXkg?pwd=7ng5)| 
| dac_cdn_ice| Res50 | 12 |  50.9 | [Google](https://drive.google.com/drive/folders/1BxzkDRsDDengzINr0jhVW-6Yp2OSOLs8?usp=drive_link), [Baidu](https://pan.baidu.com/s/1nRk7aZhop_iOlAjBHBdXrg?pwd=4g63) | [Google](https://drive.google.com/drive/folders/1BxzkDRsDDengzINr0jhVW-6Yp2OSOLs8?usp=drive_link), [Baidu](https://pan.baidu.com/s/1nRk7aZhop_iOlAjBHBdXrg?pwd=4g63) | 
| dac_cdn_ice| Res50 | 24 |  52.1 | [Google](https://drive.google.com/drive/folders/10HahQiv0KkFq6iYkM7KicPdl-HPmzm4g?usp=drive_link), [Baidu](https://pan.baidu.com/s/19Fb5dBmbym8Q5MMOe-GARQ?pwd=qw6n) | [Google](https://drive.google.com/drive/folders/10HahQiv0KkFq6iYkM7KicPdl-HPmzm4g?usp=drive_link), [Baidu](https://pan.baidu.com/s/19Fb5dBmbym8Q5MMOe-GARQ?pwd=qw6n) | 
| dac_cdn | Swin_Large | 12 |  57.3 | [Google](https://drive.google.com/drive/folders/1T2tQn8TqdhbptATp7WzFx24SRSafNCEi?usp=drive_link), [Baidu](https://pan.baidu.com/s/1MmoKCsUODnLqYnC0hoiM8w?pwd=ybq2) | [Google](https://drive.google.com/drive/folders/1T2tQn8TqdhbptATp7WzFx24SRSafNCEi?usp=drive_link), [Baidu](https://pan.baidu.com/s/1MmoKCsUODnLqYnC0hoiM8w?pwd=ybq2) | 
| dac_cdn_ice| Swin_Large | 12 |  58.1 | [Google](https://drive.google.com/drive/folders/1BuaRWEiobVXLIBOt7VlavG8ZJcFTgp6S?usp=drive_link), [Baidu](https://pan.baidu.com/s/1xUoq83gs19kzDLd_1Zu6Fg?pwd=sg3i) | [Google](https://drive.google.com/drive/folders/1BuaRWEiobVXLIBOt7VlavG8ZJcFTgp6S?usp=drive_link), [Baidu](https://pan.baidu.com/s/1xUoq83gs19kzDLd_1Zu6Fg?pwd=sg3i)  | 
| dac_cdn_ice| Swin_Large | 24 |  59.3 | [Google](https://drive.google.com/drive/folders/141PiOLUzjdafcy_HC0xOaE0ateIiNtyk?usp=drive_link), [Baidu](https://pan.baidu.com/s/1uA0tnnoztUaxQfMkD9TA7A?pwd=ffm4) | [Google](https://drive.google.com/drive/folders/141PiOLUzjdafcy_HC0xOaE0ateIiNtyk?usp=drive_link), [Baidu](https://pan.baidu.com/s/1uA0tnnoztUaxQfMkD9TA7A?pwd=ffm4) | 

## Notes

You can access the pytorch code of  'dac-detr + contrastive denoising (cdn)' and model from

1. [Baidu Netdisk.](https://pan.baidu.com/s/1kBy3NDnd6J6WGwKQSerq1Q?pwd=yz3k)
2. [Google Drive](https://drive.google.com/file/d/1QB4_vssP1xvjdTygSJKyrRAFwEqB0foR/view?usp=drive_link)




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
