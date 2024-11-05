# FIFO: Learning Fog-invariant Features for Foggy Scene Segmentation

## Overview
Robust visual recognition under adverse weather conditions is of great importance in real-world applications. In this context, we propose a new method for learning semantic segmentation models robust against fog. Its key idea is to consider the fog condition of an image as its style and close the gap between images with different fog conditions in neural style spaces of a segmentation model. In particular, since the neural style of an image is in general affected by other factors as well as fog, we introduce a fog-pass filter module that learns to extract a fog-relevant factor from the style. Optimizing the fog-pass filter and the segmentation model alternately gradually closes the style gap between different fog conditions and allows to learn fog-invariant features in consequence. Our method substantially outperforms previous work on three real foggy image datasets. Moreover, it improves performance on both foggy and clear weather images, while existing methods often degrade performance on clear scenes.

## Experimental Results
![Main_qual](https://user-images.githubusercontent.com/57887512/163107476-7e70cebe-6b38-497f-b5bd-f8d6979a8fb0.png)


## Dataset
+ **Cityscapes**: Download the [Cityscapes Dataset](https://www.cityscapes-dataset.com/), and put it in the /root/data1/Cityscapes folder

+ **Foggy Cityscapes**: Download the [Foggy Cityscapes Dataset](https://www.cityscapes-dataset.com/), and put it in the /root/data1/leftImg8bit_foggyDBF folder

+ **Foggy Zurich**: Download the [Foggy Zurich Dataset](https://people.ee.ethz.ch/~csakarid/Model_adaptation_SFSU_dense/), and put it in the /root/data1/Foggy_Zurich folder

+ **Foggy Driving and Foggy Driving Dense**: Download the [Foggy Driving Dataset](https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/), and put it in the /root/data1/Foggy_Driving folder

## Installation
This repository is developed and tested on

- Ubuntu 16.04
- Conda 4.9.2
- CUDA 11.4
- Python 3.7.7
- PaddlePaddle 2.0.0

## Environment Setup
* Required environment is presented in the 'FIFO.yaml' file
* Clone this repo
```bash
~$ git clone https://github.com/IzuiZero/fifo-main-Paddle
~$ cd fifo
~/fifo$ conda env create --file FIFO.yaml
~/fifo$ conda activate FIFO.yaml
```

## Pretrained Models
PRETRAINED_SEG_MODEL_PATH = '[./Cityscapes_pretrained_model.pth](https://drive.google.com/file/d/1IKBXXVhYfc6n5Pw23g7HsH_QzqOG03c6/view?usp=sharing)'


PRETRAINED_FILTER_PATH = '[./FogPassFilter_pretrained.pth](https://drive.google.com/file/d/1xHkL3Y8Y5sHoGkmcevrfMdhFxafVF4_G/view?usp=sharing)' 


## Testing
BEST_MODEL_PATH = '[./FIFO_final_model.pth](https://drive.google.com/file/d/1UF-uotKznN_wqqNqwIkPnpw55l8T9b62/view?usp=sharing
)'

Evaluating FIFO model
```bash
(fifo) ~/fifo$ python evaluate.py --file-name 'FIFO_model' --restore-from BEST_MODEL_PATH
```


## Training
Pretraining fog-pass filtering module
```bash
(fifo) ~/fifo$ python main.py --file-name 'fog_pass_filtering_module' --restore-from PRETRAINED_SEG_MODEL_PATH --modeltrain 'no'
```
Training FIFO
```bash
(fifo) ~/fifo$ python main.py --file-name 'FIFO_model' --restore-from PRETRAINED_SEG_MODEL_PATH --restore-from-fogpass PRETRAINED_FILTER_PATH --modeltrain 'train'
```
