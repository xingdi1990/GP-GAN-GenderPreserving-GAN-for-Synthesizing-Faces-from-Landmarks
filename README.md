# GP-GAN: Gender-Preserving-GAN-for-Synthesizing-Faces-from-Landmarks
This repository is about out ICPR work, GP-GAN: Gender Preserving GAN for Synthesizing Faces from Landmarks

## Prerequisites
This code has tested on Ubuntu 16/18 with Pytorch 0.4 and cuda 9.0/8.0

## Getting Started

Clone this repo:  
* git clone https://github.com/DetionDX/GP-GAN-GenderPreserving-GAN-for-Synthesizing-Faces-from-Landmarks.git  
* cd GP-GAN-GenderPreserving-GAN-for-Synthesizing-Faces-from-Landmarks  

Download dataset
* cd datasets
* bash download_lfw_landmark_dataset.sh

## Training
change directory into the cloned folder
* cd ..
* python train.py --dataroot ./datasets/lfw/ --which_direction BtoA --fineSize 64 --loadSize 64 --no_flip  --name lfw_gpgan

To view training results and loss plots, run:  
* python -m visdom.server  

and click the URL http://localhost:8097


If you have prblem to download the pretrained vgg16.t7 file. You can download [here](https://www.dropbox.com/s/6nkmly7onpi5uug/vgg16.t7?dl=0): and put it into directory: ./models  

## Reference
```
@INPROCEEDINGS{di2018gp, 
author={Xing Di and Vishwanath A. Sindagi and Vishal M. Patel}, 
booktitle={2018 24th International Conference on Pattern Recognition (ICPR)}, 
title={GP-GAN: Gender Preserving GAN for Synthesizing Faces from Landmarks}, 
year={2018},
pages={1079-1084}, 
month={Aug},}

```

## Acknowledgments
This is work is highlg inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix#prerequisites)  

We highly thanks to [He Zhang](https://github.com/hezhangsprinter) for his discussing about DenseUnet part
