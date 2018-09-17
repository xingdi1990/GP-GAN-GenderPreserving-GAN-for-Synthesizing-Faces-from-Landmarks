# GP-GAN: Gender-Preserving-GAN-for-Synthesizing-Faces-from-Landmarks
This repository is about out ICPR work, GP-GAN: Gender Preserving GAN for Synthesizing Faces from Landmarks

## Prerequisites
This code has tested on Ubuntu 16.06 with Pytorch 0.4 and cuda 9.0

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
* python train.py --dataroot ./datasets/lfw/ --which_direction BtoA --fineSize 64 --loadSize 64 --no_flip  

To view training results and loss plots, run:  
* python -m visdom.server  

and click the URL http://localhost:8097


## Acknowledgments
This is work is highlg inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix#prerequisites)  

We highly thanks to [He Zhang](https://github.com/hezhangsprinter) for his discussing about DenseUnet part
