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


If you have prblem to download the pretrained vgg16.t7 file. You can download [here](https://www.dropbox.com/s/jizamlhqqybo50d/vgg16.t7?dl=0): and put it into directory: ./models  

## Reference

$ @article{di2017gp,
  title={{GP-GAN}: gender preserving GAN for synthesizing faces from landmarks},
  author={Di, Xing and Sindagi, Vishwanath A and Patel, Vishal M},
  journal={arXiv preprint arXiv:1710.00962},
  year={2017}
}
$

## Acknowledgments
This is work is highlg inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix#prerequisites)  

We highly thanks to [He Zhang](https://github.com/hezhangsprinter) for his discussing about DenseUnet part
