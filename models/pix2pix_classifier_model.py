# -*- coding: utf-8 -*-
import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
from torchvision import models
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import csv
import pandas as pd

class Pix2PixClassifierModel(BaseModel):
    def name(self):
        return 'Pix2PixClassifierModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)

            self.netC = networks.define_C(self.gpu_ids)
 
            self.vgg_model = networks.define_P(self.opt.perceptual_model_dir,self.gpu_ids)

            self.freader = pd.read_csv(opt.dataroot + 'fine_grained_attribute.txt', header = 0, sep=' ')
            self.people = self.freader['imgname'].tolist()
            self.labels = self.freader['Male']
            self.labels[self.labels==-1]=0

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)
                self.vgg_model = networks.define_P(self.opt.perceptual_model_dir,self.gpu_ids)
                # self.netC = networks.define_C(self.gpu_ids)
                #         self.load_network(self.netC, 'C', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            self.old_c_lr = opt.lr*0.01;
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()
            # add the perceptual loss 
            self.criterionP = torch.nn.L1Loss()
            # add the classicier loss
            self.criterionC = torch.nn.BCELoss()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_C = torch.optim.Adam(self.netC.parameters(),
                                                lr=self.old_c_lr, betas=(opt.beta1, 0.999))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
            networks.print_network(self.netC)
            if self.opt.perceptual_model_dir is not None:
                print('using perceptual loss model')
            else:
                print('not using perceptual loss model')
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        if self.isTrain:
            imagename = self.image_paths[0].split('/')[-1]
            idx = self.people.index(imagename)
            self.gender = np.array(self.labels[idx].astype(np.float32))


    def forward(self):
        self.real_A = Variable(self.input_A).cuda()
        self.fake_B = self.netG.forward(self.real_A)
        self.real_B = Variable(self.input_B).cuda()
        self.preprocess_fake_B = Variable(util.preprocess(self.fake_B)).cuda()
        self.perceptual_fake_B_out = self.vgg_model.forward(self.preprocess_fake_B)[3]
        self.gender_fake_B_out = self.vgg_model.forward(self.preprocess_fake_B)[4]

    #self.perceptual_real_B = Variable(util.preprocess(self.real_B).data, requires_grad = False)
    
    def forward_C(self):
        self.real_B = Variable(self.input_B).cuda()
        self.preprocess_real_B = Variable(util.preprocess(self.real_B)).cuda()
        self.perceptual_real_B_out = self.vgg_model.forward(self.preprocess_real_B)[3]
        self.gender_real_B_out = self.vgg_model.forward(self.preprocess_real_B)[4]        

    # no backprop gradients
    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG.forward(self.real_A)
        self.real_B = Variable(self.input_B, volatile=True)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        self.pred_fake = self.netD.forward(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        self.pred_real = self.netD.forward(real_AB)
        self.loss_D_real = self.criterionGAN(self.pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_C(self):

        self.attr = self.Tensor(1, 1)
        self.attr = torch.from_numpy(self.gender).float().cuda()

        self.label = Variable(self.attr)

        self.classifier_real = self.netC.forward(self.gender_real_B_out)

        self.loss_C_real = self.criterionC(self.classifier_real, self.label)*0.0001

        self.loss_C_real.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD.forward(fake_AB)
        self.classifier_fake = self.netC.forward(self.gender_fake_B_out)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A
        
        # Third, other loss
        self.loss_G_perceptual = self.criterionP(self.perceptual_fake_B_out, self.perceptual_real_B_out)*self.opt.lambda_P
        self.loss_C_fake = self.criterionC(self.classifier_fake, self.label)*1

        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_perceptual+self.loss_C_fake

        self.loss_G.backward()

    def optimize_C_parameters(self):
        self.forward_C()
        self.optimizer_C.zero_grad()
        self.backward_C()
        self.optimizer_C.step()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_C_errors(self):
        return OrderedDict([('C_gender', self.loss_C_real.data[0]),
                        ]) 

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.data[0]),
                    ('G_L1', self.loss_G_L1.data[0]),
                            ('G_P', self.loss_G_perceptual.data[0]),
                            ('G_C_fake', self.loss_C_fake.data[0]),
                        ('D_real', self.loss_D_real.data[0]),
                        ('D_fake', self.loss_D_fake.data[0])
                        ]) 

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netC, 'C', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def update_C_rate(self):
        lrd = self.old_c_lr / self.opt.citer
        lr = self.old_c_lr - lrd
        for param_group in self.optimizer_C.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_c_lr, lr))
        self.old_c_lr = lr
