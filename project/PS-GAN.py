import os
import math
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter

from lightning_gan import Discriminator
writer = SummaryWriter()

# podaci!!!, inicijalizacija tezina
# u argparser staviti: learning rate, batch size

class PersonDiscriminator(nn.Module): # no ls_gan = False
    def __init__(self): # ndf = broj diskriminatorovih filtera
        super(PersonDiscriminator, self).__init__()
        self.output_num = [4, 2, 1] # za SPP
        input_nc = 3 # u boji
        ndf = 64

        self.conv1 = nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1, bias=False)
        self.LReLU1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=1, padding=1, bias=False)
        # treniranje je brže ako se standardiziraju E i Var, tj.
        # ocekivanje, varijanca = 0, 1
        self.BN1 = nn.BatchNorm2d(ndf * 2)
        self.LReLU2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=1, padding=1, bias=False)
        self.BN2 = nn.BatchNorm2d(ndf * 4)
        self.LReLU3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1, bias=False)
        self.BN3 = nn.BatchNorm2d(ndf * 8)
        self.LReLU4 = nn.LeakyReLU(0.2, inplace=True)

        self.conv5 = nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False)

    def spatial_pyramid_pool(self, previous_conv, num_sample, previous_conv_size, out_pool_size):
        # previous_conv: a tensor vector of previous convolution layer
        # num_sample: an int number of image in the batch
        # previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
        # out_pool_size: a int vector of expected output size of max pooling layer
        # returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
         
        for i in range(len(out_pool_size)):
            h_wid = math.ceil(previous_conv_size[0] / out_pool_size[i]) # visina prozora
            w_wid = math.ceil(previous_conv_size[1] / out_pool_size[i]) # širina prozora
            h_pad = math.floor((h_wid*out_pool_size[i] - previous_conv_size[0] + 1)/2)
            w_pad = math.floor((w_wid*out_pool_size[i] - previous_conv_size[1] + 1)/2)

            maxpool = nn.MaxPool2d(kernel_size=(h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
            x = maxpool(previous_conv)
            if(i == 0):
                spp = x.view(num_sample,-1)
            else:
                spp = torch.cat((spp,x.view(num_sample,-1)), 1)
        return spp

    def forward(self,x):
        x = self.conv1(x)
        x = self.LReLU1(x)

        x = self.conv2(x)
        x = self.LReLU2(self.BN1(x))

        x = self.conv3(x)
        x = self.LReLU3(self.BN2(x))
        
        x = self.conv4(x)
        x = self.LReLU4(self.BN3(x))

        x = self.conv5(x)
        spp = self.spatial_pyramid_pool(x, 1, [int(x.size(2)),int(x.size(3))], self.output_num)

        return spp


class BackgroundDiscriminator(nn.Module): #no ls_gan = true
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(BackgroundDiscriminator, self).__init__()
        # input_nc = 6 # modificirano: input + output u pozivu
        # n_layers = 3
        # ndf = 64

        kw = 4
        padw = math.ceil((kw-1)/2)
        # nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)
        sequence = [
            # 1.
            nn.Conv2d(6, 64, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
            # 2.
            nn.Conv2d(64, 128, kernel_size=kw, stride=2, padding=padw),
            norm_layer(128),
            nn.LeakyReLU(0.2, True),
            # 3.
            nn.Conv2d(128, 256, kernel_size=kw, stride=2, padding=padw),
            norm_layer(256),
            nn.LeakyReLU(0.2, True),
            # 4.
            nn.Conv2d(256, 512, kernel_size=kw, stride=1, padding=padw),
            norm_layer(512),
            nn.LeakyReLU(0.2, True),
            # 5.
            nn.Conv2d(512, 1, kernel_size=kw, stride=1, padding=padw),
            nn.Sigmoid()
        ]

        # nf_mult = 1
        # nf_mult_prev = 1
        # for n in range(1, n_layers):
          #  nf_mult_prev = nf_mult
           # nf_mult = min(2**n, 8)
            #sequence += [
             #   nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
               # norm_layer(ndf * nf_mult),
                #nn.LeakyReLU(0.2, True)
            #]

        #nf_mult_prev = nf_mult
        #nf_mult = min(2**n_layers, 8)
        #sequence += [
         #   nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
           # norm_layer(ndf * nf_mult),
            #nn.LeakyReLU(0.2, True)
        #]

        # sequence += [nn.Conv2d(self.ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        # sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
            return self.model(input)


class Generator(nn.Module):
    def __init__(self, num_downs=8, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=True):
        super(Generator, self).__init__()

        # U-Net strukturu gradimo rekurzivno - od unutarnjeg sloja (dno slova U) prema vanjskim (vrhovi slova U)
        # ali ju u forward pozovemo po redu
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(3, ngf, unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
            return self.model(input)
            

# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
# conv                                                                         relu, upconv, Tanh
#   relu, conv, norm * 3                                       3 * relu, upconv, norm                          
#           relu, conv, norm* 3               3 * relu, upconv, norm, dropout
#                         relu, conv    relu, upconv, norm
#                                   baza
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4, stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)

        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            model = down + [submodule] + up
            if use_dropout:
                model += [nn.Dropout(0.5)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)

class PSGAN (pl.LightningModule):
    def __init__(self, lr=0.0002, batch_size=1):
        super().__init__()
        self.save_hyperparameters()

        self.generator = Generator()
        self.discriminatorP = PersonDiscriminator()
        self.discriminatorB = BackgroundDiscriminator()

        # dimenzije za provjeru?
        # self.validation_z = torch.randn(1,)

    def forward(self, x):
        return self.generator(x)
    
    def Person_loss(self, y_hat, y):
        return nn.MSELoss(y_hat, y)
    
    def Background_loss(self, y_hat, y):
        return nn.BCELoss(y_hat, y)

    def L1_loss(self, y_hat, y):
        return nn.L1Loss(y_hat, y)
    
    # sve popravi
    def training_step(self, batch, batch_idx, optimizer_idx):
        real_imgs, _ = batch
        # z = torch.randn(real_imgs.shape[0], self.hparams.latent_dim)
        # !!!!! potencijalno maknemo uvjete, odmah treniramo i G i Dp i Db
        # train generator
        # train discriminatorP
        # train discriminatorB

    def configure_optimizers(self):
        lr = self.hparams.lr

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5   , 0.999))
        opt_dP = torch.optim.Adam(self.discriminatorP.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_dB = torch.optim.Adam(self.discriminatorB.parameters(), lr=lr, betas=(0.5, 0.999))
        
        # Nakon svakih 100 epoha smanji learning rate za 0.01
        # u originalu: lr_nova = lr_stara - 0.01 * lr_pocetna ----> majsličnije
        g_sch = torch.optim.lr_scheduler.StepLR(opt_g, step_size=100, gamma=0.99)
        dP_sch = torch.optim.lr_scheduler.StepLR(opt_dP, step_size=100, gamma=0.99)
        dB_sch = torch.optim.lr_scheduler.StepLR(opt_dB, step_size=100, gamma=0.99)

        return [opt_g, opt_dP, opt_dB], [g_sch, dP_sch, dB_sch] # optimizers, lr_schedulers