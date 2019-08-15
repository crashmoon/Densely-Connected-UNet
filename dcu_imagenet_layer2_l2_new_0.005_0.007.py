from __future__ import print_function, division
import torch.nn as nn
import torch.nn.modules
import numpy as np

import plot

import scipy.misc
from scipy.misc import imsave

import torch
from torch.autograd import Variable
import torch.autograd as autograd

from os.path import join
from glob import glob
from torch.utils.data import Dataset, DataLoader
from torch.nn.modules.utils import _pair

import math
import os
from skimage import io, transform
from skimage.transform import resize
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import time
import random
#import cv2

#os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'


class Param:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device = torch.device('cuda:1')
    image_size = 128
    ############## res ###################
    layer_size = [2, 2, 2, 2, 2, 2]
    resnet_channel = 32
    ############# dsu ###################
    grouth       = np.array([16, 32, 64, 128, 256, 256])         # small one
    begin_grouth = np.array([ 3, 32, 64, 128, 256, 256])         #
    #grouth       = np.array([32, 64, 128, 256, 512, 512])       # big one
    #begin_grouth = np.array([ 3, 64, 128, 256, 512, 512])       #
    num_out = 3
    num_unet = 2 
    #num_unet = 2 
    #num_unet = 8                                             # number of unet
    #######################################
    n_critic = 1
    batch_size = 128
    gan_weight = 1.0
    l2_weight = 1000.0
    weight_decay = 1e-3
    G_learning_rate = 0.007
    D_learning_rate = 0.005 # 1e-3
    #gp_learning_rate = 0.01  # 1e-3
    #max_gradient_penalty = 0.1
    gp_lamada = 10.0 #gp_learning_rate / D_learning_rate
    out_path = '/home/shenling/inpainting/dsu_imagenet_layer2_l2_new_0.005_0.007/'


################################################################
###################  model   ##################################
#############################################################

class LayerNorm(nn.Module):
    def __init__(self):
        return super(LayerNorm, self).__init__()

    def forward(self, input):
        x = input.view(input.size(0), -1)
        input_mean = torch.mean(x, 1)
        input_var = torch.var(x, 1)
        input_var = input_var.view(-1, 1, 1, 1).expand_as(input)
        input_mean = input_mean.view(-1, 1, 1, 1).expand_as(input)
        input = (input - input_mean) * torch.rsqrt(input_var)
        return input


def conv_down(dim_in, dim_out):
    return nn.Sequential(
        nn.BatchNorm2d(dim_in),
        nn.LeakyReLU(0.2),
        nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=2, padding=1, bias=False)
    )


def conv_up(dim_in, dim_out):
    return nn.Sequential(
        nn.BatchNorm2d(dim_in),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(dim_in, dim_out, 4, 2, 1, bias=False)
    )


def conv_horizon(dim_in, dim_out):
    return nn.Sequential(
        nn.BatchNorm2d(dim_in),
        nn.LeakyReLU(0.2),
        nn.Conv2d(dim_in, dim_out, 3, 1, 1, bias = False)
    )


class DenseUnet(nn.Module):
    def __init__(self, num_unet = Param.num_unet, num_classes = Param.num_out):
        super(DenseUnet, self).__init__()
        self.num_unet = num_unet
        ldim = []
        rdim = []
        ldim.append(Param.begin_grouth)
        rdim.append(ldim[-1]+Param.grouth)
        num_of_layer = len(ldim[-1]) - 1
        for step in range(1, self.num_unet):
            ldim.append(ldim[-1]+Param.grouth*2)
            rdim.append(rdim[-1]+Param.grouth*2)
        self.conv = nn.ModuleList([nn.ModuleList() for i in range(self.num_unet)])
        self.deconv = nn.ModuleList([nn.ModuleList() for i in range(self.num_unet)])
        self.mid_conv = nn.ModuleList()
        self.end_conv = nn.ModuleList()
        self.out_conv = nn.Conv2d(rdim[-1][0]+Param.grouth[0],num_classes,1,bias=False)
        for step in range(self.num_unet):
            for id in range(num_of_layer):
                self.conv[step].append(conv_down(ldim[step][id], Param.grouth[id+1]))
                self.deconv[step].append(conv_up(rdim[step][id+1], Param.grouth[id]))
            self.mid_conv.append(conv_horizon(ldim[step][-1], Param.grouth[-1]))
            self.end_conv.append(conv_horizon(rdim[step][0], Param.grouth[0]))
        ## weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal(m.weight.data, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        net = [x, None, None, None, None, None]
        num_of_layer = len(net) - 1
        for step in range(self.num_unet):
            for id in range(num_of_layer):
                out = self.conv[step][id](net[id])
                if(net[id+1] is None):
                    net[id+1] = out
                else:
                    net[id+1] = torch.cat((net[id+1],out), 1)
            out = self.mid_conv[step](net[-1])
            net[-1] = torch.cat((net[-1],out), 1)
            for id in reversed(range(num_of_layer)):
                out = self.deconv[step][id](net[id+1])
                net[id] = torch.cat((net[id],out), 1)
            out = self.end_conv[step](net[0])
            net[0] = torch.cat((net[0],out), 1)
        out = self.out_conv(net[0])
        out = F.sigmoid(out)
        return out


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, dim_in, dim_out, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(dim_in, dim_out, stride)
        self.bn1 = nn.BatchNorm2d(dim_in)
        # self.bn1 = nn.InstanceNorm2d(dim_in)
        # self.bn1 = LayerNorm()
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(dim_out, dim_out)
        self.bn2 = nn.BatchNorm2d(dim_out)
        # self.bn2 = nn.InstanceNorm2d(dim_out)
        # self.bn2 = LayerNorm()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        return out


class ResNet(nn.Module):
    def __init__(self, layers=Param.layer_size):
        super(ResNet, self).__init__()
        self.conv0 = nn.Conv2d(3, Param.resnet_channel, kernel_size=1, stride=1, bias=False)
        self.bn0 = nn.BatchNorm2d(Param.resnet_channel)
        # self.bn0 = LayerNorm()
        # self.bn0 = nn.InstanceNorm2d(Param.resnet_channel)
        self.layer1 = self._make_layer(Param.resnet_channel, Param.resnet_channel, layers[0], stride=1)
        self.layer2 = self._make_layer(Param.resnet_channel, Param.resnet_channel * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(Param.resnet_channel * 2, Param.resnet_channel * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(Param.resnet_channel * 4, Param.resnet_channel * 8, layers[3], stride=2)
        self.layer5 = self._make_layer(Param.resnet_channel * 8, Param.resnet_channel * 16, layers[4], stride=2)
        self.layer6 = self._make_layer(Param.resnet_channel * 16, Param.resnet_channel * 32, layers[5], stride=2)
        self.avgpool = nn.AvgPool2d(4)
        #self.fc = nn.Linear(Param.resnet_channel * 2 * 2 * 2 * 2 * 2, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # nn.init.kaiming_normal(m.weight.data, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                # elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_layer(self, dim_in, dim_out, layer_num, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.BatchNorm2d(dim_in),
                # nn.InstanceNorm2d(dim_out)
                # LayerNorm()
                nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=stride, bias=False)
            )
        layers = []
        layers.append(BasicBlock(dim_in, dim_out, stride, downsample))
        for i in range(1, layer_num):
            layers.append(BasicBlock(dim_out, dim_out))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = F.relu(x)          
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #x = self.fc(x)
        x = x.mean(1)
        return -x.view(-1)

##########################################################################
#################### model end ###########################################
##########################################################################

def calc_gradient_penalty(netD, real_data, fake_data):
    # print real_data.size()
    alpha = torch.rand(Param.batch_size, 1, 1, 1).to(Param.device)
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates.requires_grad_(True)
    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(Param.device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gs = gradients.view(Param.batch_size, -1)
    gradient_penalty = ((gs.norm(2, dim=1) - 1.0) ** 2).mean()
    return gradient_penalty


class ImageNetData(object):
    def __init__(self, csv_file, trans=None):
        self.lines = pd.read_csv(csv_file)
        self.trans = trans

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        image_pos = self.lines.ix[idx, 0]
        image = io.imread(image_pos)
        image = image.astype(np.float)
        h, w = image.shape[:2]
        the_min_length = random.randint(150, 200) #150.0
        if (h < w):
            factor = h / the_min_length
            w = w / factor
            h = the_min_length
        else:
            factor = w / the_min_length
            h = h / factor
            w = the_min_length
        image = transform.resize(image, (int(h), int(w), 3))
        image_id = self.lines.ix[idx, 1]
        sample = {'image': image, 'id': image_id}
        if self.trans is not None:
            sample = self.trans(sample)
        return sample


class ParisData(object):
    def __init__(self, csv_file, trans=None):
        self.lines = pd.read_csv(csv_file)
        self.trans = trans

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        image_pos = self.lines.ix[idx, 0]
        image = io.imread(image_pos)
        image = image.astype(np.float)
        h, w = image.shape[:2]
        #the_min_length = 350
        the_min_length = random.randint(150, 400)
        if (h < w):
            factor = h / the_min_length
            w = w / factor
            h = the_min_length
        else:
            factor = w / the_min_length
            h = h / factor
            w = the_min_length
        image = transform.resize(image, (int(h), int(w), 3))
        image_id = self.lines.ix[idx, 1]
        sample = {'image': image, 'id': image_id}
        if self.trans is not None:
            sample = self.trans(sample)
        return sample


class RandCrop(object):
    def __call__(self, sample):
        image = sample['image']
        image_id = sample['id']
        h, w = image.shape[:2]
        sx = random.randint(0, h - Param.image_size)
        sy = random.randint(0, w - Param.image_size)
        image = image[sx:(sx + Param.image_size), sy:(sy + Param.image_size)]
        image = image.transpose((2, 0, 1))
        if (random.randint(0, 1)):
            image = image[:, :, ::-1]
        image /= 255.0
        image_trans = np.array(image)
        return {'image': torch.FloatTensor(image_trans), 'id': torch.Tensor([image_id])}


def inf_get(train):
    while (True):
        for x in train:
            yield x['image']


def destroy(image, crop_size=64):
    re = image.clone()
    '''
    re[:, :, int((Param.image_size - crop_size) / 2):int((Param.image_size - crop_size) / 2 + crop_size),
    int((Param.image_size - crop_size) / 2):int((Param.image_size - crop_size) / 2 + crop_size)] = torch.zeros(
        Param.batch_size, 3, crop_size, crop_size).to(Param.device)
    '''
    re[:, 0, int((Param.image_size - crop_size) / 2):int((Param.image_size - crop_size) / 2 + crop_size),
    int((Param.image_size - crop_size) / 2):int((Param.image_size - crop_size) / 2 + crop_size)] = torch.zeros(
        Param.batch_size, crop_size, crop_size).fill_(0.45703125).to(Param.device)
    re[:, 1, int((Param.image_size - crop_size) / 2):int((Param.image_size - crop_size) / 2 + crop_size),
    int((Param.image_size - crop_size) / 2):int((Param.image_size - crop_size) / 2 + crop_size)] = torch.zeros(
        Param.batch_size, crop_size, crop_size).fill_(0.40625).to(Param.device)
    re[:, 2, int((Param.image_size - crop_size) / 2):int((Param.image_size - crop_size) / 2 + crop_size),
    int((Param.image_size - crop_size) / 2):int((Param.image_size - crop_size) / 2 + crop_size)] = torch.zeros(
        Param.batch_size, crop_size, crop_size).fill_(0.48046875).to(Param.device)
    
    return re


class Net_G(nn.Module):
    def __init__(self):
        super(Net_G, self).__init__()
        self.unet = DenseUnet()

    def forward(self, x):
        out = self.unet(x)
        return out


class Net_D(nn.Module):
    def __init__(self):
        super(Net_D, self).__init__()
        self.cnn = ResNet()

    def forward(self, x):
        out = self.cnn(x)
        return out


def save_image_plus(x, save_path):
    x = (255.99 * x).astype('uint8')
    x = x.transpose(0, 1, 3, 4, 2)
    nh, nw = x.shape[:2]
    h = x.shape[2]
    w = x.shape[3]
    img = np.zeros((h * nh, w * nw, 3))
    for i in range(nh):
        for j in range(nw):
            img[i * h:i * h + h, j * w:j * w + w] = x[i][j]
    imsave(save_path, img)


def cal_tv(image):
    temp = image.clone()
    temp[:, :, :Param.image_size - 1, :] = image[:, :, 1:, :]
    re = ((image - temp) ** 2).mean()
    temp = image.clone()
    temp[:, :, :, :Param.image_size - 1] = image[:, :, :, 1:]
    re += ((image - temp) ** 2).mean()
    return re


def main():

    mask = torch.ones(Param.batch_size, 3, 128, 128).to(Param.device)
    mask[:, :, 32:32 + 64, 32:32 + 64] = torch.zeros(Param.batch_size, 3, 64, 64).to(Param.device)

    netG = Net_G().to(Param.device)
    netD = Net_D().to(Param.device)

    netG = torch.nn.DataParallel(netG)
    netD = torch.nn.DataParallel(netD)

    


    netG.load_state_dict(torch.load('/home/shenling/inpainting/dsu_imagenet_layer2_l2_new_0.005_0.007/netG_29999.pickle'))
    netD.load_state_dict(torch.load('/home/shenling/inpainting/dsu_imagenet_layer2_l2_new_0.005_0.007/netD_29999.pickle'))
    #opt_G = optim.Adam(netG.parameters(), lr=Param.G_learning_rate, betas = (0.,0.9), weight_decay=Param.weight_decay)
    #opt_D = optim.Adam(netD.parameters(), lr=Param.D_learning_rate, betas = (0.,0.9), weight_decay=Param.weight_decay)
    opt_G = optim.SGD(netG.parameters(), lr=Param.G_learning_rate, weight_decay=Param.weight_decay)
    opt_D = optim.SGD(netD.parameters(), lr=Param.D_learning_rate, weight_decay=Param.weight_decay)
    #opt_D = optim.RMSprop(netD.parameters(), lr=Param.D_learning_rate, weight_decay=Param.weight_decay, eps=1e-6)
    #opt_gp_D = optim.RMSprop(netD.parameters(), lr=Param.gp_learning_rate, weight_decay=Param.weight_decay, eps=1e-6)
    #opt_gp_D = optim.SGD(netD.parameters(), lr=Param.gp_learning_rate, weight_decay=Param.weight_decay)

    #trainset = ParisData('paris.csv', RandCrop())
    trainset = ImageNetData('imagenet.csv', RandCrop()) 
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=Param.batch_size, shuffle=True, num_workers=4, drop_last=True)
    train_data = inf_get(train_loader)

    epoch = 30000
    maxepoch = 2000000
    #bce_loss = nn.BCELoss()

    ##########################
    ###########################

    while (epoch < maxepoch):
        start_time = time.time()
        ##########
        # step D #
        ##########
        for p in netD.parameters():
            p.requires_grad = True

        for D_step in range(Param.n_critic):
            real_data = train_data.next()
            real_data = real_data.to(Param.device)
            destroy_data_64 = destroy(real_data, 64)
            fake_data = netG(destroy_data_64).detach()

            netD.zero_grad()
            D_fake = netD(fake_data).mean()
            D_real = netD(real_data).mean()
            gradient_penalty = calc_gradient_penalty(netD, real_data.detach(), fake_data.detach())
            D_loss = D_fake - D_real
            D_tot = D_loss + gradient_penalty * Param.gp_lamada
            D_tot.backward()
            opt_D.step()

        ##########
        # step G #
        ##########
        for p in netD.parameters():
            p.requires_grad = False
        netG.zero_grad()

        real_data = train_data.next()
        real_data = real_data.to(Param.device)
        destroy_data_64 = destroy(real_data, 64)
        fake_data = netG(destroy_data_64)
        
        l2_loss = ((fake_data - real_data) ** 2).mean()
        #l2_loss = (((fake_data - real_data) * mask) ** 2).mean()
        

        D_fake = netD(fake_data).mean()
        G_loss = l2_loss * Param.l2_weight - D_fake * Param.gan_weight
        G_loss.backward()
        opt_G.step()

        # Write logs and save samples
        os.chdir(Param.out_path)
        plot.plot('train-D-cost', D_loss.item() )
        plot.plot('time', time.time() - start_time)
        plot.plot('train-G-cost', G_loss.item())
        plot.plot('l2-cost', l2_loss.item())
        plot.plot('G-fake-cost', -D_fake.item())
        plot.plot('gp', gradient_penalty.item())

        if epoch % 100 == 99:
            #  real_data = train_data.next()
            out_image = torch.cat(
                (
                    fake_data.data.view(Param.batch_size, 1, 3, Param.image_size, Param.image_size),
                    destroy_data_64.data.view(Param.batch_size, 1, 3, Param.image_size, Param.image_size),
                    real_data.data.view(Param.batch_size, 1, 3, Param.image_size, Param.image_size)
                ),
                1
            )
            # out_image.transpose(0,1,3,4,2)
            save_image_plus(out_image.cpu().numpy(), Param.out_path + 'train_image_new_{}.jpg'.format(epoch))

        if (epoch < 50) or (epoch % 100 == 99):
            plot.flush()
        plot.tick()

        if epoch % 5000 == 4999:
            torch.save(netD.state_dict(), Param.out_path + 'netD_{}.pickle'.format(epoch))
            torch.save(netG.state_dict(), Param.out_path + 'netG_{}.pickle'.format(epoch))
            # opt_D.param_groups[0]['lr'] /= 10.0
            # opt_G.param_groups[0]['lr'] /= 10.0
        epoch += 1


if __name__ == '__main__':
    main()















