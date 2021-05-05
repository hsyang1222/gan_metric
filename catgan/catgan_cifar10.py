import os, sys

sys.path.append(os.getcwd())

import time

from utils.utility import mkdir_p, generate_image
from utils.plot import plot, flush

import numpy as np

import torch
from torch import nn
from torch import autograd
from torch import optim
import argparse
import csv

import generative_model_score
inception_model_score = generative_model_score.GenerativeModelScore()
inception_model_score.lazy_mode(True)

import torchvision.utils as vutils
from torch.utils.data import TensorDataset, DataLoader
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms

from normal_utils import *


from pprint import pprint

import wandb

#wandb.init(entity="greeksharifa", project='catgan_1')
wandb.init(project='catgan')

parser = argparse.ArgumentParser(description='parse the input options')

parser.add_argument('--name', type=str, default='cifar10',
                    help='name of the experiment. It decides where to store the result and checkpoints')
parser.add_argument('--results_dir', type=str, default='./result', help='folder to store the result')
parser.add_argument('--image_size', type=int, default=32, help='input image size, for cifar10 is 32x32')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--workers', type=int, default=2, help='# of workers to load the dataset')
parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='folder to store the model checkpoint')
parser.add_argument('--noise_dim', type=int, default=100, help='input dim of noise')
parser.add_argument('--dim', type=int, default=64, help='# of filters in first conv layer of both discrim and gen')
parser.add_argument('--data_dir', default=dataroot, help='folder of the dataset')
parser.add_argument('--netG', type=str, default='',
                    help='checkpoints of netG you wish to use in continuing the training')
parser.add_argument('--netD', type=str, default='',
                    help='checkpoints of netD you wish to use in continuing the training')
parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
parser.add_argument('--num_epochs', type=int, default=200, help='# of epochs to train')

opt = parser.parse_args()
wandb.config.update(opt)

dtype = torch.FloatTensor

mkdir_p(os.path.join(opt.results_dir, opt.name))
mkdir_p(os.path.join(opt.checkpoints_dir, opt.name))

import glob

'''
globresult = list(glob.glob('/data/cifar10/**/*', recursive=True))
print("len(globresult): {}", len(globresult))
print('glob[]: \n')
length = min(20, len(globresult))
for i in range(length):
    print(globresult[i])
'''


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(opt.noise_dim, opt.dim * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(opt.dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (opt.dim*8) x 4 x 4
            nn.ConvTranspose2d(opt.dim * 4, opt.dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (opt.dim*4) x 8 x 8
            nn.ConvTranspose2d(opt.dim * 2, opt.dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.dim),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (opt.dim*2) x 16 x 16
            nn.ConvTranspose2d(opt.dim, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 32 x 32
        )
    
    def forward(self, input):
        output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        main = nn.Sequential(
            nn.Conv2d(3, opt.dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),  # 64x16x16
            nn.Conv2d(opt.dim, 2 * opt.dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2 * opt.dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),  # 128x8x8
            nn.Conv2d(2 * opt.dim, 4 * opt.dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4 * opt.dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),  # 256x4x4
            nn.Conv2d(4 * opt.dim, 4 * opt.dim, 4),
            nn.BatchNorm2d(4 * opt.dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),  # 256x1x1
            nn.Conv2d(4 * opt.dim, 10, 1)
        )
        
        self.main = main
        self.softmax = nn.Softmax()
    
    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 10)
        output = self.softmax(output)
        return output


# marginalized entropy
def entropy1(y):
    y1 = autograd.Variable(torch.randn(y.size(1)).type(dtype), requires_grad=True)
    y2 = autograd.Variable(torch.randn(1).type(dtype), requires_grad=True)
    y1 = y.mean(0)
    y2 = -torch.sum(y1 * torch.log(y1 + 1e-6))
    
    return y2


# entropy
def entropy2(y):
    y1 = autograd.Variable(torch.randn(y.size()).type(dtype), requires_grad=True)
    y2 = autograd.Variable(torch.randn(1).type(dtype), requires_grad=True)
    y1 = -y * torch.log(y + 1e-6)
    
    y2 = 1.0 / opt.batch_size * y1.sum()
    return y2

import torchvision.transforms as transforms
import torchvision
dataset = torchvision.datasets.CIFAR10(root='../dataset', #download=True,
                           transform=transforms.Compose([
                               transforms.Resize((opt.image_size,opt.image_size)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
train_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
print(train_loader.dataset)
import hashlib
real_images_info_file_name = inception_model_score.trainloaderinfo_to_hashedname(train_loader)
if os.path.exists('../../inception_model_info/' + real_images_info_file_name) : 
    print("Using exist inception model info from :", real_images_info_file_name)
    inception_model_score.load_real_images_info('../../inception_model_info/' + real_images_info_file_name)
else : 
    inception_model_score.model_to('cuda')

    #put real image
    for each_batch in train_loader : 
        X_train_batch = each_batch[0]
        inception_model_score.put_real(X_train_batch)

    #generate real images info
    inception_model_score.lazy_forward(batch_size=64, device='cuda', real_forward=True)
    inception_model_score.calculate_real_image_statistics()
    #save real images info for next experiments
    inception_model_score.save_real_images_info('../../inception_model_info/' + real_images_info_file_name)
    print("Save inception model info to :", real_images_info_file_name)
    #offload inception_model
    inception_model_score.model_to('cpu')




netG = Generator()
netD = Discriminator()
# wandb.watch(netG)

# continue traning by loading the latest model or the model specified in --netG and --netD
if opt.continue_train:
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    else:
        netG.load_state_dict(torch.load('%s/netG_latest.pth' % (os.path.join(opt.checkpoints_dir, opt.name))))
    
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    else:
        netD.load_state_dict(torch.load('%s/netD_latest.pth' % (os.path.join(opt.checkpoints_dir, opt.name))))

print(netG)
print(netD)

use_cuda = torch.cuda.is_available()

if use_cuda:
    netD = netD.cuda()
    netG = netG.cuda()

one = torch.tensor(1, dtype=torch.float).cuda()
mone = torch.tensor(-1, dtype=torch.float).cuda()

if use_cuda:
    one = one.cuda()
    mone = mone.cuda()

optimizerD = optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.9))

# Dataset iterator # ImageFolder
'''
dataset = dset.CIFAR10(root=opt.data_dir, download=True,
                       transform=transforms.Compose([
                           transforms.Scale(opt.image_size),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                       ]))
'''
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                         shuffle=True, num_workers=int(opt.workers))

print("Start training on %s dataset which contains %d images..." % (opt.name, len(dataset)))

fixed_noise = torch.randn(opt.batch_size, opt.noise_dim, 1, 1).cuda()
real_label = 1
fake_label = 0

iter_idx = 0
# log = open(os.path.join(opt.results_dir, opt.name, 'log.csv'), 'wb')
# log_writer = csv.writer(log, delimiter=',')

print('{:<12s}\t{:<12s}\t{:<12s}\t{:<12s}\t{:<12s}\t{:<12s}\t{:<12s}'
      .format('FID', 'Precision', 'Recall', 'Coverage', 'Density', 'is_r', 'is_f'),
      file=open(normal_opt.outf + '/log.txt', 'w'))

for epoch in range(opt.num_epochs):
    # fake_list = []
    start_time = time.time()
    
    for batch_idx, (real, labels) in enumerate(dataloader):
        ###########################
        # (1) Update D network
        ###########################
        
        # freeze G and update D
        for p in netD.parameters():
            p.requires_grad = True
        for p in netG.parameters():
            p.requires_grad = False
        netD.zero_grad()
        
        #################
        # train with real
        #################
        if use_cuda:
            real = autograd.Variable(real.cuda())
        
        D_real = netD(real)
        # minimize entropy to make certain prediction of real sample
        entorpy2_real = entropy2(D_real)
        entorpy2_real.backward(one, retain_graph=True)
        
        # maximize marginalized entropy over real samples to ensure equal usage
        entropy1_real = entropy1(D_real)
        entropy1_real.backward(mone)
        
        #################
        # train with fake
        #################
        noise = torch.randn(opt.batch_size, opt.noise_dim, 1, 1)
        if use_cuda:
            noise = autograd.Variable(noise.cuda())  # totally freeze netG
        
        fake = netG(noise)
        D_fake = netD(fake)
        
        # fake_list.append(fake.detach().cpu())
        
        # minimize entropy to make uncertain prediction of fake sample
        entorpy2_fake = entropy2(D_fake)
        entorpy2_fake.backward(mone)
        
        D_cost = entropy1_real + entorpy2_real + entorpy2_fake
        optimizerD.step()
        ############################
        # (2) Update G network
        ###########################
        
        # freeze D and update G
        for p in netD.parameters():
            p.requires_grad = False
        for p in netG.parameters():
            p.requires_grad = True
        netG.zero_grad()
        
        noise = torch.randn(opt.batch_size, opt.noise_dim, 1, 1)
        noise = autograd.Variable(noise.cuda())
        fake = netG(noise)
        D_fake = netD(fake)
        
        inception_model_score.put_fake(fake.detach().cpu())
        
        # fool D to make it believe the generated samples are real
        entropy2_fake = entropy2(D_fake)
        entropy2_fake.backward(one, retain_graph=True)
        
        # ensure equal usage of fake samples
        entropy1_fake = entropy1(D_fake)
        entropy1_fake.backward(mone)
        
        G_cost = entropy2_fake + entropy1_fake
        optimizerG.step()
        
        D_cost = D_cost.cpu().data.numpy()
        G_cost = G_cost.cpu().data.numpy()
        entorpy2_real = entorpy2_real.cpu().data.numpy()
        entorpy2_fake = entorpy2_fake.cpu().data.numpy()
        
        # monitoring the loss
        # plot('errD', D_cost, iter_idx)
        # plot('time', time.time() - start_time, iter_idx)
        # plot('errG', G_cost, iter_idx)
        # plot('errD_real', entorpy2_real, iter_idx)
        # plot('errD_fake', entorpy2_fake, iter_idx)
        
        # Save plot every  iter
        # flush(os.path.join(opt.results_dir, opt.name))
        
        # Write losses to logs
        # log_writer.writerow([D_cost[0],G_cost[0],entorpy2_real[0],entorpy2_fake[0]])
        
        # print('D_cos:', type(D_cost), D_cost)
        # print('G_cost:', type(G_cost), G_cost)
        # print('entorpy2_real:', type(entorpy2_real), entorpy2_real)
        # print('entorpy2_fake:', type(entorpy2_fake), entorpy2_fake)
        #
        # print('[{}/{}], errD={:.4f}\terrG={:.4f}\terrD_real={:.4f}errD_fake={:.4f}'
        #       .format(iter_idx, epoch, float(D_cost), float(G_cost), float(entorpy2_real), float(entorpy2_fake)))
        
        # checkpointing the latest model every 500 iteration
        if iter_idx % 500 == 0:
            torch.save(netG.state_dict(), '%s/netG_latest.pth' % (os.path.join(opt.checkpoints_dir, opt.name)))
            torch.save(netD.state_dict(), '%s/netD_latest.pth' % (os.path.join(opt.checkpoints_dir, opt.name)))
        
        iter_idx += 1
    
    #offload all GAN model to cpu and onload inception model to gpu
    netG = netG.to('cpu')
    netD = netD.to('cpu')
    inception_model_score.model_to('cuda')
    
    #generate fake images info
    inception_model_score.lazy_forward(batch_size=64, device='cuda', fake_forward=True)
    inception_model_score.calculate_fake_image_statistics()
    metrics = inception_model_score.calculate_generative_score()
    inception_model_score.clear_fake()
    
    #onload all GAN model to cpu and offload inception model to gpu
    netG = netG.to('cuda')
    netD = netD.to('cuda')
    inception_model_score.model_to('cpu')
    
    
    # generate samples every 2 epochs for surveillance
    
    '''fake_set = torch.cat(fake_list)
    fakeDataset = TensorDataset(fake_set)
    fakeDataloader = DataLoader(fakeDataset, batch_size=opt.batch_size,
                            shuffle=True, num_workers=int(opt.workers))
    mu_2,std_2=calculate_activation_statistics(fakeDataloader,inceptionV3,cuda=True)
    fid_value = calculate_frechet_distance(mu_1, std_1, mu_2, std_2)
    print('fid_value:', type(fid_value), float(fid_value), '|', fid_value)'''
    
    try:
        nsml.report(summary=True, step=epoch, errD=float(D_cost), errG=float(G_cost),
                    entorpy2_real=float(entorpy2_real),
                    entorpy2_fake=float(entorpy2_fake))  # ,        fid_value=float(fid_value))
        
        result = metrics
        
        nsml.report(summary=True, step=epoch,
                    Precision=result['precision'], Recall=result['recall'],
                    Density=result['density'], Coverage=result['coverage'],
                    fid=result['fid'],
                    is_r=result['real_is'],
                    is_f=result['fake_is'])
    except NameError as e:
        print('epoch={}, errD={:.4f}, errG={:.4f}, entorpy2_real={:.4f}, entorpy2_fake={:.4f}'
              .format(epoch, float(D_cost), float(G_cost), float(entorpy2_real), float(entorpy2_fake)))
        
        result = metrics
        
        print('{:<12s}\t{:<12s}\t{:<12s}\t{:<12s}\t{:<12s}\t{:<12s}\t{:<12s}'
              .format('FID', 'Precision', 'Recall', 'Coverage', 'Density', 'real_is', 'fake_is'))
        
        print('{:<12.4f}\t{:<12.4f}\t{:<12.4f}\t{:<12.4f}\t{:<12.4f}\t{:<12.4f}\t{:<12.4f}'
              .format(result['fid'], result['precision'], result['recall'],
                      result['coverage'], result['density'],
                      result['real_is'], result['fake_is'])
              )
        
        print('{:<12.4f}\t{:<12.4f}\t{:<12.4f}\t{:<12.4f}\t{:<12.4f}\t{:<12.4f}\t{:<12.4f}'
              .format(result['fid'], result['precision'], result['recall'],
                      result['coverage'], result['density'],
                      result['real_is'], result['fake_is']),
              file=open(normal_opt.outf + '/log.txt', 'a'))
        
        wandb.log({
            'epoch': epoch,
            'errD': float(D_cost),
            'errG': float(G_cost),
            'entorpy2_real': float(entorpy2_real),
            'entorpy2_fake': float(entorpy2_fake),
            'fid': result['fid'],
            'precision': result['precision'],
            'recall': result['recall'],
            'coverage': result['coverage'],
            'density': result['density'],
            'is_r': result['real_is'],
            'is_f': result['fake_is']
        })
    
    vutils.save_image(real, '%s/images/real_samples.png' % normal_opt.outf, normalize=True)
    fake = netG(fixed_noise)
    vutils.save_image(fake.detach(), '%s/images/fake_samples_epoch_%03d.png' % (normal_opt.outf, epoch), normalize=True)
    
    # if epoch % 2 == 0:
    #     generate_image(epoch, netG, opt)
    
    # do checkpointing every 20 epochs
    if epoch % 20 == 0:
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (os.path.join(opt.checkpoints_dir, opt.name), epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (os.path.join(opt.checkpoints_dir, opt.name), epoch))
