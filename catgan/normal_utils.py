from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import  TensorDataset, DataLoader

import numpy as np

import __main__ as main
print(main.__file__)

from datetime import datetime
import os

dataroot = ''

try:
    import nsml
    from nsml import DATASET_PATH, SESSION_NAME
    print('DATASET_PATH, SESSION_NAME:', DATASET_PATH, '\n', SESSION_NAME)
    dataroot = os.path.join(DATASET_PATH, 'train/')
except ImportError:
    dataroot = '/data/dataset/cifar10/'


import easydict
args = easydict.EasyDict({
    'dataset':'cifar10',
    'dataroot':dataroot,  # '/data/dataset/'
    'workers':2,
    'batchSize':64,
    'imageSize':64,
    'nz':100,
    'ngf':64,
    'ndf':64,
    'niter':25,
    'lr':0.0002,
    'beta1':0.5,
    'cuda':True,
    'dry_run':False,
    'ngpu':1,
    'netG':'',
    'netD':'',
    'netE':'',
    'netZ':'',
    'manualSeed':None,
    'classes':None,
    'outf':'result/' + main.__file__.split('.')[0] + '_' + str(datetime.today().month) + '_' + str(datetime.today().day) + '_' + str(datetime.today().hour),
    'n_show': 5,
})



#opt = parser.parse_args()
normal_opt = args
print(normal_opt)
os.makedirs(normal_opt.outf, exist_ok=True)
os.makedirs(normal_opt.outf + '/images/', exist_ok=True)
os.makedirs(normal_opt.outf + '/model/', exist_ok=True)


device = torch.cuda.device("cuda" if normal_opt.cuda else "cpu")

def calculate_activation_statistics(dataloader,model,batch_size=128, dims=2048,
                                    cuda=False):
    model.eval()
    act=np.empty((len(dataloader), dims))

    pred_list = []
    for data in dataloader :
        batch=data[0].cuda()
        pred = model(batch)[0]
        pred_list.append(pred.detach().cpu())

    pred = torch.cat(pred_list)

    # If model output is not scalar, apply global spatial average pooling.
    # This happens if you choose a dimensionality not equal 2048.
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

    act= pred.cpu().data.numpy().reshape(pred.size(0), -1)

    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


from scipy import linalg
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2


    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))


    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_fretchet(images_real,images_fake,model):
    mu_1,std_1=calculate_activation_statistics(images_real,model,cuda=True)
    mu_2,std_2=calculate_activation_statistics(images_fake,model,cuda=True)

    """get fretched distance"""
    fid_value = calculate_frechet_distance(mu_1, std_1, mu_2, std_2)
    return fid_value

def torchlist_to_dataloader(fake_list) :
    fake_set = torch.cat(fake_list)
    fakeDataset = TensorDataset(fake_set)
    fakeDataloader = DataLoader(fakeDataset, batch_size=normal_opt.batchSize,
                                shuffle=True, num_workers=int(normal_opt.workers))
    return fakeDataloader
