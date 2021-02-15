from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical
import time
#from torchsummary import summary

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | folder | lfw | fake | fresh_folder')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--trajectf', default='result_npy_traj', help='path to generated trajectories per training step')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--imageSize', type=int, default=7, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--genreps', type=int, default=1, help='number of additional generator training iterations per cycle')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='net_weights', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--classes', default='bedroom', help='comma separated list of classes for the lsun data set')

def npy_loader(path):
    sample = torch.from_numpy(np.transpose(np.load(path), (2, 0, 1)))#.reshape((2, opt.imageSize, 6)))
    return sample

opt = parser.parse_args()
print(opt)



try:
    os.makedirs(opt.outf)
except OSError:
    pass
try:
    os.makedirs(opt.trajectf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset == 'fresh_folder':
    # folder dataset
    dataset = dset.DatasetFolder(root=opt.dataroot,
                                 loader=npy_loader,
                                 extensions='.npy')
    nc=2
    
elif  opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    nc=3
elif opt.dataset == 'lsun':
    classes = [ c + '_train' for c in opt.classes.split(',')]
    dataset = dset.LSUN(root=opt.dataroot, classes=classes,
                        transform=transforms.Compose([
                            transforms.Resize(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
    nc=3
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    nc=3

elif opt.dataset == 'mnist':
        dataset = dset.MNIST(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                           ]))
        nc=1

elif opt.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                            transform=transforms.ToTensor())
    nc=3

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
genreps = int(opt.genreps)


def _cov(mat, rowvar=False):
    '''Estimate a covariance matrix given data.
    https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/2
    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if mat.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if mat.dim() < 2:
        mat = mat.view(1, -1)
    if not rowvar and mat.size(0) != 1:
        mat = mat.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (mat.size(1) - 1)
    mat2 = mat - torch.mean(mat, dim=1, keepdim=True)
    matt = mat2.t()  # if complex: mt = m.t().conj()
    return fact * mat2.matmul(matt).squeeze()


def diversity_metrics(matrix):
    #find covariance matrix
    new_mat = matrix.view(opt.batchSize, -1)
#    print('SIZE: ', new_mat.size(), new_mat.min(), new_mat.max())
    covariance_mat = _cov(new_mat, rowvar=True)
#    print('SIZE 2: ', covariance_mat.size())

    eigvals_cov, vecs = torch.symeig(covariance_mat, eigenvectors=True, upper=False)
#    print('SIZE 2: ', eigvals_cov.size())
#    determinant_cov = torch.cholesky(covariance_mat_for_det).diag().prod()
#    vals=[]
#    for _ in range(opt.batchSize):
#        vals.append(matrix[_, :, :, :].reshape)
#    np.cov(vals)
    max_eig = torch.max(eigvals_cov)
    condition_number = torch.max(eigvals_cov)/torch.min(eigvals_cov)
    return torch.tensor(100000000000)/max_eig, torch.abs(condition_number-1)/torch.tensor(1000000000000000)#torch.abs(condition_number-1)

def euclidean_dist(eigen_values):
    roll = opt.batchSize#eigen_values.size()[0]
    total_dist = 0
    for i in range(roll):
        for j in range(roll):
            total_dist += torch.norm(eigen_values[i, :] - eigen_values[j, :], float("inf"))
#    print('Euclidean: ', 3*torch.exp(-1/total_dist))
#    return 3*torch.exp(-1/total_dist)
    return 100000000/total_dist

def fitness_approx(eigen_values):
    roll = opt.batchSize#eigen_values.size()[0]
    total_fitness = 0
    for i in range(roll):
        total_fitness += torch.abs(torch.max(eigen_values[i, :])/torch.min(eigen_values[i, :]))
    return total_fitness/roll


def entropy_fitness(eigen_values):
    roll = opt.batchSize#eigen_values.size()[0]
    total_fitness = 0
    for i in range(roll):
        total_fitness += Categorical(probs = eigen_values[i, :]).entropy()
    return total_fitness/roll    

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def make_int(success_rates):
    for i in range(opt.batchSize):
        if success_rates[i] >=0.5:
            success_rates[i] = 1
        else:
            success_rates[i] = 0
    return success_rates

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        print(np.shape(x))
        return x

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
#            PrintLayer(),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),            
#            PrintLayer(),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
#            PrintLayer(),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
#            PrintLayer(),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 3, 1, 0, bias=False),
            nn.Tanh(),
#            PrintLayer(),
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input.type('torch.DoubleTensor'))
        return output.type('torch.DoubleTensor')


netG = (Generator(ngpu).to(device)).type('torch.DoubleTensor')
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=False),
#            PrintLayer(),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=False),
#            PrintLayer(),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=False),
#            PrintLayer(),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=False),
#            PrintLayer(),
            # state size. (ndf*8) x 4 x 4
#            nn.Conv2d(ndf * 8, 1, 3, 1, 0, bias=False),
#            nn.Sigmoid()
        )
        self.x = nn.Linear(2048, 1)
        self.sigm = nn.Sigmoid()

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input.type('torch.DoubleTensor'))
            output = self.sigm(self.x(output.view(opt.batchSize, -1)))
        return output.view(-1, 1).squeeze(1)


netD = (Discriminator(ngpu).to(device)).type('torch.DoubleTensor')
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)
#summary(netD, (100, 2, 7, 7))

class Discriminator_eigen(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator_eigen, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(1, ndf, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=False),
#            PrintLayer(),
#            PrintLayer(),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=False),
#            PrintLayer(),
#            PrintLayer(),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=False),
#            PrintLayer(),
#            PrintLayer(),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=False),
#            PrintLayer(),
#            PrintLayer(),
            # state size. (ndf*8) x 4 x 4
#            nn.Conv2d(ndf * 8, 1, 3, 1, 0, bias=False),
#            nn.Sigmoid()
        )
        self.x = nn.Linear(2560, 1)
        self.sigm = nn.Sigmoid()

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            input = input.view(opt.batchSize, 1, 35, 1)
            output = self.main(input.type('torch.DoubleTensor'))
#            print('D2 size: ', output.size())
            output = self.sigm(self.x(output.view(opt.batchSize, -1)))
        return output.view(-1, 1).squeeze(1)


netD_2 = (Discriminator_eigen(ngpu).to(device)).type('torch.DoubleTensor')
netD_2.apply(weights_init)
print(netD_2)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 200, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(200, 10, kernel_size=3, stride=2, padding=1)
        self.fc2 = nn.Linear(160, 35)

    def forward(self, x):      
#        print('Net input: ', np.shape(x))
        x = F.relu(self.conv1(x))
#        print('Net conv1(x): ', np.shape(x))
        x = F.relu(self.conv2(x))
#        print('Net conv2(x): ', np.shape(x))
        x = self.fc2(x.view(opt.batchSize, -1))        
#        print('Net fc2(x): ', np.shape(x))
        return x.view(opt.batchSize,-1).type('torch.DoubleTensor')

converter = (Net()).type('torch.DoubleTensor')
#converter.float()
converter.load_state_dict(torch.load('./model_0_50k_epochs.pth'))
#converter.load_state_dict(torch.load('./model_0_diag_50k_20error.pth'))

class SR(nn.Module):
    def __init__(self):
        super(SR, self).__init__()
        self.conv1 = nn.Conv2d(2, 200, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(200, 2, kernel_size=3, stride=2, padding=1)
        self.fc2 = nn.Linear(32, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
#        print('SR input: ', np.shape(x))
        x = F.relu(self.conv1(x))
#        print('SR conv1: ', np.shape(x))
        x = F.relu(self.conv2(x))
#        print('SR conv2: ', np.shape(x))
        x = self.fc2(x.view(opt.batchSize, -1))
#        print('SR fc2: ', np.shape(x))
        x = self.out_act(x)
#        print('SR out_act: ', np.shape(x))
        return x.view(opt.batchSize,-1).type('torch.DoubleTensor')

sr = (SR()).type('torch.DoubleTensor')
#converter.float()
sr.load_state_dict(torch.load('./model_0_SR.pth'))


criterion = nn.BCELoss()

fixed_noise = (-2 * torch.rand(10, opt.batchSize, nz, 1, 1) + 1).type('torch.DoubleTensor')#torch.randn(opt.batchSize, nz, 1, 1, device=device).type('torch.DoubleTensor')
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD_2 = optim.Adam(netD_2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu = data[0].to(device)
#        print('input: ', real_cpu.size())
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, device=device)
        output = netD(real_cpu.type('torch.DoubleTensor'))
#        print('D output: ', output.size())
        errD_real = criterion(output.type('torch.DoubleTensor'), label.type('torch.DoubleTensor'))
        errD_real.backward()
        D_x = output.mean().item()

        netD_2.zero_grad()
        real_cpu = data[0].to(device)
#        print('REAL size: ', real_cpu.size())
        pre_processed = converter(real_cpu.type('torch.DoubleTensor'))
        batch_size = real_cpu.size(0)
#        print('pre-processed: ', pre_processed.size())
        label = torch.full((batch_size,), real_label, device=device)
        output = netD_2(pre_processed.type('torch.DoubleTensor'))
#        print('D2 output: ', output.size())
#        summary(output)
        errD_2_real = criterion(output.type('torch.DoubleTensor'), label.type('torch.DoubleTensor'))
        errD_2_real.backward()
        D_2_x = output.mean().item()

        # train with fake
        noise = (-2 * torch.rand(batch_size, nz, 1, 1) + 1).type('torch.DoubleTensor')#torch.randn(batch_size, nz, 1, 1, device=device).type('torch.DoubleTensor')#
        fake = netG(noise.type('torch.DoubleTensor'))
#        summary(fake)
#        print('G output: ', fake.detach().size())
        label.fill_(fake_label)
        
        output = netD(fake.detach())
        errD_fake = criterion(output.type('torch.DoubleTensor'), label.type('torch.DoubleTensor'))
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        pre_fake = converter(fake.type('torch.DoubleTensor'))
        output = netD_2(pre_fake.detach())
        errD_2_fake = criterion(output.type('torch.DoubleTensor'), label.type('torch.DoubleTensor'))
        errD_2_fake.backward()
        D_2_G_z1 = output.mean().item()
        errD_2 = errD_2_real + errD_2_fake
        optimizerD_2.step()
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        output2 = netD_2(pre_fake)
        
        temp = pre_fake
        torch.autograd.set_detect_anomaly(True)
        max_eig, cond_number = diversity_metrics(temp)
#        print('Max Eig: ', max_eig)
#        print('condition_number: ', cond_number)
#        print('determinant: ', determinant_cov)
        
        diversity_penalty = euclidean_dist(pre_fake);
        fitness_penalty = fitness_approx(pre_fake);
        fitness_entropy = entropy_fitness(pre_fake);
        success_rates = sr(fake.type('torch.DoubleTensor'))
#        success_rates_int = make_int(success_rates)
#        print(success_rates_int.detach())
        
        errG_1 = criterion(output.type('torch.DoubleTensor'), label.type('torch.DoubleTensor'))
        errG_2 = criterion(output2.type('torch.DoubleTensor'), label.type('torch.DoubleTensor'))
        if epoch < 10:
            errG = 0.5*errG_1 + 0.5*errG_2 + (1 - torch.mean(success_rates))#+ 0.1*cond_number + 0.1*max_eig
        else:
            errG = 0.5*errG_1 + 0.5*errG_2 + (1+0.6*(epoch-10)/10)*(1 - torch.mean(success_rates)) + 2*fitness_entropy #+ 0.2*cond_number + 0.35*max_eig 
        errG.backward(retain_graph=True)
        D_G_z2 = output.mean().item()
        optimizerG.step()
        
        netG.zero_grad()
        for k in range(genreps):
            noise = (-2 * torch.rand(batch_size, nz, 1, 1) + 1).type('torch.DoubleTensor')#torch.randn(batch_size, nz, 1, 1, device=device).type('torch.DoubleTensor')#
            start = time.time()
            fake = netG(noise.type('torch.DoubleTensor'))
            end = time.time()
            print('time: ', end - start)
            pre_fake = converter(fake.type('torch.DoubleTensor'))
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            output2 = netD_2(pre_fake)
            diversity_penalty = euclidean_dist(pre_fake);
            fitness_penalty = fitness_approx(pre_fake)
            fitness_entropy = entropy_fitness(pre_fake);
            success_rates = sr(fake.type('torch.DoubleTensor'))
#            success_rates_int = make_int(success_rates)
#            print(success_rates_int.detach())
            errG_1 = criterion(output.type('torch.DoubleTensor'), label.type('torch.DoubleTensor'))
            errG_2 = criterion(output2.type('torch.DoubleTensor'), label.type('torch.DoubleTensor'))
            if epoch < 10:
                errG = 0.5*errG_1 + 0.5*errG_2 + (1 - torch.mean(success_rates))#+ 0.1*cond_number + 0.1*max_eig
            else:
                errG = 0.5*errG_1 + 0.5*errG_2  + (1+0.6*(epoch-10)/10)*(1 - torch.mean(success_rates)) + 2*fitness_entropy#+ 0.2*cond_number + 0.35*max_eig 
            errG.backward(retain_graph=True)
            D_G_z2 = output.mean().item()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f LossD_2: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f, Div: %.4f, %.4f, %.4f, Fitness: %.4f, SR: %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.item(), errD_2.item(), errG.item(), D_x, D_G_z1, D_G_z2, diversity_penalty, cond_number, max_eig, fitness_penalty/1000, torch.mean(success_rates)))
        if i % 1800 == 0:
            vutils.save_image(real_cpu[:, 0:1, :, :],
                    '%s/real_samples.png' % opt.outf,
                    normalize=True)
            for n in range(10):
                fake = netG(fixed_noise[n, :, :, :, :])
                pre_fake = converter(fake.type('torch.DoubleTensor'))
#                print('FAKE: ', fake.size(), fake.min(), fake.max())
                np_fake = fake.detach().numpy()
                np.save('./{}/fake_epoch{}_noise{}.npy'.format(opt.trajectf, epoch, n), fake.detach())
                np.save('./{}/eig_fake_epoch{}_noise{}.npy'.format(opt.trajectf, epoch, n), pre_fake.detach())
                
#                vutils.save_image(fake.detach()[:, 0:1, :, :],
#                        '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
#                        normalize=True)
            print('Range of the synthetic data (sanity check): ', np.min(np_fake), np.max(np_fake))

#     do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD_2.state_dict(), '%s/netD_2_epoch_%d.pth' % (opt.outf, epoch))

print('Training concluded. Please check: \n - {} folder fpr the trained network parameters, \n - {} for the generated trajectories per epoch in npy format.'.format(opt.outf, opt.trajectf))
print('Please remember to convert the trajectories into json before submitting them to ARDL simulator.')
