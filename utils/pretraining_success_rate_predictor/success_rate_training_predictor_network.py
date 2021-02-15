#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 14:27:38 2020

@author: marija
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import matplotlib.pyplot as plt
import os

'''This is an exmple of a pre-training the Success Rate Predictor.
   NB! The trained version is already provided in the root directory.'''

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='train', help = 'please provide the .npy arrays of training and testing trajectory parameters with corresponding eigen-values')
parser.add_argument('--target_dir', default='trained_SR', help='path to converter net parameters over training epochs')
parser.add_argument('--num_of_nets', type=int, default = 1, help='number of converters to train to choose the best out of')
parser.add_argument('--batchSize', type=int, default=100, help='input half-batch size')
parser.add_argument('--niter', type=int, default=1000, help='number of epochs to train for')
opt = parser.parse_args()
print(opt)


try:
    os.mkdir(opt.target_dir)
except:
    print(str(opt.target_dir) + 'already exists.')

folder_name_ = opt.target_dir
B = opt.batchSize#10 # (half) batch size

loss_matrix = np.zeros((opt.niter, opt.num_of_nets))
loss_check_matrix = np.zeros((opt.niter, opt.num_of_nets))
loss_check_matrix_test = np.zeros((opt.niter))
for j in range(opt.num_of_nets):
    print('network # ', j, M)
    nodes_train = torch.from_numpy((np.load('./{}/train_traj.npy'.format(opt.train_dir))))#.reshape((1440, 84)))
    distances_train = torch.from_numpy(np.load('./{}/train_label.npy'.format(opt.train_dir)))#/200000
    nodes_test = torch.from_numpy((np.load('./{}/test_traj.npy'.format(opt.train_dir))))#.reshape((360, 84)))
    eigens_test = torch.from_numpy(np.load('./{}/test_label.npy'.format(opt.train_dir)))#/200000

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
    #       input channels, number of kernels, stride, kernel size
            self.conv1 = nn.Conv2d(2, 2*B, kernel_size=3, stride=1, padding=1)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv2d(2*B, 2, kernel_size=3, stride=2, padding=1)
            self.relu2 = nn.ReLU()
            self.fc2 = nn.Linear(32, 1)
            self.out_act = nn.Softmax()

    
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.fc2(x.view(B, -1))
            x = self.out_act(x)
            return x.view(B,-1)

    model = Net()
    model.float()
    loss_func = nn.BCELoss()#torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.0001) #lr=1e-4)

    
    for epoch in range(opt.niter):
        batch_idx = np.random.choice(40000, B, replace=False)
        data = nodes_train[batch_idx, :]
        eigens = distances_train[batch_idx, :].type('torch.DoubleTensor')

    
        optimizer.zero_grad()
        
        cloud = model(data.float()).type('torch.DoubleTensor')

        loss = loss_func(cloud[:, 0], eigens[:, 0])
        loss_check_matrix[epoch, j] = loss.detach().numpy()
        if epoch % 1000 == 0:
            print(epoch, batch_idx, loss.detach().numpy())
            if epoch % 10000 == 0:
                torch.save(model.state_dict(), './{}/model_{}_epoch{}k.pth'.format(folder_name_, j, int(epoch/1000)))
        test_data = nodes_test
        test_ind = np.random.choice(10000, B, replace=False)
        cloud_ = model(test_data[test_ind, :].float()).double()
        loss_test = loss_func(cloud_, eigens_test[test_ind, :])
        loss_check_matrix_test[epoch] = loss_test.detach().numpy()

        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), './{}/model_{}.pth'.format(folder_name_, j))
    print('Train set: ', loss_check_matrix[-1, j])
    np.save('./{}/Orig_loss.npy'.format(folder_name_), loss_check_matrix)        
    test_data = nodes_test



#==============================================================================
# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# import torch.optim as optim
# 
# import matplotlib.pyplot as plt
# 
# 
# nodes_train = torch.from_numpy((np.load('./train_traj.npy')))#.reshape((1440, 84)))
# labels_train = torch.from_numpy(np.load('./train_label.npy'))#/200000
# nodes_test = torch.from_numpy((np.load('./test_traj.npy')))#.reshape((360, 84)))
# labels_test = torch.from_numpy(np.load('./test_label.npy'))#/200000
# 
# X = nodes_train
# 
# Y = labels_train
# print(X.size())
# print(Y.size())
# 
# 
# class Net(nn.Module):
#     
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(7, 7)
#         self.relu1 = nn.ReLU()
#         self.dout = nn.Dropout(0.2)
#         self.fc2 = nn.Linear(50, 100)
#         self.prelu = nn.PReLU(1)
#         self.out = nn.Linear(100, 1)
#         self.out_act = nn.Sigmoid()
#         
#     def forward(self, input_):
#         a1 = self.fc1(input_)
#         h1 = self.relu1(a1)
#         dout = self.dout(h1)
#         a2 = self.fc2(dout)
#         h2 = self.prelu(a2)
#         a3 = self.out(h2)
#         y = self.out_act(a3)
#         return y
#     
# net = Net().type('torch.DoubleTensor')
# opt = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
# criterion = nn.BCELoss().type('torch.DoubleTensor')
# 
# def train_epoch(model, opt, criterion, batch_size=50):
#     model.train()
#     losses = []
#     for beg_i in range(0, X.size(0), batch_size):
#         x_batch = X[beg_i:beg_i + batch_size, :]
#         y_batch = Y[beg_i:beg_i + batch_size, :]
#         x_batch = Variable(x_batch)
#         y_batch = Variable(y_batch)
# 
#         opt.zero_grad()
#         # (1) Forward
#         y_hat = net(x_batch)
#         # (2) Compute diff
#         loss = criterion(y_hat, y_batch)
#         # (3) Compute gradients
#         loss.backward()
#         # (4) update weights
#         opt.step()        
#         losses.append(loss.data.numpy())
#     return losses
# 
# e_losses = []
# num_epochs = 20
# for e in range(num_epochs):
#     e_losses += train_epoch(net, opt, criterion.type('torch.DoubleTensor'))
# plt.plot(e_losses)
# 
# x_t = Variable(torch.randn(1, 50))
# net.eval()
# print(net(x_t))
# x_1_t = Variable(torch.randn(1, 50) + 1.5)
# print(net(x_1_t))
# 
#==============================================================================
