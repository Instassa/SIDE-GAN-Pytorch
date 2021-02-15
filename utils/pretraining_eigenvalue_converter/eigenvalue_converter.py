#!/usr/bin/env python3
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import re

'''This function is an example of pre-training the part of the network responsible for converting  
    trajectory parameters into pseudo-eigenvalues.
    (we could calculate the actual eigenvalues but that would be slower in our case) '''


parser = argparse.ArgumentParser()
parser.add_argument('--train_dataset_dir', default='example_train_set', help = 'please provide the .npy arrays of training and testing trajectory parameters with corresponding eigen-values')
parser.add_argument('--converter_dir', default='eigen_converter', help='path to converter net parameters over training epochs')
parser.add_argument('--num_of_nets', type=int, default = 5, help='number of converters to train to choose the best out of')
parser.add_argument('--batchSize', type=int, default=100, help='input half-batch size')
parser.add_argument('--niter', type=int, default=50000, help='number of epochs to train for')

opt = parser.parse_args()
print(opt)


folder_name_ = opt.converter_dir
try:
    os.mkdir(folder_name_)
except:
    print(str(folder_name_) + 'already exists.')

B = opt.batchSize#10 # (half) batch size
averaging = opt.num_of_nets
num_eps = opt.niter


loss_matrix = np.zeros((num_eps, averaging))
loss_check_matrix = np.zeros((num_eps, averaging))
loss_check_matrix_test = np.zeros((averaging))
for j in range(averaging):
    print('network # ', j)
    nodes_train = torch.from_numpy((np.load('./{}/train_policies.npy'.format(opt.train_dataset_dir))))#.reshape((1440, 84)))
    distances_train = torch.from_numpy(np.load('./{}/train_diags.npy'.format(opt.train_dataset_dir)))#/200000
    nodes_test = torch.from_numpy((np.load('./{}/test_policies.npy'.format(opt.train_dataset_dir))))#.reshape((360, 84)))
    eigens_test = torch.from_numpy(np.load('./{}/test_diags.npy'.format(opt.train_dataset_dir)))#/200000

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(2, 2*B, kernel_size=3, stride=1, padding=1)#(2*B, 2*B, 5, 10) # 1, 16, 5, 1
            self.conv2 = nn.Conv2d(2*B, 10, kernel_size=3, stride=2, padding=1)
            self.fc2 = nn.Linear(160, 35)
    
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.fc2(x.view(B, -1))
            return x.view(B,-1)
        
    model = Net()
    model.float()
    optimizer = optim.Adam(model.parameters(), lr = 0.0001) #lr=1e-4)

    print('Ep.  Eigenvalue Loss      Fitness loss')    
    for epoch in range(num_eps):
        batch_idx = np.random.choice(1440, B, replace=False)
        #print(batch_idx)
        data = nodes_train[batch_idx, :]
        eigens = distances_train[batch_idx, :].type('torch.DoubleTensor')

        optimizer.zero_grad()
        
        cloud = model(data.float()).type('torch.DoubleTensor')

        loss_paper = torch.mean(torch.abs(cloud - eigens)/eigens)
#            loss_paper = torch.log((1/2016)*torch.sum(torch.abs(dist - dist_)/dist_)) #41664
        fitness_loss = torch.mean(torch.max(cloud, 1)[0]/torch.min(cloud, 1)[0] - torch.max(eigens, 1)[0]/torch.min(eigens, 1)[0]/(torch.max(eigens, 1)[0]/torch.min(eigens, 1)[0]))
        if epoch % 1000 == 0:
#            print(epoch, batch_idx, loss_paper.detach().numpy(),  fitness_loss.detach().numpy())
            print(epoch, loss_paper.detach().numpy(),  fitness_loss.detach().numpy())
            if epoch % 10000 == 0:
                torch.save(model.state_dict(), './{}/model_{}_epoch{}k.pth'.format(folder_name_, j, int(epoch/1000)))

        loss_check_matrix[epoch, j] = loss_paper.detach().numpy()

        loss = loss_paper
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), './{}/model_{}.pth'.format(folder_name_, j))
    print('Train set: ', loss_check_matrix[epoch, j])
#        np.save('./{}/PaperLoss_train_D{}_M{}_1kEpochs_rescaledWell_dropoutAfterC2.npy'.format(folder_name_, D, M), loss_check_matrix)        
#    np.save('./{}/PaperLoss_train.npy'.format(folder_name_), loss_check_matrix)        
    test_data = nodes_test
    optimizer.zero_grad()
    
    test_ind = np.random.choice(360, B, replace=False)
    cloud = model(test_data[test_ind, :].float()).double()

    loss_paper_test = torch.mean(torch.abs(cloud - eigens_test[test_ind, :])/eigens_test[test_ind, :])                            
    
    print('TEST SET: ', epoch, loss_paper_test.detach().numpy())

    loss_check_matrix_test[j] = loss_paper_test.detach().numpy()

min_test_loss_net = np.where(loss_check_matrix_test == np.min(loss_check_matrix_test))[0][0]

plt.plot(np.mean(loss_check_matrix, axis =1), c = 'b', label = 'Average Train Error')
plt.plot(np.mean(loss_check_matrix_test)*np.ones((num_eps)), c = 'g', label = 'Average Final Test Error')
plt.legend()
plt.xlabel('training epochs')
plt.savefig('./' + folder_name_ + '/training_averages_across_{}tries_eig_rescaling.png'.format(averaging))
#plt.show()     
plt.clf()         

plt.bar(np.arange(len(loss_check_matrix_test)), loss_check_matrix_test - np.min(loss_check_matrix_test), label = 'Test losses per network')
plt.title('Best net for test error is net number ' + str(min_test_loss_net), fontweight = 'heavy')
plt.xlabel('Differences in loss between the minimal loss net \n and each of the other options')
plt.savefig('./' + folder_name_ + '/differences_in_testlosses_from_{}_nets.png'.format(averaging))
#plt.show()
  

print('The converter with the smallest loss is converter #' + str(min_test_loss_net))
print('Recommended to use model_{}.pth in you future SIDE-GAN training. Will delete others.'.format(min_test_loss_net))
os.rename('./{}/model_{}.pth'.format(folder_name_, min_test_loss_net), './{}/min_test_loss_converter.pth'.format(folder_name_))

for _file in os.listdir(folder_name_):
    if re.search('model', _file):
        os.remove(str(folder_name_ + '/' + _file))
