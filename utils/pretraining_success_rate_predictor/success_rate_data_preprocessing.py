#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 13:43:44 2020

@author: marija
"""
import argparse
import os
import numpy as np
import json
import matplotlib.pyplot as plt


'''This step is meant to create positive and negative training examples for the Success Rate Predictor (SR) training.

    Negative examples in our case would be self-collisions of the manipulator, we created this via training a vanilla GAN
    on the training dataset consisting of positive trajectory parameters. The lack of SIDE-GAN-imposed constraints makes
    vanilla GAN trajectories very prone to self-colliding (in up to 96% cases in our empirical experiments).
    We take the trajectories from all training epochs of vanilla GAN - we are not interested in quality that much
    as in presence of negative (self-colliding) examples.
    
    We assume the usual ARDL library .json trajectories format, we provide some data for this operation in self-collisions_json
    (only some of these trajectories self-collide though - but it is different from the main dataset where all examples are positive).'''
    
parser = argparse.ArgumentParser()
parser.add_argument('--json_dir', default='self-collisions_json', help = 'directory containing ARDL trajectories in .json format')
parser.add_argument('--target_dir', default='train', help='path for processed training data')
parser.add_argument('--eps', type=int, default = 50, help='epochs at synthesis time')
parser.add_argument('--noise', type=int, default=10, help='number of noise vectors for each epoch at synthesis time')
parser.add_argument('--skip', type=int, default=100, help='sampling frequency')

opt = parser.parse_args()
print(opt)


try:
    os.mkdir(opt.target_dir)
except:
    print(str(opt.target_dir) + 'already exists.')


'''converting from json'''
i = 0
name_list = os.listdir()
number_of_samples = opt.eps * opt.noise * opt.skip
total_traj = np.zeros((number_of_samples, 2, 7, 7)) # this is hard-coded because
                                                    # the trajectory is ALWAYS defined as (2, 7, 7) for KUKA
                                                    # - 2 points defining cyclic trajectory
                                                    # - 7 joints of the manipulator
                                                    # - 6 modified Fourier transform parameters per join and point (we padded it to 7 for simplicity)
total_sr = np.zeros((number_of_samples, 1))

for eps in range(opt.eps):
    for noise in range(opt.noise):
        trajectories = np.load('{}/fake_epoch{}_noise{}.npy'.format(opt.json_dir, eps, noise))#/4.9
        sr_local = np.zeros(opt.skip)
        for j in range(opt.skip):
            json__ = '.{}/fake_epoch{}/noise{}/data_{:02d}.json'.format(opt.json_dir, eps, noise, j)
            try:
                with open(json__) as json_file:
                         eigenvalues = json.load(json_file)["eigenvalues"]
                sr_local[j] = 1
            except:
                sr_local[j] = 0
        total_traj[(eps*opt.noise * opt.skip+noise*opt.skip):(eps*opt.noise * opt.skip+(1+noise)*opt.skip), :, :, :] = trajectories
        total_sr[(eps*opt.noise * opt.skip+noise*opt.skip):(eps*opt.noise * opt.skip+(1+noise)*opt.skip), 0] = sr_local

x = np.arange(0, number_of_samples)
np.random.shuffle(x)
train_ind, test_ind = x[:int(number_of_samples*0.8)], x[int(number_of_samples*0.8):]   


'''saving the results for pre-training the SR predictor:'''
np.save('{}/train_traj.npy'.format(opt.target_dir), total_traj[train_ind, :, :, :])
np.save('{}/test_traj.npy'.format(opt.target_dir), total_traj[test_ind, :, :, :])

np.save('{}/train_label.npy'.format(opt.target_dir), total_sr[train_ind, :])
np.save('{}/test_label.npy'.format(opt.target_dir), total_sr[test_ind, :])   

print('Finished pre-processing. please run success_rate_training_predictor_network.py to pre-train the SR predictor.')           