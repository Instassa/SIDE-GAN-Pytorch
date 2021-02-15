# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 18:19:08 2019

@author: marija
"""
import os
import numpy as np
import json


'''converting from json'''
def pre_processing(init_dir, out_dir_traj, out_dir_eig, eig=True):
    os.chdir(os.path.expanduser('~'))
    os.chdir(init_dir)

    name_list = os.listdir()    
    i = 0
    data_coefA = np.zeros((len(name_list), 42))
    data_coefB = np.zeros((len(name_list), 42))
    eigenvalues = np.zeros((len(name_list), 35))

    for json__ in name_list:
        try:
            with open(json__) as json_file:
                      data_coefA[i, :] = json.load(json_file)["coefficient_a"]
            with open(json__) as json_file:
                      data_coefB[i, :] = json.load(json_file)["coefficient_b"]
            if eig == True:
                with open(json__) as json_file:
                         eigenvalues[i, :] = json.load(json_file)["eigenvalues"]
            
            datapoint = np.concatenate((data_coefA[i, :].reshape((6, 7, 1)), data_coefB[i, :].reshape((6, 7, 1))), axis = 2)*4.9
            datapoint = np.concatenate((datapoint, datapoint[0:1, :, :]), axis=0)
            np.save('./{}/train_trajectory_{:03d}.npy'.format(out_dir_traj, i), datapoint)#.reshape((1, 84, 1)))
            if eig == True:
                np.save('./{}/eigenvalues_{:03d}.npy'.format(out_dir_eig, i), eigenvalues[i, :])
            i +=1
        except:
            continue
    print(np.shape(datapoint))


'''converting to json'''
def post_processing(npy_dir):
    os.chdir(os.path.expanduser('~'))
    os.chdir(npy_dir)

    name_list_test = os.listdir()
    for npy_file in name_list_test:
        if npy_file[-4:] == '.npy':
            print(npy_file)
            os.mkdir(npy_file[:-4])
            full_matrix = np.load(npy_file)/4.9
            for i in range(np.shape(full_matrix)[0]):
                data = {'coefficient_a': full_matrix[i, 0, :42, 0].tolist(),
                        'coefficient_b': full_matrix[i, 0, 42:, 0].tolist(),
                        'coefficient_number': 6,
                        'dof': 7,
                        'frequency': 0.4}
                with open('./' + npy_file[:-4] + '/data_{:02d}.json'.format(i), 'w') as outfile:
                    json.dump(data, outfile, indent = 4)


'''converting to json: matrices:'''
def post_processing_matrices(npy_dir):
    os.chdir(os.path.expanduser('~'))
    os.chdir(npy_dir)

    name_list_test = os.listdir()
    for npy_file in name_list_test:
        if npy_file[-4:] == '.npy':
            try:
                os.mkdir(npy_file[:-11])
            except:
                print('Cannot make a directory {}. Check if already exists'.format(str(npy_file[:-11])))
            new_num = np.int(npy_file[-5])+20
            os.mkdir(npy_file[:-11] + '/' + npy_file[-10:-5] + str(new_num))
            full_matrix = np.load(npy_file)/4.9
    #        eigenvalues_sim = np.load(str('./eig/eig_' + npy_file))#*200000
            for i in range(np.shape(full_matrix)[0]):
                data = {'coefficient_a': full_matrix[i, 0, :6, :].reshape(42).tolist(),
                        'coefficient_b': full_matrix[i, 1, :6, :].reshape(42).tolist(),
                        'coefficient_number': 6,
                        'dof': 7,
                        'frequency': 0.4,
    #                    'simulated_eigenvalues': eigenvalues_sim[i, :].tolist(), 
    #                    'predicted_fitness': np.max(eigenvalues_sim[i, :])/np.min(eigenvalues_sim[i, :])
                        }
                with open('./' + npy_file[:-11] + '/' + npy_file[-10:-5] + str(new_num)+ '/data_{:02d}.json'.format(i), 'w') as outfile:
                    json.dump(data, outfile, indent = 4)


