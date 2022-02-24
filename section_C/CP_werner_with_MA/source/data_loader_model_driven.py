# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 18:05:10 2020

@author: sanjaya lohani
"""
import os

import pandas as pd
import numpy as np

def loader(out_class=16,width=6,height=6,
           training=55000,valid=50,qubits=4, ens_rank=1, trial=1, test_data_path='path_to_data',
           train_data_path='pahth to train data'):
    train_data_file = train_data_path + f'/HS_tomo_tau_dm_30000_qs_2.pickle'
    # train_data_file = train_data_path + f'/MA_tomo_tau_dm_30000_qs_2_alpha_0.1.pickle'
    test_data_file = test_data_path + f'/testdata_qubit_size_2.pkl'
    tomo, tau, dm = pd.read_pickle(train_data_file) #tomo, tau,dm
    tomo_test, dm_test, dm_mle = pd.read_pickle(test_data_file) #tomo, tau,dm




    tau_array = np.reshape(tau,[-1,out_class])
    print (np.sum(tomo,axis=1))
    x_train = np.reshape(np.stack(tomo[:training]),[-1,width*height])
    y_train = np.reshape(np.stack(tau_array[:training]),[-1,out_class])

    x_valid = np.reshape(np.stack(tomo_test[-valid:]),[-1,width*height])

    dm_train = dm[:training]
    dm_valid = dm_test[-valid:]

    return (x_train,y_train,x_valid,dm_train,dm_valid)
