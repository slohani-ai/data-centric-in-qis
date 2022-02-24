# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 18:05:10 2020

@author: sanjaya lohani
"""

import numpy as np
import pandas as pd


def loader(out_class=16, width=6, height=6,
           training=55000, valid=50, qubits=4, ens_rank=0.5, trial=1, test_data_path='path_to_data',
           train_data_path='pahth to train data', kparams=1, n_add_separable_sets=None):
    if ens_rank == 'HS_Haar' or ens_rank == 'Haar' or ens_rank == 'HS_Haar_MA':
        train_data_file = train_data_path + f'/{ens_rank}_tomo_tau_dm_30000_qs_2.pickle'
    elif isinstance(ens_rank, list) or ens_rank == 'random':
        print('*****************************|alpha are in a list or random.**************************************')
        train_data_file = train_data_path + f'/HS_MA_tomo_tau_dm_30000_qs_2_asymmetric_alpha_{ens_rank[0]}_{ens_rank[1]}_{ens_rank[2]}_{ens_rank[3]}.pickle'
        # train_data_file = train_data_path + f'/Pool_MA_tomo_tau_dm_30000_qs_2_asymmetric_alpha_{ens_rank}.pickle'

    # elif ens_rank.split('_')[0] == 'Brian':
    #     # train_data_file = train_data_path + f'/Brian_tomo_tau_dm_50K.pickle'
    #     point = ens_rank.split('_')[1]
    #     train_data_file = train_data_path + f"/Brian_tomo_tau_dm_50K_point_{point}_even.pickle"
    else:
        # train_data_file = train_data_path + f'/MA_tomo_tau_dm_30000_qs_2_alpha_{ens_rank}.pickle'
        # train_data_file = train_data_path + f'/Pool_MA_tomo_tau_dm_30000_qs_2_asymmetric_alpha_{ens_rank}_k_params_{kparams}.pickle'
        train_data_file = train_data_path + f'/Pool_{ens_rank}_qs_{qubits}_tomo_tau_dm_35000_eta_{kparams[0]}_{kparams[1]}.pickle'

    # test_data_file = test_data_path + f'/testdata_qubit_size_2.pkl'
    tomo, tau, dm = pd.read_pickle(train_data_file)  # tomo, tau,dm
    print('tau shape', tomo.shape)
    if n_add_separable_sets is not None:
        train_data_file_1 = train_data_path + f'/{ens_rank}_qs_{qubits}_tomo_tau_dm_test_n_20000_TEST_to_be_mixed.pickle'
        tomo_1, tau_1, dm_1 = pd.read_pickle(train_data_file_1)  # tomo, tau,dm
        print('tomo_1', tomo_1.shape)
        tomo = np.concatenate([tomo[:30_000 - n_add_separable_sets], tomo_1[:n_add_separable_sets]])
        tau = np.concatenate([tau[:30_000 - n_add_separable_sets], tau_1[:n_add_separable_sets]])
        dm = np.concatenate([dm[:30_000 - n_add_separable_sets], dm_1[:n_add_separable_sets]])

    test_data_file = train_data_path + f'/Pool_{ens_rank}_qs_{qubits}_tomo_tau_dm_35000_eta_{kparams[0]}_{kparams[1]}.pickle'
    # test_data_file = train_data_path + f'/Pool_{ens_rank}_qs_{qubits}_tomo_tau_dm_30000.pickle'
    # test_data_file = test_data_path + f'/{ens_rank}_qs_{qubits}_tomo_tau_dm_test_n_20000_TEST.pickle'
    tomo_test, tau_test, dm_mle = pd.read_pickle(test_data_file)  # tomo, tau,dm

    tau_array = np.reshape(tau, [-1, out_class])
    print(np.sum(tomo, axis=1))
    x_train = np.reshape(np.stack(tomo[:training]), [-1, width * height])
    y_train = np.reshape(np.stack(tau_array[:training]), [-1, out_class])
    print(y_train.shape)

    x_valid = np.reshape(np.stack(tomo_test[-valid:]), [-1, width * height])
    print(x_valid.shape)
    # x_valid = np.reshape(np.stack(x_train[-valid:]),[-1,width*height])

    dm_train = dm[:training]
    print(dm_train.shape)
    dm_valid = dm_mle[-valid:]
    # dm_valid = dm[-valid:]

    return (x_train, y_train, x_valid, dm_train, dm_valid)
