import sys

sys.path.insert(0, './source/')
sys.path.insert(0, './utils/')
from source import Network_model_driven as arlnet
import tensorflow as tf
import numpy as np
import sklearn.preprocessing as pp
import pandas as pd
# import qutip.visualization as v
# sys.path.insert(1, '../utils')
import Fidelity_Measure as FM
import _pickle as pkl
import time
import Purity_Measure as pm
import Partial_Trace as pt

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

qs = 2
test_n = 5000
n_shots = 1024  # [128, 256, 512, 1024, 2048, 4096] #[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
alpha = 'MEMS'
fids = []
# kparams = ['4_no_truncation', '5_no_truncation', '6_no_truncation', '7_no_truncation', '8_no_truncation', 4, 5, 6, 7, 8]
# kparams = ['None']  # ['16_no_truncation', '19_no_truncation', '22_no_truncation', '25_no_truncation'] # , 4, 5, 6, 7]
kparams = [[0.0, 1.0]]  # ['16_no_truncation', '19_no_truncation', '22_no_truncation', '25_no_truncation'] # , 4, 5, 6, 7]
# kparams = ['4_no_truncation', 4, 5, 6, 7]
# for k_params in kparams:
# test_data_path = './data'

# tomo_test_data_file = f'./data/Werner_qs_2_tomo_tau_dm_test_n_20000_TEST.pickle'
# tomo_test_data_file = f'./data/Werner_qs_2_tomo_tau_dm_test_n_20000_TEST_purity_gt_.25.pickle'
# tomo_test_data_file = './data/Pool_Werner_qs_2_tomo_tau_dm_20000_eta_0.33_1.0_entangled_TEST.pickle'
tomo_test_data_file = '../../section_A/DataSets/Simulation/MA_train_MA_test/data/MA_tomo_tau_dm_30000_qs_2_alpha_TEST_0.1.pickle'

x_test, tau, dm_mle_uncorrected = pd.read_pickle(tomo_test_data_file)  # tomo, tau,dm
# x_test_corr, _, dm_mle_corrected = pd.read_pickle(corr_test_data_file) #tomo, tau,dm
# print(x_test)
# print(x_test_corr)

print (len(dm_mle_uncorrected))
x_test = np.array(x_test).reshape(-1, 6 ** qs)[:test_n]  # [-20:]
# x_test = np.array(x_test_corr).reshape(-1, 6 ** qs)[:test_n]  # [-20:]
dm_test_mle = dm_mle_uncorrected[:test_n]  # [-20:]
# dm_test_mle = dm_mle_corrected[:test_n]  # [-20:]
# print(dm_test)
# _, fav = FM.Fidelity_Metric(dm_test, dm_test_mle)
# print(fav)
x_test = pp.scale(x_test, 1)
print(x_test.shape)

if qs == 1:
    x_test = x_test.reshape(-1, 2, 3, 1)
    print(x_test.shape)
elif qs == 2:
    x_test = x_test.reshape(-1, 6, 6, 1)
elif qs == 3:
    x_test = x_test.reshape(-1, 6, 36, 1)
elif qs == 4:
    x_test = x_test.reshape(-1, 36, 36, 1)
start_time = time.time()

# ens_rank = 1
params_list = []


def Run_me(train_n, alpha, dense_1, dense_2, kparams_values, trial, n_add_separable_sets):
    model = tf.keras.models.load_model(
        f'../MEMS/models/data_driven/data_driven_training_{train_n}_dense_1_{dense_1}_dense_2_{dense_2}_alpha_{alpha}_k_params_{kparams_values}_trial_{trial}_n_add_separable_sets_{n_add_separable_sets}_BEST.h5',
        custom_objects={'ErrorNode': arlnet.ErrorNode, \
                        'PredictDensityMatrix': arlnet.PredictDensityMatrix, \
                        'FidelityMetric': arlnet.FidelityMetric})
    params = np.sum([np.prod(v.get_shape()) for v in model.trainable_variables])
    print(params)
    params_list.append(params)

    _, dm_pred = model([x_test], training=False)
    fidelity, av_fid = FM.Fidelity_Metric(dm_pred, dm_test_mle)
    return [av_fid.numpy(), fidelity.numpy(), dm_pred]


train_n = 30000
# alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.8]
# alpha_values = [0.01, 0.1, 0.3, 0.8, 'HS_Haar', 'Brian_25']

# alpha_values = ['HS_Haar']
for dense in [[3050, 1650]]:
    print(alpha)
    fidelity_list = []
    fidelity_av = []
    cm_list = []
    pur_list = []
    dms = []
    ppt = []
    ent_list = []
    ent_rank_list = []
    for k_params in kparams:  # [[50, 25], [150, 75], [250, 150], [350, 200], [450, 250], [550, 300], [650, 350], [750, 400],
        # [850, 450], [950, 550], [1050, 650], [1550,900], [2050, 1150], [2550, 1400], [3050, 1650]]:
        for n_add_separable_sets in [0, 250, 500, 750, 1000, 1250, 1500, 1750]:
            for trial in range(0, 10):
                av_fid, fidelity, dm_pred = Run_me(train_n, alpha, dense[0], dense[1], k_params, trial,
                                                   n_add_separable_sets)
                fidelity_list.append(fidelity)
                fidelity_av.append(av_fid)
                dm_pred = dm_pred.numpy()
                # print(dm_pred)
                # print(dm_test_mle)
                # print(np.trace(dm_pred[0]))
                # cm_ = cm.concurrence(dm_pred)
                # cm_list.append(cm_)
                pm_ = pm.purity(dm_pred)
                pur_list.append(pm_)
                pm_test = pm.purity(dm_test_mle)
                print(pm_test.mean())
                ent, sep = pt.PPT(dm_pred)
                ent_rank_list.append(len(ent))
                print(len(ent), ent)
                # print(av_fid)
                dms.append(dm_pred)
            print('For alpha: ', alpha)
            print(fidelity_av)
            fids.append(fidelity_av)
            # print(fidelity_list)
    # print('Ent_rank_len:', [len(ent_ranks) for ent_ranks in ent_rank_list])
# with open(f'./test_prediction_results_data_driven/fidelity_list_fid_av_pred_dm_params_alpha_{alpha}_k_params_{kparams}_qs_{qs}_separability_mix_TEST.pickle', 'wb') as f:
# with open(f'./test_prediction_results_data_driven/fidelity_list_fid_av_pred_dm_params_alpha_{alpha}_k_params_{kparams}_qs_{qs}_separability_mix_TEST_purity_gt_.25.pickle', 'wb') as f:
# with open(f'./test_prediction_results_data_driven/fidelity_list_fid_av_pred_dm_params_alpha_{alpha}_k_params_{kparams}_qs_{qs}_separability_mix_Valid_Werner.pickle', 'wb') as f:
with open(f'./test_prediction_results_data_driven/fidelity_list_fid_av_pred_dm_params_alpha_{alpha}_k_params_{kparams}_qs_{qs}_separability_mix_MA.pickle', 'wb') as f:
    pkl.dump([fidelity_list, fidelity_av, dms, pur_list, params_list, ent_rank_list], f, -1)
print(fids)
