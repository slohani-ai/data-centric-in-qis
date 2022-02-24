import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import Purity_Measure as pm
from utils import Concurrence_Measure as con
def c_mems(P): # P --> Purity
    if P >= 5/9. and P<=1:
        cmems = 0.5*(1 + np.sqrt(2*P-1))
    elif P<5/9. and P>=1/3.:
        cmems = np.sqrt(2*P - 2/3.)
    else:
        cmems = 0

    return cmems

def c_w(P):
    if P >= 1/3. and P <= 1:
        cw = 0.5*(np.sqrt(12*P-3)-1)
    else:
        cw = 0
    return cw

purity = np.linspace(0, 1, 100000)

C_M = list(map(c_mems, purity))
C_M_array = np.array(C_M)

C_W = list(map(c_w, purity))
C_W_array = np.array(C_W)

fidelity_list_valid, fidelity_av_valid, dms_valid, pur_list_valid, params_list_valid, ent_list_valid = pd.read_pickle(
    "./test_prediction_results_data_driven/fidelity_list_fid_av_pred_dm_params_alpha_Werner_k_params_[[0.0, 1.0]]_qs_2_separability_mix_MA.pickle")

fids_valid = np.array(fidelity_list_valid).reshape(-1, 10, 5000)
fids_mean_valid = fids_valid.mean(axis=1)
fids_std_valid = fids_valid.std(axis=1)

print(fids_mean_valid[0].shape)

# Test sets
tomo_test_data_file = '../../section_A/DataSets/Simulation/MA_train_MA_test/data/MA_tomo_tau_dm_30000_qs_2_alpha_TEST_0.1.pickle'
x_test, tau, dm_mle_uncorrected = pd.read_pickle(tomo_test_data_file)
pur_test = pm.purity(dm_mle_uncorrected)
con_test = con.concurrence(dm_mle_uncorrected)


# Training sets
_,_,dm_train = pd.read_pickle('../data/Pool_Werner_qs_2_tomo_tau_dm_35000_eta_0.0_1.0.pickle')

# Added sep. states
_,_,dm_sep = pd.read_pickle('../data/Werner_qs_2_tomo_tau_dm_test_n_20000_TEST_to_be_mixed.pickle')

print(dm_sep[:1750].shape)
fs = 14
for j in [0, -1]:
    sep_add = 0
    if j == 0:
        dm_train = dm_train[:5000]
        pass
    else:
        sep_add = 1750
        dm_train = np.concatenate((dm_train[:5000], dm_sep[:1750]))

    pur_train = pm.purity(dm_train)
    print(pur_train.shape)
    con_train = con.concurrence(dm_train)






    plt.plot(purity,C_M_array, 'r-')
    plt.plot(purity,C_W_array, 'k-')
    plt.scatter(pur_train, con_train,  c=np.repeat('m', len(pur_train)), s = 4)
    plt.scatter(pur_test, con_test, vmin=0.20, vmax=1., c=fids_mean_valid[j], s= 6)
    plt.xlabel('Purity', fontsize=fs)
    plt.ylabel('Concurrence', fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.colorbar()
    plt.grid(alpha=0.2)
    plt.subplots_adjust(bottom=0.13,left=0.13)
    plt.savefig(f'cp_{j}.svg')
    plt.savefig(f'cp_{j}.png')
    plt.show()
# # From MA MA train
# _,_,train_dm = pd.read_pickle('../section_A/DataSets/Simulation/MA_train_MA_test/data/Pool_MA_tomo_tau_dm_30000_qs_2_asymmetric_alpha_0.1_k_params_4_no_truncation.pickle')
# train_dm = train_dm[:5000]
# train_purity = pm.purity(train_dm)
# train_con = con.concurrence(train_dm)
#
# # From MA MA test
# _,_,test_dm = pd.read_pickle('../section_A/DataSets/Simulation/MA_train_MA_test/data/MA_tomo_tau_dm_30000_qs_2_alpha_TEST_0.1.pickle')
# test_purity = pm.purity(test_dm)
# test_con = con.concurrence(test_dm)
#
# fidelity_list, fidelity_av, dms, cm_list, pur_list, params_list = pd.read_pickle("../section_A/DataSets/Simulation/MA_train_MA_test/test_prediction_results_data_driven/fidelity_list_fid_av_pred_dm_params_alpha_0.1_k_params_['4_no_truncation', '5_no_truncation', '6_no_truncation', '7_no_truncation'].pickle")
#
# fid_nn = np.array(fidelity_list).reshape(4, 10, -1)
# fid_nn_unfit = fid_nn[:5]
# fid_nn_unfit_m = np.mean(fid_nn_unfit, axis=1)
# fs = 14
# k = [4, 5 ,6 ,7]
# for i in range(4):
#     # From MA MA train
#     _, _, train_dm = pd.read_pickle(
#         f'../section_A/DataSets/Simulation/MA_train_MA_test/data/Pool_MA_tomo_tau_dm_30000_qs_2_asymmetric_alpha_0.1_k_params_{k[i]}_no_truncation.pickle')
#     train_dm = train_dm[:5000]
#     train_purity = pm.purity(train_dm)
#     train_con = con.concurrence(train_dm)
#
#
#     plt.plot(purity,C_M_array, 'r-')
#     plt.plot(purity,C_W_array, 'k-')
#     plt.xlabel('Purity', fontsize=fs)
#     plt.ylabel('Concurrence', fontsize=fs)
#     plt.xticks(fontsize=fs)
#     plt.yticks(fontsize=fs)
#     # plt.scatter(pur_train, con_train, s=2)
#     # plt.scatter(pur_test, con_test, s=2)
#     plt.scatter(train_purity, train_con, c=np.repeat('m', len(train_purity)), s = 4)
#     plt.scatter(test_purity, test_con, vmin=0.80, vmax=1., c=fid_nn_unfit_m[i], s= 6)
#     plt.colorbar()
#     plt.grid(alpha=0.2)
#     plt.subplots_adjust(bottom=0.13,left=0.13)
#     plt.savefig(f'cp_{i}.svg')
#     plt.show()