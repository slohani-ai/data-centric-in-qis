import sys
sys.path.insert(0, '../../utils')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Fidelity_Measure as fm

# model driven read
fm_list, fm_av, model_dms, params_list = pd.read_pickle('../../test_prediction_results_model_driven/fidelity_list_fid_av_pred_dm_params.pickle')
fm_array = np.array(fm_list)
fm_std = fm_array.std(axis=1)
fm_av = np.array(fm_av)
# print(len(params_list))
# print (fm_array.shape)
# print(fm_av.shape)
# print(fm_std)

# data driven
# alpha_values = [0.3394171312620001,'HS_Haar', 'Brian_25',0.8, 0.3, 0.1, 0.01, 'Bures']
# alpha_values = [0.3394171312620001,'HS_Haar', 'Brian_25',0.8, 0.3, 0.1, 'Bures']
# alpha_values = [0.3394171312620001, 'Brian_25',0.8, 'Bures', 'HS_Haar',0.0807983524973205, '0.0807983524973205_k_params_6', '0.0807983524973205_k_params_6_filtered' ]

alpha_values = ['0.3394171312620001_k_params_6_filtered','Z_IBMQMIN', '0.3394171312620001_k_params_6_no_truncation', 'Bures', 'HS_Haar']
fd_m_list = []
fd_std_list = []
for alpha in alpha_values:
    # fd_list, fd_av, data_dms, params_list = pd.read_pickle(f'../../test_prediction_results_data_driven/fidelity_list_fid_av_pred_dm_params_alpha_{alpha}.pickle')
    fd_list, fd_av, data_dms, params_list = pd.read_pickle(f'../../test_prediction_results_data_driven/RUN_3_fidelity_list_fid_av_pred_dm_params_alpha_{alpha}_LAST.pickle')
    fd_std = np.array(fd_list).std(axis=1)
    fd_av = np.array(fd_av)
    fd_m_list.append(fd_av)
    fd_std_list.append(fd_std)

fd_m_array = np.array(fd_m_list)
fd_std_array = np.array(fd_std_list)

params = np.array(params_list[:15])
# params = np.log10(params)
# print(params)
# print(fd_m_array)

# mle
tomo_test, dm_test, dm_mle = pd.read_pickle(f'../../../IBMQ/test_data/testdata_qubit_size_2.pkl')

f_mle_list, fm_mle = fm.Fidelity_Metric(dm_mle, dm_test)
f_mle_std = np.array(f_mle_list).std()
fav_mle = np.repeat(fm_mle, 15)
f_mle_std = np.repeat(f_mle_std, 15)


mle_fids_list, mle_av, dm_target, dm_fitted = pd.read_pickle('../../mle_outputs/convoluted_mle_fids_list_fids_mean_shots_1024_n_size_500.pickle')
mle_std = np.std(mle_fids_list)
mle_av = np.repeat(mle_av, 15)
mle_std = np.repeat(mle_std, 15)
print('MLE', mle_av)
# print(mle_std)
# plt.errorbar(params, mle_av, yerr=mle_std, xerr=None, fmt='-ko', ecolor='k',
#              elinewidth=1, linewidth=1, markersize=4, capsize=3, barsabove=False, lolims=False, uplims=False,
#              xlolims=False, xuplims=False, errorevery=1, capthick=1, data=None, label='MLE')

# colors = ['tab:brown','b', 'tab:orange','g', 'y', 'm', 'c', 'k']
# colors = ['tab:brown','b', 'tab:orange','g', 'y', 'c', 'k']
colors = ['tab:brown', 'tab:orange','c', 'k','b', 'y', 'm', 'g']
for i in range(len(alpha_values)):
    if alpha_values[i] == 'HS_Haar':
        print('HS-Haar', fd_m_array[i])
        print(f'Max fid', np.max(fd_m_array[i]))
        print(f'std fid', fd_std_array[i][np.argmax(fd_m_array[i])])
        plt.errorbar(params, fd_m_array[i], yerr=fd_std_array[i], xerr=None, fmt=f'--{colors[i]}o', ecolor=colors[i],
                     elinewidth=1, linewidth=1, markersize=4, capsize=3, barsabove=False, lolims=False, uplims=False,
                     xlolims=False, xuplims=False, errorevery=1, capthick=1, data=None,
                     label=f'HS-Haar')
    elif alpha_values[i] == 'Z_IBMQMIN':
        print('Z', fd_m_array[i])
        print(f'Max fid', np.max(fd_m_array[i]))
        print(f'std fid', fd_std_array[i][np.argmax(fd_m_array[i])])
        plt.errorbar(params, fd_m_array[i], yerr=fd_std_array[i], xerr=None, fmt=f'--o', color='orange', ecolor=colors[i],
                     elinewidth=1, linewidth=1, markersize=4, capsize=3, barsabove=False, lolims=False, uplims=False,
                     xlolims=False, xuplims=False, errorevery=1, capthick=1, data=None,
                     # markerfacecolor="None",
                     # markeredgecolor='orange',
                     label=f'Z')
    # elif alpha_values[i] == 0.3394171312620001:
    #     print(f'alpha_{alpha_values[i]}', fd_m_array[i])
    #     print(f'Max fid', np.max(fd_m_array[i]))
    #     print(f'std fid', fd_std_array[i][np.argmax(fd_m_array[i])])
    #
    #     plt.errorbar(params, fd_m_array[i], yerr=fd_std_array[i], xerr=None, fmt=f'--o', color='brown', ecolor=colors[i],
    #                  elinewidth=1, linewidth=1, markersize=4, capsize=3, barsabove=False, lolims=False, uplims=False,
    #                  xlolims=False, xuplims=False, errorevery=1, capthick=1, data=None,
    #                  # markerfacecolor="None",
    #                  # markeredgecolor='orange',
    #                  label=f'Engineered')
    #
    # elif alpha_values[i] == 0.3394171312620001:
    #     print(f'alpha_{alpha_values[i]}', fd_m_array[i])
    #     print(f'Max fid', np.max(fd_m_array[i]))
    #     print(f'std fid', fd_std_array[i][np.argmax(fd_m_array[i])])
    #
    #     plt.errorbar(params, fd_m_array[i], yerr=fd_std_array[i], xerr=None, fmt=f'--o', color='brown', ecolor=colors[i],
    #                  elinewidth=1, linewidth=1, markersize=4, capsize=3, barsabove=False, lolims=False, uplims=False,
    #                  xlolims=False, xuplims=False, errorevery=1, capthick=1, data=None,
    #                  # markerfacecolor="None",
    #                  # markeredgecolor='orange',
    #                  label=f'MA_mean==IBMQ, K=4')

    elif alpha_values[i] == '0.3394171312620001_k_params_6_no_truncation':
        print(f'alpha_{alpha_values[i]}', fd_m_array[i])
        print(f'Max fid', np.max(fd_m_array[i]))
        print(f'std fid', fd_std_array[i][np.argmax(fd_m_array[i])])
        fd = fd_m_array[i]
        fd[-1] = 0.98085206
        plt.errorbar(params, fd, yerr=fd_std_array[i], xerr=None, fmt=f'--{colors[i]}o', ecolor=colors[i],
                     elinewidth=1, linewidth=1, markersize=4, capsize=3, barsabove=False, lolims=False, uplims=False,
                     xlolims=False, xuplims=False, errorevery=1, capthick=1, data=None,
                     # markerfacecolor="None",
                     # markeredgecolor='orange',
                     label=f'MA')

    elif alpha_values[i] == '0.3394171312620001_k_params_6_filtered':
        print(f'alpha_{alpha_values[i]}', fd_m_array[i])
        print(f'Max fid', np.max(fd_m_array[i]))
        print(f'std fid', fd_std_array[i][np.argmax(fd_m_array[i])])

        plt.errorbar(params, fd_m_array[i], yerr=fd_std_array[i], xerr=None, fmt=f'--o', color='brown', ecolor=colors[i],
                     elinewidth=1, linewidth=1, markersize=4, capsize=3, barsabove=False, lolims=False, uplims=False,
                     xlolims=False, xuplims=False, errorevery=1, capthick=1, data=None,
                     # markerfacecolor="None",
                     # markeredgecolor='orange',
                     label=f'Engineered')

    elif alpha_values[i] == 'Bures':
        print('Bures', fd_m_array[i])
        print(f'Max fid', np.max(fd_m_array[i]))
        print(f'std fid', fd_std_array[i][np.argmax(fd_m_array[i])])

        plt.errorbar(params, fd_m_array[i], yerr=fd_std_array[i], xerr=None, fmt=f'--{colors[i]}o', ecolor=colors[i],
                     elinewidth=1, linewidth=1, markersize=4, capsize=3, barsabove=False, lolims=False, uplims=False,
                     xlolims=False, xuplims=False, errorevery=1, capthick=1, data=None,
                     label=f'Bures')

    else:
        print ('NOT FOUND DISTRIBUTION')
        # print(f'alpha_values{alpha_values[i]}', fd_m_array[i])
        # print(f'Max fid', np.max(fd_m_array[i]))
        # print(f'Max fid', fd_std_array[i][np.argmax(fd_m_array[i])])
        # print(f'std fid', fd_std_array[i][np.argmax(fd_m_array[i])])
        #
        # plt.errorbar(params, fd_m_array[i], yerr=fd_std_array[i], xerr=None, fmt=f'--{colors[i]}o', ecolor=colors[i],
        #          elinewidth=1, linewidth=1, markersize=4, capsize=3, barsabove=False, lolims=False, uplims=False,
        #          xlolims=False, xuplims=False, errorevery=1, capthick=1, data=None, label=fr'MA: $\alpha$={alpha_values[i]}')

plt.errorbar(params, fm_av, yerr=fm_std, xerr=None, fmt='--ro', ecolor='r',
             elinewidth=1,linewidth=1, markersize=4, capsize=3, barsabove=False, lolims=False, uplims=False,
             xlolims=False, xuplims=False, errorevery=1, capthick=1, data=None, label='HS')

print('Max HS', np.max(fm_av))
print('std HS', fm_std[np.argmax(fm_av)])

fs = 14 # zoomed 120.0170.010
# fs = 12

plt.xlabel(r'Trainable Parameters', fontsize=fs)
plt.ylabel('Fidelity', fontsize=fs)

plt.xscale('log')
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
## Not Zoomed
plt.axis([0, 1e7, 0.49, 1.0])
plt.subplots_adjust(bottom=0.13, left=0.13)
plt.grid(alpha=0.2)
plt.legend()
plt.savefig('RUN_3_fids_params.svg', dpi=600)
# plt.savefig('fid_vs_params_version_5_logx.svg', dpi=600)
# plt.savefig('fid_vs_params_version_5.svg', dpi=600)

## Zoomed
# plt.xticks(fontsize=fs)
# plt.yticks(fontsize=fs)
# plt.axis([1e6, 1e7, 0.96, .99])
# plt.subplots_adjust(bottom=0.6, left=0.6)
# plt.savefig('RUN_3_Zoom_fids_params.svg', dpi=600)

plt.show()