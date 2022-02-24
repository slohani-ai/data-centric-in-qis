import sys
sys.path.insert(0, '../utils')
import pandas as pd
import numpy as np
from mpl_toolkits import mplot3d

import matplotlib.pyplot as plt
import Fidelity_Measure as fm
import Purity_Measure as pm
import Concurrence_Measure as cm
import os

# tomo_HS, tau_HS, dm_HS = pd.read_pickle('../data/MA_tomo_tau_dm_5000_qs_3_alpha_TEST_8.0.pickle')
# # con_test_HS = cm.concurrence(dm_HS)
# pur_test_HS = pm.purity(dm_HS)
#
# print(np.min(pur_test_HS))
# _ = plt.hist(pur_test_HS, 20, histtype='stepfilled', density=True, color='r', alpha=0.5)
# fs = 14
# # plt.xlabel('Purity (Targets)', fontsize=fs)
# plt.xticks(np.arange(0.2, 0.45, 0.05), [])
# plt.yticks(fontsize=fs)
# plt.ylabel('Density', fontsize=fs)
# plt.axis([0.18, 0.4, 0, 22])
# plt.subplots_adjust(bottom=0.6, left=0.15)
# plt.savefig('hist_HS_pur.svg', dpi=600)
#
# plt.show()

tomo_MA, tau_MA, dm_MA = pd.read_pickle('../data/MA_tomo_tau_dm_5000_qs_4_alpha_TEST_0.015.pickle')
# con_test_MA = cm.concurrence(dm_MA)
pur_test_MA = pm.purity(dm_MA)

print(np.min(pur_test_MA))
_ = plt.hist(pur_test_MA, 20, histtype='stepfilled', density=True, color='b', alpha=0.5)

fs = 14
# plt.xlabel('Purity (Targets)', fontsize=fs)
plt.xticks(np.arange(0.3, 1.1, 0.1), [])
plt.yticks(fontsize=fs)
plt.ylabel('Density', fontsize=fs)
plt.axis([0.25, 1.0, 0, 20])
plt.subplots_adjust(bottom=0.6, left=0.15)
plt.savefig('hist_MA_pur.svg', dpi=600)
plt.show()
