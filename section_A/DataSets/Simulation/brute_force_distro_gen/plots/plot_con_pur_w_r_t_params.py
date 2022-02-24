import sys
sys.path.insert(0,'../utils/')
import pandas as pd
import Purity_Measure as pm
import Concurrence_Measure as cm
import matplotlib.pyplot as plt
import _pickle as pkl

tomo_test, dm_test, dm_mle = pd.read_pickle(f'../../../IBMQ/test_data/testdata_qubit_size_2.pkl')
con_test = cm.concurrence(dm_mle)
pur_test = pm.purity(dm_mle)
print('min_con', con_test.min())
print('max_con', con_test.max())
print('min_pur', pur_test.min())
print('max_pur', pur_test.max())
# k_unfit = ['4_no_truncation', '5_no_truncation', '6_no_truncation', '7_no_truncation', '8_no_truncation']
# k_fit = [4, 5, 6, 7, 8]


# choice = 'k_fit'
choice = 'k_unfit'
colors = ['b', 'r']

if choice == 'k_unfit':
    # kparams_set = k_unfit
    c = colors[0]

elif choice == 'k_fit':
    # kparams_set = k_fit
    c = colors[1]

con_list, pur_list = pd.read_pickle(f'con_list_pur_list_choice_{choice}.pickle')
fs = 17
for i in range(5):
    if i == 2:
        _ = plt.hist(con_list[i], 20, histtype='step', density=True, color=c, linewidth=1)
    # if i == 4:
    #     _ = plt.hist(con_list[i], 20, histtype='step', density=True, color=c, linewidth=1.5)
    _ = plt.hist(con_list[i], 20, histtype='step', density=True, color=c, linestyle='dashed', alpha=0.5, linewidth=1)
_ = plt.hist(con_test, 20, histtype='stepfilled', density=True, color='g', linestyle = 'solid', label='IBM Q', alpha=0.75)
plt.xlabel('Concurrence', fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.ylabel('Density', fontsize=fs)
plt.axis([-0.1, 1., 0, 7])
if choice == 'k_fit':
    plt.axis([-0.1, 1., 0, 2.5])
# plt.legend(fontsize='small')
plt.grid(alpha=0.2)
# plt.subplots_adjust(bottom=0.6, left=0.6)
plt.subplots_adjust(bottom=0.3, left=0.3)
plt.legend()
plt.savefig(f'histo_con_params_{choice}.svg', dpi=600)
# plt.savefig('histo_alpha.png', dpi=600)
# plt.show()

plt.show()

for j in range(5):
    if j == 2:
        _ = plt.hist(pur_list[j], 20, histtype='step', density=True, color=c, linewidth=1)

    _ = plt.hist(pur_list[j], 20, histtype='step', density=True, color=c, linestyle='dashed', alpha=0.5, linewidth=1)

_ = plt.hist(pur_test, 20, histtype='stepfilled', density=True, label='IBM Q', linestyle = 'solid', color='g', alpha=0.75)
plt.xlabel('Purity', fontsize=fs)
plt.xticks(fontsize=fs, rotation=45)
plt.yticks(fontsize=fs)
plt.ylabel('Density', fontsize=fs)

if choice == 'k_fit':
    plt.axis([0.6, 1., 0, 16])
else:
    plt.axis([0.28, 1., 0, 16])
plt.grid(alpha=0.2)
# plt.subplots_adjust(bottom=0.6, left=0.6)
plt.subplots_adjust(bottom=0.3, left=0.3)
plt.legend()
plt.savefig(f'histo_pur_params_{choice}.svg', dpi=600)
plt.show()
