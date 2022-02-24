import sys

sys.path.insert(0, './source/')
from source import Network_data_driven as arlnet
import time

# from multiprocessing import Process, Lock
# import time

##################
#### Qubit 1
# Total params: 47,404
# Trainable params: 47,404
##################
# Net = arlnet.ARL_Network(width = 3,height = 2,batch = 100 ,out_class = 4,qubit_size=1,
#                             dense_1=250,dense_2=150)


# ##################
# #### Qubit 2
# Total params: 517,316
# Trainable params: 517,316
# ##################
# Net = arlnet.ARL_Network(width=6, height=6, batch=100, out_class=16, qubit_size=2,
#                          dense_1=750, dense_2=450)


# ##################
# #### Qubit 3
# Total params: 5,945,214
# Trainable params: 5,945,214
# ##################
# Net = arlnet.ARL_Network(width=36, height=6, batch=100, out_class=64, qubit_size=3,
#                          dense_1=2500, dense_2=1000)


# ##################
# #### Qubit 4
# dense_1 = 40000, dense_2 = 2500
# Total params: 43,049,406
# Trainable params: 43,049,406

# dense_1=3500,dense_2=1500
# Total params: 33,991,906
# Trainable params: 33,991,906
# ##################
# Net = arlnet.ARL_Network(width=36, height=36, batch=100, out_class=256, qubit_size=4,
#                          dense_1=4500, dense_2=2500)

# ***********************************************************************************************************
# Net.Run_Network(epoch_n=300, eta=0.05, train_n=9500, from_checkpoint=False, ens_rank=1, trial=1)

train_n = 30000
# alpha_values = [0.8] # [0.1, 0.2, 0.3]
# alpha_values = [0.01, 'HS_Haar', 0.1, 0.3, 0.8]
# alpha_values = ['HS_Haar']
# alpha_values = ['HS_Haar_MA']
# alpha_values = ["Brian_25"]
# alpha_values = [[0.1,1.,1.,1.]]
# alpha_values = [[.8,.5,.6,.2]]
# alpha_values = [0.085]
# alpha_values = [12.248273554336155]
# alpha_values = [0.2857142857142858]
# alpha_values = [0.13793103448275856]
# alpha_values = [0.924]
# alpha_values = [2.0]
# alpha_values = [0.03]
# alpha_values = [0.015]
alpha_values = ['Werner']
# alpha_values = [4.0]
# alpha_values = [[1,.6,1,.6]]
# alpha_values = [[.9,.6,.9,.6]]
# alpha_values = [[.8,.7,.8,.7]]
# alpha_values = [[.8,.5,.8,.5]]
# alpha_values = ['random']
# alpha_values = ['HS_Haar']
for dense in [[3050,
               1650]]:  # [[50, 25], [150, 75], [250,150], [350, 200], [450, 250], [550, 300], [650,350], [750, 400], [850, 450], [950, 550], [1050, 650], [1550, 900], [2050, 1150], [2550, 1400], [3050, 1650]]:
    Net = arlnet.ARL_Network(width=6, height=6, batch=100, out_class=16, qubit_size=2,
                             dense_1=dense[0], dense_2=dense[1])
    for alphas in alpha_values:
        # for k_params in [4, 5, 6, 7, 8]:
        # for k_params in [[0., 0.05], [0., 0.1], [0., 0.2], [0., 0.3], [0., 0.4], [0., 0.5], [0., 0.6], [0., 0.7],
        #                  [0., 0.8], [0., 0.9], [0.,1.0]]:

        for k_params in [[0., 1.]]:
            # for k_params in ['4_no_truncation', 4, 5, 6, 7]:
            print('*' * 30)
            print(k_params)
            for trial in range(1, 10):
                for n_sep in [0, 250, 500, 750, 1000, 1250, 1500, 1750]:
                    print('n_sep:', n_sep)
                    Net.Run_Network(epoch_n=400, eta=0.008, train_n=train_n, valid_n=500,
                                    ens_rank=alphas,
                                    trial=trial,
                                    # test_data_path='../IBMQ/test_data/',
                                    test_data_path='./data',

                                    train_data_path='./data',
                                    save_folder_path='./models/data_driven',
                                    kparams=k_params,
                                    n_add_separable_sets=n_sep)

            # Net.Run_Network(epoch_n=400, eta=0.008, train_n=train_n, valid_n=500,
            #                 ens_rank=alphas,
            #                 trial=0,
            #                 test_data_path='../IBMQ/test_data',
            #                 # test_data_path='./data/data_shots',
            #                 train_data_path='./data',
            #
            #                 save_folder_path='./models/data_driven')

                time.sleep(5)

# def run_me_parallel(lock,trial):
#     lock.acquire()
#     Net.Run_Network(epoch_n=500, eta=0.05, train_n=train_n, from_checkpoint=False, ens_rank=ens_rank, trial=trial,
#                     data_path='../data/training_testing_Aug_17',
#                     save_folder_path='./models')
#     lock.release()
#
#
#
# for i in range(1, 6):
#     lock = Lock()
#     Process(target=run_me_parallel, args=(lock,i)).start()
