# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 13:27:32 2020

@author: Sanjaya_lohani (lohanisanjaya@gmail.com)
"""

import tensorflow as tf
import data_loader_model_driven as dl
import numpy as np
import _pickle as pkl
import time
import datetime

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


class ARL_Network(object):

    def __init__(self, width, height, batch, out_class, qubit_size, dense_1, dense_2):
        self._w = width
        self._h = height
        self._b = batch
        self._outclass = out_class
        self._qs = qubit_size
        self._dense_1 = dense_1
        self._dense_2 = dense_2

    def Network(self,
                conv_out_ch=25,
                kernel=[2, 2],
                pool_size=[2, 2],
                act_fn=tf.keras.activations.relu):
        input_to_network = tf.keras.Input(shape=(self._h, self._w, 1), \
                                          batch_size=None,
                                          name='Tomography_Measurements')

        conv_1 = tf.keras.layers.Conv2D(filters=conv_out_ch,
                                        kernel_size=kernel,
                                        strides=1,
                                        padding='same',
                                        activation=act_fn,
                                        name='First_CONV')(input_to_network)

        maxpool_1 = tf.keras.layers.MaxPool2D(pool_size=pool_size,
                                              strides=2,
                                              padding='valid',
                                              name='First_MAXPOOL')(conv_1)

        # conv_2 = tf.keras.layers.Conv2D(filters=conv_out_ch,
        #                                 kernel_size=kernel,
        #                                 strides=1,
        #                                 padding='same',
        #                                 activation=act_fn,
        #                                 name='Second_CONV')(maxpool_1)
        #
        # maxpool_2 = tf.keras.layers.MaxPool2D(pool_size = pool_size,
        #                               strides = 2,
        #                               padding = 'valid',
        #                               name = 'Second_MAXPOOL')(conv_2)
        # # #
        # conv_3 = tf.keras.layers.Conv2D(filters=conv_out_ch,
        #                                 kernel_size=kernel,
        #                                 strides=1,
        #                                 padding='same',
        #                                 activation=act_fn,
        #                                 name='Third_CONV')(maxpool_2)

        # maxpool_3 = tf.keras.layers.MaxPool2D(pool_size=pool_size,
        #                                       strides=2,
        #                                       padding='valid',
        #                                       name='Third_MAXPOOL')(conv_3)

        flatten = tf.keras.layers.Flatten(name='FLATTEN') (maxpool_1) #(conv_2)

        dense_1 = tf.keras.layers.Dense(units=self._dense_1,
                                        activation=act_fn,
                                        name='First_DENSE')(flatten)
        # dense_1_leaky_relu = tf.keras.layers.LeakyReLU()(dense_1)

        dropout_1 = tf.keras.layers.Dropout(0.5, name='First_DROPOUT')(dense_1)

        dense_2 = tf.keras.layers.Dense(units=self._dense_2,
                                        activation=act_fn,
                                        name='Sec_DENSE')(dropout_1)
        # dense_2_leaky_relu = tf.keras.layers.LeakyReLU()(dense_2)

        dropout_2 = tf.keras.layers.Dropout(0.5, name='Sec_DROPOUT')(dense_2)

        output_layer = tf.keras.layers.Dense(units=self._outclass,
                                             activation=None,
                                             name='Tau_Elements')(dropout_2)
        '''For one-hot-type output'''
        #        predictions = tf.keras.layers.Softmax()(output_layer)
        '''For regression output'''
        predictions = ErrorNode(name='Error')(output_layer)
        predictions_dm = PredictDensityMatrix(name='Density_Matrix', qubit_size=self._qs)(output_layer)
        # find_purity = Purity(name='Purity')(predictions_dm)
        # true_dm_input = tf.keras.Input(shape = (2**self._qs,2**self._qs),\
        #                                   batch_size=None,
        #                                   dtype=tf.complex128,
        #                                   name = 'True_DM')
        # fidelity_loss = FidelityLoss(name='fid_loss')([true_dm_input,predictions_dm])
        # Model = tf.keras.Model(inputs = [input_to_network,true_dm_input],outputs=[predictions,predictions_dm,fidelity_loss],name='ARL_Nets')
        Model = tf.keras.Model(inputs=input_to_network,
                               outputs=[predictions, predictions_dm], name='ARL_Nets')

        return Model

    def Run_Network(self, epoch_n=20, eta=0.01, train_n=15000, from_checkpoint=False,
                    ens_rank=1, trial=1, test_data_path='data_path', train_data_path='train_data_path',
                    save_folder_path='path_to_save'):

        model = self.Network()
        model.summary()
        tf.keras.utils.plot_model(model, f'ARL_model_qubits_{self._qs}.png', show_shapes=True)
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=eta)
        # Loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        Loss_fn = tf.keras.losses.MeanSquaredError()
        Loss_fn_purity = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        batch_size = self._b
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        test_log_dir = f'logs/qubits_size_{self._qs}/' + current_time + '/test'
        time_log_dir = f'logs/qubits_size_{self._qs}/' + current_time + '/time'
        dm_log_dir = f'logs/qubits_size_{self._qs}/' + current_time + '/dm'
        time_summary_writer = tf.summary.create_file_writer(time_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        visualize_dm_writer = tf.summary.create_file_writer(dm_log_dir)

        x_train, y_train, x_valid, dm_train, \
        dm_valid = dl.loader(out_class=self._outclass,
                                      width=self._w,
                                      height=self._h,
                                      training=train_n,
                                      valid=50,
                                      qubits=self._qs,
                                      trial=trial,
                                      test_data_path=test_data_path,
                                      train_data_path=train_data_path)

        def scaling_mean_0_std_1(row_matrix):
            m = tf.math.reduce_mean(row_matrix)
            std = tf.math.reduce_std(row_matrix)
            scaled = (row_matrix - m) / std
            return scaled

        x_train = tf.vectorized_map(scaling_mean_0_std_1, x_train)
        x_valid = tf.vectorized_map(scaling_mean_0_std_1, x_valid)

        x_train = tf.reshape(x_train, [-1, self._h, self._w, 1])
        x_valid = tf.reshape(x_valid, [-1, self._h, self._w, 1])
        print (x_train.shape)
        '''For Regression type output'''
        y_train = tf.reshape(100 * y_train, [-1, self._outclass])
        # y_valid = tf.reshape(100 * y_valid, [-1, self._outclass])
        print(y_train.shape)
        train_datasets = tf.data.Dataset.from_tensor_slices((x_train, y_train, dm_train))
        train_datasets = train_datasets.cache()
        train_datasets = train_datasets.shuffle(buffer_size=30000).batch(batch_size)

        valid_datasets = tf.data.Dataset.from_tensor_slices((x_valid, dm_valid))
        valid_datasets = valid_datasets.batch(50)
        valid_datasets = valid_datasets.cache()
        tf.print(valid_datasets.cardinality())

        Fidelity_Measure = FidelityMetric()
        epochs = epoch_n



        # checkpoints = tf.train.Checkpoint(opt=optimizer,
        #                                  model=model)
        # ckpt_manager = tf.train.CheckpointManager(checkpoints,f'./tmp/ckpt_{self._qs}',max_to_keep=2)
        #
        # if from_checkpoint:
        #     checkpoints.restore(ckpt_manager.latest_checkpoint)
        #     if ckpt_manager.latest_checkpoint:
        #         print (f'Restored from {ckpt_manager.latest_checkpoint}')
        #     else:
        #         print ('Initializing from scratch ...')
        # else:
        #     print('Initializing from scratch ...')

        @tf.function(experimental_compile=True)
        def train_me(x_batch_train, y_batch_train):
            with tf.GradientTape() as gtape:
                logits, rho_pred_train = model([x_batch_train], training=True)
                loss_mse = Loss_fn(y_batch_train, logits)
                # loss_purity = Loss_fn_purity(purity, tf.ones_like(purity))
                loss = loss_mse#+loss_purity
            grads = gtape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            return logits, loss

        start = time.time()
        rank_pred_list = []
        purity_list = []
        av_fid = [0.]
        for epoch in range(epochs):
            tf.print('|Start of epoch %d' % (epoch,))
            start_epoch_train = time.time()
            for step, (x_batch_train, y_batch_train, rhos) in enumerate(train_datasets):
                logits, loss_train = train_me(x_batch_train, y_batch_train)

                # train_acc_metric(y_batch_train,logits)
            #                valid_av_fidelity_metric(logits,rhos)
            end_epoch_train = time.time()
            # tf.print('time per epoch: ',end_epoch_train - start_epoch_train)
            with time_summary_writer.as_default():
                tf.summary.scalar(f'Time_per_epoch_qubit_size_{self._qs} (sec)',
                                  data=end_epoch_train - start_epoch_train, step=epoch)

            # tf.print('|Training Loss: {}'.format(float(loss_train)))
            # tf.print('|Training Purity: {}'.format(purity_train))
            # acc_per_epoch = train_acc_metric.result()
            # train_acc_metric.reset_states()
            # tf.print ('|Accuracy_Training: {}'.format(acc_per_epoch))
            # print ('|logits:{}'.format(logits.numpy()))
            #            av_fid_valid_data = []
            if (epoch+1) % 25 == 0:
                for x_batch_valid, rhos in valid_datasets:
                    logits_valid, rho_pred_valid = model([x_batch_valid])
                    #                av_fidelity_valid,_ = Fidelity_Measure.Fidelity_Metric(logits_valid,rhos)
                    #                av_fid_valid_data.append(av_fidelity_valid)
                    #            print ('|Fidelity_valid: {}'.format(tf.reduce_mean(av_fid_valid_data)))
                    Fidelity_Measure(rhos, rho_pred_valid)

                    # figure_1_buff = image_grid(rho_pred_valid[0],title='Predicted')
                    # figure_2_buff = image_grid(rhos[0],title='True')
                    # print (figure_1_buff)
                    # Convert to image and log
                    # with visualize_dm_writer.as_default():
                    #     tf.summary.image("Predicted_DM", plot_to_image(figure_1_buff), step=0)
                    #     tf.summary.image("True_DM", plot_to_image(figure_2_buff), step=0)

                av = Fidelity_Measure.result()
                if epoch > 20 and av > np.max(av_fid):
                    # model.save('../models/ARL_model_{}_batch_4_no_noise_qubits_{}_BEST.h5'.format('35K_pure_states',self._qs))
                    model.save(f'{save_folder_path}/model_driven_training_{train_n}_dense_1_{self._dense_1}_dense_2_{self._dense_2}_BEST.h5')
                    tf.print('|Best model is saved.')
                av_fid.append(av.numpy())

                Fidelity_Measure.reset_states()

                with test_summary_writer.as_default():
                    tf.summary.scalar(f'Fidelity_qubit_{self._qs}', data=av, step=epoch)
                tf.print('|Fidelity_valid: {}'.format(av))
                rank_pred = np.linalg.matrix_rank(rho_pred_valid, tol=1e-5)
                uniq, counts = np.unique(rank_pred, return_counts=True)
                dict_form = dict(zip(uniq, counts))
                tf.print('|Rank valid: {}'.format(dict_form))
                rank_pred_list.append(dict_form)
                rho_sq = tf.matmul(rho_pred_valid, tf.transpose(rho_pred_valid, perm=[0, 2, 1], conjugate=True))
                purity = tf.math.real(tf.linalg.trace(rho_sq))
                purity_list.append(purity.numpy())
                tf.print('|Purity_valid {}'.format(tf.reduce_mean(purity)))

            # if (epoch + 1) % 10 == 0:
            #     ckpt_manager.save()



        #            for x_batch_test,y_batch_test in test_datasets:
        #
        #                logits_test = model(x_batch_test)
        #                test_acc_metric(y_batch_test,logits_test)
        #            acc_per_epoch_test = test_acc_metric.result()
        #            test_acc_metric.reset_states()
        #            print ('|Accuracy_Test: {}'.format(acc_per_epoch_test))

        end = time.time()

        with open(f'./misc_outputs/model_driven/rank_distributions_per_5_epochs_qubits_{self._qs}_trial_{trial}_training_{train_n}_dense_1_{self._dense_1}_dense_2_{self._dense_2}.pkl','wb') as f:
            pkl.dump(rank_pred_list, f, -1)
        with open(f'./misc_outputs/model_driven/Total Training_time_qubits_{self._qs}_trial_{trial}_training_{train_n}_dense_1_{self._dense_1}_dense_2_{self._dense_2}.pkl', 'wb') as f:
            pkl.dump(end - start, f, -1)
        print(end - start)
        with open(f'./misc_outputs/model_driven/fidelity_list_qubits_{self._qs}_trial_{trial}_training_{train_n}_dense_1_{self._dense_1}_dense_2_{self._dense_2}.pkl','wb') as f:
            pkl.dump(av_fid,f,-1)
        # model.save('ARL_model_{}_batch_4_no_noise.h5'.format('60K_pure_states'))
        model.save(f'{save_folder_path}/model_driven_training_{train_n}_dense_1_{self._dense_1}_dense_2_{self._dense_2}_LAST.h5')

        # with open(f'Training_time_qubits_{self._qs}_shots_15_no_noise.pkl', 'wb') as f:
        #     pkl.dump(end - start, f, -1)
        # print(end - start)
        # with open(f'fidelity_list_qubits_{self._qs}_shots_15_no_noise.pkl', 'wb') as f:
        #     pkl.dump(av_fid, f, -1)
        # # model.save('ARL_model_{}_batch_4_no_noise.h5'.format('60K_pure_states'))
        # model.save('ARL_model_{}_batch_4_no_noise_qubits_{}_shots_15_no_noise.h5'.format('35K_pure_states', self._qs))


class FidelityMetric(tf.keras.metrics.Metric):

    def __init__(self, name='FidelityMetric', **kwargs):
        super(FidelityMetric, self).__init__(name=name, **kwargs)
        self.av_fidelity = tf.Variable(0., trainable=False)  # self.add_weight(name='fid', initializer='zeros')
        self.count = tf.Variable(0., trainable=False)

    # @tf.function(experimental_compile=True)
    def update_state(self, y_true, y_pred, sample_weight=None):
        rhos_value = y_true
        rho_mat_pred = y_pred
        sqrt_rho_pred = tf.linalg.sqrtm(rho_mat_pred)
        products = tf.matmul(tf.matmul(sqrt_rho_pred, rhos_value), sqrt_rho_pred)
        fidelity = tf.square(tf.linalg.trace(tf.math.real(tf.linalg.sqrtm(products))))
        av_fid = tf.reduce_mean(fidelity)
        av_fid = tf.cast(av_fid, tf.float32)
        self.av_fidelity.assign_add(av_fid)
        self.count.assign_add(1.)

    #        return av_fid,rho_mat_pred

    def result(self):
        return self.av_fidelity / self.count

    def reset_states(self):
        return self.count.assign(0.), self.av_fidelity.assign(0.)


class PredictDensityMatrix(tf.keras.layers.Layer):
    def __init__(self, name='Density_matrix', qubit_size=2, **kwargs):
        super(PredictDensityMatrix, self).__init__(name=name, **kwargs)

        self.qubit_size = qubit_size

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'qubit_size': self.qubit_size
        })
        return config

    def diag_and_off_diag(self, t_off_values):  # [1,2]
        t_off_values = tf.cast(t_off_values, tf.float32)
        concat = tf.cast(tf.complex(t_off_values[0], t_off_values[1]), tf.complex128)
        return concat

    def t_to_mat_layer(self, t):
        indices = list(zip(*np.tril_indices(n=2 ** self.qubit_size, k=-1)))
        indices = tf.constant([list(i) for i in indices], dtype=tf.int64)
        diags = tf.cast(t[:2 ** self.qubit_size], tf.float32)
        off_diags = tf.reshape(t[2 ** self.qubit_size:], [-1, 2])
        real, imag = tf.split(tf.cast(off_diags, tf.float32), 2, 1)
        real = tf.reshape(real, [-1])
        imag = tf.reshape(imag, [-1])

        t_mat_real = tf.sparse.SparseTensor(indices=indices, values=real,
                                            dense_shape=[2 ** self.qubit_size, 2 ** self.qubit_size])
        t_mat_imag = tf.sparse.SparseTensor(indices=indices, values=imag,
                                            dense_shape=[2 ** self.qubit_size, 2 ** self.qubit_size])

        t_mat_list_real = tf.sparse.to_dense(t_mat_real)
        t_mat_array_real_with_diag = tf.cast(tf.linalg.set_diag(t_mat_list_real, diags), tf.complex128)
        t_mat_array_imag_with_no_diag = tf.cast(tf.sparse.to_dense(t_mat_imag), tf.complex128)
        t_mat_array = tf.add(t_mat_array_real_with_diag, 1j * t_mat_array_imag_with_no_diag)

        return t_mat_array

    def t_mat_rho_layer(self, t_mat):
        rho = tf.matmul(t_mat, tf.transpose(t_mat, conjugate=True)) / \
              tf.linalg.trace(tf.matmul(t_mat, tf.transpose(t_mat, conjugate=True)))
        return rho

    def call(self, inputs):
        logits_l = tf.cast(inputs, tf.complex128)
        t_matrix_list_l = tf.vectorized_map(self.t_to_mat_layer, logits_l / 100)  ###***** divide HERE 100 ********
        t_matrix_l = tf.reshape(t_matrix_list_l, [-1, 2 ** self.qubit_size, 2 ** self.qubit_size])
        rho_mat_pred = tf.vectorized_map(self.t_mat_rho_layer, t_matrix_l)
        return rho_mat_pred

class Purity(tf.keras.layers.Layer):

    def __init__(self, name='purity', **kwargs):
        super(Purity, self).__init__(name=name, **kwargs)

    def call(self, inputs, **kwargs):
        rho_sq = tf.matmul(inputs, tf.transpose(inputs, perm=[0,2,1], conjugate=True))
        purity = tf.math.real(tf.linalg.trace(rho_sq))
        purity = tf.cast(purity,tf.float32)
        return purity

class ErrorNode(tf.keras.layers.Layer):
    def __init__(self, name='Error', **kwargs):
        super(ErrorNode, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        return inputs








