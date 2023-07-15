'''
README
# python 3.8
# coding: utf-8
# main script for /modelers
'''

#########################################################################################
'''
References
# Model averaging: 
* https://machinelearningmastery.com/weighted-average-ensemble-for-deep-learning-neural-networks/
# Activations: 
* https://github.com/siebenrock/activation-functions
* https://www.v7labs.com/blog/neural-networks-activation-functions
'''
#########################################################################################
# All Imports
# from bsdcnn import bsdcnn

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.utils import plot_model, to_categorical # unused aon
from tensorflow.keras.metrics import *

#from keras_visualizer import visualizer
#from model_profiler import model_profiler
# import coremltools as ct
# import lime
# from lime import lime_image
# from skimage.segmentation import mark_boundaries

import numpy as np
import pandas as pd
#import seaborn as sns
#from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from scikitplot.metrics import plot_confusion_matrix, plot_roc
from pandas.plotting import register_matplotlib_converters
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

from numba import cuda
import os, random, copy, warnings, math, time, csv
from pathlib import Path
from contextlib import redirect_stdout

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)

print(tf.config.list_physical_devices())


#########################################################################################
# paths
arr_dir = '/home/SharedFiles/Projects/EEG/Inputs/seq_arr/'
outputs_dir = '/home/SharedFiles/Projects/EEG/Outputs/runs'

#########################################################################################
# Models ###
engine_names = ('BSDCNN',) #('CNN1Db2', 'CNN2Db1', 'BSDCNNb1')
engines = (bsdcnn,) #(cnn1d, cnn2d, bsdcnn)

# engine_names = ('CNN1Db2', 'CNN2Db1', 'BSDCNNb1') 
# engines = (cnn1d, cnn2d, bsdcnn)

# channel picks  ### CHECKER

all_ch = ('Fp1-AVG', 'F3-AVG', 'C3-AVG','P3-AVG','Fp2-AVG', 'F4-AVG', 'C4-AVG','P4-AVG','F7-AVG',
              'T1-AVG', 'T3-AVG', 'T5-AVG', 'O1-AVG', 'F8-AVG', 'T2-AVG', 'T4-AVG', 'T6-AVG', 'O2-AVG',
              'Fz-AVG', 'Cz-AVG', 'Pz-AVG') # in order of index!

uniqNs = [
         ['SNUCH01',['Fp2-AVG', 'Fp1-AVG', 'Fz-AVG', 'F4-AVG'] ], 
         ['SNUCH02',['F8-AVG', 'T2-AVG', 'Fz-AVG', 'F7-AVG'] ], 
         ['SNUCH03',['T6-AVG', 'P4-AVG', 'C4-AVG', 'T5-AVG'] ], 
         ['SNUCH04',['F7-AVG', 'Fp1-AVG', 'F8-AVG', 'P4-AVG'] ], 
         ['SNUCH05',['Fz-AVG', 'P4-AVG', 'T6-AVG', 'Pz-AVG'] ], 
         ['SNUCH06',['Fp1-AVG', 'Fp2-AVG', 'T2-AVG', 'F4-AVG'] ], 
         ['SNUCH07',['T6-AVG', 'F4-AVG', 'Cz-AVG', 'T4-AVG'] ], 
         ['SNUCH08',['F7-AVG', 'T3-AVG', 'Fp1-AVG', 'T1-AVG'] ], 
         ['SNUCH09',['F4-AVG', 'T6-AVG', 'Cz-AVG', 'F7-AVG'] ], 
         ['SNUCH10',['C4-AVG', 'F4-AVG', 'F3-AVG', 'C3-AVG'] ], 
         ['SNUCH11',['T6-AVG', 'P4-AVG', 'Fz-AVG', 'F4-AVG'] ],
        ]


# Global Settings
learning_rate = 1e-5 
batch_size = 2** 5 # CHECKER
epochs = 10** 2 
patience = 10

# Dataset settings
train_ratio = 0.7

# Other settings
scaler = MinMaxScaler()

#########################################################################################
def fetcher(_dir):
    patients, all_preictals, all_interictals = [], [], []
    for patient in os.listdir(_dir):
        if patient.startswith('SNUCH'): # CHECKER
            #print('patient', patient) # TEST
            patient_pth = os.path.join(_dir, patient)
            patients.append(patient)
            all_preictals.append([])
            all_interictals.append([])
            for ictalType in os.listdir(patient_pth):
                ictalType_pth = os.path.join(patient_pth, ictalType) 
                if os.path.isdir(ictalType_pth):
                    if ictalType == 'preictals':
                        for preictal in os.listdir(ictalType_pth):
                            if not preictal.startswith('.'):
                                preictal_pth = os.path.join(ictalType_pth, preictal)
                                all_preictals[-1].append(preictal_pth)
                    if ictalType == 'interictals':
                        for interictal in os.listdir(ictalType_pth):
                            if not interictal.startswith('.'):
                                interictal_pth = os.path.join(ictalType_pth, interictal)
                                all_interictals[-1].append(interictal_pth)

    print('Num patients:', len(patients))
    print('Cnt equivalency:', len(patients) == len(all_preictals) == len(all_interictals))
    
    # zip and order
    zipper = list(zip(patients, all_preictals, all_interictals))
    zipper = zip(*sorted(zipper, key = lambda x:x[0]))
     
    return tuple(zipper)


# train valid test split
def split_data(preictals, interictals, train_ratio, idx, t, tail):
    # # shape test
    # prime = np.load(preictals[0]).shape
    # print('preictals shape:', prime)
    # print([np.load(ele).shape for ele in preictals if np.load(ele).shape != np.load(preictals[0]).shape])
    # prime = np.load(interictals[0]).shape
    # print('interictals shape:', prime)
    # print([np.load(ele).shape for ele in interictals if np.load(ele).shape != np.load(interictals[0]).shape])
    
    tot_timesteps = 3000 ### hardcoded
    
    mask = [all_ch.index(ch) for ch in uniqNs[idx][1][:tail]] # find alternative
    print('patient: channels')
    print('SNUCH' + str(idx+1) + ':', uniqNs[idx][1][:tail])
    print('SNUCH' + str(idx+1) + ':', uniqNs[idx][1][:tail], file=t)

        
        
    X = [np.load(npy)[mask,:tot_timesteps] if np.load(npy).shape[1] >tot_timesteps else np.load(npy)[mask,:] for npy in preictals] + \
        [np.load(npy)[mask,:tot_timesteps] if np.load(npy).shape[1] >tot_timesteps else np.load(npy)[mask,:] for npy in interictals]   
    y = [1. for it in preictals] + [0. for it in interictals]
    
    data = list(zip(X, y))
    random.shuffle(data)
    
    train_size = int(len(data) * train_ratio)
    valid_size = int((len(data) - train_size) / 2) # 1:1 = valid:test
    valid_ind = train_size + valid_size
    
    train = data[:train_size]
    valid = data[train_size:valid_ind]
    test = data[valid_ind:]
    
    X_train, y_train = zip(*train)
    X_valid, y_valid = zip(*valid)
    X_test, y_test = zip(*test)
              
    return np.array(X_train, dtype=np.float32), np.array(y_train), np.array(X_valid, dtype=np.float32), np.array(y_valid), np.array(X_test, dtype=np.float32), np.array(y_test)

class PerformanceVisualizationCallback(Callback):
    def __init__(self, model, validation_data, image_dir):
        super().__init__()
        self.model = model
        self.validation_data = validation_data
        
        os.makedirs(image_dir, exist_ok=True)
        self.image_dir = image_dir

    def on_epoch_end(self, epoch, logs={}):
        y_pred = np.asarray(self.model.predict(self.validation_data[0]))
        y_true = self.validation_data[1]             
        y_pred_class = np.argmax(y_pred, axis=1)

        # plot and save confusion matrix
        fig, ax = plt.subplots(figsize=(16,12))
        plot_confusion_matrix(y_true, y_pred_class, ax=ax)
        # fig.savefig(os.path.join(self.image_dir, f'confusion_matrix_epoch_{epoch}'))

       # plot and save roc curve
        fig, ax = plt.subplots(figsize=(16,12))
        plot_roc(y_true, y_pred, ax=ax)
        # fig.savefig(os.path.join(self.image_dir, f'roc_curve_epoch_{epoch}'))

#########################################################################################
def modeler(arr_name, patients, all_preictals, all_interictals, t, cw, run_name, run_dir, engine_name, engine, tail):
    # paths
    image_dir = os.path.join(run_dir, 'visuals')
    model_dir = os.path.join(run_dir, 'models')
    
    # Multi-GPU Set-up # https://keras.io/guides/distributed_training/
    strategy = tf.distribute.MirroredStrategy(['/gpu:0', '/gpu:1']) # multi-GPU
    print('Num devices: {}'.format(strategy.num_replicas_in_sync)) # num GPUs
    #print('Patience:', patience)
    #print('Learning rate:', learning_rate)
    print('Batch size:', batch_size)
    
    print()
    print('-' * 65)
    
    # # all auc plot
    # plt.figure(2)
    # plt.title('Model AUC (All Patients)')
    # plt.axis([0,1,0,1])
    # plt.ylabel('TPR')
    # plt.xlabel('FPR')

    all_auc_keras, all_loss, all_acc = [], [], []
    
    for idx, patient in enumerate(patients[:]): #### CHECKER
        try:
            X_train, y_train, X_valid, y_valid, X_test, y_test = split_data(
                all_preictals[idx], all_interictals[idx], train_ratio, idx, t, tail)
            
            # Scale (for 3D shaped X)            
            X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
            X_valid = scaler.transform(X_valid.reshape(-1, X_valid.shape[-1])).reshape(X_valid.shape)
            X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)


            # Reshape
            Xs = (X_train, X_valid, X_test) # pack
            
            with strategy.scope(): # multi-gpu
                Xs_reshaped, model = engine(Xs, learning_rate, batch_size)

            X_train, X_valid, X_test = Xs_reshaped # unpack
            print('out Xs:', X_train.shape, X_valid.shape, X_test.shape)
            
            train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            val_data = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
            
            train_data = train_data.batch(batch_size)
            val_data = val_data.batch(batch_size)
            
            # Disable AutoShard
            options = tf.data.Options()
            options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
            train_data = train_data.with_options(options)
            val_data = val_data.with_options(options)
            
            
            # Compile model
            if idx == 0:
                model.summary()
                with redirect_stdout(t):
                    model.summary()
                    
            print('_' * 65)
            print(patient)
            print('=' * 65)
            print('Num total sequences:', len(all_preictals[idx]) + len(all_interictals[idx]))

            fname = '{}_{}_{}'.format(engine_name, patient, arr_name)
            fpath = os.path.join(run_dir, 'models', fname)
            
            early_stopping_monitor = EarlyStopping(
                monitor='binary_accuracy',
                mode='max',
                patience=patience,
                restore_best_weights=True,
            )

            # model_checkpoint_monitor = ModelCheckpoint(
            #     fpath, monitor='loss', verbose=0, 
            #     save_best_only=True, save_weights_only=True, 
            #     mode='auto', save_freq=batch_size*2**4,
            # )

            # performance_cbk = PerformanceVisualizationCallback(
            #       model=model,
            #       validation_data=(X_valid, y_valid),
            #       image_dir=image_dir,
            # )
            # tboard_callback = tf.keras.callbacks.TensorBoard(
            #     log_dir=run_dir,
            #     histogram_freq=1,
            #     update_freq=100
            #     #profile_batch=[16, 32]
            # )

            # Fit model
            hist = model.fit(train_data, 
                  epochs=epochs,
                  verbose=1,
                  validation_data=val_data,
                  callbacks = [early_stopping_monitor]#, model_checkpoint_monitor]#, tboard_callback],# performance_cbk]
            )

            val_loss, val_acc = model.evaluate(X_valid, y_valid, verbose=1, batch_size=batch_size)
            best_val_loss = min(hist.history['val_loss'])
            best_val_acc = max(hist.history['val_binary_accuracy'])
            all_loss.append(val_loss)
            all_acc.append(val_acc)

            print('Valid loss:', round(val_loss, 4), '|',
                  'Valid accuracy:', round(val_acc,4))
            print('Best valid loss:', round(best_val_loss,4), '|',
                  'Best valid accuracy:', round(best_val_acc,4)) # all accuracy is binary accuracy

            # Check
            y_pred = model.predict(X_test).ravel()

            nn_fpr_keras, nn_tpr_keras, nn_thresholds_keras = roc_curve(y_test, y_pred)
            auc_keras = auc(nn_fpr_keras, nn_tpr_keras)
            all_auc_keras.append(auc_keras)
            auc_score = roc_auc_score(y_test,y_pred)
            print('AUC Score:', round(auc_score,4))

            # Outputs
            curr_scores = [patient, round(auc_score,4), round(val_acc,4), round(val_loss,4)]
            cw.writerow(curr_scores)

#             # Visualization
#             # loss, acc plot
#             fig, loss_ax = plt.subplots()
#             acc_ax = loss_ax.twinx()

#             loss_ax.plot(hist.history['loss'], 'y', label = 'train loss')
#             loss_ax.plot(hist.history['val_loss'], 'r', label = 'val loss')
#             loss_ax.set_ylim([0,1])

#             acc_ax.plot(hist.history['binary_accuracy'], 'b', label = 'train accuracy')
#             acc_ax.plot(hist.history['val_binary_accuracy'], 'g', label = 'val accuracy')

#             loss_ax.set_xlabel('epoch')
#             loss_ax.set_ylabel('loss')
#             acc_ax.set_ylabel('accuracy')

#             loss_ax.legend(loc = 'upper left')
#             acc_ax.legend(loc = 'lower left')

#             fname = patient + '_loss_acc_' + run_name + '.png'
#             fpath = os.path.join(image_dir, fname)
#             fig.savefig(fpath)
#             plt.clf()

            # # auc plot (all)
            # plt.figure(2) # all patients
            # plt.plot(nn_fpr_keras, nn_tpr_keras, marker='.', label='{} (auc = {})'.format(patient, round(auc_keras,4)) )
            
#             print()
            
            # # plot & visualize model summary
            # fname = engine_name + '_' + run_name + '.png'
            # fpath = os.path.join(image_dir, fname)
            # plot_model(model, to_file=fpath, show_shapes=True, show_layer_names=True)
            
                # # keras-visualizer
                # fname = engine_name + '_' + run_name + '_graphics.png'
                # fpath = os.path.join(image_dir, fname)
                # visualizer(model, filename=fpath, format='png', view=False)

            # # Profiler
            # model = ct.convert(model)
            # profile = model_profiler(model, batch_size) # measure model profile
            # print(profile)
            # print(profile, file=t)

            # Clear mem
            del model
            tf.keras.backend.clear_session() # may not work. Safety catch with cuda.close() in main()
            
        except ValueError:
            #raise # TEST # CHECKER
            
            print('Error with {}, skipping...'.format(patient) )
            print('Error with {}, skipping...'.format(patient), file=t)
            continue

    print('_' * 65)

    avg_auc_keras = np.mean(all_auc_keras)
    print("Avg. auc for all patients:", round(avg_auc_keras,4))
    print("Avg. acc for all patients:",  round(np.mean(all_acc),4))
    avg_scores = ['Avg', round(avg_auc_keras,4), round(np.mean(all_acc),4), round(np.mean(all_loss),4)]
    cw.writerow(avg_scores)
    
    # plt.figure(2) # all auc plot
    # plt.legend(loc='best')
    # fname = 'all_auc' + run_name + '.png'
    # fpath = os.path.join(image_dir, fname)
    # plt.savefig(fpath)
    # plt.clf()
    

#########################################################################################
def ignition(arr_name, run_name, run_dir, engine_name, engine, tail):
    os.makedirs(run_dir, exist_ok=False)
    t = open(os.path.join(run_dir, 'model_run_' + run_name + '.txt'), 'w')
    c = open(os.path.join(run_dir, 'model_scores_' + run_name + '.csv'), 'w')
    cw = csv.writer(c)
    header = ['patient', 'auc', 'acc', 'loss']
    cw.writerow(header)

    print(run_name)
    print(run_name, file=t)
    patients, all_preictals, all_interictals = fetcher(os.path.join(arr_dir, arr_name))
    #print(*map(len, [patients, all_preictals, all_interictals])) # test
    modeler(arr_name, patients, all_preictals, all_interictals, t, cw, run_name, run_dir, engine_name, engine, tail)
    print('\n\n')

    t.close()
    c.close()

def main():
    cuda.close() # clear VRAM
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    for arr_name in os.listdir(arr_dir): # for all datasets (of differing seg_len, if any)
        # filter out ghosts, and pick desired dataset:
        if not arr_name.startswith('.') and 'seg05m' in arr_name:
            print()
            print('_' * 65)
            print('=' * 65)
            print('Dataset:', arr_name)
            print('=' * 65)
            print()
            
            tail = len(uniqNs[0][1]) # checker: 0
            for i in range(len(uniqNs[0][1])-1):
            
                for engine_name, engine in zip(engine_names, engines):   
                    run_name = timestamp + '_' + engine_name + '_' + arr_name + '_' + 'uniqNs' + str(tail)
                    run_dir = os.path.join(outputs_dir, run_name)
                    ignition(arr_name, run_name, run_dir, engine_name, engine, tail)
                    print()
                    
                tail -= 1
                print()
        print()
                    
    ## clear VRAM 
    cuda.close()
    
    # cuda.select_device(0)
    # cuda.close()
    # cuda.select_device(1)
    # cuda.close()

#########################################################################################
main() # call main
# penultimate line
