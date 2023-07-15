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
# XAI:
* https://towardsdatascience.com/deep-learning-model-interpretation-using-shap-a21786e91d16
'''
#########################################################################################
# All Imports
#from bsdcnn import bsdcnn_relu
from mlp import mlp

import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.utils import plot_model, to_categorical # unused aon
from tensorflow.keras.metrics import *

#from keras_visualizer import visualizer
#from model_profiler import model_profiler

import shap

import numpy as np
import pandas as pd
# import seaborn as sns
# from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from scikitplot.metrics import plot_confusion_matrix, plot_roc
from pandas.plotting import register_matplotlib_converters
from sklearn.metrics import roc_curve, auc, roc_auc_score

from numba import cuda
import os, random, copy, warnings, math, time, csv
from pathlib import Path
from contextlib import redirect_stdout

# SETTINGS
# multi-gpu
device_gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(device_gpus))

tf.debugging.set_log_device_placement(True)

# os.environ['TF_CUDNN_DETERMINISTIC']='1'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
# 
# tf.config.experimental.set_memory_growth(device_gpus[0], True)
# tf.config.experimental.set_memory_growth(device_gpus[1], True)


# strategy = tf.distribute.MultiWorkerMirroredStrategy() # multi-GPU
gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)

# other tf settings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# rs 
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
#########################################################################################
# paths
arr_dir = '/home/SharedFiles/Projects/EEG/Inputs/seq_arr/'
outputs_dir = '/home/SharedFiles/Projects/EEG/Outputs/runs'

#########################################################################################
# Models ### CHECKER
engine_names = ('mlp',) #('CNN1Db2', 'CNN2Db1', 'BSDCNNb1')
engines = (mlp,) #(cnn1d, cnn2d, bsdcnn)

# engine_names = ('CNN1Db2', 'CNN2Db1', 'BSDCNNb1') 
# engines = (cnn1d, cnn2d, bsdcnn)

# channel picks  ### CHECKER
ch_picks_list = ('all',) #('rt', 'lt') 

all_ch = ('Fp1-AVG', 'F3-AVG', 'C3-AVG','P3-AVG','Fp2-AVG', 'F4-AVG', 'C4-AVG','P4-AVG','F7-AVG',
              'T1-AVG', 'T3-AVG', 'T5-AVG', 'O1-AVG', 'F8-AVG', 'T2-AVG', 'T4-AVG', 'T6-AVG', 'O2-AVG',
              'Fz-AVG', 'Cz-AVG', 'Pz-AVG') # in order of index!
#ch_picks_list += all_ch

# Global Settings
learning_rate = 1e-5 
batch_size = 32 # CHECKER #num features
epochs = 1 #10** 2 
patience = 10

# Dataset settings
train_ratio = 0.7


#########################################################################################
def fetcher(_dir):
    patients, all_preictals, all_interictals = [], [], []
    for patient in os.listdir(_dir):
        if patient.startswith('SNUCH01'): # CHECKER
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
def split_data(preictals, interictals, train_ratio, ch_picks, t):
    # # shape test
    # prime = np.load(preictals[0]).shape
    # print('preictals shape:', prime)
    # print([np.load(ele).shape for ele in preictals if np.load(ele).shape != np.load(preictals[0]).shape])
    # prime = np.load(interictals[0]).shape
    # print('interictals shape:', prime)
    # print([np.load(ele).shape for ele in interictals if np.load(ele).shape != np.load(interictals[0]).shape])
    
    tot_timesteps = 3000 ### hardcoded
    
    if ch_picks == 'all':
        mask = [i for i in range(len(all_ch))] # find alternative
            
#     elif ch_picks == 'rt':
#         rt_ch = ('Fp2-AVG', 'F4-AVG', 'C4-AVG', 'P4-AVG', 'O2-AVG', 'F8-AVG','T4-AVG', 'T6-AVG')
#         mask = [all_ch.index(ch) for ch in rt_ch]
        
#     elif ch_picks == 'lt':
#         lt_ch = ('Fp1-AVG', 'F3-AVG', 'C3-AVG','P3-AVG', 'O1-AVG' ,'F7-AVG', 'T3-AVG', 'T5-AVG')
#         mask = [all_ch.index(ch) for ch in lt_ch]
        
#     else:
#         single_ch = ch_picks
#         print('Single channel:', single_ch)
#         print('Single channel:', single_ch, file=t)
#         mask = [all_ch.index(single_ch)]
        
        
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


#########################################################################################
def modeler(arr_name, patients, all_preictals, all_interictals, t, cw, run_name, run_dir, engine_name, engine, ch_picks):
    # paths
    image_dir = os.path.join(run_dir, 'visuals')
    model_dir = os.path.join(run_dir, 'models')

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
                all_preictals[idx], all_interictals[idx], train_ratio, ch_picks, t)  

            # Reshape & compile model
            print('Shaping dataset...')
            
            Xs = (X_train, X_valid, X_test) # pack
            with strategy.scope(): # multi-gpu
                Xs_reshaped, model = engine(Xs, learning_rate, batch_size)

            X_train, X_valid, X_test = Xs_reshaped # unpack
            print('out Xs:', X_train.shape, X_valid.shape, X_test.shape)
            
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

            model_checkpoint_monitor = ModelCheckpoint(
                fpath, monitor='loss', verbose=0, 
                save_best_only=True, save_weights_only=True, 
                mode='auto', save_freq=batch_size*2**4,
            )

            # performance_cbk = PerformanceVisualizationCallback(
            #       model=model,
            #       validation_data=(X_valid, y_valid),
            #       image_dir=image_dir,
            # )
            
            # Fit model
            print('Fitting model...')
            model.fit(X_train, y_train, 
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(X_valid, y_valid),
                  callbacks = [early_stopping_monitor]#, model_checkpoint_monitor]#, tboard_callback],# performance_cbk]
            )

            #### No evaluations or predictions for SHAP (XAI) ####
            
            # # compute SHAP values ### checker
            # print('Deeply explaining model...')
            # explainer = shap.DeepExplainer(model, X_train)
            # shap_values = explainer.shap_values(X_test)
            
            # print('@@@@ shap_values.shape', shap_values.shape)
            
            # # global interpretation
            # shap.summary_plot(shap_values[0], plot_type = 'bar', feature_names = all_ch)           
            # fname = 'shap_global_' + patient + '_' + run_name + '.png'
            # fpath = os.path.join(image_dir, fname)
            # plt.savefig(fpath, format = "png", dpi=150, bbox_inches='tight')
            # plt.clf()
    
                       
            # # Profiler
            # model = ct.convert(model)
            # profile = model_profiler(model, batch_size) # measure model profile
            # print(profile)
            # print(profile, file=t)

            # Clear mem
            del model
            tf.keras.backend.clear_session() # Sometimes may not work. Safety catch with cuda.close() in main()
            
        except ValueError:
            raise # TEST # CHECKER
            
            # print('Error with {}, skipping...'.format(patient) )
            # print('Error with {}, skipping...'.format(patient), file=t)
            # continue

    print('_' * 65)

    # avg_auc_keras = np.mean(all_auc_keras)
    # print("Avg. auc for all patients:", round(avg_auc_keras,4))
    # print("Avg. acc for all patients:",  round(np.mean(all_acc),4))
    # avg_scores = ['Avg', round(avg_auc_keras,4), round(np.mean(all_acc),4), round(np.mean(all_loss),4)]
    # cw.writerow(avg_scores)
    
    # plt.figure(2) # all auc plot
    # plt.legend(loc='best')
    # fname = 'all_auc' + run_name + '.png'
    # fpath = os.path.join(image_dir, fname)
    # plt.savefig(fpath)
    # plt.clf()
    

#########################################################################################
def ignition(arr_name, run_name, run_dir, engine_name, engine, ch_picks):
    os.makedirs(run_dir, exist_ok=False)
    t = open(os.path.join(run_dir, 'model_run_' + run_name + '.txt'), 'w')
    c = open(os.path.join(run_dir, 'model_scores_' + run_name + '.csv'), 'w')
    cw = csv.writer(c)
    header = ['patient', 'auc', 'acc', 'loss']
    cw.writerow(run_name.split('_'))
    cw.writerow(header)

    print(run_name)
    print(run_name, file=t)
    patients, all_preictals, all_interictals = fetcher(os.path.join(arr_dir, arr_name))
    #print(*map(len, [patients, all_preictals, all_interictals])) # test
    modeler(arr_name, patients, all_preictals, all_interictals, t, cw, run_name, run_dir, engine_name, engine, ch_picks)
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
            
            for engine_name, engine in zip(engine_names, engines):   
                for ch_picks in ch_picks_list:
                    run_name = timestamp + '_' + engine_name + '_' + arr_name + '_' + ch_picks
                    run_dir = os.path.join(outputs_dir, run_name)
                    ignition(arr_name, run_name, run_dir, engine_name, engine, ch_picks)
                    print()
                print()
            print()
                    
    ## clear VRAM 
    cuda.close()
    
    # cuda.select_device(0)
    # cuda.close()
    # cuda.select_device(1)
    # cuda.close()
    
    print('All process complete.')

#########################################################################################
main() # call main
# penultimate line
