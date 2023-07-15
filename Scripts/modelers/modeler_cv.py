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
from cnn1d_kf import cnn1d
# from cnn_lstm_1 import cnn_lstm_1
#from lstm_1 import lstm_1

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"

## double check
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')

tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.experimental.set_memory_growth(gpus[1], True)
###

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.utils import plot_model, to_categorical # unused aon
from tensorflow.keras.metrics import *
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold, StratifiedKFold

from numba import cuda

import random, copy, warnings, math, time, csv, pickle, gc
from pathlib import Path
from contextlib import redirect_stdout

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#tf.config.experimental.set_memory_growth(gpu, True)
warnings.filterwarnings("ignore", category=DeprecationWarning)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
strategy = tf.distribute.MultiWorkerMirroredStrategy() # multi-GPU

#########################################################################################
# paths
arr_dir = '/home/SharedFiles/Projects/EEG/Inputs/seq_arr/'
outputs_dir = '/home/SharedFiles/Projects/EEG/Outputs/runs'

#########################################################################################
# Models ### CHECKER
engine_names = ('cnn1d',) #('CNN1Db2', 'CNN2Db1', 'BSDCNNb1')
engines = (cnn1d,) #(cnn_lstm_1, cnn2d, bsdcnn)

# channel picks  

all_ch = ('Fp1-AVG', 'F3-AVG', 'C3-AVG','P3-AVG','Fp2-AVG', 'F4-AVG', 'C4-AVG','P4-AVG','F7-AVG',
              'T1-AVG', 'T3-AVG', 'T5-AVG', 'O1-AVG', 'F8-AVG', 'T2-AVG', 'T4-AVG', 'T6-AVG', 'O2-AVG',
              'Fz-AVG', 'Cz-AVG', 'Pz-AVG') # in order of index!

ch_picks_list = ('all',) #('rt', 'lt') 

# ch_picks_list += all_ch[:] 


# Global Settings
learning_rate = 1e-5 
batch_size = 64 # CHECKER (16 if cnn2d, 256 if lstm)
epochs = 10** 2  # CHECKER
patience = 10

# Dataset settings
train_ratio = 0.7


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
        
    elif ch_picks.endswith('AVG'):
        print('ends with -AVG')
        single_ch = ch_picks
        print('Single channel:', single_ch)
        print('Single channel:', single_ch, file=t)
        mask = [all_ch.index(single_ch)]
        
    #else 
        
        
    X = [np.load(npy)[mask,:tot_timesteps] if np.load(npy).shape[1] >tot_timesteps else np.load(npy)[mask,:] for npy in preictals] + \
        [np.load(npy)[mask,:tot_timesteps] if np.load(npy).shape[1] >tot_timesteps else np.load(npy)[mask,:] for npy in interictals]   
    y = [1. for it in preictals] + [0. for it in interictals]

    data = list(zip(X, y))
    random.shuffle(data)
    
    train_size = int(len(data) * train_ratio)
    
    train = data[:train_size]
    test = data[train_size:]


    # zip
    X_train, y_train = zip(*train)  
    X_test, y_test = zip(*test)

    # IF LSTM ## CHECKER
    # y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
    # y_test = np.asarray(y_test).astype('float32').reshape((-1,1))

              
    return np.array(X_train, dtype=np.float32), np.array(y_train), np.array(X_test, dtype=np.float32), np.array(y_test)

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
def modeler(arr_name, patients, all_preictals, all_interictals, t, cw, run_name, run_dir, engine_name, engine, ch_picks):
    # paths
    image_dir = os.path.join(run_dir, 'visuals')
    model_dir = os.path.join(run_dir, 'models')
    
    # Multi-GPU Set-up # https://keras.io/guides/distributed_training/
    #gpus = tf.config.list_logical_devices('GPU')
    print()
    #print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
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
    
    for idx, patient in enumerate(patients[:]):
        try:
            X_train, y_train, X_test, y_test = split_data(
                all_preictals[idx], all_interictals[idx], train_ratio, ch_picks, t)

            # Reshape & compile model
            Xs = (X_train, X_test) # pack
            with strategy.scope(): # multi-gpu
                Xs_reshaped, model = engine(Xs, learning_rate, batch_size)

            X_train, X_test = Xs_reshaped # unpack
            print('out Xs:', X_train.shape, X_test.shape)

        
            # Summary
            if idx == 0:
                model.summary()
                with redirect_stdout(t):
                    model.summary()

            # Preliminaries
            print('_' * 65)
            print(patient)
            print('=' * 65)
            print('Num total sequences:', len(all_preictals[idx]) + len(all_interictals[idx]))

            fname = '{}_{}_{}'.format(engine_name, patient, arr_name)
            fpath = os.path.join(run_dir, 'models', fname)
                                   
            
            # Disable AutoShard
            options = tf.data.Options()
            options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF


            # Callbacks
            early_stopping_monitor = EarlyStopping(
                monitor='binary_accuracy',
                mode='max',
                patience=patience,
                restore_best_weights=True,)

            callbacks_list = [early_stopping_monitor,]

            ## Stratified KFold
            skf = StratifiedKFold(n_splits = 5, random_state = 42, shuffle = True) 
            fold = 1
            all_curr_scores = []

            print('Num CV kFold Splits:', skf.get_n_splits(X_train, y_train))
            #print('Pre CV Data Type check:', type(X_train), type(y_train))
            print('patient', 'fold', 'auc', 'acc', 'spec', 'sens', 'fpr', 'loss', file=t)

            for train_ind, val_ind in skf.split(X_train, y_train):
                print('Cross validation Fold:', fold)
                print("TRAIN len:", len(train_ind), "TEST: len", len(val_ind))

                fold_train_data = tf.data.Dataset.from_tensor_slices((X_train[train_ind], y_train[train_ind]))
                fold_val_data = tf.data.Dataset.from_tensor_slices((X_train[val_ind], y_train[val_ind]))

                fold_train_data = fold_train_data.batch(batch_size)
                fold_val_data = fold_val_data.batch(batch_size)

                fold_train_data = fold_train_data.with_options(options)
                fold_val_data = fold_val_data.with_options(options)

                hist = model.fit(
                                fold_train_data,
                                batch_size = batch_size,
                                verbose = 1,
                                epochs=epochs,
                                validation_data = fold_val_data,
                                callbacks=callbacks_list)
                

                val_loss, val_acc, *is_anything_else_being_returned = model.evaluate(
                                        fold_val_data, 
                                        verbose=1, 
                                        batch_size=batch_size)

                # Metrics
                best_val_loss = min(hist.history['val_loss'])
                best_val_acc = max(hist.history['val_binary_accuracy'])


                print('Valid loss:', round(val_loss, 4), '|',
                    'Valid accuracy:', round(val_acc,4))
                print('Best valid loss:', round(best_val_loss,4), '|',
                    'Best valid accuracy:', round(best_val_acc,4)) # all accuracy is binary accuracy

                
                # Check
                p_pred = model.predict(X_test).ravel()
                y_pred = np.where(p_pred > 0.5, 1, 0)

                fpr_keras, tpr_keras, th_keras = roc_curve(y_test, p_pred)
                auc_score = roc_auc_score(y_test, p_pred)

                # print('AUC Score:', round(auc_score,4), file=t)
                # print(classification_report(y_test, y_pred, zero_division=0), file=t)
                
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel() 
                val_spec = tn / (tn + fp + K.epsilon())
                val_sens = tp / (tp + fn + K.epsilon())
                val_fpr = fp / (fp + tn + K.epsilon())

                # Outputs
                curr_scores = [
                    patient, fold, 
                    round(auc_score,4), 
                    round(val_acc,4), 
                    round(val_spec,4),
                    round(val_sens,4),
                    round(val_fpr,4),
                    round(val_loss,4),
                ]
  
                print(curr_scores, file=t)
                all_curr_scores.append(curr_scores[2:])


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
                # plt.plot(fpr_keras, tpr_keras, marker='.', label='{} (auc = {})'.format(patient, round(auc_score,4)) )
                
    #             print()
                
                # # plot & visualize model summary
                # if patient == 'SNUCH01' and ch_picks=='all':
                #     fname = engine_name + '_' + run_name + '.png'
                #     fpath = os.path.join(image_dir, fname)
                #     plot_model(model, to_file=fpath, show_shapes=True, show_layer_names=True)
                
                    # # keras-visualizer
                    # fname = engine_name + '_' + run_name + '_graphics.png'
                    # fpath = os.path.join(image_dir, fname)
                    # visualizer(model, filename=fpath, format='png', view=False)

                # # Profiler
                # model = ct.convert(model)
                # profile = model_profiler(model, batch_size) # measure model profile
                # print(profile)
                # print(profile, file=t)

                # Clear vram
                tf.keras.backend.clear_session() # may not work. Safety catch with cuda.close() in main()
                
                fold += 1

            # print mean scores of all folds
            mean_scores = list(np.round(np.mean(all_curr_scores, axis=0), 4))
            cw.writerow([patient, '-'] + mean_scores)
            print([patient, 'ALL'] + mean_scores, file=t)
            print('MEAN SCORES:', 'auc', 'acc', 'spec', 'sens', 'fpr', 'loss')
            print(mean_scores)


        except ValueError:
            raise ## TEST ### CHECKER
            
            # print('Error with {}, skipping...'.format(patient) )
            # print('Error with {}, skipping...'.format(patient), file=t)
            # continue

    print('_' * 65)

    # Clear vram
    tf.keras.backend.clear_session()
    

#########################################################################################
def ignition(arr_name, run_name, run_dir, engine_name, engine, ch_picks):
    os.makedirs(run_dir, exist_ok=False)
    t = open(os.path.join(run_dir, 'model_run_' + run_name + '.txt'), 'w')
    c = open(os.path.join(run_dir, 'model_scores_' + run_name + '.csv'), 'w')
    cw = csv.writer(c)
    header = ['patient', 'fold', 'auc', 'acc', 'spec', 'sens', 'fpr', 'loss']
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
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    for arr_name in os.listdir(arr_dir): # for all datasets (of differing seg_len, if any)
        # filter out ghosts, and pick desired dataset:
        if not arr_name.startswith('.') and 'seg05m' in arr_name: ####
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
                    gc.collect()        
                    print()
                print()
                gc.collect()
            print()
            gc.collect()

    print('All process complete.')
    ## clear mem
    gc.collect()
    cuda.close()
    
    # cuda.select_device(0)
    # cuda.close()
    # cuda.select_device(1)
    # cuda.close()

#########################################################################################
main() # call main
# penultimate line
