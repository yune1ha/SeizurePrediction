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
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from scikitplot.metrics import plot_confusion_matrix, plot_roc
from pandas.plotting import register_matplotlib_converters
from sklearn.metrics import roc_curve, auc, roc_auc_score

from numba import cuda
import os, random, copy, warnings, math, time, csv
from pathlib import Path
from contextlib import redirect_stdout

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
batch_size = 32 # CHECKER
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
    
    # Multi-GPU Set-up # https://keras.io/guides/distributed_training/
    strategy = tf.distribute.MirroredStrategy() # multi-GPU
    print('Num devices: {}'.format(strategy.num_replicas_in_sync)) # num GPUs
    #print('Patience:', patience)
    #print('Learning rate:', learning_rate)
    print('Batch size:', batch_size)
    
    print()
    print('-' * 65)
    
    # all auc plot
    plt.figure(2)
    plt.title('Model AUC (All Patients)')
    plt.axis([0,1,0,1])
    plt.ylabel('TPR')
    plt.xlabel('FPR')

    all_auc_keras, all_loss, all_acc = [], [], []
    
    for idx, patient in enumerate(patients[:]): #### CHECKER
        try:
            X_train, y_train, X_valid, y_valid, X_test, y_test = split_data(
                all_preictals[idx], all_interictals[idx], train_ratio, ch_picks, t)

            # Reshape & compile model
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
            # tboard_callback = tf.keras.callbacks.TensorBoard(
            #     log_dir=run_dir,
            #     histogram_freq=1,
            #     update_freq=100
            #     #profile_batch=[16, 32]
            # )

            # Fit model
            model.fit(train_data, 
                  epochs=epochs,
                  verbose=1,
                  validation_data=val_data,
                  callbacks = [early_stopping_monitor]#, model_checkpoint_monitor]#, tboard_callback],# performance_cbk]
            )

            # no valuation for SHAP
#             val_loss, val_acc = model.evaluate(X_valid, y_valid, verbose=1, batch_size=batch_size)
#             best_val_loss = min(hist.history['val_loss'])
#             best_val_acc = max(hist.history['val_binary_accuracy'])
#             all_loss.append(val_loss)
#             all_acc.append(val_acc)

#             print('Valid loss:', round(val_loss, 4), '|',
#                   'Valid accuracy:', round(val_acc,4))
#             print('Best valid loss:', round(best_val_loss,4), '|',
#                   'Best valid accuracy:', round(best_val_acc,4)) # all accuracy is binary accuracy
            
            # 1 image for class preictal
            # class label list
            class_names = ['preictal', 'interictal']
            # example image for each class
            images_dict = dict()
            for i, l in enumerate(y_train):
                if len(images_dict)==2:
                    break
                if l not in images_dict.keys():
                    images_dict[l] = X_train[i]
            images_dict = dict(sorted(images_dict.items()))

            # example image for each class for test set
            X_test_dict = dict()
            for i, l in enumerate(y_test):
                if len(X_test_dict)==2:
                    break
                if l not in X_test_dict.keys():
                    X_test_dict[l] = X_test[i]
            # order by class
            X_test_each_class = [X_test_dict[i] for i in sorted(X_test_dict)]
            X_test_each_class = np.asarray(X_test_each_class)


            # Predict
            y_pred = model.predict(X_test).ravel()
            predicted_class = np.argmax(y_pred, axis=1)
            
            # SHAP 
            # plot actual and predicted class
            def plot_actual_predicted(images, pred_classes):
                fig, axes = plt.subplots(1, 11, figsize=(16, 15))
                axes = axes.flatten()

                # plot
                ax = axes[0]
                dummy_array = np.array([[[0, 0, 0, 0]]], dtype='uint8')
                ax.set_title("Base reference")
                ax.set_axis_off()
                ax.imshow(dummy_array, interpolation='nearest')
                
                # plot image
                for k,v in images.items():
                    ax = axes[k+1]
                    ax.imshow(v, cmap=plt.cm.binary)
                    ax.set_title(f"True: %s \nPredict: %s" % (class_names[k], class_names[pred_classes[k]]))
                    ax.set_axis_off()
                    
                plt.tight_layout()
                plt.show()
                
            # select backgroud for shap
            background = X_train[np.random.choice(X_train.shape[0], 1000, replace=False)]
            # DeepExplainer to explain predictions of the model
            explainer = shap.DeepExplainer(model, background)
            # compute shap values
            shap_values = explainer.shap_values(X_test_each_class)
            
            # plot SHAP values
            plot_actual_predicted(images_dict, predicted_class)
            print()
            shap.image_plot(shap_values, x_test_each_class)
            
            # save
            fname = 'shap_image_' + patient + '_' + run_name + '.png'
            fpath = os.path.join(image_dir, fname)
            plt.savefig(fpath, format = "png") #, dpi=150, bbox_inches='tight')
            plt.clf()
    
    
    
            # Scores
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
            
            # plot & visualize model summary 
            if patient == 'SNUCH01' and ch_picks=='all':
                fname = run_name + '.png'
                fpath = os.path.join(image_dir, fname)
                #plot_model(model, to_file=fpath, show_shapes=True, show_layer_names=True) # CHECKER
            
                # # keras-visualizer
                # fname = engine_name + '_' + run_name + '_graphics.png'
                # fpath = os.path.join(image_dir, fname)
                # visualizer(model, filename=fpath, format='png', view=False)
            
#             ## XAI (structured: channels)
#             # shap plot issues: https://github.com/slundberg/shap/issues/153
            
#             # compute SHAP values
#             explainer = shap.DeepExplainer(model, X_train)
#             shap_values = explainer.shap_values(X_test)
#             print('======shap_values shape:', shap_values.shape)
            
#             # SHAP Global Interpretation
#             plt.figure(10)
#             shap.summary_plot(shap_values[0], plot_type = 'bar', feature_names = all_ch, 
#                               show=False, matplotlib=True)
#             fname = 'shap_global_' + run_name + '.png'
#             fpath = os.path.join(image_dir, fname)
#             plt.savefig(fpath, format = "png", dpi=150, bbox_inches='tight')
#             plt.clf()
            
            # SHAP Local Interpretation
#             shap.initjs()
# shap.force_plot(explainer.expected_value[0].np(), shap_values[0][0], features = all_ch)

#             shap.decision_plot(explainer.expected_value[0].numpy(), shap_values[0][0], features = test_data.iloc[0,:], feature_names = all_ch)
    
#             shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0].numpy(), shap_values[0][0], feature_names = all_ch)
            
            ## XAI (unstructured)
            
    
                       
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

#########################################################################################
main() # call main
# penultimate line
