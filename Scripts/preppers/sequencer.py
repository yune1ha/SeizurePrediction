'''
Further slices sliced segments into (seq_len long, seq_cnt many) sequences. 
Sample balancing via PREICTAL OVERSAMPLING with SLIDING WINDOW technique or RUS on interictals.

Designed by HIRC Lab @ SNUBH in ROK.
'''

## all imports
import numpy as np
import mne  
import random

import os, shutil
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

## pathing
path = "/home/SharedFiles/Data/HospitalData/EEG/SNUCH_VEM_EDF"
seg_dir = '/home/SharedFiles/Projects/EEG/Interim/seg_raws'
seq_dir = '/home/SharedFiles/Projects/EEG/Interim/seq_raws'

def fetcher(_dir):
    ## retrieve all paths
    print('retrieving all files...')
    patients, all_preictals, all_interictals = [], [], []
    for patient in os.listdir(_dir):
        patient_pth = os.path.join(_dir, patient)
        if patient.startswith('SNUCH'):
            patients.append(patient)
            all_preictals.append([])
            all_interictals.append([])
            for ictalType in os.listdir(patient_pth):
                ictalType_pth = os.path.join(patient_pth, ictalType) 
                if os.path.isdir(ictalType_pth):
                    if ictalType == 'preictals':
                        for preictal in os.listdir(ictalType_pth):
                            if not preictal.startswith('._'):
                                preictal_pth = os.path.join(ictalType_pth, preictal)
                                all_preictals[-1].append(preictal_pth)
                    if ictalType == 'interictals':
                        for interictal in os.listdir(ictalType_pth):
                            if not interictal.startswith('._'):
                                interictal_pth = os.path.join(ictalType_pth, interictal)
                                all_interictals[-1].append(interictal_pth)
    return patients, all_preictals, all_interictals

# # Test                            
# #print('all_preictals = all_interictals = patients:', 
# #      len(all_preictals) == len(all_interictals) == len(patients) )

# ## Settings
# seg_len = 5 *60#sec
# seq_len = int(0.5 *60)sec 
# seq_cnt = seg_len // seq_len
# new_sfreq = 100.0 # og = 200.0
# period = 1 / new_sfreq

# inp_overwrite = True

def sequencer(patients, all_preictals, all_interictals):
    ## Sequence Slicer
    method = ''
    applyAll = 'm'

    print('starting slicer...')
    for p in range(len(patients)): 
        print(patients[p])
        # print('preictals count:', len(all_preictals[p]))
        # print('interictals count:', len(all_interictals[p]))
        if not method: # if method is empty 
            method = input('Oversample preictals, or undersample interictals? o/(U):').lower()
            if not method:
                method = 'u'

            while applyAll != '' and applyAll != 'y' and applyAll != 'n':
                applyAll = input('Apply to all cases? (Y)/n:').lower()
                if applyAll == 'n':
                    method = ''
        # print('method:', method) # TEST
        # print('applyAll:', applyAll) # TEST

        # create dirs
        patient_dir = os.path.join(seq_dir, patients[p])
        preictals_dir = os.path.join(patient_dir, 'preictals')
        interictals_dir = os.path.join(patient_dir, 'interictals')

        for _dir in (patient_dir, preictals_dir, interictals_dir):
            if not os.path.exists(_dir):
                os.makedirs(_dir)

        numPreictals = len(all_preictals[p])
        numInterictals = len(all_interictals[p])

        if method == 'o':

            print('Preictal segments:', numPreictals)
            if numPreictals < numInterictals:    
                # make total number of preictal seqs equivalent to total number of interictal seqs
                totInterictalSeqs = numInterictals * seq_cnt  
                singlePreictalSeqs = totInterictalSeqs // numPreictals
                totCropCnt = 0 # TEST
                print('Preictal sequences:', singlePreictalSeqs * numPreictals) #### TEST
                print('Processing..')
                for preictal in all_preictals[p]:
                    raw = mne.io.read_raw_fif(preictal, verbose=False)
                    #print(raw)
                    totTime = int(raw.times[-1]) # sec

                    startT, endT= 0, 0
                    currCropCnt = 0 #### TEST
                    #print('Cropping preictal sequences...')
                    if singlePreictalSeqs > seq_cnt:
                    # sliding window oversampling on preictals. 
                        window_gap = totTime / singlePreictalSeqs # same as totTime / singlePreictalSeqs.startTs
                        if window_gap < period: # if window gap is less than 1/new_sfreq
                            window_gap = period
                        for seq in range(singlePreictalSeqs):
                            endT = startT + seq_len 
                            if endT <= totTime: # safety net
                                currCropCnt += 1
                                realCropCnt = currCropCnt + totCropCnt
                                fname = os.path.join(preictals_dir, patients[p] + '_preictal_' + str(realCropCnt).zfill(len(str(singlePreictalSeqs))) + '_raw.fif')
                                crop = raw.copy().crop(tmin=float(startT), tmax=float(endT), include_tmax=False)
                                crop.save(fname, overwrite=inp_overwrite) # to save as mne native .fif format 
                            startT += window_gap         
                    else:
                        num_seq = totTime // seq_len
                        for seq in range(num_seq):
                            endT = startT + seq_len 
                            if endT <= totTime: # safety net
                                currCropCnt += 1
                                realCropCnt = currCropCnt + totCropCnt
                                fname = os.path.join(preictals_dir, patients[p] + '_preictal_' + str(realCropCnt).zfill(len(str(num_seq))) + '_raw.fif')
                                crop = raw.copy().crop(tmin=float(startT), tmax=float(endT), include_tmax=False)
                                crop.save(fname, overwrite=inp_overwrite) # to save as mne native .fif format 
                            startT = endT  
                    # print('final endT:', endT) #### TEST
                    # print('currCropCnt expected == actual:', singlePreictalSeqs==currCropCnt) #### TEST
                    totCropCnt += currCropCnt 
                print('Expected seqs == Actual crops:', singlePreictalSeqs * numPreictals == totCropCnt, totCropCnt)

            print('Interictal segments:', numInterictals)
            print('Interictal sequences:', seq_cnt * numInterictals)
            print('Processing..')
            totCropCnt = 0 # TEST
            for interictal in all_interictals[p]:
                raw = mne.io.read_raw_fif(interictal, verbose=False)
                #print(raw)
                totTime = int(raw.times[-1]) # sec
                num_seq = totTime // seq_len
                #print('seq_cnt = num_seq:', seq_cnt == num_seq, num_seq) #### TEST

                startT, endT= 0, 0          
                currCropCnt = 0 #### TEST
                #print('Cropping interictal sequences...')
                for seq in range(num_seq):
                    endT = startT + seq_len
                    if endT <= totTime: # safety net
                        currCropCnt += 1
                        realCropCnt = currCropCnt + totCropCnt
                        fname = os.path.join(interictals_dir, patients[p] + '_preictal_' + str(realCropCnt).zfill(len(str(num_seq))) + '_raw.fif')
                        crop = raw.copy().crop(tmin=float(startT), tmax=float(endT), include_tmax=False)
                        crop.save(fname, overwrite=inp_overwrite) # to save as mne native .fif format 
                    startT = endT
                # print('final endT:', endT) #### TEST
                # print('currCropCnt expected == actual:', num_seq==currCropCnt) #### TEST
                totCropCnt += currCropCnt 
            print('Expected seqs == Actual crops:', num_seq * numInterictals == totCropCnt, totCropCnt)

            
        #
        #
        #
        ### UNDERSAMPLING ###
        
        if method == 'u':

            print('Preictal segments:', numPreictals)
            preict_totCropCnt = 0 # TEST
            for preictal in all_preictals[p]:
                raw = mne.io.read_raw_fif(preictal, verbose=False)
                totTime = int(raw.times[-1]) # sec
                startT, endT= 0, 0
                currCropCnt = 0 # TEST
                window_gap = period 
                num_seq = int(totTime - seq_len) // window_gap
                print('Preictal sequences:', num_seq * numPreictals) #### TEST
                print('Processing..')
                for seq in range(num_seq):
                    endT = startT + seq_len 
                    if endT <= totTime: # safety net
                        currCropCnt += 1
                        realCropCnt = currCropCnt + preict_totCropCnt
                        fname = patients[p] + '_preictal_' + str(realCropCnt).zfill(len(str(len(num_seq)))) + '_raw.fif'
                        fpath = os.path.join(preictals_dir, fname)
                        crop = raw.copy().crop(tmin=float(startT), tmax=float(endT), include_tmax=False)
                        crop.save(fname, overwrite=inp_overwrite) # to save as mne native .fif format 
                    startT += window_gap 
                preict_totCropCnt += currCropCnt 
            print('preict_totCropCnt:', preict_totCropCnt)


            print('Interictal segments:', numInterictals)
            print('Interictal sequences:', seq_cnt * numInterictals)
            print('Processing..')
            intict_totCropCnt = 0 # TEST
            rand_inticts = []
            if preict_totCropCnt > len(all_interictals[p]):
                preict_modulo = preict_totCropCnt % len(all_interictals[p])
                for i in range(preict_totCropCnt // len(all_interictals[p])):
                    rand_inticts += all_interictals[p]
                rand_inticts += random.sample(all_interictals[p], preict_modulo)

            else:            
                rand_inticts += random.sample(all_interictals[p], preict_totCropCnt)

            for interictal in rand_inticts:
                raw = mne.io.read_raw_fif(interictal, verbose=False)
                totTime = int(raw.times[-1]) # sec
                startT = random.randint(0, totTime - seq_len)
                endT = startT + seq_len         
                currCropCnt = 0 #### TEST
                if endT <= totTime: # safety net
                    currCropCnt += 1
                    realCropCnt = currCropCnt + intict_totCropCnt
                    fname = patients[p] + '_interictal_' + str(realCropCnt).zfill(len(str(len(rand_inticts)))) + '_raw.fif'
                    fpath = os.path.join(interictals_dir, fname)
                    crop = raw.copy().crop(tmin=float(startT), tmax=float(endT), include_tmax=False)
                    crop.save(fname, overwrite=inp_overwrite) # to save as mne native .fif format 
                intict_totCropCnt += currCropCnt 
            print('intict_totCropCnt:', intict_totCropCnt)

        print('Saved at', patient_dir)
        print()

sequencer(patients, all_preictals, all_interictals)
print('All processes complete.\nTerminating program\n.')
# penultimate line
 