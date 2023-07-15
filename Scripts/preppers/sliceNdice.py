'''
Designed by HIRC Lab @ SNUBH in ROK.
'''

# if seq_dir exists, ask, default: skip preprocessing. 

# all imports
import numpy as np
import pandas as pd
import mne  

import os, shutil, random, time, io
from pathlib import Path

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
mne.set_log_level(verbose=False)


##################################################################################
# Settings  # CHECKER
seg_lens = (5*60, 10*60, 30*60)#sec
seq_lens = (30,)#sec 
horizon_len = 5 *60#sec # pre-seizure horizon of 5 min
buffer_len = 4 *60*60#sec # 4hr buffer for interictal (one side)
new_sfreq = 100.0 # og = 200.0

period = (1 / new_sfreq) #* 100 # QUICK TEST ### CHECKER

# Directories
path = "/home/SharedFiles/Data/HospitalData/EEG/SNUCH_VEM_EDF"
annot_dir = '/home/SharedFiles/Projects/EEG/Interim/annotated_raws'
seg_dir = '/home/SharedFiles/Projects/EEG/Interim/seg_raws'
seq_dir = '/home/SharedFiles/Projects/EEG/Interim/seq_raws'
arr_dir = '/home/SharedFiles/Projects/EEG/Inputs/seq_arr/'
output_dir = '/home/SharedFiles/Projects/EEG/Outputs/'


# R/W
inp_overwrite = True
f = open(os.path.join(output_dir, 'data_run_' + time.strftime("%Y%m%d%H%M%S") + '.txt'), 'w')
##################################################################################
# def make_dir(_dir):
#     if os.path.exists(_dir):
#         #inp = input('{} already exist. Delete? (Y)/n:'.format(_dir))
#         inp = 'n' ####TEST
#         if inp != 'n':       
#             shutil.rmtree(_dir)
#     os.makedirs(_dir, exist_ok=True)
    
def fetcher(_dir):
    ## retrieve all paths
    print('retrieving all files...')
    patients, all_preictals, all_interictals = [], [], []
    for patient in os.listdir(_dir):
        patient_pth = os.path.join(_dir, patient)
        if patient.startswith('SNUCH'):
            # and patient.startswith('SNUCH07'):# or patient.startswith('SNUCH07') or patient.startswith('SNUCH10'): # CHECKER
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
                                
    print('len of all_preictals == all_interictals == patients:', 
        len(all_preictals) == len(all_interictals) == len(patients) )
    return patients, all_preictals, all_interictals

def conductor(seg_len, seq_len, arr_type):
    for it in os.listdir(annot_dir): # for every annotated fif
        if not it.startswith('.') and it.endswith('.fif'):
            # and it.startswith('SNUCH07'): # or it.startswith('SNUCH07') or it.startswith('SNUCH10')): ##### CHECKER 
        # if non-ghost fif file
            # create dirs
            patient = it.split('_')[0]
            patient_dir = os.path.join(seg_dir, patient)
            preictals_dir = os.path.join(patient_dir, 'preictals')
            interictals_dir = os.path.join(patient_dir, 'interictals')
            for _dir in (preictals_dir, interictals_dir):
                if not os.path.exists(_dir):
                    os.makedirs(_dir)

            # load
            fif_pth = os.path.join(annot_dir, it)
            raw = mne.io.read_raw_fif(fif_pth, verbose=False)
            annots = mne.read_annotations(fif_pth)
            
            # segment
            segmenter(raw, annots, patient, preictals_dir, interictals_dir, inp_overwrite, 
                seg_len, seq_len, horizon_len, buffer_len, new_sfreq)
            
    # sequence
    patients, all_preictals, all_interictals = fetcher(seg_dir)
    sequencer(patients, all_preictals, all_interictals, seg_len, seq_len)
    
    # arrayify
    patients, all_preictals, all_interictals = fetcher(seq_dir)
    arrayifier(patients, all_preictals, all_interictals, seg_len, seq_len, arr_type)

#################################################################################
# actors
def segmenter(raw, annots, patient, preictals_dir, interictals_dir, inp_overwrite, 
        seg_len, seq_len, horizon_len, buffer_len, new_sfreq):   
    print('***starting segmenter...***')
    seq_cnt = seg_len // seq_len
    
    # constants
    tot_time = np.round(raw.times[-1])
    
    # lists
    interictal_annots = [0.0]
    preictal_segs, interictals, interictal_segs = [], [], []
    
    # check
    print(patient)
    print('Patient:', patient, file=f)
    print(annots)
    print('Onsets:', len(annots), file=f)
    
    ## create preictal segments and possible interictal points
    for annot in annots:
        # PREICTAL segments
        preictal_end = annot['onset'] - horizon_len # prior to seizure horizon (exclusive)
        preictal_start = preictal_end - seg_len 
        
        if preictal_start > 0: # if within bounds
            if len(preictal_segs) == 0: 
                preictal_segs.append([preictal_start, preictal_end])
            elif preictal_start > preictal_segs[-1][1]: # avoid preictal overlap 
                # if next preictal comes AFTER previous preictal:
                preictal_segs.append([preictal_start, preictal_end])
                
                
        # INTERICTAL segments 
        # add buffer at left or right of seizure
        interictal_left = annot['onset'] - buffer_len # buffer_len before seizure
        interictal_right = annot['onset'] + annot['duration'] + buffer_len # buffer_len after seizure
        
        if interictal_left > 0:
            interictal_annots.append(interictal_left)
        if interictal_right < tot_time - seg_len:
            interictal_annots.append(interictal_right)

    
    # interictal partitioning
    '''
        ==Note==
        - Find smarter method for interictal filtering. Too heavy right now.
    '''
    interictal_annots = sorted(interictal_annots) # order from low to high
    interictal_annots.append(tot_time)
    for a in range(len(interictal_annots)-1):
        if interictal_annots[a+1] - interictal_annots[a] > seg_len: 
        # interictal is above seg_len
            no_overlap = True
            for preictal_seg in preictal_segs: 
                #if preictal_seg[0] >= interictal_annots[a] and preictal_seg[0] <= interictal_annots[a+1]:  
                ## faster better method. double check logic. filter for preictal_seg[1] too?
                if (int(preictal_seg[0]), int(preictal_seg[1])) in range(int(interictal_annots[a]), 
                    int(interictal_annots[a+1])+1): #overkill, too heavy.
                    no_overlap = False # if overlapping with any preictal_starts 
                    
            if no_overlap:
                interictals.append([interictal_annots[a], interictal_annots[a+1]])
                                                                              

    # TEST
    print('preictal_segs:', len(preictal_segs))
    print('preictal_segs:', len(preictal_segs), file=f)
    if len(preictal_segs)==0 or len(interictals)==0:
        print('preictal_segs:', preictal_segs)
        print('interictal_portions:', interictals)
    if len(interictals) <= 0:
        print('interictal_annots', interictal_annots)
        print('***')
        print('***WARNING: num interictals is 0. This patient will be skipped. Decrease buffer_len and rerun.***')
        print('***')
        ############## ADD: decrease buffer_len by seg_len til seg_len per itertation, restart curr call from top ###########
        print()
        return
    
    notice=''
    # interictal segments    
    for interictal in interictals:
        num_interictal_segs = int((interictal[1] - interictal[0]) // seg_len)
        interictal_seg_start = interictal[0]
        for num in range(num_interictal_segs): # see if this and line above can be combined into range([0],[1],step=seg_len)
            interictal_seg_end = interictal_seg_start + seg_len
            if interictal_seg_end <= interictal[1]:
                interictal_segs.append([interictal_seg_start, interictal_seg_end])             
            else: #if interictal_seg_end > interictal[1]:
                notice = "***NOTE: Omitted last possible interictal pair; outside of total interictal range: ({} ~){} > {}.***" \
                      .format(interictal_seg_start, interictal_seg_end, interictal[1])
                break
            interictal_seg_start = interictal_seg_end
    
    if notice:
        print(notice)
    
    # TEST
    print('interictal_portions:', len(interictals))
    print('interictal_segs:', len(interictal_segs))
    print('interictal_portions:', len(interictals), file=f)
    print('interictal_segs:', len(interictal_segs), file=f)
    if len(interictal_segs)==0:
        print('preictal_segs:', preictal_segs)
        print('interictal_annots:', interictal_annots)
        print('interictal_portions:', interictals)
        print('num_interictal_segs:', num_interictal_segs)
        print('interictal_segs:', interictal_segs)

    ## Crop 'n Save segments
    print('Saving preictals...')
    for seg_num, preictal_seg in enumerate(preictal_segs[:]): ### CHECKER
        preictal_crop = raw.copy().crop(tmin=preictal_seg[0], tmax=preictal_seg[1], include_tmax=False)
        
        # Downsample
        preictal_crop = preictal_crop.resample(new_sfreq, n_jobs=-1)
        
        # Save
        fname =  patient + '_preictal_seg_' + str(seg_num+1).zfill(len(str(len(preictal_segs)))) + '_raw.fif'
        fpath = os.path.join(preictals_dir, fname)
        preictal_crop.save(fpath)
        print(fname)
    
    print('Saving interictals...')
    for seg_num, interictal_seg in enumerate(interictal_segs[:]): ### CHECKER
        interictal_crop = raw.copy().crop(tmin=interictal_seg[0], tmax=interictal_seg[1], include_tmax=False)
        
        # Downsample
        interictal_crop = interictal_crop.resample(new_sfreq, n_jobs=-1)
        
        # Save
        fname = patient + '_interictal_seg_' + str(seg_num+1).zfill(len(str(len(interictal_segs)))) + '_raw.fif'
        fpath = os.path.join(interictals_dir, fname)                      
        interictal_crop.save(fpath)
        print(fname)
    print()
    

def sequencer(patients, all_preictals, all_interictals, seg_len, seq_len):
    seq_cnt = seg_len // seq_len
    ## Sequence Slicer
    print('***starting sequencer...***')
    print('***starting sequencer...***', file=f)
    for p in range(len(patients)): 
        print(patients[p])
        print(patients[p], file=f)
        # print('preictals count:', len(all_preictals[p]))
        # print('interictals count:', len(all_interictals[p]))

        # create dirs
        patient_dir = os.path.join(seq_dir, patients[p])
        preictals_dir = os.path.join(patient_dir, 'preictals')
        interictals_dir = os.path.join(patient_dir, 'interictals')

        for _dir in (patient_dir, preictals_dir, interictals_dir):
            if not os.path.exists(_dir):
                os.makedirs(_dir)

        numPreictals = len(all_preictals[p])
        numInterictals = len(all_interictals[p])

        
        ### OVERSAMPLING ###
        ''' 
        * over sample preictals to match cnt interictal seq  
        * make total number of preictal seqs equivalent to total 
            number of interictal seqs
        ''' 
        totInterictalSeqs = numInterictals * seq_cnt
        singlePreictalSeqs = totInterictalSeqs // numPreictals
        
        print('Preictal segments:', numPreictals)
        print('singlePreictalSeqs:', singlePreictalSeqs) ## TEST
        print('Expected preictal sequences:', singlePreictalSeqs * numPreictals) #### TEST
        print('Processing..')
        totCropCnt = 0 # TEST
        for preictal in all_preictals[p]:
            raw = mne.io.read_raw_fif(preictal, verbose=False)
            #print(raw)
            totTime = int(raw.times[-1]) # sec

            startT, endT= 0, 0
            currCropCnt = 0 #### TEST
            #print('Cropping preictal sequences...')

            if singlePreictalSeqs > seq_cnt:
                #print('singlePreictalSeqs > seq_cnt') ## TEST
                # sliding window oversampling on preictals. 
                window_gap = (totTime - seq_len) / singlePreictalSeqs # same as totTime / singlePreictalSeqs.startTs
                if window_gap < period: # if window gap is less than 1/new_sfreq
                    window_gap = period
                    print('window_gap:', window_gap)
                
                for seq in range(singlePreictalSeqs):
                    endT = startT + seq_len
                    if endT <= totTime: # safety net
                        currCropCnt += 1
                        realCropCnt = currCropCnt + totCropCnt
                        fpath = os.path.join(preictals_dir, patients[p] + '_preictal_' + str(realCropCnt).zfill(
                            len(str(singlePreictalSeqs))) + '_raw.fif')
                        crop = raw.copy().crop(tmin=float(startT), tmax=float(endT), include_tmax=False)
                        crop.save(fpath) # to save as mne native .fif format 
                    startT += window_gap         
            else:
                print('singlePreictalSeqs < seq_cnt') ## TEST
                num_seq = totTime // seq_len
                for seq in range(num_seq):
                    endT = startT + seq_len 
                    if endT <= totTime: # safety net
                        currCropCnt += 1
                        realCropCnt = currCropCnt + totCropCnt
                        fpath = os.path.join(preictals_dir, patients[p] + '_preictal_' + str(realCropCnt).zfill(
                            len(str(num_seq))) + '_raw.fif')
                        crop = raw.copy().crop(tmin=float(startT), tmax=float(endT), include_tmax=False)
                        crop.save(fpath) # to save as mne native .fif format 
                    startT = endT  
            # print('final endT:', endT) #### TEST
            # print('currCropCnt expected == actual:', singlePreictalSeqs==currCropCnt) #### TEST
            totCropCnt += currCropCnt 
        #print('Expected seqs == Actual crops:', singlePreictalSeqs * numPreictals == totCropCnt, 
        # 'Actual', totCropCnt) ## TEST
        print('Actual preictal sequences:', totCropCnt)
        print('Preictal sequences:', totCropCnt, file=f)


        print('Interictal segments:', numInterictals)
        print('Expected interictal sequences:', seq_cnt * numInterictals)
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
                    fpath = os.path.join(interictals_dir, patients[p] + '_interictal_' + str(realCropCnt).zfill(
                        len(str(num_seq))) + '_raw.fif')
                    crop = raw.copy().crop(tmin=float(startT), tmax=float(endT), include_tmax=False)
                    crop.save(fpath) # to save as mne native .fif format 
                startT = endT
            # print('final endT:', endT) #### TEST
            # print('currCropCnt expected == actual:', num_seq==currCropCnt) #### TEST
            totCropCnt += currCropCnt 
        #print('Expected seqs == Actual crops:', num_seq * numInterictals == totCropCnt, '| Actual:', totCropCnt) ## TEST
        print('Actual interictal sequences:', totCropCnt)
        print('Interictal sequences:', totCropCnt, file=f)
            
            
#         ### MAX SAMPLING ###
#         # Max sample preictals and make interictal count match
#         print('Preictal segments:', numPreictals)
#         preict_totCropCnt = 0 # TEST
#         for preictal in all_preictals[p]:
#             raw = mne.io.read_raw_fif(preictal, verbose=False)
#             totTime = int(raw.times[-1]) # sec
#             startT, endT= 0, 0
#             currCropCnt = 0 # TEST
#             window_gap = period # minimum time step
#             num_seq = int((totTime - seq_len) / window_gap)
#             print('Preictal sequences:', num_seq * numPreictals) #### TEST
#             print('Processing..')
#             for seq in range(num_seq):
#                 endT = startT + seq_len 
#                 if endT <= totTime: # safety net
#                     currCropCnt += 1
#                     realCropCnt = currCropCnt + preict_totCropCnt
#                     fname = patients[p] + '_preictal_' + str(realCropCnt).zfill(len(str(num_seq))) + '_raw.fif'
#                     fpath = os.path.join(preictals_dir, fname)
#                     crop = raw.copy().crop(tmin=float(startT), tmax=float(endT), include_tmax=False)
#                     crop.save(fpath) # to save as mne native .fif format 
#                 startT += window_gap 
#             preict_totCropCnt += currCropCnt 
#         print('preict_totCropCnt:', preict_totCropCnt)
#         print('Preictal sequences:', preict_totCropCnt, file=f)


#         print('Interictal segments:', numInterictals)
#         print('Processing..')
#         intict_totCropCnt = 0 # TEST
#         rand_inticts = []
#         if preict_totCropCnt > len(all_interictals[p]):
#             preict_remain = preict_totCropCnt % len(all_interictals[p])
#             for i in range(preict_totCropCnt // len(all_interictals[p])):
#                 rand_inticts += all_interictals[p]
#             rand_inticts += random.sample(all_interictals[p], preict_remain)

#         else:            
#             rand_inticts += random.sample(all_interictals[p], preict_totCropCnt)
        
#         print('rand_inticts len:', len(rand_inticts))

#         for interictal in rand_inticts:
#             raw = mne.io.read_raw_fif(interictal, verbose=False)
#             totTime = int(raw.times[-1]) # sec
#             startT = random.randint(0, totTime - seq_len)
#             endT = startT + seq_len         
#             currCropCnt = 0 #### TEST
#             if endT <= totTime: # safety net
#                 currCropCnt += 1
#                 realCropCnt = currCropCnt + intict_totCropCnt
#                 fname = patients[p] + '_interictal_' + str(realCropCnt).zfill(len(str(num_seq))) + '_raw.fif'
#                 fpath = os.path.join(interictals_dir, fname)
#                 crop = raw.copy().crop(tmin=float(startT), tmax=float(endT), include_tmax=False)
#                 crop.save(fpath) # to save as mne native .fif format 
#             intict_totCropCnt += currCropCnt 
#         print('intict_totCropCnt:', intict_totCropCnt)
#         print('Interictal sequences:', intict_totCropCnt, file=f)
        
        

        print('Saved at', patient_dir)
        print()
        
def arrayifyNsave(raw, fname):
    # pick channels
    eeg_chans = [
        'Fp1-AVG',
        'F3-AVG',
        'C3-AVG',
        'P3-AVG',
        'Fp2-AVG',
        'F4-AVG',
        'C4-AVG',
        'P4-AVG',
        'F7-AVG',
        'T1-AVG',
        'T3-AVG',
        'T5-AVG',
        'O1-AVG',
        'F8-AVG',
        'T2-AVG',
        'T4-AVG',
        'T6-AVG',
        'O2-AVG',
        'Fz-AVG',
        'Cz-AVG',
        'Pz-AVG',
    ]
    eeg_raw = raw.pick(eeg_chans)
    
    # arrayify
    eeg_raw_data = eeg_raw.get_data()
    #print(eeg_raw_data.shape) ## TEST
    
    # save as np.array
    np.save(fname, eeg_raw_data)
        
def arrayifier(patients, all_preictals, all_interictals, seg_len, seq_len, arr_type):
    ## Applier
    print('***starting arrayifier...***')
    for p in range(len(patients)): 
        print(patients[p])
        # create dirs
        patient_dir = os.path.join(arr_dir, arr_type, patients[p])
        preictals_dir = os.path.join(patient_dir, 'preictals')
        interictals_dir = os.path.join(patient_dir, 'interictals')

        for _dir in (patient_dir, preictals_dir, interictals_dir):
            if not os.path.exists(_dir):
                os.makedirs(_dir)

        print('Arrayifying preictals...')
        for preictal in all_preictals[p]:
            raw = mne.io.read_raw_fif(preictal, verbose=False)
            fname = os.path.basename(preictal).split('.')[0][:-3] + 'arr'
            fpath = os.path.join(preictals_dir, fname)
            arrayifyNsave(raw, fpath)

        print('Arrayifying interictals...')
        for interictal in all_interictals[p]:
            raw = mne.io.read_raw_fif(interictal, verbose=False)
            fname = os.path.basename(interictal).split('.')[0][:-3] + 'arr'
            fpath = os.path.join(interictals_dir, fname)
            arrayifyNsave(raw, fpath)

        print('Saved at', patient_dir)
        print()

            
    
###################################################################################
def main():
    # delete any previous temporary saves 
    for _dir in (seg_dir, seq_dir): # CHECKER
        if os.path.exists(_dir):    
            shutil.rmtree(_dir)
            
    for seg_len in seg_lens:
        for seq_len in seq_lens:
            arr_type = 'seg{}m_seq{}s'.format(str(int(seg_len/60)).zfill(2), seq_len)
            print('Running for output type:', arr_type)
            print('Data info:', arr_type, file=f)
            conductor(seg_len, seq_len, arr_type)
            print('\n\n', file=f)
            
            # delete temporary saves
            print('Deleting temporary saves..')
            for _dir in (seg_dir, seq_dir):
                if os.path.exists(_dir):    
                    shutil.rmtree(_dir)

    print('All processes complete.\nTerminating program\n.')
    f.close()
    
main()
# penultimate line
