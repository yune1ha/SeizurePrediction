'''
Slices annotated raw files into seg_len segments
according to desired horizon_len, segment_len, seguence_len, and buffer_len(around ictal) settings.
Downsamples sfreq after cropping.

Designed by HIRC Lab @ SNUBH in ROK.
'''

# all imports
import numpy as np
import pandas as pd
import mne  

import os, shutil
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
#############################################

print('Executing program...')

## pathing
path = "/home/SharedFiles/Data/HospitalData/EEG/SNUCH_VEM_EDF"
annot_dir = '/home/SharedFiles/Projects/EEG/Interim/annotated_raws'
seg_dir = '/home/SharedFiles/Projects/EEG/Interim/seg_raws'

# Outter Settings
inp_overwrite = True

def make_dir(_dir):
    if os.path.exists(_dir):
        shutil.rmtree(_dir)
    os.makedirs(_dir)

def main():
    make_dir(seg_dir)
    
    for it in os.listdir(annot_dir): # for every annotated fif
        if not it.startswith('._') and it.endswith('.fif'):  
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
            
            # slice
            segmenter(raw, annots, patient, preictals_dir, interictals_dir, inp_overwrite)
            
    print('All processes complete.\nTerminating program.')

        
def segmenter(raw, annots, patient, preictals_dir, interictals_dir, inp_overwrite, seg_len, seq_len, horizon_len, buffer_len, new_sfreq):
    # derivables
    seq_cnt = seg_len // seq_len
    
    # constants
    tot_time = np.round(raw.times[-1])
    
    # lists
    interictal_annots = [0.0]
    preictal_segs, interictals, interictal_segs = [], [], []
    
    # check
    print(patient)
    print(annots)
    
    ## create preictal segments and possible interictal points
    for annot in annots: 
        # PREICTAL segments
        preictal_end = annot['onset'] - horizon_len # prior to seizure horizon (exclusive)
        preictal_start = preictal_end - seg_len 
        
        if preictal_start > 0: # if within bounds
            if len(preictal_segs) == 0: 
                preictal_segs.append([preictal_start, preictal_end])
            elif preictal_start > preictal_segs[-1][1]: # avoid preictal overlap
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
    print('preictal_segs:\t\t\t {}'.format(len(preictal_segs)))
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
    
    if not notice:
        print(notice)
    
    # TEST
    print('interictal_portions -> _segs: {} -> {}'.format(len(interictals), len(interictal_segs)))
    if len(interictal_segs)==0:
        print('preictal_segs:', preictal_segs)
        print('interictal_annots:', interictal_annots)
        print('interictal_portions:', interictals)
        print('num_interictal_segs:', num_interictal_segs)
        print('interictal_segs:', interictal_segs)

    ## Crop 'n Save segments
    print('Saving preictals...')
    for seg_num, preictal_seg in enumerate(preictal_segs):
        preictal_crop = raw.copy().crop(tmin=preictal_seg[0], tmax=preictal_seg[1], include_tmax=False)
        
        # Downsample
        preictal_crop = preictal_crop.resample(new_sfreq, n_jobs=-1)
        
        # Save
        fname =  patient + '_preictal_seg_' + str(seg_num+1).zfill(len(str(len(preictal_segs)))) + '_raw.fif'
        fpath = os.path.join(preictals_dir, fname)
        preictal_crop.save(fpath, overwrite=inp_overwrite)
        print(fname)
    
    print('Saving interictals...')
    for seg_num, interictal_seg in enumerate(interictal_segs):
        interictal_crop = raw.copy().crop(tmin=interictal_seg[0], tmax=interictal_seg[1], include_tmax=False)
        
        # Downsample
        interictal_crop = interictal_crop.resample(new_sfreq, n_jobs=-1)
        
        # Save
        fname = patient + '_interictal_seg_' + str(seg_num+1).zfill(len(str(len(interictal_segs)))) + '_raw.fif'
        fpath = os.path.join(interictals_dir, fname)                      
        interictal_crop.save(fpath, overwrite=inp_overwrite)
        print(fname)
    print()
    
main() # call main
# penultimate line
