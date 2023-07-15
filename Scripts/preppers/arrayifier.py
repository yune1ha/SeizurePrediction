'''
Converts sliced sequences in native mne raw.fif format to np arrays.
Performs other necessary preprocessing, such as channel selection.

Designed by HIRC Lab @ SNUBH in ROK.
'''

## all imports
import numpy as np
import mne

import os
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

## pathing
path = "/home/SharedFiles/Data/HospitalData/EEG/SNUCH_VEM_EDF"
seq_dir = '/home/SharedFiles/Projects/EEG/Interim/seq_raws'
arr_dir = '/home/SharedFiles/Projects/EEG/Inputs/seq_arr/

## retrieve all paths
print('retrieving all files...')
patients, all_preictals, all_interictals = [], [], []
for patient in os.listdir(seq_dir):
    patient_pth = os.path.join(seq_dir, patient)
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

# Test                            
print('len of all_preictals = all_interictals = patients:', 
      len(all_preictals) == len(all_interictals) == len(patients) )

## Settings
inp_overwrite = True

# Preprocess & Save
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
    
    # save as np.array
    np.save(fname, eeg_raw_data)
        
def arrayifier(patients, all_preictals, all_interictals):
    ## Applier
    print('starting arrayifier...')
    for p in range(len(patients)): 
        print(patients[p])
        # create dirs
        patient_dir = os.path.join(arr_dir, patients[p])
        preictals_dir = os.path.join(patient_dir, 'preictals')
        interictals_dir = os.path.join(patient_dir, 'interictals')

        for _dir in (patient_dir, preictals_dir, interictals_dir):
            if not os.path.exists(_dir):
                os.makedirs(_dir)

        print('Arrayifying preictals...')
        for preictal in all_preictals[p]:
            raw = mne.io.read_raw_fif(preictal, verbose=False)
            fname = os.path.join(preictals_dir, os.path.basename(preictal).split('.')[0][:-3] + 'arr')
            arrayifyNsave(raw, fname)

        print('Arrayifying interictals...')
        for interictal in all_interictals[p]:
            raw = mne.io.read_raw_fif(interictal, verbose=False)
            fname = os.path.join(interictals_dir, os.path.basename(interictal).split('.')[0][:-3] + 'arr')
            arrayifyNsave(raw, fname)

        print('Saved at', patient_dir)
        print()

print('All processes complete.\nTerminating program.')
# penultimate line
