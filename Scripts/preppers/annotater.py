'''
Annotater script for EDF files, output in RAW
Designed by HIRC Lab @ SNUBH in ROK.
'''

# all imports
import numpy as np
import pandas as pd
import openpyxl

import mne

import datetime, os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# start
print('executing program...')

# pathing

# test
input_path = "/home/SharedFiles/Data/HospitalData/EEG/SNUCH_VEM_EDF"
annot_path = '/home/SharedFiles/Projects/EEG/Interim/annotated_raws'
xl_path = os.path.join(input_path, 'SNUCH 2020 labeling_edit_20220303.xlsx')

# #final
# path = input('Enter input directory path:')
# annot_path = os.path.join(path, 'annotated_raws')
# xl_path = input('Enter path to excel sheet:')

os.makedirs(annot_path, exist_ok=True)
# load edfs
print('loading edfs as raws...')
patients=[]
raw_paths = []
for it in os.listdir(input_path):
    if not it.startswith('.') and it.lower().endswith('.edf'): # filter-out ghosts
        patients.append(it.split('.')[0])
        raw_paths.append(os.path.join(input_path, it))
        

# load excel
print('loading excel...')
df = pd.concat(pd.read_excel(xl_path, sheet_name=None), ignore_index=False)

# prep df
# fill empty date rows
for i in range(len(df.index)):
    pos = df.iat[i, 0]
    if not pd.isnull(pos):
        currDate = pos
    else:
        df.iat[i,0] = currDate
        
df = df[['Date', 'Time', 'Annotation']]
df['Datetime'] = df.apply(lambda r : datetime.datetime.combine(
    datetime.datetime.strptime(str(r['Date']),'%Y%m%d'),r['Time']),1)
df = df[['Datetime', 'Annotation']]

# Annotate
print('starting on annotations...')
for patient, raw_path in zip(patients, raw_paths): 
    raw = mne.io.read_raw_edf(raw_path, verbose=True)
    print(patient)
                         
    raw_data = raw.load_data()    
    orig_time = df.loc[patient]['Datetime'][0]
    start_time = df[df['Annotation'].str.contains(
        'Start of seizure')].loc[patient]['Datetime'].reset_index(drop=True)
    end_time = df[df['Annotation'].str.contains(
        'End of seizure')].loc[patient]['Datetime'].reset_index(drop=True)
        # leave as floats, because mne.Annotations only takes in array of floats
    
    onsets = np.round(np.array((start_time - orig_time).dt.total_seconds())) 
    durations = np.round(np.array((end_time - start_time).dt.total_seconds())) 
    
#     # TESTS
#     print('orig_time:', orig_time)
#     print('start_time:', start_time)
#     print('end_time:', end_time)
#     print('onsets:', onsets)
#     print('durations:', durations)
    
    # filter out negatives
    for i, (onset, duration) in enumerate(zip(onsets, durations)):
        if onset < 0 or duration < 0:
            print('negative warning (onset, duration):', onset, duration)
            onsets = np.delete(onsets, i)
            durations = np.delete(durations, i)
    
    descriptions = ['ictal' for i in range(len(onsets))]
    print('length equivalency:', len(onsets)==len(durations)==len(descriptions)) ### TEST
    
    curr_annot = mne.Annotations(
                            onset=onsets,
                            duration=durations,
                            description=descriptions,
                                )
    print(curr_annot)
        
    # Set
    print('setting annotations...')
    raw_annot = raw_data.set_annotations(curr_annot)

    # Save
    fname = patient + '_annotated_raw.fif'
    fpath = os.path.join(annot_path, fname)
    raw_annot.save(fpath, overwrite=True)  # saved to annot_path #given at the top
    print("Saved.")
    
    # mem clear
    del raw, raw_data, raw_annot
    
    print()
    
print('All processes complete.')
# penultimate line
