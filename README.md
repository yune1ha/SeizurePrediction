# SeizurePrediction

Predicts seizure (detects preictal pattern) 5 min + up to 30 min prior to onset (ictal). 

Script directory contains the final executable codes, whereas the notebook directory is for reference only (various EDA, prep, training processes).

Inside script directory lies preppers and modelers directories, whose main scripts are to be executed in said order: 
* 1.) The preppers directory's main script is SliceNDice.py, which performs its namesake by converting and window slicing og raw .edf files.
* 2.) The modelers directory's main script is main.py (feel free to switch out the default chosen model with any of the other models or your own).

Always change any directory path to point to your actual input/output directory. All interm directories will be created at pwd. 
