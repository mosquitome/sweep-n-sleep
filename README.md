# 🧹 sweep-n-sleep 💤
This repository contains code used to analyse mosquito nerve, and Laser Doppler Vibrometry (LDV) data, in Ellis et al. 2024. In the study, we stimulated mosquito sounds recievers (flagellae) with a series of 1 second sweeps (from 0-1000 Hz), interspersed with 1 second periods of rest. There are two pieces of code: one which quantifies spontaneous firing in the nerve during unstimulated periods (unstimulatedNerveAnalysis.py), and one which quantifies peak responses of both the nerve and flagellum to sweep stimulation (stimulatedNerveLaser.py) by fitting splines to the data and calling peaks in these splines.
### unstimulatedNerveAnalysis.py
This script requires a folder with the following structure, where run_[n] refers to a tab-delimited text file with two columns (time and voltage):
```
- individual_1
  - before
    - run_1
    - run_2
    - run_n
  - after
    - run_1
    - run_2
    - run_n
- individual_2
- individual_n
```
Text files were extracted from spike2 SMRX files using the [fileSplitter](https://github.com/hadifalex/Spike2-scripts) script from @hadifalex.
### stimulatedNerveAnalysis.py
This script analyse batches of matlab files and collates data. The output [foldername]_spline-peak-stats.txt contains various parameters extracted from splines. The script generates plots that were used to curate the stats files and remove data from erroneously fit splines. This script requires a folder with the following structure, where [direction]\_[stimulus]_laser.mat is LDV data for that individual, [direction]\_[stimulus]_nerve_DC.mat is highpass nerve data, [direction]\_[stimulus]_nerve_SMOOTH.mat is lowpass nerve data and [direction]\_[stimulus]_stim.mat is the stimulus provided:
```
- individual_1
  - before
    - bwd_H_laser.mat
    - bwd_H_nerve_DC.mat
    - bwd_H_nerve_SMOOTH.mat
    - bwd_H_stim.mat
    - fwd_h_laser.mat
    - fwd_h_nerve_DC.mat
    - fwd_h_nerve_SMOOTH.mat
    - fwd_h_stim.mat
  - after
    - bwd_H_laser.mat
    - bwd_H_nerve_DC.mat
    - bwd_H_nerve_SMOOTH.mat
    - bwd_H_stim.mat
    - fwd_h_laser.mat
    - fwd_h_nerve_DC.mat
    - fwd_h_nerve_SMOOTH.mat
    - fwd_h_stim.mat
- individual_2
- individual_n
```
