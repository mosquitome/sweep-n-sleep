# 🧹 sweep-n-sleep 💤
This repository contains code used to analyse mosquito nerve, and Laser Doppler Vibrometry (LDV) data, in Ellis et al. 2024. In the study, we stimulated mosquito sounds recievers (flagellae) with a series of 1 second sweeps (from 0-1000 Hz), interspersed with 1 second periods of rest. There are two pieces of code: one which quantifies spontaneous firing in the nerve during unstimulated periods (unstimulatedNerveAnalysis.py), and one which quantifies peak responses of both the nerve and flagellum to sweep stimulation (stimulatedNerveLaser.py) by fitting splines to the data and calling peaks in these splines.
### unstimulatedNerveAnalysis.py
This script requires a folder with the following structure, where run_* refers to a tab-delimited text file with two columns (time and voltage):
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
### stimulatedNerveAnalysis.py
This script requires...
