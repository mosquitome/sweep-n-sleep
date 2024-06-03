# 🧹 sweep-n-sleep 💤
This repository contains code used to analyse mosquito nerve, and Laser Doppler Vibrometry (LDV) data, in Ellis et al. 2024. In the study, we stimulated mosquito sounds recievers (flagellae) with a series of 1 second sweeps (from 0-1000 Hz), interspersed with 1 second periods of rest. There are two pieces of code: one which quantifies spontaneous firing in the nerve during unstimulated periods (unstimulatedNerveAnalysis.py), and one which quantifies peak responses of both the nerve and flagellum to sweep stimulation (stimulatedNerveLaser.py) by fitting splines to the data and calling peaks in these splines.
### unstimulatedNerveAnalysis.py
This script requires a folder with the following structure:
```
- individual_1
  - fileSplitter_before
    - epoch_1
    - epoch_2
    - epoch_n
  - fileSplitter_after
    - epoch_1
    - epoch_2
    - epoch_n
- individual_2
- individual_n
```
