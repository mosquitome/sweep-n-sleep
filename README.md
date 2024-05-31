# sweep-n-sleep
This repository contains code used to analyse mosquito nerve (and Laser Doppler Vibrometry) data in Ellis et al. 2024. In the study, we stimulated mosquito sounds recievers with a series of 1 second sweeps (from 0-1000 Hz), interspersed with 1 second periods of rest.

There are two pieces of code: one which quantifies firing in the nerve during unstimulated periods, and one which quantifies peak responses of both the nerve (electrical data) and sound reciever (LDV data) to sweep stimulation by fitting splines to the data.