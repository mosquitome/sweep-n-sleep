'''

#---------------------------------------------------------------------------------
#              U N S T I M U L A T E D   N E R V E   A N A L Y S I S
#---------------------------------------------------------------------------------

# batch analysis of nerve data during periods of rest.

# Author        :   David A. Ellis <https://github.com/mosquitome/>
# Organisation  :   University College London

# This script will generate two files: 
    
    spikes      : contains information on real spikes above threshold.
    spikes_sim  : contains information on simulated spikes (requires real data)

# Notes         : Below, set the sample rate (fs) and the peak threshold in standard deviations 
                  (threshold).

#---------------------------------------------------------------------------------

'''

import pandas as pd
import numpy as np
import os
import pathlib as pl
import scipy.signal as ss

fs = 20000 # <- nerve sample rate
threshold = 5 # <- number of standard deviations a peak needs to be from the baseline in order to be counted as a spike

#==============================================================================
# C A L L   S P I K E S   A N D   E X T R A C T   I S I
#==============================================================================

folders = [i for i in os.listdir() if i[-3:] not in ['txt', 'pdf', '.py']] # <- get a list fo all subdirectories in the current working directory | each subdirectory corresponds to a mosquito and contains multiple text files, each with 1 second long samples of nerve data, for before and after injection | ignore any PDF or TXT files in the current working directory 

spikes = pd.DataFrame()
for folder in folders:
    print('\n' + folder)
    for treatment in ['fileSplitter_before', 'fileSplitter_after']:
        print('\t', treatment)

        for idx, file in enumerate(os.listdir(pl.Path(folder, treatment))):
            _ = pd.read_table(pl.Path(folder, treatment, file)).dropna()
            y = _['2 nerve'] - _['2 nerve'].median()
            if len(_.dropna()) == 0:
                continue
            p = ss.find_peaks(y * -1, prominence=0, width=0)
            pdx = list(y[p[0]] < y.std() * - threshold) # <- idx of peaks below threshold (note: median has already been subtracted so this is just looking below the threshold in the negative sign)
            if not any(_['Time'].to_numpy()[p[0]][pdx]):
                continue
            s = _['Time'].to_numpy()[p[0]][pdx] # <- time of spikes exceeding threshold
            isi = np.diff(s, prepend=np.nan)# <- inter-spike interval

            temp = pd.DataFrame({'mosquito'             : [folder] * len(s),
                                 'treatment'            : [treatment.split('_')[1]] * len(s),
                                 'epoch'                : [idx] * len(s),
                                 'spike_time'           : s,
                                 'interspike_interval'  : isi })
            spikes = pd.concat([spikes, temp]).reset_index(drop=True)

spikes['group'] = spikes['mosquito'].apply(lambda x: x.split('_')[-2] + '~' + x.split('_')[-1])

#==============================================================================
# S I M U L T E   S P I K E S   A N D   E X T R A C T   I S I   I N    S A M E   W A Y
#==============================================================================

'''
To understand how interspike-intervals would be distributed if there were no
underlying structure, a dataset of simulated spikes was generated of equal size
to the real dataset. To account for any effect of our spike-calling pipeline
(e.g. variation in spike number from epoch to epoch) spike number is matched to
real data for each epoch. Spike times were randomly chosen from a uniform
distribution.
'''

spikes_sim = pd.DataFrame()
for mosquito in spikes['mosquito'].unique():
    for treatment in spikes['treatment'].unique():
        print(mosquito, treatment)
        _ = spikes.loc[(spikes['mosquito']==mosquito) & (spikes['treatment']==treatment)]
        for epoch in _['epoch'].unique():
            n = len(_.loc[_['epoch']==epoch]) # <- number of spikes in real data for current epoch
            s = np.sort(np.random.choice(np.linspace(0, 1, fs), size=n)) # <- simulated spikes for current epoch
            isi = np.diff(s, prepend=np.nan)
            __ = pd.DataFrame({'mosquito'             : [mosquito] * n, 
                               'treatment'            : [treatment] * n, 
                               'epoch'                : [epoch] * n,
                               'spike_time'           : s, 
                               'interspike_interval'  : isi })
            spikes_sim = pd.concat([spikes_sim, __]).reset_index(drop=True)

spikes_sim['group'] = spikes_sim['mosquito'].apply(lambda x: x.split('_')[-2] + '~' + x.split('_')[-1])
