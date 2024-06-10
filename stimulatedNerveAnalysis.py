'''

#---------------------------------------------------------------------------------
#                S T I M U L A T E D   N E R V E   A N A L Y S I S
#---------------------------------------------------------------------------------

# batch analysis of averaged laser and nerve data during sweep stimulation.

# Author        :   David A. Ellis <https://github.com/mosquitome/>
# Organisation  :   University College London

# Requirements  :   Python 3.9.18
#                   mat73 0.62
#                   pandas 2.1.4
#                   numpy 1.26.3
#                   scipy 1.11.4
#                   sklearn 1.3.2

# Notes         :   - The current directory should contain a number of folders, one
                      per mosquito.
                    - Each of these must contain any number of sub-folders, inside
                      which are matlab files of averaged data from sweeps.
                    - There must not be any files other than these in the current
                      dierctory.
                    - Below, "nerve" refers to high-pass filtered nerve data and
                      "nerve2" refers to lowpass filtered.
                    - return info dataframe containing details on peaks:

    sweep_direction                                 = 'bwd' or 'fwd'
    [nerve|laser]_[lower|upper]_time                = time of peak for given spline
    [nerve|laser]_[lower|upper]_left_time           = time of left edge of peak width (peak width is measured at the mid-point of the peak)
    [nerve|laser]_[lower|upper]_right_time          = time of right edge of peak width
    [nerve|laser]_[lower|upper]_width_time          = timespan of peak width
    [nerve|laser]_[lower|upper]_value               = y-value at given spline's peak
    nerve_lower_value_upper_spline                  = for the nerve data, give corresponding value from the upper spline at the time of the peak in the lower spline.
    [nerve|laser]_[lower|upper]_frequency           = corresponding frequency at stimulus of the given peak
    [nerve|laser]_[lower|upper]_width_frequency     = timespan of peak width converted to frequency domain ("frequency span")
    [nerve|laser]_[lower|upper]_spline_mse          = mean squared error of the spline's fit to data (NOTE: data has been passed through filterSpikes() to remove large, sudden, anomalous spikes)
    nerve2_[lower|upper]_*                          = details on positive (upper) or negative (lower) peaks from single spline, fitted to smoothed nerve data
    [nerve|laser]_peak_to_peak                      = distance between peak called in lower spline to peak called in upper spline
    nerve_peak_to_peak_v2                           = distance between peak called in lower spline and the same x position in the upper spline (NOTE: still using DC-remove nerve, not nerve2)


#---------------------------------------------------------------------------------

'''

import mat73
import seaborn as sb
import matplotlib.pyplot as mplp
import matplotlib.gridspec as mplg
import scipy as sp
import numpy as np
import pandas as pd
import sklearn.metrics as skm
import sklearn.preprocessing as skp
import warnings as warn
import os
import pathlib as path

warn.filterwarnings('ignore')

sb.set(style='ticks', font_scale=1.5)
tmin, tmax, peak_threshold = {}, {}, {}

# NOTE: No point thresholding MSE as it is directly related to the smoothing factor; mat73 is not case sensitive when importing files

stimulus                    = 'H'                                       # <- name of stimulus (to determine filenames)
peak_threshold['nerve']     = 90                                        # <- percentile above which peak must be
peak_threshold['laser']     = 98.5
frequency_max               = 1100                                      # <- maximum frequency to analyse when acquiring spectrogram (max sweep frequency + 10%)
tmin['fwd']                 = 0.1                                       # <- length of time (seconds) flagellum is unstimulated before sweep to ignore when acquiring spectrogram
tmin['bwd']                 = 0.05
tmax['fwd']                 = 1.05                                      # <- length of time (seconds) after sweep to ignore when acquiring spectrogram
tmax['bwd']                 = 1.0
envelope_window             = 100                                       # <- rolling window size for nerve envelope
spline_degree               = 2                                         # <- type of spline: 1=linear, 2=quadratic, 3=cubic, 4=quartic, 5=?..
spline_max                  = 2e11 # 1e10

def filterSpikes(y, window=500, threshold=2):
    '''
    Remove noisey spikes by identifying them, then setting to local median.
    '''
    y_ = y.rolling(window, center=True).apply(lambda x: np.percentile(x, 5)) * threshold
    y_ = y_.replace(np.nan, y_.min()) # <- filter will not work at edges but will also not convert them to Nan
    y2 = y.copy()
    y2[y < y_] = y_[y < y_]
    return y2

def highLowEnvIdx(s, dmin=1, dmax=1, split=False):
    '''
    https://stackoverflow.com/questions/34235530/
    
    Input :
    s           = 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax  = int, optional, size of chunks, use this if the size of the input signal is too big
    split:      = bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    
    Returns :
    lmin,lmax   = high/low envelope idx of input signal s
    '''
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1
    if split:
        s_mid = np.mean(s) 
        lmin = lmin[s[lmin]<s_mid]
        lmax = lmax[s[lmax]>s_mid]
    lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
    lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
    return lmin,lmax

def weight(x, a, b):
    return -a * (abs(x)**b)

def lspline(x, y, s, k=spline_degree):
    if y.max() > 0:
        x = x [y < 0]
        y = y [y < 0]
    xx = np.linspace(y.min(), y.max(), 1000)
    yy = skp.MinMaxScaler((y.min(), y.max())).fit_transform(np.array([i**0.5 for i in np.linspace(1, 10, 1000)]).reshape(-1, 1))[:,0]
    #yy = np.geomspace(y.min(), y.max(), 1000)
    popt, pcov = sp.optimize.curve_fit(weight, xx, yy, maxfev=10000) # p0=[8]
    w = weight(y, *popt) # <- weight the fit around the peak
    f = sp.interpolate.UnivariateSpline(x, y, w=w, k=k)
    f.set_smoothing_factor(s)
    return f

def hspline(x, y, s, k=spline_degree):
    if y.min() < 0:
        x = x [y > 0]
        y = y [y > 0]
    xx = skp.MinMaxScaler((y.min(), y.max())).fit_transform(np.array([i**0.5 for i in np.linspace(1, 10, 1000)]).reshape(-1, 1))[:,0]
    #xx = np.geomspace(y.min(), y.max(), 1000)
    yy = np.linspace(y.min(), y.max(), 1000)
    popt, pcov = sp.optimize.curve_fit(weight, xx, yy, maxfev=10000) # p0=[8]
    w = weight(y, *popt) # <- weight the fit around the peak
    f = sp.interpolate.UnivariateSpline(x, y, w=w, k=k)
    f.set_smoothing_factor(s)
    return f

def mspline(x, y, s, k=spline_degree):
    f = sp.interpolate.UnivariateSpline(x, y, k=k)
    f.set_smoothing_factor(s)
    return f  

def sf(x, a, b):
    return x**a*b

def returnNan(x):
    return np.full(len(x), np.nan)

def optimiseSpline(x, y, f, threshold=90, smax='infer', n=500, infer_exponent=3.5, infer_high=spline_max):
    '''
    Find the spline fit that leads to a single strong peak.

    Parameters
    ----------
    x               = time
    y               = envelope 
    f               = Either 'lspline' (for negative peaks) or 'hspline' for positive peaks.
    threshold       = Percentile above which there should only be one peak. The default is 90.
    smax            = The maximum smoothing factor (used by scipy.interpolate.UnivariateSpline(). The default is 'infer'.
    n               = The number of smoothing factors to test. The default is 500.
    infer_exponent  = For adjusting the slope of the curve used to interpolate smoothing factors.
    infer_high      = the maximum possible value for smax

    Returns
    -------
    _f  = spline function to apply to x array.
    px  = x at peak
    py  = y at peak
    pw  = width of peak (in units of x)
    pl  = left-most point of peak width (in units of x)
    pr  = right-most point of peak width (in units of x)
    mse = mean squared error of y vs _f(x)

    '''
    
    if smax=='infer':
        xx = np.linspace(0, 160, 10000)
        yy = skp.MinMaxScaler((0, infer_high)).fit_transform(np.array([i**infer_exponent for i in np.linspace(1, 10, 10000)]).reshape(-1, 1))[:,0]
        popt, pcov = sp.optimize.curve_fit(sf, xx, yy, maxfev=10000) # p0=[8]
        smax = sf(y.std(), *popt)
    if f=='mspline':
        smax = smax * 15
    smooth_increments = skp.MinMaxScaler((0, smax)).fit_transform(np.array([np.exp(i) for i in np.linspace(1, 10, n)]).reshape(-1, 1))[:,0]
    for s in smooth_increments:
        _x = np.linspace(x.min(), x.max(), 1000) # <- equally-spaced x values (important for peak width information)
        if f=='mspline':
            _f = mspline(x, y, s)
            _y, peaks, info, count = {}, {}, {}, {}
            _y[0] = _f(_x)
            _y[1] = -1 * _f(_x)
            for i in [0,1]:
                peaks[i], info[i] = sp.signal.find_peaks(_y[i], prominence=0, width=0)
                t = np.percentile(_y[i], threshold)
                count[i] = sum(j > t for j in _y[i][peaks[i]])
            if count[0]==1 and count[1]==1:
                pdx = np.argmax(_y[0][peaks[0]]), np.argmax(_y[1][peaks[1]])
                px = _x[peaks[0][pdx[0]]], _x[peaks[1][pdx[1]]]
                py = _f(_x)[peaks[0][pdx[0]]], _f(_x)[peaks[1][pdx[1]]]
                pw = _x.max() / len(_x) * info[0]['widths'][pdx[0]], _x.max() / len(_x) * info[1]['widths'][pdx[1]]
                pl = _x.max() / len(_x) * info[0]['left_ips'][pdx[0]], _x.max() / len(_x) * info[1]['left_ips'][pdx[1]]
                pr = _x.max() / len(_x) * info[0]['right_ips'][pdx[0]], _x.max() / len(_x) * info[1]['right_ips'][pdx[1]]
                mse = skm.mean_squared_error(y, _f(x))
                print('')
                return _f, px, py, pw, pl, pr, mse
            elif count[0]==1 and count[1]!=1 and s==smooth_increments[-1]:
                pdx = np.argmax(_y[0][peaks[0]]), np.nan
                px = _x[peaks[0][pdx[0]]], np.nan
                py = _f(_x)[peaks[0][pdx[0]]], np.nan
                pw = _x.max() / len(_x) * info[0]['widths'][pdx[0]], np.nan
                pl = _x.max() / len(_x) * info[0]['left_ips'][pdx[0]], np.nan
                pr = _x.max() / len(_x) * info[0]['right_ips'][pdx[0]], np.nan
                mse = skm.mean_squared_error(y, _f(x))
                print('\n\toptimiseSpline() did not reach single upper peak. Try adjusting smax or power_function.')
                return _f, px, py, pw, pl, pr, mse
            elif count[0]!=1 and count[1]==1 and s==smooth_increments[-1]:
                pdx = np.nan, np.argmax(_y[1][peaks[1]])
                px = np.nan, _x[peaks[1][pdx[1]]]
                py = np.nan, _f(_x)[peaks[1][pdx[1]]]
                pw = np.nan, _x.max() / len(_x) * info[1]['widths'][pdx[1]]
                pl = np.nan, _x.max() / len(_x) * info[1]['left_ips'][pdx[1]]
                pr = np.nan, _x.max() / len(_x) * info[1]['right_ips'][pdx[1]]
                mse = skm.mean_squared_error(y, _f(x))
                print('\n\toptimiseSpline() did not reach single lower peak. Try adjusting smax or power_function.')
                return _f, px, py, pw, pl, pr, mse
        elif f=='lspline':
            _f = lspline(x, y, s)
            _y = abs(_f(_x))
            peaks, info = sp.signal.find_peaks(_y, prominence=0, width=0)
            t = np.percentile(_y, threshold)
            count = sum(i > t for i in _y[peaks]) 
        elif f=='hspline':
            _f = hspline(x, y, s)
            _y = _f(_x)
            peaks, info = sp.signal.find_peaks(_y, prominence=0, width=0)
            t = np.percentile(_y, threshold)
            count = sum(i > t for i in _y[peaks])        
        if f in ['lspline', 'hspline']:
            c_ = str(count).ljust(5)
        else:
            c_ = str(count[0])+','+str(count[1]).ljust(5)
        s_ = str(s).ljust(8)
        print('\r\t' + 'optimiseSpline() | peaks:', c_, ', smoothing:', s_, end='')
        if count==1:
            pdx = np.argmax(_y[peaks])
            px = _x[peaks[pdx]]
            py = _f(_x)[peaks[pdx]]
            pw = _x.max() / len(_x) * info['widths'][pdx]
            pl = _x.max() / len(_x) * info['left_ips'][pdx]
            pr = _x.max() / len(_x) * info['right_ips'][pdx]
            mse = skm.mean_squared_error(y, _f(x))
            print('')
            return _f, px, py, pw, pl, pr, mse
    if f=='mspline':
        print('\n\toptimiseSpline() did not reach single peak (either upper or lower). Try adjusting smax or power_function.' )
        return returnNan, (np.nan, np.nan), (np.nan, np.nan), (np.nan, np.nan), (np.nan, np.nan), (np.nan, np.nan), np.nan
    else:
        print('\n\toptimiseSpline() did not reach single peak. Try adjusting smax or power_function.' )
        return returnNan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

folders = os.listdir()

for folder in folders:
    
    subfolders = os.listdir(folder)
    
    for folder_ in subfolders:
    
        foldername = path.Path(folder, folder_)
        print('\n', foldername, '\n')
        
        data_fs20k = {} # <- all data sampled at 20k (nerve)
        data_fs100k = {} # <- all data sampled at 100k (stim, laser)
        info = pd.DataFrame(columns = [ 'sweep_direction',
                                        'nerve_lower_time',
                                        'nerve_lower_left_time',
                                        'nerve_lower_right_time',
                                        'nerve_lower_width_time',
                                        'nerve_lower_value',
                                        'nerve_lower_value_upper_spline',
                                        'nerve_lower_frequency',
                                        'nerve_lower_width_frequency',
                                        'nerve_lower_spline_mse',
                                        'nerve_upper_time',
                                        'nerve_upper_left_time',
                                        'nerve_upper_right_time',
                                        'nerve_upper_width_time',
                                        'nerve_upper_value',
                                        'nerve_upper_frequency',
                                        'nerve_upper_width_frequency',
                                        'nerve_upper_spline_mse',
                                        'nerve2_lower_time',
                                        'nerve2_lower_left_time',
                                        'nerve2_lower_right_time',
                                        'nerve2_lower_width_time',
                                        'nerve2_lower_value',
                                        'nerve2_lower_frequency',
                                        'nerve2_lower_width_frequency',
                                        'nerve2_upper_time',
                                        'nerve2_upper_left_time',
                                        'nerve2_upper_right_time',
                                        'nerve2_upper_width_time',
                                        'nerve2_upper_value',
                                        'nerve2_upper_frequency',
                                        'nerve2_upper_width_frequency',
                                        'nerve2_spline_mse',
                                        'laser_lower_time',
                                        'laser_lower_value',
                                        'laser_lower_left_time',
                                        'laser_lower_right_time',
                                        'laser_lower_width_time',
                                        'laser_lower_spline_mse',
                                        'laser_upper_time',
                                        'laser_upper_value',
                                        'laser_upper_left_time',
                                        'laser_upper_right_time',
                                        'laser_upper_width_time',
                                        'laser_upper_spline_mse']) # <- information about the maximal peak in splines from each dataset

    
        for d in ['bwd', 'fwd']:
        
            data_fs20k[d] = pd.DataFrame(columns=['time',
                                         'nerve',
                                         'nerve_lower_envelope',
                                         'nerve_upper_envelope',
                                         'nerve_lower_spline',
                                         'nerve_upper_spline',
                                         'nerve2',
                                         'nerve2_spline',
                                         'laser_lower_spline',
                                         'laser_upper_spline'])
            info = pd.concat([info, pd.DataFrame({'sweep_direction': [d]})]).reset_index(drop=True)
        
            # - - - N E R V E - 1 - - - #    
        
            print('\tNerve (DC removed)', d)
        
            file = d + '_' + stimulus + '_nerve_DC.mat'
            filename = path.Path(folder, folder_, file)
            
            _ = mat73.loadmat(filename)
            _.pop('file')
            key = list(_.keys())[0]
            data_fs20k[d]['nerve'] = _[key]['values']
            data_fs20k[d]['time'] = np.arange(_[key]['start'], _[key]['length'] * _[key]['interval'], _[key]['interval'])
            y_ = filterSpikes(data_fs20k[d]['nerve'])
            data_fs20k[d]['nerve_lower_envelope'] = y_.rolling(envelope_window, center=True).min()
            data_fs20k[d]['nerve_upper_envelope'] = y_.rolling(envelope_window, center=True).max()
            ly_, ldx_ = data_fs20k[d]['nerve_lower_envelope'].dropna(), data_fs20k[d]['nerve_lower_envelope'].dropna().index
            hy_, hdx_ = data_fs20k[d]['nerve_upper_envelope'].dropna(), data_fs20k[d]['nerve_upper_envelope'].dropna().index
            lf, lpx, lpy, lpw, lpl, lpr, lmse = optimiseSpline(data_fs20k[d]['time'][ldx_], ly_, f='lspline')
            hf, hpx, hpy, hpw, hpl, hpr, hmse = optimiseSpline(data_fs20k[d]['time'][hdx_], hy_, f='hspline')
            data_fs20k[d]['nerve_lower_spline'] = lf(data_fs20k[d]['time'])
            data_fs20k[d]['nerve_upper_spline'] = hf(data_fs20k[d]['time'])
            info.loc[info['sweep_direction']==d, 'nerve_lower_value_upper_spline'] = hf(lpx)
            for i, j in zip(['lower', 'upper'], [[lpx, lpy, lpl, lpr, lpw, lmse], [hpx, hpy, hpl, hpr, hpw, hmse]]):
                info.loc[info['sweep_direction']==d, ['nerve_' + i + '_time',
                                                      'nerve_' + i + '_value',
                                                      'nerve_' + i + '_left_time',
                                                      'nerve_' + i + '_right_time',
                                                      'nerve_' + i + '_width_time',
                                                      'nerve_' + i + '_spline_mse']] = [j[0], j[1], j[2], j[3], j[4], j[5]]
        
            # - - - N E R V E - 2 - - - #    
        
            print('\tNerve 2 (smooth)', d)
            
            file = d + '_' + stimulus + '_nerve_smooth.mat'
            filename = path.Path(folder, folder_, file)
            
            _ = mat73.loadmat(filename)
            _.pop('file')
            key = list(_.keys())[0]
            data_fs20k[d]['nerve2'] = _[key]['values'] - np.median(_[key]['values'])
            mf, mpx, mpy, mpw, mpl, mpr, mmse = optimiseSpline(data_fs20k[d]['time'], data_fs20k[d]['nerve2'], f='mspline')
            data_fs20k[d]['nerve2_spline'] = mf(data_fs20k[d]['time'])
            info.loc[info['sweep_direction']==d, 'nerve2_spline_mse'] = mmse
            for idx, i in enumerate(['upper','lower']):
                info.loc[info['sweep_direction']==d, ['nerve2_' + i + '_time',
                                                      'nerve2_' + i + '_width_time',
                                                      'nerve2_' + i + '_value',
                                                      'nerve2_' + i + '_left_time',
                                                      'nerve2_' + i + '_right_time',]] = [mpx[idx], mpw[idx], mpy[idx], mpl[idx], mpr[idx]]
            
            # - - - L A S E R - - - #
        
            print('\tLaser', d)
        
            file = d + '_' + stimulus + '_laser.mat'
            filename = path.Path(folder, folder_, file)
        
            data_fs100k[d] = pd.DataFrame(columns=['time',
                                         'laser',
                                         'laser_lower_spline',
                                         'laser_upper_spline',
                                         'stim',
                                         'frequency'])
        
            _ = mat73.loadmat(filename)
            _.pop('file')
            key = list(_.keys())[0]
            data_fs100k[d]['laser'] = _[key]['values']
            data_fs100k[d]['time'] = np.arange(_[key]['start'], _[key]['length'] * _[key]['interval'], _[key]['interval'])
            x_ = sp.signal.decimate(data_fs100k[d]['time'], q=5) # <- downsample data to improve envelope (q=5 should convert from fs=100k to fs=20k)
            y_ = sp.signal.decimate(data_fs100k[d]['laser'], q=5) # <- downsample data to improve envelope
            ldx, hdx = highLowEnvIdx(y_) # <- NOTE: indices are not necessarily evenly distributed; this uses a different envelope to nerve as it produced better results
            
            lf, lpx, lpy, lpw, lpl, lpr, lmse = optimiseSpline(x_[ldx], y_[ldx], f='lspline', threshold=peak_threshold['laser'])
            hf, hpx, hpy, hpw, hpl, hpr, hmse = optimiseSpline(x_[hdx], y_[hdx], f='hspline', threshold=peak_threshold['laser'])
            
            for i, j in zip(['lower', 'upper'], [[lpx, lpy, lpw, lpl, lpr, lmse], [hpx, hpy, hpw, hpl, hpr, hmse]]):
                info.loc[info['sweep_direction']==d, ['laser_' + i + '_time', 
                                                      'laser_' + i + '_value',
                                                      'laser_' + i + '_width_time',
                                                      'laser_' + i + '_left_time',
                                                      'laser_' + i + '_right_time',
                                                      'laser_' + i + '_spline_mse']] = [j[0], j[1], j[2], j[3], j[4], j[5]]    
            data_fs20k[d]['laser_lower_spline'] = lf(data_fs20k[d]['time']) # <- NOTE: interpolated at timepoints from nerve sampling (which had a different fs)
            data_fs20k[d]['laser_upper_spline'] = hf(data_fs20k[d]['time'])
            data_fs100k[d]['laser_lower_spline'] = lf(data_fs100k[d]['time'])
            data_fs100k[d]['laser_upper_spline'] = hf(data_fs100k[d]['time'])
        
            # - - - S T I M - - - #
        
            file = d + '_' + stimulus + '_stim.mat'
            filename = path.Path(folder, folder_, file)
        
            _ = mat73.loadmat(filename)
            _.pop('file')
            key = list(_.keys())[0]
            y = _[key]['values']
            x = np.arange(_[key]['start'], _[key]['length'] * _[key]['interval'], _[key]['interval'])
            fs = _[key]['length'] / (_[key]['length'] * _[key]['interval'])
            yy, xx, zz = sp.signal.spectrogram(y, fs=fs, nperseg=1000, nfft=5000) # <- break stimulus down into frequencies to allow interpolation of frequency from time
            idx = np.where(yy >= frequency_max)[0][0] # <- limit frequencies analysed (no need to analyse frequencies that were not played)
        
            matrix = pd.DataFrame(zz[:idx][::-1], index=yy[:idx][::-1], columns=xx)
            y2 = matrix.index[matrix.apply(lambda x: np.argmax(x), axis=0)]
            x2 = matrix.columns
            x2_ = x2[np.where((x2 > tmin[d]) & (x2 < tmax[d]))] # <- ignore data before sweep (to remove noisey outliers)
            y2_ = y2[np.where((x2 > tmin[d]) & (x2 < tmax[d]))] # <- ignore data after sweep (to remove noisey outliers)
        
            f = np.poly1d(np.polyfit(x2_, y2_, deg=1)) # <- fit a line through spectrogram (time-vs-frequency) to interpolate which frequency was being played at what time
        
            # - - - C O N S O L I D A T E - 1 - - - #
        
            data_fs20k[d]['frequency'] = np.nan
            data_fs20k[d].loc[(data_fs20k[d]['time'] > tmin[d]) & (data_fs20k[d]['time'] < tmax[d]), 'frequency'] = data_fs20k[d].loc[(data_fs20k[d]['time'] > tmin[d]) & (data_fs20k[d]['time'] < tmax[d])]['time'].apply(lambda x: f(x)) # <- only enter frequency information for timepoints > tmin and < tmax (else nan)
            data_fs100k[d]['stim'] = _[key]['values']
            data_fs100k[d]['frequency'] = np.nan
            data_fs100k[d].loc[(data_fs100k[d]['time'] > tmin[d]) & (data_fs100k[d]['time'] < tmax[d]), 'frequency'] = data_fs100k[d].loc[(data_fs100k[d]['time'] > tmin[d]) & (data_fs100k[d]['time'] < tmax[d])]['time'].apply(lambda x: f(x)) # <- only enter frequency information for timepoints > tmin and < tmax (else nan)
            for i in ['nerve', 'nerve2', 'laser']:
                info.loc[info['sweep_direction']==d, i + '_lower_frequency'] = f(info.loc[info['sweep_direction']==d][i + '_lower_time'].values[0]) # <- get the frequency of the stimulus at the time of peak nerve
                info.loc[info['sweep_direction']==d, i + '_upper_frequency'] = f(info.loc[info['sweep_direction']==d][i + '_upper_time'].values[0]) # <- get the frequency of the stimulus at the time of peak nerve
                info.loc[info['sweep_direction']==d, i + '_lower_width_frequency'] = abs(f(1) - f(1 - info.loc[info['sweep_direction']==d][i + '_lower_width_time'].values[0])) # <- convert peak width from time to frequency domain
                info.loc[info['sweep_direction']==d, i + '_upper_width_frequency'] = abs(f(1) - f(1 - info.loc[info['sweep_direction']==d][i + '_upper_width_time'].values[0])) # <- convert peak width from time to frequency domain
            
            for i in ['time', 'left_time', 'right_time', 'width_time', 'frequency', 'width_frequency', 'spline_mse']:
                info.loc[info['sweep_direction']==d, 'laser_average_' + i] = np.mean([info.loc[info['sweep_direction']==d]['laser_lower_' + i], info.loc[info['sweep_direction']==d]['laser_upper_' + i]])
                info.loc[info['sweep_direction']==d, 'nerve_average_' + i] = np.mean([info.loc[info['sweep_direction']==d]['nerve_lower_' + i], info.loc[info['sweep_direction']==d]['nerve_upper_' + i]])
            info.loc[info['sweep_direction']==d, 'laser_peak_to_peak'] = info.loc[info['sweep_direction']==d]['laser_upper_value'] - info.loc[info['sweep_direction']==d]['laser_lower_value']
            info.loc[info['sweep_direction']==d, 'nerve_peak_to_peak'] = info.loc[info['sweep_direction']==d]['nerve_upper_value'] - info.loc[info['sweep_direction']==d]['nerve_lower_value']
            info.loc[info['sweep_direction']==d, 'nerve_peak_to_peak_v2'] = info.loc[info['sweep_direction']==d]['nerve_lower_value_upper_spline'] - info.loc[info['sweep_direction']==d]['nerve_lower_value']
            info.loc[info['sweep_direction']==d, 'nerve2_peak_to_peak'] = info.loc[info['sweep_direction']==d]['nerve2_upper_value'] - info.loc[info['sweep_direction']==d]['nerve2_lower_value']
            
            # - - - P L O T - 1 - - - #
        
            fig = mplp.figure(figsize=[4, 12.5])
            xmin, xmax = data_fs20k[d]['time'].min(), data_fs20k[d]['time'].max()
            gs = mplg.GridSpec(5, 1, hspace=0.075, height_ratios=[1.0, 0.75, 1.0, 1.0, 1.0])
            ax = {}
            c = {'nerve1': '#1b9e77', 'nerve2': '#d95f02', 'laser': '#7570b3'}
            
            ax[0] = fig.add_subplot(gs[0])
            ax[0].plot(data_fs100k[d]['time'], data_fs100k[d]['stim'], c='.15')
            ax[0].set_xlim(xmin, xmax)
            ax[0].set_ylabel('Stimulus (V)')
            
            ax[1] = fig.add_subplot(gs[1])
            #ax[1].pcolormesh(xx, yy, zz, cmap='viridis') # 'gray'
            ax[1].pcolormesh(xx, yy, zz, cmap='cividis') # 'gray'
            #ax[1].plot(data_fs100k[d]['time'], data_fs100k[d]['frequency'], c='white', ls=':', lw=2) # '#BA1B1D'
            ax[1].set_xlim(xmin, xmax)
            ax[1].set_ylim(ymax=frequency_max)
            ax[1].set_ylabel('Frequency\n(Hz)')
            
            ymin, ymax = [abs(data_fs100k[d]['laser']).max() * -1.2, abs(data_fs100k[d]['laser']).max() * 1.2]
            ax[2] = fig.add_subplot(gs[2])
            ax[2].plot(data_fs100k[d]['time'], data_fs100k[d]['laser'], c='.15')
            ax[2].plot(data_fs100k[d]['time'], data_fs100k[d]['laser_lower_spline'], c=c['laser']) # c='.78'
            ax[2].plot(data_fs100k[d]['time'], data_fs100k[d]['laser_upper_spline'], c=c['laser']) # c='.78'
            ax[2].set_xlim(xmin, xmax)
            ax[2].set_ylim(ymin=ymin, ymax=ymax)
            ax[2].set_ylabel('Antennal\ndisplacement (nm)')
            
            ax[3] = fig.add_subplot(gs[3])
            ax[3].plot(data_fs20k[d]['time'], data_fs20k[d]['nerve'], c='.15')
            ax[3].plot(data_fs20k[d]['time'], data_fs20k[d]['nerve_lower_spline'], c=c['nerve1']) # c='.78'
            ax[3].plot(data_fs20k[d]['time'], data_fs20k[d]['nerve_upper_spline'], c=c['nerve1']) # c='.78'
            ax[3].set_xlim(xmin, xmax)
            ax[3].set_ylabel('Highpass\nNerve CAP (V)')
            
            ax[4] = fig.add_subplot(gs[4])
            ax[4].plot(data_fs20k[d]['time'], data_fs20k[d]['nerve2'], c='.15')
            ax[4].plot(data_fs20k[d]['time'], data_fs20k[d]['nerve2_spline'], c=c['nerve2']) # c='.78'
            ax[4].set_xlim(xmin, xmax)
            ax[4].set_xlabel('Time (sec)')
            ax[4].set_ylabel('Lowpass\nNerve CAP (V)')
            
            x = {}
            y = {}
            x['nerve1'] = info.loc[info['sweep_direction']==d, 'nerve_average_time'].values[0]
            y['nerve1'] = info.loc[info['sweep_direction']==d, 'nerve_average_frequency'].values[0]
            x['nerve1_left'] = info.loc[info['sweep_direction']==d, 'nerve_average_left_time'].values[0]
            x['nerve1_right'] = info.loc[info['sweep_direction']==d, 'nerve_average_right_time'].values[0]
            x['nerve2'] = info.loc[info['sweep_direction']==d, 'nerve2_lower_time'].values[0]
            y['nerve2'] = info.loc[info['sweep_direction']==d, 'nerve2_lower_frequency'].values[0]
            x['nerve2_left'] = info.loc[info['sweep_direction']==d, 'nerve2_lower_left_time'].values[0]
            x['nerve2_right'] = info.loc[info['sweep_direction']==d, 'nerve2_lower_right_time'].values[0]
            x['laser'] = info.loc[info['sweep_direction']==d, 'laser_average_time'].values[0]
            y['laser'] = info.loc[info['sweep_direction']==d, 'laser_average_frequency'].values[0]
            x['laser_left'] = info.loc[info['sweep_direction']==d, 'laser_average_left_time'].values[0]
            x['laser_right'] = info.loc[info['sweep_direction']==d, 'laser_average_right_time'].values[0]
            for i in [0, 1, 2, 3, 4]:
                ymin, ymax = ax[i].get_ylim()
                ax[i].axvline(x['nerve1'], c=c['nerve1'])
                ax[i].fill_between([x['nerve1_left'], x['nerve1_right']], [ax[i].get_ylim()[1]] * 2, \
                                   y2=[ax[i].get_ylim()[0]] * 2, facecolor=c['nerve1'], \
                                   zorder=0, edgecolor=None, alpha=0.6)
                ax[i].axvline(x['nerve2'], c=c['nerve2'])
                ax[i].fill_between([x['nerve2_left'], x['nerve2_right']], [ax[i].get_ylim()[1]] * 2, \
                                   y2=[ax[i].get_ylim()[0]] * 2, facecolor=c['nerve2'], \
                                   zorder=0, edgecolor=None, alpha=0.6)
                ax[i].axvline(x['laser'], c=c['laser'])
                ax[i].fill_between([x['laser_left'], x['laser_right']], [ax[i].get_ylim()[1]] * 2, \
                                   y2=[ax[i].get_ylim()[0]] * 2, facecolor=c['laser'], \
                                   zorder=0, edgecolor=None, alpha=0.6)
                ax[i].set_ylim(ymin, ymax)
                if i==4:
                    break
                sb.despine(ax=ax[i], bottom=True, trim=True)
                ax[i].tick_params(axis='x', bottom=False, labelbottom=False)
            sb.despine(ax=ax[4], trim=True)
            ax[1].axhline(y['nerve1'], c=c['nerve1'])
            ax[1].axhline(y['laser'], c=c['laser'])
            
            # - - - O U T P U T - 1 - - - #
            
            mplp.subplots_adjust(left=0.35)
            mplp.savefig(foldername.as_posix() + '_' + d + '-time-plots.pdf', dpi=600, format='pdf')
        
        # - - - C O N S O L I D A T E - 2 - - - #
        
        _ = [i for i in info.columns if i != 'sweep_direction']
        info.loc[2, 'sweep_direction'] = 'avg'
        info.loc[2, _] = info.loc[[0,1], _].mean()
        
        # - - - P L O T - 2 - - - #
            
        fig = mplp.figure(figsize=[5, 7.5])
        #xmin, xmax = data_fs100k[d]['frequency'].min(), data_fs100k[d]['frequency'].max()
        xmin, xmax = 0, data_fs100k[d]['frequency'].max()
        gs = mplg.GridSpec(3, 1, hspace=0.075, height_ratios=[1.0, 1.0, 1.0])
        ax = {}
        c = {'fwd1': '#762a83', 'fwd2': '#9970ab', 'fwd3': '#c2a5cf',  \
             'bwd1': '#1b7837', 'bwd2': '#5aae61', 'bwd3': '#a6dba0'}
        
        ax[0] = fig.add_subplot(gs[0])
        ax[0].plot(data_fs100k['fwd']['frequency'], data_fs100k['fwd']['laser'], c=c['fwd3'])
        ax[0].plot(data_fs100k['bwd']['frequency'], data_fs100k['bwd']['laser'], c=c['bwd3'])
        ax[0].plot(data_fs100k['fwd']['frequency'], data_fs100k['fwd']['laser_lower_spline'], c=c['fwd2'])
        ax[0].plot(data_fs100k['fwd']['frequency'], data_fs100k['fwd']['laser_upper_spline'], c=c['fwd2'])
        ax[0].plot(data_fs100k['bwd']['frequency'], data_fs100k['bwd']['laser_lower_spline'], c=c['bwd2'])
        ax[0].plot(data_fs100k['bwd']['frequency'], data_fs100k['bwd']['laser_upper_spline'], c=c['bwd2'])
        ax[0].axvline(info.loc[info['sweep_direction']=='fwd', 'laser_average_frequency'].values, c=c['fwd1'])
        ax[0].axvline(info.loc[info['sweep_direction']=='bwd', 'laser_average_frequency'].values, c=c['bwd1'])
        ax[0].set_ylabel('Antennal\ndisplacement (nm)')
        
        ax[1] = fig.add_subplot(gs[1])
        ax[1].plot(data_fs20k['fwd']['frequency'], data_fs20k['fwd']['nerve'], c=c['fwd3'])
        ax[1].plot(data_fs20k['bwd']['frequency'], data_fs20k['bwd']['nerve'], c=c['bwd3'])
        ax[1].plot(data_fs20k['fwd']['frequency'], data_fs20k['fwd']['nerve_lower_spline'], c=c['fwd2'])
        ax[1].plot(data_fs20k['fwd']['frequency'], data_fs20k['fwd']['nerve_upper_spline'], c=c['fwd2'])
        ax[1].plot(data_fs20k['bwd']['frequency'], data_fs20k['bwd']['nerve_lower_spline'], c=c['bwd2'])
        ax[1].plot(data_fs20k['bwd']['frequency'], data_fs20k['bwd']['nerve_upper_spline'], c=c['bwd2'])
        ax[1].axvline(info.loc[info['sweep_direction']=='fwd', 'nerve_average_frequency'].values, c=c['fwd1'])
        ax[1].axvline(info.loc[info['sweep_direction']=='bwd', 'nerve_average_frequency'].values, c=c['bwd1'])
        ax[1].set_ylabel('Nerve highpass\nCAP (V)')
        
        ax[2] = fig.add_subplot(gs[2])
        ax[2].plot(data_fs20k['fwd']['frequency'], data_fs20k['fwd']['nerve2'], c=c['fwd3'])
        ax[2].plot(data_fs20k['bwd']['frequency'], data_fs20k['bwd']['nerve2'], c=c['bwd3'])
        ax[2].plot(data_fs20k['fwd']['frequency'], data_fs20k['fwd']['nerve2_spline'], c=c['fwd2'])
        ax[2].plot(data_fs20k['bwd']['frequency'], data_fs20k['bwd']['nerve2_spline'], c=c['bwd2'])
        ax[2].axvline(info.loc[info['sweep_direction']=='fwd', 'nerve2_lower_frequency'].values, c=c['fwd1'])
        ax[2].axvline(info.loc[info['sweep_direction']=='bwd', 'nerve2_lower_frequency'].values, c=c['bwd1'])
        ax[2].set_ylabel('Nerve lowpass\nCAP (V)')
        ax[2].set_xlabel('Stimulus frequency (Hz)')
        
        ax[0].set_xlim(xmin=xmin, xmax=xmax)
        ax[1].set_xlim(xmin=xmin, xmax=xmax)
        ax[2].set_xlim(xmin=xmin, xmax=xmax)
        sb.despine(ax=ax[0], bottom=True, trim=True)
        sb.despine(ax=ax[1], bottom=True, trim=True)
        sb.despine(ax=ax[2], trim=True)
        ax[0].tick_params(axis='x', bottom=False, labelbottom=False)
        ax[1].tick_params(axis='x', bottom=False, labelbottom=False)
        
        mplp.savefig(foldername.as_posix() + '_' + 'frequency-plots.pdf', dpi=600, format='pdf')
        
        #
        
        info.to_csv(foldername.as_posix() + '_' + 'spline-peak-stats.txt', sep='\t', index=False, header=True)
