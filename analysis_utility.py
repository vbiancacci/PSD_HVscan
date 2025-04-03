import os
import json
import time
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py, sys
import pickle
import psutil

from scipy.optimize import curve_fit
#from scipy.ndimage import gaussian_filter1d

import lgdo.lh5_store as lh5
#from pygama.lgdo import Array, Scalar, WaveformTable
#from pygama.lgdo import show
from pygama.raw import build_raw

import pygama.math.histogram as pgh
from pygama.dsp.processors import fixed_time_pickoff
from pygama.dsp.processors import bl_subtract
from pygama.dsp.processors import double_pole_zero
from pygama.dsp.processors import trap_norm
from pygama.dsp.processors import fixed_time_pickoff
from pygama.dsp.processors import linear_slope_fit
from pygama.dsp.processors import t0_filter
from pygama.dsp.processors import time_point_thresh
from pygama.dsp.processors import min_max
from pygama.dsp.processors import histogram, histogram_stats
from pygama.dsp.processors import get_multi_local_extrema
from pygama.dsp.processors import peak_snr_threshold
from pygama.dsp.processors import gaussian_filter1d
#from pygama.dsp.processors import dplms_ge
from pygama.dsp.processors import zac_filter
from pygama.dsp.processors import cusp_filter
from pygama.dsp.processors import presum

from pygama.pargen.energy_optimisation import fom_FWHM_with_dt_corr_fit
from pygama.math.peak_fitting import extended_radford_pdf, radford_pdf, extended_gauss_step_pdf, gauss_step_pdf, goodness_of_fit
from pygama.pargen.energy_optimisation import new_fom, fwhm_slope, event_selection
from pygama.pargen.cuts import find_pulser_properties

from iminuit import Minuit

from legendmeta import LegendMetadata
meta_path = '/lfs/l1/legend/users/bianca/sw/legend-metadata'
lmeta = LegendMetadata(path=meta_path)

import calibration as pcal
from PIL import Image

cols = ['b','r','g','m','y','c','orange','gray','darkblue','pink']

def load_ge_lh5(run, chn, file_dir, n_file = None, s_file = 0):
    store = lh5.LH5Store()
    count = 0
    for p, d, files in os.walk(file_dir):
        d.sort()    
        for i, f in enumerate(sorted(files)):
            if (f.endswith(".lh5")) & ("raw" in f) & (i >= s_file):
                lh5_file = f"{file_dir}/{f}"
                #print(lh5_file)
                wfs0, n_rows = store.read_object(f"{chn}/raw/waveform/values", lh5_file)
                daqenergy0, n_rows = store.read_object(f"{chn}/raw/daqenergy", lh5_file)
                baseline0, n_rows = store.read_object(f"{chn}/raw/baseline", lh5_file)
                timestamp0, n_rows = store.read_object(f"{chn}/raw/timestamp", lh5_file)
                t_pulser0, n_rows = store.read_object(f"ch001/raw/timestamp", lh5_file)
                if count == 0:
                    wfs, daqenergy, baseline, timestamp, t_pulser = wfs0.nda, daqenergy0.nda, baseline0.nda, timestamp0.nda, t_pulser0.nda
                else:
                    wfs, daqenergy, baseline = np.append(wfs, wfs0.nda, axis=0), np.append(daqenergy, daqenergy0.nda), np.append(baseline, baseline0.nda)
                    timestamp, t_pulser = np.append(timestamp, timestamp0.nda), np.append(t_pulser, t_pulser0.nda)
                count += 1
            if n_file is not None and count >= n_file: break
    return wfs, daqenergy, baseline, timestamp, t_pulser


def select_baselines(run, chn, det, file_dir, size = None, n_sel = 10000, nwf = 100, times_sigma = 10, n_split = None,
                     blsub = False, plot = False, plot_dir = None, down_sample = None):
    if plot: fig, axis = plt.subplots(nrows=1, ncols=2,figsize=(16,9), facecolor='white')
    count, n_bls, n_tot = 0, 0, 0
    t_start = time.time()
    while n_bls < n_sel:
        #try:
        bls0, daqenergy0, baseline0, t0, tp = load_ge_lh5(run, chn, file_dir, n_file = 1, s_file = count)
            #mean, std, slope, intercept = linear_slope_fit(wfs[:,:1000])
        #except:
        #    break
        n_tot += len(bls0)
        #mask = (daqenergy == 0) & ( np.abs(slope) < 0.1 ) & (std < 40)
        bls0, baseline0 = bls0[(daqenergy0 == 0)], baseline0[(daqenergy0 == 0)]
        if size is not None: bls0 = bls0[:,:size]
        mean, std, slope, intercept = linear_slope_fit(bls0)
        #mask = ( np.abs(slope) < 0.1 ) & (std < 40)
        # std cut
        lim0 = std[(std>0)&(std<100)].mean() - times_sigma * std[(std>0)&(std<100)].std()
        lim1 = std[(std>0)&(std<100)].mean() + times_sigma * std[(std>0)&(std<100)].std()
        mask = (std > lim0) & (std < lim1)
        if count == 0 and plot:
            h, b = np.histogram(std, bins = np.linspace(0,100,100))
            hc, b = np.histogram(std[mask], bins = np.linspace(0,100,100))
            axis[0].plot(pgh.get_bin_centers(b),h,color='b',ds='steps',label='std before')
            axis[0].plot(pgh.get_bin_centers(b),hc,color='r',ds='steps',label='std after')
            axis[0].axvline(lim0,color='k',label=f'selection {lim0:.1f} - {lim1:.1f}')
            axis[0].axvline(lim1,color='k')
            axis[0].set_yscale('log')
            axis[0].legend(title=f'{chn} - {det}',loc='upper right')
        # slope
        lim0 = slope[np.abs(slope) < 1].mean() - times_sigma * slope[np.abs(slope) < 1].std()
        lim1 = slope[np.abs(slope) < 1].mean() + times_sigma * slope[np.abs(slope) < 1].std()
        mask &= (slope > lim0) & (slope < lim1)
        if count == 0 and plot:
            h, b = np.histogram(slope, bins = np.linspace(-1,1,100))
            hc, b = np.histogram(slope[mask], bins = np.linspace(-1,1,100))
            axis[1].plot(pgh.get_bin_centers(b),h,color='b',ds='steps',label='slope before')
            axis[1].plot(pgh.get_bin_centers(b),hc,color='r',ds='steps',label='slope after')
            axis[1].axvline(lim0,color='k',label=f'selection {lim0:.1f} - {lim1:.1f}')
            axis[1].axvline(lim1,color='k')
            axis[1].set_yscale('log')
            axis[1].legend(title=f'{chn} - {det}',loc='upper right')
        bls0, baseline0 = bls0[mask], baseline0[mask]
        if n_split is not None: bls0, baseline0 = bls0[:int(n_sel/n_split)], baseline0[:int(n_sel/n_split)]
        if count == 0: bls, baseline = bls0, baseline0
        else: bls, baseline = np.append(bls, bls0, axis = 0), np.append(baseline, baseline0)
        n_bls = len(bls)
        if plot: print('n.file',count,'n. selected events', n_bls )
        count += 1
    if blsub: bls = bl_subtract(bls, baseline)
    n_bls = len(bls)
    diff = time.time() - t_start
    print(f"Time to select the baselines: {diff:.2f} sec")
    print(f'{chn}, n. file: {count}, Total events: {n_tot}, selected {n_bls} -> {n_bls/n_tot*100:.2f}%')
    if down_sample is not None:
        n_ev, n_size = len(bls), round(len(bls[0]) / down_sample)
        bls_presum = np.zeros(( n_ev, n_size ))
        presum(bls,1,down_sample,bls_presum)
        bls = bls_presum
    if plot:
        fig, ax = plt.subplots(figsize=(12,6.75))
        plt.xlabel("Samples")
        plt.ylabel("ADC")
        nwf = 50
        if nwf > len(bls): nwf = len(bls)
        for j in range(nwf):
            if j == 0: plt.plot(bls[j],label=f'{chn} - {det}')
            else: plt.plot(bls[j])
        plt.legend(loc='upper right')
        if plot_dir is not None:
            fig.savefig(f'{plot_dir}/bls_{chn}.png',dpi=300, bbox_inches='tight')
    return bls


def select_signals(run, chn, det, file_dir, size, n_sel = 10000, nwf=500, blsub = True, n_split = None,
                   e_low = 1500, e_high = 2700, selection = True, plot = False, plot_dir = None, down_sample = None):
    if plot: fig, axis = plt.subplots(nrows=2, ncols=3, figsize=(16,9), facecolor='white')
    count, s_count, n_wfs, n_tot = 0, 0, 0, 0
    t_start = time.time()
    while n_wfs < n_sel:
        try:
            wfs0, daqenergy0, baseline0, timestamp, t_pulser = load_ge_lh5(run, chn, file_dir, n_file=1, s_file=s_count)
            mean, std, slope, intercept = linear_slope_fit(wfs0[:,:100])
        except:
            break
        if down_sample is not None:
            wfs_presum = np.zeros(( len(wfs0), int(len(wfs0[0])/down_sample) ))
            presum(wfs0,1,down_sample,wfs_presum)
            wfs0 = wfs_presum
            del wfs_presum
        n_tot += len(wfs0)
        if (n_tot < 50000) or (daqenergy0.mean() > 6000):
            s_count += 1
            continue
        try:
            peak_max, peak_last = pcal.get_first_last_peaks(daqenergy0, pulser = 0)
        except:
            s_count += 1
            continue
        rough_kev_per_adc = (pcal.cal_peaks_th228[0] - pcal.cal_peaks_th228[-1])/(peak_max-peak_last)
        rough_kev_offset = pcal.cal_peaks_th228[0] - rough_kev_per_adc * peak_max
        e_cal_rough = rough_kev_per_adc * daqenergy0 + rough_kev_offset
        mask = pulser_tag(timestamp, t_pulser) & (e_cal_rough > e_low) & (e_cal_rough < e_high)
        if count == 0 and plot:
            he, be = np.histogram(e_cal_rough, bins = np.linspace(0,3000,500))
            hec, be = np.histogram(e_cal_rough[mask], bins = np.linspace(0,3000,500))
            axis[0][0].plot(be[1:],he,color='b',ds='steps',label='before')
            axis[0][0].plot(be[1:],hec,color='r',ds='steps',label='after')
            axis[0][0].set_yscale('log')
            axis[0][0].legend(title=f'{chn} - {det}',loc='upper right')
        if selection:
            times_sigma = 5
            # std cut
            h, b = np.histogram(std, bins = np.linspace(0,100,100))
            #idx = np.where(h > h.max()*0.01)
            #lim0, lim1 = b[idx[0][0]], b[idx[0][-1]]
            guess, b_ = get_gaussian_guess(h, pgh.get_bin_centers(b))
            par, pcov = curve_fit(gaussian, pgh.get_bin_centers(b), h, p0=guess)
            lim0, lim1 = par[1] - times_sigma * par[2], par[1] + times_sigma * par[2]
            mask &= (std > lim0) & (std < lim1)
            if count == 0 and plot:
                hc, b = np.histogram(std[mask], bins = np.linspace(0,100,100))
                axis[0][1].plot(pgh.get_bin_centers(b),h,color='b',ds='steps',label='std before')
                axis[0][1].plot(pgh.get_bin_centers(b),hc,color='r',ds='steps',label='std after')
                axis[0][1].axvline(lim0,color='k',label='selection')
                axis[0][1].axvline(lim1,color='k')
                axis[0][1].set_yscale('log')
                axis[0][1].legend(title=f'{chn} - {det}',loc='upper right')
            # slope
            h, b = np.histogram(slope, bins = np.linspace(-1,1,100))
            guess, b_ = get_gaussian_guess(h, pgh.get_bin_centers(b))
            par, pcov = curve_fit(gaussian, pgh.get_bin_centers(b), h, p0=guess)
            lim0, lim1 = par[1] - times_sigma * par[2], par[1] + times_sigma * par[2]
            mask &= (slope > lim0) & (slope < lim1)
            if count == 0 and plot:
                hc, b = np.histogram(slope[mask], bins = np.linspace(-1,1,100))
                axis[0][2].plot(pgh.get_bin_centers(b),h,color='b',ds='steps',label='slope before')
                axis[0][2].plot(pgh.get_bin_centers(b),hc,color='r',ds='steps',label='slope after')
                axis[0][2].axvline(lim0,color='k',label='selection')
                axis[0][2].axvline(lim1,color='k')
                axis[0][2].set_yscale('log')
                axis[0][2].legend(title=f'{chn} - {det}',loc='upper right')
            # mean
            bspace = np.linspace(12000,18000,500)
            h, b = np.histogram(mean, bins = bspace)
            guess, b_ = get_gaussian_guess(h, pgh.get_bin_centers(b))
            par, pcov = curve_fit(gaussian, pgh.get_bin_centers(b), h, p0=guess)
            lim0, lim1 = par[1] - 2 * times_sigma * par[2], par[1] + 10 * times_sigma * par[2]
            mask &= (mean > lim0) & (mean < lim1)
            if count == 0 and plot:
                hc, b = np.histogram(mean[mask], bins = bspace)
                axis[1][0].plot(pgh.get_bin_centers(b),h,color='b',ds='steps',label='mean before')
                axis[1][0].plot(pgh.get_bin_centers(b),hc,color='r',ds='steps',label='mean after')
                axis[1][0].axvline(lim0,color='k',label='selection')
                axis[1][0].axvline(lim1,color='k')
                axis[1][0].set_yscale('log')
                axis[1][0].legend(title=f'{chn} - {det}',loc='upper right')
        del mean, std, slope, intercept
        wfs0, baseline0 = wfs0[mask], baseline0[mask]
        if n_split is not None: wfs0, baseline0 = wfs0[:int(n_sel/n_split)], baseline0[:int(n_sel/n_split)]
        if count == 0:
            wfs, baseline, daqenergy = wfs0, baseline0, daqenergy0
        else:
            wfs, baseline, daqenergy = np.append(wfs, wfs0, axis = 0), np.append(baseline, baseline0), np.append(daqenergy, daqenergy0)
        del wfs0, daqenergy0, baseline0
        n_wfs = len(wfs)
        if plot: print('n.file',s_count,'n. selected events',n_wfs,'size',len(wfs[0]))
        count += 1
        s_count += 1
    
    if selection:
        # pile up
        peaks_pos = np.zeros((len(wfs),20))
        gaus_func = gaussian_filter1d(3,4)
        wfd = np.zeros((n_wfs,len(wfs[0])))
        gaus_func(wfs,wfd)
        for i, wf in enumerate(wfd):
            #diff = np.diff(np.array(wf, dtype=float))
            #peaks, pr = signal.find_peaks(gaussian_filter1d(diff,5), height=diff.max()/50, distance=50)
            peaks = extract_peaks(np.gradient(wf), a_delta = 20, times_fwhm = 3, ratio = 0.8, width = 10, check = 0)
            peaks_pos[i] = peaks
            #peaks = peaks[((peaks > 0) & (peaks < size/2 - 100)) | ((peaks > size/2 + 100) & (peaks < size))]
        win_sel = 50
        peaks_array = peaks_pos[(peaks_pos > size/2-win_sel) & (peaks_pos < size/2+win_sel)]
        h, b = np.histogram(peaks_array, bins = np.linspace(size/2-win_sel,size/2+win_sel,400))
        lim0 = peaks_array[(peaks_array>size/2-win_sel)&(peaks_array<size/2+win_sel)].mean() - times_sigma * peaks_array[(peaks_array>size/2-win_sel)&(peaks_array<size/2+win_sel)].std()
        lim1 = peaks_array[(peaks_array>size/2-win_sel)&(peaks_array<size/2+win_sel)].mean() + times_sigma * peaks_array[(peaks_array>size/2-win_sel)&(peaks_array<size/2+win_sel)].std()
        npeaks = [len(p[((p>0) & (p<lim0)) | ((p>lim1) & (p<size))]) for p in peaks_pos]
        if plot:
            print('Before pile-up selection',len(wfs))
            axis[1][1].plot(pgh.get_bin_centers(b),h,color='b',ds='steps',label='peak positions')
            axis[1][1].axvline(lim0,color='k',label='region excluded for pile-up')
            axis[1][1].axvline(lim1,color='k')
            axis[1][1].set_yscale('log')
            axis[1][1].legend(title=f'{chn} - {det}',loc='upper right')
        mask = np.array(npeaks) == 0
        wfs, baseline = wfs[mask], baseline[mask]
        if plot: print('After pile-up selection',len(wfs))
        """# alignment with current max
        wfs_new, baseline_new = np.zeros((len(wfs),size)), np.zeros(len(wfs))
        curr_max = np.array([np.argmax(np.gradient(wf)) for wf in wfs])
        h, b = np.histogram(curr_max, bins = np.linspace(size/2-win_sel,size/2+win_sel,400))
        lim0 = int(curr_max[(curr_max>size/2-win_sel)&(curr_max<size/2+win_sel)].mean() - 2*times_sigma * curr_max[(curr_max>size/2-win_sel)&(curr_max<size/2+win_sel)].std())
        lim1 = int(curr_max[(curr_max>size/2-win_sel)&(curr_max<size/2+win_sel)].mean() + 2*times_sigma * curr_max[(curr_max>size/2-win_sel)&(curr_max<size/2+win_sel)].std())
        for i, wf in enumerate(wfs):
            if (curr_max[i] >= size/2) and (curr_max[i] < lim1):
                wfs_new[i] = wf[int(curr_max[i] - size/2): int(curr_max[i] + size/2)]
                baseline_new[i] = baseline[i]
            elif (curr_max[i] > lim0) and (curr_max[i] < lim1):
                ss = int((size+1)/2 - curr_max[i])
                wfs_new[i][:ss] = baseline[i]
                wfs_new[i][ss:] = wf[: int(curr_max[i] + size/2)]
                baseline_new[i] = baseline[i]"""
        # alignment with centroid
        wfs_new, baseline_new, centroid = np.zeros((len(wfs),size)), np.zeros(len(wfs)), np.zeros(len(wfs))
        step = step_function(lenght = 80)
        for i, wf in enumerate(wfs):
            wf_step = np.convolve(wf, step, 'valid')
            ss0 = int((len(wf)-len(wf_step)+1)/2)
            try:
                c_a = np.where(wf_step[wf_step.argmin():wf_step.argmax()] > 0)[0][0] + wf_step.argmin() + ss0
                c_b = np.where(wf_step[wf_step.argmin():wf_step.argmax()] < 0)[0][-1] + wf_step.argmin() + ss0
                centroid[i] = round((c_a+c_b)/2)
            except: continue
            
        h, b = np.histogram(centroid, bins = np.linspace(size/2-win_sel,size/2+win_sel,400))
        lim0 = int(centroid[(centroid>size/2-win_sel)&(centroid<size/2+win_sel)].mean() - 2*times_sigma * centroid[(centroid>size/2-win_sel)&(centroid<size/2+win_sel)].std())
        lim1 = int(centroid[(centroid>size/2-win_sel)&(centroid<size/2+win_sel)].mean() + 2*times_sigma * centroid[(centroid>size/2-win_sel)&(centroid<size/2+win_sel)].std())
        for i, wf in enumerate(wfs):
            if (centroid[i] >= size/2) and (centroid[i] < lim1):
                wfs_new[i] = wf[int(centroid[i] - size/2): int(centroid[i] + size/2)]
                baseline_new[i] = baseline[i]
            elif (centroid[i] > lim0) and (centroid[i] < lim1):
                ss = int((size+1)/2 - centroid[i])
                wfs_new[i][:ss] = baseline[i]
                wfs_new[i][ss:] = wf[: int(centroid[i] + size/2)]
                baseline_new[i] = baseline[i]
        if plot:
            print('Before alignment',len(wfs))
            axis[1][2].plot(pgh.get_bin_centers(b),h,color='b',ds='steps',label='centroid')
            axis[1][2].axvline(lim0,color='m',label='limit for the alignment')
            axis[1][2].axvline(lim1,color='m')
            axis[1][2].set_yscale('log')
            axis[1][2].legend(title=f'{chn} - {det}',loc='upper right')
        #mask = (curr_max > lim0) & (curr_max < lim1)
        mask = (centroid > lim0) & (centroid < lim1)
        mask &= ~np.all(wfs_new == 0, axis=1)
        wfs, baseline = wfs_new[mask], baseline_new[mask]
        if plot: print('After alignment',len(wfs))
        del wfd, wfs_new
    else: wfs, baseline = wfs[:n_sel,:size], baseline[:n_sel]
    if blsub: wfs = bl_subtract(wfs, baseline)
    n_wfs = len(wfs)
    diff = time.time() - t_start
    del baseline, mask, e_cal_rough
    print(f"Time to select signals: {diff:.2f} sec")
    print(f'{chn}, n. file: {count}, Total events: {n_tot}, selected {n_wfs} -> {n_wfs/n_tot*100:.2f}%')
    if plot:
        fig1, ax1 = plt.subplots(figsize=(12,6.75), facecolor='white')
        plt.xlabel("Samples")
        plt.ylabel("ADC")
        if nwf > len(wfs): nwf = len(wfs)
        for j in range(nwf):
            if j == 0: plt.plot(wfs[j],label=f'{chn} - {det}')
            else: plt.plot(wfs[j])
            #if j == 0: plt.plot(np.gradient(wfs[j]),label=f'{chn} - {det}')
            #else: plt.plot(np.gradient(wfs[j]))
        plt.legend(loc='upper right')
        axin = ax1.inset_axes([0.1, 0.15, 0.35, 0.5])
        for j in range(nwf):
            axin.plot(wfs[j])
            #axin.plot(np.gradient(wfs[j]))
        axin.set_xlim(len(wfs[0])/2 - 20, len(wfs[0])/2 + 20)
        axin.set_yticklabels('')
        if plot_dir is not None:
            fig.savefig(f'{plot_dir}/wfs_selection_{chn}.png',dpi=300, bbox_inches='tight')
            fig1.savefig(f'{plot_dir}/wfs_{chn}.png',dpi=300, bbox_inches='tight')
        del fig, fig1, ax1, axin, axis, he, hec, be
    check_memory()
    return wfs

def step_function(lenght):
    x = np.linspace(0, lenght, lenght)
    step_function = np.piecewise(x, [((x >= 0) & (x < lenght/4)), ((x >= lenght/4) & (x <= 3*lenght/4)), 
                                     ((x > 3*lenght/4) & (x <= lenght))], [-1, 1, -1])
    return step_function

def calculate_fft(bls, nbls = 1000,sample_time = 0.016, plot = False):
    if nbls > len(bls): nbls = len(bls)
    sampling_rate = 1/sample_time
    for i in range(nbls):
        fft = np.fft.rfft(bls[i][:])
        abs_fft = np.abs(fft)
        if i == 0: power_spectrum = np.square(abs_fft)
        else: power_spectrum += np.square(abs_fft)
    power_spectrum /= nbls
    frequency = np.linspace(0, sampling_rate/2, len(power_spectrum))
    if plot:
        plt.figure(figsize=(12,6.75))
        plt.plot(frequency[1:], power_spectrum[1:])
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("frequency (MHz)", ha='right', x=1)
        plt.ylabel(f"power spectral density", ha='right', y=1)
    return frequency, power_spectrum


def process_signals(chn, x, file_dir, run = 22, n_file = 1, plot = False):
    t_start = time.time()
    wfs, daqenergy = load_ge_lh5(chn, file_dir, run, n_file)
    wfs = wfs[daqenergy>100]
    diff = time.time() - t_start
    print(f"Time to open n. {len(wfs)} signals: {diff:.2f} sec")
    t_start = time.time()
    unc_ene = []
    #with open (f'{filter_dir}/filter_{chn}.pkl', 'rb') as fp:
    #x = pickle.load(fp)
    if plot: plt.figure(figsize=(12,6.75), facecolor='white')
    for i, wf in enumerate(wfs):
        conv = np.convolve(-wf[850:5150], x, mode = 'valid')
        unc_ene.append(conv.max())
        if i < 10 and plot: plt.plot(conv)
    unc_ene = np.array(unc_ene)
    print(f"Time to calculate energies: {(time.time() - t_start):.2f} sec")
    if plot:
        plt.figure(figsize=(12,6.75), facecolor='white')
        h, b = np.histogram(unc_ene,bins=np.linspace(0,unc_ene.max(),3500))
        plt.plot(b[1:], h)
    return unc_ene


def get_gaussian_guess(h,b):
    mu = b[np.argmax(h)]
    imax = np.argmax(h)
    hmax = h[imax]
    idx = np.where(h > hmax/2) # fwhm 
    ilo, ihi = idx[0][0], idx[0][-1]
    sig = (b[ihi]-b[ilo]) / 2.355
    if sig == 0: sig = 2*(b[1]-b[0])
    #idx = np.where(((t0-mu) > -8 * sig) & ((t0-mu) < 8 * sig))
    #ilo, ihi = idx[0][0], idx[0][-1]
    #t0, h0 = t0[ilo:ihi], h0[ilo:ihi]
    guess = (hmax, mu, sig)
    bounds = ((hmax*0.8, mu*0.98, 0),(hmax*1.2, mu*1.02, 2*sig))
    return guess, bounds


# ##### ALIGNMENT ######

def step_function(lenght):
    x = np.linspace(0, lenght, lenght)
    step_function = np.piecewise(x, [((x >= 0) & (x < lenght/4)), ((x >= lenght/4) & (x <= 3*lenght/4)), 
                                     ((x > 3*lenght/4) & (x <= lenght))], [-1, 1, -1])
    return step_function

def align_waveform(wfs, nw=10, new_len = 200, smin = 100, plot = True):
    fig1, ax1 = plt.subplots(nrows=5, ncols=2,figsize=(16,16), facecolor='white')
    fig2, ax2 = plt.subplots(nrows=5, ncols=2,figsize=(16,16), facecolor='white')
    nwfs, wsize = wfs.shape[0], wfs.shape[1]
    step = step_function(lenght = 16)
    i = 0
    for wf in wfs:
        if i >= 10: break
        wf_step = np.convolve(wf, step, 'valid')
        ss = int((len(wf)-len(wf_step)+1)/2)
        if wf_step.max() < smin: continue
        peaks, pr = signal.find_peaks(wf_step, height=smin)
        if len(peaks) > 1: continue
        centroid = np.where(wf_step[wf_step.argmin():] > 0)[0][0] + wf_step.argmin()
        sleft, sright = int(centroid - new_len/2 + ss), int(centroid + new_len/2 + ss)
        wf_a = wf[sleft:sright]
        if len(wf_a) < new_len: continue
        axis1, axis2 = ax1.flat[i], ax2.flat[i]
        axis1.plot(wf_step,label='wf step convoluted')
        axis1.plot(peaks,wf_step[peaks], "x",color='m')
        axis1.axvline(centroid,color='r',label='centroid')
        axis1.axhline(smin,color='g',label='threshold')
        axis1.axhline(0,c='k')
        axis1.legend()
        #axis1.set_xlim(centroid-20,centroid+20)
        axis2.plot(wf_a)
        axis2.axvline(new_len/2,color='r')
        #axis2.set_xlim(new_len/2-20,new_len/2+20)
        i += 1     

def gaussian(x, a, mu, sigma):
    return a * np.exp(-(x - mu)**2 / (2. * sigma**2))

def expo(x, a, t):
    return a*np.exp(-x/t)

def linear(x, a, b):
    return a * x + b

def gauss_const(x, A, mu, sigma, c ):
    return A * np.exp(-(x - mu)**2 / (2. * sigma**2)) + c

def res_model(x, a, b):
    return np.sqrt(a + b * x)


def qdrift_processing(chn, wfs, db_file, baseline = None, n_file = 1, dt_lim = 10000, plot = False):
    n_ev, wsize = len(wfs), len(wfs[0])
    print('Number of events',n_ev)
    with open(db_file, 'r') as fp:
        db = json.load(fp)
    ns, us = 1/16, 1000/16
    tau, tau2, frac = float(db[chn]['pz']['tau']), db[chn]['pz']['tau2'], float(db[chn]['pz']['frac'])
    tau2 = float(tau2.split('*us')[0])*us
    
    t_start = time.time()
    if baseline is not None: wf_blsub = bl_subtract(wfs, baseline)
    else: wf_blsub = wfs
    wf_pz = double_pole_zero(wf_blsub,tau,tau2,frac)
    wf_trap2 = trap_norm(wf_pz, int(4*us), int(96*ns))
    bl_mean , bl_std, bl_slope, bl_intercept = linear_slope_fit(wf_blsub[:,:2500])
    
    wf_t0_filter_func = t0_filter(128*ns, 2*us)
    wf_t0_filter = np.empty((n_ev,wsize))
    wf_t0_filter_func(wf_pz,wf_t0_filter)
    conv_tmin, tp_start, conv_min, conv_max = min_max(wf_t0_filter)
    tp_0_est = time_point_thresh(wf_t0_filter, (bl_mean/8) + bl_std, tp_start, 0)
    trapQftp = np.empty(n_ev)
    for j, wf in enumerate(wf_trap2):
        if not np.isnan(tp_0_est[j]):
            trapQftp[j] = fixed_time_pickoff(wf, int(tp_0_est[j] + 8.096*us), 1)
            #except:
            #    print('fixed_time_pickoff ERROR',j,tp_0_est[j])
            #    continue
    QDrift = np.multiply(trapQftp, 4000)
    # dt_eff
    riset, flatt = '10*us', '3.008*us'#db[chn]['ttrap']['rise'], db[chn]['ttrap']['flat']
    #riset, flatt = db[chn]['etrap']['rise'], db[chn]['etrap']['flat']
    rise, flat = float(riset.split('*us')[0])*us, float(flatt.split('*us')[0])*us
    wf_trap = trap_norm(wf_pz, int(rise), int(flat))
    trapTmax = np.empty(n_ev)
    for j, wf in enumerate(wf_trap):
        try: trapTmax[j] = np.amax(wf[round((2*rise+flat)):len(wf)-round((2*rise+flat))])
        except: print(j,wf)
    #trapTmax = np.amax(wf_trap[:,round((2*rise+flat)):len(wf_trap)-round((2*rise+flat))],axis=1)
    dt_eff = np.divide(QDrift, trapTmax, out=np.zeros_like(QDrift), where=trapTmax!=0)
    dt_eff[(dt_eff < 0) | (dt_eff > dt_lim)] = 0
    print(f"Time to calculate dt_eff: {(time.time() - t_start):.2f} sec")
    if plot:
        plt.figure(figsize=(12,6.75), facecolor='white')
        plt.plot(wf_blsub[1],label='wf_blsub')
        plt.plot(wf_pz[1],label='wf_pz')
        plt.plot(wf_trap2[1],label='wf_trap2 (4*us, 96*ns)')
        plt.plot(wf_t0_filter[1],label='wf_t0_filter (128*ns, 2*us)')
        plt.plot(wf_trap[1],label=f'wf_trap ({riset}, {flatt})')
        plt.axvline(tp_start[1],label=f'tp_start = {tp_start[1]}',c='k',ls=':')
        plt.axvline(tp_0_est[1],label=f'tp_0_est (threshold point) = {tp_0_est[1]}',c='r',ls=':')
        tp_pickoff = int(tp_0_est[1] + 8.096*us)
        plt.axvline(tp_pickoff,label=f'tp_pickoff (tp_0_est + 8.096*us) = {tp_pickoff}',c='g',ls=':')
        plt.plot(tp_pickoff, trapQftp[1],label=f'trapQftp (wf_trap2[tp_pickoff]) = {trapQftp[1]:.1f}',c='b',ls='',
             ms=5,marker='o')
        plt.plot(wf_trap[1].argmax(), trapTmax[1],label=f'trapTmax = {trapTmax[1]:.1f}',c='r',ls='',ms=5,marker='o')
        plt.xlim(2000,4500)
        plt.legend(title=f'QDrift = trapQftp*4000 = {QDrift[1]:.1f}\ndt_eff = QDrift/trapTmax = {dt_eff[1]:.2f}')
    return dt_eff

from multihist import Hist1d, Histdd

def plot_energy_drift(ene, dt_eff, lim=(2600,2630), a1 = None, a2 = None, title = 'energy'):
    ue_space, dt_space = np.linspace(lim[0],lim[1],100), np.linspace(0,1500,300)
    ph = Histdd(ene, dt_eff, bins=(ue_space, dt_space))
    plt.figure(figsize=(12,6.75), facecolor='white')
    ph.plot(log_scale=True,cmap='viridis')
    plt.axvline(2614.5,label='peak at 2614.5 keV',c='k')
    if a1 is not None:
        m = (a2[1]-a1[1]) / (a2[0]-a1[0])
        q = a1[1]-m*a1[0]
        plt.plot(ue_space,ue_space*m+q,c='r',label=f'correction: {-1/m:.4f}')
    plt.legend(title=f'QDrift vs {title}',loc='lower right')
    plt.xlabel('energy (keV)')
    plt.ylabel('dt eff')


def zac_filter_loc(length: int, sigma: float, flat: int, decay: int, dec = False):

    lt = int((length - flat) / 2)
    flat_int = int(flat)

    # calculate cusp filter and negative parables
    cusp = np.zeros(length)
    par = np.zeros(length)
    for ind in range(0, lt, 1):
        cusp[ind] = float(np.sinh(ind / sigma) / np.sinh(lt / sigma))
        par[ind] = np.power(ind - lt / 2, 2) - np.power(lt / 2, 2)
    for ind in range(lt, lt + flat_int + 1, 1):
        cusp[ind] = 1
    for ind in range(lt + flat_int + 1, length, 1):
        cusp[ind] = float(np.sinh((length - ind) / sigma) / np.sinh(lt / sigma))
        par[ind] = np.power(length - ind - lt / 2, 2) - np.power(lt / 2, 2)

    # calculate area of cusp and parables
    areapar, areacusp = 0, 0
    for i in range(0, length, 1):
        areapar += par[i]
        areacusp += cusp[i]

    # normalize parables area
    par = -par / areapar * areacusp
    
    # create zac filter
    zac = cusp + par
    
    # deconvolve zac filter
    den = [1, -np.exp(-1 / decay)]
    if not dec: zac = np.convolve(zac, den, "same")
    return zac

def cusp_filter_loc(length: int, sigma: float, flat: int, decay: int, dec = False):

    lt = int((length - flat) / 2)
    flat_int = int(flat)
    cusp = np.zeros(length)
    for ind in range(0, lt, 1):
        cusp[ind] = float(np.sinh(ind / sigma) / np.sinh(lt / sigma))
    for ind in range(lt, lt + flat_int + 1, 1):
        cusp[ind] = 1
    for ind in range(lt + flat_int + 1, length, 1):
        cusp[ind] = float(np.sinh((length - ind) / sigma) / np.sinh(lt / sigma))
    
    den = [1, -np.exp(-1 / decay)]
    if not dec: cusp = np.convolve(cusp, den, "same")
    return cusp

def check_memory():
    load1, load5, load15 = psutil.getloadavg()
    cpu_usage = (load15/os.cpu_count()) * 100
    print("The CPU usage is : ", cpu_usage)
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)


def delete_variables():
    for obj in dir():
        if not obj.startswith('__'):
            del globals()[obj]


def extract_peaks(wf, a_delta = 5, times_fwhm = 3, ratio = 0.8, width = 10, check = True):
    vt_max_out, vt_min_out, trigger_pos = np.zeros(20), np.zeros(20), np.zeros(20)
    n_max_out, n_min_out, flag_out, no_out = 0, 0, 0, 0
    hist_weights, hist_borders = np.zeros(100), np.zeros(101)
    histogram(wf, hist_weights, hist_borders)
    idx_out, max_out, fwhm = 0, 0, 0
    histogram_stats(hist_weights, hist_borders, idx_out, max_out, fwhm, np.nan)
    #idx_out, max_out, fwhm = hist_stats(hist_weights, hist_borders)
    get_multi_local_extrema(wf, a_delta, 0.1, 1, times_fwhm*fwhm, 0, vt_max_out, vt_min_out, n_max_out, n_min_out)
    if check: peak_snr_threshold(wf, vt_max_out, ratio, width, trigger_pos, no_out)
    else: trigger_pos = vt_max_out
    return trigger_pos

def hist_stats(weights_in, edges_in):
    if np.isnan(weights_in).any():
        return

    # find global maximum search from left to right
    max_index = 0
    if np.isnan(max_in):
        for i in range(0, len(weights_in), 1):
            if weights_in[i] > weights_in[max_index]:
                max_index = i

    # is user specifies mean justfind mean index
    else:
        if max_in > edges_in[-2]:
            max_index = len(weights_in) - 1
        else:
            for i in range(0, len(weights_in), 1):
                if abs(max_in - edges_in[i]) < abs(max_in - edges_in[max_index]):
                    max_index = i

    # returns left bin edge
    max_out = edges_in[max_index]

    # and the approx fwhm
    for i in range(max_index, len(weights_in), 1):
        if weights_in[i] <= 0.5 * weights_in[max_index] and weights_in[i] != 0:
            fwhm_out[0] = abs(max_out[0] - edges_in[i])
            break

    # look also into the other direction
    for i in range(0, max_index, 1):
        if weights_in[i] >= 0.5 * weights_in[max_index] and weights_in[i] != 0:
            if fwhm_out[0] < abs(max_out[0] - edges_in[i]):
                fwhm_out[0] = abs(max_out[0] - edges_in[i])
            break
    return max_index, max_out, fwhm_out

def dsp_analysis_old(run, chns, file_dir, db_file, config, n_file = None, s_file = 0, plot=False):
    store = lh5.LH5Store()
    fig1, axis1 = plt.subplots(nrows=5, ncols=2,figsize=(16,16), facecolor='white')
    for i, chn in enumerate(chns):
        print('\n',chn)
        ax1 = axis1.flat[i]
        count = 0
        for p, d, files in os.walk(file_dir):
            d.sort()    
            for i, f in enumerate(sorted(files)):
                if (f.endswith(".lh5")) & ("dsp" in f) & (i >= s_file):
                    lh5_file = f"{file_dir}/{f}"
                    trapEmax0, n_rows = store.read_object(f"{chn}/dsp/trapEmax", lh5_file)
                    cuspEmax0, n_rows = store.read_object(f"{chn}/dsp/cuspEmax", lh5_file)
                    zacEmax0, n_rows = store.read_object(f"{chn}/dsp/zacEmax", lh5_file)
                    dplmsEmax0, n_rows = store.read_object(f"{chn}/dsp/dplmsEmax", lh5_file)
                    dt_eff0, n_rows = store.read_object(f"{chn}/dsp/dt_eff", lh5_file)
                    if count == 0:
                        trapEmax, cuspEmax, zacEmax = trapEmax0.nda, cuspEmax0.nda, zacEmax0.nda
                        dplmsEmax, dt_eff = dplmsEmax0.nda, dt_eff0.nda
                    else:
                        trapEmax = np.append(trapEmax, trapEmax0.nda, axis=0)
                        cuspEmax = np.append(cuspEmax, cuspEmax0.nda, axis=0)
                        zacEmax = np.append(zacEmax, zacEmax0.nda, axis=0)
                        dplmsEmax = np.append(dplmsEmax, dplmsEmax0.nda, axis=0)
                        dt_eff = np.append(dt_eff, dt_eff0.nda, axis=0)
                    count += 1
                if n_file is not None and count >= n_file: break
        #l_cal_c, fwhm_c, fwhm_err_c, chi2 = pcal.calibrate_th228(cuspEmax, plot=0)
        #l_cal_z, fwhm_z, fwhm_err_z, chi2 = pcal.calibrate_th228(zacEmax, plot=0)
        #l_cal, fwhm, fwhm_err, chi2 = pcal.calibrate_th228(dplmsEmax, plot=0)
        #plot_energy_drift(l_cal_c[0] * cuspEmax + l_cal_c[1], dt_eff , title='cuspEmax')
        #plot_energy_drift(l_cal_z[0] * zacEmax + l_cal_z[1], dt_eff )
        #plot_energy_drift(l_cal[0] * dplmsEmax + l_cal[1], dt_eff, title='dplmsEmax' )
    
        with open(db_file, 'r') as fp:
            db = json.load(fp)
        alpha_cusp = db[chn]['ctc_params']['cuspEmax_ctc']['parameters']['a']
        alpha_zac = db[chn]['ctc_params']['zacEmax_ctc']['parameters']['a']
        alpha_dplms = db[chn]['ctc_params']['dplms']['parameters']['a']
        print('dplms',db[chn]['dplms']['a5'],db[chn]['dplms']['ff'],'alpha_dplms',alpha_dplms)
        corr_cusp = np.multiply(np.multiply(alpha_cusp, dt_eff, dtype="float64"), cuspEmax, dtype="float64")
        cuspEmax_ctc = np.add(corr_cusp, cuspEmax)
        corr_zac = np.multiply(np.multiply(alpha_zac, dt_eff, dtype="float64"), zacEmax, dtype="float64")
        zacEmax_ctc = np.add(corr_zac, zacEmax)
        corr = np.multiply(np.multiply(alpha_dplms, dt_eff, dtype="float64"), dplmsEmax, dtype="float64")
        dplmsEmax_ctc = np.add(corr, dplmsEmax)
        result_map_c = pcal.calibrate_th228(cuspEmax_ctc, plot=0)
        result_map_z = pcal.calibrate_th228(zacEmax_ctc, plot=0)
        result_map = pcal.calibrate_th228(dplmsEmax_ctc, plot=0)
        #plot_energy_drift(l_cal_c[0] * cuspEmax_ctc + l_cal_c[1], dt_eff , title='cuspEmax_ctc')
        #plot_energy_drift(l_cal[0] * dplmsEmax_ctc + l_cal[1], dt_eff, title='dplmsEmax_ctc' )
        
        ebins = np.arange(0, 2700)
        res_par_c = [result_map_c['resolution_curve'][a] for a in result_map_c['resolution_curve'].keys()]
        res_par_z = [result_map_z['resolution_curve'][a] for a in result_map_z['resolution_curve'].keys()]
        res_par = [result_map['resolution_curve'][a] for a in result_map['resolution_curve'].keys()]
        #plt.errorbar(res_peaks, fwhm_fit, yerr = fwhm_efit, fmt='o',color='b',label='data')
        #eb1=plt.errorbar(no_peaks, fwhm_no, yerr = fwhm_eno, fmt='o',color='k',label='peaks not used')
        #eb1[-1][0].set_linestyle(':')
        ax1.set_xlabel('energy [keV]')
        ax1.set_ylabel('FWHM [keV]')
        ax1.plot(ebins, res_model(ebins, *res_par_c),'g-')#, label='cusp)
        ax1.plot(ebins, res_model(ebins, *res_par_z),'b-')#, label='zac')
        ax1.plot(ebins, res_model(ebins, *res_par),'r-')#, label='dplms')
        fwhmqbb_c, fwhmqbb_err_c = result_map_c['qbb']['fwhm'], result_map_c['qbb']['fwhm_err']
        fwhmqbb_z, fwhmqbb_err_z = result_map_z['qbb']['fwhm'], result_map_z['qbb']['fwhm_err']
        fwhmqbb, fwhmqbb_err = result_map['qbb']['fwhm'], result_map['qbb']['fwhm_err']
        fwhm_c, fwhm_err_c = result_map_c[2614.553]['fwhm'], result_map_c[2614.553]['fwhm_err']
        fwhm_z, fwhm_err_z = result_map_z[2614.553]['fwhm'], result_map_z[2614.553]['fwhm_err']
        fwhm, fwhm_err = result_map[2614.553]['fwhm'], result_map[2614.553]['fwhm_err']
        print(f'cusp: FWHM at 2.6 MeV = {fwhm_c:.2f} +/- {fwhm_err_c:.2f} keV')
        print(f'zac: FWHM at 2.6 MeV  = {fwhm_z:.2f} +/- {fwhm_err_z:.2f} keV')
        print(f'dplms: FWHM at 2.6 MeV = {fwhm:.2f} +/- {fwhm_err:.2f} keV\n')
        ax1.plot(2039,fwhmqbb_c,'gx',ms=10)#,label=f'$FWHM={fwhmqbb_c:.2f}\pm{fwhmqbb_err_c:.2f}$ keV')
        ax1.plot(2039,fwhmqbb_z,'bx',ms=10)#,label=f'$FWHM={fwhmqbb_z:.2f}\pm{fwhmqbb_err_z:.2f}$ keV')
        ax1.plot(2039,fwhmqbb,'rx',ms=10)#,label=f'$FWHM={fwhmqbb:.2f}\pm{fwhmqbb_err:.2f}$ keV')
        ax1.errorbar(result_map_c['res_peaks'], result_map_c['res_fwhm'], yerr = result_map_c['res_fwhm_err'],
                     fmt='o',color='g',label=f'cusp: FWHM$={fwhmqbb_c:.2f}\pm{fwhmqbb_err_c:.2f}$ keV')
        ax1.errorbar(result_map_z['res_peaks'], result_map_z['res_fwhm'], yerr = result_map_z['res_fwhm_err'],
                     fmt='o',color='b',label=f'zac: FWHM$={fwhmqbb_z:.2f}\pm{fwhmqbb_err_z:.2f}$ keV')
        ax1.errorbar(result_map['res_peaks'], result_map['res_fwhm'], yerr = result_map['res_fwhm_err'],
                     fmt='o',color='r',label=f'dplms: FWHM$={fwhmqbb:.2f}\pm{fwhmqbb_err:.2f}$ keV')
        eb0 = ax1.errorbar(result_map_c['peaks_not'], result_map_c['fwhm_not'], yerr = result_map_c['fwhm_err_not'],
                           fmt='o',color='g')
        eb1 = ax1.errorbar(result_map_z['peaks_not'], result_map_z['fwhm_not'], yerr = result_map_z['fwhm_err_not'],
                           fmt='o',color='b')
        eb2 = ax1.errorbar(result_map['peaks_not'], result_map['fwhm_not'], yerr = result_map['fwhm_err_not'],
                           fmt='o',color='r')
        eb0[-1][0].set_linestyle(':')
        eb1[-1][0].set_linestyle(':')
        eb2[-1][0].set_linestyle(':')
        det_id = config['hardware_configuration']['channel_map'][chn]['det_id']
        ax1.legend(loc='upper left',title=f'{chn} - {det_id}')


def dsp_analysis(time_string, period, run, proc_channels, db_file, db_file_prod,
                 pulser_id = 'ch1027201', n_file = None, s_file = 0, res_file = None):
    store = lh5.LH5Store()
    chmap = lmeta.hardware.configuration.channelmaps.on(time_string)
    ge_all = [ch for ch in chmap.keys() if chmap[ch]['system']=='geds']
    ge_off = ['V07298B', 'P00665A', 'V01386A', 'V01403A', 'V01404A', 'B00091D', 'P00537A', 'B00091B', 'P00538B', 'P00661A', 'P00665B', 'P00698B']
    ge_drift = ['V01406A', 'V01415A', 'V01387A', 'P00665C','P00748B', 'P00748A']
    ge_keys = [ch for ch in ge_all if ch not in ge_off]
    ge_keys = [ch for ch in ge_keys if ch not in ge_drift]
    ge_rawid = [chmap[ch]['daq']['rawid']  for ch in ge_keys]
    ge_table = [f'ch{rawid}' for rawid in ge_rawid]
    ge_keys, ge_table = ge_keys[proc_channels[0]:1+proc_channels[1]], ge_table[proc_channels[0]:1+proc_channels[1]]
    
    if res_file is not None:
        try:
            with open (res_file, 'rb') as filehandler:
                results = pickle.load(filehandler)
            print('pickle file for results already existing')
        except:
            print('new pickle file for results')
            results = {}
    with open(db_file, 'r') as fp:
        db = json.load(fp)
    with open(db_file_prod, 'r') as fp:
        db_prod = json.load(fp)
    for i, (chn, det) in enumerate(zip(ge_table,ge_keys)):
        print('\n',chn,det)
        alpha_cusp = db_prod[chn]['ctc_params']['cuspEmax_ctc']['parameters']['a']
        alpha_zac = db_prod[chn]['ctc_params']['zacEmax_ctc']['parameters']['a']
        alpha_dplms = db[chn]['ctc_params']['dplms']['parameters']['a']
        file_dir = f'/data1/users/dandrea/test_data/dsp/{period}/{run}'
        
        count = 0
        for p, d, files in os.walk(file_dir):
            d.sort()    
            for ii, f in enumerate(sorted(files)):
                if (f.endswith(".lh5")) & ("dsp" in f) & (ii >= s_file):
                    lh5_file = f"{file_dir}/{f}"
                    trapEmax0, n_rows = store.read_object(f"{chn}/dsp/trapEmax", lh5_file)
                    cuspEmax0, n_rows = store.read_object(f"{chn}/dsp/cuspEmax", lh5_file)
                    zacEmax0, n_rows = store.read_object(f"{chn}/dsp/zacEmax", lh5_file)
                    dplmsEmax0, n_rows = store.read_object(f"{chn}/dsp/dplmsEmax", lh5_file)
                    dt_eff0, n_rows = store.read_object(f"{chn}/dsp/dt_eff", lh5_file)
                    if count == 0:
                        trapEmax, cuspEmax, zacEmax = trapEmax0.nda, cuspEmax0.nda, zacEmax0.nda
                        dplmsEmax, dt_eff = dplmsEmax0.nda, dt_eff0.nda
                    else:
                        trapEmax = np.append(trapEmax, trapEmax0.nda, axis=0)
                        cuspEmax = np.append(cuspEmax, cuspEmax0.nda, axis=0)
                        zacEmax = np.append(zacEmax, zacEmax0.nda, axis=0)
                        dplmsEmax = np.append(dplmsEmax, dplmsEmax0.nda, axis=0)
                        dt_eff = np.append(dt_eff, dt_eff0.nda, axis=0)
                    count += 1
                if n_file is not None and count >= n_file: break
        dsp_files = sorted(os.listdir(file_dir))
        if n_file is not None:
            dsp_files = [f'{file_dir}/{f}' for j, f in enumerate(dsp_files) if j < n_file]
        else:
            dsp_files = [f'{file_dir}/{f}' for j, f in enumerate(dsp_files)]
        dsp_data = store.read_object(f"{chn}/dsp", dsp_files)[0]
        #dsp_pul = store.read_object(f"{pulser_id}/dsp", dsp_files)[0]
        #pulser_mask = pulser_tag(dsp_data["timestamp"].nda, dsp_pul["timestamp"].nda)
        
        dsp_config='/data1/users/dandrea/software/legend-dataflow-config/pars_dplms_ge/L200-p03-r%-T%-ICPC-dsp_proc_chain_minimal.json'
        peaks_keV = np.array([238.632, 583.191, 727.330, 860.564, 1620.5, 2614.553])
        kev_widths = [[8,8],[25,30], [25,35],[25,40],[25,55], [70,70]]
        cal_list_file = open(f'l200-{period}/{run}/l200-{period}-{run}-cal-raw.list', "r")
        cal_list = cal_list_file.read().split('\n')
        tb_data, idx_list = event_selection(
            cal_list, f'{chn}/raw',
            dsp_config,
            db_prod[chn],
            peaks_keV,
            np.arange(0,len(peaks_keV),1).tolist(),
            kev_widths,
            cut_parameters = {"bl_std": 4,"bl_mean": 4},
            n_events = 5000,
            threshold = 0
        )
        
        a = True
        if a:
            #res_c = pcal.calibrate_th228(cuspEmax)#[pulser_mask])
            #res_z = pcal.calibrate_th228(zacEmax)#[pulser_mask])
            #res = pcal.calibrate_th228(dplmsEmax)#[pulser_mask])
            #cusp_cal = res_c['calibration_curve']['m'] * cuspEmax + res_c['calibration_curve']['q']
            #zac_cal = res_z['calibration_curve']['m'] * zacEmax + res_z['calibration_curve']['q']
            #dplms_cal = res['calibration_curve']['m'] * dplmsEmax + res['calibration_curve']['q']
            
            cusp_dict = dd.get_peaks_dict(chn, dsp_data, peaks_keV, kev_widths, idx_list=idx_list, ene_par = 'cuspEmax')
            zac_dict = dd.get_peaks_dict(chn, dsp_data, peaks_keV, kev_widths, idx_list=idx_list, ene_par = 'zacEmax')
            dplms_dict = dd.get_peaks_dict(chn, dsp_data, peaks_keV, kev_widths, idx_list=idx_list, ene_par = 'dplmsEmax')
            
            start_t = time.time()
            result_map_c = new_fom(dsp_data, cusp_dict, alpha = alpha_cusp)
            result_map_z = new_fom(dsp_data, zac_dict, alpha = alpha_zac)
            result_map = new_fom(dsp_data, dplms_dict, alpha = alpha_dplms)
            #result_map_c['calibration_curve'] =  res_c['calibration_curve']
            #result_map_z['calibration_curve'] =  res_z['calibration_curve']
            #result_map['calibration_curve'] =  res['calibration_curve']
            print(f'Time to calculate FOMs {time.time()-start_t:.1f}')
            fwhmqbb_c, fwhmqbb_err_c = result_map_c['qbb_fwhm'], result_map_c['qbb_fwhm_err']
            fwhmqbb_z, fwhmqbb_err_z = result_map_z['qbb_fwhm'], result_map_z['qbb_fwhm_err']
            fwhmqbb, fwhmqbb_err = result_map['qbb_fwhm'], result_map['qbb_fwhm_err']
            fwhm_c, fwhm_err_c = result_map_c['fwhms'][-1], result_map_c['fwhm_errs'][-1]
            fwhm_z, fwhm_err_z = result_map_z['fwhms'][-1], result_map_z['fwhm_errs'][-1]
            fwhm, fwhm_err = result_map['fwhms'][-1], result_map['fwhm_errs'][-1]
            print(f'cusp: FWHM at Qbb = {fwhmqbb_c:.2f} +/- {fwhmqbb_err_c:.2f} keV, FWHM at 2.6 MeV = {fwhm_c:.2f} +/- {fwhm_err_c:.2f} keV')
            print(f'zac: FWHM at Qbb = {fwhmqbb_z:.2f} +/- {fwhmqbb_err_z:.2f} keV, FWHM at 2.6 MeV = {fwhm_z:.2f} +/- {fwhm_err_z:.2f} keV')
            print(f'dplms: FWHM at Qbb = {fwhmqbb:.2f} +/- {fwhmqbb_err:.2f} keV, FWHM at 2.6 MeV = {fwhm:.2f} +/- {fwhm_err:.2f} keV')
        #except:
        #    continue
        if res_file is not None:
            try:
                bls_fwhm_c, bls_fwhm_err_c = results[chn]['cusp']['bls_fwhm'], results[chn]['cusp']['bls_fwhm_err']
                bls_fwhm_z, bls_fwhm_err_z = results[chn]['zac']['bls_fwhm'], results[chn]['zac']['bls_fwhm_err']
                bls_fwhm, bls_fwhm_err = results[chn]['dplms']['bls_fwhm'], results[chn]['dplms']['bls_fwhm_err']
            except:
                print('Baseline FWHM not in',res_file,'for',chn)
            try:
                results[chn]['cusp'], results[chn]['zac'], results[chn]['dplms'] = result_map_c, result_map_z, result_map
            except:
                results[chn] = {}
                results[chn]['cusp'], results[chn]['zac'], results[chn]['dplms'] = result_map_c, result_map_z, result_map
            try:
                results[chn]['cusp']['bls_fwhm'], results[chn]['cusp']['bls_fwhm_err'] = bls_fwhm_c, bls_fwhm_err_c
                results[chn]['zac']['bls_fwhm'], results[chn]['zac']['bls_fwhm_err'] = bls_fwhm_z, bls_fwhm_err_z
                results[chn]['dplms']['bls_fwhm'], results[chn]['dplms']['bls_fwhm_err'] = bls_fwhm, bls_fwhm_err
            except: pass
    if res_file is not None:
        filehandler = open(res_file, 'wb')
        pickle.dump(results, filehandler)
    return results


def plot_results(time_string, res_file, strings=[1], plot_dir=None):
    chmap = lmeta.hardware.configuration.channelmaps.on(time_string)
    ge_all = [ch for ch in chmap.keys() if chmap[ch]['system']=='geds']
    ge_off = ['V07298B', 'P00665A', 'V01386A', 'V01403A', 'V01404A', 'B00091D', 'P00537A', 'B00091B', 'P00538B', 'P00661A', 'P00665B', 'P00698B']
    ge_drift = ['V01406A', 'V01415A', 'V01387A', 'P00665C','P00748B', 'P00748A']
    
    with open (res_file, 'rb') as filehandler:
        results = pickle.load(filehandler)
    
    #strings = np.array([int(chmap[ch]['location']['string']) for ch in chmap.keys() if chmap[ch]['system']=='geds'])
    #if nstring is None: nstring = strings.max()
    #for string in range(1,nstring+1):
    string_line, n_ge = [0], 0
    for string in strings:
        print('string',string)
        ge_keys = [ch for ch in chmap.keys() if chmap[ch]['system']=='geds' and chmap[ch]['location']['string']==string]
        ge_keys = [ch for ch in ge_keys if ch not in ge_off]
        ge_keys = [ch for ch in ge_keys if ch not in ge_drift]
        ge_rawid = [chmap[ch]['daq']['rawid']  for ch in ge_keys]
        ge_table = [f'ch{rawid}' for rawid in ge_rawid]
        n_ge += len(ge_table)
        fc, fc_e = np.zeros(len(ge_table)), np.zeros(len(ge_table))
        fz, fz_e = np.zeros(len(ge_table)), np.zeros(len(ge_table))
        fd, fd_e = np.zeros(len(ge_table)), np.zeros(len(ge_table))
        fc_fep, fc_fep_e = np.zeros(len(ge_table)), np.zeros(len(ge_table))
        fz_fep, fz_fep_e = np.zeros(len(ge_table)), np.zeros(len(ge_table))
        fd_fep, fd_fep_e = np.zeros(len(ge_table)), np.zeros(len(ge_table))
        fc_puls, fz_puls, fd_puls = np.zeros(len(ge_table)), np.zeros(len(ge_table)), np.zeros(len(ge_table))
        fc_puls_e, fz_puls_e, fd_puls_e = np.zeros(len(ge_table)), np.zeros(len(ge_table)), np.zeros(len(ge_table))
        fc_bls, fz_bls, fd_bls = np.zeros(len(ge_table)), np.zeros(len(ge_table)), np.zeros(len(ge_table))
        fc_bls_e, fz_bls_e, fd_bls_e = np.zeros(len(ge_table)), np.zeros(len(ge_table)), np.zeros(len(ge_table))
        fig1, axis1 = plt.subplots(nrows=7, ncols=2,figsize=(16,28), facecolor='white')
        for i, (chn, det) in enumerate(zip(ge_table,ge_keys)):
            if det == 'V01389A': continue #Ortec ICPC with long tails on all peaks
            ax1 = axis1.flat[i]
            ebins = np.arange(0, 2700)
            ax1.set_xlabel('energy [keV]')
            ax1.set_ylabel('FWHM [keV]')
            failed = 0
            try:
                fc[i], fc_e[i] = results[chn]['cusp']['qbb_fwhm'], results[chn]['cusp']['qbb_fwhm_err']
                fc_fep[i], fc_fep_e[i] = results[chn]['cusp']['fwhms'][-1], results[chn]['cusp']['fwhm_errs'][-1]
                peaks, fwhms = results[chn]['cusp']['peaks'], results[chn]['cusp']['fwhms']
                fwhms = np.array([p for p, pe in zip(fwhms,peaks) if pe != 1620.5])
                peaks = np.array([pe for pe in peaks if pe != 1620.5])
                slope_cusp = fwhm_slope(ebins, *results[chn]['cusp']['fit_pars'])
                ax1.plot(ebins, slope_cusp,'b-')
                ax1.plot(2039,fc[i],'bx',ms=10)
                ax1.plot(peaks, fwhms,'ob',ls='',label=f'cusp: FWHM$_Q={fc[i]:.2f}({fc_e[i]*100:.0f})$ keV')
            except:
                failed += 1
                pass
            try:
                fz[i], fz_e[i] = results[chn]['zac']['qbb_fwhm'], results[chn]['zac']['qbb_fwhm_err']
                fz_fep[i], fz_fep_e[i] = results[chn]['zac']['fwhms'][-1], results[chn]['zac']['fwhm_errs'][-1]
                peaks, fwhms = results[chn]['zac']['peaks'], results[chn]['zac']['fwhms']
                fwhms = np.array([p for p, pe in zip(fwhms,peaks) if pe != 1620.5])
                peaks = np.array([pe for pe in peaks if pe != 1620.5])    
                slope_zac = fwhm_slope(ebins, *results[chn]['zac']['fit_pars'])
                ax1.plot(ebins, slope_zac,'g-')
                ax1.plot(2039,fz[i],'gx',ms=10)
                ax1.plot(peaks,fwhms,'og',ls='',label=f'zac: FWHM$_Q={fz[i]:.2f}({fz_e[i]*100:.0f})$ keV')
            except:
                failed += 1
                pass
            try:
                fd[i], fd_e[i] = results[chn]['dplms']['qbb_fwhm'], results[chn]['dplms']['qbb_fwhm_err']
                fd_fep[i], fd_fep_e[i] = results[chn]['dplms']['fwhms'][-1], results[chn]['dplms']['fwhm_errs'][-1]
                peaks, fwhms = results[chn]['dplms']['peaks'], results[chn]['dplms']['fwhms']
                fwhms = np.array([p for p, pe in zip(fwhms,peaks) if pe != 1620.5])
                peaks = np.array([pe for pe in peaks if pe != 1620.5])
                slope_dplms = fwhm_slope(ebins, *results[chn]['dplms']['fit_pars'])
                ax1.plot(ebins, slope_dplms,'m-')
                ax1.plot(2039,fd[i],'mx',ms=10)
                ax1.plot(peaks,fwhms,'om',ls='',label=f'dplms: FWHM$_Q={fd[i]:.2f}({fd_e[i]*100:.0f})$ keV')
            except:
                failed += 1
                pass
            if failed > 1:
                print('Results not found for',chn,det)
                n_ge -= 1
                continue
            else:
                ax1.legend(loc='lower right',title=f'{chn} - {det}',fontsize=9.5)
            # plotting pulser results
            pulser_results = True
            try:
                fc_puls[i] = results[chn]['pulser']['cusp']['fwhm']
                fc_puls_e[i] = results[chn]['pulser']['cusp']['fwhm_err']
                if fc_puls_e[i] > 0.02: fc_puls_e[i] = 0.02
                if fc_puls[i] > 0:
                    puls_c, = ax1.plot(0, fc_puls[i],ls='',marker='o', ms=10, markerfacecolor='none', markeredgecolor='b')
            except:
                pulser_results = False
                print('Pulser results not found for cusp',chn,det)
                pass
            try:
                fz_puls[i] = results[chn]['pulser']['zac']['fwhm']
                fz_puls_e[i] = results[chn]['pulser']['zac']['fwhm_err']
                if fz_puls_e[i] > 0.02: fz_puls_e[i] = 0.02
                if fz_puls[i] > 0:
                    puls_z, = ax1.plot(0, fz_puls[i],ls='',marker='o', ms=10, markerfacecolor='none', markeredgecolor='g')
            except:
                pulser_results = False
                print('Pulser results not found for zac',chn,det)
                pass
            try:
                fd_puls[i] = results[chn]['pulser']['dplms_pul']['fwhm']
                fd_puls_e[i] = results[chn]['pulser']['dplms_pul']['fwhm_err']
                if fd_puls[i] > 0:
                    puls_d, = ax1.plot(0, fd_puls[i],ls='',ms=10,marker='o',markerfacecolor='none',markeredgecolor='m')
            except KeyError:
                print('Pulser results with standard dplms', chn, det)
                fd_puls[i] = results[chn]['pulser']['dplms']['fwhm']
                fd_puls_e[i] = results[chn]['pulser']['dplms']['fwhm_err']
                if fd_puls[i] > 0:
                    puls_d, = ax1.plot(0, fd_puls[i],ls='',ms=10,marker='o',markerfacecolor='none',markeredgecolor='m')
            except Exception as e:
                pulser_results = False
                print('Pulser results not found for dplms', chn, det)
                print(e)
            try:
                fc_bls[i] = results[chn]['baseline']['cusp']['fwhm']
                fc_bls_e[i] = results[chn]['baseline']['cusp']['fwhm_err']
                #fz_bls[i] = results[chn]['baseline']['zac']['fwhm']
                #fz_bls_e[i] = results[chn]['baseline']['zac']['fwhm_err']
                fd_bls[i] = results[chn]['baseline']['dplms']['fwhm']
                fd_bls_e[i] = results[chn]['baseline']['dplms']['fwhm_err']
            except:
                #print('Baseline results not found',chn,det)
                pass
            if pulser_results:
                second_legend_handles = [puls_c, puls_z, puls_d]
                second_legend = ax1.legend(second_legend_handles,
                                           [f'cusp: FWHM$ = {fc_puls[i]:.2f}({fc_puls_e[i]*100:.0f})$ keV',
                                            f'zac: FWHM$ = {fz_puls[i]:.2f}({fz_puls_e[i]*100:.0f})$ keV',
                                            f'dplms: FWHM$ = {fd_puls[i]:.2f}({fd_puls_e[i]*100:.0f})$ keV'],
                                           loc='upper left',fontsize=9.5)
                second_legend.set_title('pulser results')
                ax1.add_artist(second_legend)
                ax1.legend(loc='lower right',title=f'{chn} - {det}',fontsize=9.5)
        
        string_line.append(n_ge)
        if string == strings[0]:
            dets, fwhmc, fwhmz, fwhmd = np.array(ge_keys), fc, fz, fd
            fwhmc_err, fwhmz_err, fwhmd_err = fc_e, fz_e, fd_e
            fwhmc_fep, fwhmz_fep, fwhmd_fep = fc_fep, fz_fep, fd_fep
            fwhmc_fep_err, fwhmz_fep_err, fwhmd_fep_err = fc_fep_e, fz_fep_e, fd_fep_e
            fwhmc_puls, fwhmz_puls, fwhmd_puls = fc_puls, fz_puls, fd_puls
            fwhmc_puls_err, fwhmz_puls_err, fwhmd_puls_err = fc_puls_e, fz_puls_e, fd_puls_e
            fwhmc_bls, fwhmz_bls, fwhmd_bls = fc_bls, fz_bls, fd_bls
            fwhmc_bls_err, fwhmz_bls_err, fwhmd_bls_err = fc_bls_e, fz_bls_e, fd_bls_e
        else:
            dets, fwhmc, fwhmz, fwhmd = np.append(dets,ge_keys), np.append(fwhmc,fc), np.append(fwhmz,fz), np.append(fwhmd,fd)
            fwhmc_err, fwhmz_err, fwhmd_err = np.append(fwhmc_err,fc_e), np.append(fwhmz_err,fz_e), np.append(fwhmd_err,fd_e)
            fwhmc_fep, fwhmz_fep, fwhmd_fep = np.append(fwhmc_fep,fc_fep), np.append(fwhmz_fep,fz_fep), np.append(fwhmd_fep,fd_fep)
            fwhmc_fep_err, fwhmz_fep_err, fwhmd_fep_err = np.append(fwhmc_fep_err,fc_fep_e), np.append(fwhmz_fep_err,fz_fep_e), np.append(fwhmd_fep_err,fd_fep_e)
            fwhmc_puls, fwhmz_puls, fwhmd_puls = np.append(fwhmc_puls,fc_puls), np.append(fwhmz_puls,fz_puls), np.append(fwhmd_puls,fd_puls)
            fwhmc_puls_err, fwhmz_puls_err, fwhmd_puls_err = np.append(fwhmc_puls_err,fc_puls_e), np.append(fwhmz_puls_err,fz_puls_e), np.append(fwhmd_puls_err,fd_puls_e)
            fwhmc_bls, fwhmz_bls, fwhmd_bls = np.append(fwhmc_bls,fc_bls), np.append(fwhmz_bls,fz_bls), np.append(fwhmd_bls,fd_bls)
            fwhmc_bls_err, fwhmz_bls_err, fwhmd_bls_err = np.append(fwhmc_bls_err,fc_bls_e), np.append(fwhmz_bls_err,fz_bls_e), np.append(fwhmd_bls_err,fd_bls_e)
        if plot_dir is not None:
            fig1.savefig(f'{plot_dir}/resolution_results_string{string:02}.png',dpi=200, bbox_inches='tight')
    print('Total detectors',n_ge)
    mask = (fwhmc>0) & (fwhmz>0) & (fwhmd>0)
    fwhmc = fwhmc[mask]
    fwhmz = fwhmz[mask]
    fwhmd = fwhmd[mask]
    fwhmc_err = fwhmc_err[mask]
    fwhmz_err = fwhmz_err[mask]
    fwhmd_err = fwhmd_err[mask]
    fwhmc_fep = fwhmc_fep[mask]
    fwhmz_fep = fwhmz_fep[mask]
    fwhmd_fep = fwhmd_fep[mask]
    fwhmc_fep_err = fwhmc_fep_err[mask]
    fwhmz_fep_err = fwhmz_fep_err[mask]
    fwhmd_fep_err = fwhmd_fep_err[mask]
    fwhmc_puls = fwhmc_puls[mask]
    fwhmz_puls = fwhmz_puls[mask]
    fwhmd_puls = fwhmd_puls[mask]
    fwhmc_puls_err = fwhmc_puls_err[mask]
    fwhmz_puls_err = fwhmz_puls_err[mask]
    fwhmd_puls_err = fwhmd_puls_err[mask]
    fwhmc_bls = fwhmc_bls[mask]
    fwhmz_bls = fwhmz_bls[mask]
    fwhmd_bls = fwhmd_bls[mask]
    fwhmc_bls_err = fwhmc_bls_err[mask]
    fwhmz_bls_err = fwhmz_bls_err[mask]
    fwhmd_bls_err = fwhmd_bls_err[mask]
    dets = dets[mask]
    
    # Qbb
    fig, axis = plt.subplots(figsize=(20,8), facecolor='white')
    for i, string in enumerate(strings):
        plt.axvspan(string_line[i]-0.5,string_line[i+1]-0.5,color=cols[i],alpha=0.1)#, label=f'string {string}')
    plt.errorbar(dets, fwhmc, yerr=fwhmc_err,fmt='^',color='b',ms=8,ls='',label=f'cusp average {fwhmc.mean():.2f} keV')
    plt.errorbar(dets, fwhmz, yerr=fwhmz_err,fmt='v',color='g',ms=8,ls='',label=f'zac average {fwhmz.mean():.2f} keV')
    plt.errorbar(dets, fwhmd, yerr=fwhmd_err,fmt='o',color='m',ms=8,ls='',label=f'dplms average {fwhmd.mean():.2f} keV')
    plt.ylabel('FWHM at Qbb (keV)')
    plt.xlabel('channel')
    plt.xticks(rotation=90)
    leg = plt.legend(title=r'FWHM at Q$_{\beta\beta}$',fontsize=14)
    leg.get_title().set_fontsize(14)
    if plot_dir is not None:
        fig.savefig(f'{plot_dir}/resolution_results.png',dpi=200, bbox_inches='tight')
    
    # pulser
    figp, axisp = plt.subplots(figsize=(20,8), facecolor='white')
    for i, string in enumerate(strings):
        plt.axvspan(string_line[i]-0.5,string_line[i+1]-0.5,color=cols[i],alpha=0.1)#, label=f'string {string}')
    plt.errorbar(dets, fwhmc_puls, yerr=fwhmc_puls_err, fmt='^',color='b',ms=8,ls='',label=f'cusp {fwhmc_puls.mean():.2f} keV')
    plt.errorbar(dets, fwhmz_puls, yerr=fwhmz_puls_err, fmt='v',color='g',ms=8,ls='',label=f'zac {fwhmz_puls.mean():.2f} keV')
    plt.errorbar(dets, fwhmd_puls, yerr=fwhmd_puls_err, fmt='o',color='m',ms=8,ls='',label=f'dplms {fwhmd_puls.mean():.2f} keV')
    plt.ylabel('FWHM at pulser (keV)')
    plt.xlabel('channel')
    plt.xticks(rotation=90)
    leg = plt.legend(title='FWHM at Pulser',fontsize=14)
    leg.get_title().set_fontsize(14)
    if plot_dir is not None:
        figp.savefig(f'{plot_dir}/resolution_results_puls.png',dpi=200, bbox_inches='tight')
    """
    # baseline
    figb, axisb = plt.subplots(figsize=(20,8), facecolor='white')
    for i, string in enumerate(strings):
        plt.axvspan(string_line[i]-0.5,string_line[i+1]-0.5,color=cols[i],alpha=0.1)#, label=f'string {string}')
    plt.errorbar(dets, fwhmc_bls, yerr=fwhmc_bls_err, fmt='^',color='g',ms=8,ls='',label='cusp')
    plt.errorbar(dets, fwhmz_bls, yerr=fwhmz_bls_err, fmt='v',color='b',ms=8,ls='',label='zac')
    plt.errorbar(dets, fwhmd_bls, yerr=fwhmd_bls_err, fmt='o',color='r',ms=8,ls='',label='dplms')
    plt.ylabel('FWHM at pulser (keV)')
    plt.xlabel('channel')
    plt.xticks(rotation=90)
    leg = plt.legend(title='FWHM at Pulser',fontsize=14)
    leg.get_title().set_fontsize(14)
    if plot_dir is not None:
        figb.savefig(f'{plot_dir}/resolution_results_bls.png',dpi=200, bbox_inches='tight')
    """
    # relative
    fig0, axis0 = plt.subplots(figsize=(20,8), facecolor='white')
    for i, string in enumerate(strings):
        plt.axvspan(string_line[i]-0.5,string_line[i+1]-0.5,color=cols[i],alpha=0.1)#, label=f'string {string}')
    fwhm_rel, fwhm_rel_err = calculate_fwhm_rel(fwhmc,fwhmd,fwhmc_err,fwhmd_err)
    fwhm_rel_puls, fwhm_rel_puls_err = calculate_fwhm_rel(fwhmc_puls,fwhmd_puls,fwhmc_puls_err,fwhmd_puls_err)
    fwhm_rel_bls, fwhm_rel_bls_err = calculate_fwhm_rel(fwhmc_bls,fwhmd_bls,fwhmc_bls_err,fwhmd_bls_err)
    mean, mean_err = calculate_mean_and_error(fwhm_rel, fwhm_rel_err)
    mean_puls, mean_err_puls = calculate_mean_and_error(fwhm_rel_puls, fwhm_rel_puls_err)
    mean_bls, mean_err_bls = calculate_mean_and_error(fwhm_rel_bls, fwhm_rel_bls_err)
    
    plt.errorbar(dets, fwhm_rel, yerr=fwhm_rel_err,fmt='o',ms=12,ls='',
                 label=f'cusp-dplms at Qbb\nmean improv.$= {mean:.2f}$ keV')
    plt.errorbar(dets, fwhm_rel_puls,yerr=fwhm_rel_puls_err,fmt='^',ms=12,ls='',
                 label=f'cusp-dplms at pulser\nmean improv.$= {mean_puls:.2f}$ keV')
    #plt.errorbar(dets, fwhm_rel_bls,yerr=fwhm_rel_bls_err,fmt='v',ms=12,ls='',
    #             label=f'cusp-dplms at baseline\nmean improv.$= {mean_bls:.2f}$ keV')
    plt.axhline(0,color='k',ls=':')
    plt.ylabel('FWHM improvement (keV)')
    plt.xlabel('channel')
    plt.xticks(rotation=90)
    plt.legend(fontsize=14)
    plt.ylim(-0.2,0.6)
    if plot_dir is not None:
        fig0.savefig(f'{plot_dir}/resolution_results_rel.png',dpi=200, bbox_inches='tight')
    """
    # ICPC
    fig1, axis1 = plt.subplots(figsize=(20,8), facecolor='white')
    icpc = [d for d in dets if 'V0' in d]
    fwhm_ = np.array([f for f, d in zip(fwhm_rel, dets) if 'V0' in d])
    fwhm_err_ = np.array([f for f, d in zip(fwhm_rel_err, dets) if 'V0' in d])
    fwhm_1_ = np.array([f for f, d in zip(fwhm_rel_1MeV, dets) if 'V0' in d])
    mean_icpc, _ = calculate_mean_and_error(fwhm_, fwhm_)
    mean_1_icpc, _ = calculate_mean_and_error(fwhm_1_, fwhm_1_)
    plt.errorbar(icpc, fwhm_, yerr=fwhm_err_,fmt='^',ms=12,ls='',label=f'cusp-dplms at Qbb\nICPC = {mean_icpc:.2f}%')
    plt.errorbar(icpc, fwhm_1_,fmt='v',ms=12,ls='',label=f'cusp-dplms at {comp_energy} keV\nICPC = {mean_1_icpc:.2f}%')
    plt.axhline(0,color='k',ls=':')
    plt.ylabel('FWHM improvement (%)')
    plt.xlabel('channel')
    plt.xticks(rotation=90)
    plt.legend(fontsize=14)
    if plot_dir is not None:
        fig1.savefig(f'{plot_dir}/resolution_results_icpc.png',dpi=200, bbox_inches='tight')
    # BEGe
    fig2, axis2 = plt.subplots(figsize=(20,8), facecolor='white')
    bege = [d for d in dets if 'B00' in d]
    fwhm_ = np.array([f for f, d in zip(fwhm_rel, dets) if 'B00' in d])
    fwhm_err_ = np.array([f for f, d in zip(fwhm_rel_err, dets) if 'B00' in d])
    fwhm_1_ = np.array([f for f, d in zip(fwhm_rel_1MeV, dets) if 'B00' in d])
    mean_bege, _ = calculate_mean_and_error(fwhm_, fwhm_)
    mean_1_bege, _ = calculate_mean_and_error(fwhm_1_, fwhm_1_)
    plt.errorbar(bege, fwhm_, yerr=fwhm_err_,fmt='^',ms=12,ls='',label=f'cusp-dplms at Qbb\nBEGe = {mean_bege:.2f}%')
    plt.errorbar(bege, fwhm_1_,fmt='v',ms=12,ls='',label=f'cusp-dplms at {comp_energy} keV\nBEGe = {mean_1_bege:.2f}%')
    plt.axhline(0,color='k',ls=':')
    plt.ylabel('FWHM improvement (%)')
    plt.xlabel('channel')
    plt.xticks(rotation=90)
    plt.legend(fontsize=14)
    if plot_dir is not None:
        fig2.savefig(f'{plot_dir}/resolution_results_bege.png',dpi=200, bbox_inches='tight')
    # PPC
    fig3, axis3 = plt.subplots(figsize=(20,8), facecolor='white')
    ppc = [d for d in dets if 'P00' in d]
    fwhm_ = np.array([f for f, d in zip(fwhm_rel, dets) if 'P00' in d])
    fwhm_err_ = np.array([f for f, d in zip(fwhm_rel_err, dets) if 'P00' in d])
    fwhm_1_ = np.array([f for f, d in zip(fwhm_rel_1MeV, dets) if 'P00' in d])
    mean_ppc, _ = calculate_mean_and_error(fwhm_, fwhm_)
    mean_1_ppc, _ = calculate_mean_and_error(fwhm_1_, fwhm_1_)
    plt.errorbar(ppc, fwhm_, yerr=fwhm_err_,fmt='^',ms=12,ls='',label=f'cusp-dplms at Qbb\nPPC = {mean_ppc:.2f}%')
    plt.errorbar(ppc, fwhm_1_,fmt='v',ms=12,ls='',label=f'cusp-dplms at {comp_energy} keV\nPPC = {mean_1_ppc:.2f}%')
    plt.axhline(0,color='k',ls=':')
    plt.ylabel('FWHM improvement (%)')
    plt.xlabel('channel')
    plt.xticks(rotation=90)
    plt.legend(fontsize=14)
    if plot_dir is not None:
        fig3.savefig(f'{plot_dir}/resolution_results_ppc.png',dpi=200, bbox_inches='tight')
    # COAX
    fig4, axis4 = plt.subplots(figsize=(20,8), facecolor='white')
    coax = [d for d in dets if 'C00' in d]
    fwhm_ = np.array([f for f, d in zip(fwhm_rel, dets) if 'C00' in d])
    fwhm_err_ = np.array([f for f, d in zip(fwhm_rel_err, dets) if 'C00' in d])
    fwhm_1_ = np.array([f for f, d in zip(fwhm_rel_1MeV, dets) if 'C00' in d])
    mean_coax, _ = calculate_mean_and_error(fwhm_, fwhm_)
    mean_1_coax, _ = calculate_mean_and_error(fwhm_1_, fwhm_1_)
    plt.errorbar(coax, fwhm_, yerr=fwhm_err_,fmt='^',ms=12,ls='',label=f'cusp-dplms at Qbb\nCOAX = {mean_coax:.2f}%')
    plt.errorbar(coax, fwhm_1_,fmt='v',ms=12,ls='',label=f'cusp-dplms at {comp_energy} keV\nCOAX = {mean_1_coax:.2f}%')
    plt.axhline(0,color='k',ls=':')
    plt.ylabel('FWHM improvement (%)')
    plt.xlabel('channel')
    plt.xticks(rotation=90)
    plt.legend(fontsize=14)
    if plot_dir is not None:
        fig4.savefig(f'{plot_dir}/resolution_results_coax.png',dpi=200, bbox_inches='tight')
    """

def calculate_mean_and_error(data, errors):
    mask = (errors > 0)
    data, errors = data[mask], errors[mask]
    mean = np.mean(data)
    squared_errors = np.square(errors)
    sum_squared_errors = np.sum(squared_errors)
    error = np.sqrt(sum_squared_errors) / np.sqrt(len(data))
    return mean, error

def calculate_fwhm_rel(fwhm_1,fwhm_2,fwhm_1_err,fwhm_2_err):
    fwhm_rel, fwhm_rel_err = np.zeros(len(fwhm_1)), np.zeros(len(fwhm_1))
    for i, (a, b, error_a, error_b) in enumerate(zip(fwhm_1,fwhm_2,fwhm_1_err,fwhm_2_err)):
        """fwhm_rel[i] = (a - b) / a * 100
        partial_derivative_c_a = (b / (a ** 2)) * 100
        partial_derivative_c_b = (-1 / a) * 100
        squared_error_a = error_a ** 2
        squared_error_b = error_b ** 2
        sum_squared_errors_partial_derivatives = (squared_error_a * (partial_derivative_c_a ** 2)) + (squared_error_b * (partial_derivative_c_b ** 2))
        fwhm_rel_err[i] = math.sqrt(sum_squared_errors_partial_derivatives)"""
        fwhm_rel[i] = (a - b)
        fwhm_rel_err[i] = (error_a**2 + error_b**2)**0.5
    return fwhm_rel, fwhm_rel_err


def get_peaks_dict(detector, ene_cal, dsp_data, ene_par = "dplmsEmax"):
    peaks = np.array([238.632, 583.191, 727.330, 860.564, 1620.5, 2614.553])
    idx_list, peak_dicts = [], []
    for pp in peaks:
        idxs = (ene_cal > pp - 100) & (ene_cal < pp + 100) & (~np.isnan(dsp_data["dt_eff"].nda))
        idx_list.append(np.where(idxs)[0])
        if pp == 2614.553:
            keV_width = [70,70]
            func = extended_radford_pdf
            gof_func = radford_pdf
        else:
            keV_width = [20,20]
            func = extended_gauss_step_pdf
            gof_func = gauss_step_pdf
        peak_dict = {"peak":pp, "kev_width": keV_width, "parameter": ene_par, 'func': func, 'gof_func': gof_func }
        peak_dicts.append(peak_dict)
    kwarg_dict = {}
    kwarg_dict["detector"] = detector
    kwarg_dict["peaks_keV"] = peaks
    kwarg_dict["idx_list"] = idx_list
    kwarg_dict["peak_dicts"] = peak_dicts
    kwarg_dict["ctc_param"] = "QDrift"
    return kwarg_dict


def filter_grids(chns, plot_dir, run = 22 ):
    fig1, axis1 = plt.subplots(nrows=5, ncols=2,figsize=(16,16), facecolor='white')
    for i, chn in enumerate(chns):
        ax1 = axis1.flat[i]
        df_grid = pd.read_pickle(f'{plot_dir}/filter_grid_{chn}.pkl')
        df_g = df_grid.loc[ (df_grid['chisquare2614'] < 30) & (df_grid['fwhm2614'] < 10) & (df_grid['fwhm2614_err'] < 1)]
        minidx = df_g['fwhm'].idxmin()
        df_min = df_g.loc[minidx]
        best, best_err, best_2614, best_2614_err = df_min['fwhm'], df_min['fwhm_err'], df_min['fwhm2614'], df_min['fwhm2614_err']
        a1, a5, ff, alpha = df_min['a1'], df_min['a5'], df_min['ff'], df_min['alpha']
        print(chn,a1, a5, ff, alpha,f'FWHM (Qbb) = {best:.2f} +/- {best_err:.2f} keV, FWHM (2.6 MeV) = {best_2614:.2f}  +/- {best_2614_err:.2f} keV')
        #df_g = df_g.loc[ (df_g['a1'] == df_min['a1']) & (df_g['a5'] == df_min['a5']) & (df_g['ff'] == df_min['ff']) ]
        ax1.errorbar(df_g['alpha'],df_g['fwhm'],yerr = df_g['fwhm_err'], fmt='.',c='b',label=r'FWHM at $Q_{\beta\beta}$')
        ax1.errorbar(df_g['alpha'],df_g['fwhm2614'],yerr = df_g['fwhm2614_err'], fmt='.',c='r',label=f'FWHM at 2.6 MeV')
        ax1.plot(df_min['alpha'],df_min['fwhm'],'bo',ms=8,label=f'FWHM = {best:.2f}+/-{best_err:.2f} keV')
        ax1.plot(df_min['alpha'],df_min['fwhm2614'],'ro',ms=8,label=f'FWHM = {best_2614:.2f}+/-{best_2614_err:.2f} keV')
        #ax1.axhline(0, color='black', linestyle=':')
        det_id = config['hardware_configuration']['channel_map'][chn]['det_id']
        ax1.set_xlabel('CT correction')
        ax1.set_ylabel('FWHM [keV]')
        ax1.legend(loc='upper right',title=f'{chn} - {det_id}')


def baseline_processing(run, chns, dets, phy_dir, db_file, res_file, n_sel = 1000, nwf = 100, down_sample = 6, plot_dir = None):
    fig1, axis1 = plt.subplots(nrows=2, ncols=2,figsize=(16,6.4), facecolor='white')
    fig2, axis2 = plt.subplots(nrows=2, ncols=2,figsize=(16,6.4), facecolor='white')
    
    with open(db_file, "r") as file:
        db = json.load(file)
    with open (res_file, 'rb') as filehandler:
        results = pickle.load(filehandler)
    for i, chn in enumerate(chns):
        det = dets[i]
        print('\n',chn,det)
        ax1, ax2 = axis1.flat[i], axis2.flat[i]
        bls = select_baselines(run, chn, det, phy_dir, n_sel = n_sel, blsub = 0, down_sample = down_sample)
        bls_sub = select_baselines(run, chn, det, phy_dir, n_sel = n_sel, blsub = 1, down_sample = down_sample)
        n_ev = len(bls)
        if nwf > n_ev: nwf = n_ev
        for j in range(nwf):
            if j == 0: ax1.plot(bls[j],label=chn)
            else: ax1.plot(bls[j])
        ax1.legend()
        #dplms
        t_start = time.time()
        wsize, fsize, sconv = 1015, 982, 33
        flo = int(wsize/2 - fsize/2)
        fhi = int(wsize/2 + fsize/2)
        a1, a2, a3 = db[chn]['dplms']['a1'], db[chn]['dplms']['a2'], db[chn]['dplms']['a3']
        a4, a5, ff = db[chn]['dplms']['a4'], db[chn]['dplms']['a5'], db[chn]['dplms']['ff']
        print('dplms filter',a1,a2,a3,a4,a5,ff)
        dplms_func = dplms_ge(db[chn]['dplms']['n_mat'],
                              db[chn]['dplms']['ref'],
                              db[chn]['dplms']['dec'],
                              db[chn]['dplms']['ft_mat'],
                              wsize, fsize, a1, a2, a3, a4, a5, ff)
        wf_dplms = np.zeros(( n_ev, sconv + 1 ))
        dplms_func(bls[:,:wsize],wf_dplms)
        dplms_ene = np.max(wf_dplms,axis=1)
        print(f"Time to process w/ dplms: {(time.time() - t_start):.2f} sec")
        # cusp
        t_start = time.time()
        ns, us = 1/16/down_sample, 1000/16/down_sample
        #tau, tau2, frac = float(db[chn]['pz']['tau']), db[chn]['pz']['tau2'], float(db[chn]['pz']['frac'])
        #tau2 = float(tau2.split('*us')[0])*us
        #bls_pz = double_pole_zero(bls, tau, tau2, frac)
        tau = db[chn]['pz']['tau']
        tau = float(tau.split('*ns')[0])*ns
        s_cusp, f_cusp = db[chn]['cusp']['sigma'], db[chn]['cusp']['flat']
        sigma_cusp, flat_cusp = float(s_cusp.split('*us')[0])*us, float(f_cusp.split('*us')[0])*us
        wf_cusp = np.zeros(( n_ev, sconv + 1 ))
        cusp_func = cusp_filter(fsize, sigma_cusp, int(flat_cusp), np.inf)
        cusp_func(bls_sub[:,:wsize],wf_cusp)
        cusp_ene = np.max(wf_cusp,axis=1)
        print(f"Time to process w/ cusp: {(time.time() - t_start):.2f} sec")
        # zac
        t_start = time.time()
        s_zac, f_zac = db[chn]['zac']['sigma'], db[chn]['zac']['flat']
        sigma_zac, flat_zac = float(s_zac.split('*us')[0])*us, float(f_zac.split('*us')[0])*us
        zac_func = zac_filter(fsize, sigma_zac, int(flat_zac), np.inf)
        wf_zac = np.zeros(( n_ev, sconv + 1 ))
        zac_func(bls_sub[:,:wsize],wf_zac)
        zac_ene = np.max(wf_zac,axis=1)
        print(f"Time to process w/ zac: {(time.time() - t_start):.2f} sec")
        
        xlo = np.percentile(dplms_ene, 0)
        xhi = np.percentile(dplms_ene, 100)
        nb = 500#int((xhi-xlo)/xpb)
        hd, bd = np.histogram(dplms_ene, bins=np.linspace(xlo,xhi,nb))
        hc, bc = np.histogram(cusp_ene, bins=np.linspace(xlo,xhi,nb))
        hz, bz = np.histogram(zac_ene, bins=np.linspace(xlo,xhi,nb))
        bc = pgh.get_bin_centers(bd)
        guess, b_ = get_gaussian_guess(hd, bc)
        pard, pcov = curve_fit(gaussian, bc, hd, p0=guess)
        perrd = np.sqrt(np.diag(pcov))
        fwhm = results[chn]['dplms']['calibration_curve']['m'] * pard[2] * 2.355
        fwhm_err = results[chn]['dplms']['calibration_curve']['m'] * perrd[2] * 2.355
        results[chn]['dplms']['bls_fwhm'] = fwhm
        results[chn]['dplms']['bls_fwhm_err'] = fwhm_err
        guess, b_ = get_gaussian_guess(hd, bc)
        parc, pcov = curve_fit(gaussian, bc, hc, p0=guess)
        perrc = np.sqrt(np.diag(pcov))
        fwhm_c = results[chn]['cusp']['calibration_curve']['m'] * parc[2] * 2.355
        fwhm_err_c = results[chn]['cusp']['calibration_curve']['m'] * perrc[2] * 2.355
        results[chn]['cusp']['bls_fwhm'] = fwhm_c
        results[chn]['cusp']['bls_fwhm_err'] = fwhm_err_c
        guess, b_ = get_gaussian_guess(hd, bc)
        parz, pcov = curve_fit(gaussian, bc, hz, p0=guess)
        perrz = np.sqrt(np.diag(pcov))
        fwhm_z = results[chn]['zac']['calibration_curve']['m'] * parz[2] * 2.355
        fwhm_err_z = results[chn]['zac']['calibration_curve']['m'] * perrz[2] * 2.355
        results[chn]['zac']['bls_fwhm'] = fwhm_z
        results[chn]['zac']['bls_fwhm_err'] = fwhm_err_z
        ax2.plot(bc,hd,c='r',label=f'dplms: FWHM = {fwhm:.2f} +/- {fwhm_err:.2f} keV')
        ax2.plot(bc,hc,c='g',label=f'cusp: FWHM = {fwhm_c:.2f} +/- {fwhm_err_c:.2f} keV')
        ax2.plot(bc,hz,c='b',label=f'zac: FWHM = {fwhm_z:.2f} +/- {fwhm_err_z:.2f} keV')
        ax2.legend(title=chn)
        filehandler = open(res_file, 'wb')
        pickle.dump(results, filehandler)

def select_zero_energy(energySeries, thr = 100):
    energySeries = energySeries[np.abs(energySeries) < thr]
    xlo = np.percentile(energySeries, 0)
    xhi = np.percentile(energySeries, 100)
    nb = 3000#int((xhi-xlo)/xpb)
    
    hist, bin_edges = np.histogram(energySeries, bins=np.linspace(xlo,xhi,nb))
    bin_centers = pgh.get_bin_centers(bin_edges)
    return hist, bin_centers

def dsp_analysis_phy(run, chns, file_dir, n_file = None, s_file = 0, plot = False, res_file = None):
    store = lh5.LH5Store()
    if res_file is not None:
        try:
            with open (res_file, 'rb') as filehandler:
                results = pickle.load(filehandler)
            print('pickle file for results already existing')
        except:
            print('new pickle file for results')
            results = {}
    fig1, axis1 = plt.subplots(nrows=5, ncols=2,figsize=(16,16), facecolor='white')
    for i, chn in enumerate(chns):
        ax1 = axis1.flat[i]
        count = 0
        for p, d, files in os.walk(file_dir):
            d.sort()    
            for ii, f in enumerate(sorted(files)):
                if (f.endswith(".lh5")) & ("dsp" in f) & (ii >= s_file):
                    lh5_file = f"{file_dir}/{f}"
                    trapEmax0, n_rows = store.read_object(f"{chn}/dsp/trapEmax", lh5_file)
                    cuspEmax0, n_rows = store.read_object(f"{chn}/dsp/cuspEmax", lh5_file)
                    zacEmax0, n_rows = store.read_object(f"{chn}/dsp/zacEmax", lh5_file)
                    dplmsEmax0, n_rows = store.read_object(f"{chn}/dsp/dplmsEmax", lh5_file)
                    dt_eff0, n_rows = store.read_object(f"{chn}/dsp/dt_eff", lh5_file)
                    if count == 0:
                        trapEmax, cuspEmax, zacEmax = trapEmax0.nda, cuspEmax0.nda, zacEmax0.nda
                        dplmsEmax, dt_eff = dplmsEmax0.nda, dt_eff0.nda
                    else:
                        trapEmax = np.append(trapEmax, trapEmax0.nda, axis=0)
                        cuspEmax = np.append(cuspEmax, cuspEmax0.nda, axis=0)
                        zacEmax = np.append(zacEmax, zacEmax0.nda, axis=0)
                        dplmsEmax = np.append(dplmsEmax, dplmsEmax0.nda, axis=0)
                        dt_eff = np.append(dt_eff, dt_eff0.nda, axis=0)
                    count += 1
                if n_file is not None and count >= n_file: break
        dsp_files = sorted(os.listdir(file_dir))
        dsp_files = [f'{file_dir}/{f}' for j, f in enumerate(dsp_files) if j < n_file]
        dsp_data = store.read_object(f"{chn}/dsp", dsp_files)[0]
        
        # calibration
        #result_map_c = pcal.calibrate_th228(cuspEmax, plot=1)
        #result_map_z = pcal.calibrate_th228(zacEmax, plot=0)
        #result_map = pcal.calibrate_th228(dplmsEmax, plot=0)
        hc,bc = select_zero_energy(cuspEmax)
        hz,bz = select_zero_energy(zacEmax)
        hd,bd = select_zero_energy(dplmsEmax)
        ax1.plot(bc,hc,c='g',label='cusp')
        ax1.plot(bz,hz,c='b',label='zac')
        ax1.plot(bd,hd,c='r',label='dplms')
        ax1.set_yscale('log')
        ax1.legend(title=chn)


def daq_convertion(file_dir, file_key, raw_file, stream_type, raw_config):
    
    if stream_type == 'FlashCam':
        file_ext = ".fcio"
        raw_config = None
    if stream_type == 'ORCA':
        file_ext = ".orca"
    
    daq_file = file_dir + '/' + file_key + file_ext
    print('daq file:',daq_file)
    #if raw_dir is None: raw_dir = file_dir.replace('daq','raw')
    #raw_file = raw_dir + '/' + file_key + '_raw.lh5'
    print('raw file:',raw_file)
    build_raw(daq_file, in_stream_type='ORCA', out_spec = raw_config,
              filekey = raw_file, buffer_size=1024, overwrite=True)

def pulser_tag(timestamp, t_pulser, gap = 0.01):
    pulser_idx = []
    for t in t_pulser:
        idx = np.where(np.abs(timestamp-t)<gap)[0]
        if len(idx)>0: pulser_idx.append(idx)
    pulser_idx = np.concatenate(pulser_idx)
    return np.array([False if i in pulser_idx else True for i in range(len(timestamp))])

def save_png_as_pdf(directory,string):
    imagelist = []
    for p, d, files in os.walk(directory):
        d.sort()    
        for filename in sorted(files):
            if (filename.endswith(".png")) & (string in filename) & ('checkpoint' not in filename):
                image = Image.open(os.path.join(directory, filename))
                im = image.convert('RGB')
                imagelist.append(im)
    pdf_file = f'{directory}/{string}_merged.pdf'
    print('save pdf',pdf_file)
    im.save(pdf_file,save_all=True, append_images=imagelist)


def guassian_minuit_fit(energies, fit_width = [8, 8], dx=0.1):
    lower_bound = np.percentile(energies, 50) - fit_width[0]
    upper_bound = np.percentile(energies, 50) + fit_width[1]
    fit_range = (lower_bound, upper_bound)
    hist, bins, var = pgh.get_hist(energies, dx=dx, range=fit_range)
    bin_cs = (bins[:-1] + bins[1:]) / 2
    guess, bounds = get_gaussian_guess(hist, bins)

    def cost_func(a, mu, sigma):
        return np.sum((gaussian(bin_cs, a, mu, sigma) - hist)**2)
    m = Minuit(cost_func, *guess)
    m.migrad()
    pars = np.array(m.values)
    errs = np.array(m.errors)
    cov = np.array(m.covariance)
    valid = ( m.valid & (~np.isnan(errs).any()))

    # goodness of fit
    gof_range = (pars[1]-3*pars[2], pars[1]+3*pars[2])
    hist_gof, bins_gof, var = pgh.get_hist(energies, dx=dx, range=gof_range)
    cs = goodness_of_fit(hist_gof, bins_gof, None, gaussian, m.values, method="Pearson")
    chisqr = (cs[0] / cs[1])
    fwhm = pars[2] * 2 * np.sqrt(2 * np.log(2))
    fwhm_err = np.sqrt(cov[2][2]) * 2 * np.sqrt(2 * np.log(2))
    result_dict = {
        'hist': hist,
        'bins': bins,
        'bin_cs': bin_cs,
        'fwhm': fwhm,
        'fwhm_err': fwhm_err,
        'chisqr': chisqr,
        'valid': valid,
        'gof_range': gof_range,
        'pars': pars
    }
    return result_dict

def moving_average(x, w=10):
    nn = len(x)
    a = np.zeros(nn)
    for i in range(nn):
        if i > w/2: ll = int(i - w/2)
        else: ll = 0
        if i < nn - w/2 - 1: hh = int(i+w/2)
        else: hh = int(nn-1)
        a[i] = np.mean(x[ll:hh])
    return a
