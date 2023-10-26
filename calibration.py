import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from scipy.signal import argrelextrema, medfilt, find_peaks_cwt, find_peaks
from scipy.ndimage.filters import gaussian_filter1d
from scipy.stats import norm
import scipy.optimize as op
from scipy.optimize import curve_fit

import pygama.math.histogram as pgh
#import pygama.math.peak_fitting as pgf
#import pygama.math.utils as pgu

import analysis_utility as au

def get_most_prominent_peaks(energySeries, xlo, xhi, xpb, min_snr=2, max_cwt_width = 20,
                             max_num_peaks=np.inf, plot=False):
    """                                                                                                  
    find the most prominent peaks in a spectrum by looking for spikes in derivative of spectrum          
    energySeries: array of measured energies                                                             
    max_num_peaks = maximum number of most prominent peaks to find                                       
    return a histogram around the most prominent peak in a spectrum of a given percentage of width       
    """
    nb = int((xhi-xlo)/xpb)
    
    hist, bin_edges = np.histogram(energySeries, range=(xlo, xhi), bins=nb)
    bin_centers = pgh.get_bin_centers(bin_edges)

    # median filter along the spectrum, do this as a "baseline subtraction"                              
    #hist_med = medfilt(hist, 15)
    #hist = hist - hist_med
    # identify peaks with a scipy function (could be improved ...)                
 
    peak_idxs = find_peaks_cwt(hist, np.arange(1, max_cwt_width, 0.1), min_snr=2)
    peak_energies = bin_centers[peak_idxs]
    
    # pick the num_peaks most prominent peaks        
    if max_num_peaks < len(peak_energies):
        peak_vals = hist[peak_idxs]
        sort_idxs = np.argsort(peak_vals)
        peak_idxs_max = peak_idxs[sort_idxs[-max_num_peaks:]]
        peak_energies = np.sort(bin_centers[peak_idxs_max])

    if plot:
        plt.plot(bin_centers, hist, ls='steps', lw=1, c='b')
        for e in peak_energies:
            plt.axvline(e, color="r", lw=1, alpha=0.6)
        plt.xlabel("Energy [uncal]", ha='right', x=1)
        plt.ylabel("Filtered Spectrum", ha='right', y=1)
        plt.yscale('log')
        #plt.tight_layout()                                    
        #plt.show()    
        
    return peak_energies


def match_peaks(data_pks, cal_pks, plot=False):
    """
    Match uncalibrated peaks with literature energy values.
    """
    from itertools import combinations
    from scipy.stats import linregress

    n_pks = len(cal_pks) if len(cal_pks) < len(data_pks) else len(data_pks)

    cal_sets = combinations(range(len(cal_pks)), n_pks)
    data_sets = combinations(range(len(data_pks)), n_pks)

    best_err, best_m, best_b = np.inf, None, None
    for i,cal_set in enumerate(cal_sets):

        cal = cal_pks[list(cal_set)] # lit energies for this set

        for data_set in data_sets:

            data = data_pks[list(data_set)] # uncal energies for this set

            m, b, _, _, _ = linregress(data, y=cal)
            err = np.sum((cal - (m * data + b))**2)

            if err < best_err:
                best_err, best_m, best_b = err, m, b

    if test:
        print(i, best_err)
        print("cal:",cal)
        print("data:",data)
        plt.scatter(data, cal, label='min.err:{:.2e}'.format(err))
        xs = np.linspace(data[0], data[-1], 10)
        plt.plot(xs, best_m * xs + best_b , c="r",
                 label="y = {:.2f} x + {:.2f}".format(best_m,best_b) )
        plt.xlabel("Energy (uncal)", ha='right', x=1)
        plt.ylabel("Energy (keV)", ha='right', y=1)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    return best_m, best_b


def calibrate_tl208(energy_series, cal_peaks=None, xlo = 1000, xhi = 15000, xpb = 25, min_snr = 2, max_cwt_width = 20, plot=None):
    """
    energy_series: array of energies we want to calibrate
    cal_peaks: array of peaks to fit

    1.) we find the 2614 peak by looking for the tallest peak at >0.1 the max adc value
    2.) fit that peak to get a rough guess at a calibration to find other peaks with
    3.) fit each peak in peak_energies
    4.) do a linear fit to the peak centroids to find a calibration
    """

    if cal_peaks is None:
        cal_peaks = np.array(
            [238.632, 510.770, 583.191, 727.330, 860.564,
             2614.553])  #get_calibration_energies(peak_energies)
    else:
        cal_peaks = np.array(cal_peaks)

    if len(energy_series) < 100:
        return 1, 0

    #get 10 most prominent ~high e peaks
    max_adc = np.amax(energy_series)
    energy_hi = energy_series  #[ (energy_series > np.percentile(energy_series, 20)) & (energy_series < np.percentile(energy_series, 99.9))]

    peak_energies = get_most_prominent_peaks(energy_hi, xlo = xlo, xhi = xhi, xpb = xpb, min_snr = min_snr, max_cwt_width = max_cwt_width, max_num_peaks = len(cal_peaks),plot=plot)
    rough_kev_per_adc, rough_kev_offset = match_peaks(peak_energies, cal_peaks,plot=plot)
    e_cal_rough = rough_kev_per_adc * energy_series + rough_kev_offset

    # return rough_kev_per_adc, rough_kev_offset
    # print(energy_series)
    # plt.ion()
    # plt.figure()
    # # for peak in cal_peaks:
    # #     plt.axvline(peak, c="r", ls=":")
    # # energy_series.hist()
    # # for peak in peak_energies:
    # #      plt.axvline(peak, c="r", ls=":")
    # #
    # plt.hist(energy_series)
    # # plt.hist(e_cal_rough[e_cal_rough>100], bins=2700)
    # val = input("do i exist?")
    # exit()

    ###############################################
    #Do a real fit to every peak in peak_energies
    ###############################################
    max_adc = np.amax(energy_series)

    peak_num = len(cal_peaks)
    centers = np.zeros(peak_num)
    fit_result_map = {}
    bin_size = 0.2  #keV

    if plot is not None:
        plot_map = {}

    for i, energy in enumerate(cal_peaks):
        window_width = 10  #keV
        window_width_in_adc = (window_width) / rough_kev_per_adc
        energy_in_adc = (energy - rough_kev_offset) / rough_kev_per_adc
        bin_size_adc = (bin_size) / rough_kev_per_adc

        peak_vals = energy_series[
            (energy_series > energy_in_adc - window_width_in_adc) &
            (energy_series < energy_in_adc + window_width_in_adc)]

        peak_hist, bins = np.histogram(
            peak_vals,
            bins=np.arange(energy_in_adc - window_width_in_adc,
                           energy_in_adc + window_width_in_adc + bin_size_adc,
                           bin_size_adc))
        bin_centers = pgh.get_bin_centers(bins)
        # plt.ion()
        # plt.figure()
        # plt.plot(bin_centers,peak_hist,  color="k", ls="steps")

        # inpu = input("q to quit...")
        # if inpu == "q": exit()

        try:
            guess, bounds = pgh.get_gaussian_guess(peak_hist, bin_centers)
            #guess_e, guess_sigma, guess_area = *guess
        except IndexError:
            print("\n\nIt looks like there may not be a peak at {} keV".format(
                energy))
            print("Here is a plot of the area I'm searching for a peak...")
            plt.ion()
            plt.figure(figsize=(12, 6), facecolor='white')
            plt.subplot(121)
            plt.plot(bin_centers, peak_hist, color="k", ls="steps")
            plt.subplot(122)
            plt.hist(e_cal_rough, bins=2700, histtype="step")
            input("-->press any key to continue...")
            #sys.exit()
        if plot is not None:
            plt.plot(
                bin_centers,
                gauss(bin_centers, *guess),
                color="b")

        # inpu = input("q to quit...")
        # if inpu == "q": exit()

        bounds = ([0.9 * guess_e, 0.5 * guess_sigma, 0, 0, 0, 0, 0], [
            1.1 * guess_e, 2 * guess_sigma, 0.1, 0.75, window_width_in_adc, 10,
            5 * guess_area
        ])
        params = fit_binned(
            radford_peak,
            peak_hist,
            bin_centers,
            [guess_e, guess_sigma, 1E-3, 0.7, 5, 0, guess_area],
        )  #bounds=bounds)
        if plot is not None:
            plt.plot(bin_centers, radford_peak(bin_centers, *params), color="r")

        # inpu = input("q to quit...")
        # if inpu == "q": exit()

        fit_result_map[energy] = params
        centers[i] = params[0]

        if plot is not None:
            plot_map[energy] = (bin_centers, peak_hist)

    #Do a linear fit to find the calibration
    linear_cal = np.polyfit(centers, cal_peaks, deg=1)

    if plot is not None:

        plt.figure(figsize=(15, 6), facecolor='white')
        #plt.clf()

        grid = gs.GridSpec(3, 3)
        #ax_line = plt.subplot(grid[:, 1])
        #ax_spec = plt.subplot(grid[:, 2])
        for i, energy in enumerate(cal_peaks):
            ax_peak = plt.subplot(grid[i])
            bin_centers, peak_hist = plot_map[energy]
            params = fit_result_map[energy]
            ax_peak.plot(
                bin_centers * rough_kev_per_adc + rough_kev_offset,
                peak_hist,
                ls="steps-mid",
                color="r")
            fit = radford_peak(bin_centers, *params)
            ax_peak.plot(bin_centers * rough_kev_per_adc + rough_kev_offset,fit,color="b")

        ax_peak.set_xlabel("Energy [keV]")
        
        #ax_line.scatter(centers,cal_peaks,)
        #plt.scatter(centers,cal_peaks,)
        x = np.arange(0, max_adc, 1)
        #ax_line.plot(x, linear_cal[0] * x + linear_cal[1])
        #ax_line.set_xlabel("ADC")
        #ax_line.set_ylabel("Energy [keV]")
        
        plt.figure(figsize=(12, 6.75), facecolor='white')
        energies_cal = energy_series * linear_cal[0] + linear_cal[1]
        peak_hist, bins = np.histogram(energies_cal, bins=np.arange(0, 2700))
        plt.semilogy(pgh.get_bin_centers(bins), peak_hist, ls="steps-mid")
        plt.xlabel("Energy [keV]")

    return linear_cal


def get_calibration_energies(cal_type):
    if cal_type == "th228":
        return np.array([238, 277, 300, 452, 510.77, 583.191,
                         727, 763, 785, 860.564, 1620, 2614.533],
                        dtype="double")

    elif cal_type == "uwmjlab":
        # return np.array([239, 295, 351, 510, 583, 609, 911, 969, 1120,
        #                  1258, 1378, 1401, 1460, 1588, 1764, 2204, 2615],
        #                 dtype="double")
        return np.array([239, 911, 1460, 1764, 2615],
                        dtype="double")

    else:
        raise ValueError


def get_first_last_peaks(energySeries, detector_type, pulser = True, plot=False):
    """                                                                                                  
    find the most prominent peaks in a spectrum by looking for spikes in derivative of spectrum          
    energySeries: array of measured energies                                                             
    max_num_peaks = maximum number of most prominent peaks to find                                       
    return a histogram around the most prominent peak in a spectrum of a given percentage of width       
    """
    xlo = np.percentile(energySeries, 5)
    xhi = np.percentile(energySeries, 100)
    nb = 3000#int((xhi-xlo)/xpb)
    
    hist, bin_edges = np.histogram(energySeries, bins=np.linspace(xlo,xhi,nb))
    bin_centers = pgh.get_bin_centers(bin_edges)
    if pulser:
        xp = bin_centers[np.where(hist > hist.max()*0.01)][-1]
        hist = hist[np.where(bin_centers < xp-500)]
        bin_centers = bin_centers[np.where(bin_centers < xp-500)]
    # median filter along the spectrum, do this as a "baseline subtraction"                              
    #hist_med = medfilt(hist, 15)
    #hist = hist - hist_med
    # identify peaks with a scipy function (could be improved ...)
    
    #peak_idxs = find_peaks_cwt(hist, np.arange(1, max_cwt_width, 0.1), min_snr=min_snr)
    peak_idxs, _ = find_peaks(hist,height=hist.max()/30,distance=50)
    peak_energies = bin_centers[peak_idxs]
    
    # pick the num_peaks most prominent peaks        
    #if max_num_peaks < len(peak_energies):
    #    peak_vals = hist[peak_idxs]
    #    sort_idxs = np.argsort(peak_vals)
    #    peak_idxs_max = peak_idxs[sort_idxs[-max_num_peaks:]]
    #    peak_energies = np.sort(bin_centers[peak_idxs_max])
    peak_max = bin_centers[np.argmax(hist)]
    peak_last = peak_energies[-1]
    if (detector_type=="PPC" and peak_max > 1500):   
        pp = peak_idxs[peak_energies < 1500]
        hh = hist[np.where(bin_centers < 1500)]
        peak_max = bin_centers[np.argmax(hh)]
    if peak_last == peak_max:
        hh = hist[np.where(bin_centers < peak_last-13500)]
        bb = bin_centers[np.where(bin_centers < peak_last-13500)]
        peak_max = bb[np.argmax(hh)]
        if peak_max < 1000:
            hh = hist[np.where((bin_centers>1500) & (bin_centers < 1700))]
            bb = bin_centers[np.where((bin_centers>1500) & (bin_centers < 1700))]
            peak_max = bb[np.argmax(hh)]
    elif detector_type=="ICPC" and peak_max < 1000:   
        hh = hist[np.where((bin_centers>1500) & (bin_centers < 1700))]
        bb = bin_centers[np.where((bin_centers>1500) & (bin_centers < 1700))]
        peak_max = bb[np.argmax(hh)]
    elif peak_last <= peak_max + 4000:
        hh = hist[np.where(bin_centers > peak_max + 4000)]
        bb = bin_centers[np.where(bin_centers > peak_max + 4000)]
        peak_max = bb[np.argmax(hh)]
        
    if plot:
        plt.figure(figsize=(12, 6.75), facecolor='white')
        plt.plot(bin_centers, hist, ds='steps', lw=1, c='b')
        #for e in peak_energies:
        #    plt.axvline(e, color="r", lw=1, alpha=0.6)
        plt.axvline(peak_max, color='r', lw=1, alpha=0.6,label='max peak')
        plt.axvline(peak_last, color='g', lw=1, alpha=0.6,label='last peak')
        plt.xlabel("Energy [uncal]", ha='right', x=1)
        plt.ylabel("Filtered Spectrum", ha='right', y=1)
        plt.yscale('log')
        plt.legend()
    return peak_max, peak_last


cal_peaks_th228 = np.array([238.632, 510.770, 583.191, 727.330, 860.564, 1592.5, 2103.5, 2614.553])

def calibrate_th228(energy_series, detector_type="ICPC", cal_peaks=None, pulser = False, plot = False):
    if cal_peaks is None: cal_peaks = cal_peaks_th228
    else: cal_peaks = np.array(cal_peaks)
    energy_series = energy_series[~np.isnan(energy_series)]
    peak_max, peak_last = get_first_last_peaks(energy_series,detector_type,pulser=pulser,plot=plot)
    rough_kev_per_adc = (cal_peaks[0]-cal_peaks[-1])/(peak_max-peak_last)
    rough_kev_offset = cal_peaks[0] - rough_kev_per_adc * peak_max
    e_cal_rough = rough_kev_per_adc * energy_series + rough_kev_offset
    #fit peaks
    max_adc = np.amax(energy_series)
    bin_size = 0.2  #keV
    fit_result_map = {}
    if plot:
        plot_map = {}
    for i, energy in enumerate(cal_peaks):
        window_width = 10  #keV
        window_width_in_adc = (window_width) / rough_kev_per_adc
        energy_in_adc = (energy - rough_kev_offset) / rough_kev_per_adc
        bin_size_adc = (bin_size) / rough_kev_per_adc

        peak_vals = energy_series[
            (energy_series > energy_in_adc - window_width_in_adc) &
            (energy_series < energy_in_adc + window_width_in_adc)]
        
        peak_hist, bins = np.histogram(
            peak_vals,
            bins=np.arange(energy_in_adc - window_width_in_adc,
                           energy_in_adc + window_width_in_adc + bin_size_adc,
                           bin_size_adc))
            #bins=np.arange(energy_in_adc - window_width_in_adc,
            #               energy_in_adc + window_width_in_adc + bin_size_adc,
             #              bin_size_adc))
        bin_centers = pgh.get_bin_centers(bins)

        try:
            guess, b_ = au.get_gaussian_guess(peak_hist, bin_centers)
            guess = np.append(guess,peak_hist[0])
        except IndexError:
            print("\n\nIt looks like there may not be a peak at {} keV".format(
                energy))
            #print("Here is a plot of the area I'm searching for a peak...")
            #plt.ion()
            #plt.figure(figsize=(12, 6))
            #plt.subplot(121)
            #plt.plot(bin_centers, peak_hist, color="k", ls="steps")
            #plt.subplot(122)
            #plt.hist(e_cal_rough, bins=2700, histtype="step")
            #input("-->press any key to continue...")
        try:
            params, pcov = curve_fit(au.gauss_const, bin_centers, peak_hist, p0=guess)
            perr = np.sqrt(np.diag(pcov))
            chisq = []
            for j, h in enumerate(peak_hist):
                model = au.gauss_const(bin_centers[j], *params)
                diff = (model - h)**2 / model
                chisq.append(abs(diff))
            chi2 = sum(np.array(chisq) / len(peak_hist))
            if np.isnan(chi2) or chi2 > 50:
                print(energy,chi2)
                raise ValueError('Bad fit')
            fit_result_map[energy] = {}
            fit_result_map[energy]['parameters'] = params
            fit_result_map[energy]['centers'] = params[1]
            fit_result_map[energy]['fwhm'] = params[2]*2.355
            fit_result_map[energy]['fwhm_err'] = perr[2]*2.355
            fit_result_map[energy]['chi2'] = chi2
            if plot:
                plot_map[energy] = (bin_centers, peak_hist)
        except:
            print('Fit not performed for', energy)
    
    #Do a linear fit to find the calibration
    centers = [fit_result_map[key]['centers'] for key in fit_result_map.keys()]
    energies = [key for key in fit_result_map.keys()]
    linear_cal = np.polyfit(centers, energies, deg=1)
    for i, energy in enumerate(energies):
        fit_result_map[energy]['fwhm'] = fit_result_map[energy]['fwhm'] * linear_cal[0]
        fit_result_map[energy]['fwhm_err'] = fit_result_map[energy]['fwhm_err'] * linear_cal[0]
    fwhm = [fit_result_map[key]['fwhm'] for key in fit_result_map.keys()]
    fwhm_err = [fit_result_map[key]['fwhm_err'] for key in fit_result_map.keys()]
    chi2 = [fit_result_map[key]['chi2'] for key in fit_result_map.keys()]
    fit_result_map['calibration_curve'] = {}
    fit_result_map['calibration_curve']['m'] = linear_cal[0]
    fit_result_map['calibration_curve']['q'] = linear_cal[1]
    if plot:
        plt.figure(figsize=(15, 6.75), facecolor='white')
        #plt.clf()

        grid = gs.GridSpec(3, 3)
        #ax_line = plt.subplot(grid[:, 1])
        #ax_spec = plt.subplot(grid[:, 2])
        for i, energy in enumerate(energies):
            ax_peak = plt.subplot(grid[i])
            bin_centers, peak_hist = plot_map[energy]
            params = fit_result_map[energy]['parameters']
            ax_peak.plot(
                bin_centers * rough_kev_per_adc + rough_kev_offset,
                peak_hist,
                ds="steps-mid",
                color="r",label=f'peak at {energy} keV')
            #fit = radford_peak(bin_centers, *params)
            fit = au.gauss_const(bin_centers, *params)
            ax_peak.plot(bin_centers * rough_kev_per_adc + rough_kev_offset,fit,color="b",
                        label=f'FWHM = {fwhm[i]:.2f} keV')
            ax_peak.legend()

        ax_peak.set_xlabel("Energy [keV]")
        # plot calibrated spectrum
        x = np.arange(0, max_adc, 1)
        plt.figure(figsize=(12, 6.75), facecolor='white')
        energies_cal = energy_series * linear_cal[0] + linear_cal[1]
        peak_hist, bins = np.histogram(energies_cal, bins=np.arange(0, 2700))
        plt.semilogy(pgh.get_bin_centers(bins), peak_hist, ds="steps-mid")
        plt.xlabel("Energy [keV]", x=1,fontsize=14)
        
        #calibration
        curve = au.linear(np.array(centers), *linear_cal)
        plt.figure(figsize=(12,6.75), facecolor='white')
        plt.plot(centers, energies, 'r.',ms=8)
        plt.xlabel('uncalibrated energy', ha='right', x=1,fontsize=14)
        plt.ylabel('energy [keV]', ha='right', y=1,fontsize=14)
        plt.plot(centers, curve, label=f'calibration curve:\nE = ${linear_cal[1]:.2f}+{linear_cal[0]:.2f}\cdot ADC$',c='b')
        plt.legend(fontsize=14)
        
    # resolution curve
    res_peaks = np.array([cal_peaks[0], cal_peaks[2], cal_peaks[3], cal_peaks[4], cal_peaks[-1]])
    res_peaks = [a for a in res_peaks if a in energies]
    fwhm_fit = [a for i, a in enumerate(fwhm) if energies[i] in res_peaks]
    fwhm_efit = [a for i, a in enumerate(fwhm_err) if energies[i] in res_peaks]
    no_peaks = [a for a in energies if a not in res_peaks]
    fwhm_no = [a for i, a in enumerate(fwhm) if energies[i] not in res_peaks]
    fwhm_eno = [a for i, a in enumerate(fwhm_err) if energies[i] not in res_peaks]
    fit_result_map['res_peaks'] = res_peaks
    fit_result_map['res_fwhm'] = fwhm_fit
    fit_result_map['res_fwhm_err'] = fwhm_efit
    fit_result_map['peaks_not'] = no_peaks
    fit_result_map['fwhm_not'] = fwhm_no
    fit_result_map['fwhm_err_not'] = fwhm_eno
    res_par, res_cov = curve_fit(au.res_model, res_peaks, fwhm_fit, p0=(0.7,0.003),sigma=fwhm_efit)
    res_err = np.sqrt(np.diag(res_cov))
    energy_qbb = 2039
    fwhmqbb = np.sqrt(res_par[0] + res_par[1] * energy_qbb)
    fwhmqbb_err = 1/(2*fwhmqbb)*np.sqrt(res_err[0]**2 + energy_qbb * res_err[1]**2)
    fit_result_map['qbb'] = {}
    fit_result_map['qbb']['fwhm'] = fwhmqbb
    fit_result_map['qbb']['fwhm_err'] = fwhmqbb_err
    resolution = au.res_model(bins, *res_par)
    fit_result_map['resolution_curve'] = {}
    fit_result_map['resolution_curve']['a'] = res_par[0]
    fit_result_map['resolution_curve']['b'] = res_par[1]
    if plot:
        plt.figure(figsize=(12,6.75), facecolor='white')
        plt.errorbar(res_peaks, fwhm_fit, yerr = fwhm_efit, fmt='o',color='b',label='data')
        eb1=plt.errorbar(no_peaks, fwhm_no, yerr = fwhm_eno, fmt='o',color='k',label='peaks not used')
        eb1[-1][0].set_linestyle(':')
        plt.xlabel('energy [keV]', ha='right', x=1,fontsize=14)
        plt.ylabel('FWHM [keV]', ha='right', y=1,fontsize=14)
        plt.plot(bins, resolution,'r-', label='$FWHM = \sqrt{%.4f+%.4f \cdot E}$)' % (res_par[0],res_par[1]))
        plt.plot(energy_qbb,fwhmqbb,'gx',ms=10,label=f'$FWHM={fwhmqbb:.2f}\pm{fwhmqbb_err:.2f}$ keV')
        plt.legend(fontsize=14)
        
    return fit_result_map
