import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cycler, patches
from matplotlib.colors import LogNorm
import os, json
import lgdo.lh5_store as lh5
import matplotlib.pyplot as plt
import numpy as np
from pygama.lgdo import ls  
from legendmeta import LegendMetadata
import pandas as pd
import pygama.math.histogram as pgh
from tqdm.notebook import tqdm
import random
from scipy.optimize import curve_fit
from scipy import spatial
from pygama.dsp.processors import bl_subtract

IPython_default = plt.rcParams.copy()
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

figsize = (4.5, 3)

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title
plt.rcParams["font.family"] = "serif"

matplotlib.rcParams['mathtext.fontset'] = 'stix'
#matplotlib.rcParams['font.family'] = 'STIXGeneral'

marker_size = 2
line_width = 0.5
cap_size = 0.5
cap_thick = 0.5

colors = cycler('color', ['b', 'g', 'r', 'm', 'y', 'k', 'c', '#8c564b'])
plt.rc('axes', facecolor='white', edgecolor='black',
       axisbelow=True, grid=True, prop_cycle=colors)

def plot_event(sto, files, channels, idx, is_valid):
    
    wfs=[]
    bls=[]
    energies=[]
    
    events_per_file = [sto.read_n_rows("ch1027201/raw/baseline", raw_file)for raw_file in files]
    cum_events = np.cumsum(events_per_file)
    for i,n in enumerate(cum_events):
        if idx<n:
            file_n = i
            if file_n >0:
                start_r = idx-cum_events[i-1]
            else:
                start_r = idx
            break
    
    
    wfs0 = sto.read_object(f'ch1027201/raw/waveform/values', files[file_n],start_row = start_r, n_rows=1 )[0].nda
    bls0 = sto.read_object(f'ch1027201/raw/baseline', files[file_n],start_row = start_r, n_rows=1 )[0].nda
    energy0 = sto.read_object(f'ch1027201/dsp/wf_max', files[file_n],start_row = start_r, n_rows=1  )[0].nda
    
    wfs.append(wfs0)
    bls.append(bls0)
    energies.append(energy0)
    
    for channel in channels:
        wfsc = sto.read_object(f'{channel}/raw/waveform/values', files[file_n], start_row = start_r, n_rows=1)[0].nda
        blsc = sto.read_object(f'{channel}/raw/baseline', files[file_n],start_row = start_r, n_rows=1)[0].nda
        energyc = sto.read_object(f'{channel}/dsp/trapTmax', files[file_n],start_row = start_r, n_rows=1)[0].nda
        
        wfs.append(wfsc)
        bls.append(blsc)
        energies.append(energyc)
    
    xs = np.arange(0,len(wfs[0][0]),1)
    plt.figure()
    
    energies[0][0] = energies[0][0]-bls[0][0]
    
    scale = np.nanmax(energies)

    wfs0_blsub = np.zeros(len(wfs[0][0]))
    bl_subtract(wfs[0][0], bls[0][0], wfs0_blsub)
    plt.step(xs*16 , wfs0_blsub/(scale), label='Pulser')
    #plt.xlim([45000,60000])
    #plt.ylim([-0.25,1.1])
    plt.xlabel('Time (ns)')
    plt.ylabel('Value (ADC)')
    plt.legend(ncol=1, loc='upper left')
    plt.show()

    for j in range(len(wfs[1:])):
        plt.figure()
        wf_blsub = np.zeros(len(wfs[j+1][0]))
        bl_subtract(wfs[j+1][0], bls[j+1][0], wf_blsub)
        if is_valid[j]:
            color = 'b'
        else:
            color = 'r'
        plt.plot(xs*16 , wf_blsub, color, label=f'{chmap.map("daq.rawid")[int(ch[2:])]["name"]}')#
        plt.text(0,0,f'{chmap.map("daq.rawid")[int(channels[j][2:])]["name"]}')


        #plt.xlim([45000,60000])
        #plt.ylim([-0.25,1.1])
        plt.xlabel('Time (ns)')
        plt.ylabel('Value (ADC)')
        plt.legend(ncol=1, loc='upper left')
        plt.show()

def plot_waveform(sto, files_obj, chmap, idx, channel, all_channels, ret_wav=False, plot=False, do_o_ch=True):
    
    events_per_file = [sto.read_n_rows("ch1027201/raw/baseline", raw_file)for raw_file in files_obj.raw]
    cum_events = np.cumsum(events_per_file)
    for i,n in enumerate(cum_events):
        if idx<n:
            file_n = i
            if file_n >0:
                start_r = idx-cum_events[i-1]
            else:
                start_r = idx
            break
    
    wfsc = sto.read_object(f'{channel}/raw/waveform/values', files_obj.raw[file_n], start_row = start_r, n_rows=1)[0].nda
    blsc = sto.read_object(f'{channel}/raw/baseline', files_obj.raw[file_n],start_row = start_r, n_rows=1)[0].nda
    energyc = sto.read_object(f'{channel}/dsp/trapTmax', files_obj.dsp[file_n],start_row = start_r, n_rows=1)[0].nda

    # is_valid_event = is_valid_0vbb & (~is_negative_energy) & is_valid_tail & is_valid_bl & (~is_noise_burst) & (~is_saturated)
    is_valid_0vbb = sto.read_object(f'{channel}/hit/is_valid_0vbb', files_obj.hit[file_n],start_row = start_r, n_rows=1)[0].nda
    is_negative_energy = sto.read_object(f'{channel}/hit/is_neg_energy', files_obj.hit[file_n],start_row = start_r, n_rows=1)[0].nda
    is_valid_tail = sto.read_object(f'{channel}/hit/is_valid_tail', files_obj.hit[file_n],start_row = start_r, n_rows=1)[0].nda
    is_valid_bl = sto.read_object(f'{channel}/hit/is_valid_baseline', files_obj.hit[file_n],start_row = start_r, n_rows=1)[0].nda
    is_noise_burst = sto.read_object(f'{channel}/hit/is_noise_burst', files_obj.hit[file_n],start_row = start_r, n_rows=1)[0].nda
    is_saturated = sto.read_object(f'{channel}/hit/is_saturated', files_obj.hit[file_n],start_row = start_r, n_rows=1)[0].nda
    
    is_valid_event = is_valid_0vbb & (~is_negative_energy) & is_valid_tail & is_valid_bl & (~is_noise_burst) & (~is_saturated)

    is_pulser = sto.read_object(f'ch1027201/dsp/trapTmax', files_obj.dsp[file_n],start_row = start_r, n_rows=1)[0].nda > 100
    other_dets = {}
    if not is_pulser and do_o_ch:
        for ichannel in all_channels:
            temp_is_valid_0vbb = sto.read_object(f'{ichannel}/hit/is_valid_0vbb', files_obj.hit[file_n],start_row = start_r, n_rows=1)[0].nda
            temp_is_negative_energy = sto.read_object(f'{ichannel}/hit/is_neg_energy', files_obj.hit[file_n],start_row = start_r, n_rows=1)[0].nda
            temp_is_valid_tail = sto.read_object(f'{ichannel}/hit/is_valid_tail', files_obj.hit[file_n],start_row = start_r, n_rows=1)[0].nda
            temp_is_valid_bl = sto.read_object(f'{ichannel}/hit/is_valid_baseline', files_obj.hit[file_n],start_row = start_r, n_rows=1)[0].nda
            temp_is_noise_burst = sto.read_object(f'{ichannel}/hit/is_noise_burst', files_obj.hit[file_n],start_row = start_r, n_rows=1)[0].nda
            temp_is_saturated = sto.read_object(f'{ichannel}/hit/is_saturated', files_obj.hit[file_n],start_row = start_r, n_rows=1)[0].nda
        
            temp_is_valid_event = temp_is_valid_0vbb & (~temp_is_negative_energy) & temp_is_valid_tail & temp_is_valid_bl & (~temp_is_noise_burst) & (~temp_is_saturated)

            temp_wfsc = sto.read_object(f'{ichannel}/raw/waveform/values', files_obj.raw[file_n], start_row = start_r, n_rows=1)[0].nda
            temp_blsc = sto.read_object(f'{ichannel}/raw/baseline', files_obj.raw[file_n],start_row = start_r, n_rows=1)[0].nda
            temp_energyc = sto.read_object(f'{ichannel}/hit/cuspEmax_ctc_cal', files_obj.hit[file_n],start_row = start_r, n_rows=1)[0].nda

            temp_xs = np.arange(0,len(temp_wfsc[0]),1)
            temp_wf_blsub = np.zeros(len(temp_wfsc[0]))
            bl_subtract(temp_wfsc[0], temp_blsc[0], temp_wf_blsub)

            if temp_is_valid_event and temp_energyc>25:
                other_dets[ichannel] = temp_wf_blsub

    xs = np.arange(0,len(wfsc[0]),1)
    wf_blsub = np.zeros(len(wfsc[0]))
    bl_subtract(wfsc[0], blsc[0], wf_blsub)

    if plot:
        plt.figure()
        plt.title(f'EVT: {idx} CH:{chmap.map("daq.rawid")[int(channel[2:])]["name"]}')
        
        plt.plot(xs*16 , wf_blsub, label=f'{chmap.map("daq.rawid")[int(channel[2:])]["name"]}')#
        plt.xlabel('Time (ns)')
        plt.ylabel('Value (ADC)')
        # plt.legend(ncol=1, loc='upper left')
        plt.show()
    
    if ret_wav:
        return wf_blsub, is_valid_event, other_dets, is_pulser

def plot_event_no_scale(sto, files, chmap, channels, idx, is_valid):
    
    wfs=[]
    bls=[]
    energies=[]
    
    events_per_file = [sto.read_n_rows("ch1027201/raw/baseline", raw_file)for raw_file in files]
    cum_events = np.cumsum(events_per_file)
    for i,n in enumerate(cum_events):
        if idx<n:
            file_n = i
            if file_n >0:
                start_r = idx-cum_events[i-1]
            else:
                start_r = idx
            break
    
    
    wfs0 = sto.read_object(f'ch1027201/raw/waveform/values', files[file_n],start_row = start_r, n_rows=1 )[0].nda
    bls0 = sto.read_object(f'ch1027201/raw/baseline', files[file_n],start_row = start_r, n_rows=1 )[0].nda
    energy0 = sto.read_object(f'ch1027201/dsp/wf_max', files[file_n],start_row = start_r, n_rows=1  )[0].nda
    
    wfs.append(wfs0)
    bls.append(bls0)
    energies.append(energy0)
    
    for channel in channels:
        wfsc = sto.read_object(f'{channel}/raw/waveform/values', files[file_n], start_row = start_r, n_rows=1)[0].nda
        blsc = sto.read_object(f'{channel}/raw/baseline', files[file_n],start_row = start_r, n_rows=1)[0].nda
        energyc = sto.read_object(f'{channel}/dsp/trapTmax', files[file_n],start_row = start_r, n_rows=1)[0].nda
        
        wfs.append(wfsc)
        bls.append(blsc)
        energies.append(energyc)
    
    xs = np.arange(0,len(wfs[0][0]),1)
    fig, ax = plt.subplots(nrows=len(channels)+1, ncols=1, sharex=True)
    
    energies[0][0] = energies[0][0]-bls[0][0]

    wfs0_blsub = np.zeros(len(wfs[0][0]))
    bl_subtract(wfs[0][0], bls[0][0], wfs0_blsub)
    ax[0].plot(xs*16 , wfs0_blsub, label='ch1027201')
    ax[0].text(0,0,f'Pulser')

    for j in range(len(wfs[1:])):
        wf_blsub = np.zeros(len(wfs[j+1][0]))
        bl_subtract(wfs[j+1][0], bls[j+1][0], wf_blsub)
        if is_valid[j]:
            color = 'b'
        else:
            color = 'r'
        ax[j+1].plot(xs*16 , wf_blsub, color, label=f'{chmap.map("daq.rawid")[int(channel[2:])]["name"]}')#
        ax[j+1].text(0,0,f'{chmap.map("daq.rawid")[int(channels[j][2:])]["name"]}')
        '''if np.nanmax(np.abs(wf_blsub))>500:
            plt.annotate(xy=(xs[-1]*16,wf_blsub[-1]), xytext=(5,0), textcoords='offset points', text=f'{channels[j]}', va='center')

        '''
        # ax[j+1].set_ylabel('Value (ADC)')
    #plt.xlim([45000,60000])
    #plt.ylim([-0.25,1.1])
    plt.xlabel('Time (ns)')
    
    plt.show()

def plot_event_raw(sto, files, channels, idx):
    
    wfs=[]
    bls=[]
    energies=[]
    
    events_per_file = [sto.read_n_rows("ch1027201/raw/baseline", raw_file)for raw_file in files]
    cum_events = np.cumsum(events_per_file)
    for i,n in enumerate(cum_events):
        if idx<n:
            file_n = i
            if file_n >0:
                start_r = idx-cum_events[i-1]
            else:
                start_r = idx
            break
    
    
    wfs0 = sto.read_object(f'ch1027201/raw/waveform/values', files[file_n],start_row = start_r, n_rows=1 )[0].nda
    bls0 = sto.read_object(f'ch1027201/raw/baseline', files[file_n],start_row = start_r, n_rows=1 )[0].nda
    energy0 = sto.read_object(f'ch1027201/dsp/wf_max', files[file_n],start_row = start_r, n_rows=1  )[0].nda
    
    wfs.append(wfs0)
    bls.append(bls0)
    energies.append(energy0)
    
    for channel in channels:
        wfsc = sto.read_object(f'{channel}/raw/waveform/values', files[file_n], start_row = start_r, n_rows=1)[0].nda
        blsc = sto.read_object(f'{channel}/raw/baseline', files[file_n],start_row = start_r, n_rows=1)[0].nda
        energyc = sto.read_object(f'{channel}/dsp/trapTmax', files[file_n],start_row = start_r, n_rows=1)[0].nda
        
        wfs.append(wfsc)
        bls.append(blsc)
        energies.append(energyc)
    
    xs = np.arange(0,len(wfs[0][0]),1)
    plt.figure()
    
    energies[0][0] = energies[0][0]-bls[0][0]

    wfs0_blsub = np.zeros(len(wfs[0][0]))
    # bl_subtract(wfs[0][0], bls[0][0], wfs0_blsub)
    plt.step(xs*16 , wfs[0][0], label='ch1027201')

    for j in range(len(wfs[1:])):
        wf_blsub = np.zeros(len(wfs[j+1][0]))
        # bl_subtract(wfs[j+1][0], bls[j+1][0], wf_blsub)
        # plt.step(xs*16 , wf_blsub, label=f'{channels[j]}')#
        plt.step(xs*16 , wfs[j+1][0], label=f'{channels[j]}')
        if np.nanmax(np.abs(wf_blsub))>500:
            plt.annotate(xy=(xs[-1]*16,wfs[j+1][0]), xytext=(5,0), textcoords='offset points', text=f'{channels[j]}', va='center')

    plt.legend()
    #plt.xlim([45000,60000])
    #plt.ylim([-0.25,1.1])
    plt.xlabel('Time (ns)')
    plt.ylabel('Value (ADC)')
    plt.show()

def gaus(x, mu, sig, n):
    return n * 1/(np.sqrt(2*np.pi)*sig) * np.exp(-0.5*((x-mu)/sig)**2)

def plot_baseline(df, ch, chmap):

    df_ch = df[(df.ch == ch) & (df.is_pulser == False) & (df.cusp_E > 30)]
    df_ch_qc = df_ch[df_ch.is_valid == True]
    fig = plt.figure(figsize=(9,6))
    
    bin_width = 1
    baselines = df_ch.baseline
    hh,bb,vv = pgh.get_hist(baselines, bins=int(abs(min(baselines) - max(baselines))/4), range=(min(baselines), max(baselines)))
    pgh.plot_hist(hh, bb, label=f'{chmap.map("daq.rawid")[int(ch[2:])]["name"]}')

    baselines = df_ch_qc.baseline
    hh,bb,vv = pgh.get_hist(baselines, bins=int(abs(min(baselines) - max(baselines))/4), range=(min(baselines), max(baselines)))
    pgh.plot_hist(hh, bb, label=f'{chmap.map("daq.rawid")[int(ch[2:])]["name"]} QC')
    
    popt, pcov = curve_fit(gaus, bb[:-1], hh, p0 = [np.average(baselines), np.std(baselines), len(baselines)])
    plt.plot(bb[:-1], gaus(bb[:-1], *popt))

    plt.title('p03 phy, string 2')
    plt.xlabel('baseline /ADC')
    # plt.xlim(-50,3000)
    plt.ylabel('counts')
    # plt.yscale('log')
    plt.legend(loc = 'upper right')
    plt.tight_layout()
    #plt.savefig(f'string2_phy_p03_baselines_{ch}.pdf')


def plot_base_bl(df, ch):
    
    df_ch = df[(df.ch == ch) & (df.is_pulser == False) & (df.cusp_E > 30)]
    df_ch_qc = df_ch[df_ch.is_valid == True]
    fig = plt.figure(figsize=(9,6))

    baselines = df_ch.baseline - df_ch.bl_mean
    hh,bb,vv = pgh.get_hist(baselines, bins=int(abs(min(baselines) - max(baselines))/2), range=(min(baselines), max(baselines)))
    pgh.plot_hist(hh, bb, label=f"{ch}")

    baselines = df_ch_qc.baseline - df_ch_qc.bl_mean
    hh,bb,vv = pgh.get_hist(baselines, bins=int(abs(min(baselines) - max(baselines))/2), range=(min(baselines), max(baselines)))
    pgh.plot_hist(hh, bb, label=f"{ch} QC")

    plt.title('p03 phy, string 2')
    plt.xlabel('baseline - bl_mean /ADC')
    # plt.xlim(-50,3000)
    plt.ylabel('counts')
    plt.yscale('log')
    plt.legend(loc = 'upper right')
    plt.tight_layout()
    # plt.savefig(f'string2_phy_p03_baselines_bl_{ch}.pdf')

def plot_baseline_std(df, ch):
    fig = plt.figure(figsize=(9,6))

    df_ch = df[(df.ch == ch) & (df.is_pulser == False) & (df.cusp_E > 30)]
    df_ch_qc = df_ch[df_ch.is_valid == True]

    baselines = df_ch.bl_std
    hh,bb,vv = pgh.get_hist(baselines, bins=int(abs(min(baselines) - max(baselines))/2), range=(min(baselines), max(baselines)))
    pgh.plot_hist(hh, bb, label=f"{ch}")

    baselines = df_ch_qc.bl_std
    hh,bb,vv = pgh.get_hist(baselines, bins=int(abs(min(baselines) - max(baselines))/2), range=(min(baselines), max(baselines)))
    pgh.plot_hist(hh, bb, label=f"{ch} QC")

    plt.title('bl_std')
    plt.xlabel('baseline /ADC')
    # plt.xlim(-50,3000)
    plt.ylabel('counts')
    plt.yscale('log')
    plt.legend(loc = 'upper right')

def plot_discharge(df_dict, chmap, channels, string):
    ts = []
    for channel in channels:
        temp_df = df_dict[channel]
        temp_df = temp_df[(temp_df.t_sat_lo.notna()) & (temp_df.t_sat_lo>0)]
        ts.append(temp_df.timestamp.values.tolist())
    t_min = df_dict[channels[0]].timestamp.values[0]

    fig, ax = plt.subplots(nrows=len(channels), figsize=(9,4))   
    plt.subplots_adjust(hspace=0)
    
    t_min = ts[0][0]

    ax[0].text(805, 2.1, 'Total')
    for i, ch in enumerate(channels):
        for t in ts[i]:
            ax[len(channels) - 1 - i].axvline((t-t_min)/3600, 0, 1, color='k', alpha=0.2)
        ax[len(channels) - 1 - i].set_ylim(0,2)
        ax[len(channels) - 1 - i].set_xlim(0,800)
        ax[len(channels) - 1 - i].set_yticks([1], labels=[f'{chmap.map("daq.rawid")[int(ch[2:])]["name"]}'], rotation = 0)
        ax[len(channels) - 1 - i].grid(False)

        ax[len(channels) - 1 - i].text(805, 0.75, f'{len(ts[i])}')
        if i != 0: ax[len(channels) - 1 - i].set_xticks([])
    

    plt.xlabel('run time /hr')
    fig.suptitle(f'string {string} discharge rate')
    # plt.tight_layout()
    plt.savefig(f'sting_{string}_discharge_rate.png')
    plt.show()


def plot_discharge_2D(df_dist, chmap, channels, string, ob=False):
    ts = []
    for channel in channels:
        temp_df = df_dist[channel]
        temp_df = temp_df[(temp_df.t_sat_lo.notna()) & (temp_df.t_sat_lo>0)]
        ts.append(temp_df.timestamp.values.tolist())
    fig, ax = plt.subplots(figsize=(9,4))

    t_min = df_dist[channels[0]].timestamp.values[0]
    n_bins = 100
    x = np.linspace(t_min/3600, t_min/3600 + 800, n_bins)
    y = np.linspace(0, len(channels), len(channels))
    X,Y = np.meshgrid(x,y)
    zs = []

    ax.axhline(0-0.5, ls='-', color='k', linewidth=1)
    ax.text(n_bins+1, len(channels)-0.5, 'Av')

    for i, ch in enumerate(channels):
        
        hh,bb,vv = pgh.get_hist(ts[i], bins=n_bins, range=(t_min, t_min + 800*3600))
        # plt.plot((bb - bb[0])[:-1]/(3600), hh, label=f'{chmap.map("daq.rawid")[int(ch[2:])]["name"]}')
        zs.append(np.array(hh))
        ax.text(n_bins+1, i-0.05, '{:.2f}'.format(np.average(hh)))
        ax.axhline(i+0.5, ls='-', color='k', linewidth=1)

        # plt.gca().add_patch(patches.Rectangle((n_bins, i), 5, 1, linewidth=1, edgecolor='b', facecolor='none'))

    zs = np.array(zs)
    Z = zs.reshape(X.shape)
    if ob:
        cm = 'Paired'
    else:
        cm = 'Reds'
    im = ax.imshow(Z, aspect='auto', origin='lower', cmap=cm, vmin=0, vmax=65)
    ax.set_xlabel('run time /hr')
    ax.set_xticks([i*12.5 for i in range(9)], labels=[i*12.5*8 for i in range(9)])
    ax.set_yticks([i for i in range(len(channels))], labels=[f'{chmap.map("daq.rawid")[int(ch[2:])]["name"]}' for ch in channels])
    
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.5)
    cbar.set_label('counts per 8hrs')
    ax.set_title(f'string {string} discharge rate')

    plt.tight_layout()
    plt.grid()
    plt.savefig(f'sting_{string}_discharge_rate_2d.png')
    plt.show()


def plot_is_valids_2d(XY, plot_dict, channels, n_bins, event_flag, string, chmap, cm):
    fig, ax = plt.subplots(nrows=3, figsize=(9,12))
    cms = ['Blues', 'Reds', 'Greens']
    for idx,typ in enumerate(['pulser', 'baseline', 'other']):
        ax[idx].axhline(0-0.5, ls='-', color='k', linewidth=1)
        ax[idx].text(n_bins+1, len(channels)-0.5, 'Av')
        Z = np.array(plot_dict[typ]['zs']).reshape(XY.shape)
        for i,channel in enumerate(channels):
            ax[idx].text(n_bins+1, i-0.05, '{:.2f}'.format(plot_dict[typ]['av'][i]))
            ax[idx].axhline(i+0.5, ls='-', color='k', linewidth=1)

        im = ax[idx].imshow(Z, aspect='auto', origin='lower', cmap=cms[idx], vmin=0, vmax=100)
        cbar = fig.colorbar(im, ax=ax[idx], shrink=0.5)
        cbar.set_label('percentage per 8hrs')

        if idx == 2:
            ax[idx].set_xticks([i*100 for i in range(9)], labels=[i*100 for i in range(9)])
            ax[idx].set_xlabel('run time /hr')
        ax[idx].set_yticks([i for i in range(len(channels))], labels=[f'{chmap.map("daq.rawid")[int(ch[2:])]["name"]}' for ch in channels])
        ax[idx].set_title(f'{typ}')
        ax[idx].set_xticks([i*100 for i in range(9)], labels=[i*100 for i in range(9)])
    
    ax[-1].set_xlabel('run time /hr')
    
    fig.suptitle(f'string {string} {event_flag} rate', horizontalalignment='left')
    plt.tight_layout()
    plt.grid()
    plt.savefig(f'sting_{string}_{event_flag}_rate_2d.png')
    plt.close(fig)


def plot_is_valids(files_obj, event_flag, string, channels, chmap, cm, baseline_data, pulser_data):
    
    n_bins = 800

    dict_data = {}
    plot_dict = {
        'pulser': {'zs':[], 'av':[]},
        'baseline': {'zs':[], 'av':[]},
        'other': {'zs':[], 'av':[]}
    }

    is_pulser     = pulser_data['trapTmax']>100
    is_baseline   = baseline_data['trapTmax']>100

    for i in tqdm(range(len(channels))):
        channel = channels[i]
        hit_data = lh5.load_dfs(files_obj.hit, ['timestamp', event_flag, 'cuspEmax_ctc_cal'], f"{channel}/hit")
        hit_data['is_pulser']   = is_pulser
        hit_data['is_baseline'] = is_baseline

        t_min = hit_data.timestamp.values[0]
        # Cut on energy
        hit_data = hit_data[hit_data.cuspEmax_ctc_cal > 1000]
        dict_data[channel] = hit_data

        is_pass = hit_data[(hit_data[event_flag].notna()) & (hit_data[event_flag]==True)]
        is_fail = hit_data[(hit_data[event_flag].notna()) & (hit_data[event_flag]==False)]
        is_naan = hit_data[(hit_data[event_flag].notna())]
        tot = len(is_pass) + len(is_fail) + len(is_naan)

        # Pulser events
        hh_0,bb_0,vv_0 = pgh.get_hist(is_pass[is_pass.is_pulser==True].timestamp.values.tolist(), bins=n_bins, range=(t_min, t_min + 800*3600))
        hh_1,bb_1,vv_1 = pgh.get_hist(is_fail[is_fail.is_pulser==True].timestamp.values.tolist(), bins=n_bins, range=(t_min, t_min + 800*3600))
        hh_2,bb_2,vv_2 = pgh.get_hist(is_naan[is_naan.is_pulser==True].timestamp.values.tolist(), bins=n_bins, range=(t_min, t_min + 800*3600))
        counts = [sum([hh_0[k], hh_1[k], hh_2[k]]) for k in range(len(hh_0))]
        plot_dict['pulser']['zs'].append(np.array([hh_1[k]/counts[k]*100 if counts[k]>0 else 0 for k in range(len(hh_1))]))
        plot_dict['pulser']['av'].append(sum(hh_1)/tot*100)

        # Baseline events
        hh_0,bb_0,vv_0 = pgh.get_hist(is_pass[(is_pass.is_pulser==False) & (is_pass.is_baseline==True)].timestamp.values.tolist(), bins=n_bins, range=(t_min, t_min + 800*3600))
        hh_1,bb_1,vv_1 = pgh.get_hist(is_fail[(is_fail.is_pulser==False) & (is_fail.is_baseline==True)].timestamp.values.tolist(), bins=n_bins, range=(t_min, t_min + 800*3600))
        hh_2,bb_2,vv_2 = pgh.get_hist(is_naan[(is_naan.is_pulser==False) & (is_naan.is_baseline==True)].timestamp.values.tolist(), bins=n_bins, range=(t_min, t_min + 800*3600))
        counts = [sum([hh_0[k], hh_1[k], hh_2[k]]) for k in range(len(hh_0))]
        plot_dict['baseline']['zs'].append(np.array([hh_1[k]/counts[k]*100 if counts[k]>0 else 0 for k in range(len(hh_1))]))
        plot_dict['baseline']['av'].append(sum(hh_1)/tot*100)

        # Other events
        hh_0,bb_0,vv_0 = pgh.get_hist(is_pass[(is_pass.is_pulser==False) & (is_pass.is_baseline==False)].timestamp.values.tolist(), bins=n_bins, range=(t_min, t_min + 800*3600))
        hh_1,bb_1,vv_1 = pgh.get_hist(is_fail[(is_fail.is_pulser==False) & (is_fail.is_baseline==False)].timestamp.values.tolist(), bins=n_bins, range=(t_min, t_min + 800*3600))
        hh_2,bb_2,vv_2 = pgh.get_hist(is_naan[(is_naan.is_pulser==False) & (is_naan.is_baseline==False)].timestamp.values.tolist(), bins=n_bins, range=(t_min, t_min + 800*3600))
        counts = [sum([hh_0[k], hh_1[k], hh_2[k]]) for k in range(len(hh_0))]
        plot_dict['other']['zs'].append(np.array([hh_1[k]/counts[k]*100 if counts[k]>0 else 0 for k in range(len(hh_1))]))
        plot_dict['other']['av'].append(sum(hh_1)/tot*100)

        #ax.text(n_bins+1, i-0.05, '{:.2f}'.format(sum(hh_1)/tot*100))
        #ax.axhline(i+0.5, ls='-', color='k', linewidth=1)

    x = np.linspace(t_min/3600, t_min/3600 + 800, n_bins)
    y = np.linspace(0, len(channels), len(channels))
    X,Y = np.meshgrid(x,y)
    plot_is_valids_2d(X, plot_dict, channels, n_bins, event_flag, string, chmap, cm) 


def plot_event_flags(files_obj, event_flag, string, channels, chmap, cm):

    fig, ax = plt.subplots(figsize=(9,4))
    n_bins = 800

    ax.axhline(0-0.5, ls='-', color='k', linewidth=1)
    ax.text(n_bins+1, len(channels)-0.5, 'Av')

    dict_data = {}
    ts = []
    zs = []

    for i in tqdm(range(len(channels))):
        channel = channels[i]
        hit_data = lh5.load_dfs(files_obj.hit, ['timestamp', event_flag, 'cuspEmax_ctc_cal'], f"{channel}/hit")
        t_min = hit_data.timestamp.values[0]
        hit_data = hit_data[hit_data.cuspEmax_ctc_cal > 25]
        dict_data[channel] = hit_data
        ts.append(hit_data[(hit_data[event_flag].notna()) & (hit_data[event_flag]==True)].timestamp.values.tolist())

    x = np.linspace(t_min/3600, t_min/3600 + 800, n_bins)
    y = np.linspace(0, len(channels), len(channels))
    X,Y = np.meshgrid(x,y)

    # plot 2D
    for i, ch in enumerate(channels):
            
        hh,bb,vv = pgh.get_hist(ts[i], bins=n_bins, range=(t_min, t_min + 800*3600))
        # plt.plot((bb - bb[0])[:-1]/(3600), hh, label=f'{chmap.map("daq.rawid")[int(ch[2:])]["name"]}')
        zs.append(np.array(hh))
        ax.text(n_bins+1, i-0.05, '{:.2f}'.format(np.average(hh)))
        ax.axhline(i+0.5, ls='-', color='k', linewidth=1)

        # plt.gca().add_patch(patches.Rectangle((n_bins, i), 5, 1, linewidth=1, edgecolor='b', facecolor='none'))

    zs = np.array(zs)
    Z = zs.reshape(X.shape)

    im = ax.imshow(Z, aspect='auto', origin='lower', cmap=cm)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.5)
    cbar.set_label('counts per 8hrs')

    ax.set_xlabel('run time /hr')
    ax.set_xticks([i*12.5 for i in range(9)], labels=[i*12.5*8 for i in range(9)])
    ax.set_yticks([i for i in range(len(channels))], labels=[f'{chmap.map("daq.rawid")[int(ch[2:])]["name"]}' for ch in channels])
    ax.set_title(f'string {string} {event_flag} rate')

    plt.tight_layout()
    plt.grid()
    plt.savefig(f'sting_{string}_{event_flag}_rate_2d.png')
    plt.close(fig)

    '''# plot 1D
    fig, ax = plt.subplots(nrows=len(channels), figsize=(9,4))   
    plt.subplots_adjust(hspace=0)

    ax[0].text(805, 2.1, 'Total')
    for i, ch in enumerate(channels):
        for t in ts[i]:
            ax[len(channels) - 1 - i].axvline((t-t_min)/3600, 0, 1, color='k', alpha=0.2)
        ax[len(channels) - 1 - i].set_ylim(0,2)
        ax[len(channels) - 1 - i].set_xlim(0,800)
        ax[len(channels) - 1 - i].set_yticks([1], labels=[f'{chmap.map("daq.rawid")[int(ch[2:])]["name"]}'], rotation = 0)
        ax[len(channels) - 1 - i].grid(False)

        ax[len(channels) - 1 - i].text(805, 0.75, f'{len(ts[i])}')
        if i != 0: ax[len(channels) - 1 - i].set_xticks([])
        

    plt.xlabel('run time /hr')
    fig.suptitle(f'string {string} {event_flag} rate')
    # plt.tight_layout()
    plt.savefig(f'sting_{string}_{event_flag}_rate.png')
    plt.close(fig)'''

    fig = plt.figure()
    for i in zs:
        plt.plot(i)
    plt.show()

class LoadData():
    def __init__(self, phase, n_runs, path='/data2/public/prodenv/prod-blind/tmp/auto', dtype='phy') -> None:
        hit_files = []
        for r in range(n_runs):
            try:
                r_file_path = f"{path}/generated/tier/hit/{dtype}/{phase}/r00{r}/"
                hit_files += sorted([os.path.join(r_file_path, file) for file in os.listdir(r_file_path)])
            except: 
                continue
        raw_files = [
            file.replace("hit", "raw").replace(r_file_path.split("/generated")[0],
                                            "/data2/public/prodenv/prod-orig/archive/raw-v01.00") for file in hit_files
            ]
        dsp_files = [file.replace("hit", "dsp") for file in hit_files]

        self.hit = hit_files
        self.dsp = dsp_files
        self.raw = raw_files


def build_string_array(chan_map):
    dets = []
    strings = []
    positions = []
    for key,entry in chan_map.items():
        if entry.system == "geds":
            string = entry.location.string
            pos = entry.location.position
            dets.append(key)
            strings.append(string)
            positions.append(int(pos))
    return dets, strings, positions


def build_string_map(chan_map, data):
    dets, strings, positions = build_string_array(chan_map)
    string_nos = np.array(sorted(np.unique(strings)))
    pos_nos = np.array(sorted(np.unique(positions)))
    n_strings = len(string_nos)
    max_pos = np.max(positions)
    data_array = np.full((max_pos *2+1, n_strings*2+1), np.nan)
    annot_array = np.empty((max_pos *2+1, n_strings*2+1), dtype="object")
    for i,det in enumerate(dets):
        index = (2*positions[i]-1, 2*(np.where(strings[i] == string_nos)[0]+1)-1)
        try:
            if isinstance(data[det],float):
                annot_array[index] = f"{data[det]:.2f}"
            else:
                annot_array[index] = f"{data[det]}"
            data_array[index] =data[det]
            if data[det] == -1:
                annot_array[index] = " "
        except:
            annot_array[index] = ' '
            data_array[index] = -1
    x_axes = np.full(n_strings*2+1, " ",dtype = object)
    for i, s in enumerate(string_nos):
        x_axes[2*(i+1)-1] = f'Str {s}'
    y_axes = np.full(max_pos *2+1, " ", dtype = object)
    for i, n in enumerate(pos_nos):
        y_axes[2*(i+1)-1] = f'Pos {n}'
    return data_array, x_axes, y_axes, annot_array
