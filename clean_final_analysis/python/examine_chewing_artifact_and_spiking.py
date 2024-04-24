# -*- coding: utf-8 -*-
"""
Created on Fri May 13 11:53:14 2022

@author: Dalton
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pathlib import Path

import dill
import re
import seaborn as sns
from scipy import sparse, signal
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import  gaussian_filter
from scipy.fft import rfft, rfftfreq

from sklearn.metrics.cluster import normalized_mutual_info_score

import neo
import elephant
from quantities import s

from pynwb import NWBHDF5IO
import ndx_pose
from importlib import sys

data_path = Path('/project/nicho/projects/dalton/network_encoding_paper/clean_final_analysis/data')
code_path = Path('/project/nicho/projects/marmosets/code_database/analysis/trajectory_encoding_model/clean_final_analysis/')

sys.path.insert(0, str(code_path))
from hatlab_nwb_functions import get_sorted_units_and_apparatus_kinematics_with_metadata   
from utils import choose_units_for_model

marm = 'TY'

if marm == 'TY':
    nwb_infile   = data_path / 'TY' / 'TY20210211_freeAndMoths-003_resorted_20230612_DM_with_functional_networks.nwb'
    nwb_acqfile  = data_path / 'TY' / 'TY20210211_freeAndMoths-003_acquisition.nwb'
    annotation  = data_path / 'TY' / 'spontaneous_behavior_annotation_TY20210211.csv'
    bad_units_list = None
    mua_to_fix = []
elif marm == 'MG':
    nwb_infile  = data_path / 'MG' / 'MG20230416_1505_mothsAndFree-002_processed_DM_with_functional_networks.nwb'
    annotation  = data_path / 'MG' / 'spontaneous_behavior_annotation_MG20230416.csv'
    bad_units_list = [181, 440]
    mua_to_fix = [745, 796]


class params:
    frate_thresh = 2
    snr_thresh = 3
    bad_units_list = bad_units_list
    binwidth = 10
    FN_metric = 'fMI'
    mua_to_fix = mua_to_fix
    
class plot_params:
    axis_fontsize = 20
    dpi = 300



def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def butter_bandpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = [cut / nyq for cut in cutoff]
    b, a = signal.butter(order, normal_cutoff, btype='bandpass', analog=False)
    return b, a

def butter_bandpass_filter(data, cutoff, fs, order=5):
    b, a = butter_bandpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def compute_signal_fft(signal, srate = 30000):
    
    sig_fft = rfft(signal, axis = 0)
    fft_freq = rfftfreq(signal.shape[0], d = 1./srate)    
    
    fig, ax = plt.subplots()
    ax.plot(fft_freq, 2.0/signal.shape[0] * np.abs(sig_fft))
    
    plt.show()
    

if __name__ == "__main__":

    mode = 'save_chewing_segments'  
    chans = [1, 3]

    with NWBHDF5IO(nwb_infile, 'r') as io_in:
        nwb = io_in.read()    
        reaches_key = [key for key in nwb.intervals.keys() if 'reaching_segments' in key][0]
        
        units, reaches, kin_module = get_sorted_units_and_apparatus_kinematics_with_metadata(nwb, reaches_key, mua_to_fix=params.mua_to_fix, plot=False)    
        
        units = choose_units_for_model(units, quality_key = 'snr', quality_thresh = params.snr_thresh, frate_thresh = params.frate_thresh, bad_units_list = bad_units_list)

        event_timestamps = dict()
        for data_key, data in nwb.processing['video_event_timestamps_moths'].data_interfaces.items():
            event_timestamps[data_key] = data.timestamps[:]

    first_event, last_event = 180, 187 #180, 187
    raw_samples_start = int(event_timestamps[f'moths_s_1_e_{first_event}_timestamps'][0]*30000) 
    raw_samples_end   = int(event_timestamps[f'moths_s_1_e_{last_event}_timestamps'][-1]*30000) 
    # raw_samples_end   = int(event_timestamps[f'moths_s_1_e_{last_event}_timestamps'][0]*30000) + 1000

    
    if mode == 'all_chans':
        with NWBHDF5IO(nwb_acqfile, 'r') as io_acq:
            nwb_acq = io_acq.read()
            
            es_key = [key for key in nwb_acq.acquisition.keys() if 'Electrical' in key][0]
            start = nwb_acq.acquisition[es_key].starting_time
            step = 1/nwb_acq.acquisition[es_key].rate
            stop = start + step*nwb_acq.acquisition[es_key].data.shape[0]
            raw_timestamps = np.arange(start, stop, step)
            
            raw_signal = nwb_acq.acquisition[es_key].data[raw_samples_start:raw_samples_end, :4].T 
        
        spiketrains = []
        for idx, single_unit in units.iterrows(): 
            if single_unit.quality == 'good':
                cut_spike_times = single_unit.spike_times[(single_unit.spike_times > raw_timestamps[raw_samples_start]) & \
                                                          (single_unit.spike_times < raw_timestamps[raw_samples_end])]
                spiketrain = neo.spiketrain.SpikeTrain(cut_spike_times*s, 
                                                       t_start=raw_timestamps[raw_samples_start]*s, 
                                                       t_stop =raw_timestamps[raw_samples_end  ]*s)
                spiketrains.append(spiketrain)
    
        PETH = elephant.statistics.time_histogram(spiketrains, 
                                                  0.05*s, 
                                                  t_start=None, 
                                                  t_stop=None, 
                                                  output='rate', 
                                                  binary=False)
        
        highpass_signal = butter_highpass_filter(raw_signal,200,30000)
        fig, (ax0, ax1) = plt.subplots(2,1, sharex=True)
        for chan_data, adj in zip(highpass_signal, [0, 1000, 2000, 3000]):  
            ax0.plot(raw_timestamps[raw_samples_start:raw_samples_end], chan_data)
    
        ax1.plot(np.linspace(raw_timestamps[raw_samples_start], raw_timestamps[raw_samples_end], PETH.shape[0]),
                 PETH,
                 '-b',
                 linewidth=1)
    
        for t_key, timestamps in event_timestamps.items():
            if int(re.findall(re.compile('e_\d{3}'), t_key)[0].split('e_')[-1]) in range(first_event, last_event+1):
                ax0.vlines(timestamps[ 0], -4000, 4000, 'black', '-')
                ax0.vlines(timestamps[-1], -4000, 4000, 'black', ':')
    
                ax1.vlines(timestamps[ 0], 0, 50, 'black', '-')
                ax1.vlines(timestamps[-1], 0, 50, 'black', ':')
        
        ax0.set_ylim(-5000, 4000)
    
        plt.show()
    elif mode == 'individual_chans':
        
        kernel = elephant.kernels.GaussianKernel(sigma=0.025*s)
        
        with NWBHDF5IO(nwb_acqfile, 'r') as io_acq:
            nwb_acq = io_acq.read()
            
            es_key = [key for key in nwb_acq.acquisition.keys() if 'Electrical' in key][0]
            start = nwb_acq.acquisition[es_key].starting_time
            step = 1/nwb_acq.acquisition[es_key].rate
            stop = start + step*nwb_acq.acquisition[es_key].data.shape[0]
            raw_timestamps = np.arange(start, stop, step)
            
            channel_data_list = []
            for channel_idx in range(nwb_acq.acquisition[es_key].data.shape[1]): 
                if channel_idx < chans[0] or channel_idx > chans[1]:
                    continue
                print(channel_idx)
                raw_signal = nwb_acq.acquisition[es_key].data[raw_samples_start:raw_samples_end, channel_idx].T 
                # highpass_signal = butter_highpass_filter(raw_signal,200,30000)
                highpass_signal = butter_bandpass_filter(raw_signal,[200, 500],30000)
                analytic_signal = signal.hilbert(highpass_signal)
                envelope = np.abs(analytic_signal)
                envelope = gaussian_filter(envelope, sigma=200)
                
                fr_list = []
                quality = []
                unit_id = []
                units_on_chan = units.loc[units['channel_index'] == channel_idx, :]
                for idx, single_unit in units_on_chan.iterrows(): 
                    cut_spike_times = single_unit.spike_times[(single_unit.spike_times > raw_timestamps[raw_samples_start]) & \
                                                              (single_unit.spike_times < raw_timestamps[raw_samples_end])]
                    spiketrain = neo.spiketrain.SpikeTrain(cut_spike_times*s, 
                                                           t_start=raw_timestamps[raw_samples_start]*s, 
                                                           t_stop =raw_timestamps[raw_samples_end  ]*s)

                    try:
                        fr = elephant.statistics.instantaneous_rate(spiketrain, 
                                                                    sampling_period = 1/30000*s, 
                                                                    kernel = kernel,
                                                                    t_start=None, 
                                                                    t_stop=None)
                        
                        fr_list.append(fr)
                        quality.append(single_unit.quality)
                        unit_id.append(single_unit.unit_name)
                    except Exception as error:
                        print(f'Channel_index = {channel_idx}, unit_id = {single_unit.unit_name}\nError: {error}')
                
                peaks, _ = find_peaks(envelope,height=500, prominence=350, width=[600, 4000])
                results_full = peak_widths(envelope, peaks, rel_height=.9)   
                
                
                data_dict = {'highpass_signal' : highpass_signal,
                             'envelope'        : envelope,
                             'peaks'           : peaks,
                             'chew_segments'   : results_full,
                             'firing_rate'     : fr_list,
                             'unit_class'      : quality,
                             'unit_id'         : unit_id,
                             'channel_index'   : channel_idx}
                
                channel_data_list.append(data_dict)
                    
        
        for channel_data in channel_data_list:
            if len(channel_data['unit_class']) == 0:
                continue
            fig, (ax0, ax1, ax2) = plt.subplots(3,1, sharex=True)
            ax0.plot(raw_timestamps[raw_samples_start:raw_samples_end], channel_data['highpass_signal'])
            ax0.plot(raw_timestamps[raw_samples_start:raw_samples_end], channel_data['envelope'])
            ax0.plot(raw_timestamps[raw_samples_start:raw_samples_end][channel_data['peaks']], channel_data['envelope'][channel_data['peaks']], 'ok')
            ax0.hlines(channel_data['chew_segments'][1], 
                       raw_timestamps[raw_samples_start:raw_samples_end][[int(idx) for idx in channel_data['chew_segments'][2]]],
                       raw_timestamps[raw_samples_start:raw_samples_end][[int(idx) for idx in channel_data['chew_segments'][3]]],
                       'black')
            
            ax0.set_title(f'Channel_index = {channel_data["channel_index"]}')
            
            for fr, unit_class, unit_id in zip(channel_data['firing_rate'], channel_data['unit_class'], channel_data['unit_id']):    
                plot_ax = ax1 if unit_class == 'good' else ax2
                plot_ax.plot(raw_timestamps[raw_samples_start:raw_samples_end-1],
                             fr.squeeze(),
                             linestyle = '-',
                             linewidth = 1)
                
            
            for t_key, timestamps in event_timestamps.items():
                if int(re.findall(re.compile('e_\d{3}'), t_key)[0].split('e_')[-1]) in range(first_event, last_event+1):
                    ax0.vlines(timestamps[ 0], -4000, 4000, 'black', '-')
                    ax0.vlines(timestamps[-1], -4000, 4000, 'black', ':')
        
                    ax1.vlines(timestamps[ 0], 0, 50, 'black', '-')
                    ax1.vlines(timestamps[-1], 0, 50, 'black', ':')
                    ax2.vlines(timestamps[ 0], 0, 50, 'black', '-')
                    ax2.vlines(timestamps[-1], 0, 50, 'black', ':')
            
            ax1.legend([uID for uID, uClass in zip(channel_data['unit_id'], channel_data['unit_class']) if uClass=='good'], loc="upper left", bbox_to_anchor=(1, 1))
            ax2.legend([uID for uID, uClass in zip(channel_data['unit_id'], channel_data['unit_class']) if uClass=='mua'], loc="upper left", bbox_to_anchor=(1, 1))
        
            ax0.set_ylim(-4000, 4000)
            ax1.set_ylim(0,100)
            ax2.set_ylim(0,150)
            
            plt.show()   
            
    elif mode == 'save_chewing_segments':
         kernel = elephant.kernels.GaussianKernel(sigma=0.025*s)
         
         with NWBHDF5IO(nwb_acqfile, 'r') as io_acq:
             nwb_acq = io_acq.read()
             
             es_key = [key for key in nwb_acq.acquisition.keys() if 'Electrical' in key][0]
             start = nwb_acq.acquisition[es_key].starting_time
             step = 1/nwb_acq.acquisition[es_key].rate
             stop = start + step*nwb_acq.acquisition[es_key].data.shape[0]
             raw_timestamps = np.arange(start, stop, step)
             
             channel_data_list = []
             for channel_idx in range(nwb_acq.acquisition[es_key].data.shape[1]): 
                 if channel_idx < chans[0] or channel_idx > chans[1]:
                     continue
                 print(channel_idx)
                 raw_signal = nwb_acq.acquisition[es_key].data[:, channel_idx].T 
                 # highpass_signal = butter_highpass_filter(raw_signal,200,30000)
                 highpass_signal = butter_bandpass_filter(raw_signal,[200, 500],30000)
                 analytic_signal = signal.hilbert(highpass_signal)
                 envelope = np.abs(analytic_signal)
                 envelope = gaussian_filter(envelope, sigma=200)
                 
                 fr_list = []
                 quality = []
                 unit_id = []
                 units_on_chan = units.loc[units['channel_index'] == channel_idx, :]
                 for idx, single_unit in units_on_chan.iterrows(): 
                     cut_spike_times = single_unit.spike_times[(single_unit.spike_times > raw_timestamps[raw_samples_start]) & \
                                                               (single_unit.spike_times < raw_timestamps[raw_samples_end])]
                     spiketrain = neo.spiketrain.SpikeTrain(cut_spike_times*s, 
                                                            t_start=raw_timestamps[raw_samples_start]*s, 
                                                            t_stop =raw_timestamps[raw_samples_end  ]*s)
        
                     try:
                         fr = elephant.statistics.instantaneous_rate(spiketrain, 
                                                                     sampling_period = 1/30000*s, 
                                                                     kernel = kernel,
                                                                     t_start=None, 
                                                                     t_stop=None)
                         
                         fr_list.append(fr)
                         quality.append(single_unit.quality)
                         unit_id.append(single_unit.unit_name)
                     except Exception as error:
                         print(f'Channel_index = {channel_idx}, unit_id = {single_unit.unit_name}\nError: {error}')
                 
                 peaks, _ = find_peaks(envelope,height=500, prominence=350, width=[600, 4000])
                 results_full = peak_widths(envelope, peaks, rel_height=.9)   
                 
                 
                 data_dict = {'highpass_signal' : highpass_signal,
                              'envelope'        : envelope,
                              'peaks'           : peaks,
                              'chew_segments'   : results_full,
                              'firing_rate'     : fr_list,
                              'unit_class'      : quality,
                              'unit_id'         : unit_id,
                              'channel_index'   : channel_idx}
                 
                 channel_data_list.append(data_dict)
                   
                 if channel_idx == chans[0]:
                     chew_df = pd.DataFrame(data=zip(raw_timestamps[results_full[2].astype(int)], raw_timestamps[results_full[2].astype(int)]),
                                            columns=['start', 'end'])
                     chew_df.to_hdf(data_path /'TY'/'chewing_times.h5', key='timestamps')
         
           # for channel_data in channel_data_list:
           #     if len(channel_data['unit_class']) == 0:
           #         continue
             
           #       fig, (ax0, ax1, ax2) = plt.subplots(3,1, sharex=True)
           #     ax0.plot(raw_timestamps[raw_samples_start:raw_samples_end], channel_data['highpass_signal'])
           #     ax0.plot(raw_timestamps[raw_samples_start:raw_samples_end], channel_data['envelope'])
           #     ax0.plot(raw_timestamps[raw_samples_start:raw_samples_end][channel_data['peaks']], channel_data['envelope'][channel_data['peaks']], 'ok')
           #     ax0.hlines(channel_data['chew_segments'][1], 
           #                raw_timestamps[raw_samples_start:raw_samples_end][[int(idx) for idx in channel_data['chew_segments'][2]]],
           #                raw_timestamps[raw_samples_start:raw_samples_end][[int(idx) for idx in channel_data['chew_segments'][3]]],
           #                'black')
             
           #    ax0.set_title(f'Channel_index = {channel_data["channel_index"]}')
             
           #    for fr, unit_class, unit_id in zip(channel_data['firing_rate'], channel_data['unit_class'], channel_data['unit_id']):    
           #        plot_ax = ax1 if unit_class == 'good' else ax2
           #        plot_ax.plot(raw_timestamps[raw_samples_start:raw_samples_end-1],
           #                     fr.squeeze(),
           #                     linestyle = '-',
           #                     linewidth = 1)
                 
             
           #    for t_key, timestamps in event_timestamps.items():
           #        if int(re.findall(re.compile('e_\d{3}'), t_key)[0].split('e_')[-1]) in range(first_event, last_event+1):
           #            ax0.vlines(timestamps[ 0], -4000, 4000, 'black', '-')
           #            ax0.vlines(timestamps[-1], -4000, 4000, 'black', ':')
         
           #            ax1.vlines(timestamps[ 0], 0, 50, 'black', '-')
           #            ax1.vlines(timestamps[-1], 0, 50, 'black', ':')
           #            ax2.vlines(timestamps[ 0], 0, 50, 'black', '-')
           #            ax2.vlines(timestamps[-1], 0, 50, 'black', ':')
             
           #    ax1.legend([uID for uID, uClass in zip(channel_data['unit_id'], channel_data['unit_class']) if uClass=='good'], loc="upper left", bbox_to_anchor=(1, 1))
           #    ax2.legend([uID for uID, uClass in zip(channel_data['unit_id'], channel_data['unit_class']) if uClass=='mua'], loc="upper left", bbox_to_anchor=(1, 1))
         
           #    ax0.set_ylim(-4000, 4000)
           #    ax1.set_ylim(0,100)
           #    ax2.set_ylim(0,150)
             
           #    plt.show()   
                    
            
                    