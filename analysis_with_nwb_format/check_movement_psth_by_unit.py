#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 10:09:47 2023

@author: daltonm
"""

from pynwb import NWBHDF5IO
from nwbwidgets import nwb2widget
import ndx_pose
import numpy as np
import matplotlib.pyplot as plt
from importlib import sys
import neo
import elephant
from quantities import s
from os.path import join as pjoin
from scipy.stats import median_test
import os
import seaborn as sns
import dill
from scipy.signal import savgol_filter, medfilt
import pandas as pd

sys.path.insert(0, '/project/nicho/projects/marmosets/code_database/data_processing/nwb_tools/hatlab_nwb_tools/')
from hatlab_nwb_functions import remove_duplicate_spikes_from_good_single_units, get_raw_timestamps   


#nwb_file = '/project/nicho/data/marmosets/electrophys_data_for_processing/TY20210211_freeAndMoths/TY20210211_freeAndMoths-003.nwb'
#nwb_file = '/project/nicho/data/marmosets/electrophys_data_for_processing/MG20230416_1505_mothsAndFree/MG20230416_1505_mothsAndFree-002_acquisition.nwb'
# nwb_acquisition_file = '/project/nicho/data/marmosets/electrophys_data_for_processing/TY20210211_freeAndMoths/TY20210211_freeAndMoths-003_acquisition.nwb'

marmcode = 'MG'

if marmcode == 'TY':
    nwb_analysis_file = '/beagle3/nicho/projects/dalton/data/TY/TY20210211_freeAndMoths-003_resorted_20230612_DM.nwb'
    plot_storage = '/project/nicho/projects/dalton/plots/TY20210211/reaching_PETHs'
    reach_specific_units = [15, 19, 26, 48, 184, 185, 246, 267, 273, 321, 327, 358, 375, 417, 457, 762, 790, 856, 887]
elif marmcode == 'MG':
    nwb_analysis_file = '/beagle3/nicho/projects/dalton/data/MG/MG20230416_1505_mothsAndFree-002_processed_DM.nwb'
    plot_storage = '/project/nicho/projects/dalton/plots/MG20230416/reaching_PETHs'

os.makedirs(plot_storage, exist_ok=True)

def get_sorted_units_and_apparatus_kinematics_with_metadata(nwb_prc, reaches_key, plot=False):
    units          = nwb_prc.units.to_dataframe()
    units = remove_duplicate_spikes_from_good_single_units(units, plot=plot)
    reaches        = nwb_prc.intervals[reaches_key].to_dataframe()
    
    kin_module_key = reaches.iloc[0].kinematics_module
    kin_module = nwb_prc.processing[kin_module_key]
    
    return units, reaches, kin_module

def get_aligned_spiketrains_and_PETH(units, spike_times, align_times, preTime=1, postTime=1, mod_index_mode = 'start'):

    spiketrains = [[] for i in align_times]
    for idx, t_align in enumerate(align_times):
        spike_times_aligned = spike_times - t_align
        spike_times_aligned = [spk for spk in spike_times_aligned if spk > -1*preTime and spk < postTime]
        spiketrains[idx] = neo.spiketrain.SpikeTrain(spike_times_aligned*s, 
                                                     t_start=-1*preTime*s, 
                                                     t_stop =postTime*s)
    
    PETH = elephant.statistics.time_histogram(spiketrains, 
                                              0.05*s, 
                                              t_start=None, 
                                              t_stop=None, 
                                              output='rate', 
                                              binary=False)
    
    if mod_index_mode == 'savgol':
        savFilt = savgol_filter(PETH.as_array().flatten(), 13, 3)
        # savFilt = medfilt(PETH.as_array().flatten(), 7)

        # mod_index = round((savFilt.max() - savFilt.min())/savFilt.mean(), 2)
        mod_index = round((savFilt.max() - savFilt.min()), 2)

    else:
        center_bin = int(preTime / .05)
        if mod_index_mode == 'start':
            baseline_bins = list(range(0, int(center_bin-.25/.05)))
            mod_bins = list(range(center_bin, int(center_bin+.75/.05+1)))
        elif mod_index_mode == 'stop':
            baseline_bins = list(range(0, center_bin))
            mod_bins = list(range(center_bin, int(center_bin+postTime/.05+1)))
        elif mod_index_mode == 'peak':
            baseline_bins = list(range(0, int(center_bin-.75/.05+1))) + list(range(int(center_bin+.75/.05), int(center_bin+postTime/.05+1)))
            mod_bins = list(range(int(center_bin-.25/.05), int(center_bin+.25/.05+1)))
        
        baseline_mask = np.array([True if idx in baseline_bins else False for idx in range(PETH.shape[0])])
        mod_mask      = np.array([True if idx in      mod_bins else False for idx in range(PETH.shape[0])])
    
        mod_index = round(PETH.as_array()[mod_mask].mean() / PETH.as_array()[baseline_mask].mean(), 2)
            
    return spiketrains, PETH, mod_index

def generate_PETHs_aligned_to_reaching(units, reaches, kin_module, preTime=1, postTime=1):
    reach_start_times = [reach.start_time for idx, reach in reaches.iterrows()]
    reach_end_times   = [reach.stop_time for idx, reach in reaches.iterrows()]
    reach_peak_times  = [float(reach.peak_extension_times.split(',')[0]) for idx, reach in reaches.iterrows() if len(reach.peak_extension_times)>0]
    
    modulation_df = pd.DataFrame()
    for units_row, unit in units.iterrows():
        
        # if int(unit.unit_name) < 260:
        #     continue
    
        if reach_specific_units is None:
            fg = 'none'
        elif int(unit.unit_name) in reach_specific_units:
            fg = 'Reach-Specific'
        else:
            fg = 'Non-Specific'
    
        spike_times = unit.spike_times
        
        spiketrains_RS, PETH_RS, mod_RS = get_aligned_spiketrains_and_PETH(units, spike_times, reach_start_times, preTime=preTime, postTime=postTime, mod_index_mode = 'savgol')
        spiketrains_RE, PETH_RE, mod_RE = get_aligned_spiketrains_and_PETH(units, spike_times, reach_end_times  , preTime=preTime, postTime=postTime, mod_index_mode = 'savgol')
        spiketrains_RP, PETH_RP, mod_RP = get_aligned_spiketrains_and_PETH(units, spike_times, reach_peak_times , preTime=preTime, postTime=postTime, mod_index_mode = 'savgol')

        PETH_ymax = np.max([np.max(PETH_RS.magnitude.flatten()), np.max(PETH_RP.magnitude.flatten()), np.max(PETH_RE.magnitude.flatten())])
        PETH_ymin = np.max([np.min(PETH_RS.magnitude.flatten()), np.min(PETH_RP.magnitude.flatten()), np.min(PETH_RE.magnitude.flatten())])

        fig, ((P0, P1, P2), (M0, M1, M2), (R0, R1, R2)) = plt.subplots(3, 3, sharex='col', figsize=(8, 8), dpi=300)
        left_plots = True
        
        mod_list = []
        dev_list = []
        mod_label_list = []
        dev_label_list = []
        for axP, axM, axR, spiketrains, PETH, label, mod in zip([P0, P1, P2],
                                                                [M0, M1, M2],
                                                                [R0, R1, R2], 
                                                                [spiketrains_RS, spiketrains_RP, spiketrains_RE], 
                                                                [PETH_RS, PETH_RP, PETH_RE],
                                                                ['RO', 'RP', 'RE'],
                                                                [mod_RS, mod_RP, mod_RE]):

            axP.bar(PETH.times, PETH.magnitude.flatten(), width=PETH.sampling_period, align='edge', alpha=0.3, label='time histogram (rate)')
            axP.vlines(0, 0, PETH_ymax, colors='black', linestyles='solid')
            axP.set_ylim(0, np.ceil(PETH_ymax)+1)            
            
            savFilt = savgol_filter(PETH.as_array().flatten(), 13, 3)
            # savFilt = medfilt(PETH.as_array().flatten(), 7)

            deviance = round(np.max(np.abs(savFilt - np.linspace(savFilt[0], savFilt[-1], PETH.shape[0]))), 1)
            axP.plot(PETH.times, savFilt, '-r')
            
            axM.plot(PETH.times, savFilt, '-r')
            axM.plot(PETH.times, np.linspace(savFilt[0], savFilt[-1], PETH.shape[0]), '-k')
            axM.set_ylim(np.floor(PETH_ymin), np.floor(PETH_ymin)+40)
            
            
            axR.eventplot([st.magnitude for st in spiketrains], linelengths=0.75, linewidths=0.75, color='black')
            axR.vlines(0, 0, len(spiketrains), colors='black', linestyles='solid')
            axR.set_xlabel("Time, s")
            axR.set_xlim(-1*preTime, postTime)
            axR.set_ylim(0, len(spiketrains))
            axR.set_xticks([-1*preTime, 0, postTime])
            axR.set_xticklabels([-1*preTime, label, postTime])
            axR.set_title(f'mod = {mod}, dev={deviance}')


            if left_plots:
                axP.set_ylabel('Rate (spikes/sec)')
                axR.set_ylabel("Trial")
                axP.set_yticks([0, round(PETH_ymax)])
                axR.set_yticks([0, len(spiketrains)])
                sns.despine(ax=axP)
                sns.despine(ax=axR)
                left_plots = False
            elif axP == P1:
                axP.set_title('Unit %s, %s, amp=%d, x=%d, y=%d, %s' % (unit.unit_name, 
                                                                       unit.quality, 
                                                                       int(unit.amp), 
                                                                       int(unit.x), 
                                                                       int(unit.y),
                                                                       unit.electrode_label))

                sns.despine(ax=axP, top=True, left=True, right=True)                
                sns.despine(ax=axR, top=True, left=True, right=True)    
                axP.set_yticks([])
                axR.set_yticks([])

            else:
                sns.despine(ax=axP, top=True, left=True, right=True)                
                sns.despine(ax=axR, top=True, left=True, right=True)
                axP.set_yticks([])
                axR.set_yticks([])
            
            mod_list.append(mod)
            dev_list.append(deviance)
            mod_label_list.append(f'modulation_{label}')
            dev_label_list.append(f'maxDev_{label}')
        
        plt.savefig(pjoin(plot_storage, 'unit_%s.png' % unit.unit_name), dpi='figure', format=None, metadata=None,
                    bbox_inches=None, pad_inches=0.1,
                    facecolor='auto', edgecolor='auto',
                    backend=None
                    )
        
        plt.show()
        
        tmp_df = pd.DataFrame(data=mod_list + dev_list + [unit.unit_name] + [fg]).T
        tmp_df.columns = mod_label_list + dev_label_list + ['unit_name'] + ['FG']
        modulation_df = pd.concat((modulation_df, tmp_df), axis=0, ignore_index=True)
        # if units_row > 50:
        #     break
    return modulation_df     

def modulation_in_functional_group(modulation_df, metric='modulation_RO', hue_order=None):
    
    med_out = median_test(modulation_df.loc[modulation_df['FG'] == 'Reach-Specific', metric].astype(float), 
                          modulation_df.loc[modulation_df['FG'] ==   'Non-Specific', metric].astype(float))
    print(f'{metric}: RS v NS, p={np.round(med_out[1], 4)}, {med_out[-1][0,0]}a-{med_out[-1][1,0]}b, {med_out[-1][0,1]}a-{med_out[-1][1,1]}b')
    
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.kdeplot(data=modulation_df, ax=ax, x=metric, hue='FG',
                palette='Dark2', hue_order=hue_order,
                common_norm=False, cumulative=True, legend=False)
    ax.text(modulation_df[metric].max()*0.9, 0.25, f'p={np.round(med_out[1], 4)}', horizontalalignment='center', fontsize = 12)
    plt.show()


if __name__ == '__main__':
    # io_acq = NWBHDF5IO(nwb_acquisition_file, mode='r')
    # nwb_acq = io_acq.read()
    
    io_prc = NWBHDF5IO(nwb_analysis_file, mode='r')
    nwb_prc = io_prc.read()

    reaches_key = [key for key in nwb_prc.intervals.keys() if 'reaching_segments' in key][0]
    
    units, reaches, kin_module = get_sorted_units_and_apparatus_kinematics_with_metadata(nwb_prc, reaches_key, plot=False)

    modulation_df = generate_PETHs_aligned_to_reaching(units, reaches, kin_module, preTime=1, postTime=1)
    
    io_prc.close()
        
    for met in modulation_df.columns[:6]:
        modulation_df[met] = modulation_df[met].astype(float)
        modulation_in_functional_group(modulation_df, metric=met, hue_order=['Reach-Specific', 'Non-Specific'])
    
    with open(f'{nwb_analysis_file.split(".nwb")[0]}_modulationData.pkl', 'wb') as f:
        dill.dump(modulation_df, f, recurse=True) 
    
    
# unit_to_plot = units.loc[units.electrode_label == elabel, :]
# spike_times = unit_to_plot.spike_times.iloc[0]
# # Get electrodes table, extract the channel index matching the desired electrode_label
# raw_elec_table = nwb_acq.acquisition['ElectricalSeriesRaw'].electrodes.to_dataframe()
# raw_elec_index = raw_elec_table.index[raw_elec_table.electrode_label == elabel]

# # Get first 100000 samples raw data for that channel index
# raw_data_single_chan = nwb_acq.acquisition['ElectricalSeriesRaw'].data[:100000, raw_elec_index.values]


# # ##### Pull out data around spike time in raw neural data (using tMod = 0 or tMod = nwbfile.acqusition['ElectricalSeriesRaw'] starting time)

# # In[38]:


# tMod = 0 #nwb_acq.acquisition['ElectricalSeriesRaw'].starting_time
# spikes_indexed_in_raw = [np.where(np.isclose(raw_timestamps, spk_time+tMod, atol=1e-6))[0][0] for spk_time in spike_times[:10]]


# # In[39]:


# spkNum = 1
# plt.plot(raw_timestamps[spikes_indexed_in_raw[spkNum] - 100 : spikes_indexed_in_raw[spkNum] + 100], 
#          raw_data_single_chan[spikes_indexed_in_raw[spkNum] - 100 : spikes_indexed_in_raw[spkNum] + 100])
# plt.plot(raw_timestamps[spikes_indexed_in_raw[spkNum]], raw_data_single_chan[spikes_indexed_in_raw[spkNum]], 'or')


# # ### Look at an individual reaching segment and link it to the correct kinematics

# # In[40]:


# segment_idx = 39

# # get info in dataframe for specific segment_idx
# segment_df = nwb_prc.intervals['reaching_segments_moths'].to_dataframe()
# segment_info = segment_df.iloc[segment_idx]

# # get event data using container and ndx_pose names from segment_info table following form below:
# # nwb.processing['goal_directed_kinematics'].data_interfaces['moths_s_1_e_004_position']
# event_data      = nwb_prc.processing[segment_info.kinematics_module].data_interfaces[segment_info.video_event] 
# hand_kinematics = event_data.pose_estimation_series['hand'].data[:] 
# timestamps      = event_data.pose_estimation_series['hand'].timestamps[:]
# reproj_error    = event_data.pose_estimation_series['hand'].confidence[:]

# # plot full_event 
# fig, axs = plt.subplots(2, 1, sharex=True)
# axs[0].plot(timestamps, hand_kinematics)
# axs[0].vlines(x=[segment_info.start_time, segment_info.stop_time], ymin=-3,ymax=14, colors='black', linestyle='dashdot')
# axs[1].plot(timestamps, reproj_error, '.b')
# axs[0].set_ylabel('Position (cm) for x (blue), y (orange), z (green)')
# axs[0].set_title('Entire video event hand kinematics')
# axs[1].set_ylabel('Reprojection Error b/w Cameras (pixels)')
# axs[1].set_xlabel('Time (sec)')
# plt.show()

# # extract kinematics of this single reaching segment and plot
# reach_hand_kinematics = hand_kinematics[segment_info.start_idx:segment_info.stop_idx]
# reach_reproj_error    = reproj_error   [segment_info.start_idx:segment_info.stop_idx]
# reach_timestamps      = timestamps     [segment_info.start_idx:segment_info.stop_idx]
# peak_idxs = segment_info.peak_extension_idxs.split(',')
# peak_idxs = [int(idx) for idx in peak_idxs]
# peak_timestamps = timestamps[peak_idxs]
# peak_ypos = hand_kinematics[peak_idxs, 1]

# # plot single reaching segment 
# fig, axs = plt.subplots(2, 1, sharex=True)
# axs[0].plot(reach_timestamps, reach_hand_kinematics)
# axs[0].plot(peak_timestamps, peak_ypos, 'or')
# axs[1].plot(reach_timestamps, reach_reproj_error, '.b')
# axs[0].set_ylabel('Position (cm) for x (blue), y (orange), z (green)')
# axs[0].set_title('Reaching segment hand kinematics')
# axs[1].set_ylabel('Reprojection Error b/w Cameras (pixels)')
# axs[1].set_xlabel('Time (sec)')
# plt.show()


# # In[41]:


# # get table of sorted unit info
# units_df = nwb_prc.units.to_dataframe()
# elec_positions = units_df.loc[:, ['x', 'y', 'z', 'electrode_label']]
# elec_positions


# # ### Load and isolate analog channels using electrodes table

# # In[42]:


# raw = nwb_acq.acquisition['ElectricalSeriesRaw']

# start = raw.starting_time
# step = 1/raw.rate
# stop = start + step*raw.data.shape[0]
# raw_timestamps = np.arange(start, stop, step)

# elec_df = raw.electrodes.to_dataframe()
# analog_idx = [idx for idx, name in elec_df['electrode_label'].iteritems() if 'ainp' in name]
# electrode_labels = elec_df.loc[analog_idx, 'electrode_label']

# # plot the first 3 minutes of data for the channels
# time_to_plot = 3*60
# num_samples = int(raw.rate * time_to_plot)
# num_channels = np.min([2, len(analog_idx)])
# fig, axs = plt.subplots(num_channels, 1, sharex=True) 
# for cIdx in range(num_channels):
#     analog_signals = raw.data[:num_samples, analog_idx[cIdx]] * elec_df['gain_to_uV'][analog_idx[cIdx]] * raw.conversion
#     axs[cIdx].plot(raw_timestamps[:num_samples], analog_signals)
#     axs[cIdx].set_title(electrode_labels.iloc[cIdx])
#     axs[cIdx].set_ylabel('Raw Signal (V)')

# axs[cIdx].set_xlabel('Timestamps (sec)')
    
# plt.show()


# # ### Now for a few neural channels

# # In[43]:


# raw = nwb_acq.acquisition['ElectricalSeriesRaw']
# elec_df = raw.electrodes.to_dataframe()
# analog_idx = [idx for idx, name in elec_df['electrode_label'].iteritems() if 'elec' in name]
# electrode_labels = elec_df.loc[analog_idx, 'electrode_label']

# # plot the first 3 minutes of data for the channels
# time_to_plot = 3*60
# num_samples = int(raw.rate * time_to_plot)
# num_channels = np.min([3, len(analog_idx)])
# fig, axs = plt.subplots(num_channels, 1, sharex=True) 
# for cIdx in range(num_channels):
#     analog_signals = raw.data[:num_samples, analog_idx[cIdx]] * elec_df['gain_to_uV'][analog_idx[cIdx]] * raw.conversion
#     axs[cIdx].plot(raw_timestamps[:num_samples], analog_signals)
#     axs[cIdx].set_title(electrode_labels.iloc[cIdx])

# axs[cIdx].set_ylabel('Raw Signal (V)')
# axs[cIdx].set_xlabel('Timestamps (sec)')
    
# plt.show()


# # ### When you finish working with the data, close the files

# # In[44]:


# io_acq.close()
# io_prc.close()
