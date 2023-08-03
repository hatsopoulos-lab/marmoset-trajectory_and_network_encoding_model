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
import seaborn as sns

sys.path.insert(0, '/project/nicho/projects/marmosets/code_database/data_processing/nwb_tools/hatlab_nwb_tools/')
from hatlab_nwb_functions import remove_duplicate_spikes_from_good_single_units, get_raw_timestamps   


#nwb_file = '/project/nicho/data/marmosets/electrophys_data_for_processing/TY20210211_freeAndMoths/TY20210211_freeAndMoths-003.nwb'
#nwb_file = '/project/nicho/data/marmosets/electrophys_data_for_processing/MG20230416_1505_mothsAndFree/MG20230416_1505_mothsAndFree-002_acquisition.nwb'
# nwb_acquisition_file = '/project/nicho/data/marmosets/electrophys_data_for_processing/TY20210211_freeAndMoths/TY20210211_freeAndMoths-003_acquisition.nwb'
nwb_analysis_file   = '/project/nicho/data/marmosets/electrophys_data_for_processing/MG20230416_1505_mothsAndFree/MG20230416_1505_mothsAndFree-002_processed.nwb' 

plot_storage = '/project/nicho/projects/dalton/plots/MG20230416/reaching_PETHs'

def get_sorted_units_and_apparatus_kinematics_with_metadata(nwb_prc, reaches_key, plot=False):
    units          = nwb_prc.units.to_dataframe()
    units = remove_duplicate_spikes_from_good_single_units(units, plot=plot)
    reaches        = nwb_prc.intervals[reaches_key].to_dataframe()
    
    kin_module_key = reaches.iloc[0].kinematics_module
    kin_module = nwb_prc.processing[kin_module_key]
    
    return units, reaches, kin_module

def get_aligned_spiketrains_and_PETH(units, spike_times, align_times, preTime=1, postTime=1):

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
    
    return spiketrains, PETH

def generate_PETHs_aligned_to_reaching(units, reaches, kin_module, preTime=1, postTime=1):
    reach_start_times = [reach.start_time for idx, reach in reaches.iterrows()]
    reach_end_times   = [reach.stop_time for idx, reach in reaches.iterrows()]
    reach_peak_times  = [float(reach.peak_extension_times.split(',')[0]) for idx, reach in reaches.iterrows()]
    for units_row, unit in units.iterrows():
        spike_times = unit.spike_times
        
        spiketrains_RS, PETH_RS = get_aligned_spiketrains_and_PETH(units, spike_times, reach_start_times, preTime=preTime, postTime=postTime)
        spiketrains_RE, PETH_RE = get_aligned_spiketrains_and_PETH(units, spike_times, reach_end_times  , preTime=preTime, postTime=postTime)
        spiketrains_RP, PETH_RP = get_aligned_spiketrains_and_PETH(units, spike_times, reach_peak_times , preTime=preTime, postTime=postTime)


        PETH_ymax = np.max([np.max(PETH_RS.magnitude.flatten()), np.max(PETH_RP.magnitude.flatten()), np.max(PETH_RE.magnitude.flatten())])
        fig, ((P0, P1, P2), (R0, R1, R2)) = plt.subplots(2, 3, sharex='col', figsize=(8, 6), dpi=300)
        left_plots = True
        for axP, axR, spiketrains, PETH, label in zip([P0, P1, P2], 
                                                      [R0, R1, R2], 
                                                      [spiketrains_RS, spiketrains_RP, spiketrains_RE], 
                                                      [PETH_RS, PETH_RP, PETH_RE],
                                                      ['RO', 'RP', 'RE']):

            axP.bar(PETH.times, PETH.magnitude.flatten(), width=PETH.sampling_period, align='edge', alpha=0.3, label='time histogram (rate)')
            axP.vlines(0, 0, PETH_ymax, colors='black', linestyles='solid')
            axP.set_ylim(0, np.ceil(PETH_ymax)+1)
            
            axR.eventplot([st.magnitude for st in spiketrains], linelengths=0.75, linewidths=0.75, color='black')
            axR.vlines(0, 0, len(spiketrains), colors='black', linestyles='solid')
            axR.set_xlabel("Time, s")
            axR.set_xlim(-1*preTime, postTime)
            axR.set_ylim(0, len(spiketrains))
            axR.set_xticks([-1*preTime, 0, postTime])
            axR.set_xticklabels([-1*preTime, label, postTime])


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
        
        plt.savefig(pjoin(plot_storage, 'unit_%s.png' % unit.unit_name), dpi='figure', format=None, metadata=None,
                    bbox_inches=None, pad_inches=0.1,
                    facecolor='auto', edgecolor='auto',
                    backend=None
                    )
        
        plt.show()
        

if __name__ == '__main__':
    # io_acq = NWBHDF5IO(nwb_acquisition_file, mode='r')
    # nwb_acq = io_acq.read()
    
    io_prc = NWBHDF5IO(nwb_analysis_file, mode='r')
    nwb_prc = io_prc.read()

    reaches_key = [key for key in nwb_prc.intervals.keys() if 'reaching_segments' in key][0]
    
    units, reaches, kin_module = get_sorted_units_and_apparatus_kinematics_with_metadata(nwb_prc, reaches_key, plot=False)

    generate_PETHs_aligned_to_reaching(units, reaches, kin_module, preTime=1, postTime=1)
    
    io_prc.close()
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
