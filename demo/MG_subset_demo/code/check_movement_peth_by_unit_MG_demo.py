#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 10:09:47 2023

@author: daltonm
"""

from pynwb import NWBHDF5IO
import ndx_pose
import numpy as np
import matplotlib.pyplot as plt
from importlib import sys
import neo
import elephant
from quantities import s
from scipy.stats import median_test
import os
import seaborn as sns
from pathlib import Path
from scipy.signal import savgol_filter
import pandas as pd

script_directory = Path(os.path.dirname(os.path.abspath(sys.argv[0])))
code_path = script_directory.parent.parent.parent / 'clean_final_analysis/'
data_path = script_directory.parent.parent / 'data' / 'demo'

sys.path.insert(0, str(code_path))
from hatlab_nwb_functions import remove_duplicate_spikes_from_good_single_units   

marmcode = 'MG'

if marmcode == 'TY':
    nwb_analysis_file = data_path / 'TY' / 'TY20210211_freeAndMoths-003_resorted_20230612_DM.nwb'
    reach_specific_units = [15, 19, 26, 48, 184, 185, 246, 267, 273, 321, 327, 358, 375, 417, 457, 762, 790, 856, 887]
elif marmcode == 'MG':
    nwb_analysis_file = data_path / 'MG' / 'MG20230416_1505_mothsAndFree-002_processed_DM.nwb'
    reach_specific_units = [87, 96, 225, 231, 246, 253, 524, 711, 780]#[225, 246, 253, 524, 780]#[87, 225, 246, 253, 524, 711, 780]

dataset_code = nwb_analysis_file.stem.split('_')[0]
plot_storage = plots = script_directory.parent / 'plots' / dataset_code / 'reaching_PETHs'
modulation_data_storage = data_path / f'{nwb_analysis_file.stem}_modulationData.h5'

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
        # mod_index = round((savFilt.max() - savFilt.min()) / savFilt.min(), 2)
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
        
        baseline_rate, modulated_rate = PETH.as_array()[baseline_mask].mean(), PETH.as_array()[mod_mask].mean() 
        mod_index = round((modulated_rate - baseline_rate) / baseline_rate, 2)
            
    return spiketrains, PETH, mod_index

def generate_PETHs_aligned_to_reaching(units, reaches, kin_module, preTime=1, postTime=1):
    reach_start_times = [reach.start_time for idx, reach in reaches.iterrows()]
    reach_end_times   = [reach.stop_time for idx, reach in reaches.iterrows()]
    reach_peak_times  = [float(reach.peak_extension_times.split(',')[0]) for idx, reach in reaches.iterrows() if len(reach.peak_extension_times)>0]
    
    modulation_df = pd.DataFrame()
    for units_row, unit in units.iterrows():
        
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
        
        plt.savefig(plot_storage / f'unit_{unit.unit_name}.png', 
                    dpi='figure', format=None, metadata=None,
                    bbox_inches=None, pad_inches=0.1,
                    facecolor='auto', edgecolor='auto',
                    backend=None
                    )
        
        plt.show()
        
        tmp_df = pd.DataFrame(data=mod_list + dev_list + [unit.unit_name] + [fg]).T
        tmp_df.columns = mod_label_list + dev_label_list + ['unit_name'] + ['FG']
        modulation_df = pd.concat((modulation_df, tmp_df), axis=0, ignore_index=True)

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
    
    io_prc = NWBHDF5IO(nwb_analysis_file, mode='r')
    nwb_prc = io_prc.read()

    reaches_key = [key for key in nwb_prc.intervals.keys() if 'reaching_segments' in key][0]
    
    units, reaches, kin_module = get_sorted_units_and_apparatus_kinematics_with_metadata(nwb_prc, reaches_key, plot=False)

    modulation_df = generate_PETHs_aligned_to_reaching(units, reaches, kin_module, preTime=1, postTime=1)
    
    io_prc.close()
        
    for met in modulation_df.columns[:6]:
        modulation_df[met] = modulation_df[met].astype(float)
        modulation_in_functional_group(modulation_df, metric=met, hue_order=['Reach-Specific', 'Non-Specific'])
    
    # with open(modulation_data_storage, 'wb') as f:
    #     dill.dump(modulation_df, f, recurse=True) 
    
    modulation_df.to_hdf(modulation_data_storage, 'modulation')