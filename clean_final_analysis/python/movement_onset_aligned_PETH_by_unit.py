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
from os.path import join as pjoin
from scipy.stats import median_test
import os
import seaborn as sns
import dill
from pathlib import Path
from scipy.signal import savgol_filter
import pandas as pd
from scipy.ndimage import median_filter

data_path = Path('/project/nicho/projects/dalton/network_encoding_paper/clean_final_analysis/data')
code_path = Path('/project/nicho/projects/marmosets/code_database/analysis/trajectory_encoding_model/clean_final_analysis/')
plot_path = Path('/project/nicho/projects/dalton/grant_work_with_context_specific_units/plots/')

sys.path.insert(0, str(code_path))
from hatlab_nwb_functions import remove_duplicate_spikes_from_good_single_units   
from utils import load_dict_from_hdf5

marmcode = 'TY'
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(context='notebook', style="ticks", palette='Dark2', rc=custom_params)

if marmcode == 'TY':
    nwb_analysis_file = data_path / 'TY' / 'TY20210211_freeAndMoths-003_resorted_20230612_DM.nwb'
    reach_specific_units = [15, 19, 26, 48, 184, 185, 246, 267, 273, 321, 327, 358, 375, 417, 457, 762, 790, 856, 887]
    best_lead_lag_key = 'lead_100_lag_300'
    bad_units_list = None
elif marmcode == 'MG':
    nwb_analysis_file = data_path / 'MG' / 'MG20230416_1505_mothsAndFree-002_processed_DM.nwb'
    reach_specific_units = [87, 96, 225, 231, 246, 253, 524, 711, 780]#[225, 246, 253, 524, 780]#[87, 225, 246, 253, 524, 711, 780]
    best_lead_lag_key = 'lead_100_lag_300'
    bad_units_list = [181, 440]

frate_thresh = 2
snr_thresh = 3

dataset_code = nwb_analysis_file.stem.split('_')[0]
plot_storage = plot_path / 'reaching_PETHs' / marmcode
modulation_data_storage = nwb_analysis_file.parent / f'{nwb_analysis_file.stem}_modulationData_normalizedIndex.pkl'

results_file_tag = 'network_models_created'
results_file   = nwb_analysis_file.parent / f'{nwb_analysis_file.stem}_{results_file_tag}.h5'

gen_file_tag = 'generalization_experiments_LocClimbCombined'
gen_results_file = nwb_analysis_file.parent / f'{nwb_analysis_file.stem}_{gen_file_tag}.h5'

os.makedirs(plot_storage, exist_ok=True)
os.makedirs(plot_storage / 'Context-specific', exist_ok=True)
os.makedirs(plot_storage / 'Context-invariant', exist_ok=True)

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
        mod_index = round((savFilt.max() - savFilt.min()) / savFilt.min(), 2)
        # mod_index = round((savFilt.max() - savFilt.min()), 2)
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

def generate_PETHs_aligned_to_reaching(units, units_res, reaches, kin_module, preTime=1, postTime=1):
    reach_start_times = [reach.start_time for idx, reach in reaches.iterrows()]
    reach_end_times   = [reach.stop_time for idx, reach in reaches.iterrows()]
    reach_peak_times  = [float(reach.peak_extension_times.split(',')[0]) for idx, reach in reaches.iterrows() if len(reach.peak_extension_times)>0]
    
    modulation_df = pd.DataFrame()
    for units_row, unit in units.iterrows():
        
        if unit["unit_name"] not in units_res["unit_name"].values: 
            continue
        
        units_res_row = units_res.loc[units_res["unit_name"] == unit.unit_name, :]
    
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
            axP.plot(PETH.times, savFilt)
            
            axM.plot(PETH.times, savFilt)
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
        
        plt.savefig(plot_storage / units_res_row["Functional Group"].values[0] / f'unit_{unit.unit_name}.png', dpi='figure', format=None, metadata=None,
                    bbox_inches=None, pad_inches=0.1,
                    facecolor='auto', edgecolor='auto',
                    backend=None
                    )
        
        plt.show()
        
        tmp_df = pd.DataFrame(data=mod_list + dev_list + [unit.unit_name] + [units_res_row["Functional Group"]]).T
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

def add_generalization_experiments_to_units_df(units_res, gen_res):
    
    for model_key in gen_res['model_results'].keys():
        if 'reach_test_FN' in model_key:
            units_res[f'{model_key}_auc'] = gen_res['model_results'][model_key]['AUC'].mean(axis=-1)
    
    return units_res  

def compute_performance_difference_by_unit(units_res, model_1, model_2):
    
    if model_1[-4:] != '_auc':
        model_1 = model_1 + '_auc'
    if model_2[-4:] != '_auc':
        model_2 = model_2 + '_auc'
    
    cols = [col for col in units_res.columns if 'auc' not in col]
    diff_df = units_res.loc[:, cols]
    
    dist_from_unity = np.abs(-1*units_res[model_1] + 1*units_res[model_2]) / np.sqrt(1**2+1**2)
    
    model_diff = pd.DataFrame(data = units_res[model_2] - units_res[model_1],
                              columns = ['auc_diff'])
    
    model_dist = pd.DataFrame(data = dist_from_unity,
                              columns = ['dist_from_unity'])
    diff_df = pd.concat((diff_df, model_diff, model_dist), axis = 1)
    
    return diff_df

def find_context_specific_group(diff_df, model_1, model_2, gen_test_behavior=None):


    sorted_diff_df = diff_df.sort_values(by='auc_diff', ascending=False)
    # sorted_diff_df['dist_positive_grad'] = np.hstack((np.abs(np.diff(sorted_diff_df['dist_from_unity'])),
    #                                                   [np.nan]))     
    sorted_diff_df['derivative_auc_diff'] = np.hstack((np.abs(np.diff(sorted_diff_df['auc_diff'])),
                                                      [np.nan]))  
    medFilt_grad = median_filter(sorted_diff_df['derivative_auc_diff'], 8) #TODO 9
    sorted_diff_df['medfilt_derivative_auc_diff'] = medFilt_grad
    lastUnit = np.where(medFilt_grad < 0.1  * np.nanmax(medFilt_grad))[0][0] #TODO 0.075
    top_value_cut = -12 if marmcode=='TY' else -3 
    tmp = sorted_diff_df['derivative_auc_diff'].values
    tmp = tmp[~np.isnan(tmp)]
    lastUnit = np.where(medFilt_grad < 0.1 * np.median(np.sort(tmp)[top_value_cut:]))[0][0] #TODO 0.075
    if marmcode=='TY' and gen_test_behavior.lower()=='rest':
        lastUnit = 60 # TODO

    context_specific_units  = sorted_diff_df.index[:lastUnit] 
    context_invariant_units = sorted_diff_df.index[lastUnit:]
    
    return context_specific_units, context_invariant_units    

def summarize_model_results(units, lead_lag_keys):  
    
    if type(lead_lag_keys) != list:
        lead_lag_keys = [lead_lag_keys]
    
    for lead_lag_key in lead_lag_keys:
        
        if 'all_models_summary_results' in results_dict[lead_lag_key].keys():
            all_units_res = results_dict[lead_lag_key]['all_models_summary_results']
        else:
            all_units_res = units.copy()
            
        try:
            all_units_res.drop(columns=['spike_times', 'n_spikes'], inplace=True)
        except:
            pass
        
        for model_key in results_dict[lead_lag_key]['model_results'].keys():
            col_names = ['%s_auc' % model_key]  
            results_keys = ['AUC']  

            for col_name, results_key in zip(col_names, results_keys):                
                if col_name not in all_units_res.columns and results_key in results_dict[lead_lag_key]['model_results'][model_key].keys(): 
                    all_units_res[col_name] = results_dict[lead_lag_key]['model_results'][model_key][results_key].mean(axis=-1)
                else:
                    print('This model (%s, %s) has already been summarized in the all_models_summary_results dataframe' % (lead_lag_key, model_key))                    

        results_dict[lead_lag_key]['all_models_summary_results'] = all_units_res.copy()
        
        return results_dict
        
if __name__ == '__main__':
    # io_acq = NWBHDF5IO(nwb_acquisition_file, mode='r')
    # nwb_acq = io_acq.read()
    
    io_prc = NWBHDF5IO(nwb_analysis_file, mode='r')
    nwb_prc = io_prc.read()

    reaches_key = [key for key in nwb_prc.intervals.keys() if 'reaching_segments' in key][0]
    
    units, reaches, kin_module = get_sorted_units_and_apparatus_kinematics_with_metadata(nwb_prc, reaches_key, plot=False)
    # units = choose_units_for_model(units, quality_key = 'snr', quality_thresh = params.snr_thresh, frate_thresh = params.frate_thresh, bad_units_list=bad_units_list)

    results_dict = load_dict_from_hdf5(results_file, top_level_list=False, convert_4d_array_to_list = True)    
    results_dict = summarize_model_results(units=None, lead_lag_keys = best_lead_lag_key)

    units_res = results_dict[best_lead_lag_key]['all_models_summary_results']

    generalization_results = load_dict_from_hdf5(gen_results_file)
    units_res = add_generalization_experiments_to_units_df(units_res, generalization_results[best_lead_lag_key])

    gen_test_behavior = 'spont'
    diff_df = compute_performance_difference_by_unit(units_res, f'traj_avgPos_{gen_test_behavior}_train_reach_test_FN_auc', 'traj_avgPos_reach_FN_auc')   
    # reach_specific_units = diff_df.index[(diff_df.auc_diff > 0) & (diff_df.dist_from_unity > params.reach_specific_thresh) & (diff_df.dist_from_unity < 0.04)]
    context_specific_units, context_invariant_units = find_context_specific_group(diff_df,
                                                                           f'traj_avgPos_{gen_test_behavior}_train_reach_test_FN', 
                                                                           'traj_avgPos_reach_FN',
                                                                           gen_test_behavior=gen_test_behavior)
    
    units_res['Functional Group'] = ['Context-invariant' for idx in range(units_res.shape[0])]
    units_res.loc[context_specific_units, 'Functional Group'] = ['Context-specific' for idx in range(context_specific_units.size)]
    
    modulation_df = generate_PETHs_aligned_to_reaching(units, units_res, reaches, kin_module, preTime=1, postTime=1)  
    
    io_prc.close()
        
    # for met in modulation_df.columns[:6]:
    #     modulation_df[met] = modulation_df[met].astype(float)
    #     modulation_in_functional_group(modulation_df, metric=met, hue_order=['Reach-Specific', 'Non-Specific'])
    
    # with open(modulation_data_storage, 'wb') as f:
    #     dill.dump(modulation_df, f, recurse=True) 
