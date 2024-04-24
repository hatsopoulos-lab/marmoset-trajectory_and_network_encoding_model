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
from scipy import sparse
from sklearn.metrics.cluster import normalized_mutual_info_score

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
    annotation   = data_path / 'TY' / 'spontaneous_behavior_annotation_TY20210211.csv'
    chew_path    = data_path / 'TY' / 'chewing_times.h5'
    ext_ret_path = data_path / 'TY' / 'reaching_extend_retract_segments.h5'
    bad_units_list = None
    mua_to_fix = []
elif marm == 'MG':
    nwb_infile   = data_path / 'MG' / 'MG20230416_1505_mothsAndFree-002_processed_DM_with_functional_networks.nwb'
    annotation   = data_path / 'MG' / 'spontaneous_behavior_annotation_MG20230416.csv'
    chew_path    = data_path / 'MG' / 'chewing_times.h5'
    ext_ret_path = data_path / 'MG' / 'reaching_extend_retract_segments.h5'
    bad_units_list = [181, 440]
    mua_to_fix = [745, 796]

nwb_outfile = nwb_infile.parent / f'{nwb_infile.stem}_single_reaches{nwb_infile.suffix}'    

frates_outfile = nwb_infile.parent / f'{nwb_infile.stem}_single_reach_average_firing_rates.h5' 

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

def MI(prob_matrix):

    # input: prob_matrix of shape (s,r)
    # stimulus: presyn: rows
    # (number of stimulus = rows in matrix = prob_matrix.shape[0])
    # response: postsyn: columns
    # (number of stimulus = columns in matrix = prob_matrix.shape[0])

    MI = 0
    for r, response in enumerate(np.sum(prob_matrix,axis=1)):
        for s, stimulus in enumerate(np.sum(prob_matrix,axis=0)):
            
            pr = response 
            ps = stimulus
            
            if (prob_matrix[r,s] != 0) and (ps!=0) and (pr!=0):
                MI += prob_matrix[r,s]*(np.log2(prob_matrix[r,s]) - np.log2(pr*ps))
    
    return MI

# create list of binary raster, each entry for each reaching bout. Iterate over bouts, 
# summing successes of each category and summing total changes of each category. 
# At end, compute probability matrix and MI 

def make_FN(raster_list, metric='MI',plot=True,self_edge=False, norm=False, mode='all'):
    
    if mode == 'split':
        rng = np.random.default_rng(seed=11)
        raster_idxs = list(range(len(raster_list)))
        reaches_set_1 = rng.choice(raster_idxs, int(len(raster_idxs) / 2), replace=False)
        reaches_set_2 = np.setdiff1d(raster_idxs, reaches_set_1)
        raster_set = [[rast for idx, rast in enumerate(raster_list) if idx in reaches_set_1],
                      [rast for idx, rast in enumerate(raster_list) if idx in reaches_set_2]]
        # raster_set = [raster_list[0 : len(raster_list) : 2],
        #               raster_list[1 : len(raster_list) : 2]]

        reach_set_df = pd.DataFrame(data = zip(list(reaches_set_1) + list(reaches_set_2), 
                                               [1]*len(reaches_set_1) + [2]*len(reaches_set_2)),
                                    columns = ['reach_num', 'FN_reach_set'])
        reach_set_df.sort_values(by = 'reach_num', ignore_index=True, inplace=True)

        # nwb.add_scratch(reach_set_df, name = 'split_FNs_reach_sets', description='pandas DataFrame holding a record of which reaches were used to create each network for split reaches.')

    elif mode == 'single_reach':
        raster_set = [[rast] for rast in raster_list] 
    elif mode == 'half':
        raster_set = [[rast for idx, rast in enumerate(raster_list) if idx <  len(raster_list)/2],
                      [rast for idx, rast in enumerate(raster_list) if idx >= len(raster_list)/2]]       
    else:
        raster_set = [raster_list]
    
    FN_list = []
    for setNum, rasters in enumerate(raster_set):
        
        print(mode, setNum)
 
        FN = np.zeros((rasters[0].shape[0], rasters[0].shape[0]))
        
        aa_matrix    = np.zeros_like(FN)
        ii_matrix    = np.zeros_like(FN)
        ai_matrix    = np.zeros_like(FN)
        ia_matrix    = np.zeros_like(FN)
        count_vector = np.zeros_like(FN[0])
        for rNum, FN_data in enumerate(rasters):
                        
            FN_data = FN_data.toarray()
            
            # Mutual Information
            if metric=='MI':
                for i, presyn in enumerate(FN_data):
                    count_vector[i] += len(presyn) 
                    for j, postsyn in enumerate(FN_data):    
                        # Both are active
                        aa = np.where(np.logical_and(presyn>0,postsyn>0))[0]
                        aa_matrix[i,j] += len(aa)
        
                        # Both are inactive
                        ii =  np.where(np.logical_and(presyn==0,postsyn==0))[0]
                        ii_matrix[i,j] += len(ii)
        
                        # Only presyn is active
                        ai =  np.where(np.logical_and(presyn>0,postsyn==0))[0]
                        ai_matrix[i,j] += len(ai)
        
                        # only postsyn is active
                        ia =  np.where(np.logical_and(presyn==0,postsyn>0))[0]
                        ia_matrix[i,j] += len(ia)
                        
            # Consecutive Mutual Information
            elif metric=='cMI':
                for i, presyn in enumerate(FN_data):
                    presyn  = presyn[:-1] #don't include last (to keep everything in the correct order)
                    count_vector[i] += len(presyn) 
                    for j, postsyn in enumerate(FN_data):
                        # shift one timebin over
                        postsyn = postsyn[1:] #don't include the first (because that is rolled value)
                        
                        # Both are active
                        aa = np.where(np.logical_and(presyn>0,postsyn>0))[0]
                        aa_matrix[i,j] += len(aa)
        
                        # Both are inactive
                        ii =  np.where(np.logical_and(presyn==0,postsyn==0))[0]
                        ii_matrix[i,j] += len(ii)
        
                        # Only presyn is active
                        ai =  np.where(np.logical_and(presyn>0,postsyn==0))[0]
                        ai_matrix[i,j] += len(ai)
        
                        # only postsyn is active
                        ia =  np.where(np.logical_and(presyn==0,postsyn>0))[0]
                        ia_matrix[i,j] += len(ia)
                        
            #full Mutual Information
            elif metric=='fMI':
                for i, presyn in enumerate(FN_data):
                    presyn  = presyn[:-1] #don't include last (to keep everything in the correct order)
                    count_vector[i] += len(presyn) 
                    for j, postsyn in enumerate(FN_data):
                        # shift one timebin over
                        postsyn = postsyn[1:] + postsyn[:-1] # look at both consecutive and simoulaneous timebins!
                        postsyn[postsyn>0] = 1
                    
                        # Both are active
                        aa = np.where(np.logical_and(presyn>0,postsyn>0))[0]
                        aa_matrix[i,j] += len(aa)
        
                        # Both are inactive
                        ii =  np.where(np.logical_and(presyn==0,postsyn==0))[0]
                        ii_matrix[i,j] += len(ii)
        
                        # Only presyn is active
                        ai =  np.where(np.logical_and(presyn>0,postsyn==0))[0]
                        ai_matrix[i,j] += len(ai)
        
                        # only postsyn is active
                        ia =  np.where(np.logical_and(presyn==0,postsyn>0))[0]
                        ia_matrix[i,j] += len(ia)
            #elif metric=='corr':
                 
            #elif metric=='lagcorr':
            else:
                print('That metric does not exist!')
        
        for i in range(len(FN)):
            for j in range(len(FN)):
                prob_matrix = np.zeros((2, 2))
                prob_matrix[1,1] = aa_matrix[i, j] / count_vector[i]
                prob_matrix[0,0] = ii_matrix[i, j] / count_vector[i]
                prob_matrix[1,0] = ai_matrix[i, j] / count_vector[i]
                prob_matrix[0,1] = ia_matrix[i, j] / count_vector[i]
        
                if norm:
                    FN[i,j] = normalized_mutual_info_score(presyn,postsyn) # this requires reorganizing everything
                else:
                    MIij = MI(prob_matrix)
                    FN[i,j] = MIij   
         
        if not self_edge: #zero out diagonal
            np.fill_diagonal(FN,0)
        
        FN = FN.T # transpose so that target units vary by row and input units vary by column. This allows for simple matrix multiplication with FN @ input_activity producing the dot product of weight*activity into a single target neuron
        
        FN_list.append(FN)
    
    if mode in ['split', 'single_reach', 'half']:
        return FN_list
    else:
        return FN_list[0]

def df2binMat_csc(df, binwidth, nUnits=None, duration=None):
    units = df.unit_idx
    spikes_ms = df.spike_time * 1e3
    if not nUnits:
        nUnits = int(units.max()+1)
    nrow = nUnits
    if not duration:
        duration = spikes_ms.max() - spikes_ms.min()        
    else:
        duration = duration * 1e3
    ncol = int(duration) // binwidth + 1
    binMat_lil = sparse.lil_matrix((nrow, ncol))
    for u in units.unique():
        spike_train_of_a_neuron = spikes_ms[units == u]
        bins = np.arange(spikes_ms.min(), spikes_ms.max(), binwidth)
        digitized_spike_train_of_a_neuron = np.digitize(spike_train_of_a_neuron, bins) - 1
        binned_spike_train_of_a_neuron = np.bincount(digitized_spike_train_of_a_neuron)
        binMat_lil[u, digitized_spike_train_of_a_neuron] = binned_spike_train_of_a_neuron[digitized_spike_train_of_a_neuron]
    return binMat_lil.tocsc()

def arrange_all_unit_spikes_chronologically(units): 
    spike_times = list()
    unit_names  = list()
    unit_idxs   = list() 
    for units_row, unit in units.iterrows():
        spike_times.extend(unit.spike_times)
        unit_idxs.extend([units_row for i in range(len(unit.spike_times))])
        unit_names.extend([int(unit.unit_name) for i in range(len(unit.spike_times))])
        
    chronological_spikes_df = pd.DataFrame(data=zip(unit_idxs, unit_names, spike_times), columns=['unit_idx', 'unit_name', 'spike_time'])
    chronological_spikes_df.sort_values(by='spike_time', ascending=True, inplace=True, ignore_index=True)
    
    return chronological_spikes_df

def create_binarized_rasters(units, reaches, kin_module, nwb, chew_df=None, reach_seg_df = None, mode=None):
    
    try:
        chronological_spikes_df = nwb.scratch['spikes_chronological'].to_dataframe() 
    except:
        chronological_spikes_df = arrange_all_unit_spikes_chronologically(units)
        nwb.add_scratch(chronological_spikes_df, name = 'spikes_chronological', description='pandas DataFrame with spikes ordered chronologically for all units.')
    
    nUnits = chronological_spikes_df.unit_idx.max()+1
    
    reach_start_times = [reach.start_time for idx, reach in reaches.iterrows()]
    reach_end_times   = [reach.stop_time for idx, reach in reaches.iterrows()]
    spontaneous_start_times = [nwb.intervals['video_events_free'].start_time[0]] + reach_end_times.copy()
    spontaneous_end_times   = reach_start_times.copy() + [nwb.intervals['video_events_free'].stop_time[0]]    

    raster_list = []
    if mode == 'reach':
        start_times = reach_start_times
        stop_times  = reach_end_times
    elif mode == 'spontaneous':
        start_times = spontaneous_start_times
        stop_times  = spontaneous_end_times
    elif mode == 'reach_segments':
        start_times = reach_seg_df['start']
        stop_times  = reach_seg_df['stop']
    
    for start, stop in zip(start_times, stop_times):
        
        print(mode, start, stop)
        if start == stop:
            continue
        
        segment_spikes_df = chronological_spikes_df.copy()
        start_mask = segment_spikes_df.loc[:, 'spike_time'] > start
        stop_mask  = segment_spikes_df.loc[:, 'spike_time'] < stop
        segment_spikes_df = segment_spikes_df.loc[(start_mask) & (stop_mask), :]
            
        raster = df2binMat_csc(segment_spikes_df, params.binwidth, nUnits=nUnits, duration=stop-start)
        raster_list.append(raster)

    return raster_list

def plot_functional_networks(FN, units_res, FN_key = 'split_reach_FNs', cmin=None, cmax=None, subset_idxs = None, subset_type='both'):
    
    if units_res is not None:
    
        # units_sorted = units_res.copy()    
        # units_sorted.sort_values(by='cortical_area', inplace=True, ignore_index=False)
        units_sorted = pd.DataFrame()
        for area in ['3b', '3a', 'M1', '6dc']: 
            tmp_df = units_res.loc[units_res['cortical_area']==area, :]
            units_sorted = pd.concat((units_sorted, tmp_df), axis=0, ignore_index=True)
    
        if subset_idxs is not None:
            units_sorted_subset = units_res.copy().loc[subset_idxs, :]
            units_sorted_subset.sort_values(by='cortical_area', inplace=True, ignore_index=False) 
            
            subset_tick_3a = np.sum(units_sorted_subset['cortical_area'] == '3a')
            subset_tick_3b = subset_tick_3a + np.sum(units_sorted_subset['cortical_area'] == '3b') 
            subset_tick_m1 = subset_tick_3b + np.sum(units_sorted_subset['cortical_area'] == 'M1')
        
        tick_3a = np.sum(units_sorted['cortical_area'] == '3a')
        tick_3b = tick_3a + np.sum(units_sorted['cortical_area'] == '3b') 
        tick_m1 = tick_3b + np.sum(units_sorted['cortical_area'] == 'M1')
    
    if FN.ndim < 3:
        FN = np.expand_dims(FN, axis = 0)
    
    if cmin is None:
        net_min = []
        net_max = []
        for network in FN:
            net_min.append(np.nanmin(network))
            net_max.append(np.nanmax(network))
        cmin = min(net_min)
        cmax = max(net_max)
    
    if FN_key == 'split_reach_FNs':
        titles = ['Reach Set 1', 'Reach Set 2']
    elif FN_key == 'spontaneous_FN':
        titles = ['Spontaneous']
    else:
        titles = [FN_key]
    
    for network, title in zip(FN, titles):
        fig, ax = plt.subplots(figsize=(6,6), dpi = plot_params.dpi)
        network_copy = network.copy()
        
        if units_res is not None:
            if subset_idxs is not None:
                if subset_idxs.size > FN.shape[-1]/2:
                    title += ' Non'
                if subset_type == 'both':
                    title += ' Reach Specific'
                    target_idx, source_idx = units_sorted_subset.index.values, units_sorted_subset.index.values 
                    xtick_3a, xtick_3b, xtick_m1 = subset_tick_3a, subset_tick_3b, subset_tick_m1
                    ytick_3a, ytick_3b, ytick_m1 = subset_tick_3a, subset_tick_3b, subset_tick_m1               
                elif subset_type == 'target':
                    title += ' Reach Specific Targets'
                    target_idx, source_idx = units_sorted_subset.index.values, units_sorted.index.values  
                    xtick_3a, xtick_3b, xtick_m1 =        tick_3a,        tick_3b,        tick_m1
                    ytick_3a, ytick_3b, ytick_m1 = subset_tick_3a, subset_tick_3b, subset_tick_m1  
                elif subset_type == 'source':
                    title += ' Reach Specific Sources'
                    target_idx, source_idx = units_sorted.index.values, units_sorted_subset.index.values  
                    xtick_3a, xtick_3b, xtick_m1 = subset_tick_3a, subset_tick_3a, subset_tick_3a
                    ytick_3a, ytick_3b, ytick_m1 =        tick_3a,        tick_3b,        tick_m1  
            else:
                target_idx, source_idx = units_sorted.index.values, units_sorted.index.values  
                xtick_3a, xtick_3b, xtick_m1 = tick_3a, tick_3b, tick_m1
                ytick_3a, ytick_3b, ytick_m1 = tick_3a, tick_3b, tick_m1  

            network_copy = network_copy[np.ix_(target_idx, source_idx)]
        sns.heatmap(network_copy,ax=ax,cmap= 'viridis',square=True, norm=colors.PowerNorm(0.5, vmin=cmin, vmax=cmax)) # norm=colors.LogNorm(vmin=cmin, vmax=cmax)
        if units_res is not None:
            ax.set_xticks([np.mean([0, xtick_3a]), xtick_3a, np.mean([xtick_3a, xtick_3b]), xtick_3b, np.mean([xtick_3b, xtick_m1])])
            ax.set_yticks([np.mean([0, ytick_3a]), ytick_3a, np.mean([ytick_3a, ytick_3b]), tick_3b, np.mean([ytick_3b, ytick_m1])])
            ax.set_xticklabels(['3a', '', '3b', '', 'Motor'])
            ax.set_yticklabels(['3a', '', '3b', '', 'Motor'])
        ax.set_title(title, fontsize=plot_params.axis_fontsize)
        ax.set_ylabel('Target Unit', fontsize=plot_params.axis_fontsize)
        ax.set_xlabel('Input Unit' , fontsize=plot_params.axis_fontsize)
        plt.show()
            
        # plt.hist(network_copy.flatten(), bins = 30)
        # plt.show()
        
    return cmin, cmax        

def compute_average_firing_rates(reach_raster_list, modes = ['single_reach', 'half']):
    
    total_spikes = [0 for i in range(reach_raster_list[0].shape[0])]
    total_time = 0
    for rast in reach_raster_list:
        for uIdx, unitSpikes in enumerate(rast):  
            total_spikes[uIdx] += unitSpikes.sum()
        
        total_time += rast.shape[1] * params.binwidth*1e-3
    
    average_reach_fr = [spkCount / total_time for spkCount in total_spikes]
    
    average_frates = pd.DataFrame(data = average_reach_fr,
                                  columns = ['Reach'])    
    
    if 'half' in modes:
        total_spikes = [0 for i in range(reach_raster_list[0].shape[0])]
        total_time = 0
        for rast in reach_raster_list[:int(len(reach_raster_list)/2)]:
            for uIdx, unitSpikes in enumerate(rast):  
                total_spikes[uIdx] += unitSpikes.sum()
            
            total_time += rast.shape[1] * params.binwidth*1e-3
        
        average_fr = [spkCount / total_time for spkCount in total_spikes]
        average_frates['first_half'] = average_fr
        
        total_spikes = [0 for i in range(reach_raster_list[0].shape[0])]
        total_time = 0
        for rast in reach_raster_list[int(len(reach_raster_list)/2):]:
            for uIdx, unitSpikes in enumerate(rast):  
                total_spikes[uIdx] += unitSpikes.sum()
            
            total_time += rast.shape[1] * params.binwidth*1e-3
        
        average_fr = [spkCount / total_time for spkCount in total_spikes]
        average_frates['second_half'] = average_fr
    
    if 'single_reach' in modes:
        for reachNum, rast in enumerate(reach_raster_list):
            total_spikes = [0 for i in range(reach_raster_list[0].shape[0])]
            total_time = 0
            for uIdx, unitSpikes in enumerate(rast):  
                total_spikes[uIdx] += unitSpikes.sum()
            
            total_time += rast.shape[1] * params.binwidth*1e-3
        
            average_fr = [spkCount / total_time for spkCount in total_spikes]
            average_frates[f'reach{reachNum}'] = average_fr    
    
    average_frates.to_hdf(frates_outfile, 'frates')
    
    # with open(frates_outfile, 'wb') as f:
    #     dill.dump(average_frates, f, recurse=True)

if __name__ == "__main__":

    with NWBHDF5IO(nwb_infile, 'r') as io_in:
        nwb = io_in.read()

        reaches_key = [key for key in nwb.intervals.keys() if 'reaching_segments' in key][0]
        
        units, reaches, kin_module = get_sorted_units_and_apparatus_kinematics_with_metadata(nwb, reaches_key, mua_to_fix=params.mua_to_fix, plot=False)    
        
        units = choose_units_for_model(units, quality_key = 'snr', 
                                       quality_thresh = params.snr_thresh, 
                                       frate_thresh = params.frate_thresh, 
                                       bad_units_list = bad_units_list)

        reach_raster_list       = create_binarized_rasters(units, reaches, kin_module, nwb, chew_df = None, mode = 'reach')
    
        compute_average_firing_rates(reach_raster_list)
    
        single_reach_FNs = make_FN(reach_raster_list, metric=params.FN_metric, plot=True, self_edge=False, norm=False, mode='single_reach')
        half_FNs         = make_FN(reach_raster_list, metric=params.FN_metric, plot=True, self_edge=False, norm=False, mode='half')

        # for key, FN in enumerate(half_FNs):
        #     cmin, cmax = plot_functional_networks(FN, units_res=None, FN_key = key)
        # # cmin, cmax = plot_functional_networks(np.concatenate((np.expand_dims(FN['split_reach_FNs'][0], 0), np.expand_dims(FN['split_reach_FNs'][1], 0)), axis=0), units_res=None, FN_key = 'split_reach_FNs', cmin=cmin, cmax=cmax)
        # cmin, cmax = plot_functional_networks(np.concatenate((np.expand_dims(FN['split_reach_FNs'][0], 0), 
        #                                                       np.expand_dims(FN['split_reach_FNs'][1], 0)), axis=0), 
        #                                       units_res=None, 
        #                                       FN_key = 'split_reach_FNs')
        
        # plot_functional_networks(FN['spontaneous_FN'], units_res=None, FN_key ='spontaneous_FN', cmin=cmin, cmax=cmax)
        
        for reachNum, FN in enumerate(single_reach_FNs):
            nwb.add_scratch(FN, name = f'reach{str(reachNum).zfill(3)}', description= 'Functional network for single reach. Generated using %d ms bins, FN_metric = %s, and transposed so target units vary in the first dimension (rows) and input units in the columns' % (params.binwidth, params.FN_metric))

        for half, FN in zip(['half1', 'half2'], half_FNs):
            nwb.add_scratch(FN, name = half, description= 'Functional network for half of session. Generated using %d ms bins, FN_metric = %s, and transposed so target units vary in the first dimension (rows) and input units in the columns' % (params.binwidth, params.FN_metric))

        # nwb.add_scratch(reach_FN, name = 'all_reach_FN',           description='Functional network for all reaches. Generated using %d ms bins, FN_metric = %s, and transposed so target units vary in the first dimension (rows) and input units in the columns' % (params.binwidth, params.FN_metric))
        # nwb.add_scratch(split_reach_FNs, name = 'split_reach_FNs', description='Functional networks for odd (first element of list) or even (second element) reaches. Generated using %d ms bins, FN_metric = %s, and transposed so target units vary in the first dimension (rows) and input units in the columns' % (params.binwidth, params.FN_metric))
        # nwb.add_scratch(spontaneous_FN, name = 'spontaneous_FN',   description='Functional network for all non-reaching (spontaneous) behavior. Some of this behavior occurs in apparatus before/after reaches, but most is in home enclosure. Generated using %d ms bins, FN_metric = %s, and transposed so target units vary in the first dimension (rows) and input units in the columns' % (params.binwidth, params.FN_metric))
    
        nwb.scratch.pop('spikes_chronological')
    
        with NWBHDF5IO(nwb_outfile, mode='w') as export_io:
            export_io.export(src_io=io_in, nwbfile=nwb)
            