# -*- coding: utf-8 -*-
"""
Created on Fri May 13 11:53:14 2022

@author: Dalton
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import dill
import os
import glob
import seaborn as sns
from scipy import sparse
from scipy.ndimage import gaussian_filter
from sklearn.metrics.cluster import normalized_mutual_info_score

from pynwb import NWBHDF5IO
from nwbwidgets import nwb2widget
import ndx_pose
from importlib import sys
from os.path import join as pjoin

sys.path.insert(0, '/project/nicho/projects/marmosets/code_database/data_processing/nwb_tools/hatlab_nwb_tools/')
from hatlab_nwb_functions import get_sorted_units_and_apparatus_kinematics_with_metadata   

sys.path.insert(0, '/project/nicho/projects/marmosets/code_database/analysis/trajectory_encoding_model/')
from utils import choose_units_for_model


nwb_infile   = '/project/nicho/projects/dalton/data/TY20210211_freeAndMoths-003_resorted_20230612_DM.nwb'
base, ext = os.path.splitext(nwb_infile)
base, old_tag = base.split('DM')
new_tag = '_with_functional_networks'  
nwb_outfile = base + 'DM' + new_tag + ext

class params:
    frate_thresh = 2
    snr_thresh = 3
    binwidth = 10
    FN_metric = 'fMI'
    
class plot_params:
    axis_fontsize = 20
    dpi = 300
    axis_linewidth = 2
    tick_length = 2
    tick_width = 1
    tick_fontsize = 18

    map_figSize = (6, 8)
    weights_by_distance_figsize = (6, 4)
    aucScatter_figSize = (6, 6)

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
        rng = np.random.default_rng(seed=10)
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

        nwb.add_scratch(reach_set_df, name = 'split_FNs_reach_sets', description='pandas DataFrame holding a record of which reaches were used to create each network for split reaches.')

    else:
        raster_set = [raster_list]
    
    FN_list = []
    for rasters in raster_set:
        FN = np.zeros((rasters[0].shape[0], rasters[0].shape[0]))
        
        aa_matrix    = np.zeros_like(FN)
        ii_matrix    = np.zeros_like(FN)
        ai_matrix    = np.zeros_like(FN)
        ia_matrix    = np.zeros_like(FN)
        count_vector = np.zeros_like(FN[0])
        for rNum, FN_data in enumerate(rasters):
            
            print(mode, rNum)
            
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
                        print((rNum, i, j))
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
            
        if plot:
            fig, ax = plt.subplots()
            sns.heatmap(FN,ax=ax,cmap= 'magma',square=True)
            plt.show()
        
        FN = FN.T # transpose so that target units vary by row and input units vary by column. This allows for simple matrix multiplication with FN @ input_activity producing the dot product of weight*activity into a single target neuron
        
        FN_list.append(FN)
    
    if mode == 'split':
        return FN_list
    else:
        return FN_list[0]

def df2binMat_csc(df, binwidth):
    units = df.unit_idx
    spikes_ms = df.spike_time * 1e3
    nUnits = int(units.max()+1)
    nrow = nUnits 
    ncol = int(spikes_ms.max() - spikes_ms.min()) // binwidth + 1
    binMat_lil = sparse.lil_matrix((nrow, ncol))
    for u in range(nUnits):
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

def create_binarized_rasters(units, reaches, kin_module, nwb, mode):
    
    raster_list = []

    reach_start_times = [reach.start_time for idx, reach in reaches.iterrows()]
    reach_end_times   = [reach.stop_time for idx, reach in reaches.iterrows()]
    spontaneous_start_times = [nwb.intervals['video_events_free'].start_time[0]] + reach_end_times.copy()
    spontaneous_end_times   = reach_start_times.copy() + [nwb.intervals['video_events_free'].stop_time[0]]

    if mode == 'reach':
        start_times = reach_start_times
        stop_times  = reach_end_times
    elif mode == 'spontaneous':
        start_times = spontaneous_start_times
        stop_times  = spontaneous_end_times
    
    try:
        chronological_spikes_df = nwb.scratch['spikes_chronological'].to_dataframe() 
    except:
        chronological_spikes_df = arrange_all_unit_spikes_chronologically(units)
        nwb.add_scratch(chronological_spikes_df, name = 'spikes_chronological', description='pandas DataFrame with spikes ordered chronologically for all units.')
    
    for start, stop in zip(start_times, stop_times):
        
        print(mode, start, stop)
        
        segment_spikes_df = chronological_spikes_df.copy()
        start_mask = segment_spikes_df.loc[:, 'spike_time'] > start
        stop_mask  = segment_spikes_df.loc[:, 'spike_time'] < stop
        segment_spikes_df = segment_spikes_df.loc[(start_mask) & (stop_mask), :]
            
        raster = df2binMat_csc(segment_spikes_df, params.binwidth)
        raster_list.append(raster)

    return raster_list
        

if __name__ == "__main__":

    with NWBHDF5IO(nwb_infile, 'r') as io_in:
        nwb = io_in.read()

        reaches_key = [key for key in nwb.intervals.keys() if 'reaching_segments' in key][0]
        
        units, reaches, kin_module = get_sorted_units_and_apparatus_kinematics_with_metadata(nwb, reaches_key, plot=False)    
        
        units = choose_units_for_model(units, quality_key = 'snr', quality_thresh = params.snr_thresh, frate_thresh = params.frate_thresh)
        # units = choose_units_for_model(units, quality_percentile=5, frate_thresh=params.frate_thresh)
        
        reach_raster_list       = create_binarized_rasters(units, reaches, kin_module, nwb, mode = 'reach')
        spontaneous_raster_list = create_binarized_rasters(units, reaches, kin_module, nwb, mode = 'spontaneous')
    
        reach_FN        = make_FN(reach_raster_list      , metric=params.FN_metric, plot=True, self_edge=False, norm=False, mode='all')
        split_reach_FNs = make_FN(reach_raster_list      , metric=params.FN_metric, plot=True, self_edge=False, norm=False, mode='split')
        spontaneous_FN  = make_FN(spontaneous_raster_list, metric=params.FN_metric, plot=True, self_edge=False, norm=False, mode='all')
        
        FN = {'reach_FN'        : reach_FN,
              'split_reach_FNs' : split_reach_FNs,
              'spontaneous_FN'  : spontaneous_FN,
              'split_by' : 'even or odd reaches'}
        
        nwb.add_scratch(reach_FN, name = 'all_reach_FN',           description='Functional network for all reaches. Generated using %d ms bins, FN_metric = %s, and transposed so target units vary in the first dimension (rows) and input units in the columns' % (params.binwidth, params.FN_metric))
        nwb.add_scratch(split_reach_FNs, name = 'split_reach_FNs', description='Functional networks for odd (first element of list) or even (second element) reaches. Generated using %d ms bins, FN_metric = %s, and transposed so target units vary in the first dimension (rows) and input units in the columns' % (params.binwidth, params.FN_metric))
        nwb.add_scratch(spontaneous_FN, name = 'spontaneous_FN',   description='Functional network for all non-reaching (spontaneous) behavior. Some of this behavior occurs in apparatus before/after reaches, but most is in home enclosure. Generated using %d ms bins, FN_metric = %s, and transposed so target units vary in the first dimension (rows) and input units in the columns' % (params.binwidth, params.FN_metric))
    
        with NWBHDF5IO(nwb_outfile, mode='w') as export_io:
            export_io.export(src_io=io_in, nwbfile=nwb)
            
    
    
    # with open(os.path.join(path.intermediate_save_path, 'FN_%s_fMI_10ms_bins_dict.pkl' % path.date), 'wb') as f:
    #     dill.dump([FN], f, recurse= True)