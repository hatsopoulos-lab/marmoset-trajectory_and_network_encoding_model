# -*- coding: utf-8 -*-
"""
Created on Fri May 13 11:53:14 2022

@author: Dalton
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dill
import os
import glob
import seaborn as sns
from scipy import sparse
from scipy.ndimage import gaussian_filter
from sklearn.metrics.cluster import normalized_mutual_info_score

class path:
    storage = r'Z:/marmosets/processed_datasets'
    # intermediate_save_path = r'C:\Users\daltonm\Documents\Lab_Files\encoding_model\intermediate_variable_storage'
    intermediate_save_path = r'C:\Users\Dalton\Documents\lab_files\analysis_encoding_model\intermediate_variable_storage'
    date = '20210211'

class params:
    frate_thresh = 2
    binwidth = 10
    only_include_good_shoulder_frames = False

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
        raster_set = [raster_list[0 : len(raster_list) : 2],
                      raster_list[1 : len(raster_list) : 2]]
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
        
        FN_list.append(FN)
    
    if mode == 'split':
        return FN_list
    else:
        return FN_list[0]

def df2binMat_csc(df, binwidth):
    units = df.unit
    spikes_ms = df.spikeTime * 1e3
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

def choose_units_for_model(spike_data):
    cluster_info = spike_data['cluster_info']
    
    unit_info = cluster_info.loc[cluster_info.group != 'noise', :]
    
    if 'snr' in unit_info.keys():
        quality = unit_info.snr
    else:
        quality = unit_info.amp
    
    quality_thresh = np.percentile(quality, 5)      
    frate_thresh   = params.frate_thresh
    
    unit_info = unit_info.loc[(quality > quality_thresh) | (cluster_info.group == 'good'), :]
    unit_info = unit_info.loc[unit_info.fr > frate_thresh, :]
        
    return unit_info

def compute_derivatives(marker_pos, smooth = True):
    marker_vel = np.diff(marker_pos, axis = -1)
    if smooth:
        for dim in range(3):
            marker_vel[dim] = gaussian_filter(marker_vel[dim], sigma=1.5)
    return marker_vel

def get_frames_in_bouts(kin, vel):
    
    bout_boundaries = []
    for start, stop in zip(kin['starts'], kin['stops']):
        
        if params.only_include_good_shoulder_frames:
        
            tmp_reach_idxs = np.array(range(start, stop-1))
            tmp_reach = vel[0, 0, tmp_reach_idxs]
            tmp_reach_idxs = tmp_reach_idxs[~np.isnan(tmp_reach)]
            
            new_start = tmp_reach_idxs[0]
            new_stop  = tmp_reach_idxs[-1]
            
            bout_boundaries.append([new_start, new_stop])
        else:
            bout_boundaries.append([start, stop-1])        
        
    return bout_boundaries

def create_binarized_rasters(allData, mode):
    
    unit_info = choose_units_for_model(allData.spike_data)
    valid_units = unit_info.cluster_id.values

    goalIdx = [idx for idx, exp in enumerate(allData.analog_and_video['experiments']) if 'free' not in exp][0]
    
    raster_list = []
    prev_reach_end_time = 0
    for kin in allData.kinematics:

        eventIdx = kin['event'] - 1        

        handIdx     = [idx for idx, mark in enumerate(kin['marker_names']) if mark == 'hand'][0]
        shoulderIdx = [idx for idx, mark in enumerate(kin['marker_names']) if mark == 'shoulder'][0]
        
        eventExpTimes = allData.analog_and_video['frameTimes_byEvent'][goalIdx][eventIdx]

        pos = np.empty((1, 3, kin['position'].shape[-1]))
        vel = np.empty_like(pos[..., :-1])
        pos[0] = kin['position'][handIdx] - kin['position'][shoulderIdx]
        for mark in range(pos.shape[0]): 
            vel[mark] = compute_derivatives(pos[mark], smooth = True)
            
        bout_boundaries = get_frames_in_bouts(kin, vel)      
    
        for bounds in bout_boundaries:
            
            if mode == 'reach':
                startTime = eventExpTimes[bounds[0]]
                stopTime = eventExpTimes[bounds[1]]
            elif mode == 'spontaneous':
                startTime = prev_reach_end_time
                stopTime =  eventExpTimes[bounds[0]]
                prev_reach_end_time = eventExpTimes[bounds[1]]
                
            bout_idxs = np.where((allData.spike_data['spike_times'] > startTime) 
                                 & (allData.spike_data['spike_times'] < stopTime)) 
            times = allData.spike_data['spike_times'][bout_idxs]
            units = allData.spike_data['spike_clusters'][bout_idxs]
            keep_idx = [idx for idx, unit in enumerate(units) if unit in valid_units]
            times = times[keep_idx]
            units = units[keep_idx]
            simpleIdx = np.arange(unit_info.shape[0])
            units = [int(simpleIdx[unit_info.cluster_id.to_numpy() == clust]) for clust in units]
            
            spikes_df = pd.DataFrame(data=zip(units, times), columns=['unit', 'spikeTime'])
            
            raster = df2binMat_csc(spikes_df, params.binwidth)
            raster_list.append(raster)

    return raster_list
    
def load_data():
    # spike_path = glob.glob(os.path.join(path.storage, 'formatted_spike_data', '%s*.pkl' % path.date))
    # spike_path = [f for f in spike_path if 'sleep' not in f][0]
    spike_path = r'C:\Users\Dalton\Documents\lab_files\local_spikesort_curation\20210211_freeAndMoths_spike_data.pkl'
    with open(spike_path, 'rb') as fp:
        spike_data = dill.load(fp)
    
    kin_path = glob.glob(os.path.join(path.storage, 'reach_and_trajectory_information', '%s*.pkl' % path.date))[0]
    with open(kin_path, 'rb') as fp:
        kinematics = dill.load(fp)

    analog_path = glob.glob(os.path.join(path.storage, 'analog_signal_and_video_frame_information/pickle_files', '%s*.pkl' % path.date))[0]
    with open(analog_path, 'rb') as fp:
        analog_and_video = dill.load(fp)

    return spike_data, kinematics, analog_and_video
        
if __name__ == "__main__":

    spike_data, kinematics, analog_and_video = load_data() 
    
    class allData:
        spike_data = spike_data
        analog_and_video = analog_and_video
        kinematics = kinematics   
        
    spontaneous_raster_list = create_binarized_rasters(allData, mode = 'spontaneous')
    reach_raster_list       = create_binarized_rasters(allData, mode = 'reach')

    reach_FN        = make_FN(reach_raster_list      , metric='fMI', plot=True, self_edge=False, norm=False, mode='all')
    split_reach_FNs = make_FN(reach_raster_list      , metric='fMI', plot=True, self_edge=False, norm=False, mode='split')
    spontaneous_FN  = make_FN(spontaneous_raster_list, metric='fMI', plot=True, self_edge=False, norm=False, mode='all')
    
    FN = {'reach_FN'        : reach_FN,
          'split_reach_FNs' : split_reach_FNs,
          'spontaneous_FN'  : spontaneous_FN,
          'split_by' : 'even or odd reaches'}
    
    with open(os.path.join(path.intermediate_save_path, 'FN_%s_fMI_10ms_bins_dict.pkl' % path.date), 'wb') as f:
        dill.dump([FN], f, recurse= True)