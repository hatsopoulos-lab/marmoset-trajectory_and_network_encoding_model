#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 08:00:51 2020

@author: daltonm
"""
#%matplotlib notebook
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import dill
import os
import time
import glob
import math
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc
import statsmodels.api as sm
from scipy.stats import binomtest, ttest_rel
from scipy.integrate import cumtrapz
from scipy.integrate import simps
from mpl_toolkits import mplot3d 
from scipy.ndimage import median_filter, gaussian_filter
from importlib import sys, reload

from pynwb import NWBHDF5IO
import ndx_pose
from importlib import sys
from os.path import join as pjoin

sys.path.insert(0, '/project/nicho/projects/marmosets/code_database/data_processing/nwb_tools/hatlab_nwb_tools/')
from hatlab_nwb_functions import get_sorted_units_and_apparatus_kinematics_with_metadata

sys.path.insert(0, '/project/nicho/projects/marmosets/code_database/analysis/trajectory_encoding_model/')
from utils import get_interelectrode_distances_by_unit, choose_units_for_model

marmcode = 'TY'

if marmcode=='TY':
    nwb_infile = '/project/nicho/projects/dalton/data/TY20210211_freeAndMoths-003_resorted_20230612_DM_with_functional_networks.nwb' 
    pkl_infile = '/project/nicho/projects/dalton/data/TY20210211_freeAndMoths-003_resorted_20230612_DM_encoding_model_sorting_corrected_30ms_shift_v6.pkl'
    models_already_stored = True
elif marmcode=='MG':
    nwb_infile = ''
    pkl_infile = ''
    models_already_stored = False


split_pattern = '_shift_v' # '_results_v'
base, ext = os.path.splitext(pkl_infile)
base, in_version = base.split(split_pattern)
out_version = str(int(in_version) + 1)  
pkl_outfile = base + '_random_FN_split_' + split_pattern + out_version + ext

if models_already_stored:
    models_storage_folder, ext = os.path.splitext(pkl_infile)    
else:
    models_storage_folder, ext = os.path.splitext(pkl_outfile)
models_storage_folder = models_storage_folder + '_stored_encoding_models'

remove_models = ['shuffled_weights_FN_5_percent_by_strength', 
                 'shuffled_weights_FN_10_percent_by_strength', 
                 'shuffled_weights_FN_15_percent_by_strength', 
                 'shuffled_weights_FN_20_percent_by_strength', 
                 'shuffled_weights_FN_25_percent_by_strength', 
                 'shuffled_weights_FN_30_percent_by_strength', 
                 'shuffled_weights_FN_35_percent_by_strength', 
                 'shuffled_weights_FN_40_percent_by_strength', 
                 'shuffled_weights_FN_45_percent_by_strength', 
                 'shuffled_weights_FN_50_percent_by_strength', 
                 'shuffled_topology_FN_5_percent_by_strength', 
                 'shuffled_topology_FN_10_percent_by_strength', 
                 'shuffled_topology_FN_15_percent_by_strength', 
                 'shuffled_topology_FN_20_percent_by_strength', 
                 'shuffled_topology_FN_25_percent_by_strength', 
                 'shuffled_topology_FN_30_percent_by_strength', 
                 'shuffled_topology_FN_35_percent_by_strength', 
                 'shuffled_topology_FN_40_percent_by_strength', 
                 'shuffled_topology_FN_45_percent_by_strength', 
                 'shuffled_topology_FN_50_percent_by_strength', 
                 'shuffled_weights_FN_5_percent_by_random', 
                 'shuffled_weights_FN_10_percent_by_random', 
                 'shuffled_weights_FN_15_percent_by_random', 'shuffled_weights_FN_20_percent_by_random', 'shuffled_weights_FN_25_percent_by_random', 'shuffled_weights_FN_30_percent_by_random', 'shuffled_weights_FN_35_percent_by_random', 'shuffled_weights_FN_40_percent_by_random', 'shuffled_weights_FN_45_percent_by_random', 'shuffled_weights_FN_50_percent_by_random', 'shuffled_topology_FN_5_percent_by_random', 'shuffled_topology_FN_10_percent_by_random', 'shuffled_topology_FN_15_percent_by_random', 'shuffled_topology_FN_20_percent_by_random', 'shuffled_topology_FN_25_percent_by_random', 'shuffled_topology_FN_30_percent_by_random', 'shuffled_topology_FN_35_percent_by_random', 'shuffled_topology_FN_40_percent_by_random', 'shuffled_topology_FN_45_percent_by_random', 
                 'shuffled_topology_FN_50_percent_by_random']

dataset_code = os.path.basename(pkl_infile)[:10] 
plots = os.path.join(os.path.dirname(os.path.dirname(pkl_infile)), 'plots', dataset_code)
shift_set = int(pkl_infile.split('ms_shift')[0][-2:])

debugging = False
run_model_only = False


# if run_model_only:
#     model_infile  = pkl_outfile
#     base, ext = os.path.splitext(model_infile)
#     model_outfile = base + '_with_models_tmp' + ext

class params:
    
    use_preset_regParams = True
    alpha = 0.00001
    l1 = 0
    
    significant_proportion_thresh = 0.99
    numThresh = 100
    trainRatio = 0.8
    if debugging:
        num_model_samples = 2
    else:
        num_model_samples = 100
    
    primary_traj_model = 'traj_avgPos'
    best_lead_lag_key = 'lead_200_lag_300'
    lead_lag_keys_for_network = [best_lead_lag_key]
    FN_key = 'split_reach_FNs'
    encoding_model_for_trainPredictions = '%s_full_FN' % primary_traj_model
    
    transpose_FN = False
    minSpikeRatio = .005
    
    networkSampleBins = 2
    networkFeatureBins = 2 
    
    axis_fontsize = 24
    dpi = 300
    axis_linewidth = 2
    tick_length = 2
    tick_width = 1
    tick_fontsize = 18
    boxplot_figSize = (5.5, 5.5)
    aucScatter_figSize = (7, 7)
    FN_figSize = (7,7)
    map_figSize = (7, 7)
    plot_linewidth = 3
    
    channel_sep_horizontal = 0.4 # in mm

def shuffle_selected_inputs(inputs, shuffle_idxs, rng):    
    inputs[shuffle_idxs] = inputs[rng.permutation(shuffle_idxs)]
    return inputs

def modify_FN_weights_by_distance(weights, subset_dict):
    elec_dist = subset_dict['electrode_distances']
    if subset_dict['upper_bound'] is not None:
        weights[elec_dist > subset_dict['upper_bound']] = 0
    if subset_dict['lower_bound'] is not None:
        weights[elec_dist < subset_dict['lower_bound']] = 0
    
    return weights

def modify_FN_weights_by_tuning(weights, subset_dict):
    elec_dist = subset_dict['electrode_distances']
    unit_info_sorted = subset_dict['unit_info'].sort_values(by='proportion_sign', ascending = False)
    unit_info_sorted.reset_index(drop=False, inplace=True)
    unit_info_sorted.columns = ['original_index' if col == 'index' else col for col in unit_info_sorted.columns]
    if subset_dict['upper_bound'] is not None:
        print('This option has not yet been enabled. Add to this code in the function "modify_FN_weights_by_tuning"', flush=True)
        return
    if subset_dict['lower_bound'] is not None:
        if subset_dict['bound_type'] == 'proportion':
            pro_inputs_to_keep  = list(unit_info_sorted.loc[unit_info_sorted.proportion_sign >= subset_dict['lower_bound'], 'original_index'].values)
            num_to_keep = len(pro_inputs_to_keep)
            pro_inputs_to_keep.extend(list(unit_info_sorted.original_index.iloc[len(pro_inputs_to_keep):len(pro_inputs_to_keep)+10]))         
            anti_inputs_to_keep = list(unit_info_sorted.original_index.iloc[-len(pro_inputs_to_keep):].values)

            pro_weights  = weights.copy()
            anti_weights = weights.copy()

            seed_count = 0
            permutation_seed_count = 100
            for unit, (pro_inputs, anti_inputs, dist) in enumerate(zip(pro_weights, anti_weights, elec_dist)):
                unit_pro_to_keep  =  pro_inputs_to_keep.copy()
                unit_anti_to_keep = anti_inputs_to_keep.copy()
                
                unit_pro_to_keep  = [u for u in  unit_pro_to_keep if not np.isnan(dist[u])]
                unit_anti_to_keep = [u for u in unit_anti_to_keep if not np.isnan(dist[u])]
                unit_anti_to_keep = unit_anti_to_keep[::-1]

                pro_zero_count     = sum(dist[ unit_pro_to_keep[:num_to_keep]] == 0)
                pro_nonzero_count  = sum(dist[unit_pro_to_keep [:num_to_keep]] > 0) 
                anti_zero_count    = sum(dist[unit_anti_to_keep[:num_to_keep]] == 0)
                anti_nonzero_count = sum(dist[unit_anti_to_keep[:num_to_keep]] > 0)
                
                while any([pro_zero_count!=anti_zero_count, 
                           pro_nonzero_count!=anti_nonzero_count, 
                           pro_zero_count+pro_nonzero_count!=num_to_keep]):
                    
                    if pro_zero_count > anti_zero_count:
                        pro_zero_idxs = [idx for idx, u in enumerate(unit_pro_to_keep[:num_to_keep]) if dist[u] == 0]    
                        diff = pro_zero_count - anti_zero_count
                        drop_idxs = np.random.default_rng(seed_count).choice(pro_zero_idxs, size = diff, replace = False)
                        unit_pro_to_keep = [u for idx, u in enumerate(unit_pro_to_keep) if idx not in drop_idxs]
                        seed_count += 1
                   
                    elif anti_zero_count > pro_zero_count:
                        anti_zero_idxs = [idx for idx, u in enumerate(unit_anti_to_keep[:num_to_keep]) if dist[u] == 0]    
                        diff = anti_zero_count - pro_zero_count
                        drop_idxs = np.random.default_rng(seed_count).choice(anti_zero_idxs, size = diff, replace = False)
                        unit_anti_to_keep = [u for idx, u in enumerate(unit_anti_to_keep) if idx not in drop_idxs]
                        seed_count += 1 

                    pro_zero_count     = sum(dist[ unit_pro_to_keep[:num_to_keep]] == 0)
                    pro_nonzero_count  = sum(dist[unit_pro_to_keep [:num_to_keep]] > 0) 
                    anti_zero_count    = sum(dist[unit_anti_to_keep[:num_to_keep]] == 0)
                    anti_nonzero_count = sum(dist[unit_anti_to_keep[:num_to_keep]] > 0)                                                                
                                                                 
                unit_pro_to_keep  =  unit_pro_to_keep[:num_to_keep]
                unit_anti_to_keep = unit_anti_to_keep[:num_to_keep]
                
                pro_remove  = [u for u in range(weights.shape[1]) if u not in unit_pro_to_keep  and u != unit]
                anti_remove = [u for u in range(weights.shape[1]) if u not in unit_anti_to_keep and u != unit]
                
                if subset_dict['modify_method'] == 'remove':
                    pro_inputs [ pro_remove] = 0
                    anti_inputs[anti_remove] = 0
                elif subset_dict['modify_method'] == 'shuffle':
                    pro_inputs  = shuffle_selected_inputs( pro_inputs,  pro_remove, np.random.default_rng(permutation_seed_count))
                    anti_inputs = shuffle_selected_inputs(anti_inputs, anti_remove, np.random.default_rng(permutation_seed_count))
                    permutation_seed_count += 1
                
    return pro_weights, anti_weights

def get_num_inputs_by_area_for_all_units(weights, cortical_area, elec_dist):
    num_inputs_from_3b = []
    num_inputs_from_3a = []
    num_inputs_from_motor = []
    for unit, (inputs, dist) in enumerate(zip(weights.copy(), elec_dist)):
        inputs[dist == 0] = np.nan
        inputs[np.isnan(dist)] = np.nan
        not_nan_idxs = np.where(~np.isnan(inputs))[0] 

        input_idxs_from_3b = np.where(cortical_area == '3b')[0]
        input_idxs_from_3a = np.where(cortical_area == '3a')[0]
        input_idxs_from_motor = np.where((cortical_area == 'M1') | (cortical_area == '6Dc'))[0]
                
        num_inputs_from_3b.append   (np.intersect1d(input_idxs_from_3b   , not_nan_idxs).shape[0]) 
        num_inputs_from_3a.append   (np.intersect1d(input_idxs_from_3a   , not_nan_idxs).shape[0]) 
        num_inputs_from_motor.append(np.intersect1d(input_idxs_from_motor, not_nan_idxs).shape[0]) 
    
    area_inputs_counts_df = pd.DataFrame(data    = zip(num_inputs_from_3b, num_inputs_from_3a, num_inputs_from_motor),
                                         columns = ['3b', '3a', 'Motor'])
    
    return area_inputs_counts_df

def modify_FN_weights_by_cortical_area(weights, subset_dict):
    elec_dist = subset_dict['electrode_distances']
    cortical_area = subset_dict['cortical_area']
    model_names = [item for key, item in subset_dict.items() if 'model_name' in key]
    
    area_input_counts_df = get_num_inputs_by_area_for_all_units(weights, cortical_area, elec_dist)
    
    output_weights_list = [np.nan for idx in range(len(model_names))]
    seed_count = 500
    for model_idx, model_name in enumerate(model_names): 

        model_weights  = weights.copy()

        for unit, (inputs, dist) in enumerate(zip(model_weights, elec_dist)):
            # nonzero_dist_idxs = np.where(dist > 0)
            inputs[dist == 0] = np.nan
            inputs[np.isnan(dist)] = np.nan
            not_nan_idxs = np.where(~np.isnan(inputs))[0]
            
            if 'motor' in model_name:
                area_idxs = np.where((cortical_area == 'M1') | (cortical_area == '6Dc'))[0]
            elif 'sensory' in model_name:
                area_idxs = np.where((cortical_area == '3a') | (cortical_area == '3b' ))[0]
            elif '3a' in model_name:
                area_idxs = np.where(cortical_area == '3a')[0]
            elif '3b' in model_name:
                area_idxs = np.where(cortical_area == '3a')[0]
                
            idxs_to_modify = np.intersect1d(not_nan_idxs, area_idxs)
            
            if subset_dict['modify_method'] == 'remove':
                inputs[idxs_to_modify] = 0
            elif subset_dict['modify_method'] == 'shuffle':
                inputs  = shuffle_selected_inputs(inputs,  idxs_to_modify, np.random.default_rng(seed_count))
                inputs[np.isnan(inputs)] = 0
                
            seed_count += 1
        
        output_weights_list[model_idx] = model_weights
    
    return output_weights_list

def select_units_to_shuffle(weights, mode, percentile, seed_count):
    if mode == 'strength':
        shuffle_idxs = np.where(weights > np.percentile(weights, 100-percentile))
    elif mode == 'random':
        num_to_shuffle = np.where(weights > np.percentile(weights, 100-percentile))[0].size
        idx_pairs = []
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                if i != j:
                    idx_pairs.append((i, j))
        idx_choice_rng = np.random.default_rng(seed_count * 2)
        shuffle_idxs = idx_choice_rng.choice(idx_pairs, size = num_to_shuffle, replace = False)
        shuffle_idxs = (shuffle_idxs[:, 0], shuffle_idxs[:, 1])
    else:
        raise Exception('\n "%s" mode for selecting units to shuffle has not been implemented\n\n' % mode)
    
    weights_at_idxs = np.full_like(shuffle_idxs[0], np.nan, dtype=weights.dtype)
    for idx, (target_unit, source_unit) in enumerate(zip(shuffle_idxs[0], shuffle_idxs[1])):
        weights_at_idxs[idx] = weights[target_unit, source_unit]        
    
    shuffle_set = (shuffle_idxs[0], shuffle_idxs[1], weights_at_idxs)
    
    return shuffle_set
def shuffle_selected_network_idxs(weights, shuffle_set, shuffle_edges_mode, seed_count):
    
    target = shuffle_set[0].copy()
    source = shuffle_set[1].copy()
    wji    = shuffle_set[2].copy()

    rng = np.random.default_rng(seed_count)
    
    if shuffle_edges_mode   == 'weights':
        wji_shuf    = rng.permutation(wji)
        weights[target, source] =  wji_shuf
    elif shuffle_edges_mode == 'topology':
        target_shuf = rng.permutation(target)
        self_edges_idx = np.where(target_shuf == source)[0]
        for count, idx in enumerate(self_edges_idx):
            self_unit = target_shuf[idx]
            possible_new_idx = np.where((source != self_unit) & (target_shuf != self_unit))[0]
            rng_swap = np.random.default_rng(seed_count+count)
            swap_idx = rng_swap.choice(possible_new_idx, 1)[0]
            target_shuf[[idx, swap_idx]] = target_shuf[[swap_idx, idx]]            

        # original_weights = weights.copy()        

        for idx, (targ, targ_shuf, sour) in enumerate(zip(target, target_shuf, source)):
            if targ == targ_shuf:
                continue
            
            weights[[targ, targ_shuf], sour] = weights[[targ_shuf, targ], sour]
            # print((idx, weights.sum(), np.sum(weights!=original_weights)))
            
    else:
        raise Exception('\n "%s" mode for shuffling weights has not been implemented\n\n' % shuffle_edges_mode)
    
    return weights
    
def shuffle_FN_by_strength(weights, subset_dict):
    
    store_model_names = False
    if subset_dict['model_name'] == 'tmp':
        store_model_names = True
        model_names = []
    
    seed_count = 25
    output_weights_list = []
    for mode in subset_dict['mode']:
        for shuffle_edges_mode in subset_dict['edges_to_shuffle']:
            for percent in subset_dict['percents']:
                if percent is None:
                    rng = np.random.default_rng(seed_count)
                    if subset_dict['edges_to_shuffle'] == 'weights':
                        shuf_weights = weights.copy()
                        rng.shuffle(shuf_weights, axis = 1)
                    elif subset_dict['edges_to_shuffle'] == 'topology':
                        print('topology method not yet implemented for full_network shuffle', flush=True)
                else:
                    shuffle_set  = select_units_to_shuffle(weights, mode, percent, seed_count)
                    
                    shuf_weights = shuffle_selected_network_idxs(weights.copy(), shuffle_set, shuffle_edges_mode, seed_count)
                
                if percent is None:
                    percent = 100
                if store_model_names:
                    model_names.append('shuffled_%s_FN_%d_percent_by_%s' % (shuffle_edges_mode, percent, mode))
                
                output_weights_list.append(shuf_weights.copy())
                
                seed_count += 1
                
    if store_model_names:
        subset_dict['model_name'] = model_names
    
    return output_weights_list

def modify_FN_weights(weights, subset_dict):
    if subset_dict['mode'] == 'distance':
        weights_out = modify_FN_weights_by_distance(weights.copy(), subset_dict)
    elif subset_dict['mode'] == 'tuning':
        pro_weights, anti_weights = modify_FN_weights_by_tuning(weights.copy(), subset_dict)
        weights_out = [pro_weights, anti_weights]
    elif subset_dict['mode'] == 'cortical_area':
        weights_out = modify_FN_weights_by_cortical_area(weights.copy(), subset_dict)
    elif 'strength' in subset_dict['mode']:
        weights_out = shuffle_FN_by_strength(weights.copy(), subset_dict)
    elif subset_dict['mode'] == 'original':
        weights_out = [weights.copy()]

    if type(weights_out) != list:
        weights_out = [weights_out]
        
    return weights_out

def apply_standard_scaler(samples, mode):
    
    scaled_samples = [[] for i in range(len(samples))]

    if mode == 'pca_features':
        scaler = StandardScaler()
        scaled_samples = scaler.fit_transform(samples)
    elif mode == 'network':
        scaled_samples = np.full_like(samples, np.nan)
        for unit, unit_samples in enumerate(samples):
            scaler = StandardScaler()
            scaled_samples[unit] = scaler.fit_transform(unit_samples)
    else:
        for mark, kin in enumerate(samples):
            scaler = StandardScaler()
        
            if mode == 'traj':
                kin_reshaped = kin.transpose(0, 2, 1).reshape(-1, 3, order='C')
                kin_scaled   = scaler.fit_transform(kin_reshaped)        
                kin_scaled   = kin_scaled.reshape(kin.shape[0], 
                                                  kin.shape[2], 
                                                  kin.shape[1], order='C').transpose(0, 2, 1) 
            elif mode == 'avg_kin':
                kin_scaled = scaler.fit_transform(kin)
            
            scaled_samples[mark] = kin_scaled
                
    return scaled_samples

def create_network_features_and_store_in_dict(FN, subset_dict, lead_lag_key, reach_set_df, FN_key = 'split_reach_FNs'):
    
    model_features_dict = results_dict[lead_lag_key]['model_features']        
    model_names = [item for key, item in subset_dict.items() if 'model_name' in key]
    
    if all([name in model_features_dict.keys() for name in model_names]):
        return [], []
        
    spike_samples = results_dict[lead_lag_key]['sampled_data']['spike_samples']
    sample_info   = results_dict[lead_lag_key]['sampled_data']['sample_info']
    
    network_features = [np.full((spike_samples.shape[0], spike_samples.shape[1], params.networkFeatureBins), np.nan) for i in range(subset_dict['n_models'])]
    split_weights = []
    if 'split' in FN_key:
        for tmp_FN in FN:
            tmp_weights = tmp_FN.copy()
            if params.transpose_FN:
                tmp_weights = tmp_weights.T
            tmp_weights_modified = modify_FN_weights(tmp_weights.copy(), subset_dict)
            split_weights.append(tmp_weights_modified)

        for sampleNum, reach_idx in enumerate(sample_info['reach_idx']):
            if sampleNum % 500 == 0:
                print(sampleNum, flush=True)
            if debugging:
                if sampleNum > 2:
                    break
            
            FN_reach_source = reach_set_df.loc[reach_set_df['reach_num'] == reach_idx, 'FN_reach_set'].values[0]
            if FN_reach_source == 2:
                weights = split_weights[0]
            else:
                weights = split_weights[1]
            
            for wIdx, w_arr in enumerate(weights):
                for leadBin in range(params.networkFeatureBins):
                    network_features[wIdx][:, sampleNum, leadBin] = w_arr @ spike_samples[:, sampleNum, (params.networkSampleBins-1) - leadBin] 

    else:
        weights = FN.copy()
        if params.transpose_FN:
            weights = weights.T
        weights_modified = modify_FN_weights(weights.copy(), subset_dict)

        for wIdx, w_arr in enumerate(weights_modified):
            for leadBin in range(params.networkFeatureBins):
                network_features[wIdx][..., leadBin] = w_arr @ spike_samples[..., (params.networkSampleBins-1) - leadBin]         

    model_names = [item for key, item in subset_dict.items() if 'model_name' in key]
    if type(model_names[0]) == list and len(model_names) == 1:
        model_names = model_names[0]
    
    if len(network_features) != len(model_names):
        raise Exception("The length of network features (should be 1 per model) does not match the length of the model_names list.")

    if all([name in model_features_dict.keys() for name in model_names]):
        return [], []
    for model_name, features in zip(model_names, network_features):
        features = apply_standard_scaler(features, mode = 'network')

        model_features_dict[model_name] = features
    
    return network_features, model_names

def compute_area_under_ROC(predictions, testSpikes):
    single_sample_auc = np.empty((len(predictions),)) 
    allHitProbs = []
    allFalsePosProbs = []
    for unit, preds in enumerate(predictions):
        thresholds = np.linspace(preds.min(), preds.max(), params.numThresh)            
        hitProb = np.empty((len(thresholds),))
        falsePosProb = np.empty((len(thresholds),))
        for t, thresh in enumerate(thresholds):    
            posIdx = np.where(preds > thresh)
            hitProb[t] = np.sum(testSpikes[unit][posIdx] >= 1) / np.sum(testSpikes[unit] >= 1)
            falsePosProb[t] = np.sum(testSpikes[unit][posIdx] == 0) / np.sum(testSpikes[unit] == 0)
        
        single_sample_auc[unit] = auc(falsePosProb, hitProb)
        
        allHitProbs.append(hitProb)
        allFalsePosProbs.append(falsePosProb)
        
    return single_sample_auc, allHitProbs, allFalsePosProbs

def train_and_test_glm(traj_features, network_features, spike_samples, model_name, training_glm_source, RNGs, lead_lag_key, alpha, l1, retrain=True):   
    
    areaUnderROC = np.empty((spike_samples.shape[0], params.num_model_samples))
    trainingSet_areaUnderROC = np.full((spike_samples.shape[0], params.num_model_samples), np.nan)

    aic          = np.empty_like(areaUnderROC)
    loglike = np.empty_like(aic)
    coefs = np.empty((traj_features.shape[-1] + network_features.shape[-1] + 1, spike_samples.shape[0], params.num_model_samples))
    pVals = np.empty_like(coefs)  
    
    allTrainPredictions = []
    for n, split_rng in enumerate(RNGs['train_test_split']):

        # Create train/test datasets for cross-validation
        print(lead_lag_key + ', ' + model_name + ', iteration = ' +str(n), flush=True)
        testSpikes = []
        trainSpikes = []
        trainFeatures = []
        testFeatures = []
        for unit, spikes in enumerate(spike_samples[..., -1]):
            
            spikeIdxs   = np.where(spikes >= 1)[0]
            noSpikeIdxs = np.where(spikes == 0)[0]
                
            idxs = np.union1d(spikeIdxs, noSpikeIdxs)
            trainIdx = np.hstack((split_rng.choice(spikeIdxs  , size = round(params.trainRatio*len(spikeIdxs  )), replace = False), 
                                  split_rng.choice(noSpikeIdxs, size = round(params.trainRatio*len(noSpikeIdxs)), replace = False)))
            testIdx  = np.setdiff1d(idxs, trainIdx)
                
            if np.sum(spikes[testIdx] >= 1) / np.sum(spikes[testIdx] == 0) >= params.minSpikeRatio:
                trainSpikes.append(spikes[trainIdx])
                testSpikes.append(spikes[testIdx])
                
                if traj_features.shape[0] == 0:
                    trainFeatures.append(network_features[unit, trainIdx])
                    testFeatures.append (network_features[unit, testIdx ])                 
                elif network_features.shape[0] == 0:
                    trainFeatures.append(traj_features[trainIdx])
                    testFeatures.append (traj_features[testIdx])                    
                else:
                    trainFeatures.append(np.hstack((traj_features[trainIdx], network_features[unit, trainIdx])))
                    testFeatures.append (np.hstack((traj_features[testIdx ], network_features[unit, testIdx ]))) 

       
            else:
                if n == 0:
                    print('unit ' + str(unit) + ' had only ' + str(np.sum(spikes[testIdx] >= 1)) + ' spikes in the sampled time windows and is removed from analysis', flush=True)
            
        # Train GLM
        predictions = []
        trainPredictions = []
        for unit, (trainSpks, trainFts, testFts, shuf_rng) in enumerate(zip(trainSpikes, trainFeatures, testFeatures, RNGs['spike_shuffle'])):
            if 'shuffle' in model_name:
                trainSpks = shuf_rng.permutation(trainSpks)
            if retrain:
                glm = sm.GLM(trainSpks,
                             sm.add_constant(trainFts), 
                             family=sm.families.Poisson(link=sm.families.links.log()))
                encodingModel = glm.fit_regularized(method='elastic_net', alpha=alpha, L1_wt=l1)
            
                coefs  [:, unit, n] = encodingModel.params            
                # pVals  [:, unit, n] = np.round(encodingModel.pvalues, decimals = 4)            
                # aic    [   unit, n] = round(encodingModel.aic, 1)
                # loglike[   unit, n] = round(encodingModel.llf, 1)
            
                if model_name == params.encoding_model_for_trainPredictions:
                    model_filepath = os.path.join(models_storage_folder, '%s_sample_%s_unit_%s_encoding_model.pkl' % (model_name, str(n).zfill(4), str(unit).zfill(3)))
                    with open(model_filepath, 'wb') as f:
                        dill.dump(encodingModel, f, recurse=True)
                    
                predictions.append(encodingModel.predict(sm.add_constant(testFts)))
            
            if training_glm_source:
                if model_name == params.encoding_model_for_trainPredictions:
                    stored_encoding_model = encodingModel 
                else:   
                    model_filepath = os.path.join(models_storage_folder, '%s_sample_%s_unit_%s_encoding_model.pkl' % (params.encoding_model_for_trainPredictions, str(n).zfill(4), str(unit).zfill(3)))
                    with open(model_filepath, 'rb') as f:
                        stored_encoding_model = dill.load(f) 

                trainPredictions.append(stored_encoding_model.predict(sm.add_constant(trainFts))) 

                
        # Test GLM --> area under ROC
        if retrain: 
            areaUnderROC[:, n], allHitProbs, allFalsePosProbs = compute_area_under_ROC(predictions, testSpikes)

        # Get AUC on training data
        if training_glm_source:
            trainingSet_areaUnderROC[:, n], _, _ = compute_area_under_ROC(trainPredictions, trainSpikes)
            
    if np.all(np.isnan(trainingSet_areaUnderROC)):
        trainingSet_areaUnderROC = []
    
    description = '''Model = %s. 
    To understand dimensions: 
        %d units,
        %d model parameters (1 constant, %d kinematic features, %d network features)
        %d shuffles of the train/test split. 
    The order of model parameters is: 
        Constant term
        Trajectory features projected onto principal components 
        Average position terms (3 terms, x/y/z, if model includes average position)
        Average speed terms if included in model
        Network Features
    The keys hold the following information:
        AUC: area under the ROC curve on %d%% held-out test data
        trainAUC: area under the ROC curve on %d%% train data. The encoding model used is pulled from the %s model.
        AIC: AIC criterion value for trained model on training data
        coefs: the parameter coefficents that have been fit to the input features  
        pvals: the p-values describing the significance of the parameter coefficients
        logLikelihood: the output of encodingModel.llf, the log likelihood of the model,
        all_predictions_on_training_set: list of length = num_samples. Within each element is a list of length = numUnits.
                                         Each element in this list contains an array of predicted spike activity on the training dataset.
        encoding_models: list of encodingModel GLMs for each train/test split sample (first level of list) and each unit (second level)
    If this is a shuffled model, that means the spike samples were shuffled to eliminate the relationship
    between model features and spiking. A new spike_samples shuffle was performed for each train/test split. 
    For network used to create network features for this model was "%s".
    ''' % (model_name, areaUnderROC.shape[0], coefs.shape[0], traj_features.shape[-1], network_features.shape[-1], 
    params.num_model_samples, int((1-params.trainRatio)*1e2),  int((params.trainRatio)*1e2), training_glm_source,
    params.FN_key)
    
    model_results = {'AUC'                            : areaUnderROC,
                     'trainAUC'                       : trainingSet_areaUnderROC,
                     'coefs'                          : coefs,
                     'all_predictions_on_training_set': allTrainPredictions,
                     'description'                    : description,
                     'alpha'                          : alpha,
                     'l1'                             : l1}    
    
    return model_results

if __name__ == "__main__":
    
    print('\n\n Began creating models at %s\n\n' % time.strftime('%c', time.localtime()), flush=True)
    
    # if debugging:
    #     task_id = 0
    #     n_tasks = 1
    # else:
    #     task_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
    #     n_tasks = int(os.getenv('SLURM_ARRAY_TASK_COUNT')) 
    
    # if n_tasks != len(params.lead_lag_keys_for_network) * params.num_models_including_shuffles * len(params.sample_ranges) and not debugging:
    #     print('number of jobs in array does not equal length of leads or lags to be tested')
    #     print('ending job', flush=True)
    
    os.makedirs(models_storage_folder, exist_ok=True)
    
    with open(pkl_infile, 'rb') as f:
        results_dict = dill.load(f)
    
    if not run_model_only:
        
        with NWBHDF5IO(nwb_infile, 'r') as io:
            nwb = io.read()

            FN = nwb.scratch[params.FN_key].data[:]
            spontaneous_FN = nwb.scratch['spontaneous_FN'].data[:]
            reach_set_df = nwb.scratch['split_FNs_reach_sets'].to_dataframe()
                
        units_res = results_dict[params.best_lead_lag_key]['all_models_summary_results']
        electrode_distances = get_interelectrode_distances_by_unit(units_res, array_type='utah')
        
        RNGs = {'train_test_split'      : [np.random.default_rng(n) for n in range(params.num_model_samples)],
                'partial_traj'          : [np.random.default_rng(n) for n in range(1000,  1000+params.num_model_samples)],
                'spike_shuffle'         : [np.random.default_rng(n) for n in range(5000,  5000+results_dict[params.best_lead_lag_key]['sampled_data']['spike_samples'].shape[0])],
                'weight_shuffle'        : [np.random.default_rng(n) for n in range(10000, 10000+params.num_model_samples)],
                'network_input_shuffles': [[np.random.default_rng(n) for n in range(unit_idx*2000,  unit_idx*2000+params.num_model_samples)] for unit_idx in range(units_res.shape[0])]}

        all_subset_dict = {'full_FN'        :     {'model_name'          : 'full_FN',
                                                   'mode'                : 'original',
                                                   'n_models'            : 1,
                                                   'glm_source'          : params.encoding_model_for_trainPredictions},
                           'spontaneous_FN' :     {'model_name'          : 'spontaneous_FN',
                                                   'mode'                : 'original',
                                                   'n_models'            : 1,
                                                   'FN'                  : spontaneous_FN.copy(),
                                                   'FN_key'              : 'spontaneous_FN',
                                                   'glm_source'          : params.encoding_model_for_trainPredictions},
                           'strength'       :     {'model_name'          : 'tmp',
                                                   'mode'                : ['strength', 'random'],
                                                   'edges_to_shuffle'    : ['weights','topology'],
                                                   'percents'            : [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99],
                                                   'n_models'            : 84,
                                                   'retrain_model'       : False,
                                                   'glm_source'          : params.encoding_model_for_trainPredictions}}

        # all_subset_dict = {'full_FN'        :     {'model_name'          : 'full_FN',
        #                                            'mode'                : 'full',
        #                                            'n_models'            : 1,
        #                                            'glm_source'          : params.encoding_model_for_trainPredictions},
        #                    'zero_dist_FN'   :     {'model_name'          : 'zero_dist_FN',
        #                                            'mode'                : 'distance',
        #                                            'n_models'            : 1,
        #                                            'modify_method'       : 'remove',
        #                                            'upper_bound'         : 1,
        #                                            'lower_bound'         : None,
        #                                            'electrode_distances' : electrode_distances,
        #                                            'glm_source'          : params.encoding_model_for_trainPredictions},
        #                    'nonzero_dist_FN':     {'model_name'          : 'nonzero_dist_FN',
        #                                            'mode'                : 'distance',
        #                                            'n_models'            : 1,
        #                                            'modify_method'       : 'remove',
        #                                            'upper_bound'         : None,
        #                                            'lower_bound'         : 1,
        #                                            'electrode_distances' : electrode_distances,
        #                                            'glm_source'          : params.encoding_model_for_trainPredictions},
        #                    'strength'       :     {'model_name'          : 'tmp',
        #                                            'mode'                : ['strength', 'random'],
        #                                            'edges_to_shuffle'    : ['weights','topology'],
        #                                            'percents'            : [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        #                                            'retrain_model'       : False,
        #                                            'glm_source'          : params.encoding_model_for_trainPredictions}}
                
        for lead_lag_key in params.lead_lag_keys_for_network:
            
            for model_key in remove_models:
                tmp_results_key = '%s_%s' % (params.primary_traj_model, model_key)
                del results_dict[lead_lag_key]['model_features'][model_key]
                del results_dict[lead_lag_key]['model_results'][tmp_results_key] 
            
            for model_key in all_subset_dict.keys():                    

                if 'FN' in all_subset_dict[model_key].keys():
                    FN_for_features     = all_subset_dict[model_key]['FN']
                    FN_key_for_features = all_subset_dict[model_key]['FN_key']
                else:
                    FN_for_features     = FN.copy()
                    FN_key_for_features = params.FN_key

                network_features_list, model_names = create_network_features_and_store_in_dict(FN_for_features, 
                                                                                               all_subset_dict[model_key],
                                                                                               lead_lag_key=lead_lag_key,
                                                                                               reach_set_df=reach_set_df,
                                                                                               FN_key = FN_key_for_features)
                
                for network_features, model_name in zip(network_features_list, model_names):
                    model_results_key = '%s_%s' % (params.primary_traj_model, model_name)
                    if params.use_preset_regParams:
                        alpha = params.alpha
                        l1 = params.l1
                    else:
                        alpha = results_dict[lead_lag_key]['model_results'][params.primary_traj_model]['alpha']
                        l1    = results_dict[lead_lag_key]['model_results'][params.primary_traj_model]['l1']
                    if model_results_key in results_dict[lead_lag_key]['model_results'].keys():
                        continue
                    
                    try:   
                        retrain_and_test_model = all_subset_dict[model_key]['retrain_model'] 
                    except:
                        retrain_and_test_model = True
                    
                    model_results = train_and_test_glm(results_dict[lead_lag_key]['model_features'][params.primary_traj_model], 
                                                       network_features, 
                                                       results_dict[lead_lag_key]['sampled_data']['spike_samples'], 
                                                       model_results_key,
                                                       all_subset_dict[model_key]['glm_source'],
                                                       RNGs,
                                                       lead_lag_key,
                                                       alpha,
                                                       l1,
                                                       retrain = retrain_and_test_model)
                    
                    results_dict[lead_lag_key]['model_results'][model_results_key] = model_results
                    
                    with open(pkl_outfile, 'wb') as f:
                        dill.dump(results_dict, f, recurse=True) 
                
                # run full_FN_only model
                if 'full_FN' in model_names:
                    model_results_key = 'full_FN'
                    model_idx = [idx for idx, name in enumerate(model_names) if name == model_results_key][0]
                    model_results = train_and_test_glm(np.array(()), 
                                                       network_features_list[0], 
                                                       results_dict[lead_lag_key]['sampled_data']['spike_samples'], 
                                                       model_results_key, 
                                                       False,
                                                       RNGs,
                                                       lead_lag_key,
                                                       alpha,
                                                       l1)
                    
                    results_dict[lead_lag_key]['model_results'][model_results_key] = model_results
                    
                    with open(pkl_outfile, 'wb') as f:
                        dill.dump(results_dict, f, recurse=True) 

                if 'spontaneous_FN' in model_names:
                    model_results_key = 'spontaneous_FN'
                    model_idx = [idx for idx, name in enumerate(model_names) if name == model_results_key][0]
                    model_results = train_and_test_glm(np.array(()), 
                                                       network_features_list[0], 
                                                       results_dict[lead_lag_key]['sampled_data']['spike_samples'], 
                                                       model_results_key, 
                                                       False,
                                                       RNGs,
                                                       lead_lag_key,
                                                       alpha,
                                                       l1)
                    
                    results_dict[lead_lag_key]['model_results'][model_results_key] = model_results
                    
                    with open(pkl_outfile, 'wb') as f:
                        dill.dump(results_dict, f, recurse=True)                                     
        
        with open(pkl_outfile, 'wb') as f:
            dill.dump(results_dict, f, recurse=True)  
            
    print('\n\n Finished creating models at %s\n\n' % time.strftime('%c', time.localtime()), flush=True)

'''
    To do:
        - Make entry and model in subset_dict for inter- vs intra-area connections
        - Make entry and model in subset_dict for narrow vs wide-spiking units
        - Re-run "tuning_of_inputs" with percentile bound in instead of significance bound, or with higher significance bound (used 0.95 previously which has overlapping populations of tuned/non-tuned)
'''
        
