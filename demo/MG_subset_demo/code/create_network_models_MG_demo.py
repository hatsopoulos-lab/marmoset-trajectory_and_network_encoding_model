#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 08:00:51 2020

@author: daltonm
"""
#%matplotlib notebook
 
import numpy as np
import pandas as pd
import dill
import os
import time
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc
import statsmodels.api as sm
from importlib import sys
from pathlib import Path

from pynwb import NWBHDF5IO
import ndx_pose

script_directory = Path(os.path.dirname(os.path.abspath(sys.argv[0])))
code_path = script_directory.parent.parent.parent / 'clean_final_analysis/'
data_path = script_directory.parent.parent / 'data' / 'demo'

sys.path.insert(0, str(code_path))
from utils import get_interelectrode_distances_by_unit, load_dict_from_hdf5, save_dict_to_hdf5

pkl_in_tag  = 'kinematic_models_summarized'
pkl_out_tag = 'network_models_created' 

show_plots=False
marmcode = 'MG'
debugging = False
demo = True
shuffles_only = False
no_shuffles = False

if marmcode=='TY':
    nwb_infile   = data_path / 'TY' / 'TY20210211_freeAndMoths-003_resorted_20230612_DM_with_functional_networks.nwb' 
    models_already_stored = False
elif marmcode=='MG':
    nwb_infile   = data_path / 'MG' / 'MG20230416_1505_mothsAndFree-002_processed_DM_with_functional_networks.nwb'
    models_already_stored = False

pkl_infile   = nwb_infile.parent / f'{nwb_infile.stem.split("_with_functional_networks")[0]}_{pkl_in_tag}.pkl'
pkl_outfile  = nwb_infile.parent / f'{nwb_infile.stem.split("_with_functional_networks")[0]}_{pkl_out_tag}.pkl'

tmp_job_array_folder  = pkl_outfile.parent / 'network_jobs_tmp_files' / f'{pkl_outfile.stem}'          
models_storage_folder = pkl_outfile.parent / 'stored_encoding_models' / f'{pkl_outfile.stem}'
os.makedirs(models_storage_folder, exist_ok=True)
os.makedirs(tmp_job_array_folder, exist_ok=True) 

remove_models = []

class params:
    
    alpha = 1e-6#1e-7#5e-6
    l1 = 0
    
    numThresh = 100
    if debugging:
        num_model_samples = 2
    elif demo:
        num_model_samples = 5
    else:
        num_model_samples = 500
    
    primary_traj_model = 'traj_avgPos'
    if marmcode == 'TY':
        lead_lag_keys_for_network = ['lead_100_lag_300']
        intra_inter_areas_list = [['Motor'], ['Sensory']]
        intra_inter_names_list = ['Motor', 'Sensory']
        trainRatio = 0.8

    elif marmcode == 'MG':
        lead_lag_keys_for_network = ['lead_100_lag_300']
        intra_inter_areas_list = [['Motor'], ['Sensory']]
        intra_inter_names_list = ['Motor', 'Sensory']
        trainRatio = 0.8
        
    reach_FN_key = 'split_reach_FNs'
    encoding_model_for_trainPredictions = '%s_reach_FN' % primary_traj_model
    
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

def modify_FN_weights_by_distance(weights, task_info):
    elec_dist = task_info['electrode_distances']
    if task_info['upper_bound'] is not None:
        weights[elec_dist > task_info['upper_bound']] = 0
    if task_info['lower_bound'] is not None:
        weights[elec_dist < task_info['lower_bound']] = 0
    
    return weights

def modify_FN_weights_by_tuning(weights, task_info):
    elec_dist = task_info['electrode_distances']
    unit_info_sorted = task_info['unit_info'].sort_values(by='proportion_sign', ascending = False)
    unit_info_sorted.reset_index(drop=False, inplace=True)
    unit_info_sorted.columns = ['original_index' if col == 'index' else col for col in unit_info_sorted.columns]
    if task_info['upper_bound'] is not None:
        print('This option has not yet been enabled. Add to this code in the function "modify_FN_weights_by_tuning"', flush=True)
        return
    if task_info['lower_bound'] is not None:
        if task_info['bound_type'] == 'proportion':
            pro_inputs_to_keep  = list(unit_info_sorted.loc[unit_info_sorted.proportion_sign >= task_info['lower_bound'], 'original_index'].values)
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
                
                if task_info['modify_method'] == 'remove':
                    pro_inputs [ pro_remove] = 0
                    anti_inputs[anti_remove] = 0
                elif task_info['modify_method'] == 'shuffle':
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


def modify_FN_weights_by_cortical_area_OLD(weights, task_info):
    elec_dist = task_info['electrode_distances']
    cortical_area = task_info['cortical_area']
    model_names = [item for key, item in task_info.items() if 'model_name' in key]
    
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
            
            if task_info['modify_method'] == 'remove':
                inputs[idxs_to_modify] = 0
            elif task_info['modify_method'] == 'shuffle':
                inputs  = shuffle_selected_inputs(inputs,  idxs_to_modify, np.random.default_rng(seed_count))
                inputs[np.isnan(inputs)] = 0
                
            seed_count += 1
        
        output_weights_list[model_idx] = model_weights
    
    return output_weights_list

def select_units_to_shuffle(weights, mode, percent, seed_count, mod_name):
    if mode == 'strength':
        shuffle_idxs = np.where(weights >= np.percentile(weights, 100-percent))
    elif mode == 'random':
        num_to_shuffle = np.where(weights >= np.percentile(weights, 100-percent))[0].size
        idx_pairs = []
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                if i != j or 'inter' in mod_name:
                    idx_pairs.append((i, j))
        idx_choice_rng = np.random.default_rng(seed_count * 2)

        shuffle_idxs = idx_choice_rng.choice(idx_pairs, size = np.min([num_to_shuffle, np.shape(idx_pairs)[0]]), replace = False)
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

        for idx, (targ, targ_shuf, sour) in enumerate(zip(target, target_shuf, source)):
            if targ == targ_shuf:
                continue
            
            weights[[targ, targ_shuf], sour] = weights[[targ_shuf, targ], sour]            
    else:
        raise Exception('\n "%s" mode for shuffling weights has not been implemented\n\n' % shuffle_edges_mode)
    
    return weights

def shuffle_FN_by_strength(weights, task_info, target_idxs = None, source_idxs = None):
    
    original_mod_name = task_info.copy()['model_name']
    FN_key   = task_info['train_FN_key']
    
    store_model_names = False
    if 'tmp' in original_mod_name:
        store_model_names = True
        model_names = []
    
    seed_count = 25
    output_weights_list = []
    for mode in task_info['shuffle_mode']:
        for shuffle_edges_mode in task_info['edges_to_shuffle']:
            for percent in task_info['percents']:
                if percent is None:
                    rng = np.random.default_rng(seed_count)
                    if task_info['edges_to_shuffle'] == 'weights':
                        shuf_weights = weights.copy()
                        rng.shuffle(shuf_weights, axis = 1)
                    elif task_info['edges_to_shuffle'] == 'topology':
                        print('topology method not yet implemented for full_network shuffle', flush=True)
                else:
                    if target_idxs is not None:
                        shuf_weights = np.full_like(weights, 0)
                        for target, source in zip(target_idxs, source_idxs): 
                            subgraph_weights = weights.copy()[np.ix_(target, source)]
                            shuffle_set  = select_units_to_shuffle(subgraph_weights, mode, percent, seed_count, original_mod_name)
                            shuf_weights[np.ix_(target, source)] = shuffle_selected_network_idxs(subgraph_weights.copy(), shuffle_set, shuffle_edges_mode, seed_count)
                            seed_count += 1
                    else:
                        shuffle_set  = select_units_to_shuffle(weights, mode, percent, seed_count, original_mod_name)
                        shuf_weights = shuffle_selected_network_idxs(weights.copy(), shuffle_set, shuffle_edges_mode, seed_count)
                
                if percent is None:
                    percent = 100
                if store_model_names:
                    if original_mod_name == 'tmp':   
                        tmp_name = '%s_shuffled_%s_%d_percent_by_%s' % (FN_key, shuffle_edges_mode, percent, mode)
                    else:
                        tmp_name = '%s_%s_shuffled_%s_%d_percent_by_%s' % (original_mod_name.split('_tmp')[0], FN_key, shuffle_edges_mode, percent, mode)
                    model_names.append(tmp_name)
                
                output_weights_list.append(shuf_weights.copy())
                
                seed_count += 1
                
    if store_model_names:
        task_info['model_names'] = model_names
    
    return output_weights_list

def modify_FN_weights(weights, task_info, target_idxs = None, source_idxs = None):
    if task_info['shuffle_mode'] == 'distance':
        weights_out = modify_FN_weights_by_distance(weights.copy(), task_info)
    elif task_info['shuffle_mode'] == 'tuning':
        pro_weights, anti_weights = modify_FN_weights_by_tuning(weights.copy(), task_info)
        weights_out = [pro_weights, anti_weights]
    elif task_info['shuffle_mode'] == 'cortical_area':
        weights_out = modify_FN_weights_by_cortical_area_OLD(weights.copy(), task_info)
    elif 'strength' in task_info['shuffle_mode']:
        weights_out = shuffle_FN_by_strength(weights.copy(), task_info, target_idxs, source_idxs)
    elif task_info['shuffle_mode'] == 'unaltered':
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

def compute_network_features(spike_samples, sample_info, FN, FN_key, models_per_set, task_info, cortical_area_idxs = None):

    mod_name = task_info['model_name']
    if 'intra' in mod_name:
        if marmcode == 'TY':
            target_idxs = [cortical_area_idxs['motor'], cortical_area_idxs['3a'], cortical_area_idxs['3b']]
            source_idxs = [cortical_area_idxs['motor'], cortical_area_idxs['3a'], cortical_area_idxs['3b']]
        elif marmcode == 'MG':
            target_idxs = [cortical_area_idxs['motor'], cortical_area_idxs['sensory']]
            source_idxs = [cortical_area_idxs['motor'], cortical_area_idxs['sensory']]    
        subgraphs = True
    elif 'inter' in mod_name:
        if marmcode == 'TY':
            target_idxs = [cortical_area_idxs['motor']  , cortical_area_idxs['3a']          , cortical_area_idxs['3b']]
            source_idxs = [cortical_area_idxs['sensory'], cortical_area_idxs['3b_and_motor'], cortical_area_idxs['3a_and_motor']]
        elif marmcode == 'MG':
            target_idxs = [cortical_area_idxs['motor']  , cortical_area_idxs['sensory']]
            source_idxs = [cortical_area_idxs['sensory'], cortical_area_idxs['motor']]  
        subgraphs = True
    else:
        target_idxs = None
        source_idxs = None
        subgraphs = False

    network_features = [np.full((spike_samples.shape[0], spike_samples.shape[1], params.networkFeatureBins), np.nan) for i in range(models_per_set)]

    if FN_key == 'reach_FN' and 'split' in params.reach_FN_key:
        split_weights = []
        for tmp_FN in FN:
            tmp_weights = tmp_FN.copy()
            if params.transpose_FN:
                tmp_weights = tmp_weights.T
            tmp_weights_modified = modify_FN_weights(tmp_weights.copy(), task_info, target_idxs, source_idxs)
            split_weights.append(tmp_weights_modified)

        for sampleNum, reach_idx in enumerate(sample_info['reach_idx']):
            if sampleNum % 500 == 0:
                print(sampleNum, flush=True)
            
            FN_reach_source = reach_set_df.loc[reach_set_df['reach_num'] == reach_idx, 'FN_reach_set'].values[0]
            if FN_reach_source == 2:
                weights = split_weights[0]
            else:
                weights = split_weights[1]
            
            for wIdx, w_arr in enumerate(weights):
                for leadBin in range(params.networkFeatureBins):
                    if subgraphs:
                        for target, source in zip(target_idxs, source_idxs): 
                            network_features[wIdx][target, sampleNum, leadBin] = w_arr[np.ix_(target, source)] @ spike_samples[source, sampleNum, (params.networkSampleBins-1) - leadBin] 
                    else:
                        network_features[wIdx][:, sampleNum, leadBin] = w_arr @ spike_samples[:, sampleNum, (params.networkSampleBins-1) - leadBin] 

    else:
        weights = FN.copy()
        if params.transpose_FN:
            weights = weights.T
        weights_modified = modify_FN_weights(weights.copy(), task_info, target_idxs, source_idxs)

        for wIdx, w_arr in enumerate(weights_modified):
            for leadBin in range(params.networkFeatureBins):
                if subgraphs:
                    for target, source in zip(target_idxs, source_idxs): 
                        network_features[wIdx][target, :, leadBin] = w_arr[np.ix_(target, source)] @ spike_samples[source, :, (params.networkSampleBins-1) - leadBin] 
                else:
                    network_features[wIdx][..., leadBin] = w_arr @ spike_samples[..., (params.networkSampleBins-1) - leadBin]         

    return network_features

def create_network_features_and_store_in_dict(task_info, reach_set_df):

    lead_lag_key = task_info['lead_lag_key']
    model_class = task_info['model_class']
    model_name  = task_info['model_name']
    train_FN    = task_info['train_FN']
    test_FN     = task_info['test_FN']
    train_FN_key    = task_info['train_FN_key']
    test_FN_key     = task_info['test_FN_key']
    models_per_set = task_info['models_per_set']
    cortical_area_idxs = task_info['cortical_area_idxs'] if 'cortical_area_idxs' in task_info.keys() else None  
    
    model_train_features_dict = results_dict[lead_lag_key]['model_features']
    
    spike_samples = results_dict[lead_lag_key]['sampled_data']['spike_samples']
    sample_info   = results_dict[lead_lag_key]['sampled_data']['sample_info']

    network_features_train_FN = compute_network_features(spike_samples, sample_info, train_FN, train_FN_key, models_per_set, task_info, cortical_area_idxs=cortical_area_idxs)
    if test_FN is not None: 
        network_features_test_FN  = compute_network_features(spike_samples, sample_info,  test_FN,  test_FN_key, models_per_set, task_info, cortical_area_idxs=cortical_area_idxs)
    else:
        network_features_test_FN = [None] * len(network_features_train_FN)

    try:
        model_names = [item for key, item in task_info.items() if key == 'model_names']
        tmp = model_names[0]
    except:
        model_names = [item for key, item in task_info.items() if key == 'model_name']

    if type(model_names[0]) == list and len(model_names) == 1:
        model_names = model_names[0]
    
    if len(network_features_train_FN) != len(model_names):
        raise Exception("The length of network features (should be 1 per model) does not match the length of the model_names list.")

    if all([name in model_train_features_dict.keys() for name in model_names]):
        return [], []
    
    for model_name, features_train, features_test in zip(model_names, network_features_train_FN, network_features_test_FN):
        features_train = apply_standard_scaler(features_train, mode = 'network')
        if features_test is not None:
            features_test  = apply_standard_scaler(features_test , mode = 'network')
            
    task_info['model_names'] = model_names
    task_info['network_features_train_FN'] = network_features_train_FN
    task_info['network_features_test_FN' ] = network_features_test_FN
    
    return task_info

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

def train_and_test_glm(task_info, network_features_train_FN, network_features_test_FN, 
                       spike_samples, model_name, RNGs, alpha, l1):   
    
    traj_features       = task_info['traj_features']
    train_FN_key        = task_info['train_FN_key'] 
    test_FN_key         = task_info['test_FN_key']
    lead_lag_key        = task_info['lead_lag_key']
    trained_glm_source  = task_info['trained_glm_source']
    save_GLMs           = task_info['save_GLMs']
    
    areaUnderROC = np.empty((spike_samples.shape[0], params.num_model_samples))
    trainingSet_areaUnderROC = np.full((spike_samples.shape[0], params.num_model_samples), np.nan)

    coefs = np.empty((traj_features.shape[-1] + network_features_train_FN.shape[-1] + 1, spike_samples.shape[0], params.num_model_samples))
    
    allTrainPredictions = []
    for samp, split_rng in enumerate(RNGs['train_test_split']):

        # Create train/test datasets for cross-validation
        print(lead_lag_key + ', ' + model_name + ', iteration = ' +str(samp), flush=True)
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
                    trainFeatures.append(network_features_train_FN[unit, trainIdx])
                    if network_features_test_FN is not None:
                        testFeatures.append (network_features_test_FN [unit, testIdx ])                                   
                else:
                    trainFeatures.append(np.hstack((traj_features[trainIdx], network_features_train_FN[unit, trainIdx])))
                    if network_features_test_FN is not None:
                        testFeatures.append (np.hstack((traj_features[testIdx ], network_features_test_FN [unit, testIdx ]))) 
            else:
                if samp == 0:
                    print('unit ' + str(unit) + ' had only ' + str(np.sum(spikes[testIdx] >= 1)) + ' spikes in the sampled time windows and is removed from analysis', flush=True)
            
        train_model = True
        if len(testFeatures)==0:
            train_model = False
            testFeatures=[[]]*len(trainFeatures)
            
        # Train GLM
        predictions = []
        trainPredictions = []
        for unit, (trainSpks, trainFts, testFts) in enumerate(zip(trainSpikes, trainFeatures, testFeatures)):
            if train_model:
                glm = sm.GLM(trainSpks,
                             sm.add_constant(trainFts), 
                             family=sm.families.Poisson(link=sm.families.links.Log()))
                encodingModel = glm.fit_regularized(method='elastic_net', alpha=alpha, L1_wt=l1)
            
                coefs  [:, unit, samp] = encodingModel.params            
            
                if save_GLMs:
                    model_filepath = os.path.join(models_storage_folder, '%s_%s_sample_%s_unit_%s_encoding_model.pkl' % (lead_lag_key, model_name, str(samp).zfill(4), str(unit).zfill(3)))
                    with open(model_filepath, 'wb') as f:
                        dill.dump(encodingModel, f, recurse=True)
                    
                predictions.append(encodingModel.predict(sm.add_constant(testFts)))
            
            if trained_glm_source is not None:
                if save_GLMs:
                    stored_encoding_model = encodingModel 
                else:   
                    model_filepath = os.path.join(models_storage_folder, '%s_%s_sample_%s_unit_%s_encoding_model.pkl' % (lead_lag_key, trained_glm_source, str(samp).zfill(4), str(unit).zfill(3)))
                    with open(model_filepath, 'rb') as f:
                        stored_encoding_model = dill.load(f) 
                    
                trainPredictions.append(stored_encoding_model.predict(sm.add_constant(trainFts))) 

                
        # Test GLM --> area under ROC
        if train_model: 
            areaUnderROC[:, samp], allHitProbs, allFalsePosProbs = compute_area_under_ROC(predictions, testSpikes)

        # Get AUC on training data
        if trained_glm_source is not None:
            trainingSet_areaUnderROC[:, samp], _, _ = compute_area_under_ROC(trainPredictions, trainSpikes)
            
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
    Network features for training and testing are train_FN_key = %s, test_FN_key = %s.
    ''' % (model_name, areaUnderROC.shape[0], coefs.shape[0], traj_features.shape[-1], network_features_train_FN.shape[-1], 
    params.num_model_samples, int((1-params.trainRatio)*1e2),  int((params.trainRatio)*1e2), trained_glm_source,
    train_FN_key, test_FN_key)
    
    model_results = {'AUC'                            : areaUnderROC,
                     'trainAUC'                       : trainingSet_areaUnderROC,
                     'coefs'                          : coefs,
                     'all_predictions_on_training_set': allTrainPredictions,
                     'description'                    : description,
                     'alpha'                          : alpha,
                     'l1'                             : l1}    
    
    return model_results

def assign_models_to_job_tasks(models_dict, task_id):
    all_models_info_list = []
    for lead_lag_key in params.lead_lag_keys_for_network:
                
        for model_name in remove_models:
            tmp_results_key = '%s_%s' % (params.primary_traj_model, model_name)
            if model_name in results_dict[lead_lag_key]['model_features'].keys():
                del results_dict[lead_lag_key]['model_features'][model_name]
            if tmp_results_key in results_dict[lead_lag_key]['model_results'].keys():
                del results_dict[lead_lag_key]['model_results'][tmp_results_key] 
        
        for model_class in models_dict.keys():                    
            
            for model_idx, (model_name, train_FN, train_FN_key, trained_glm_source, save_GLMs, save_feat) in enumerate(zip(models_dict[model_class]['model_names'],
                                                                                                                           models_dict[model_class]['FN'],
                                                                                                                           models_dict[model_class]['FN_key'],
                                                                                                                           models_dict[model_class]['trained_glm_source'],
                                                                                                                           models_dict[model_class]['save_GLMs'],
                                                                                                                           models_dict[model_class]['save_network_features'])):
            
                if trained_glm_source:
                    trained_glm_source = trained_glm_source.replace('kin_model', params.primary_traj_model)
                
                model_name = model_name.replace('kin_model', params.primary_traj_model)
                if params.primary_traj_model in model_name:
                    traj_features = results_dict[lead_lag_key]['model_features'][params.primary_traj_model]
                else:
                    traj_features = np.array(())
                
                submodel_info = dict(lead_lag_key          = lead_lag_key,
                                     model_name            = model_name,
                                     model_class           = model_class,
                                     shuffle_mode          = models_dict[model_class]['shuf_mode'],
                                     models_per_set        = models_dict[model_class]['models_per_set'],
                                     train_FN              = train_FN, 
                                     train_FN_key          = train_FN_key, 
                                     traj_features         = traj_features,
                                     trained_glm_source    = trained_glm_source,
                                     save_GLMs             = save_GLMs,
                                     save_network_features = save_feat)

                if 'test_FN' in models_dict[model_class].keys():
                    submodel_info['test_FN']     = models_dict[model_class]['test_FN'][model_idx]
                    submodel_info['test_FN_key'] = models_dict[model_class]['test_FN_key'][model_idx]
                elif ('retrain_model' not in models_dict[model_class].keys() or models_dict[model_class]['retrain_model'] is True):   
                    submodel_info['test_FN']     = submodel_info['train_FN']
                    submodel_info['test_FN_key'] = submodel_info['train_FN_key']
                elif not models_dict[model_class]['retrain_model']:
                    submodel_info['test_FN']     = None
                    submodel_info['test_FN_key'] = None    
                    
                if 'edges_to_shuffle' in models_dict[model_class].keys():
                    submodel_info['edges_to_shuffle'] = models_dict[model_class]['edges_to_shuffle']

                if 'percents' in models_dict[model_class].keys():
                    submodel_info['percents'] = models_dict[model_class]['percents']

                if 'cortical_area_idxs' in models_dict[model_class].keys():
                    submodel_info['cortical_area_idxs'] = models_dict[model_class]['cortical_area_idxs']
                
                all_models_info_list.append(submodel_info)        
    
    all_tasks_info_list = []            
    for submodel_info in all_models_info_list:
        if 'tmp' in submodel_info['model_name']:
            for idx, task_list in enumerate(all_tasks_info_list):
                if np.any([task_info['model_name'] == submodel_info['trained_glm_source'] and task_info['lead_lag_key'] == submodel_info['lead_lag_key'] for task_info in task_list]):
                    task_index = idx
            all_tasks_info_list[task_index].append(submodel_info)
        else:
            all_tasks_info_list.append([submodel_info])
            
    task_model_list = all_tasks_info_list[task_id]
    
    return all_tasks_info_list, task_model_list

def grab_cortical_area_FN_idxs(units_res):
    
    cortical_area_idxs= dict()
    for regions, set_name in zip(params.intra_inter_areas_list, params.intra_inter_names_list):       
        area_units = units_res.copy()
        mask = area_units['cortical_area'] == 0
        for reg in regions:
            mask = mask | (area_units['cortical_area'] == reg)
            
        area_units = area_units.loc[mask, 'cortical_area']

        cortical_area_idxs[set_name] = area_units.index.values
            
    return cortical_area_idxs

def add_tmp_files_to_pkl():   

    job_array_files = glob.glob(os.path.join(tmp_job_array_folder, '*'))    

    for job_file in job_array_files:
        with open(job_file, 'rb') as f:
            stored_model_info = dill.load(f)
                
        lead_lag_key       = stored_model_info['lead_lag_key']
        model_name_list    = stored_model_info['model_names']
        model_results_list = stored_model_info['model_results']
        # sample_range     = range(0, params.num_model_samples)
        if type(model_name_list) != list:
            model_name_list = [model_name_list]

        if 'model_results' not in results_dict[lead_lag_key].keys():
            results_dict[lead_lag_key]['model_results'] = dict()    
        
        for model_name, model_results in zip(model_name_list, model_results_list): 
            results_dict[lead_lag_key]['model_results' ][model_name] = model_results
        
        if 'network_features_train_FN' in stored_model_info.keys():
            network_features = stored_model_info['network_features_train_FN'][0]
            results_dict[lead_lag_key]['model_features'][model_name] = network_features
    
    # with open(pkl_outfile, 'wb') as f:
    #     dill.dump(results_dict, f, recurse=True)  
    
    save_dict_to_hdf5(results_dict, pkl_outfile.with_suffix('.h5'))

if __name__ == "__main__":
    
    print('\n\n Began creating models at %s\n\n' % time.strftime('%c', time.localtime()), flush=True)
    
    if debugging:
        task_id = 5
        file_creation_task = task_id
    elif demo:
        task_id = 0
        file_creation_task = task_id    
    else:
        task_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
        file_creation_task = int(os.getenv('SLURM_ARRAY_TASK_MAX'))
    
    # with open(pkl_infile, 'rb') as f:
    #     results_dict = dill.load(f)
    results_dict = load_dict_from_hdf5(pkl_infile.with_suffix('.h5'))    
        
    with NWBHDF5IO(nwb_infile, 'r') as io:
        nwb = io.read()

        reach_FN = nwb.scratch[params.reach_FN_key].data[:]
        spontaneous_FN = nwb.scratch['spontaneous_FN'].data[:]
        reach_set_df = nwb.scratch['split_FNs_reach_sets'].to_dataframe()
            
    units_res = results_dict[params.lead_lag_keys_for_network[0]]['all_models_summary_results']
    electrode_distances = get_interelectrode_distances_by_unit(units_res, array_type='utah')
    
    cortical_area_idxs = grab_cortical_area_FN_idxs(units_res)
    
    '''
        Models are set up, now just need to make changes to feature creation function to deal with different sets of inputs. 
    '''
    
    RNGs = {'train_test_split'      : [np.random.default_rng(n) for n in range(params.num_model_samples)],
            'partial_traj'          : [np.random.default_rng(n) for n in range(1000,  1000+params.num_model_samples)],
            'spike_shuffle'         : [np.random.default_rng(n) for n in range(5000,  5000+results_dict[params.lead_lag_keys_for_network[0]]['sampled_data']['spike_samples'].shape[0])],
            'weight_shuffle'        : [np.random.default_rng(n) for n in range(10000, 10000+params.num_model_samples)],
            'network_input_shuffles': [[np.random.default_rng(n) for n in range(unit_idx*2000,  unit_idx*2000+params.num_model_samples)] for unit_idx in range(units_res.shape[0])]}

    models_dict = {'full_vs_spont_FN' : {'model_names'           : ['kin_model_spont_train_reach_test_FN', 'kin_model_reach_train_spont_test_FN', 'spont_train_reach_test_FN', 'reach_train_spont_test_FN'],  
                                         'shuf_mode'             : 'unaltered',
                                         'models_per_set'        : 1,
                                         'FN'                    : [spontaneous_FN.copy(), reach_FN.copy(), spontaneous_FN.copy(), reach_FN.copy()],
                                         'FN_key'                : ['spontaneous_FN', 'reach_FN', 'spontaneous_FN', 'reach_FN'],
                                         'test_FN'               : [reach_FN.copy(), spontaneous_FN.copy(), reach_FN.copy(), spontaneous_FN.copy()],
                                         'test_FN_key'           : ['reach_FN', 'spontaneous_FN', 'reach_FN', 'spontaneous_FN'],
                                         'trained_glm_source'    : [None, None, None, None],
                                         'save_GLMs'             : [False, False, False, False],
                                         'save_network_features' : [False, False, False, False]},
                   
                   'reach_FN'         : {'model_names'           : ['reach_FN', 'kin_model_reach_FN'],
                                         'shuf_mode'             : 'unaltered',
                                         'models_per_set'        : 1,
                                         'FN'                    : [reach_FN.copy(), reach_FN.copy()],
                                         'FN_key'                : ['reach_FN', 'reach_FN'],
                                         'trained_glm_source'    : ['reach_FN', 'kin_model_reach_FN'],
                                         'save_GLMs'             : [True, True],
                                         'save_network_features' : [True, False]},
                   
                   'spontaneous_FN'   : {'model_names'           : ['spontaneous_FN', 'kin_model_spontaneous_FN'],
                                         'shuf_mode'             : 'unaltered',
                                         'models_per_set'        : 1,
                                         'FN'                    : [spontaneous_FN.copy(), spontaneous_FN.copy()],
                                         'FN_key'                : ['spontaneous_FN', 'spontaneous_FN'],
                                         'trained_glm_source'    : ['spontaneous_FN', 'kin_model_spontaneous_FN'],
                                         'save_GLMs'             : [True, True],
                                         'save_network_features' : [True, False]},
                    'strength_shuffles': {'model_names'           : ['kin_model_tmp', 'kin_model_tmp'],
                                          'shuf_mode'             : ['strength', 'random'],
                                          'edges_to_shuffle'      : ['weights','topology'],
                                          'percents'              : [1, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100],
                                          'retrain_model'         : False,
                                          'models_per_set'        : 60,
                                          'FN'                    : [reach_FN.copy(), spontaneous_FN.copy(), reach_FN.copy(), spontaneous_FN.copy()],
                                          'FN_key'                : ['reach_FN', 'spontaneous_FN'],
                                          'trained_glm_source'    : ['kin_model_reach_FN', 'kin_model_spontaneous_FN'],
                                          'save_GLMs'             : [False, False],
                                          'save_network_features' : [False, False]}}

                   # 'cortical_area_FN' : {'model_names'           : ['kin_model_intra_reach_FN', 'kin_model_inter_reach_FN', 'intra_reach_FN', 'inter_reach_FN'],  
                   #                       'shuf_mode'             : 'unaltered',
                   #                       'models_per_set'        : 1,
                   #                       'cortical_area_idxs'    : cortical_area_idxs,
                   #                       'FN'                    : [reach_FN.copy(), reach_FN.copy(), reach_FN.copy(), reach_FN.copy()],
                   #                       'FN_key'                : ['reach_FN', 'reach_FN', 'reach_FN', 'reach_FN'],
                   #                       'trained_glm_source'    : ['kin_model_intra_reach_FN', 'kin_model_inter_reach_FN', 'intra_reach_FN', 'inter_reach_FN'],
                   #                       'save_GLMs'             : [True, True, True, True],
                   #                       'save_network_features' : [True, True, True, True]},
            
    all_tasks_info_list, task_model_list = assign_models_to_job_tasks(models_dict, task_id)        
    n_job_files = sum([len(single_task_list) for single_task_list in all_tasks_info_list])
    
    for task_id, task_model_list in enumerate(all_tasks_info_list):
        for model_set, task_info in enumerate(task_model_list):
            
            if shuffles_only and 'edges_to_shuffle' not in task_info.keys():
                continue
            elif no_shuffles and 'edges_to_shuffle' in task_info.keys():
                continue
            
            pkl_tmp_job_file = tmp_job_array_folder / f'{pkl_outfile.stem}_tmp_job_{str(task_id).zfill(3)}_model_set_{str(model_set).zfill(2)}.pkl'
            
            task_info = create_network_features_and_store_in_dict(task_info, reach_set_df)
            
            lead_lag_key = task_info['lead_lag_key']
            
            task_info['model_results'] = []
            
            for network_features_train_FN, network_features_test_FN, model_name in zip(task_info['network_features_train_FN'], 
                                                                                       task_info['network_features_test_FN'], 
                                                                                       task_info['model_names']):
                alpha = params.alpha
                l1 = params.l1       
                
                model_results = train_and_test_glm(task_info, 
                                                   network_features_train_FN,
                                                   network_features_test_FN,
                                                   results_dict[lead_lag_key]['sampled_data']['spike_samples'], 
                                                   model_name,
                                                   RNGs,
                                                   alpha,
                                                   l1)
                                
                task_info['model_results'].append(model_results)
                    
            del task_info['traj_features']
            del task_info['network_features_test_FN']
            if not task_info['save_network_features']:
                del task_info['network_features_train_FN']
                
            with open(pkl_tmp_job_file, 'wb') as f:
                dill.dump(task_info, f, recurse=True)
                
            print('Just saved model results for %s' % lead_lag_key, flush=True)
                                               
    if demo:
        add_tmp_files_to_pkl()
    else:
        if task_id == file_creation_task:
            while len(glob.glob(os.path.join(tmp_job_array_folder, '*'))) < n_job_files:
                print('waiting for all jobs to finish model creation')
                time.sleep(100)
            add_tmp_files_to_pkl()

            
    print('\n\n Finished creating models at %s\n\n' % time.strftime('%c', time.localtime()), flush=True)

        
