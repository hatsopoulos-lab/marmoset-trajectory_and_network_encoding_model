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
import itertools
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

marmcode = 'MG'
debugging = False

if marmcode=='TY':
    nwb_infile = '/project/nicho/projects/dalton/data/TY/TY20210211_freeAndMoths-003_resorted_20230612_DM_with_functional_networks.nwb' 
    pkl_infile = '/project/nicho/projects/dalton/data/TY/TY20210211_freeAndMoths-003_resorted_20230612_DM_alpha_pt00001_encoding_models_30ms_shift_v3.pkl' 
    extra_tag = 'REGTEST_V2'
elif marmcode=='MG':
    nwb_infile = '/project/nicho/projects/dalton/data/MG/MG20230416_1505_mothsAndFree-002_processed_DM_with_functional_networks.nwb'
    pkl_infile = '/project/nicho/projects/dalton/data/MG/MG20230416_1505_mothsAndFree-002_processed_DM_dlcIter5_resortedUnits_trajectory_shuffled_encoding_models_30ms_shift_v3.pkl'
    extra_tag = 'REGTEST_V2'

split_pattern = '_shift_v' # '_results_v'
base, ext = os.path.splitext(pkl_infile)
base, in_version = base.split(split_pattern)
out_version = str(int(in_version) + 1)  
pkl_outfile = base + extra_tag + split_pattern + out_version + ext

data_folder, base_file = os.path.split(base)
tmp_job_array_folder = os.path.join(data_folder, 'jobs_tmp_saved_files', '%s_v%s' % (base_file+extra_tag, out_version))   

dataset_code = os.path.basename(pkl_infile)[:10] 
plots = os.path.join(os.path.dirname(os.path.dirname(pkl_infile)), 'plots', dataset_code)

os.makedirs(tmp_job_array_folder, exist_ok=True)

class params:
    
    significant_proportion_thresh = 0.99
    numThresh = 100
    if debugging:
        num_model_samples = 2
    else:
        num_model_samples = 100
    
    primary_traj_model = 'traj_avgPos'
    if marmcode == 'TY':
        lead_lag_keys_for_network = ['lead_100_lag_300', 'lead_200_lag_300']
        trainRatio = 0.8

    elif marmcode == 'MG':
        lead_lag_keys_for_network = ['lead_200_lag_300', 'lead_100_lag_300']
        trainRatio = 0.9
        
    reach_FN_key = 'split_reach_FNs'
    
    transpose_FN = False
    minSpikeRatio = .005
    
    networkSampleBins = 2
    networkFeatureBins = 2 

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

    network_features = [np.full((spike_samples.shape[0], spike_samples.shape[1], params.networkFeatureBins), np.nan) for i in range(models_per_set)]

    if FN_key == 'reach_FN' and 'split' in params.reach_FN_key:
        split_weights = []
        for tmp_FN in FN:
            tmp_weights = tmp_FN.copy()
            if params.transpose_FN:
                tmp_weights = tmp_weights.T
            split_weights.append(tmp_weights)

        for sampleNum, reach_idx in enumerate(sample_info['reach_idx']):
            if sampleNum % 500 == 0:
                print(sampleNum, flush=True)
            
            FN_reach_source = reach_set_df.loc[reach_set_df['reach_num'] == reach_idx, 'FN_reach_set'].values[0]
            if FN_reach_source == 2:
                weights = split_weights[0]
            else:
                weights = split_weights[1]
            
            for leadBin in range(params.networkFeatureBins):
                network_features[0][:, sampleNum, leadBin] = weights @ spike_samples[:, sampleNum, (params.networkSampleBins-1) - leadBin] 

    return network_features

def create_network_features_and_store_in_dict(task_info, reach_set_df):

    lead_lag_key       = task_info['lead_lag_key']
    model_name         = task_info['model_name']
    FN                 = task_info['FN']
    FN_key             = task_info['FN_key']
    models_per_set     = task_info['models_per_set']
    cortical_area_idxs = task_info['cortical_area_idxs'] if 'cortical_area_idxs' in task_info.keys() else None  
    
    model_train_features_dict = results_dict[lead_lag_key]['model_features']
    
    spike_samples = results_dict[lead_lag_key]['sampled_data']['spike_samples']
    sample_info   = results_dict[lead_lag_key]['sampled_data']['sample_info']

    model_names = [model_name]
    task_info['model_names'] = model_names
    if FN is None:
        task_info['network_features'] = [None for i in range(len(model_names))]
    else:
        network_features = compute_network_features(spike_samples, sample_info, FN, FN_key, 
                                                    models_per_set, task_info, cortical_area_idxs=cortical_area_idxs)

        for model_name, features in zip(model_names, network_features):
            features = apply_standard_scaler(features, mode = 'network')
           
        task_info['network_features'] = network_features
    
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

def train_and_test_glm(task_info, network_features, 
                       spike_samples, model_name, RNGs, alpha, l1):   
    
    traj_features       = task_info['traj_features']
    lead_lag_key        = task_info['lead_lag_key']
    
    areaUnderROC = np.empty((spike_samples.shape[0], params.num_model_samples))

    if network_features is None:
        coefs = np.empty((traj_features.shape[-1] + 1, spike_samples.shape[0], params.num_model_samples))
    else:    
        coefs = np.empty((traj_features.shape[-1] + network_features.shape[-1] + 1, spike_samples.shape[0], params.num_model_samples))
    
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
                    trainFeatures.append(network_features[unit, trainIdx])
                    testFeatures.append (network_features[unit, testIdx ])  
                elif network_features is None:
                    trainFeatures.append(traj_features[trainIdx])
                    testFeatures.append (traj_features[testIdx ])                                     
                else:
                    trainFeatures.append(np.hstack((traj_features[trainIdx], network_features[unit, trainIdx])))
                    testFeatures.append (np.hstack((traj_features[testIdx ], network_features[unit, testIdx ]))) 
            else:
                if samp == 0:
                    print('unit ' + str(unit) + ' had only ' + str(np.sum(spikes[testIdx] >= 1)) + ' spikes in the sampled time windows and is removed from analysis', flush=True)
            
        # Train GLM
        predictions = []
        for unit, (trainSpks, trainFts, testFts) in enumerate(zip(trainSpikes, trainFeatures, testFeatures)):
            glm = sm.GLM(trainSpks,
                         sm.add_constant(trainFts), 
                         family=sm.families.Poisson(link=sm.families.links.log()))
            encodingModel = glm.fit_regularized(method='elastic_net', alpha=alpha, L1_wt=l1)
            coefs  [:, unit, samp] = encodingModel.params            
            predictions.append(encodingModel.predict(sm.add_constant(testFts)))
                                
        # Test GLM --> area under ROC
        areaUnderROC[:, samp], allHitProbs, allFalsePosProbs = compute_area_under_ROC(predictions, testSpikes)
    
    model_results = {'AUC'                            : areaUnderROC,
                     'coefs'                          : coefs,
                     'alpha'                          : alpha,
                     'l1'                             : l1}    
    
    return model_results

def assign_models_to_job_tasks(models_dict, task_id):
    all_models_info_list = []
    for lead_lag_key in params.lead_lag_keys_for_network:
        
        for model_class in models_dict.keys():                    
            
            for model_idx, (model_name, FN, FN_key) in enumerate(zip(models_dict[model_class]['model_names'],
                                                                     models_dict[model_class]['FN'],
                                                                     models_dict[model_class]['FN_key'])):            
                
                model_name = model_name.replace('kin_model', params.primary_traj_model)
                if params.primary_traj_model in model_name:
                    traj_features = results_dict[lead_lag_key]['model_features'][params.primary_traj_model]
                else:
                    traj_features = np.array(())
                
                for idx, (alpha, l1) in enumerate(itertools.product(models_dict[model_class]['alpha'], models_dict[model_class]['l1'])): 
                
                    submodel_info = dict(lead_lag_key          = lead_lag_key,
                                         model_name            = f'{model_name}_{str(idx).zfill(3)}',
                                         model_class           = model_class,
                                         shuffle_mode          = models_dict[model_class]['shuf_mode'],
                                         models_per_set        = models_dict[model_class]['models_per_set'],
                                         FN                    = FN, 
                                         FN_key                = FN_key, 
                                         traj_features         = traj_features,
                                         alpha                 = alpha,
                                         l1                    = l1)
                           
                    all_models_info_list.append(submodel_info)        
    
    all_tasks_info_list = []            
    for submodel_info in all_models_info_list:
        all_tasks_info_list.append([submodel_info])
            
    task_model_list = all_tasks_info_list[task_id]
    
    return all_tasks_info_list, task_model_list

def test_regularization_params(trainFts, testFts, trainSpks, testSpks, alpha_range, l1_range):
    
    areaUnderROC = np.empty((alpha_range.size, l1_range.size))
    for aIdx, alpha_val in enumerate(alpha_range):
        for lIdx, l1_val in enumerate(l1_range):
    
            glm = sm.GLM(trainSpks,
                         sm.add_constant(trainFts), 
                         family=sm.families.Poisson(link=sm.families.links.log()))
            encodingModel = glm.fit_regularized(method='elastic_net', alpha=alpha_val, L1_wt=l1_val)
            
            predictions = encodingModel.predict(sm.add_constant(testFts))
            
            thresholds = np.linspace(predictions.min(), predictions.max(), params.numThresh)            
            hitProb = np.empty((len(thresholds),))
            falsePosProb = np.empty((len(thresholds),))
            for t, thresh in enumerate(thresholds):    
                posIdx = np.where(predictions > thresh)
                hitProb[t] = np.sum(testSpks[posIdx] >= 1) / np.sum(testSpks >= 1)
                falsePosProb[t] = np.sum(testSpks[posIdx] == 0) / np.sum(testSpks == 0)
            
            areaUnderROC[aIdx, lIdx] = auc(falsePosProb, hitProb)
    
    return areaUnderROC

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
    
    with open(pkl_outfile, 'wb') as f:
        dill.dump(results_dict, f, recurse=True)  

if __name__ == "__main__":
    
    print('\n\n Began creating models at %s\n\n' % time.strftime('%c', time.localtime()), flush=True)
    
    if debugging:
        task_id = 0
        n_tasks = 1
        file_creation_task = task_id
    else:
        task_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
        n_tasks = int(os.getenv('SLURM_ARRAY_TASK_COUNT')) 
        file_creation_task = n_tasks-1
    
    with open(pkl_infile, 'rb') as f:
        results_dict = dill.load(f)
        
    with NWBHDF5IO(nwb_infile, 'r') as io:
        nwb = io.read()

        reach_FN = nwb.scratch[params.reach_FN_key].data[:]
        spontaneous_FN = nwb.scratch['spontaneous_FN'].data[:]
        reach_set_df = nwb.scratch['split_FNs_reach_sets'].to_dataframe()
            
    units_res = results_dict[params.lead_lag_keys_for_network[0]]['all_models_summary_results']
    electrode_distances = get_interelectrode_distances_by_unit(units_res, array_type='utah')
    
    RNGs = {'train_test_split'      : [np.random.default_rng(n) for n in range(params.num_model_samples)],
            'partial_traj'          : [np.random.default_rng(n) for n in range(1000,  1000+params.num_model_samples)],
            'spike_shuffle'         : [np.random.default_rng(n) for n in range(5000,  5000+results_dict[params.lead_lag_keys_for_network[0]]['sampled_data']['spike_samples'].shape[0])],
            'weight_shuffle'        : [np.random.default_rng(n) for n in range(10000, 10000+params.num_model_samples)],
            'network_input_shuffles': [[np.random.default_rng(n) for n in range(unit_idx*2000,  unit_idx*2000+params.num_model_samples)] for unit_idx in range(units_res.shape[0])]}

    models_dict = {'reach_FN'         : {'model_names'           : ['kin_model', 'kin_model_reach_FN'],
                                         'shuf_mode'             : 'unaltered',
                                         'models_per_set'        : 1,
                                         'FN'                    : [None, reach_FN.copy()],
                                         'FN_key'                : [None, 'reach_FN'],
                                         'alpha'                 : [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7],
                                         'l1'                    : [0]}}
            
    all_tasks_info_list, task_model_list = assign_models_to_job_tasks(models_dict, task_id)        
    n_job_files = sum([len(single_task_list) for single_task_list in all_tasks_info_list])
    
    for model_set, task_info in enumerate(task_model_list):
        
        pkl_tmp_job_file = os.path.join(tmp_job_array_folder, 
                                        base_file + split_pattern + out_version + '_tmp_job_%s_model_set_%s' % (str(task_id).zfill(2), str(model_set).zfill(2)) + ext)
        
        task_info = create_network_features_and_store_in_dict(task_info, reach_set_df)
        
        lead_lag_key = task_info['lead_lag_key']
        
        task_info['model_results'] = []
        
        for network_features, model_name in zip(task_info['network_features'],
                                                task_info['model_names']):

            alpha = task_info['alpha']
            l1    = task_info['l1']          
            
            model_results = train_and_test_glm(task_info, 
                                               network_features,
                                               results_dict[lead_lag_key]['sampled_data']['spike_samples'], 
                                               model_name,
                                               RNGs,
                                               alpha,
                                               l1)
                            
            task_info['model_results'].append(model_results)
                
        del task_info['traj_features']
        del task_info['network_features']
            
        with open(pkl_tmp_job_file, 'wb') as f:
            dill.dump(task_info, f, recurse=True)
            
        print('Just saved model results for %s' % lead_lag_key, flush=True)
                                               
    if task_id == file_creation_task:
        while len(glob.glob(os.path.join(tmp_job_array_folder, '*'))) < n_job_files:
            print('waiting for all jobs to finish model creation')
            time.sleep(100)
        add_tmp_files_to_pkl()

    print('\n\n Finished creating models at %s\n\n' % time.strftime('%c', time.localtime()), flush=True)

        
