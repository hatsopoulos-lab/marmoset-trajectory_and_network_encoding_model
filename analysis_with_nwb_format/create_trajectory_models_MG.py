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
import dill
import os
import time
import glob
import re
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc
import statsmodels.api as sm
from scipy.fft import rfft, rfftfreq
from scipy.ndimage import gaussian_filter
from importlib import sys

from pynwb import NWBHDF5IO
import ndx_pose

sys.path.insert(0, '/project/nicho/projects/marmosets/code_database/data_processing/nwb_tools/hatlab_nwb_tools/')
from hatlab_nwb_functions import get_sorted_units_and_apparatus_kinematics_with_metadata   

sys.path.insert(0, '/project/nicho/projects/marmosets/code_database/analysis/trajectory_encoding_model/')
from utils import choose_units_for_model

marmcode = 'MG'

if marmcode=='TY':
    nwb_infile = '/project/nicho/projects/dalton/data/TY/TY20210211_freeAndMoths-003_resorted_20230612_DM.nwb' 
    # new_tag = '_FINAL_trajectory_shuffled_encoding_models_30ms_shift_v1'
    new_tag = '_alpha_pt00001_encoding_models_30ms_shift_v1'
elif marmcode=='MG':
    nwb_infile   = '/project/nicho/projects/dalton/data/MG/MG20230416_1505_mothsAndFree-002_processed_DM.nwb'
    # new_tag = '_dlcIter5_noBadUnitsList_trajectory_shuffled_encoding_models_30ms_shift_v1' 
    new_tag = '_alpha_pt00001_removedUnits_181_440_fixedMUA_745_796_encoding_models_30ms_shift_v1'  
    
base, ext = os.path.splitext(nwb_infile)

base, old_tag = base.split('DM')
pkl_infile = base + 'DM' + new_tag + '.pkl'

debugging=False
redo_feature_creation=False

use_traj_regParams_for_traj_avgPos = False

class params:
    
    if marmcode == 'MG':
        fps = 200
        trainRatio  = 0.9
    elif marmcode =='TY':
        fps = 150
        trainRatio  = 0.8

        
    num_models_including_shuffles = 15
    num_model_iters = 100
    if debugging:
        # iter_ranges = [range(0, 2), range(2, 4)]
        iter_ranges = [range(0, 33), range(33, 66), range(66, 100)]
    else:
        # iter_ranges = [range(start, stop) for start, stop in zip(range(0, 91, 10), range(10, 101, 10))]
        # iter_ranges = [range(0, 50), range(50, 100)]
        # iter_ranges = [range(0, 25), range(25, 50), range(50, 75), range(75, 100)]
        # iter_ranges = [range(0, 33), range(33, 66), range(66, 100)]
        iter_ranges = [range(0, 100)]

    num_features_after_traj = 3

    use_regularization = True
    alpha = 0.00001 #0.01
    l1 = 0

    spkSampWin = 0.01
    trajShift = 0.03 #sample every 30ms
    lead = [0.5, 0.4, 0.4, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0  , 0  , 0  , 0.15, 0.25] # lead time
    lag  = [0  , 0  , 0.1, 0  , 0.1, 0.2, 0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.3, 0.4, 0.5, 0.15, 0.25] # lag time
    # lead = [0.2, 0.1] # lead time
    # lag  = [0.3, 0.3] # lag time
    numThresh = 100
    selectRatio = 0.1
    shuffle_to_test = ['traj_avgPos']
    kin_to_split = 'tortuosity'
    models_to_split_by_kin = ['traj_avgPos', 'shortTraj_avgPos', 'traj', 'shortTraj']
    minSpikeRatio = .005
    nDims = 3
    frate_thresh = 2
    snr_thresh = 3
    subsamp_fps = 40
    
    # alpha_range = np.linspace(.005, 0.015, 3)
    # l1_range = np.linspace(0.7, 0.7, 1)
    regularization_iters = 5
    regTest_nUnits = 60
    average_reg_choice_over = 'lead_lag' # can be one of ['iterations', 'units', 'lead_lag']
    
    pca_var_thresh = 0.9# MAKE SURE that the idx being pulled is for the hand in run_pca_on_trajectories()
    idx_for_avg_pos_and_speed = 0
    hand_traj_idx = 0 
    FN_source = 'split_reach_FNs'
    
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

def choose_regularization_params(model, choose_by, plot=False):
    
    if choose_by == 'lead_lag':
        for ll_idx, (lead, lag) in enumerate(zip(params.lead, params.lag)):    
            # if debugging:
            #     if lead == params.lead[2] and lag == params.lag[2]:
            #         break  
            lead_lag_key = 'lead_%d_lag_%d' % (int(lead*1e3), int(lag*1e3))            

            ll_hyperparams_array = results_dict[lead_lag_key]['model_regularization_hyperparameter_results'][model]
            if 'full_hyperparams_array' not in locals():
                full_hyperparams_array = np.full((ll_hyperparams_array.shape[0],
                                                  ll_hyperparams_array.shape[1],
                                                  ll_hyperparams_array.shape[2],
                                                  ll_hyperparams_array.shape[3],
                                                  len(params.lead)), np.nan)
            
            full_hyperparams_array[..., ll_idx] = ll_hyperparams_array
        
        average_auc = np.full((ll_hyperparams_array.shape[0],
                               ll_hyperparams_array.shape[1]), np.nan)
        for aIdx in range(ll_hyperparams_array.shape[0]):
            for lIdx in range(ll_hyperparams_array.shape[1]):
                average_auc[aIdx, lIdx] = np.nanmean(full_hyperparams_array[aIdx, lIdx])
        
        alpha_idx, l1_idx = np.where(average_auc == average_auc.max())
        alpha = params.alpha_range[alpha_idx][0]
        l1    = params.l1_range[l1_idx][0]
        
        if plot:
            fig, ax = plt.subplots(figsize=(6,6), dpi = 300)
            sns.heatmap(average_auc,ax=ax,cmap='viridis', vmin=average_auc[average_auc>0].min(), vmax=average_auc.max()) 
            ax.set_ylabel('Alpha')
            ax.set_xlabel('L1 weight')
            ax.set_yticklabels(params.alpha_range)
            ax.set_xticklabels(params.l1_range)
            plt.show()

    return alpha, l1

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

def train_and_test_glm(task_info, RNGs):   
    
    traj_features = task_info['traj_features']
    spike_samples = task_info['spike_samples']
    model_key     = task_info['model_key']
    lead_lag_key  = task_info['lead_lag_key']
    alpha         = task_info['alpha']
    l1            = task_info['l1']
    iter_range    = task_info['iter_range']
    sample_info   = task_info['sample_info']
    good_samples  = task_info['good_samples']
    
    areaUnderROC = np.full((spike_samples.shape[0], params.num_model_iters), np.nan)
    trainingSet_areaUnderROC = np.full((spike_samples.shape[0], params.num_model_iters), np.nan)

    # aic          = np.empty_like(areaUnderROC)
    coefs = np.full((traj_features.shape[-1] + 1, spike_samples.shape[0], params.num_model_iters), np.nan)
    # pVals = np.empty_like(coefs)    
    
    if params.kin_to_split in model_key:
        threshold = sample_info[f'mean_{params.kin_to_split}'].median()
        good_samples_info = sample_info.iloc[good_samples, :]
        if 'low' in model_key:
            traj_samples_subset_idx = np.where(good_samples_info[f'mean_{params.kin_to_split}'] <  threshold)
        elif 'high' in model_key:
            traj_samples_subset_idx = np.where(good_samples_info[f'mean_{params.kin_to_split}'] >= threshold)
        
        traj_features = traj_features[traj_samples_subset_idx]
    
    for samp in iter_range:
        # Create train/test datasets for cross-validation
        print(lead_lag_key + ', ' + model_key + ', iteration = ' +str(samp), flush=True)
        split_rng = RNGs['train_test_split'][samp]
        
        testSpikes = []
        trainSpikes = []
        trainFeatures = []
        testFeatures = []
        for unit, spikes in enumerate(spike_samples[..., -1]):
            # if debugging and unit > 2:
            #     break

            if params.kin_to_split in model_key:                
                spikes = spikes[traj_samples_subset_idx]
                
            spikeIdxs   = np.where(spikes >= 1)[0]
            noSpikeIdxs = np.where(spikes == 0)[0]    
                
            idxs = np.union1d(spikeIdxs, noSpikeIdxs)
            
            trainIdx = np.hstack((split_rng.choice(spikeIdxs  , size = round(params.trainRatio*len(spikeIdxs  )), replace = False), 
                                  split_rng.choice(noSpikeIdxs, size = round(params.trainRatio*len(noSpikeIdxs)), replace = False)))
            testIdx  = np.setdiff1d(idxs, trainIdx)
                
            trainSpikes.append(spikes[trainIdx])
            testSpikes.append(spikes[testIdx])
              
            trainFeatures.append(traj_features[trainIdx])
            testFeatures.append (traj_features[testIdx])
          
        # Train GLM
        
        models = []
        predictions = []
        trainPredictions = []
        for unit, (trainSpks, trainFts, testFts, testSpks, shuf_rng) in enumerate(zip(trainSpikes, trainFeatures, testFeatures, testSpikes, RNGs['spike_shuffle'])):
            # if debugging and unit > 2:
            #     break
            if 'shuffled_spikes' in model_key:
                trainSpks = shuf_rng.permutation(trainSpks)
            elif 'shuffled_traj' in model_key:
                trainFts[:, :-params.num_features_after_traj] = shuf_rng.permutation(trainFts[:, :-params.num_features_after_traj], axis = 0)
                    
                            
            glm = sm.GLM(trainSpks,
                         sm.add_constant(trainFts), 
                         family=sm.families.Poisson(link=sm.families.links.log()))
            if params.use_regularization:
                encodingModel = glm.fit_regularized(method='elastic_net', alpha=alpha, L1_wt=l1)
            else:
                encodingModel = glm.fit()
                
            
            coefs[:, unit, samp] = encodingModel.params            
            # pVals[:, unit, samp] = np.round(encodingModel.pvalues, decimals = 4)            
            # aic  [   unit, samp] = round(encodingModel.aic, 1)
            
            models.append(encodingModel)
            predictions.append(encodingModel.predict(sm.add_constant(testFts)))
            trainPredictions.append(encodingModel.predict(sm.add_constant(trainFts))) 
            
        # Test GLM --> area under ROC
        areaUnderROC[:, samp], allHitProbs, allFalsePosProbs = compute_area_under_ROC(predictions, testSpikes)
       
        # Get AUC on training data
        if model_key in params.shuffle_to_test:
            trainingSet_areaUnderROC[:, samp], _, _ = compute_area_under_ROC(trainPredictions, trainSpikes)
            
        
    description = '''Model = %s. 
    To understand dimensions: 
        %d units,
        %d model parameters (1 constant, %d kinematic features)
        %d shuffles of the train/test split. 
    The order of model parameters is: 
        Constant term
        Trajectory features 
            - If pca in model name, features are data transformed onto the number of PCAcomps that explain 90%% of variance
            - If pca not in model name, features are the subsampled velocity or position terms (traj=velocity). 
              The coefs are ordered by [x*nsamples ... y*nsamples ... z*nsamples]. So if there are 60 traj features, 
              the first 20 are x, middle 20 are y, last 20 are z. 
              See the info stored in model_features for details on how the data was standardized and the timestamps corresponding to trajectory coefs.   
        Average position terms (3 terms, x/y/z, if model includes average position) are at the end.
        Average speed terms if included in model, after position terms. 
    The keys hold the following information:
        AUC: cross-validated area under the ROC curve on %d%% held-out test data 
        AIC: AIC criterion value for trained model on training data
        coefs: the parameter coefficents that have been fit to the input features  
        pvals: the p-values describing the significance of the parameter coefficients
    If this is a shuffled model, that means the spike samples were shuffled to eliminate the relationship
    between model features and spiking. A new spike_samples shuffle was performed for each train/test split. 
    ''' % (model_key, areaUnderROC.shape[0], coefs.shape[0], traj_features.shape[-1], params.num_model_iters, int((1-params.trainRatio)*1e2))

    model_results = {'AUC'         : areaUnderROC,
                     'trainAUC'    : trainingSet_areaUnderROC,
                     'coefs'       : coefs,
                     'description' : description,
                     'alpha'       : alpha,
                     'l1'          : l1}     
    # model_results = {'AUC'         : areaUnderROC,
    #                  'coefs'       : coefs,
    #                  'pvals'       : pVals,
    #                  'AIC'         : aic,
    #                  'alpha'       : all_alpha,
    #                  'L1_weight'   : all_l1,
    #                  'description' : description}    
    
    return model_results

def add_tmp_files_to_pkl():   

    job_array_files = glob.glob(os.path.join(tmp_job_array_folder, '*'))    

    for job_file in job_array_files:
        with open(job_file, 'rb') as f:
            task_info = dill.load(f)
        
        lead_lag_key  = task_info['lead_lag_key']
        model_key     = task_info['model_key']
        model_results = task_info['model_results']
        iter_range  = task_info['iter_range']

        if 'model_results' not in results_dict[lead_lag_key].keys():
            results_dict[lead_lag_key]['model_results'] = dict()    

        if model_key not in results_dict[lead_lag_key]['model_results'].keys():
            results_dict[lead_lag_key]['model_results'][model_key] = model_results
        else:
            tmp_auc   = results_dict[lead_lag_key]['model_results'][model_key]['AUC']
            tmp_coefs = results_dict[lead_lag_key]['model_results'][model_key]['coefs']

            tmp_auc  [:  , iter_range.start:iter_range.stop] = model_results[  'AUC'][:  , iter_range.start:iter_range.stop]
            tmp_coefs[..., iter_range.start:iter_range.stop] = model_results['coefs'][..., iter_range.start:iter_range.stop]

            results_dict[lead_lag_key]['model_results'][model_key]['AUC'] = tmp_auc
            results_dict[lead_lag_key]['model_results'][model_key]['coefs'] = tmp_coefs
            
            if 'trainAUC' in model_results.keys():
                tmp_train_auc   = results_dict[lead_lag_key]['model_results'][model_key]['trainAUC']
                tmp_train_auc[:, iter_range.start:iter_range.stop] = model_results['trainAUC'][:, iter_range.start:iter_range.stop]
                results_dict[lead_lag_key]['model_results'][model_key]['trainAUC'] = tmp_train_auc    
    
    with open(pkl_outfile, 'wb') as f:
        dill.dump(results_dict, f, recurse=True)     

def run_models():

    lead_lag_keys = list(results_dict.keys())    

    subset_count = 0
    task_info_stored = False
    for lead_lag_key in lead_lag_keys:
        if task_info_stored:
            break
        spike_samples = results_dict[lead_lag_key]['sampled_data']['spike_samples'] 
        sample_info   = results_dict[lead_lag_key]['sampled_data']['sample_info'] 
        model_keys = [key for key in results_dict[lead_lag_key]['model_features'].keys() if key not in ['description', 'traj_PCA_components', 'subsample_times']]        
        model_keys.extend([f'{shuf_model}_shuffled_spikes' for shuf_model in params.shuffle_to_test])
        model_keys.extend([f'{shuf_model}_shuffled_traj'   for shuf_model in params.shuffle_to_test])
        model_keys.extend([f'{kin_split_model}_low_{params.kin_to_split}' for kin_split_model in params.models_to_split_by_kin])
        model_keys.extend([f'{kin_split_model}_high_{params.kin_to_split}' for kin_split_model in params.models_to_split_by_kin])
        for model in model_keys:
            
            if 'tortuosity' in model:
                print(subset_count)
            
            if task_info_stored:
                break
            features_key = model.split('_shuffled')[0].split(f'_low_{params.kin_to_split}')[0].split(f'_high_{params.kin_to_split}')[0]
            if use_traj_regParams_for_traj_avgPos and features_key == 'traj_avgPos':
                regParams_key = 'traj'
            else:
                regParams_key = features_key
            traj_features = results_dict[lead_lag_key]['model_features'][features_key]
            if params.use_regularization:
                if params.alpha:
                    alpha = params.alpha
                    l1    = params.l1
                else:
                    alpha, l1 = choose_regularization_params(regParams_key, choose_by = params.average_reg_choice_over)
            else:
                alpha = None
                l1    = None
                
            good_samples = np.where(~np.isnan(traj_features[:, 0]))[0]
            tmp_traj_features = traj_features.copy()[good_samples]
            tmp_spike_samples = spike_samples.copy()[:, good_samples]
            
            for iter_range in params.iter_ranges:
                if subset_count == task_id:
                    task_info = dict(lead_lag_key  = lead_lag_key,
                                     model_key     = model,
                                     iter_range  = iter_range,
                                     spike_samples = tmp_spike_samples,
                                     traj_features = tmp_traj_features,
                                     sample_info   = sample_info,
                                     good_samples  = good_samples,
                                     alpha         = alpha,
                                     l1            = l1)
                    task_info_stored = True
                    lead_lag_key_to_print = lead_lag_key
                    break
                else:
                    subset_count += 1
                
    RNGs = {'train_test_split'            : [np.random.default_rng(n) for n in range(params.num_model_iters)],
            'regTesting_train_test_split' : [np.random.default_rng(n) for n in range(1000, 1000+params.regularization_iters)],
            'spike_shuffle'               : [np.random.default_rng(n) for n in range(5000, 5000+spike_samples.shape[0])]}        

    task_info['model_results'] = train_and_test_glm(task_info, RNGs)
    
    del task_info['spike_samples']
    del task_info['sample_info']
    del task_info['traj_features']
    del task_info['alpha']
    del task_info['l1']

    with open(pkl_tmp_job_file, 'wb') as f:
        dill.dump(task_info, f, recurse=True)
        
    print('Just saved model results for %s' % lead_lag_key_to_print, flush=True)


def compute_trajectories_fft(traj_samples, srate = 150):

    if type(traj_samples) == list:
        traj_samples = traj_samples[0]
    
    traj_reshaped = traj_samples.transpose(0, 2, 1).reshape(-1, 3, order='C')

    traj_fft = rfft(traj_reshaped, axis = 0)
    fft_freq = rfftfreq(traj_reshaped.shape[0], d = 1./srate)    
    
    fig, axs = plt.subplots(1, 3, figsize=(8, 3), dpi=300)
    for dim, dimlabel in enumerate(['x_vel', 'y_vel', 'z_vel']):
        axs[dim].plot(fft_freq, 2.0/traj_reshaped.shape[0] * np.abs(traj_fft[:, dim]))
        axs[dim].set_title(dimlabel)
    
    plt.show()
    
    

if __name__ == "__main__":
    
    print('\n\n Began creating models at %s\n\n' % time.strftime('%c', time.localtime()), flush=True)
    
    if debugging:
        task_id = 3
        n_tasks = 1
    else:
        task_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
        n_tasks = int(os.getenv('SLURM_ARRAY_TASK_COUNT'))       
    
    with open(pkl_infile, 'rb') as f:
        results_dict = dill.load(f)
    
    n_tasks_needed = len(results_dict.keys()) * params.num_models_including_shuffles * len(params.iter_ranges)
    if n_tasks != n_tasks_needed and not debugging:
        print('number of jobs in array (%d) does not equal necessary number of models %d' % (n_tasks, n_tasks_needed))
        print('ending job', flush=True)
    else:
        
        split_pattern = '_shift_v' # '_results_v'
        base, ext = os.path.splitext(pkl_infile)
        base, in_version = base.split(split_pattern)
        data_folder, base_file = os.path.split(base)
        out_version = str(int(in_version) + 1)  
        pkl_outfile = base + split_pattern + out_version + ext
    
        tmp_job_array_folder = os.path.join(data_folder, 'trajectory_only_jobs_tmp_files', f'{base_file}_v{out_version}')
        pkl_tmp_job_file = os.path.join(tmp_job_array_folder, base_file + split_pattern + out_version + '_tmp_job_' + str(task_id).zfill(2) + ext)
        
        os.makedirs(tmp_job_array_folder, exist_ok=True) 
        
        run_models()
        
        print('\n\n Finished running models  at %s\n\n' % time.strftime('%c', time.localtime()), flush=True)
        
        if task_id == n_tasks-1:
            while len(glob.glob(os.path.join(tmp_job_array_folder, '*'))) < n_tasks:
                completed_jobs = len(glob.glob(os.path.join(tmp_job_array_folder, '*')))
                print(f'completed jobs = {completed_jobs}, n_tasks = {n_tasks}. Waiting for all jobs to finish model creation', flush=True)
                time.sleep(10)
            add_tmp_files_to_pkl()
            
        
        # if redo_feature_creation:
        #     lead_lag_keys = list(results_dict.keys())
        #     nComps = None
        #     for lead_lag_key in lead_lag_keys:
        #         del results_dict[lead_lag_key]['model_features']
        #         create_model_features_and_store_in_dict(nComps, lead_lag_key) 
            
        #     with open(pkl_outfile, 'wb') as f:
        #         dill.dump(results_dict, f, recurse=True)
        
        # else:
        #     models()
            
        #     print('\n\n Finished running models  at %s\n\n' % time.strftime('%c', time.localtime()), flush=True)
        
        #     while len(glob.glob(os.path.join(tmp_job_array_folder, '*'))) < n_tasks:
        #         print('waiting for all jobs to finish model creation')
        #         time.sleep(10)
            
        #     if task_id == 0:
        #         add_tmp_files_to_pkl()

        
    # nComps = find_number_of_trajectory_components(units, reaches, kin_module)

    # # RNGs = {'train_test_split' : [np.random.default_rng(n) for n in range(params.num_model_iters)],
    # #         'partial_traj'     : [np.random.default_rng(n) for n in range(1000,  1000+params.num_model_iters)],
    # #         'spike_shuffle'    : [np.random.default_rng(n) for n in range(5000,  5000+single_lead_lag_models['sampled_spikes'].shape[0])],
    # #         'weight_shuffle'   : [np.random.default_rng(n) for n in range(10000, 10000+params.num_model_iters)]}
    
    # for network_features, model_name in zip(network_features_list, model_names):
    #     new_model_results = train_and_test_glm(single_lead_lag_models['traj_features'], 
    #                                             network_features, 
    #                                             single_lead_lag_models['sampled_spikes'], 
    #                                             model_name, 
    #                                             RNGs)

    #     unit_info['%s_%s' % (model_name, 'AUC')] = new_model_results['AUC'].mean(axis=-1)    
        
    #     all_models_data['%s_%s' % (model_name, 'network_features')] = [network_features if idx == ll_idx else [] for idx in range(len(all_models_data['lead_lag']))]
    #     all_models_data['unit_info'][ll_idx] = unit_info
    #     all_models_data['model_details'][ll_idx]['model_names'  ].append(model_name) 
    #     all_models_data['model_details'][ll_idx]['model_results'].append(new_model_results)
            
    #     save_all_models_dict(new_models_dict_path, all_models_data)   
            
        # '''
        #     Notes: 
        #         - Set up preceding and subsequent section to do pro (tuned) and anti (untuned) sets in the same run
        #         - Set up compute_network_features and modify_weights to take 100 samples of network features, accounting for zero and nonzero input counts
        #         - Set up train_and_test_glm to use 100 samples of network features
        # '''


        
        
