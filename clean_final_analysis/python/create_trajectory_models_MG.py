#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 08:00:51 2020

@author: daltonm
"""
#%matplotlib notebook
 
import numpy as np
import matplotlib.pyplot as plt
import dill
import os
import time
import glob
from sklearn.metrics import auc
import statsmodels.api as sm
from scipy.fft import rfft, rfftfreq
from pathlib import Path

data_path = Path('/project/nicho/projects/dalton/network_encoding_paper/clean_final_analysis/data')

marmcode = 'MG'

if marmcode=='TY':
    nwb_infile = data_path / 'TY' / 'TY20210211_freeAndMoths-003_resorted_20230612_DM.nwb' 
elif marmcode=='MG':
    nwb_infile = data_path / 'MG' / 'MG20230416_1505_mothsAndFree-002_processed_DM.nwb'

pkl_in_tag  = 'samples_collected'
pkl_out_tag = 'kinematic_models_created'

pkl_infile  = nwb_infile.parent / f'{nwb_infile.stem}_{pkl_in_tag}.pkl' 
pkl_outfile = nwb_infile.parent / f'{nwb_infile.stem}_{pkl_out_tag}.pkl' 

debugging=False

class params:
    
    if marmcode == 'MG':
        fps = 200
        trainRatio  = 0.8
    elif marmcode =='TY':
        fps = 150
        trainRatio  = 0.8
       
    num_models_including_shuffles = 7
    num_model_iters = 500
    if debugging:
        # iter_ranges = [range(0, 2), range(2, 4)]
        iter_ranges = [range(0, 33), range(33, 66), range(66, 100)]
    else:
        iter_ranges = [range(0, 250), range(250, 500)]

    num_features_after_traj = 3

    alpha = 0.05
    l1 = 0

    numThresh = 100
    shuffle_to_test = ['traj_avgPos']
    kin_to_split = 'tortuosity'
    models_to_split_by_kin = [] #['traj_avgPos', 'shortTraj_avgPos', 'traj', 'shortTraj']
            
    dpi = 300

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

    coefs = np.full((traj_features.shape[-1] + 1, spike_samples.shape[0], params.num_model_iters), np.nan)
    
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

            if 'shuffled_spikes' in model_key:
                trainSpks = shuf_rng.permutation(trainSpks)
            elif 'shuffled_traj' in model_key:
                trainFts[:, :-params.num_features_after_traj] = shuf_rng.permutation(trainFts[:, :-params.num_features_after_traj], axis = 0)
                    
                            
            glm = sm.GLM(trainSpks,
                         sm.add_constant(trainFts), 
                         family=sm.families.Poisson(link=sm.families.links.Log()))
            encodingModel = glm.fit_regularized(method='elastic_net', alpha=alpha, L1_wt=l1)     
            
            coefs[:, unit, samp] = encodingModel.params
            
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

            traj_features = results_dict[lead_lag_key]['model_features'][features_key]

            alpha = params.alpha
            l1    = params.l1
                
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
        last_task = task_id
    else:
        task_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
        n_tasks = int(os.getenv('SLURM_ARRAY_TASK_COUNT'))  
        last_task = int(os.getenv('SLURM_ARRAY_TASK_MAX'))
    
    with open(pkl_infile, 'rb') as f:
        results_dict = dill.load(f)
    
    n_tasks_needed = len(results_dict.keys()) * params.num_models_including_shuffles * len(params.iter_ranges)
    if n_tasks != n_tasks_needed and not debugging:
        print('number of jobs in array (%d) does not equal necessary number of models %d' % (n_tasks, n_tasks_needed))
        print('ending job', flush=True)
    else:
        
        tmp_job_array_folder = pkl_outfile.parent / 'trajectory_only_jobs_tmp_files' / f'{pkl_outfile.stem}'        
        pkl_tmp_job_file = tmp_job_array_folder / f'{pkl_outfile.stem}_tmp_job_{str(task_id).zfill(3)}.pkl'
        
        os.makedirs(tmp_job_array_folder, exist_ok=True) 
        
        run_models()
        
        print('\n\n Finished running models  at %s\n\n' % time.strftime('%c', time.localtime()), flush=True)
        
        if task_id == last_task:
            while len(glob.glob(os.path.join(tmp_job_array_folder, '*'))) < n_tasks:
                completed_jobs = len(glob.glob(os.path.join(tmp_job_array_folder, '*')))
                print(f'completed jobs = {completed_jobs}, n_tasks = {n_tasks}. Waiting for all jobs to finish model creation', flush=True)
                time.sleep(10)
            add_tmp_files_to_pkl()
