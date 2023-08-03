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
    nwb_infile = '/project/nicho/projects/dalton/data/TY20210211_freeAndMoths-003_resorted_20230612_DM.nwb' 
    pkl_infile = '/project/nicho/projects/dalton/data/TY20210211_freeAndMoths-003_resorted_20230612_DM_encoding_model_sorting_corrected_30ms_shift_v4.pkl'
elif marmcode=='MG':
    nwb_infile = ''
    pkl_infile = ''

split_pattern = '_shift_v' # '_results_v'
base, ext = os.path.splitext(pkl_infile)
base, in_version = base.split(split_pattern)
out_version = str(int(in_version) + 1)  
pkl_outfile = base + split_pattern + out_version + ext

remove_models = ['tuning_of_inputs']

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
    significant_proportion_thresh = 0.99
    numThresh = 100
    trainRatio = 0.8
    if debugging:
        num_model_samples = 2
    else:
        num_model_samples = 100
    
    primary_traj_model = 'traj_avgPos'
    best_lead_lag_key = 'lead_200_lag_300'
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

def modify_FN_weights_by_tuning(weights, subset_dict, sample_num):
    elec_dist = subset_dict['electrode_distances']
    unit_info_sorted = subset_dict['unit_info'].sort_values(by='proportion_sign', ascending = False)
    unit_info_sorted.reset_index(drop=False, inplace=True)
    unit_info_sorted.columns = ['original_index' if col == 'index' else col for col in unit_info_sorted.columns]
    if subset_dict['upper_bound'] is not None:
        print('This option has not yet been enabled. Add to this code in the function "modify_FN_weights_by_tuning"')
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

def modify_FN_weights_by_cortical_area(weights, subset_dict, sample_num):
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
                print(sample_num, unit, model_name)
                inputs  = shuffle_selected_inputs(inputs,  idxs_to_modify, np.random.default_rng(seed_count))
                inputs[np.isnan(inputs)] = 0
                
            seed_count += 1
        
        output_weights_list[model_idx] = model_weights
    
    return output_weights_list

def modify_FN_weights(weights, subset_dict, sample_num):
    if subset_dict['mode'] == 'distance':
        weights = modify_FN_weights_by_distance(weights, subset_dict)
    elif subset_dict['mode'] == 'tuning':
        pro_weights, anti_weights = modify_FN_weights_by_tuning(weights, subset_dict, sample_num)
        weights = [pro_weights, anti_weights]
    elif subset_dict['mode'] == 'cortical_area':
        weights = modify_FN_weights_by_cortical_area(weights, subset_dict, sample_num)
    elif subset_dict['mode'] == 'original':
        pass

    if type(weights) != list:
        weights = [weights]
        
    return weights

def create_network_features_and_store_in_dict(FN, subset_dict, lead_lag_key, FN_key = 'split_reach_FNs'):
    
    model_features_dict = results_dict[lead_lag_key]['model_features']        
    model_names = [item for key, item in subset_dict.items() if 'model_name' in key]
    
    if all([name in model_features_dict.keys() for name in model_names]):
        return [], []
        
    spike_samples = results_dict[lead_lag_key]['sampled_data']['spike_samples']
    sample_info   = results_dict[lead_lag_key]['sampled_data']['sample_info']
    
    network_features = [np.full((spike_samples.shape[0], spike_samples.shape[1], params.networkFeatureBins), np.nan) for i in range(subset_dict['n_models'])]
    if 'split' in FN_key:
        for sampleNum, reach_idx in enumerate(sample_info['reach_idx']):
            if sampleNum % 500 == 0:
                print(sampleNum)
            if reach_idx % 2 == 0:
                weights = FN[1].copy()
            else:
                weights = FN[0].copy()
            
            if params.transpose_FN:
                weights = weights.T
            
            weights = modify_FN_weights(weights, subset_dict, sampleNum)
            
            for wIdx, w_arr in enumerate(weights):
                for leadBin in range(params.networkFeatureBins):
                    network_features[wIdx][:, sampleNum, leadBin] = w_arr @ spike_samples[:, sampleNum, (params.networkSampleBins-1) - leadBin] 
                
    else:
        weights = FN.copy()
        if params.transpose_FN:
            weights = weights.T
        weights = modify_FN_weights(weights, subset_dict)
        for wIdx, w_arr in enumerate(weights):
            for leadBin in range(params.networkFeatureBins):
                network_features[wIdx][..., leadBin] = w_arr @ spike_samples[..., (params.networkSampleBins-1) - leadBin]         

    if len(network_features) != len(model_names):
        raise Exception("The length of network features (should be 1 per model) does not match the length of the model_names list.")

    for model_name, features in zip(model_names, network_features):
        model_features_dict[model_name] = features
    
    return network_features, model_names
    
def shuffle_network_features(FN, spike_samples, rng, percentile=None, mode=None):
    
    shuffled_FN = FN.copy()
    if percentile is None:
        rng.shuffle(shuffled_FN, axis = 1)
    else:
        percentIdx = np.where(FN > np.percentile(FN, percentile))
        if mode == 'weights':
            for source_unit in np.unique(percentIdx[1]):
                target_unit = percentIdx[0][percentIdx[1] == source_unit]
                shuffled_FN[target_unit, source_unit] = FN[source_unit, rng.permutation(target_unit)]     
        elif mode == 'topology':
            shuffled_FN[percentIdx[0], percentIdx[1]] = rng.permutation(shuffled_FN[percentIdx[0], percentIdx[1]])
    
    shuffled_weights_network_features = np.empty((spike_samples.shape[0], spike_samples.shape[1], params.networkFeatureBins))
    for leadBin in range(params.networkFeatureBins):
        shuffled_weights_network_features[..., leadBin] = shuffled_FN @ spike_samples[..., (params.networkSampleBins-1) - leadBin] 
    return shuffled_weights_network_features 

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

def train_and_test_glm(traj_features, network_features, spike_samples, model_name, training_glm_source, RNGs, lead_lag_key):   
    
    areaUnderROC = np.empty((spike_samples.shape[0], params.num_model_samples))
    trainingSet_areaUnderROC = np.full((spike_samples.shape[0], params.num_model_samples), np.nan)

    aic          = np.empty_like(areaUnderROC)
    loglike = np.empty_like(aic)
    coefs = np.empty((traj_features.shape[-1] + network_features.shape[-1] + 1, spike_samples.shape[0], params.num_model_samples))
    pVals = np.empty_like(coefs)  
    
    allTrainPredictions = []
    all_encoding_models = []
    for n, split_rng in enumerate(RNGs['train_test_split']):

        try:
            trainPredict_encoding_models = results_dict[lead_lag_key]['model_results'][training_glm_source]['encoding_models']
        except:
            pass

        # Create train/test datasets for cross-validation
        print(lead_lag_key + ', ' + model_name + ', iteration = ' +str(n))
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
                    print('unit ' + str(unit) + ' had only ' + str(np.sum(spikes[testIdx] >= 1)) + ' spikes in the sampled time windows and is removed from analysis')
            
        # Train GLM
        predictions = []
        trainPredictions = []
        encodingModel_list = []
        for unit, (trainSpks, trainFts, testFts, shuf_rng) in enumerate(zip(trainSpikes, trainFeatures, testFeatures, RNGs['spike_shuffle'])):
            if 'shuffle' in model_name:
                trainSpks = shuf_rng.permutation(trainSpks)
            glm = sm.GLM(trainSpks,
                         sm.add_constant(trainFts), 
                         family=sm.families.Poisson(link=sm.families.links.log()))
            encodingModel = glm.fit()
            
            coefs  [:, unit, n] = encodingModel.params            
            pVals  [:, unit, n] = np.round(encodingModel.pvalues, decimals = 4)            
            aic    [   unit, n] = round(encodingModel.aic, 1)
            loglike[   unit, n] = round(encodingModel.llf, 1)
            
            encodingModel_list.append(encodingModel)
            predictions.append(encodingModel.predict(sm.add_constant(testFts)))
            
            if training_glm_source:
                try:
                    trainPredictions.append(trainPredict_encoding_models[n][unit].predict(sm.add_constant(trainFts))) 
                except:
                    trainPredictions.append(encodingModel.predict(sm.add_constant(trainFts))) 
                
        # Test GLM --> area under ROC
        areaUnderROC[:, n], allHitProbs, allFalsePosProbs = compute_area_under_ROC(predictions, testSpikes)

        # Get AUC on training data
        if training_glm_source:
            trainingSet_areaUnderROC[:, n], _, _ = compute_area_under_ROC(trainPredictions, trainSpikes)
            
        if model_name == params.encoding_model_for_trainPredictions:
            # allTrainPredictions.append(trainPredictions)
            all_encoding_models.append(encodingModel_list)
            
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
    
    model_results = {'AUC'           : areaUnderROC,
                     'trainAUC'      : trainingSet_areaUnderROC,
                     'coefs'         : coefs,
                     'pvals'         : pVals,
                     'AIC'           : aic,
                     'logLikelihood' : loglike,
                     'encoding_models' : all_encoding_models,
                     'all_predictions_on_training_set': allTrainPredictions,
                     'description'   : description}    
    
    return model_results

if __name__ == "__main__":
    
    with open(pkl_infile, 'rb') as f:
        results_dict = dill.load(f)
    
    if not run_model_only:
        
        with NWBHDF5IO(nwb_infile, 'r') as io:
            nwb = io.read()

            FN = nwb.scratch[params.FN_key].data[:]
                
        units_res = results_dict[params.best_lead_lag_key]['all_models_summary_results']
        electrode_distances = get_interelectrode_distances_by_unit(units_res, array_type='utah')
        
        RNGs = {'train_test_split'      : [np.random.default_rng(n) for n in range(params.num_model_samples)],
                'partial_traj'          : [np.random.default_rng(n) for n in range(1000,  1000+params.num_model_samples)],
                'spike_shuffle'         : [np.random.default_rng(n) for n in range(5000,  5000+results_dict[params.best_lead_lag_key]['sampled_data']['spike_samples'].shape[0])],
                'weight_shuffle'        : [np.random.default_rng(n) for n in range(10000, 10000+params.num_model_samples)],
                'network_input_shuffles': [[np.random.default_rng(n) for n in range(unit_idx*2000,  unit_idx*2000+params.num_model_samples)] for unit_idx in range(units_res.shape[0])]}

        subset_dict = {
                       'full_FN'        :     {'model_name'          : 'full_FN',
                                               'mode'                : 'full',
                                               'n_models'            : 1,
                                               'glm_source'          : params.encoding_model_for_trainPredictions},
                       'zero_dist_FN'   :     {'model_name'          : 'zero_dist_FN',
                                               'mode'                : 'distance',
                                               'n_models'            : 1,
                                               'modify_method'       : 'remove',
                                               'upper_bound'         : 1,
                                               'lower_bound'         : None,
                                               'electrode_distances' : electrode_distances,
                                               'glm_source'          : params.encoding_model_for_trainPredictions},
                        'nonzero_dist_FN':    {'model_name'         : 'nonzero_dist_FN',
                                               'mode'                : 'distance',
                                               'n_models'            : 1,
                                               'modify_method'       : 'remove',
                                               'upper_bound'         : None,
                                               'lower_bound'         : 1,
                                               'electrode_distances' : electrode_distances,
                                               'glm_source'          : params.encoding_model_for_trainPredictions},
                        'tuning_of_inputs':   {'model_name'          : 'tuned_inputs_FN',
                                               'anti_model_name'     : 'untuned_inputs_FN',
                                               'mode'                : 'tuning',
                                               'n_models'            : 2, 
                                               'bound_type'          : 'proportion',
                                               'modify_method'       : 'shuffle',
                                               'rng'                 : RNGs['network_input_shuffles'],
                                               'upper_bound'         : None,
                                               'lower_bound'         : params.significant_proportion_thresh,
                                               'unit_info'           : units_res.loc[:, ['proportion_sign']].copy(),
                                               'electrode_distances' : electrode_distances,
                                               'glm_source'          : params.encoding_model_for_trainPredictions},
                        'cortical_area':      {'model_name_1'        : 'permuted_motor_inputs_FN',
                                               'model_name_2'        : 'permuted_3a_inputs_FN',
                                               'model_name_3'        : 'permuted_3b_inputs_FN',
                                               'model_name_4'        : 'permuted_sensory_inputs_FN',
                                               'mode'                : 'cortical_area',
                                               'n_models'            : 4, 
                                               'bound_type'          : 'proportion',
                                               'modify_method'       : 'shuffle',
                                               'rng'                 : RNGs['network_input_shuffles'],
                                               'upper_bound'         : None,
                                               'lower_bound'         : params.significant_proportion_thresh,
                                               'cortical_area'       : units_res.loc[:, ['cortical_area']].copy().values,
                                               'electrode_distances' : electrode_distances,
                                               'glm_source'          : params.encoding_model_for_trainPredictions}
                      }
        for lead_lag_key in [params.best_lead_lag_key]:
            for model_key in subset_dict.keys():

                if model_key in remove_models:
                    tmp_model_names = [item for key, item in subset_dict[model_key].items() if 'model_name' in key]                
                    for tmp_mod in tmp_model_names:
                        tmp_results_key = '%s_%s' % (params.primary_traj_model, tmp_mod)
                        del results_dict[lead_lag_key]['model_features'][tmp_mod]
                        del results_dict[lead_lag_key]['model_results'][tmp_results_key] 
                    

                network_features_list, model_names = create_network_features_and_store_in_dict(FN, 
                                                                                               subset_dict[model_key],
                                                                                               lead_lag_key=lead_lag_key,
                                                                                               FN_key = params.FN_key)
                
                for network_features, model_name in zip(network_features_list, model_names):
                    model_results_key = '%s_%s' % (params.primary_traj_model, model_name)
                    if model_results_key in results_dict[lead_lag_key]['model_results'].keys():
                        continue
                    
                    model_results = train_and_test_glm(results_dict[lead_lag_key]['model_features'][params.primary_traj_model], 
                                                       network_features, 
                                                       results_dict[lead_lag_key]['sampled_data']['spike_samples'], 
                                                       model_results_key,
                                                       subset_dict[model_key]['glm_source'],
                                                       RNGs,
                                                       lead_lag_key)
                    
                    results_dict[lead_lag_key]['model_results'][model_results_key] = model_results
                
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
                                                       lead_lag_key)
                    
                    results_dict[lead_lag_key]['model_results'][model_results_key] = model_results
                                    
        
        with open(pkl_outfile, 'wb') as f:
            dill.dump(results_dict, f, recurse=True)  

'''
    To do:
        - Make entry and model in subset_dict for inter- vs intra-area connections
        - Make entry and model in subset_dict for narrow vs wide-spiking units
        - Re-run "tuning_of_inputs" with percentile bound in instead of significance bound, or with higher significance bound (used 0.95 previously which has overlapping populations of tuned/non-tuned)
'''
        
