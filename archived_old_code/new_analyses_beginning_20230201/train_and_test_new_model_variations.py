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


sys.path.insert(0, '/project/nicho/projects/marmosets/code_database/analysis/trajectory_encoding_model/')
from utils_before_nwb import save_all_models_dict, load_all_models_dict, get_single_lead_lag_models, load_channel_map_from_prb, fix_unit_info_elec_labels, get_interelectrode_distances_by_unit, load_data, choose_units_for_model

class path:
    storage = '/project2/nicho/dalton/processed_datasets'
    intermediate_save_path = '/project2/nicho/dalton/analysis/encoding_model/intermediate_variable_storage'
    new_save_path = '/project2/nicho/dalton/analysis/encoding_model/intermediate_variable_storage/10pt0_ms_bins/data_updated_february_2023'
    plots = '/project2/nicho/dalton/analysis/encoding_model/plots'
    date = '20210211'
    
    
class params:
    spkSampWin = 0.01
    trajShift = 0.05 #sample every 50ms
    lead = [0.2] #[0.2, 0.15, 0.1, 0.05, 0] #[0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0] #[0.05] # 100ms lead time
    lag = [0.3] #[0.2, 0.25, 0.3, 0.35, 0.4] #[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4] #[0.35] # 300ms lag time
    lead_to_analyze = [0.2]
    lag_to_analyze  = [0.3]
    spkRatioCheck = 'off'
    normalize = 'off'
    numThresh = 100
    trainRatio = 0.9
    numIters = 100
    minSpikeRatio = .005
    nDims = 3
    nShuffles = 1
    starting_eventNum = None 
    frate_thresh = 2
    fps = 150
    
    pca_var_thresh = 0.9# MAKE SURE that the idx being pulled is for the hand in run_pca_on_trajectories()
    include_avg_speed = False
    include_avg_pos = False
    network = 'on'
    hand_traj_idx = 0 
    FN_source = 'split_reach_FNs'
    transpose_FN = True
    
    networkSampleBins = 3
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
        weights[elec_dist > subset_dict['upper_bound']] = 0
    if subset_dict['lower_bound'] is not None:
        if subset_dict['bound_type'] == 'proportion':
            pro_inputs_to_keep  = list(unit_info_sorted.loc[unit_info_sorted.proportion_sign >= subset_dict['lower_bound'], 'original_index'].values)
            num_to_keep = len(pro_inputs_to_keep)
            pro_inputs_to_keep.extend(list(unit_info_sorted.original_index.iloc[len(pro_inputs_to_keep):len(pro_inputs_to_keep)+10]))         
            anti_inputs_to_keep = list(unit_info_sorted.original_index.iloc[-len(pro_inputs_to_keep):].values)

            pro_weights  = weights.copy()
            anti_weights = weights.copy()

            seed_count = 0
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
                
                pro_remove  = [u for u in range(weights.shape[1]) if u not in unit_pro_to_keep ]
                anti_remove = [u for u in range(weights.shape[1]) if u not in unit_anti_to_keep]
                pro_inputs [ pro_remove] = 0
                anti_inputs[anti_remove] = 0
                
                
                
                
            # inputs_to_remove      = unit_info_sorted.loc[unit_info_sorted.proportion_sign < subset_dict['lower_bound'], 'original_index'].values
            # anti_inputs_to_remove = unit_info_sorted.original_index[:len(inputs_to_remove)].values 
            # num_units_to_use = weights.shape[0] - len(inputs_to_remove)            
            # for unit, (pro_inputs, anti_inputs, dist) in enumerate(zip(pro_weights, anti_weights, elec_dist)):
                # unit_pro_remove  = inputs_to_remove.copy()
                # unit_anti_remove = anti_inputs_to_remove.copy()
                # pro_inputs [unit_pro_remove ] = 0
                # anti_inputs[unit_anti_remove] = 0
                # anti_zero_count = sum(dist[anti_inputs > 0] == 0)  
                # pro_zero_count  = sum(dist[ pro_inputs > 0] == 0)  
                # anti_total      = sum(anti_inputs > 0) 
                # pro_total       = sum( pro_inputs > 0)
                # while anti_zero_count != pro_zero_count or anti_total != pro_total or pro_total < num_units_to_use:
                #     if pro_total > anti_total:
                #         next_input = unit_info_sorted.original_index.iloc[-anti_total - 2]
                #         next_dist  = dist[next_input]
                #         if pro_zero_count > anti_zero_count:
                #             if next_dist == 0:
                #                 tmp = []
                #             else:
                #                 anti_inputs[next_input] = weights[unit, next_input]      
                                
                            
                #     anti_zero_count = sum(dist[anti_inputs > 0] == 0)  
                #     pro_zero_count  = sum(dist[ pro_inputs > 0] == 0)  
                #     anti_total      = sum(anti_inputs > 0) 
                #     pro_total       = sum( pro_inputs > 0)                                    
                
                
            # pro_weights[:, inputs_to_remove] = 0
            
            # zerodist = []
            # nonzerodist = []
            # for unit, (inputs, dist) in enumerate(zip(pro_weights, elec_dist)):
            #     zerodist.append   ( sum(dist[inputs > 0] == 0)   )
            #     nonzerodist.append( sum(dist[inputs > 0] >  0)   )    
            # pro_df = pd.DataFrame(data = zip(zerodist, nonzerodist),
            #                       columns = ['zero', 'nonzero'])

            # anti_inputs_to_remove = unit_info_sorted.original_index[:len(inputs_to_remove)].values 
            # anti_weights = weights.copy()
            # anti_weights[:, anti_inputs_to_remove] = 0
            
            # zerodist = []
            # nonzerodist = []
            # for unit, (inputs, dist) in enumerate(zip(anti_weights, elec_dist)):
            #     zerodist.append   ( sum(dist[inputs > 0] == 0)   )
            #     nonzerodist.append( sum(dist[inputs > 0] >  0)   )    
            # anti_df = pd.DataFrame(data = zip(zerodist, nonzerodist),
            #                        columns = ['zero', 'nonzero'])
            
            # zerodist_min_counts = [min(pro, anti) for pro, anti in zip(pro_df.zero, anti_df.zero)]
                        
    # pro_weights[pro_weights == 0] = np.nan
    # anti_weights[anti_weights == 0] = np.nan
    # plot_idxs = range(140, 169)
    # fig, ax = plt.subplots()
    # ax.errorbar(x = plot_idxs, y = np.nanmean(pro_weights, axis=1)[plot_idxs], yerr = np.nanstd(pro_weights, axis=1)[plot_idxs], mfc='red', mec='red')
    # # ax.errorbar(x = plot_idxs, y = np.nanmean(anti_weights, axis=1)[plot_idxs], yerr = np.nanstd(anti_weights, axis=1)[plot_idxs], mfc='blue', mec='blue')
    # plt.show()
    
    return pro_weights, anti_weights

def modify_FN_weights(weights, subset_dict):
    if subset_dict['mode'] == 'distance':
        weights = modify_FN_weights_by_distance(weights, subset_dict)
    elif subset_dict['mode'] == 'tuning':
        pro_weights, anti_weights = modify_FN_weights_by_tuning(weights, subset_dict)
        weights = [pro_weights, anti_weights]

    if type(weights) != list:
        weights = [weights]
        
    return weights

def compute_network_features_with_subset(FN, sampledSpikes, sample_info, subset_dict, FN_source = 'split_reach_FNs'):
    FN_tmp = FN[FN_source]
    
    network_features = [np.full((sampledSpikes.shape[0], sampledSpikes.shape[1], params.networkFeatureBins), np.nan) for i in range(subset_dict['n_models'])]
    if 'split' in FN_source:
        for sampleNum, kinIdx in enumerate(sample_info['kinIdx']):
            print(sampleNum)
            # if sampleNum > 10:
            #     break
            if kinIdx % 2 == 0:
                weights = FN_tmp[1].copy()
            else:
                weights = FN_tmp[0].copy()
            
            if params.transpose_FN:
                weights = weights.T
            
            weights = modify_FN_weights(weights, subset_dict)
            
            for wIdx, w_arr in enumerate(weights):
                for leadBin in range(params.networkFeatureBins):
                    network_features[wIdx][:, sampleNum, leadBin] = w_arr @ sampledSpikes[:, sampleNum, (params.networkSampleBins-1) - leadBin] 
                
    else:
        weights = FN_tmp.copy()
        if params.transpose_FN:
            weights = weights.T
        weights = modify_FN_weights(weights, subset_dict)
        for wIdx, w_arr in enumerate(weights):
            for leadBin in range(params.networkFeatureBins):
                network_features[wIdx][..., leadBin] = w_arr @ sampledSpikes[..., (params.networkSampleBins-1) - leadBin]         

    model_names = [item for key, item in subset_dict.items() if 'model_name' in key]

    if len(network_features) != len(model_names):
        raise Exception("The length of network features (should be 1 per model) does not match the length of the model_names list.")

    return network_features, model_names
    
def shuffle_network_features(FN, sampledSpikes, rng, percentile=None, mode=None):
    
    shuffled_FN = FN.copy()
    if percentile is None:
        rng.shuffle(shuffled_FN, axis = 1)
    else:
        if mode == 'weights':
            percentIdx = np.where(FN > np.percentile(FN, percentile))
            for presyn in np.unique(percentIdx[0]):
                postsyn = percentIdx[1][percentIdx[0] == presyn]
                shuffled_FN[presyn, postsyn] = FN[presyn, rng.permutation(postsyn)]     
        elif mode == 'topology':
            percentIdx = np.where(FN > np.percentile(FN, percentile))
            shuffled_FN[percentIdx[0], percentIdx[1]] = rng.permutation(shuffled_FN[percentIdx[0], percentIdx[1]])
    
    shuffled_weights_network_features = np.empty((sampledSpikes.shape[0], sampledSpikes.shape[1], params.networkFeatureBins))
    for leadBin in range(params.networkFeatureBins):
        shuffled_weights_network_features[..., leadBin] = shuffled_FN @ sampledSpikes[..., (params.networkSampleBins-1) - leadBin] 
    return shuffled_weights_network_features 
   
def train_and_test_glm(traj_features, network_features, sampledSpikes, model_name, RNGs):   
    areaUnderROC = np.empty((sampledSpikes.shape[0], params.numIters))
    aic          = np.empty_like(areaUnderROC)
    
    if model_name == 'network_partial_traj':
        coefs = np.empty((traj_features.shape[-1] + 1, sampledSpikes.shape[0], params.numIters))
        pVals = np.empty_like(coefs)
    elif model_name == 'pathlet_top_two_comps':
        coefs = np.empty((3 + 3*np.sum(params.include_avg_pos) + np.sum(params.include_avg_speed), sampledSpikes.shape[0], params.numIters))
        pVals = np.empty_like(coefs)        
    else:        
        coefs = np.empty((traj_features.shape[-1] + network_features.shape[-1] + 1, sampledSpikes.shape[0], params.numIters))
        pVals = np.empty_like(coefs)    
    
    for n, (split_rng, traj_rng, FN_rng) in enumerate(zip(RNGs['train_test_split'], RNGs['partial_traj'], RNGs['weight_shuffle'])):

        # Create train/test datasets for cross-validation
        print(model_name + ', iteration = ' +str(n))
        testSpikes = []
        trainSpikes = []
        trainFeatures = []
        testFeatures = []
        for unit, spikes in enumerate(sampledSpikes[..., -1]):
            
            spikeIdxs   = np.where(spikes >= 1)[0]
            noSpikeIdxs = np.where(spikes == 0)[0]
                
            idxs = np.union1d(spikeIdxs, noSpikeIdxs)
            trainIdx = np.hstack((split_rng.choice(spikeIdxs  , size = round(params.trainRatio*len(spikeIdxs  )), replace = False), 
                                  split_rng.choice(noSpikeIdxs, size = round(params.trainRatio*len(noSpikeIdxs)), replace = False)))
            testIdx  = np.setdiff1d(idxs, trainIdx)
                
            if np.sum(spikes[testIdx] >= 1) / np.sum(spikes[testIdx] == 0) >= params.minSpikeRatio:
                trainSpikes.append(spikes[trainIdx])
                testSpikes.append(spikes[testIdx])
                
                if model_name == 'full':
                    trainFeatures.append(np.hstack((traj_features[trainIdx], network_features[unit, trainIdx])))
                    testFeatures.append (np.hstack((traj_features[testIdx ], network_features[unit, testIdx ])))   
                elif model_name == 'network':
                    trainFeatures.append(network_features[unit, trainIdx])
                    testFeatures.append (network_features[unit, testIdx ])  
                elif model_name in ['network_FN_shuffled', 'network_FN_25percent_shuffled']:
                    if 'percent' in mode: 
                        shuffled_weights_network_features = shuffle_network_features(FN, sampledSpikes, FN_rng, percentile=100-int(model_name.split('percent')[0][-2:]))
                    else:
                        shuffled_weights_network_features = shuffle_network_features(FN, sampledSpikes, FN_rng, percentile=None)
                    trainFeatures.append(shuffled_weights_network_features[unit, trainIdx])
                    testFeatures.append (shuffled_weights_network_features[unit, testIdx ]) 
                elif model_name in ['full_FN_shuffled', 'full_FN_25percent_shuffled_topology', 'full_FN_25percent_shuffled_weights']:
                    if 'percent' in model_name: 
                        shuffled_weights_network_features = shuffle_network_features(FN, sampledSpikes, FN_rng, percentile=100-int(model_name.split('percent')[0][-2:]), mode=model_name.split('shuffled_')[1])
                    else:
                        shuffled_weights_network_features = shuffle_network_features(FN, sampledSpikes, FN_rng, percentile=None)
                    trainFeatures.append(np.hstack((traj_features[trainIdx], shuffled_weights_network_features[unit, trainIdx])))
                    testFeatures.append (np.hstack((traj_features[testIdx ], shuffled_weights_network_features[unit, testIdx ]))) 
                elif model_name in ['traj', 'short_traj', 'shuffle']:
                    trainFeatures.append(traj_features[trainIdx])
                    testFeatures.append (traj_features[testIdx])
                elif model_name == 'network_partial_traj':
                    trajFeatureIdx = traj_rng.choice(np.arange(traj_features.shape[-1] - 3*np.sum(params.include_avg_pos) - np.sum(params.include_avg_speed)),
                                                     size = traj_features.shape[-1]-3*np.sum(params.include_avg_pos)-np.sum(params.include_avg_speed) - params.networkFeatureBins,
                                                     replace = False)
                    trainFeatures.append(np.hstack((traj_features[np.ix_(trainIdx, trajFeatureIdx)], network_features[unit, trainIdx])))
                    testFeatures.append (np.hstack((traj_features[np.ix_(testIdx , trajFeatureIdx)], network_features[unit, testIdx ]))) 
                elif model_name == 'pathlet_top_two_comps':
                    trajFeatureIdx = [0, 1]
                    trajFeatureIdx.extend(list(range(traj_features.shape[-1] - 3*np.sum(params.include_avg_pos) - np.sum(params.include_avg_speed), traj_features.shape[-1])))
                    trainFeatures.append(traj_features[np.ix_(trainIdx, trajFeatureIdx)])
                    testFeatures.append (traj_features[np.ix_(testIdx , trajFeatureIdx)])
                elif model_name in ['zero_dist_FN', 'nonzero_dist_FN', 'untuned_inputs_FN', 'tuned_inputs_FN', 'transposed_FN']:
                    trainFeatures.append(np.hstack((traj_features[trainIdx], network_features[unit, trainIdx])))
                    testFeatures.append (np.hstack((traj_features[testIdx ], network_features[unit, testIdx ])))                  
                else:
                    print('\n\n You must choose a mode from ["full", "network", "traj", "short_traj", "shuffle", "network_partial_traj", "pathlet_top_2_comps"] \n\n')
                
                # if n == 0:
                #     print('unit ' + str(unit) + ' is prepared for GLM with ' + str(int(params.trainRatio*100)) + '/' + str(int(100-params.trainRatio*100)) + 
                #           ' split, with train/test spikes = ' + str((np.sum(spikes[trainIdx] >= 1), np.sum(spikes[testIdx] >= 1))))        
            else:
                if n == 0:
                    print('unit ' + str(unit) + ' had only ' + str(np.sum(spikes[testIdx] >= 1)) + ' spikes in the sampled time windows and is removed from analysis')
            
        # Train GLM
        
        models = []
        predictions = []
        # trainPredictions = []
        for unit, (trainSpks, trainFts, testFts, shuf_rng) in enumerate(zip(trainSpikes, trainFeatures, testFeatures, RNGs['spike_shuffle'])):
            if model_name == 'shuffle':
                trainSpks = shuf_rng.permutation(trainSpks)
            glm = sm.GLM(trainSpks,
                         sm.add_constant(trainFts), 
                         family=sm.families.Poisson(link=sm.families.links.log()))
            encodingModel = glm.fit()
            
            coefs[:, unit, n] = encodingModel.params            
            pVals[:, unit, n] = np.round(encodingModel.pvalues, decimals = 4)            
            aic     [unit, n] = round(encodingModel.aic, 1)
            
            models.append(encodingModel)
            predictions.append(encodingModel.predict(sm.add_constant(testFts)))
            # trainPredictions.append(encodingModel.predict(sm.add_constant(trainFts))) 
            
        # Test GLM --> area under ROC
        
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
            
            areaUnderROC[unit, n] = auc(falsePosProb, hitProb)
            
            allHitProbs.append(hitProb)
            allFalsePosProbs.append(falsePosProb)
    
            # if areaUnderROC[unit, n] > 0.79 and areaUnderROC[unit, n] < 0.8:
            # if unit == 88:
            #     fig, ax = plt.subplots()
            #     ax.plot(preds)
            #     ax.set_title('Unit' + str(unit))
            #     tmp = np.array(testSpikes[unit], dtype=np.float16)
            #     tmp[tmp == 0] = np.nan
            #     tmp[~np.isnan(tmp)] = preds[~np.isnan(tmp)]
            #     ax.plot(tmp, 'o', c = 'orange')
            #     plt.show()
            
    #        
    # fig, ax = plt.subplots()
    # ax.plot(allFalsePosProbs[-1], allHitProbs[-1])
        
    #    plt.plot(np.linspace(0,1,params.numThresh), np.linspace(0,1,params.numThresh), '-k')
    #    plt.show()
        
        # # ROC on train data
        # areaUnderROC_train = []
        # for unit, preds in enumerate(trainPredictions):
        #     thresholds = np.linspace(preds.min(), preds.max(), params.numThresh)
        #     hitProb = np.empty((len(thresholds),))
        #     falsePosProb = np.empty((len(thresholds),))
        #     for t, thresh in enumerate(thresholds):    
        #         posIdx = np.where(preds > thresh)
        #         hitProb[t] = np.sum(trainSpikes[unit][posIdx] >= 1) / np.sum(trainSpikes[unit] >= 1)
        #         falsePosProb[t] = np.sum(trainSpikes[unit][posIdx] == 0) / np.sum(trainSpikes[unit] == 0)
            
        #     areaUnderROC_train.append(auc(falsePosProb, hitProb))
            
        # # plt.plot(falsePosProb, hitProb)
        
        # # plt.plot(np.linspace(0,1,params.numThresh), np.linspace(0,1,params.numThresh), '-k')
        # # plt.show()
            
        # aucComb.append(np.vstack((np.array(areaUnderROC), np.array(areaUnderROC_train))).transpose())
        
    metadata = 'Results for %s, model_name = %s. To understand dimensions: there are %d units, \
        %d model parameters (1 constant, %d kinematic features, and %d network features) and \
        %d shuffles of the train/test split. The order of model parameters is the constant, then \
        the trajectory features projected onto principal components, then the 3 average position \
        and 1 average speed terms if they are enabled, then the network interaction terms, \
        starting with lead=0 and working backward (lead=1, lead=2, etc). For this model, \
        avg_pos=%s, avg_speed=%s, normalize=%s.' % (path.date, model_name, areaUnderROC.shape[0], coefs.shape[0], 
                                                    traj_features.shape[-1], params.networkFeatureBins, 
                                                    params.numIters, params.include_avg_pos, 
                                                    params.include_avg_speed, params.normalize)
    
    model_results = {'AUC'         : areaUnderROC,
                     'param_coefs' : coefs,
                     'param_pvals' : pVals,
                     'AIC'         : aic,
                     'metadata'    : metadata}    
    
    return model_results

if __name__ == "__main__":
    
    spike_data, kinematics, analog_and_video, FN = load_data(path)         

    # with open(os.path.join(path.intermediate_save_path, '10pt0_ms_bins', '20210211_model_trajectories_and_spikes_lead_200_lag_300_shift_50_with_network_lags.pkl'), 'rb') as f:
    #     trajectoryList, shortTrajectoryList, avgPos, avgSpeed, sampledSpikes, reachSpikes, sample_info, unit_info = dill.load(f)  
        
    # del unit_info
    
    # models_dict_path = os.path.join(path.new_save_path, 'all_models_data_dict.pkl')
    models_dict_path = os.path.join(path.new_save_path, 'all_models_data_dict_beforeAdding_Tuning.pkl')
    new_models_dict_path = os.path.join(path.new_save_path, 'all_models_data_dict_with_pt95_tuning_thresh.pkl')
    all_models_data = load_all_models_dict(models_dict_path) 
    
    single_lead_lag_models, ll_idx = get_single_lead_lag_models(all_models_data, params.lead[0], params.lag[0])
    unit_info = single_lead_lag_models['unit_info']
    
    chan_map_df = load_channel_map_from_prb(marm = 'Tony')   
    
    unit_info = fix_unit_info_elec_labels(unit_info, chan_map_df)
    
    electrode_distances = get_interelectrode_distances_by_unit(unit_info, chan_map_df, array_type='utah')

    RNGs = {'train_test_split' : [np.random.default_rng(n) for n in range(params.numIters)],
            'partial_traj'     : [np.random.default_rng(n) for n in range(1000,  1000+params.numIters)],
            'spike_shuffle'    : [np.random.default_rng(n) for n in range(5000,  5000+single_lead_lag_models['sampled_spikes'].shape[0])],
            'weight_shuffle'   : [np.random.default_rng(n) for n in range(10000, 10000+params.numIters)]}

    # unit_info.drop('network_FN_10percent_shuffled_AUC', axis=1, inplace=True)
    # unit_info.drop('full_FN_10percent_shuffled_AUC', axis=1, inplace=True)
    # all_model_results['model_results'] = [model_res for model_res, name in zip(all_model_results['model_results'], all_model_results['model_names'])
    #                                       if '10percent' not in name]
    # all_model_results['model_names'] = [name for name in all_model_results['model_names'] if '10percent' not in name]
    
    
    subset_dict = {'zero_dist_FN':        {'model_name'          : 'zero_dist_FN',
                                           'mode'                : 'distance',
                                           'n_models'            : 1,
                                           'upper_bound'         : 1,
                                           'lower_bound'         : None,
                                           'electrode_distances' : electrode_distances},
                    'nonzero_dist_FN':    {'model_name'         : 'nonzero_dist_FN',
                                           'mode'                : 'distance',
                                           'n_models'            : 1,
                                           'upper_bound'         : None,
                                           'lower_bound'         : 1,
                                           'electrode_distances' : electrode_distances},
                    'tuning_of_inputs':   {'model_name'          : 'tuned_inputs_FN',
                                           'anti_model_name'     : 'untuned_inputs_FN',
                                           'mode'                : 'tuning',
                                           'n_models'            : 2, 
                                           'bound_type'          : 'proportion',
                                           'upper_bound'         : None,
                                           'lower_bound'         : 0.95,
                                           'unit_info'           : unit_info.loc[:, ['tuning', 'proportion_sign']].copy(),
                                           'electrode_distances' : electrode_distances},
                    'untuned_inputs_FN':  {'model_name'          : 'untuned_inputs_FN',
                                           'mode'                : 'tuning',
                                           'n_models'            : 1,
                                           'bound_type'          : 'quantity',
                                           'upper_bound'         : None,
                                           'lower_bound'         : None,
                                           'unit_info'           : unit_info.loc[:, ['tuning', 'proportion_sign']].copy(),
                                           'electrode_distances' : electrode_distances},
                    'transposed_FN':      {'model_name'  : 'transposed_FN',
                                           'mode'        : 'transpose'}                 
                  }
    
    model_key = 'tuning_of_inputs'
    network_features_list, model_names = compute_network_features_with_subset(FN, 
                                                                              single_lead_lag_models['sampled_spikes'], 
                                                                              single_lead_lag_models['sample_info'], 
                                                                              subset_dict[model_key], 
                                                                              FN_source = params.FN_source)

    
    for network_features, model_name in zip(network_features_list, model_names):
        new_model_results = train_and_test_glm(single_lead_lag_models['traj_features'], 
                                               network_features, 
                                               single_lead_lag_models['sampled_spikes'], 
                                               model_name, 
                                               RNGs)

        unit_info['%s_%s' % (model_name, 'AUC')] = new_model_results['AUC'].mean(axis=-1)    
        
        all_models_data['%s_%s' % (model_name, 'network_features')] = [network_features if idx == ll_idx else [] for idx in range(len(all_models_data['lead_lag']))]
        all_models_data['unit_info'][ll_idx] = unit_info
        all_models_data['model_details'][ll_idx]['model_names'  ].append(model_name) 
        all_models_data['model_details'][ll_idx]['model_results'].append(new_model_results)
        
    save_all_models_dict(new_models_dict_path, all_models_data)   
        
    # '''
    #     Notes: 
    #         - Set up preceding and subsequent section to do pro (tuned) and anti (untuned) sets in the same run
    #         - Set up compute_network_features and modify_weights to take 100 samples of network features, accounting for zero and nonzero input counts
    #         - Set up train_and_test_glm to use 100 samples of network features
    # '''
####################################### KEEP FOR NOW
        # unit_info['%s_%s' % (subset_dict[model_key]['model_name'], 'AUC')] = new_model_results['AUC'].mean(axis=-1)    
        
        # all_models_data['%s_%s' % (subset_dict[model_key]['model_name'], 'network_features')] = [network_features if idx == ll_idx else [] for idx in range(len(all_models_data['lead_lag']))]
        # all_models_data['unit_info'][ll_idx] = unit_info
        # all_models_data['model_details'][ll_idx]['model_names'  ].append(subset_dict[model_key]['model_name']) 
        # all_models_data['model_details'][ll_idx]['model_results'].append(new_model_results)
        
        # save_all_models_dict(models_dict_path, all_models_data)
###############################################
    
    # network_FN_shuffled_model_results = train_and_test_glm(np.array([]) , network_features, sampledSpikes, 'network_FN_shuffled', RNGs)
    # full_FN_shuffled_model_results    = train_and_test_glm(traj_features, network_features, sampledSpikes, 'full_FN_shuffled'   , RNGs)
    # full_FN_25percent_shuffled_weights_model_results  = train_and_test_glm(traj_features, network_features, sampledSpikes, 'full_FN_25percent_shuffled_weights'    , RNGs)
    # full_FN_25percent_shuffled_topology_model_results = train_and_test_glm(traj_features, network_features, sampledSpikes, 'full_FN_25percent_shuffled_topology'   , RNGs)


    # unit_info['network_FN_shuffled_AUC'] = network_FN_shuffled_model_results['AUC'].mean(axis=-1)
    # unit_info['full_FN_shuffled_AUC'   ] = full_FN_shuffled_model_results   ['AUC'].mean(axis=-1)
    # unit_info['full_FN_25percent_shuffled_weights_AUC' ] = full_FN_25percent_shuffled_weights_model_results ['AUC'].mean(axis=-1)    
    # unit_info['full_FN_25percent_shuffled_topology_AUC'] = full_FN_25percent_shuffled_topology_model_results['AUC'].mean(axis=-1)    

    # all_model_results['model_results'].extend([full_FN_25percent_shuffled_weights_model_results, full_FN_25percent_shuffled_topology_model_results])
    # all_model_results['model_names'].extend(['full_FN_25percent_shuffled_weights', 'full_FN_25percent_shuffled_topology'])
    
    # with open(r'C:/Users/Dalton/Documents/lab_files/analysis_encoding_model/intermediate_variable_storage/20210211_encoding_model_results_network_on_lead_100_lag_300_shift_50_PCAthresh_90_norm_off_NETWORK_SHUFFLES.pkl', 'wb') as f:
    #     dill.dump([all_model_results, unit_info], f, recurse=True) 
    
    
