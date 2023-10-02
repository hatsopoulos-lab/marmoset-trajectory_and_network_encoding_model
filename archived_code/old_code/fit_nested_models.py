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


class path:
    storage = r'Z:/marmosets/processed_datasets'
    # intermediate_save_path = r'C:\Users\daltonm\Documents\Lab_Files\encoding_model\intermediate_variable_storage'
    intermediate_save_path = r'C:\Users\Dalton\Documents\lab_files\analysis_encoding_model\intermediate_variable_storage'
    date = '20210211'
    
class params:
    spkSampWin = 0.01
    trajShift = 0.05 #sample every 50ms
    lead = [0.1] #[0.2, 0.15, 0.1, 0.05, 0] #[0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0] #[0.05] # 100ms lead time
    lag = [0.3] #[0.2, 0.25, 0.3, 0.35, 0.4] #[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4] #[0.35] # 300ms lag time
    spkRatioCheck = 'off'
    normalize = 'on'
    numThresh = 1000
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
    
    networkSampleBins = 3
    networkFeatureBins = 2    
    
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

    FN_path = os.path.join(path.intermediate_save_path, 'FN_%s_fMI_10ms_bins_dict.pkl' % path.date)
    with open(FN_path, 'rb') as fp:
        FN = dill.load(fp)

    return spike_data, kinematics, analog_and_video, FN[0]

def compute_new_network_features(FN, sampledSpikes, sample_info, FN_source = 'split_reach_FNs'):
    FN_tmp = FN[FN_source]
    if 'split' in FN_source:
        tmp = []
        
    
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
    
def train_and_test_glm(traj_features, network_features, sampledSpikes, mode, RNGs):   
    areaUnderROC = np.empty((sampledSpikes.shape[0], params.numIters))
    aic          = np.empty_like(areaUnderROC)
    
    if mode == 'network_partial_traj':
        coefs = np.empty((traj_features.shape[-1] + 1, sampledSpikes.shape[0], params.numIters))
        pVals = np.empty_like(coefs)
    elif mode == 'pathlet_top_two_comps':
        coefs = np.empty((3 + 3*np.sum(params.include_avg_pos) + np.sum(params.include_avg_speed), sampledSpikes.shape[0], params.numIters))
        pVals = np.empty_like(coefs)        
    else:        
        coefs = np.empty((traj_features.shape[-1] + network_features.shape[-1] + 1, sampledSpikes.shape[0], params.numIters))
        pVals = np.empty_like(coefs)    
    
    for n, (split_rng, traj_rng, FN_rng) in enumerate(zip(RNGs['train_test_split'], RNGs['partial_traj'], RNGs['weight_shuffle'])):

        # Create train/test datasets for cross-validation
        print(mode + ', iteration = ' +str(n))
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
                
                if mode == 'full':
                    trainFeatures.append(np.hstack((traj_features[trainIdx], network_features[unit, trainIdx])))
                    testFeatures.append (np.hstack((traj_features[testIdx ], network_features[unit, testIdx ])))   
                elif mode == 'network':
                    trainFeatures.append(network_features[unit, trainIdx])
                    testFeatures.append (network_features[unit, testIdx ])  
                elif mode in ['network_FN_shuffled', 'network_FN_25percent_shuffled']:
                    if 'percent' in mode: 
                        shuffled_weights_network_features = shuffle_network_features(FN, sampledSpikes, FN_rng, percentile=100-int(mode.split('percent')[0][-2:]))
                    else:
                        shuffled_weights_network_features = shuffle_network_features(FN, sampledSpikes, FN_rng, percentile=None)
                    trainFeatures.append(shuffled_weights_network_features[unit, trainIdx])
                    testFeatures.append (shuffled_weights_network_features[unit, testIdx ]) 
                elif mode in ['full_FN_shuffled', 'full_FN_25percent_shuffled_topology', 'full_FN_25percent_shuffled_weights']:
                    if 'percent' in mode: 
                        shuffled_weights_network_features = shuffle_network_features(FN, sampledSpikes, FN_rng, percentile=100-int(mode.split('percent')[0][-2:]), mode=mode.split('shuffled_')[1])
                    else:
                        shuffled_weights_network_features = shuffle_network_features(FN, sampledSpikes, FN_rng, percentile=None)
                    trainFeatures.append(np.hstack((traj_features[trainIdx], shuffled_weights_network_features[unit, trainIdx])))
                    testFeatures.append (np.hstack((traj_features[testIdx ], shuffled_weights_network_features[unit, testIdx ]))) 
                elif mode in ['traj', 'short_traj', 'shuffle']:
                    trainFeatures.append(traj_features[trainIdx])
                    testFeatures.append (traj_features[testIdx])
                elif mode == 'network_partial_traj':
                    trajFeatureIdx = traj_rng.choice(np.arange(traj_features.shape[-1] - 3*np.sum(params.include_avg_pos) - np.sum(params.include_avg_speed)),
                                                     size = traj_features.shape[-1]-3*np.sum(params.include_avg_pos)-np.sum(params.include_avg_speed) - params.networkFeatureBins,
                                                     replace = False)
                    trainFeatures.append(np.hstack((traj_features[np.ix_(trainIdx, trajFeatureIdx)], network_features[unit, trainIdx])))
                    testFeatures.append (np.hstack((traj_features[np.ix_(testIdx , trajFeatureIdx)], network_features[unit, testIdx ]))) 
                elif mode == 'pathlet_top_two_comps':
                    trajFeatureIdx = [0, 1]
                    trajFeatureIdx.extend(list(range(traj_features.shape[-1] - 3*np.sum(params.include_avg_pos) - np.sum(params.include_avg_speed), traj_features.shape[-1])))
                    trainFeatures.append(traj_features[np.ix_(trainIdx, trajFeatureIdx)])
                    testFeatures.append (traj_features[np.ix_(testIdx , trajFeatureIdx)])
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
            if mode == 'shuffle':
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
        
    metadata = 'Results for %s, mode = %s. To understand dimensions: there are %d units, \
        %d model parameters (1 constant, %d kinematic features, and %d network features) and \
        %d shuffles of the train/test split. The order of model parameters is the constant, then \
        the trajectory features projected onto principal components, then the 3 average position \
        and 1 average speed terms if they are enabled, then the network interaction terms, \
        starting with lead=0 and working backward (lead=1, lead=2, etc). For this model, \
        avg_pos=%s, avg_speed=%s, normalize=%s.' % (path.date, mode, areaUnderROC.shape[0], coefs.shape[0], 
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

    spike_data, kinematics, analog_and_video, FN = load_data()        

    with open(r'C:/Users/Dalton/Documents/lab_files/analysis_encoding_model/intermediate_variable_storage/20210211_model_features_and_components_network_on_lead_100_lag_300_shift_50_PCAthresh_90_norm_off.pkl', 'rb') as f:
        traj_features, network_features, short_features, compsOut = dill.load(f)

    with open(r'C:/Users/Dalton/Documents/lab_files/analysis_encoding_model/intermediate_variable_storage/20210211_model_trajectories_and_spikes_lead_100_lag_300_shift_50_with_network_lags.pkl', 'rb') as f:
        trajectoryList, shortTrajectoryList, avgPos, avgSpeed, sampledSpikes, reachSpikes, sample_info, unit_info = dill.load(f)  
        
    del unit_info

    with open(r'C:/Users/Dalton/Documents/lab_files/analysis_encoding_model/intermediate_variable_storage/20210211_encoding_model_results_network_on_lead_100_lag_300_shift_50_PCAthresh_90_norm_off_NETWORK_SHUFFLES.pkl', 'rb') as f:
        all_model_results, unit_info = dill.load(f)    

    RNGs = {'train_test_split' : [np.random.default_rng(n) for n in range(params.numIters)],
            'partial_traj'     : [np.random.default_rng(n) for n in range(1000,  1000+params.numIters)],
            'spike_shuffle'    : [np.random.default_rng(n) for n in range(5000,  5000+sampledSpikes.shape[0])],
            'weight_shuffle'   : [np.random.default_rng(n) for n in range(10000, 10000+params.numIters)]}

    # unit_info.drop('network_FN_10percent_shuffled_AUC', axis=1, inplace=True)
    # unit_info.drop('full_FN_10percent_shuffled_AUC', axis=1, inplace=True)
    # all_model_results['model_results'] = [model_res for model_res, name in zip(all_model_results['model_results'], all_model_results['model_names'])
    #                                       if '10percent' not in name]
    # all_model_results['model_names'] = [name for name in all_model_results['model_names'] if '10percent' not in name]

    network_features = compute_new_network_features(FN, sampledSpikes, sample_info, FN_source = 'split_reach_FNs')

    # tmp = train_and_test_glm(traj_features, network_features, sampledSpikes, 'full', RNGs)

            
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
    
    
