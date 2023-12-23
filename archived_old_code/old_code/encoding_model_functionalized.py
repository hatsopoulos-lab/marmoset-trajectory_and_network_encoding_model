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
    storage = '/project/nicho/data/marmosets/processed_datasets'
    intermediate_save_path = '/project2/nicho/dalton/analysis/encoding_model/intermediate_variable_storage'
    date = '20210211'
    
class params:
    spkSampWin = 0.01
    trajShift = 0.05 #sample every 50ms
    lead = [0.3, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0  , 0  , 0  ] # [0.5, 0.4, 0.4, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0  , 0  , 0  ] #[0.2, 0.15, 0.1, 0.05, 0] #[0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0] #[0.05] # 100ms lead time
    lag  = [0.2, 0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.3, 0.4, 0.5] # [0  , 0  , 0.1, 0  , 0.1, 0.2, 0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.3, 0.4, 0.5] #[0.2, 0.25, 0.3, 0.35, 0.4] #[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4] #[0.35] # 300ms lag time
    spkRatioCheck = 'off'
    normalize = 'off' # NEED TO FIX NORMALIZATION SO THAT IT NORMALIZES ENTIRE TRAJECTORY SET PRIOR TO PCA, not individual chunks during collection!!
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
    FN_source = 'split_reach_FNs'
    hand_traj_idx = 0
    compute_shortTraj = True
    
    networkSampleBins = 3
    networkFeatureBins = 2    
     
def load_data():
    
    spike_path = glob.glob(os.path.join(path.storage, 'formatted_spike_dir', '%s*.pkl' % path.date))
    spike_path = [f for f in spike_path if 'sleep' not in f][0]
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
        # raster_list, FN = dill.load(fp)
        FN = dill.load(fp)

    return spike_data, kinematics, analog_and_video, FN[0]

def compute_derivatives(marker_pos, smooth = True):
    marker_vel = np.diff(marker_pos, axis = -1) * params.fps
    if smooth:
        for dim in range(3):
            marker_vel[dim] = gaussian_filter(marker_vel[dim], sigma=1.5)

    marker_acc = np.diff(marker_vel, axis = -1) * params.fps
    if smooth:
        for dim in range(3):
            marker_acc[dim] = gaussian_filter(marker_acc[dim], sigma=1.5)
    
    return marker_vel, marker_acc

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

def get_frames_to_sample(kin, vel, leadSamps, lagSamps, shortSamps100, shortSamps150, trajSampShift):
    
    traj_slices       = []
    short_traj_slices = []
    spike_samp_idx    = []
    for start, stop in zip(kin['starts'], kin['stops']):      
        tmp_reach_idxs = np.array(range(start, stop-1))
        tmp_reach = vel[0, 0, tmp_reach_idxs]
        #notnan_idx = tmp_reach_idxs[~np.isnan(tmp_reach)]
        tmp_reach_idxs = tmp_reach_idxs[~np.isnan(tmp_reach)]
        
        new_start = tmp_reach_idxs[0]
        new_stop  = tmp_reach_idxs[-1]
        
        # trajLength = new_stop + 1 - new_start    
        for centerIdx in range(new_start + leadSamps, new_stop - lagSamps, trajSampShift):
            traj_slices.append(slice(centerIdx - leadSamps, centerIdx + lagSamps - 1))
            short_traj_slices.append(slice(centerIdx + shortSamps100, centerIdx + shortSamps150))
            spike_samp_idx.append(centerIdx)            
        
    return traj_slices, short_traj_slices, spike_samp_idx 

def run_pca(traj, avgPos, avgSpeed, nComps = None, return_components = False, plot = False):
    # traj = StandardScaler().fit_transform(traj) 
    traj_tmp = traj.copy()
    
    # find out how many PCs to use
    pca = PCA()
    pca.fit(traj_tmp)      
    
    cumVar = np.cumsum(pca.explained_variance_ratio_)
    
    if nComps is None:
        cutComp = np.where(cumVar >= params.pca_var_thresh)[0][0]
        pca = PCA(n_components = cutComp+1)
    else:
        print((nComps, traj.shape[0], traj.shape[1]))
        nComps = min(nComps, traj.shape[0], traj.shape[1])
        cutComp = nComps-1
        print(nComps)
        pca = PCA(n_components = nComps)
        
    if plot:
        fig, ax = plt.subplots()
        ax.plot(cumVar, '-o')
        ax.plot(0.9*np.ones(np.shape(cumVar)), '--k', linewidth = 2)
        ax.vlines(cutComp, 0, 1, color='black', linewidth=2, linestyle='--')
        ax.set_ylabel('Cumulative Variance', fontsize = 14)
        ax.set_xlabel('Principal Component', fontsize = 14)
        ax.set_xticks([0, cutComp, len(cumVar) - 1])
        ax.set_xticklabels([1, cutComp+1, len(cumVar)])
        plt.show()
    projectedTraj = pca.fit_transform(traj)
    
    # if params.normalize == 'on':
    #     trajNorm = np.tile(np.expand_dims(np.linalg.norm(projectedTraj, axis = 0), 0), (projectedTraj.shape[0], 1))
    #     projectedTraj = projectedTraj / trajNorm
    
    if params.include_avg_pos and params.include_avg_speed:
        features = np.hstack((projectedTraj, np.expand_dims(avgSpeed[params.hand_traj_idx], axis=1), avgPos[params.hand_traj_idx]))
    elif params.include_avg_speed and not params.include_avg_pos:
        features = np.hstack((projectedTraj, np.expand_dims(avgSpeed[2], axis=1)))
    elif params.include_avg_pos and not params.include_avg_speed:
        features = np.hstack((projectedTraj, avgPos[2]))
    else:
        features = projectedTraj
    
    compsOut = pca.components_
    
    if return_components:
        return features, compsOut
    else:
        return features
    

def extract_features(trajectoryList, shortTrajectoryList, sampledSpikes, sample_info, avgPos, avgSpeed, nComps, plot = False):
    
    for mark, (traj, shortTraj) in enumerate(zip(trajectoryList, shortTrajectoryList)):
        traj = np.reshape(traj, (np.shape(traj)[0], np.shape(traj)[1] * np.shape(traj)[2]))
        if params.compute_shortTraj:
            shortTraj = np.reshape(shortTraj, (np.shape(shortTraj)[0], np.shape(shortTraj)[1] * np.shape(shortTraj)[2]))        
        if 'allTraj' not in locals():
            allTraj = traj
            if params.compute_shortTraj:            
                allShortTraj = shortTraj
        else:
            allTraj = np.hstack(allTraj, traj)
            if params.compute_shortTraj:
                allShortTraj = np.hstack(allShortTraj, shortTraj)
    
    if nComps is None:
        traj_features, compsOut = run_pca(allTraj,      avgPos, avgSpeed, nComps = None, return_components = True,  plot = plot)
        if params.compute_shortTraj:
            short_features          = run_pca(allShortTraj, avgPos, avgSpeed, nComps = compsOut.shape[0], return_components = False, plot = plot)    
        else:
            short_features = []
    else:
        traj_features, compsOut = run_pca(allTraj,      avgPos, avgSpeed, nComps = nComps, return_components = True,  plot = plot)
        if params.compute_shortTraj:
            short_features          = run_pca(allShortTraj, avgPos, avgSpeed, nComps = nComps, return_components = False, plot = plot)    
        else:
            short_features = []

    if params.network == 'on':
        network_features = np.empty((sampledSpikes.shape[0], sampledSpikes.shape[1], params.networkFeatureBins))
        FN_tmp = allData.FN[params.FN_source]
        if 'split' in params.FN_source:
            for sampleNum, kinIdx in enumerate(sample_info['kinIdx']):
                if kinIdx % 2 == 0:
                    weights = FN_tmp[1]
                else:
                    weights = FN_tmp[0]
                    
                for leadBin in range(params.networkFeatureBins):
                    network_features[:, sampleNum, leadBin] = weights @ sampledSpikes[:, sampleNum, (params.networkSampleBins-1) - leadBin] 
                    
        else:
            for leadBin in range(params.networkFeatureBins):
                network_features[..., leadBin] = allData.FN @ sampledSpikes[..., (params.networkSampleBins-1) - leadBin]   

    return traj_features, network_features, short_features, compsOut

def extract_traj_features_only(trajectoryList, plot = False):
    
    for mark, traj in enumerate(trajectoryList):
        traj = np.reshape(traj, (np.shape(traj)[0], np.shape(traj)[1] * np.shape(traj)[2]))        
        if 'allTraj' not in locals():
            allTraj = traj
        else:
            allTraj = np.hstack(allTraj, traj)
    
    traj_features, compsOut = run_pca(allTraj, False, False, nComps = None, return_components = True,  plot = plot) 

    return compsOut

def extract_trajectories_and_define_model_features(allData, lead, lag, nComps, filemod):
    
    try:
        with open(os.path.join(path.intermediate_save_path, '10pt0_ms_bins', '%s_model_trajectories_and_spikes_%s_with_network_lags.pkl' % (path.date, filemod.split('_PCAthresh')[0])), 'rb') as f:
            trajectoryList, shortTrajectoryList, avgPos, avgSpeed, sampledSpikes, reachSpikes, sample_info, unit_info = dill.load(f)   
        print('opened a saved set of trajectories and sample spikes, located at ' + (os.path.join(path.intermediate_save_path, 'model_trajectories_and_spikes_%s.pkl' % filemod.split('_PCAthresh')[0])))
    except:
        goalIdx = [idx for idx, exp in enumerate(allData.analog_and_video['experiments']) if 'free' not in exp][0]
        
        camPeriod = np.mean(np.diff(allData.analog_and_video['frameTimes_byEvent'][goalIdx][0]))  
        
        trajSampShift = int(np.round(params.trajShift / camPeriod))
        leadSamps = int(np.round(lead / camPeriod))
        lagSamps = int(np.round(lag / camPeriod))
        
        shortSamps100 = int(np.round(.1 / camPeriod))
        shortSamps150 = int(np.round(.15 / camPeriod))
            
        unit_info = choose_units_for_model(allData.spike_data)
        
        if params.starting_eventNum is None:
            first_event = kinematics[0]['event']
        else:
            first_event = params.starting_eventNum
        
        sample_kinIdx   = []
        sample_eventNum = []
        for kIdx, kin in enumerate(allData.kinematics):

            print((lead, lag, kIdx))            

            eventIdx = kin['event'] - 1
            
            handIdx     = [idx for idx, mark in enumerate(kin['marker_names']) if mark == 'hand'][0]
            shoulderIdx = [idx for idx, mark in enumerate(kin['marker_names']) if mark == 'shoulder'][0]
            
            eventExpTimes = allData.analog_and_video['frameTimes_byEvent'][goalIdx][eventIdx]
    
            pos = np.empty((1, 3, kin['position'].shape[-1]))
            vel = np.empty_like(pos[..., :-1])
            pos[0] = kin['position'][handIdx] - kin['position'][shoulderIdx]
            for mark in range(pos.shape[0]): 
                vel[mark], tmp_acc = compute_derivatives(pos[mark], smooth = True)
            
            traj_slices, short_traj_slices, spike_samp_idx = get_frames_to_sample(kin,
                                                                                  vel,
                                                                                  leadSamps, 
                                                                                  lagSamps, 
                                                                                  shortSamps100, 
                                                                                  shortSamps150, 
                                                                                  trajSampShift)
            
            numTraj = len(traj_slices)            
                    
            traj = np.empty((np.shape(vel)[0], numTraj, 3, traj_slices[0].stop - traj_slices[0].start))
            shortTraj = np.empty((np.shape(vel)[0], numTraj, 3, shortSamps150 - shortSamps100))
            avgSpeed = np.empty((np.shape(vel)[0], numTraj))
            avgPos = np.empty((np.shape(vel)[0], numTraj, 3))
            spikes = np.zeros((unit_info.shape[0], numTraj, params.networkSampleBins), dtype='int8')
            for mark in range(vel.shape[0]):
                for t, (slc, short_slc, spkIdx) in enumerate(zip(traj_slices, short_traj_slices, spike_samp_idx)):                    
                    traj[mark, t]      = vel[mark, :, slc]
                    if params.compute_shortTraj:
                        shortTraj[mark, t] = vel[mark, :, short_slc]                        
                    
                    avgSpeed[mark, t] = np.mean(np.linalg.norm(traj[mark, t], axis = -2))
                    avgPos[mark, t]   = np.mean(pos[mark, :, slc], axis = -1)
    
                    if params.normalize == 'on':
                        traj[mark, t] = traj[mark, t] / np.linalg.norm(traj[mark, t], axis = -2)
                        if params.compute_shortTraj:    
                            shortTraj[mark, t] = shortTraj[mark, t] / np.linalg.norm(shortTraj[mark, t], axis = -2)
                
                    # get spike/no-spike in 10ms window centered around idx
                    startBound = eventExpTimes[spkIdx] - (params.networkSampleBins - 0.5) * params.spkSampWin
                    stopBound  = eventExpTimes[spkIdx] + params.spkSampWin/2
                    bins = np.arange(startBound, stopBound, params.spkSampWin)
                    
                    bout_spikes = allData.spike_data['spike_times']
                    bout_idx = np.where((bout_spikes > startBound) & (bout_spikes < stopBound))[0]
                    bout_spikes = bout_spikes[bout_idx]
                    bout_clusters = allData.spike_data['spike_clusters'][bout_idx]
                    if mark == 0:
                        for s, cluster in enumerate(unit_info.cluster_id):
                            unitSpikes = bout_spikes[bout_clusters == cluster]
                            digitized_spikes = np.digitize(unitSpikes, bins) - 1
                            binned_spikes = np.bincount(digitized_spikes)
                            spikes[s, t, digitized_spikes] = binned_spikes[digitized_spikes]
                        sample_kinIdx.append(kIdx)
                        sample_eventNum.append(kin['event'])
    
            if eventIdx+1 == first_event: 
                stackedTraj      = traj
                if params.compute_shortTraj:
                    stackedShortTraj = shortTraj
                sampledSpikes    = spikes
                stackedSpeed     = avgSpeed
                stackedPos       = avgPos
            else: 
                stackedTraj = np.hstack((stackedTraj, traj))
                if params.compute_shortTraj:
                    stackedShortTraj = np.hstack((stackedShortTraj, shortTraj))
                sampledSpikes = np.hstack((sampledSpikes, spikes))
                stackedSpeed = np.hstack((stackedSpeed, avgSpeed))
                stackedPos = np.hstack((stackedPos, avgPos))
            
            if params.spkRatioCheck == 'on':
                bins = np.arange(np.floor(eventExpTimes[0] * 1e2), eventExpTimes[-1] * 1e2 + 1, 1) / 1e2
                # bins = np.arange(traj_slices[0].start, traj_slices[-1].stop)
                tmpSpikes = np.empty((unit_info.shape[0], len(bins) - 1), dtype=np.int8)
                for s, cluster in enumerate(unit_info.cluster_id):
                    unit_spikes = allData.spike_data['spike_times'][allData.spike_data['spike_clusters'] == cluster]
                    binnedSpikes = pd.DataFrame(data = unit_spikes[(unit_spikes >= eventExpTimes[0]) & 
                                                                   (unit_spikes <= eventExpTimes[-1])], 
                                                columns = ['spikeTimes'])
                    binnedSpikes['bins'] = pd.cut(binnedSpikes['spikeTimes'], bins = bins)
                    tmpSpikes[s] = np.array(binnedSpikes['bins'].value_counts(sort=False), dtype = np.int8)
                    
                if eventIdx+1 == first_event: 
                    reachSpikes = tmpSpikes
                else:
                    reachSpikes = np.hstack((reachSpikes, tmpSpikes))
            else:
                reachSpikes = []
                                   
        # rearrange traj array into a list of arrays, with each element being the array of trajectories for a single marker
        trajectoryList = [stackedTraj[mark] for mark in range(np.shape(stackedTraj)[0])]
        if params.compute_shortTraj:
            shortTrajectoryList = [stackedShortTraj[mark, ...] for mark in range(np.shape(stackedShortTraj)[0])]
        else:
            shortTrajectoryList = [0]*len(trajectoryList)
        avgPos   = [stackedPos  [mark, ...] for mark in range(np.shape(stackedTraj)[0])]
        avgSpeed = [stackedSpeed[mark, ...] for mark in range(np.shape(stackedTraj)[0])]
        
        sampledSpikes = np.delete(sampledSpikes, np.where(np.isnan(avgSpeed[0]))[0], axis = 1)
        
        for mark in range(len(trajectoryList)):
            samples_with_nan = np.where(np.isnan(avgSpeed[mark]))[0]
            trajectoryList[mark]      = np.delete(trajectoryList[mark],      samples_with_nan, axis = 0)      
            if params.compute_shortTraj:
                shortTrajectoryList[mark] = np.delete(shortTrajectoryList[mark], samples_with_nan, axis = 0) 
            avgPos[mark]              = np.delete(avgPos[mark],              samples_with_nan, axis = 0)      
            avgSpeed[mark]            = np.delete(avgSpeed[mark],            samples_with_nan, axis = 0)
            sample_kinIdx   = [kinIdx for i, kinIdx in enumerate(sample_kinIdx) if i not in samples_with_nan]
            sample_eventNum = [eNum   for i, eNum   in enumerate(sample_eventNum) if i not in samples_with_nan]
            
        sample_info = pd.DataFrame(data = zip(sample_kinIdx, sample_eventNum), columns = ['kinIdx', 'eventNum'])

        del stackedTraj, stackedPos, stackedSpeed
        if params.compute_shortTraj:
            del stackedShortTraj
            
        if params.spkRatioCheck == 'on':
            fullRatio = np.sum(reachSpikes >= 1) / np.sum(reachSpikes == 0)
            sampledRatio = np.sum(sampledSpikes[..., -1] >= 1) / np.sum(sampledSpikes[..., -1] == 0)
            print('ratio of spike/no-spike for full dataset is ' + str(fullRatio) + ', ratio for sampled set is ' + str(sampledRatio)) 
        
        if filemod is not None:
            with open(os.path.join(path.intermediate_save_path, '10pt0_ms_bins', '%s_model_trajectories_and_spikes_%s_with_network_lags.pkl' % (path.date, filemod.split('_PCAthresh')[0])), 'wb') as f:
                dill.dump([trajectoryList, shortTrajectoryList, avgPos, avgSpeed, sampledSpikes, reachSpikes, sample_info, unit_info], f, recurse=True)
    try:
        with open(os.path.join(path.intermediate_save_path, '10pt0_ms_bins', '%s_model_features_and_components_network_%s_%s.pkl'  % (path.date, params.FN_source, filemod)), 'rb') as f:
            traj_features, network_features, short_features, compsOut = dill.load(f)
        print('opened a saved set of features and components, located at ' + (os.path.join(path.intermediate_save_path, 'model_features_and_components_%s.pkl' % filemod)))

    except:
        traj_features, network_features, short_features, compsOut = extract_features(trajectoryList, 
                                                                                     shortTrajectoryList, 
                                                                                     sampledSpikes,
                                                                                     sample_info,
                                                                                     avgPos, 
                                                                                     avgSpeed,
                                                                                     nComps,
                                                                                     plot=True)
        if filemod is not None:
            with open(os.path.join(path.intermediate_save_path, '%s_model_features_and_components_network_%s_%s.pkl' % (path.date, params.FN_source, filemod)), 'wb') as f:
                dill.dump([traj_features, network_features, short_features, compsOut], f, recurse=True)    
        
    return traj_features, network_features, short_features, sampledSpikes, compsOut, sample_info, unit_info 

def count_model_features_from_trajectories(allData, lead, lag, nComps, filemod):
    
    goalIdx = [idx for idx, exp in enumerate(allData.analog_and_video['experiments']) if 'free' not in exp][0]
    
    camPeriod = np.mean(np.diff(allData.analog_and_video['frameTimes_byEvent'][goalIdx][0]))  
    
    trajSampShift = int(np.round(params.trajShift / camPeriod))
    leadSamps = int(np.round(lead / camPeriod))
    lagSamps = int(np.round(lag / camPeriod))
    
    shortSamps100 = int(np.round(.1 / camPeriod))
    shortSamps150 = int(np.round(.15 / camPeriod))
            
    if params.starting_eventNum is None:
        first_event = kinematics[0]['event']
    else:
        first_event = params.starting_eventNum
    
    sample_kinIdx   = []
    sample_eventNum = []
    for kIdx, kin in enumerate(allData.kinematics):

        print((lead, lag, kIdx))            

        eventIdx = kin['event'] - 1
        
        handIdx     = [idx for idx, mark in enumerate(kin['marker_names']) if mark == 'hand'][0]
        shoulderIdx = [idx for idx, mark in enumerate(kin['marker_names']) if mark == 'shoulder'][0]
        
        pos = np.empty((1, 3, kin['position'].shape[-1]))
        vel = np.empty_like(pos[..., :-1])
        pos[0] = kin['position'][handIdx] - kin['position'][shoulderIdx]
        for mark in range(pos.shape[0]): 
            vel[mark], tmp_acc = compute_derivatives(pos[mark], smooth = True)
        
        traj_slices, short_traj_slices, spike_samp_idx = get_frames_to_sample(kin,
                                                                              vel,
                                                                              leadSamps, 
                                                                              lagSamps, 
                                                                              shortSamps100, 
                                                                              shortSamps150, 
                                                                              trajSampShift)
        
        numTraj = len(traj_slices)            
                
        traj = np.empty((np.shape(vel)[0], numTraj, 3, traj_slices[0].stop - traj_slices[0].start))
        shortTraj = np.empty((np.shape(vel)[0], numTraj, 3, shortSamps150 - shortSamps100))
        avgSpeed = np.empty((np.shape(vel)[0], numTraj))
        avgPos = np.empty((np.shape(vel)[0], numTraj, 3))
        for mark in range(vel.shape[0]):
            for t, (slc, short_slc, spkIdx) in enumerate(zip(traj_slices, short_traj_slices, spike_samp_idx)):                    
                traj[mark, t]      = vel[mark, :, slc]
                if params.compute_shortTraj:
                    shortTraj[mark, t] = vel[mark, :, short_slc]                        
                
                avgSpeed[mark, t] = np.mean(np.linalg.norm(traj[mark, t], axis = -2))
                avgPos[mark, t]   = np.mean(pos[mark, :, slc], axis = -1)

                if params.normalize == 'on':
                    traj[mark, t] = traj[mark, t] / np.linalg.norm(traj[mark, t], axis = -2)
                    if params.compute_shortTraj:    
                        shortTraj[mark, t] = shortTraj[mark, t] / np.linalg.norm(shortTraj[mark, t], axis = -2)
                
                if mark == 0:
                    sample_kinIdx.append(kIdx)
                    sample_eventNum.append(kin['event'])

        if eventIdx+1 == first_event: 
            stackedTraj      = traj
            if params.compute_shortTraj:
                stackedShortTraj = shortTraj
            stackedSpeed     = avgSpeed
            stackedPos       = avgPos
        else: 
            stackedTraj = np.hstack((stackedTraj, traj))
            if params.compute_shortTraj:
                stackedShortTraj = np.hstack((stackedShortTraj, shortTraj))
            stackedSpeed = np.hstack((stackedSpeed, avgSpeed))
            stackedPos = np.hstack((stackedPos, avgPos))
                               
    # rearrange traj array into a list of arrays, with each element being the array of trajectories for a single marker
    trajectoryList = [stackedTraj[mark] for mark in range(np.shape(stackedTraj)[0])]
    if params.compute_shortTraj:
        shortTrajectoryList = [stackedShortTraj[mark, ...] for mark in range(np.shape(stackedShortTraj)[0])]
    else:
        shortTrajectoryList = [0]*len(trajectoryList)
    avgPos   = [stackedPos  [mark, ...] for mark in range(np.shape(stackedTraj)[0])]
    avgSpeed = [stackedSpeed[mark, ...] for mark in range(np.shape(stackedTraj)[0])]
        
    for mark in range(len(trajectoryList)):
        samples_with_nan = np.where(np.isnan(avgSpeed[mark]))[0]
        trajectoryList[mark]      = np.delete(trajectoryList[mark],      samples_with_nan, axis = 0)      
        if params.compute_shortTraj:
            shortTrajectoryList[mark] = np.delete(shortTrajectoryList[mark], samples_with_nan, axis = 0) 
        avgPos[mark]              = np.delete(avgPos[mark],              samples_with_nan, axis = 0)      
        avgSpeed[mark]            = np.delete(avgSpeed[mark],            samples_with_nan, axis = 0)
        sample_kinIdx   = [kinIdx for i, kinIdx in enumerate(sample_kinIdx) if i not in samples_with_nan]
        sample_eventNum = [eNum   for i, eNum   in enumerate(sample_eventNum) if i not in samples_with_nan]
    
    del stackedTraj, stackedPos, stackedSpeed
    if params.compute_shortTraj:
        del stackedShortTraj

    compsOut = extract_traj_features_only(trajectoryList, plot=True)  
        
    return compsOut

#%%

def train_and_test_glm(traj_features, network_features, sampledSpikes, mode, RNGs):   
    # done = np.zeros((np.shape(sampledSpikes)[0],))
    # bestAUC = np.ones_like(done) * 0.75
    
    areaUnderROC = np.empty((sampledSpikes.shape[0], params.numIters))
    aic          = np.empty_like(areaUnderROC)
    
    if mode == 'network_partial_traj':
        coefs = np.empty((traj_features.shape[-1] + 1, sampledSpikes.shape[0], params.numIters))
        pVals = np.empty_like(coefs)
    else:        
        coefs = np.empty((traj_features.shape[-1] + network_features.shape[-1] + 1, sampledSpikes.shape[0], params.numIters))
        pVals = np.empty_like(coefs)    
    
    for n, (split_rng, traj_rng) in enumerate(zip(RNGs['train_test_split'], RNGs['partial_traj'])):

        # Create train/test datasets for cross-validation
        # print(mode + ', iteration = ' +str(n))
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
                    testFeatures.append (np.hstack((traj_features[testIdx] , network_features[unit, testIdx])))   
                elif mode == 'network':
                    trainFeatures.append(network_features[unit, trainIdx])
                    testFeatures.append (network_features[unit, testIdx ])  
                elif mode in ['traj', 'short_traj', 'shuffle']:
                    trainFeatures.append(traj_features[trainIdx])
                    testFeatures.append (traj_features[testIdx])
                elif mode == 'network_partial_traj':
                    trajFeatureIdx = traj_rng.choice(np.arange(traj_features.shape[-1] - 3*np.sum(params.include_avg_pos) - np.sum(params.include_avg_speed)),
                                                     size = traj_features.shape[-1]-3*np.sum(params.include_avg_pos)-np.sum(params.include_avg_speed) - params.networkFeatureBins,
                                                     replace = False)
                    trainFeatures.append(np.hstack((traj_features[np.ix_(trainIdx, trajFeatureIdx)], network_features[unit, trainIdx])))
                    testFeatures.append (np.hstack((traj_features[np.ix_(testIdx , trajFeatureIdx)], network_features[unit, testIdx ]))) 
                else:
                    print('\n\n You must choose a mode from ["full", "network", "traj", "short_traj", "shuffle", "network_partial_traj"] \n\n')
                
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
            # trainSpks[trainSpks >= 1] = 1
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
            # testSpikes[unit][testSpikes[unit] >= 1] = 1
            for t, thresh in enumerate(thresholds):    
                posIdx = np.where(preds > thresh)
                hitProb[t] = np.sum(testSpikes[unit][posIdx] >= 1) / np.sum(testSpikes[unit] >= 1)
                falsePosProb[t] = np.sum(testSpikes[unit][posIdx] == 0) / np.sum(testSpikes[unit] == 0)
            
            areaUnderROC[unit, n] = auc(falsePosProb, hitProb)
            
            allHitProbs.append(hitProb)
            allFalsePosProbs.append(falsePosProb)
    
#            if mode == 'real' and areaUnderROC[-1] > bestAUC[unit] and done[unit] == 0:
            # if unit == 88:
                # fig, ax = plt.subplots()
                # ax.plot(preds)
                # ax.set_title('Unit' + str(unit))
                # tmp = np.array(testSpikes[unit], dtype=np.float16)
                # tmp[tmp == 0] = np.nan
                # tmp[~np.isnan(tmp)] = preds[~np.isnan(tmp)]
                # ax.plot(tmp, 'o', c = 'orange')
                # plt.show()
            
            
    #        plt.plot(falsePosProb, hitProb)
        
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
        %d model parameters (1 constant, %d kinematic features, and %d network features if included) and \
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
                
def test_model_significance(trueAUC_means, shuffleAUC_means):
    
    p_val = np.empty((np.shape(trueAUC_means)[0], np.shape(trueAUC_means)[1]))
    for unit, (trueMean, shuffleMeans) in enumerate(zip(trueAUC_means, shuffleAUC_means)):
        p_val[unit, 0] = np.sum(shuffleMeans[0, :] > trueMean[0]) / np.shape(shuffleMeans)[-1]     
        p_val[unit, 1] = np.sum(shuffleMeans[1, :] > trueMean[1]) / np.shape(shuffleMeans)[-1]     
    
    return p_val

def plot_pathlet_vs_brief_AUC(unit_info):
    fig, ax = plt.subplots()
    sns.scatterplot(ax = ax, data = unit_info, x = "brief_AUC", y = "pathlet_AUC", 
                    hue = "amp", style = "group")
    ax.plot(np.arange(0.5, 1.0, 0.1), np.arange(0.5, 1.0, 0.1), '--k')
    ax.set_xlim(0.5, 1)
    ax.set_ylim(0.5, 1)
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color('black')
    for axis in ['top','right']:
        ax.spines[axis].set_linewidth(0)
    ax.tick_params(width=2, length = 4, labelsize = 12)
    ax.set_xlabel('ROC area (+100 to +150 ms)', fontsize = 18, fontweight = 'bold')
    ax.set_ylabel('ROC area (-100 to +300 ms)', fontsize = 18, fontweight = 'bold')
    ax.grid(False)
    plt.show()
    
def find_number_of_trajectory_components():
    nComps = []
    for lead, lag in zip(params.lead, params.lag):
        if lag > 0.15:
            params.compute_shortTraj = True
        else:
            params.compute_shortTraj = False
        pComps = count_model_features_from_trajectories(allData, lead, lag, None, None)
        nComps.append(pComps.shape[0])
    
    return int(np.max(nComps))

def run_model(allData):
    
    nComps = find_number_of_trajectory_components()
    
    full_models                 = []
    traj_models                 = []
    network_models              = []
    short_models                = []
    shuffle_traj_models         = []
    # network_partial_traj_models = []
    for lead, lag in zip(params.lead, params.lag):
        
        if lag > 0.15:
            params.compute_shortTraj = True
        else:
            params.compute_shortTraj = False
        
        if params.include_avg_pos and params.include_avg_speed:
            filemod = 'lead_%d_lag_%d_shift_%d_PCAthresh_%d_with_speed_and_position_norm_%s' % (int(lead * 1e3), 
                                                                                                int(lag * 1e3),
                                                                                                int(params.trajShift * 1e3),
                                                                                                int(params.pca_var_thresh * 1e2), 
                                                                                                params.normalize)
        elif params.include_avg_speed and not params.include_avg_pos:
            filemod = 'lead_%d_lag_%d_shift_%d_PCAthresh_%d_with_speed_norm_%s' % (int(lead * 1e3), 
                                                                                   int(lag * 1e3),
                                                                                   int(params.trajShift * 1e3),
                                                                                   int(params.pca_var_thresh * 1e2),  
                                                                                   params.normalize)
        elif params.include_avg_pos and not params.include_avg_speed:
            filemod = 'lead_%d_lag_%d_shift_%d_PCAthresh_%d_with_position_norm_%s' % (int(lead * 1e3), 
                                                                                      int(lag * 1e3), 
                                                                                      int(params.trajShift * 1e3),
                                                                                      int(params.pca_var_thresh * 1e2),  
                                                                                      params.normalize)
        else:
            filemod = 'lead_%d_lag_%d_shift_%d_PCAthresh_%d_norm_%s' % (int(lead * 1e3), 
                                                                        int(lag * 1e3), 
                                                                        int(params.trajShift * 1e3),
                                                                        int(params.pca_var_thresh * 1e2),  
                                                                        params.normalize)
        traj_features, network_features, short_features, sampledSpikes, pComps, sample_info, unit_info = extract_trajectories_and_define_model_features(allData, lead, lag, nComps, filemod)

        try:
            with open(os.path.join(path.intermediate_save_path, 'RR%s_encoding_model_results_network_%s_%s.pkl'  % (path.date, params.FN_source, filemod)), 'rb') as f:
                full_model_results, traj_model_results, network_model_results, \
                    short_model_results, shuffle_model_results, network_partial_traj_model_results, unit_info = dill.load(f)  
            print('opened a saved set of single unit encoding results, located at' + (os.path.join(path.intermediate_save_path, 'single_unit_encoding_results_%s.pkl' % filemod)))
        except:
            
            RNGs = {'train_test_split' : [np.random.default_rng(n) for n in range(params.numIters)],
                    'partial_traj'     : [np.random.default_rng(n) for n in range(1000, 1000+params.numIters)],
                    'spike_shuffle'    : [np.random.default_rng(n) for n in range(5000, 5000+sampledSpikes.shape[0])]}
            
            print((lead, lag, 'full'))
            full_model_results                 = train_and_test_glm(traj_features, network_features, sampledSpikes, 'full', RNGs)
            print((lead, lag, 'network'))
            network_model_results              = train_and_test_glm( np.array([]), network_features, sampledSpikes, 'network', RNGs)
            # print((lead, lag, 'network_partial_traj'))
            # network_partial_traj_model_results = train_and_test_glm(traj_features, network_features, sampledSpikes, 'network_partial_traj', RNGs)
            print((lead, lag, 'traj'))
            traj_model_results                 = train_and_test_glm(traj_features,     np.array([]), sampledSpikes, 'traj', RNGs)
            if params.compute_shortTraj:
                print((lead, lag, 'short_traj'))
                short_model_results                = train_and_test_glm(short_features,    np.array([]), sampledSpikes, 'short_traj', RNGs) 
            else: 
                short_model_results = []
            print((lead, lag, 'shuffle'))
            shuffle_model_results              = train_and_test_glm(traj_features,     np.array([]), sampledSpikes, 'shuffle', RNGs)
            # shuffleAUC_means = np.empty((np.shape(trueAUC_means)[0], np.shape(trueAUC_means)[1], params.nShuffles))
            # for s in range(params.nShuffles):
            #     print('')
            #     print('shuffle = ' + str(s))
            #     print('')
            #     shuffleAUC_tmp = train_and_test_glm(features, sampledSpikes, 'shuffle')[0] 
            #     shuffleAUC_means[..., s] = np.mean(np.moveaxis(np.array(shuffleAUC_tmp), 0, -1), axis = -1)            
            
            unit_info['pathlet_shuffled_AUC']     = shuffle_model_results['AUC'].mean(axis=-1)
            if params.compute_shortTraj:
                unit_info['brief_AUC']                = short_model_results['AUC'].mean(axis=-1)
            unit_info['pathlet_AUC']              = traj_model_results['AUC'].mean(axis=-1)
            unit_info['network_AUC']              = network_model_results['AUC'].mean(axis=-1)
            # unit_info['network_partial_traj_AUC'] = network_partial_traj_model_results['AUC'].mean(axis=-1)
            unit_info['full_AUC']                 = full_model_results['AUC'].mean(axis=-1)
                        
            # p_val = test_model_significance(trueAUC_means, shuffleAUC_means)
                    
            full_models.append(full_model_results)
            traj_models.append(traj_model_results)
            network_models.append(network_model_results)
            if params.compute_shortTraj:
                short_models.append(short_model_results)
            shuffle_traj_models.append(shuffle_model_results)
            # network_partial_traj_models.append(network_partial_traj_model_results)
            
            all_model_results = {'model_results' : [full_model_results, traj_model_results, 
                                                    network_model_results, short_model_results, 
                                                    shuffle_model_results],
                                 'model_names'   : ['full', 'trajectory', 'network', 
                                                    'velocity', 'shuffle']} 
            
            with open(os.path.join(path.intermediate_save_path, '%s_encoding_model_results_network_%s_%s.pkl'  % (path.date, params.FN_source, filemod)), 'wb') as f:
                dill.dump([all_model_results, unit_info], f, recurse=True)  
        
        if params.compute_shortTraj:
            plot_pathlet_vs_brief_AUC(unit_info)
    
    return full_models, traj_models, network_models, short_models, shuffle_traj_models, unit_info, sampledSpikes

def sig_tests(unit_info, unit_info_reduced = None):
    
    if unit_info_reduced is None:
        nPathlet = np.sum(unit_info.pathlet_AUC > unit_info.brief_AUC)
        nUnits = np.shape(unit_info)[0]
        
        sign_test = binomtest(nPathlet, nUnits, p = 0.5, alternative='greater')
        
        ttest_paired = ttest_rel(unit_info.pathlet_AUC, unit_info.brief_AUC, alternative='greater')

    else:
        nPathlet = np.sum(unit_info.pathlet_AUC > unit_info_reduced.pathlet_AUC)
        nUnits = np.shape(unit_info)[0]
        sign_test = binomtest(nPathlet, nUnits, p = 0.5, alternative='greater')
        ttest_paired = ttest_rel(unit_info.pathlet_AUC, unit_info_reduced.pathlet_AUC, alternative='greater')

    return sign_test, ttest_paired

if __name__ == "__main__":

    spike_data, kinematics, analog_and_video, FN = load_data()    

    class allData:
        spike_data = spike_data
        analog_and_video = analog_and_video
        kinematics = kinematics    
        FN = FN

    full_models, traj_models, network_models, short_models, shuffle_traj_models, unit_info, sampledSpikes = run_model(allData)
    
    # sign_test_xv, ttest_xv = sig_tests(unit_info_xv, unit_info) 
