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
from hatlab_nwb_functions import remove_duplicate_spikes_from_good_single_units, get_sorted_units_and_apparatus_kinematics_with_metadata   

sys.path.insert(0, '/project/nicho/projects/marmosets/code_database/analysis/trajectory_encoding_model/')
from utils import get_interelectrode_distances_by_unit, choose_units_for_model

nwb_infile   = '/project/nicho/projects/dalton/data/TY20210211_freeAndMoths-003_DM.nwb'
base, ext = os.path.splitext(nwb_infile)
base, old_tag = base.split('DM')
# new_tag = '_analyze_trajectory_only_models'  
new_tag = '_encoding_model_regularized_results_30ms_shift_v1'  
pkl_outfile = base + 'DM' + new_tag + '.pkl'


debugging = False
run_model_only = True

if run_model_only:
    pkl_infile  = pkl_outfile
    
    split_pattern = '_shift_v' # '_results_v'
    base, ext = os.path.splitext(pkl_infile)
    base, in_version = base.split(split_pattern)
    out_version = str(int(in_version) + 1)  
    pkl_outfile = base + split_pattern + out_version + ext

class params:
    spkSampWin = 0.01
    trajShift = 0.03 #sample every 30ms
    lead = [0.5, 0.4, 0.4, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0  , 0  , 0  , 0.5] # lead time
    lag  = [0  , 0  , 0.1, 0  , 0.1, 0.2, 0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.3, 0.4, 0.5, 0.5] # lag time
    normalize = 'off'
    compute_shortTraj = True
    numThresh = 100
    trainRatio = 0.9
    if debugging:
        num_model_samples = 3
    else:
        num_model_samples = 100
    shuffle_to_test = 'traj_and_avgPos'
    minSpikeRatio = .005
    nDims = 3
    frate_thresh = 2
    snr_thresh=3
    fps = 150
    
    pca_var_thresh = 0.9# MAKE SURE that the idx being pulled is for the hand in run_pca_on_trajectories()
    idx_for_avg_pos_and_speed = 0
    hand_traj_idx = 0 
    FN_source = 'split_reach_FNs'
    transpose_FN = True
    
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

def get_frames_to_sample(timestamps, vel, leadSamps, lagSamps, shortSamps100, shortSamps150, trajSampShift):
    
    traj_slices       = []
    short_traj_slices = []
    spike_sample_time = []
    
    # trajLength = new_stop + 1 - new_start    
    for centerIdx in range(leadSamps, vel.shape[-1] - lagSamps, trajSampShift):
        tmp_traj_slice  = slice(centerIdx - leadSamps, centerIdx + lagSamps - 1)
        tmp_short_slice = slice(centerIdx + shortSamps100, centerIdx + shortSamps150)
        
        if   np.sum(np.isnan(vel[0, 0, tmp_traj_slice ])) > 0 or tmp_traj_slice.stop  > vel.shape[-1]:
            continue
        elif np.sum(np.isnan(vel[0, 0, tmp_short_slice])) > 0 or tmp_short_slice.stop > vel.shape[-1]:
            tmp_short_slice = None
            
        traj_slices.append(tmp_traj_slice)
        short_traj_slices.append(tmp_short_slice)
        spike_sample_time.append(timestamps[centerIdx])            
        
    return traj_slices, short_traj_slices, spike_sample_time 

def run_pca(traj, nComps = None, plot = False):
    traj_tmp = traj.copy()
    
    good_samples = np.where(~np.isnan(traj_tmp[:, 0]))[0]
    
    traj_tmp = traj_tmp[good_samples, :]
    
    # find out how many PCs to use
    pca = PCA()
    pca.fit(traj_tmp)      
    
    cumVar = np.cumsum(pca.explained_variance_ratio_)
    
    if nComps is None:
        cutComp = np.where(cumVar >= params.pca_var_thresh)[0][0]
        # pca = PCA(n_components = cutComp+1)
        nComps = cutComp+1
        
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
        
        return [], [], nComps
        
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
    
        traj_features = np.full((traj.shape[0], nComps), np.nan)
        traj_features[good_samples] = pca.fit_transform(traj[good_samples])
        
        compsOut = pca.components_
        
        # if params.normalize == 'on':
        #     trajNorm = np.tile(np.expand_dims(np.linalg.norm(projectedTraj, axis = 0), 0), (projectedTraj.shape[0], 1))
        #     projectedTraj = projectedTraj / trajNorm
                
        return traj_features, compsOut, nComps

def extract_traj_pca_features(traj_samples, nComps = None, plot = False):
    
    for mark, traj in enumerate(traj_samples):
        traj = np.reshape(traj, (np.shape(traj)[0], np.shape(traj)[1] * np.shape(traj)[2]))        
        if 'allTraj' not in locals():
            allTraj = traj
        else:
            allTraj = np.hstack(allTraj, traj)
    
    traj_features, compsOut, nComps = run_pca(allTraj, nComps = nComps,  plot = plot) 

    return traj_features, compsOut, nComps

def get_trajectory_samples(pos, vel, traj_slices, short_traj_slices, trajLength, shortLength):

    numTraj = len(traj_slices)            
    traj         = np.empty((np.shape(vel)[0], numTraj, 3, trajLength))
    shortTraj    = np.empty((np.shape(vel)[0], numTraj, 3, shortLength))
    avgSpeed     = np.empty((np.shape(vel)[0], numTraj))
    avgPos       = np.empty((np.shape(vel)[0], numTraj, 3)) 
    short_avgPos = np.empty((np.shape(vel)[0], numTraj, 3))

    for mark in range(vel.shape[0]):
        for trajIdx, (slc, short_slc) in enumerate(zip(traj_slices, short_traj_slices)):                    
            traj[mark, trajIdx] = vel[mark, :, slc]
            if params.compute_shortTraj:
                try:
                    shortTraj   [mark, trajIdx] =         vel[mark, :, short_slc] 
                    short_avgPos[mark, trajIdx] = np.mean(pos[mark, :, short_slc], axis = -1)                      
                except:
                    shortTraj   [mark, trajIdx] = np.full_like(   shortTraj[mark, trajIdx], np.nan)
                    short_avgPos[mark, trajIdx] = np.full_like(short_avgPos[mark, trajIdx], np.nan)                    
            
            avgSpeed[mark, trajIdx] = np.mean(np.linalg.norm(traj[mark, trajIdx], axis = -2))
            avgPos  [mark, trajIdx] = np.mean(pos[mark, :, slc], axis = -1)

            if params.normalize == 'on':
                traj[mark, trajIdx] = traj[mark, trajIdx] / np.linalg.norm(traj[mark, trajIdx], axis = -2)
                if params.compute_shortTraj:    
                        shortTraj[mark, trajIdx] = shortTraj[mark, trajIdx] / np.linalg.norm(shortTraj[mark, trajIdx], axis = -2)  
                        
    return traj, shortTraj, avgSpeed, avgPos, short_avgPos
                        
def get_spike_samples(units, spike_sample_times):

    numTraj = len(spike_sample_times)            
    spikes = np.zeros((units.shape[0], numTraj, params.networkSampleBins), dtype='int8')

    for trajIdx, t_spk_samp in enumerate(spike_sample_times):                    

        # get spike/no-spike in 10ms window centered around idx
        startBound = t_spk_samp - (params.networkSampleBins - 0.5)*params.spkSampWin
        stopBound  = t_spk_samp + 1.5*params.spkSampWin 
        bins = np.arange(startBound, stopBound, params.spkSampWin)
        
        for uIdx, unit in units.iterrows():
            unit_spikes = unit.spike_times
            spike_bins  = np.digitize(unit_spikes, bins) - 1
            spike_bins  = spike_bins[(spike_bins > -1) & (spike_bins < params.networkSampleBins)]
            bin_counts  = np.bincount(spike_bins)
            spikes[uIdx, trajIdx, spike_bins] = bin_counts[spike_bins]
            
    return spikes

def sample_trajectories_and_spikes(units, reaches, kin_module, nwb, lead, lag):
    
    first_event_key = [key for idx, key in enumerate(kin_module.data_interfaces.keys()) if idx == 0][0]
    camPeriod = np.mean(np.diff(kin_module.data_interfaces[first_event_key].pose_estimation_series['origin'].timestamps[:]))
    dlc_scorer = kin_module.data_interfaces[first_event_key].scorer    
    
    if 'simple_joints_model' in dlc_scorer:
        wrist_label = 'hand'
        shoulder_label = 'shoulder'
    elif 'marmoset_model' in dlc_scorer and nwb.subject.subject_id == 'TY':
        wrist_label = 'l-wrist'
        shoulder_label = 'l-shoulder'
    elif 'marmoset_model' in dlc_scorer and nwb.subject.subject_id == 'MG':
        wrist_label = 'r-wrist'
        shoulder_label = 'r-shoulder'
    
    
    trajSampShift = int(np.round(params.trajShift / camPeriod))
    leadSamps = int(np.round(lead / camPeriod))
    lagSamps = int(np.round(lag / camPeriod))
    
    shortSamps100 = int(np.round(.1 / camPeriod))
    shortSamps150 = int(np.round(.15 / camPeriod))
    
    trajLength = leadSamps + lagSamps - 1
    shortLength = shortSamps150 - shortSamps100
    
    sample_reach_idx   = []
    sample_video_event = []

    for rIdx, reach in reaches.iterrows():
        
        if debugging:
            if rIdx > 1:
                break
        
        # get event data using container and ndx_pose names from segment_info table following form below:
        # nwb.processing['goal_directed_kinematics'].data_interfaces['moths_s_1_e_004_position']
        event_data      = kin_module.data_interfaces[reach.video_event] 
        
        wrist_kinematics    = event_data.pose_estimation_series[   wrist_label].data[reach.start_idx:reach.stop_idx+1].T
        shoulder_kinematics = event_data.pose_estimation_series[shoulder_label].data[reach.start_idx:reach.stop_idx+1].T
        timestamps          = event_data.pose_estimation_series[   wrist_label].timestamps[reach.start_idx:reach.stop_idx+1]
        
        pos = np.empty((1, 3, timestamps.shape[-1]))
        vel = np.empty_like(pos[..., :-1])
        pos[0] = wrist_kinematics - shoulder_kinematics
        for mark in range(pos.shape[0]): 
            vel[mark], tmp_acc = compute_derivatives(pos[mark], smooth = True)
            
        traj_slices, short_traj_slices, spike_sample_times = get_frames_to_sample(timestamps,
                                                                                  vel,
                                                                                  leadSamps, 
                                                                                  lagSamps, 
                                                                                  shortSamps100, 
                                                                                  shortSamps150, 
                                                                                  trajSampShift)
        
        
        print('lead = %d, lag = %d, rIdx = %d' % (int(lead*1e3), int(lag*1e3), rIdx))
        if len(traj_slices) == 0:
            continue
        traj, shortTraj, avgSpeed, avgPos, short_avgPos = get_trajectory_samples(pos, vel, traj_slices, short_traj_slices, trajLength, shortLength)
        spikes = get_spike_samples(units, spike_sample_times)

        if 'stackedTraj' not in locals(): 
            stackedTraj      = traj
            spike_samples    = spikes
            stackedSpeed     = avgSpeed
            stackedPos       = avgPos
            if params.compute_shortTraj:
                stackedShortTraj = shortTraj
                stackedShortPos  = short_avgPos
        else: 
            stackedTraj   = np.hstack((stackedTraj, traj))
            spike_samples = np.hstack((spike_samples, spikes))
            stackedSpeed  = np.hstack((stackedSpeed, avgSpeed))
            stackedPos    = np.hstack((stackedPos, avgPos))
            if params.compute_shortTraj:
                stackedShortTraj = np.hstack((stackedShortTraj, shortTraj))
                stackedShortPos  = np.hstack((stackedShortPos,  short_avgPos))
        
        
        sample_reach_idx.extend([rIdx for i in range(traj.shape[1])])
        sample_video_event.extend([reach.video_event for i in range(traj.shape[1])])
        
    # rearrange traj array into a list of arrays, with each element being the array of trajectories for a single marker
    traj_samples = [stackedTraj[mark] for mark in range(np.shape(stackedTraj)[0])]
    avg_pos_samples   = [stackedPos  [mark, ...] for mark in range(np.shape(stackedTraj)[0])]
    avg_speed_samples = [stackedSpeed[mark, ...] for mark in range(np.shape(stackedTraj)[0])]
    if params.compute_shortTraj:
        short_traj_samples    = [stackedShortTraj[mark, ...] for mark in range(np.shape(stackedShortTraj)[0])]
        short_avg_pos_samples = [stackedShortPos [mark, ...] for mark in range(np.shape(stackedShortPos )[0])]
    else:
        short_traj_samples    = [0]*len(traj_samples)
        short_avg_pos_samples = [0]*len(traj_samples) 
    
    # sampledSpikes = np.delete(sampledSpikes, np.where(np.isnan(avgSpeed[0]))[0], axis = 1)
    for mark in range(len(traj_samples)):
        samples_with_nan = np.where(np.isnan(avg_speed_samples[mark]))[0]
        # trajectoryList[mark]      = np.delete(trajectoryList[mark],      samples_with_nan, axis = 0)      
        # if params.compute_shortTraj:
        #     shortTrajectoryList[mark] = np.delete(shortTrajectoryList[mark], samples_with_nan, axis = 0) 
        # avgPos[mark]              = np.delete(avgPos[mark],              samples_with_nan, axis = 0)      
        # avgSpeed[mark]            = np.delete(avgSpeed[mark],            samples_with_nan, axis = 0)
        # sample_kinIdx   = [kinIdx for i, kinIdx in enumerate(sample_kinIdx) if i not in samples_with_nan]
        # sample_eventNum = [eNum   for i, eNum   in enumerate(sample_eventNum) if i not in samples_with_nan]
        
    sample_info = pd.DataFrame(data = zip(sample_reach_idx, sample_video_event), columns = ['reach_idx', 'video_event'])
    
    lead_lag_key = 'lead_%d_lag_%d' % (int(lead*1e3), int(lag*1e3))
    lead_lag_dict = dict()
    lead_lag_dict['description'] = '''All-inclusive dict variable that holds all information for the given lead-lag combination. 
    This includes:
        - The sampled data used to produce model test/train features 
        - The extracted features 
        - Model results
        - Additional metadata.  
    '''
    
    sampled_data_description = '''Sampled data used to extract features for models.
        traj_samples: All the full-length trajectory samples for the lead and lag of this model. 
                      For this model, the position is wrist_pos - shoulder_pos, 
                      and the velocity is computed from that with some minor smoothing.
        avg_pos_samples: average position for the corresponding element in traj_samples
        avg_speed_samples: average speed for the corresponding element in traj_samples
        short_traj_samples: brief trajectory samples from +100 to +150 lag
        spike_samples: spike samples corresponding to the traj_samples. 
                       The last element of each row is the coincident time bin and 
                       preceding elements are leading bins, moving back in time 
                       such that the first element at idx=0 corresponds to the longest lead time. 
        sample_info: reach index and video event from which each sample was grabbed. 
                     Used for selecting the correct FN from split_reach_FNs for extracting network features. 
    '''
    lead_lag_dict['sampled_data'] = {'traj_samples'          : traj_samples,
                                     'short_traj_samples'    : short_traj_samples,
                                     'short_avg_pos_samples' : short_avg_pos_samples,
                                     'avg_pos_samples'       : avg_pos_samples,
                                     'avg_speed_samples'     : avg_speed_samples,
                                     'spike_samples'         : spike_samples,
                                     'sample_info'           : sample_info,
                                     'description'           : sampled_data_description} 
    
    results_dict[lead_lag_key] = lead_lag_dict
    
    return traj_samples, short_traj_samples, avg_pos_samples, avg_speed_samples, spike_samples, sample_info 

def sample_trajectories_and_spikes_for_model(units, reaches, kin_module, nwb):
    all_lead_lag_nComps = []
    for lead, lag in zip(params.lead, params.lag):
        
        if debugging:
            if lead == params.lead[2] and lag == params.lag[2]:
                break
        
        sampled_data = sample_trajectories_and_spikes(units, reaches, kin_module, nwb, lead, lag)
        
        traj_samples = sampled_data[0]
        
        _, _, nComps = extract_traj_pca_features(traj_samples, nComps = None, plot = False)
        
        all_lead_lag_nComps.append(nComps)
    
    max_nComps = int(np.max(all_lead_lag_nComps))
    
    return max_nComps

def create_model_features_and_store_in_dict(nComps):
    
    for lead, lag in zip(params.lead, params.lag):
        
        if debugging:
            if lead == params.lead[2] and lag == params.lag[2]:
                break
        
        lead_lag_key = 'lead_%d_lag_%d' % (int(lead*1e3), int(lag*1e3))
        lead_lag_dict = results_dict[lead_lag_key]
        sampled_data = lead_lag_dict['sampled_data']
        
        traj_samples          = sampled_data['traj_samples']
        short_traj_samples    = sampled_data['short_traj_samples']
        avg_speed_samples     = sampled_data['avg_speed_samples']
        avg_pos_samples       = sampled_data['avg_pos_samples']
        short_avg_pos_samples = sampled_data['short_avg_pos_samples']
        
        traj_features      , traj_comps      , _ = extract_traj_pca_features(traj_samples      , nComps = nComps, plot = False)
        short_traj_features, short_traj_comps, _ = extract_traj_pca_features(short_traj_samples, nComps = nComps, plot = False)
        
        idx = params.idx_for_avg_pos_and_speed
        traj_features_pos       = np.hstack((traj_features, avg_pos_samples[idx]))
        traj_features_speed     = np.hstack((traj_features, np.expand_dims(avg_speed_samples[idx], axis=1)))
        # traj_features_pos_and_speed = np.hstack((traj_features, np.expand_dims(avg_speed_samples[idx], axis=1), avg_pos_samples[idx]))
        short_traj_features_pos = np.hstack((short_traj_features, short_avg_pos_samples[idx]))
        
        lead_lag_dict = results_dict[lead_lag_key]        
        features_description = '''Model features used as inputs for testing and training models.
        The key for each feature set describes what features were included. 
        '''
        lead_lag_dict['model_features'] = {'traj'                     : traj_features,
                                           'traj_and_avgPos'          : traj_features_pos,
                                           'traj_and_avgSpeed'        : traj_features_speed,
                                           'short_traj_and_avgPos'    : short_traj_features_pos,
                                           'traj_PCA_components'      : traj_comps,
                                           'description'              : features_description} 
        
        results_dict[lead_lag_key] = lead_lag_dict

def choose_regularization_params(trainFts, testFts, trainSpks, testSpks):
    
    alpha_range = np.arange(0, 20, 1)
    l1_range = np.arange(0,1,0.1)
    
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
            
    best_alpha_idx, best_L1_wt_idx = np.where(areaUnderROC == areaUnderROC.max())
    best_alpha = alpha_range[best_alpha_idx][0]
    best_L1_wt = l1_range[best_L1_wt_idx][0]
    
    return best_alpha, best_L1_wt

def train_and_test_glm(traj_features, spike_samples, model_name, RNGs, lead_lag_key):   
    
    areaUnderROC = np.empty((spike_samples.shape[0], params.num_model_samples))
    # aic          = np.empty_like(areaUnderROC)
    coefs = np.empty((traj_features.shape[-1] + 1, spike_samples.shape[0], params.num_model_samples))
    # pVals = np.empty_like(coefs)    
    all_alpha = np.empty((spike_samples.shape[0],)) 
    all_l1    = np.empty((spike_samples.shape[0],))
    
    for n, split_rng in enumerate(RNGs['train_test_split']):

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
                  
                trainFeatures.append(traj_features[trainIdx])
                testFeatures.append (traj_features[testIdx])
       
            else:
                if n == 0:
                    print('unit ' + str(unit) + ' had only ' + str(np.sum(spikes[testIdx] >= 1)) + ' spikes in the sampled time windows and is removed from analysis')
            
        # Train GLM
        
        models = []
        predictions = []
        # trainPredictions = []
        for unit, (trainSpks, trainFts, testFts, testSpks, shuf_rng) in enumerate(zip(trainSpikes, trainFeatures, testFeatures, testSpikes, RNGs['spike_shuffle'])):
            if 'shuffle' in model_name:
                trainSpks = shuf_rng.permutation(trainSpks)
            
            if n == 0:
                alpha, L1_wt = choose_regularization_params(trainFts, testFts, trainSpks, testSpks)
                all_alpha[unit] = alpha
                all_l1   [unit] = L1_wt
            else:
                alpha = all_alpha[unit]
                L1_wt = all_l1[unit]
                
            print('best alpha and L1_wt combo is alpha=%.1f, L1_wt=%.1f' % (alpha, L1_wt))
            
            glm = sm.GLM(trainSpks,
                         sm.add_constant(trainFts), 
                         family=sm.families.Poisson(link=sm.families.links.log()))
            encodingModel = glm.fit_regularized(method='elastic_net', alpha=alpha, L1_wt=L1_wt)
            
            coefs[:, unit, n] = encodingModel.params            
            # pVals[:, unit, n] = np.round(encodingModel.pvalues, decimals = 4)            
            # aic  [   unit, n] = round(encodingModel.aic, 1)
            all_alpha[unit] = alpha
            all_l1[   unit] = L1_wt
            
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
        
    description = '''Model = %s. 
    To understand dimensions: 
        %d units,
        %d model parameters (1 constant, %d kinematic features)
        %d shuffles of the train/test split. 
    The order of model parameters is: 
        Constant term
        Trajectory features projected onto principal components 
        Average position terms (3 terms, x/y/z, if model includes average position)
        Average speed terms if included in model
    The keys hold the following information:
        AUC: cross-validated area under the ROC curve on %d%% held-out test data 
        AIC: AIC criterion value for trained model on training data
        coefs: the parameter coefficents that have been fit to the input features  
        pvals: the p-values describing the significance of the parameter coefficients
    If this is a shuffled model, that means the spike samples were shuffled to eliminate the relationship
    between model features and spiking. A new spike_samples shuffle was performed for each train/test split. 
    ''' % (model_name, areaUnderROC.shape[0], coefs.shape[0], traj_features.shape[-1], params.num_model_samples, int((1-params.trainRatio)*1e2))

    model_results = {'AUC'         : areaUnderROC,
                     'coefs'       : coefs,
                     'alpha'       : all_alpha,
                     'L1_weight'   : all_l1,
                     'description' : description}     
    # model_results = {'AUC'         : areaUnderROC,
    #                  'coefs'       : coefs,
    #                  'pvals'       : pVals,
    #                  'AIC'         : aic,
    #                  'alpha'       : all_alpha,
    #                  'L1_weight'   : all_l1,
    #                  'description' : description}    
    
    return model_results

def run_models():
    for lead, lag in zip(params.lead, params.lag):    
        if debugging:
            if lead == params.lead[2] and lag == params.lag[2]:
                break
        
        lead_lag_key = 'lead_%d_lag_%d' % (int(lead*1e3), int(lag*1e3))
        
        spike_samples = results_dict[lead_lag_key]['sampled_data']['spike_samples'] 

        model_keys = [key for key in results_dict[lead_lag_key]['model_features'].keys() if key not in ['description', 'traj_PCA_components' ]]
        if debugging:
            model_keys = [model_keys[0]]

        model_keys.append('traj_and_avgPos_shuffled_spike_samples')

        RNGs = {'train_test_split' : [np.random.default_rng(n) for n in range(params.num_model_samples)],
                'spike_shuffle'    : [np.random.default_rng(n) for n in range(5000, 5000+spike_samples.shape[0])]}        

        model_results = dict()

        for model in model_keys:
            if 'shuffle' in model:
                traj_features = results_dict[lead_lag_key]['model_features'][params.shuffle_to_test]
            else:
                traj_features = results_dict[lead_lag_key]['model_features'][model]
            
            good_samples = np.where(~np.isnan(traj_features[:, 0]))[0]
            tmp_traj_features = traj_features.copy()[good_samples]
            tmp_spike_samples = spike_samples.copy()[:, good_samples]

            model_results[model] = train_and_test_glm(tmp_traj_features, tmp_spike_samples, model, RNGs, lead_lag_key)

        results_dict[lead_lag_key]['model_results'] = model_results
        
        with open(pkl_outfile, 'wb') as f:
            dill.dump(results_dict, f, recurse=True) 
            
        print('Just saved model results for %s' % lead_lag_key)

if __name__ == "__main__":
    
    if not run_model_only:
    
        results_dict = dict()
        
        with NWBHDF5IO(nwb_infile, 'r') as io:
        # io = NWBHDF5IO(nwb_analysis_file, 'r')
            nwb = io.read()
    
            reaches_key = [key for key in nwb.intervals.keys() if 'reaching_segments' in key][0]
            
            units, reaches, kin_module = get_sorted_units_and_apparatus_kinematics_with_metadata(nwb, reaches_key, plot=False) 

            units = choose_units_for_model(units, quality_key = 'snr', quality_thresh = params.snr_thresh, frate_thresh = params.frate_thresh)
            # units = choose_units_for_model(units, quality_percentile=5, frate_thresh=params.frate_thresh)
        
            nComps = sample_trajectories_and_spikes_for_model(units, reaches, kin_module, nwb)
        
            create_model_features_and_store_in_dict(nComps)   
            
            with open(pkl_outfile, 'wb') as f:
                dill.dump(results_dict, f, recurse=True)
            
            run_models()
            
            with open(pkl_outfile, 'wb') as f:
                dill.dump(results_dict, f, recurse=True)  

    else:
        with open(pkl_infile, 'rb') as f:
            results_dict = dill.load(f)
            
        run_models()
        
        with open(pkl_outfile, 'wb') as f:
            dill.dump(results_dict, f, recurse=True) 
        
    # nComps = find_number_of_trajectory_components(units, reaches, kin_module)

    # # RNGs = {'train_test_split' : [np.random.default_rng(n) for n in range(params.num_model_samples)],
    # #         'partial_traj'     : [np.random.default_rng(n) for n in range(1000,  1000+params.num_model_samples)],
    # #         'spike_shuffle'    : [np.random.default_rng(n) for n in range(5000,  5000+single_lead_lag_models['sampled_spikes'].shape[0])],
    # #         'weight_shuffle'   : [np.random.default_rng(n) for n in range(10000, 10000+params.num_model_samples)]}
    
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


        
        
