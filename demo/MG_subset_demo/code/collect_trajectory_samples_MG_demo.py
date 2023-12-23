#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sat Dec 23 11:30:02 2023

@author: daltonm
"""
#%matplotlib notebook
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dill
import os
import seaborn as sns
import re
from sklearn.preprocessing import StandardScaler
from scipy.integrate import simpson
from scipy.fft import rfft, rfftfreq
from scipy.ndimage import gaussian_filter
from importlib import sys
from pathlib import Path

from pynwb import NWBHDF5IO
import ndx_pose

script_directory = Path(os.path.dirname(os.path.abspath(sys.argv[0])))
code_path = script_directory.parent.parent.parent / 'clean_final_analysis/'
data_path = script_directory.parent.parent / 'data' / 'demo'

sys.path.insert(0, str(code_path))
from utils import choose_units_for_model, save_dict_to_hdf5
from hatlab_nwb_functions import get_sorted_units_and_apparatus_kinematics_with_metadata   

marmcode='MG'

debugging = False
demo = True
show_plots=False

if marmcode=='TY':
    nwb_infile   = data_path / 'TY' / 'TY20210211_freeAndMoths-003_resorted_20230612_DM.nwb'
    bad_units_list = None
    mua_to_fix = []
    new_tag = 'samples_collected'
elif marmcode=='MG':
    nwb_infile   = data_path / 'MG' / 'MG20230416_1505_mothsAndFree-002_processed_DM.nwb'
    bad_units_list = [181, 440]
    mua_to_fix = [745, 796]
    new_tag = 'samples_collected'  

pkl_outfile = nwb_infile.parent / f'{nwb_infile.stem}_{new_tag}.pkl' 

class params:
    
    if marmcode == 'MG':
        fps = 200
        mua_to_fix = mua_to_fix
    elif marmcode =='TY':
        fps = 150
        mua_to_fix = mua_to_fix
    
    spkSampWin = 0.01
    trajShift = 0.03 #sample every 30ms
    if demo:
        lead = [0.1] # lead time
        lag  = [0.3] # lag time
    else:
        lead = [0.5, 0.4, 0.4, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0  , 0  , 0  , 0.15, 0.25] # lead time
        lag  = [0  , 0  , 0.1, 0  , 0.1, 0.2, 0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.3, 0.4, 0.5, 0.15, 0.25] # lag time

    frate_thresh = 2
    snr_thresh = 3
    subsamp_fps = 40
        
    idx_for_avg_pos_and_speed = 0
    
    networkSampleBins = 2
    
    dpi = 300
    
def compute_derivatives(marker_pos=None, marker_vel=None, smooth = True):
    
    if marker_pos is not None and marker_vel is None:
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

def get_trajectory_samples(pos, vel, traj_slices, short_traj_slices, trajLength, shortLength):

    numTraj = len(traj_slices)            
    traj         = np.empty((np.shape(vel)[0], numTraj, 3, trajLength))
    posTraj      = np.empty((np.shape(vel)[0], numTraj, 3, trajLength))
    shortTraj    = np.empty((np.shape(vel)[0], numTraj, 3, shortLength))
    avgSpeed     = np.empty((np.shape(vel)[0], numTraj))
    avgPos       = np.empty((np.shape(vel)[0], numTraj, 3)) 
    short_avgPos = np.empty((np.shape(vel)[0], numTraj, 3))

    for mark in range(vel.shape[0]):
        for trajIdx, (slc, short_slc) in enumerate(zip(traj_slices, short_traj_slices)):                    
            traj   [mark, trajIdx] = vel[mark, :, slc]
            posTraj[mark, trajIdx] = pos[mark, :, slc]
            try:
                shortTraj   [mark, trajIdx] =         vel[mark, :, short_slc] 
                short_avgPos[mark, trajIdx] = np.mean(pos[mark, :, short_slc], axis = -1)                      
            except:
                shortTraj   [mark, trajIdx] = np.full_like(   shortTraj[mark, trajIdx], np.nan)
                short_avgPos[mark, trajIdx] = np.full_like(short_avgPos[mark, trajIdx], np.nan)                    
            
            avgSpeed[mark, trajIdx] = np.mean(np.linalg.norm(traj[mark, trajIdx], axis = -2))
            avgPos  [mark, trajIdx] = np.mean(pos[mark, :, slc], axis = -1)

    return traj, posTraj, shortTraj, avgSpeed, avgPos, short_avgPos
                        
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

def compute_tortuosity(kin_df, neighborhood = None):
    # computes toruosity in a neighborhood for each point in bout df
    if neighborhood is None:
        start_index = 0
        end_index  = kin_df.shape[0]-1
        start_point = kin_df[['x','y','z']].iloc[start_index].to_numpy()
        end_point   = kin_df[['x','y','z']].iloc[end_index].to_numpy()   
        # arc_length = np.sqrt(np.square(kin_df[['vx', 'vy', 'vz']].iloc[start_index:end_index]).sum(axis=1)).sum()   
        arc_length = simpson(np.linalg.norm(kin_df[['vx', 'vy', 'vz']].loc[start_index:end_index], axis = 1), dx = 1/params.fps, axis = 0)

        cord_length = np.linalg.norm(start_point-end_point)
        mean_tortuosity = arc_length/cord_length         
    else:
        num_samp = neighborhood
        half_window = round(num_samp/2)  
        tortuosity = []
        index = kin_df.index
        for i in index:
            if i - half_window > index[0] and i + half_window < index[-1]:
                start_index = i - half_window
                end_index = i + half_window
                start_point = kin_df[['x','y','z']].loc[start_index].to_numpy()
                end_point = kin_df[['x','y','z']].loc[end_index].to_numpy()   
                arc_length = np.sqrt(np.square(kin_df[['vx', 'vy', 'vz']].loc[start_index:end_index]).sum(axis=1)).sum()
                arc_length_2 = simpson(np.linalg.norm(kin_df[['vx', 'vy', 'vz']].loc[start_index:end_index], axis = 1), dx = 1/params.fps, axis = 0)

                cord_length =np.linalg.norm(start_point-end_point)
                tort_val = arc_length/cord_length      
            else:
                tort_val = np.nan 
            tortuosity.append(tort_val)
        kin_df['tortuosity'] = tortuosity
        mean_tortuosity = np.nanmean(tortuosity)
    return kin_df, mean_tortuosity

def compute_curvature(kin_df):
    # computes curvature of bout 
    curvature = []
    index = kin_df.index
    for i in index:
#         print(i)
        if i -1 in index:
            pt = i
            vx = kin_df['vx'].loc[pt]
            ax = kin_df['ax'].loc[pt]
            vy = kin_df['vy'].loc[pt]
            ay= kin_df['ay'].loc[pt]
            vz = kin_df['vz'].loc[pt]
            az = kin_df['az'].loc[pt]
            K = (np.sqrt(np.square(az*vy - ay*vz) + np.square(ax*vz - az*vx)+np.square(ay*vx - ax*vy)) 
                     / ((np.square(vx)+np.square(vy)+np.square(az))**1.5))
            if K > 600: # mask anomolously high values
                K = np.nan
            curv_val = K
#             print(f'curvature = {curv_val}')
        else:
            curv_val = np.nan 
        curvature.append(curv_val)
    kin_df['curvature'] = curvature
    mean_curvature = np.nanmean(curvature)
    peak_curvature = np.nanmax(curvature)
    
    return kin_df, mean_curvature, peak_curvature

def plot_kinematic_features(sample_info, mark_pos):
    
    x_key = 'mean_tortuosity'
    y_key = 'peak_speed'
    z_key = 'mean_curvature'
    fig = plt.figure(figsize=(8, 8),dpi=300)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(sample_info[x_key], sample_info[y_key], sample_info[z_key], s=1)
    # sns.scatterplot(ax=ax, data=sample_info, x=x_key, y=y_key)
    ax.set_xlim(0, np.nanpercentile(sample_info[x_key], 95))
    ax.set_ylim(0, np.nanpercentile(sample_info[y_key], 95))
    ax.set_zlim(0, np.nanpercentile(sample_info[z_key], 95))
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_zlabel(z_key)
    # plt.show()
    fig.savefig('/project/nicho/projects/dalton/plots/TY20210211/kinematic_complexity_3D_%s_%s_%s.png' % (x_key, y_key, z_key), bbox_inches='tight', dpi=300)
    
    fig, ax = plt.subplots(figsize=(6, 6),dpi=300)
    sns.scatterplot(ax=ax, data=sample_info, x=x_key, y=y_key, s = 1)
    ax.set_xlim(0, np.nanpercentile(sample_info[x_key], 95))
    ax.set_ylim(0, np.nanpercentile(sample_info[y_key], 95))
    # plt.show()
    fig.savefig('/project/nicho/projects/dalton/plots/TY20210211/kinematic_complexity_2D_%s_%s.png' % (x_key, y_key), bbox_inches='tight', dpi=300)

    sample_info.sort_values(by = 'mean_tortuosity', ascending=True, inplace=True)
    fig = plt.figure(figsize=(8, 8),dpi=300)
    ax = fig.add_subplot(projection='3d')
    for samp in sample_info.index[500:600]:
        sample_pos = mark_pos[samp]        
        ax.plot(sample_pos[0], sample_pos[1], sample_pos[2], linewidth=0.5)
    ax.set_title(f'Tortuosity = {sample_info.loc[samp, "mean_tortuosity"]}')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-2, 10)
    ax.set_zlim(-5, 8)
    # plt.show()
    
    sample_info.sort_values(by = 'mean_tortuosity', ascending=False, inplace=True)
    fig = plt.figure(figsize=(8, 8),dpi=300)
    ax = fig.add_subplot(projection='3d')
    for samp in sample_info.index[100:200]:
        sample_pos = mark_pos[samp]        
        ax.plot(sample_pos[0], sample_pos[1], sample_pos[2], linewidth=0.5)
    ax.set_title(f'Tortuosity = {sample_info.loc[samp, "mean_tortuosity"]}')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-2, 10)
    ax.set_zlim(-5, 8)
    # plt.show()
    
    fig, ax = plt.subplots()
    sns.histplot(data=sample_info, ax=ax, x='mean_tortuosity', bins=100)
    ax.vlines(sample_info['mean_tortuosity'].median(), 0, 1200, 'black')
    # plt.show()
    
    sample_info.loc[sample_info['mean_tortuosity']<sample_info['mean_tortuosity'].median(), 'mean_speed'].mean()
    sample_info.loc[sample_info['mean_tortuosity']>sample_info['mean_tortuosity'].median(), 'mean_speed'].mean()

def compute_kinematic_features(pos_samples, traj_samples, sample_reach_idx, sample_video_event):
    mark_pos = pos_samples[0] 
    mark_vel = traj_samples[0]
    mean_speed_list = []
    peak_speed_list = []
    mean_curve_list = []
    mean_tort_list  = []
    peak_curve_list = []
    for sample_pos, sample_vel in zip(mark_pos, mark_vel):
        tmp_vel, sample_acc = compute_derivatives(marker_pos=None, marker_vel=sample_vel, smooth=True)     
        kin_df = pd.DataFrame(data=np.concatenate((sample_pos.transpose()[:-1], sample_vel.transpose()[:-1], sample_acc.transpose()), axis = 1),
                              columns=['x', 'y', 'z',
                                       'vx', 'vy', 'vz',
                                       'ax', 'ay', 'az'])     
        kin_df, mean_curvature, peak_curvature = compute_curvature(kin_df)
        kin_df, mean_tortuosity = compute_tortuosity(kin_df)

        speed = np.linalg.norm(kin_df[['vx', 'vy', 'vz']].to_numpy(), axis = 1)
        mean_speed = np.nanmean(speed)
        peak_speed = np.nanmax(speed)
    
        mean_speed_list.append(mean_speed)
        peak_speed_list.append(peak_speed)
        mean_curve_list.append(mean_curvature)
        peak_curve_list.append(peak_curvature)
        mean_tort_list.append(mean_tortuosity)
        
        # fig, ax = plt.subplots()
        # ax.plot(kin_df.loc[:, ['vx', 'vy', 'vz']])
        # ax.plot(speed)
        # plt.show()
    
    sample_info = pd.DataFrame(data = zip(sample_reach_idx, sample_video_event, 
                                          mean_speed_list, peak_speed_list, mean_curve_list, peak_curve_list, mean_tort_list), 
                               columns = ['reach_idx', 'video_event',
                                          'mean_speed', 'peak_speed', 'mean_curvature', 'peak_curvature', 'mean_tortuosity'])
    return sample_info, mark_pos

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
            vel[mark], tmp_acc = compute_derivatives(marker_pos=pos[mark], marker_vel=None, smooth = True)
            
        traj_slices, short_traj_slices, spike_sample_times = get_frames_to_sample(timestamps,
                                                                                  vel,
                                                                                  leadSamps, 
                                                                                  lagSamps, 
                                                                                  shortSamps100, 
                                                                                  shortSamps150, 
                                                                                  trajSampShift)
        
        
        print('lead = %d, lag = %d, rIdx = %d' % (int(lead*1e3), int(lag*1e3), rIdx), flush=True)
        if len(traj_slices) == 0:
            continue
        traj, posTraj, shortTraj, avgSpeed, avgPos, short_avgPos = get_trajectory_samples(pos, vel, traj_slices, short_traj_slices, trajLength, shortLength)
        spikes = get_spike_samples(units, spike_sample_times)

        if 'stackedTraj' not in locals(): 
            stackedTraj      = traj
            stackedPosTraj   = posTraj
            spike_samples    = spikes
            stackedSpeed     = avgSpeed
            stackedPos       = avgPos
            stackedShortTraj = shortTraj
            stackedShortPos  = short_avgPos
        else: 
            stackedTraj    = np.hstack((stackedTraj, traj))
            stackedPosTraj = np.hstack((stackedPosTraj, posTraj))
            spike_samples  = np.hstack((spike_samples, spikes))
            stackedSpeed   = np.hstack((stackedSpeed, avgSpeed))
            stackedPos     = np.hstack((stackedPos, avgPos))
            stackedShortTraj = np.hstack((stackedShortTraj, shortTraj))
            stackedShortPos  = np.hstack((stackedShortPos,  short_avgPos))
              
        sample_reach_idx.extend([rIdx for i in range(traj.shape[1])])
        sample_video_event.extend([reach.video_event for i in range(traj.shape[1])])
        
    # rearrange traj array into a list of arrays, with each element being the array of trajectories for a single marker
    traj_samples = [stackedTraj[mark] for mark in range(np.shape(stackedTraj)[0])]
    pos_samples  = [stackedPosTraj[mark] for mark in range(np.shape(stackedTraj)[0])]
    avg_pos_samples   = [stackedPos  [mark, ...] for mark in range(np.shape(stackedTraj)[0])]
    avg_speed_samples = [stackedSpeed[mark, ...] for mark in range(np.shape(stackedTraj)[0])]
    short_traj_samples    = [stackedShortTraj[mark, ...] for mark in range(np.shape(stackedShortTraj)[0])]
    short_avg_pos_samples = [stackedShortPos [mark, ...] for mark in range(np.shape(stackedShortPos )[0])]
    
    for mark in range(len(traj_samples)):
        samples_with_nan = np.where(np.isnan(avg_speed_samples[mark]))[0]
    
    sample_info, mark_pos = compute_kinematic_features(pos_samples, traj_samples, sample_reach_idx, sample_video_event)
    
    plot_kinematic_features(sample_info, mark_pos)
    
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
        pos_samples:  All the full-length position samples for the lead and lag of this model. 
                      For this model, the position is wrist_pos - shoulder_pos.        
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
                                     'pos_samples'           : pos_samples,
                                     'short_traj_samples'    : short_traj_samples,
                                     'short_avg_pos_samples' : short_avg_pos_samples,
                                     'avg_pos_samples'       : avg_pos_samples,
                                     'avg_speed_samples'     : avg_speed_samples,
                                     'spike_samples'         : spike_samples,
                                     'sample_info'           : sample_info,
                                     'description'           : sampled_data_description} 
    
    results_dict[lead_lag_key] = lead_lag_dict
    
    return

def convert_traj_samples_to_features(traj_samples, lead, lag):
    
    for mark, traj in enumerate(traj_samples):
        num_samples = int(round(traj.shape[-1]*params.subsamp_fps/params.fps))
        subsamp_idx = np.round(np.linspace(0, traj.shape[-1]-1, num=num_samples)).astype(int)
        full_times   = np.linspace(-1*lead, lag, num=traj.shape[-1]) 

        subsamp_traj = traj[..., subsamp_idx]
        subsamp_traj = subsamp_traj.reshape(subsamp_traj.shape[0], -1)        
        if 'traj_features' not in locals():
            traj_features = subsamp_traj
            subsamp_times = full_times[subsamp_idx] 
        else:
            traj_features = np.hstack((traj_features, subsamp_traj))
            subsamp_times = np.hstack((subsamp_times, full_times[subsamp_idx]))
    
    return traj_features, subsamp_times

def apply_standard_scaler(samples, mode):
    
    scaled_samples = [[] for i in range(len(samples))]

    if mode == 'pca_features':
        scaler = StandardScaler()
        scaled_samples = scaler.fit_transform(samples)
    elif mode == 'network':
        scaler = StandardScaler()
        scaled_samples = scaler.fit_transform(samples)
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

def sample_trajectories_and_spikes_for_model(units, reaches, kin_module, nwb):
    for lead, lag in zip(params.lead, params.lag):
        sample_trajectories_and_spikes(units, reaches, kin_module, nwb, lead, lag)
                                
    return 

def create_model_features_and_store_in_dict(lead_lag_key):
        
    lead = float(re.findall(re.compile('lead_[0-9]{1,3}'), lead_lag_key)[0].split('_')[-1]) * 1e-3
    lag  = float(re.findall(re.compile('lag_[0-9]{1,3}' ), lead_lag_key)[0].split('_')[-1]) * 1e-3
    
    lead_lag_dict = results_dict[lead_lag_key]
    sampled_data = lead_lag_dict['sampled_data']
    
    traj_samples          = sampled_data['traj_samples']
    pos_samples           = sampled_data['pos_samples']
    short_traj_samples    = sampled_data['short_traj_samples']
    avg_pos_samples       = sampled_data['avg_pos_samples']
    
    scaled_traj_samples          = apply_standard_scaler(traj_samples      , mode='traj')        
    scaled_pos_samples           = apply_standard_scaler(pos_samples       , mode='traj')        
    scaled_short_traj_samples    = apply_standard_scaler(short_traj_samples, mode='traj')        
    scaled_avg_pos_samples       = apply_standard_scaler(avg_pos_samples   , mode='avg_kin') 
    scaled_short_avg_pos_samples = apply_standard_scaler(avg_pos_samples   , mode='avg_kin') 
    
    traj_features, subsamp_times = convert_traj_samples_to_features(scaled_traj_samples, lead, lag)
    pos_features , _             = convert_traj_samples_to_features(scaled_pos_samples , lead, lag)
    short_traj_features, _       = convert_traj_samples_to_features(scaled_short_traj_samples, lead, lag)        
    traj_and_avgPos_features     = np.hstack((traj_features, scaled_avg_pos_samples[params.idx_for_avg_pos_and_speed]))        
    shortTraj_and_avgPos_features= np.hstack((short_traj_features, scaled_short_avg_pos_samples[params.idx_for_avg_pos_and_speed]))  

    lead_lag_dict = results_dict[lead_lag_key]        
    features_description = f'''
    Model features used as inputs for testing and training models.
    
    The key for each feature set describes what features were included.
    
    For all models, StandardScaler was used to standardize x,y,z separately. The samples were 
    concatenated to a matrix with dimension [numTrajSamples * datapointsPerTraj, 3]. 
    This means that x,y,z model inputs have identical means and variances,
    but full kinematic variability is maintained across samples.
    
    For PCA feature sets, trajectories were standardized prior to PCA. Appended features,
    such as the three average position terms, were standardized separately afterward.
    
    The timestamps for subsampled trajectory/position features are stored in subsample_times. The trajectories were
    subsampled to new_sample_rate={params.subsamp_fps} 
    ''' 
    lead_lag_dict['model_features'] = {'traj_avgPos'              : traj_and_avgPos_features,
                                       'shortTraj_avgPos'         : shortTraj_and_avgPos_features,
                                       'traj'                     : traj_features,
                                       'shortTraj'                : short_traj_features,
                                       'position'                 : pos_features,
                                       'subsample_times'          : subsamp_times,
                                       'description'              : features_description} 
    
    # lead_lag_dict['model_features'] = {'traj_pca'                 : traj_pca_features,
    #                                    'traj_pca_and_avgPos'      : traj_pca_features_pos,
    #                                    'traj'                     : traj_features,
    #                                    'position'                 : pos_features,
    #                                    'traj_and_avgPos'          : traj_and_avgPos_features,
    #                                    'traj_PCA_components'      : traj_comps,
    #                                    'subsample_times'          : subsamp_times,
    #                                    'description'              : features_description} 
    
    results_dict[lead_lag_key] = lead_lag_dict

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
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
if __name__ == "__main__":
    
    # if not skip_data_sampling:
    
    results_dict = dict()
    
    with NWBHDF5IO(nwb_infile, 'r') as io:
        nwb = io.read()

        reaches_key = [key for key in nwb.intervals.keys() if 'reaching_segments' in key][0]
        
        units, reaches, kin_module = get_sorted_units_and_apparatus_kinematics_with_metadata(nwb, reaches_key, mua_to_fix=params.mua_to_fix, plot=False) 
        
        units = choose_units_for_model(units, quality_key = 'snr', quality_thresh = params.snr_thresh, frate_thresh = params.frate_thresh, bad_units_list=bad_units_list)
    
        sample_trajectories_and_spikes_for_model(units, reaches, kin_module, nwb)

        for lead_lag_key in list(results_dict.keys()):
            create_model_features_and_store_in_dict(lead_lag_key) 

        # with open(pkl_outfile, 'wb') as f:
        #     dill.dump(results_dict, f, recurse=True)  
        
        save_dict_to_hdf5(results_dict, pkl_outfile.with_suffix('.h5')) 