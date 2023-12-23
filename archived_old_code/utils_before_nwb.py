#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 15:56:34 2023

@author: daltonm
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pickle
import dill
import os
import glob
import math
import re
import seaborn as sns
from scipy.ndimage import median_filter, gaussian_filter
from importlib import sys, reload

sys.path.insert(0, '/project/nicho/projects/marmosets/code_database/data_processing/nwb_tools/hatlab_nwb_tools/')
from hatlab_nwb_functions import plot_prb, read_prb_hatlab 

def load_data(path):
    
    print('loading original spike and position/reach data, analog camera signals, and functional networks')
    
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

def load_models(path, params):
    print('loading results from all models')
    
    spikeBins = math.modf(params.spkSampWin * 1e3)
    models_path = os.path.join(path.intermediate_save_path, '%dpt%d_ms_bins' % (spikeBins[1], spikeBins[0]*10))    
    results_paths = sorted(glob.glob(os.path.join(models_path, '%s_encoding_model_results_*%s*_shift_%d_PCAthresh_%d_norm_%s.pkl' % (path.date, 
                                                                                                                                     params.FN_source, 
                                                                                                                                     params.trajShift*1e3, 
                                                                                                                                     params.pca_var_thresh*100,
                                                                                                                                     params.normalize))))
    feature_paths = sorted(glob.glob(os.path.join(models_path, '%s_model_features_and_components_*%s*_shift_%d_PCAthresh_%d_norm_%s.pkl' % (path.date, 
                                                                                                                                            params.FN_source, 
                                                                                                                                            params.trajShift*1e3, 
                                                                                                                                            params.pca_var_thresh*100,
                                                                                                                                            params.normalize))))
    sampled_data_paths = sorted(glob.glob(os.path.join(models_path, '%s_model_trajectories_and_spikes*_shift_%d_with_network_lags.pkl' % (path.date, 
                                                                                                                                          params.trajShift*1e3))))

    results_list            = []
    unit_info_list          = []
    traj_features_list      = []
    network_features_list   = []
    short_features_list     = []
    components_list         = []
    sampled_spikes_list     = []
    lead_lag_list           = []
    sample_info_list        = []
    for r_path, f_path, s_path in zip(results_paths, feature_paths, sampled_data_paths):    
        with open(r_path, 'rb') as f:
            models_results, unit_info = dill.load(f)

        # all_model_results = {'model_results' : [full_model_results, traj_model_results, 
        #                                         network_model_results, short_model_results, 
        #                                         shuffle_model_results],
        #                      'model_names'   : ['full', 'trajectory', 'network', 
        #                                         'velocity', 'shuffle']}
        # with open(r_path.split('_separatedResults')[0] + '.pkl', 'wb') as f:
        #     dill.dump([all_model_results, unit_info], f, recurse=True)  
        with open(f_path, 'rb') as f:
            traj_features, network_features, short_features, compsOut = dill.load(f)
        with open(s_path, 'rb') as f:
            trajectoryList, shortTrajectoryList, avgPos, avgSpeed, sampledSpikes, reachSpikes, sample_info, tmp = dill.load(f)

        lead = int(re.findall(re.compile('lead_\d{1,3}'), r_path)[0].split('lead_')[-1])
        lag  = int(re.findall(re.compile('lag_\d{1,3}' ), r_path)[0].split('lag_' )[-1])
        results_list.append(models_results)
        unit_info_list.append(unit_info)
        traj_features_list.append(traj_features)
        network_features_list.append(network_features)
        short_features_list.append(short_features)
        components_list.append(compsOut)
        lead_lag_list.append((lead, lag))
        sampled_spikes_list.append(sampledSpikes)
        sample_info_list.append(sample_info)
    
    all_models_data = {'model_details'    : results_list,
                       'unit_info'        : unit_info_list,
                       'traj_features'    : traj_features_list,
                       'network_features' : network_features_list,
                       'short_features'   : short_features_list,
                       'components'       : components_list,
                       'lead_lag'         : lead_lag_list,
                       'sampled_spikes'   : sampled_spikes_list,
                       'sample_info'      : sample_info_list}
    
    return all_models_data

def load_all_models_dict(models_dict_path):
    with open(models_dict_path, 'rb') as f:
        all_models_data = dill.load(f)

    return all_models_data

def save_all_models_dict(models_dict_path, all_models_data):
    with open(models_dict_path, 'wb') as f:
        dill.dump(all_models_data, f, recurse=True)

def get_single_lead_lag_models(all_models_data, lead, lag):
    ll_idx = [idx for idx, ll in enumerate(all_models_data['lead_lag']) if ll == (lead*1e3, lag*1e3)][0]
    
    single_lead_lag_models = {}
    for key in all_models_data.keys(): 
        single_lead_lag_models[key] = all_models_data[key][ll_idx]
        
    return single_lead_lag_models, ll_idx

def filter_units(spike_data, frate_thresh):
    cluster_info = spike_data['cluster_info']
    
    unit_info = cluster_info.loc[cluster_info.group != 'noise', :]
    
    if 'snr' in unit_info.keys():
        quality = unit_info.snr
    else:
        quality = unit_info.amp
    
    quality_thresh = np.percentile(quality, 5)      
    # frate_thresh = np.percentile(unit_info.loc[unit_info.group == 'good', 'fr'], 5)
    
    unit_info = unit_info.loc[(quality > quality_thresh) | (cluster_info.group == 'good'), :]
    unit_info = unit_info.loc[unit_info.fr > frate_thresh, :]
    
    unit_info.reset_index(inplace=True, drop=False)
    
    return unit_info

def get_interelectrode_distances_by_unit(unit_info, chan_map_df, array_type ='utah'):
        
    electrode_distances = np.full((unit_info.shape[0], unit_info.shape[0]), np.nan)
    if array_type.lower() == 'utah':
        # for i, i_chanID in enumerate(unit_info['ns6_elec_id']):
        #     for j, j_chanID in enumerate(unit_info['ns6_elec_id']):
        #         if i == j:
        #             continue
        #         i_pos = chan_map_df.loc[chan_map_df.shank_ids == str(i_chanID), ['x','y']].values[0]
        #         j_pos = chan_map_df.loc[chan_map_df.shank_ids == str(j_chanID), ['x','y']].values[0] 
        #         electrode_distances[i, j] = np.sqrt((i_pos[0] - j_pos[0])**2 + (i_pos[1] - j_pos[1])**2)

        for i, (ix, iy) in enumerate(zip(unit_info['x'], unit_info['y'])):
            for j, (jx, jy) in enumerate(zip(unit_info['x'], unit_info['y'])):
                if i == j:
                    continue
                electrode_distances[i, j] = np.sqrt((ix - jx)**2 + (iy - jy)**2)

    return electrode_distances 

def fix_unit_info_elec_labels(unit_info, chan_map_df):
    if unit_info.index.to_list() != list(range(unit_info.shape[0])):
        unit_info.reset_index(drop=False, inplace=True)
    
    fixed_labels = [0]*unit_info.shape[0]
    x            = [0]*unit_info.shape[0]
    y            = [0]*unit_info.shape[0]
    z            = [-1]*unit_info.shape[0]
    for row, unit_row in unit_info.iterrows():
        fixed_labels[row] = chan_map_df.loc[unit_row.ch, 'shank_ids']
        x[row] = int(chan_map_df.loc[unit_row.ch, 'x'])
        y[row] = int(chan_map_df.loc[unit_row.ch, 'y'])
        try:
            z[row] = int(chan_map_df.loc[unit_row.ch, 'z'])
        except:
            pass
        
    unit_info['uncorrected_ns6_elec_id'] = unit_info['ns6_elec_id']
    unit_info['ns6_elec_id'] = fixed_labels    
    unit_info['x'] = x
    unit_info['y'] = y
    if not all([el==-1 for el in z]):
        unit_info['z'] = z
    
    return unit_info  
# def fix_unit_info_elec_labels(unit_info, chan_map_df):
#     if unit_info.index.to_list() != list(range(unit_info.shape[0])):
#         unit_info.reset_index(drop=False, inplace=True)
    
#     if 'x' not in unit_info.columns:    
#         fixed_labels = [0]*unit_info.shape[0]
#         x            = [0]*unit_info.shape[0]
#         y            = [0]*unit_info.shape[0]
#         z            = [-1]*unit_info.shape[0]
#         for row, unit_row in unit_info.iterrows():
#             fixed_labels[row] = int(chan_map_df.loc[unit_row.ch, 'shank_ids'])
#             x[row] = int(chan_map_df.loc[unit_row.ch, 'x'])
#             y[row] = int(chan_map_df.loc[unit_row.ch, 'y'])
#             try:
#                 z[row] = int(chan_map_df.loc[unit_row.ch, 'z'])
#             except:
#                 pass
            
#         unit_info['uncorrected_ns6_elec_id'] = unit_info['ns6_elec_id']
#         unit_info['ns6_elec_id'] = fixed_labels    
#         unit_info['x'] = x
#         unit_info['y'] = y
#         if not all([el==-1 for el in z]):
#             unit_info['z'] = z       
    
#     return unit_info     

def choose_units_for_model(units, quality_percentile = 5, frate_thresh = 2):
    
    if 'snr' in units.keys():
        quality = units.snr
    else:
        quality = units.amp
    
    quality_thresh = np.percentile(quality, quality_percentile)      
    
    units = units.loc[(quality > quality_thresh) | (units.quality == 'good'), :]
    units = units.loc[units.fr > frate_thresh, :]
        
    return units

def compute_derivatives(marker_pos, fps, smooth = True):
    marker_vel = np.diff(marker_pos, axis = -1) * fps
    if smooth:
        for dim in range(3):
            marker_vel[dim] = gaussian_filter(marker_vel[dim], sigma=1.5)

    marker_acc = np.diff(marker_vel, axis = -1) * fps
    if smooth:
        for dim in range(3):
            marker_acc[dim] = gaussian_filter(marker_acc[dim], sigma=1.5)
    
    return marker_vel, marker_acc


def load_channel_map_from_prb(marm = 'Tony'):
    if marm.lower() == 'tony':
        map_path = '/project/nicho/data/marmosets/prbfiles/TY_02.prb'
    elif marm.lower() == 'midge':
        map_path = '/project/nicho/data/marmosets/prbfiles/MG_01.prb'
        
    chan_map_probegroup, imp = read_prb_hatlab(map_path)
    plot_prb(chan_map_probegroup)
    chan_map_df = chan_map_probegroup.to_dataframe()
    
    return chan_map_df

def load_color_palette(palette_path):
    LinL = np.loadtxt(palette_path, delimiter=',')
    
    b3=LinL[:,2] # value of blue at sample n
    b2=LinL[:,2] # value of blue at sample n
    b1=np.linspace(0,1,len(b2)) # position of sample n - ranges from 0 to 1
    
    # setting up columns for list
    g3=LinL[:,1]
    g2=LinL[:,1]
    g1=np.linspace(0,1,len(g2))
    
    r3=LinL[:,0]
    r2=LinL[:,0]
    r1=np.linspace(0,1,len(r2))
    
    # creating list
    R=zip(r1,r2,r3)
    G=zip(g1,g2,g3)
    B=zip(b1,b2,b3)
    
    # transposing list
    RGB=zip(R,G,B)
    rgb=zip(*RGB)
    # print rgb
    
    # creating dictionary
    k=['red', 'green', 'blue']
    LinearL=dict(zip(k,rgb)) # makes a dictionary from 2 lists
    my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',LinearL)
    
    return my_cmap
