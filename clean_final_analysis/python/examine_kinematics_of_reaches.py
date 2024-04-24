#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 10:09:47 2023

@author: daltonm
"""

from pynwb import NWBHDF5IO
import ndx_pose
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D
from importlib import sys
import neo
import elephant
from viziphant.rasterplot import rasterplot
from viziphant.events import add_event
from quantities import s
from scipy.ndimage import gaussian_filter
import os
import seaborn as sns
import dill
from scipy.signal import savgol_filter
import pandas as pd
from pathlib import Path

data_path = Path('/project/nicho/projects/dalton/network_encoding_paper/clean_final_analysis/data')
code_path = Path('/project/nicho/projects/marmosets/code_database/analysis/trajectory_encoding_model/clean_final_analysis/')

sys.path.insert(0, str(code_path))
from hatlab_nwb_functions import get_sorted_units_and_apparatus_kinematics_with_metadata, remove_duplicate_spikes_from_good_single_units   

marmcode = 'TY'
lead_lag_key = 'lead_100_lag_300'
fig_mode='paper'
pkl_in_tag = 'kinematic_models_summarized'

if marmcode == 'TY':
    nwb_infile = data_path / 'TY' / 'TY20210211_freeAndMoths-003_resorted_20230612_DM_with_functional_networks.nwb' 
    # area_order=['M1', '3a', '3b']
    area_order=['Motor', 'Sensory']
    reaches_to_plot = [3]
    mod_start_timestamps = [None, 190.477]
    mod_end_timestamps = [189.875, 191.375]
    raster_size_multiple = 1
    fps = 150
    view_angle = (28, 148)#(14, 146)
    apparatus_min_dims = [0, 0, 0]
    apparatus_max_dims = [14, 12.5, 7]
elif marmcode == 'MG':
    nwb_infile = data_path / 'MG' / 'MG20230416_1505_mothsAndFree-002_processed_DM_with_functional_networks.nwb'
    # area_order=['6dc', 'M1', '3a']
    area_order=['Motor', 'Sensory']
    reaches_to_plot=[14]
    mod_start_timestamps = [None]
    mod_end_timestamps=[1779]
    raster_size_multiple = 73/175
    fps = 200
    view_angle = (28, 11)
    apparatus_min_dims = [6, -3, 0]
    apparatus_max_dims = [14, 10, 5]

pkl_infile   = nwb_infile.parent / f'{nwb_infile.stem.split("_with_functional_networks")[0]}_{pkl_in_tag}.pkl'

# color1     = (159/255, 206/255, 239/255)
color1     = (  0/255, 141/255, 208/255)
color2     = (  0/255, 141/255, 208/255)
spontColor = (183/255, 219/255, 165/255)

dataset_code = pkl_infile.stem.split('_')[0]
if fig_mode == 'paper':
    plots = nwb_infile.parent.parent.parent / 'plots' / dataset_code
elif fig_mode == 'pres':
    plots = nwb_infile.parent.parent.parent / 'presentation_plots' / dataset_code    
    
os.makedirs(plots, exist_ok=True)

class plot_params:
    
    fps = fps
    
    if fig_mode == 'paper':
        axis_fontsize = 8
        dpi = 300
        axis_linewidth = 1
        tick_length = 1.75
        tick_width = 0.5
        tick_fontsize = 8
        
        spksamp_markersize = 4
        vel_markersize = 2
        
        # raster_ticksize = 0.01
        # raster_figsize = (10, 2)
        # raster_pretime = 1
        # raster_posttime= 7
        
        modulation_palette = 'cool'
        raster_figsize = (2.7, 3*raster_size_multiple)
        raster_ticksize = 0.003
        raster_pretime = 1
        raster_posttime= 3
        
        traj_pos_sample_figsize = (2.1, 1.9)
        traj_vel_sample_figsize = (1.4, 1.5)
        reaches_figsize = (2.5, 2.25) 
        traj_linewidth = 1
        traj_leadlag_linewidth = 2
        vel_sample_tickfontsize = 8
        
        weights_by_distance_figsize = (2.5, 1.5)
        aucScatter_figSize = (2.5, 2.5)
        FN_figsize = (3, 3)
        feature_corr_figSize = (3, 3)


    elif fig_mode == 'pres':
        
        axis_fontsize = 20
        dpi = 300
        axis_linewidth = 2
        tick_length = 2
        tick_width = 1
        tick_fontsize = 18
            
        spksamp_markersize = 12
        vel_markersize = 8
        traj_length_markersize = 6
        scatter_markersize = 30
        stripplot_markersize = 2
        feature_corr_markersize = 8
        wji_vs_trajcorr_markersize = 2
        
        raster_figsize = (6, 8)
        raster_ticksize = 0.05
        raster_pretime = 1
        raster_posttime= 3
        
        shuffle_markerscale = 1
        shuffle_errwidth = 3
        shuffle_sigmarkersize = 3
        shuffle_figsize = (12, 3)
        
        traj_pos_sample_figsize = (3, 3.25)
        traj_vel_sample_figsize = (2.5, 2.5)
        traj_linewidth = 3
        traj_leadlag_linewidth = 4
        vel_sample_tickfontsize = 16
        
        preferred_traj_linewidth = 2
        distplot_linewidth = 3
        preferred_traj_figsize = (5, 5)
        
        weights_by_distance_figsize = (6, 4)
        aucScatter_figSize = (4.5, 4.5)
        FN_figsize = (5, 5)
        feature_corr_figSize = (4, 4)
        trajlength_figsize = (5, 5)
        pearsonr_histsize = (3, 3)
        distplot_figsize = (3, 2)
        stripplot_figsize = (6, 3)
        scatter_figsize = (5, 5)

        corr_marker_color = 'gray'

class fig3_params:
    dirColors = np.array([[27 , 158, 119],
                          [217, 95 , 2  ],
                          [117, 112, 179]])
    figSize   = (3.5, 2.75)
    figSize3d = (4.25, 4.25)
    angles3d = [18, -170]
    ylim = [-9, 12]
    thickMult = 0.012 
    textMult = 4
    lw = 1.25
    startMarkSize = 6
    unit = 'cm'
    yLen = 5
    panelB_scalebar_yStart = 6
    panelCD_scalebar_yStart = 2
    
    histFigSize = (6, 3)
    hist_lw = 2
    hist_median_lw = 2
    
    dlc_fps = 200
    xromm_fps = 200   
    
plt.rcParams['figure.dpi'] = plot_params.dpi
plt.rcParams['savefig.dpi'] = plot_params.dpi
plt.rcParams['font.family'] = 'Dejavu Sans'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.labelsize'] = plot_params.axis_fontsize 
plt.rcParams['axes.linewidth'] = plot_params.axis_linewidth
plt.rcParams['xtick.labelsize'] = plot_params.tick_fontsize
plt.rcParams['xtick.major.size'] = plot_params.tick_length
plt.rcParams['xtick.major.width'] = plot_params.tick_width
plt.rcParams['ytick.labelsize'] = plot_params.tick_fontsize
plt.rcParams['ytick.major.size'] = plot_params.tick_length
plt.rcParams['ytick.major.width'] = plot_params.tick_width
plt.rcParams['legend.fontsize'] = plot_params.axis_fontsize
plt.rcParams['legend.loc'] = 'upper left'
plt.rcParams['legend.borderaxespad'] = 1.1
plt.rcParams['legend.borderpad'] = 1.1


def get_sorted_units_and_apparatus_kinematics_with_metadata(nwb_prc, reaches_key, plot=False):
    units          = nwb_prc.units.to_dataframe()
    units = remove_duplicate_spikes_from_good_single_units(units, plot=plot)
    reaches        = nwb_prc.intervals[reaches_key].to_dataframe()
    
    kin_module_key = reaches.iloc[0].kinematics_module
    kin_module = nwb_prc.processing[kin_module_key]
    
    return units, reaches, kin_module

def add_in_weight_to_units_df(units_res, FN):
    
    units_res = units_res.copy()
    
    in_weights     = []
    for unit_idx in units_res.index:
        if FN.ndim == 3:
            w_in  = (FN[0, unit_idx].sum() + FN[1, unit_idx].sum())/2 / FN.shape[-1]
        else:
            w_in  = FN[unit_idx].sum()
        in_weights.append(w_in)
    
    tmp_df = pd.DataFrame(data = zip(in_weights),
                          columns = ['W_in'],
                          index = units_res.index)

    
    units_res = pd.concat((units_res, tmp_df), axis = 1)
    
    return units_res

def get_event_spiketrains_and_events(units, spike_times, align_times, preTime=1, postTime=1, mod_index_mode = 'start'):

    spiketrains = [[] for i in align_times]
    for idx, t_align in enumerate(align_times):
        spike_times_aligned = spike_times - t_align
        spike_times_aligned = [spk for spk in spike_times_aligned if spk > -1*preTime and spk < postTime]
        spiketrains[idx] = neo.spiketrain.SpikeTrain(spike_times_aligned*s, 
                                                     t_start=-1*preTime*s, 
                                                     t_stop =postTime*s)
    
    PETH = elephant.statistics.time_histogram(spiketrains, 
                                              0.05*s, 
                                              t_start=None, 
                                              t_stop=None, 
                                              output='rate', 
                                              binary=False)
    
    if mod_index_mode == 'savgol':
        savFilt = savgol_filter(PETH.as_array().flatten(), 13, 3)
        # savFilt = medfilt(PETH.as_array().flatten(), 7)

        # mod_index = round((savFilt.max() - savFilt.min())/savFilt.mean(), 2)
        mod_index = round((savFilt.max() - savFilt.min()), 2)

    else:
        center_bin = int(preTime / .05)
        if mod_index_mode == 'start':
            baseline_bins = list(range(0, int(center_bin-.25/.05)))
            mod_bins = list(range(center_bin, int(center_bin+.75/.05+1)))
        elif mod_index_mode == 'stop':
            baseline_bins = list(range(0, center_bin))
            mod_bins = list(range(center_bin, int(center_bin+postTime/.05+1)))
        elif mod_index_mode == 'peak':
            baseline_bins = list(range(0, int(center_bin-.75/.05+1))) + list(range(int(center_bin+.75/.05), int(center_bin+postTime/.05+1)))
            mod_bins = list(range(int(center_bin-.25/.05), int(center_bin+.25/.05+1)))
        
        baseline_mask = np.array([True if idx in baseline_bins else False for idx in range(PETH.shape[0])])
        mod_mask      = np.array([True if idx in      mod_bins else False for idx in range(PETH.shape[0])])
    
        mod_index = round(PETH.as_array()[mod_mask].mean() / PETH.as_array()[baseline_mask].mean(), 2)
            
    return spiketrains, PETH, mod_index

        
def compute_derivatives(marker_pos=None, marker_vel=None, smooth = True):
    
    if marker_pos is not None and marker_vel is None:
        marker_vel = np.diff(marker_pos, axis = -1) * plot_params.fps
        if smooth:
            for dim in range(3):
                marker_vel[dim] = gaussian_filter(marker_vel[dim], sigma=1.5)
        
    marker_acc = np.diff(marker_vel, axis = -1) * plot_params.fps
    if smooth:
        for dim in range(3):
            marker_acc[dim] = gaussian_filter(marker_acc[dim], sigma=1.5)
    
    return marker_vel, marker_acc

def identify_extension_and_retraction(reaches, plot=False):
    
    pos_color = 'black'
    extension_color = 'green'
    retraction_color = 'purple'
    linewidth = 1
    
    first_event_key = [key for idx, key in enumerate(kin_module.data_interfaces.keys()) if idx == 0][0]
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
    
    ext_start, ext_stop, ret_start, ret_stop = [], [], [], []
    for reachNum, reach in reaches.iterrows():      
                
        # get event data using container and ndx_pose names from segment_info table following form below:
        # nwb.processing['goal_directed_kinematics'].data_interfaces['moths_s_1_e_004_position']
        event_data      = kin_module.data_interfaces[reach.video_event] 
        
        wrist_kinematics    = event_data.pose_estimation_series[   wrist_label].data[reach.start_idx:reach.stop_idx+1].T
        shoulder_kinematics = event_data.pose_estimation_series[shoulder_label].data[reach.start_idx:reach.stop_idx+1].T
        timestamps          = event_data.pose_estimation_series[   wrist_label].timestamps[reach.start_idx:reach.stop_idx+1]
    
        extension_distance = np.sqrt(np.square(wrist_kinematics - shoulder_kinematics).sum(axis=0))
        extension_distance_filtered = gaussian_filter(extension_distance, sigma=11)
        extension_distance_filtered[np.isnan(extension_distance_filtered)] = extension_distance[np.isnan(extension_distance_filtered)]
        
        distance_change     = np.diff(extension_distance_filtered)
        extension_idxs  = np.where(distance_change > 0)[0]
        retraction_idxs = np.where(distance_change < 0)[0]
        ext_bound_idxs = np.hstack((0, 
                                    np.where(np.diff(extension_idxs) > 1)[0]+1,
                                    extension_idxs.size))
        ret_bound_idxs = np.hstack((0, 
                                    np.where(np.diff(retraction_idxs) > 1)[0]+1,
                                    retraction_idxs.size))
        
        extension_segments  = [extension_idxs [start:stop] for start, stop in zip(ext_bound_idxs[:-1], ext_bound_idxs[1:])]
        retraction_segments = [retraction_idxs[start:stop] for start, stop in zip(ret_bound_idxs[:-1], ret_bound_idxs[1:])]
        
        for seg in extension_segments:
            ext_start.append(timestamps[seg[ 0]])
            ext_stop.append (timestamps[seg[-1]])
        for seg in retraction_segments:
            ret_start.append(timestamps[seg[ 0]])
            ret_stop.append (timestamps[seg[-1]])
        
        if plot:
            fig0 = plt.figure(figsize = plot_params.traj_pos_sample_figsize, dpi=plot_params.dpi)
            ax0 = plt.axes(projection='3d')
            
            ax0.plot3D(wrist_kinematics[0] , wrist_kinematics[1], wrist_kinematics[2], linewidth=linewidth, color=pos_color, linestyle='-')
            for ext_seg in extension_segments:
                ax0.plot3D(wrist_kinematics[0, ext_seg], wrist_kinematics[1, ext_seg], wrist_kinematics[2, ext_seg], linewidth=linewidth, color= extension_color, linestyle='-')
            for ret_seg in retraction_segments:
                ax0.plot3D(wrist_kinematics[0, ret_seg], wrist_kinematics[1, ret_seg], wrist_kinematics[2, ret_seg], linewidth=linewidth, color=retraction_color, linestyle='-')
            ax0.plot3D(wrist_kinematics[0, 0], wrist_kinematics[1, 0], wrist_kinematics[2, 0], 
                       marker='o', markersize=plot_params.spksamp_markersize, markeredgecolor='black', markerfacecolor='white')
            
            for ax in [ax0]:
                ax.set_xlabel('', fontsize = plot_params.axis_fontsize)
                ax.set_ylabel('', fontsize = plot_params.axis_fontsize)
                ax.set_zlabel('', fontsize = plot_params.axis_fontsize)
                ax.set_xlim(apparatus_min_dims[0], apparatus_max_dims[0]),
                ax.set_ylim(apparatus_min_dims[1], apparatus_max_dims[1])
                ax.set_zlim(apparatus_min_dims[2], apparatus_max_dims[2])
                ax.set_xticks([apparatus_min_dims[0], apparatus_max_dims[0]], labels=['','']),
                ax.set_yticks([apparatus_min_dims[1], apparatus_max_dims[1]], labels=['',''])
                ax.set_zticks([apparatus_min_dims[2], apparatus_max_dims[2]], labels=['',''])
                ax.set_title(f'Reach = {reachNum}')
    
                ax.view_init(view_angle[0], view_angle[1])
            
            fig1, ax1 = plt.subplots(figsize = plot_params.traj_vel_sample_figsize, dpi=plot_params.dpi)        
            ax1.plot(timestamps, extension_distance)
            for ext_seg in extension_segments:
                ax1.plot(timestamps[ext_seg], extension_distance_filtered[ext_seg], color=extension_color)
            for ret_seg in retraction_segments:
                ax1.plot(timestamps[ret_seg], extension_distance_filtered[ret_seg], color=retraction_color)
            
            plt.show()
    
            # fig0.savefig(plots / paperFig / f'{marmcode}_trajectory_sampling_pos_{sampleNum}.png', bbox_inches='tight', dpi=plot_params.dpi)
    
        
    extension_times = pd.DataFrame(data=zip(ext_start, ext_stop),
                                   columns=['start', 'stop'])   
    extension_times.to_hdf(data_path / 'TY' / 'reaching_extend_retract_segments.h5', key='extension')
    retraction_times = pd.DataFrame(data=zip(ret_start, ret_stop),
                                   columns=['start', 'stop'])   
    retraction_times.to_hdf(data_path / 'TY' / 'reaching_extend_retract_segments.h5', key='retraction')  
    
    return extension_times, retraction_times

def examine_single_reach_kinematic_distributions(reaches, plot=False):
    
    pos_color = 'black'
    linewidth = 1
    
    first_event_key = [key for idx, key in enumerate(kin_module.data_interfaces.keys()) if idx == 0][0]
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
    

    kin_df = pd.DataFrame()
    for start, stop in zip(range(0, 91, 10), range(10, 102, 10)):
        # fig0 = plt.figure(figsize = plot_params.traj_pos_sample_figsize, dpi=plot_params.dpi)
        # ax0 = plt.axes(projection='3d')
        for reachNum, reach in reaches.iloc[start:stop, :].iterrows():      
                    
            # get event data using container and ndx_pose names from segment_info table following form below:
            # nwb.processing['goal_directed_kinematics'].data_interfaces['moths_s_1_e_004_position']
            event_data      = kin_module.data_interfaces[reach.video_event] 
            
            wrist_kinematics    = event_data.pose_estimation_series[   wrist_label].data[reach.start_idx:reach.stop_idx+1].T
            shoulder_kinematics = event_data.pose_estimation_series[shoulder_label].data[reach.start_idx:reach.stop_idx+1].T
            timestamps          = event_data.pose_estimation_series[   wrist_label].timestamps[reach.start_idx:reach.stop_idx+1]
        
            pos = wrist_kinematics - shoulder_kinematics
            vel, tmp_acc = compute_derivatives(marker_pos=pos, marker_vel=None, smooth = True)
            
            tmp_df = pd.DataFrame(data=zip(np.sqrt(np.square(vel).sum(axis=0)),
                                           vel[0],
                                           vel[1],
                                           vel[2],
                                           pos[0, :-1],
                                           pos[1, :-1],
                                           pos[2, :-1],
                                           np.repeat(reachNum, vel.shape[-1]),),
                                  columns=['speed', 'vx', 'vy', 'vz', 'x', 'y', 'z', 'reach',])
            kin_df = pd.concat((kin_df, tmp_df))
                
        #     ax0.scatter(wrist_kinematics[0], wrist_kinematics[1], wrist_kinematics[2], 
        #                   marker='.', s=plot_params.spksamp_markersize/50)
            
        #     for ax in [ax0]:
        #         ax.set_xlabel('', fontsize = plot_params.axis_fontsize)
        #         ax.set_ylabel('', fontsize = plot_params.axis_fontsize)
        #         ax.set_zlabel('', fontsize = plot_params.axis_fontsize)
        #         ax.set_xlim(apparatus_min_dims[0], apparatus_max_dims[0]),
        #         ax.set_ylim(apparatus_min_dims[1], apparatus_max_dims[1])
        #         ax.set_zlim(apparatus_min_dims[2], apparatus_max_dims[2])
        #         ax.set_xticks([apparatus_min_dims[0], apparatus_max_dims[0]], labels=['','']),
        #         ax.set_yticks([apparatus_min_dims[1], apparatus_max_dims[1]], labels=['',''])
        #         ax.set_zticks([apparatus_min_dims[2], apparatus_max_dims[2]], labels=['',''])
        #         ax.set_title(f'Reach = {reachNum}')
    
        #         ax.view_init(view_angle[0], view_angle[1])
                
        # plt.show()
    
    kin_df = kin_df.loc[~np.isnan(kin_df['speed'])]
    for key in kin_df.columns:
        if key == 'reach':
            continue

        fig, ax = plt.subplots(figsize = (8, 4), dpi=plot_params.dpi)
        sns.kdeplot(data=kin_df, ax=ax, x=key, hue='reach', legend = False, linewidth = 1, common_norm=False, bw_adjust=0.5)
        ax.set_title(key)
        if key in ['vx', 'vy', 'vz',]:
            ax.set_xlim(-75, 75)
        elif key in ['x', 'y', 'z',]:
            ax.set_xlim(-10, 10)
        elif key == 'speed':
            ax.set_xlim(0, 100)
        plt.show()
    
    for vel in ['vx', 'vy', 'vz']:
        kin_df[f'{vel}_mag'] = np.abs(kin_df[vel])
    g = sns.PairGrid(kin_df.loc[:, ['vx_mag', 'vy_mag', 'vz_mag']])
    g.map_upper(sns.scatterplot, s=2)
    g.map_lower(sns.scatterplot, s=2)
    g.map_diag(sns.kdeplot, lw=2)
    plt.show()
            # fig0.savefig(plots / paperFig / f'{marmcode}_trajectory_sampling_pos_{sampleNum}.png', bbox_inches='tight', dpi=plot_params.dpi)
     
    
if __name__ == '__main__':
    # io_acq = NWBHDF5IO(nwb_acquisition_file, mode='r')
    # nwb_acq = io_acq.read()
    
    with open(pkl_infile, 'rb') as f:
        results_dict = dill.load(f)
    units_res = results_dict[lead_lag_key]['all_models_summary_results']
    
    with NWBHDF5IO(nwb_infile, 'r') as io:
        nwb = io.read()

        FN = nwb.scratch['split_reach_FNs'].data[:] 
        units_res = add_in_weight_to_units_df(units_res, FN.copy())

        reaches_key = [key for key in nwb.intervals.keys() if 'reaching_segments' in key][0]
        
        units, reaches, kin_module = get_sorted_units_and_apparatus_kinematics_with_metadata(nwb, reaches_key, plot=False)
    
        examine_single_reach_kinematic_distributions(reaches, plot=True)        
    
        # extension_times, retraction_times = identify_extension_and_retraction(reaches, plot=True)
    
        # plot_fig1_trajectory_sampling(units_res, reaches, reachNum = reaches_to_plot[0], 
        #                               mod_start_timestamps = mod_start_timestamps, 
        #                               mod_end_timestamps = mod_end_timestamps,
        #                               pos_color = 'black', lead_color = 'blue', lag_color = 'red',
        #                               modulation_palette = plot_params.modulation_palette, paperFig='Fig1')    
    

    
