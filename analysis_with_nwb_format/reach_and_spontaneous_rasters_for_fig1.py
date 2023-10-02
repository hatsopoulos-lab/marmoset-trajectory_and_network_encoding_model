#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 10:09:47 2023

@author: daltonm
"""

from pynwb import NWBHDF5IO
from nwbwidgets import nwb2widget
import ndx_pose
import numpy as np
import matplotlib.pyplot as plt
from importlib import sys
import neo
import elephant
from viziphant.rasterplot import rasterplot
from viziphant.events import add_event
from quantities import s
from os.path import join as pjoin
from scipy.ndimage import median_filter, gaussian_filter
import os
import seaborn as sns
import dill
from scipy.signal import savgol_filter, medfilt
import pandas as pd

sys.path.insert(0, '/project/nicho/projects/marmosets/code_database/data_processing/nwb_tools/hatlab_nwb_tools/')
from hatlab_nwb_functions import remove_duplicate_spikes_from_good_single_units, get_raw_timestamps   


#nwb_file = '/project/nicho/data/marmosets/electrophys_data_for_processing/TY20210211_freeAndMoths/TY20210211_freeAndMoths-003.nwb'
#nwb_file = '/project/nicho/data/marmosets/electrophys_data_for_processing/MG20230416_1505_mothsAndFree/MG20230416_1505_mothsAndFree-002_acquisition.nwb'
# nwb_acquisition_file = '/project/nicho/data/marmosets/electrophys_data_for_processing/TY20210211_freeAndMoths/TY20210211_freeAndMoths-003_acquisition.nwb'

marmcode = 'TY'

if marmcode == 'TY':
    nwb_analysis_file = '/project/nicho/projects/dalton/data/TY/TY20210211_freeAndMoths-003_resorted_20230612_DM_with_functional_networks.nwb' 
    pkl_infile = '/project/nicho/projects/dalton/data/TY/TY20210211_freeAndMoths-003_resorted_20230612_DM_alpha_pt00001_encoding_models_30ms_shift_v4.pkl' 
    plot_storage = '/project/nicho/projects/dalton/plots/TY20210211/Fig1'
    area_order=['M1', '3a', '3b']
    # reaches_to_plot=[76, 77, 78, 79, 80, 81]
elif marmcode == 'MG':
    nwb_analysis_file = '/project/nicho/projects/dalton/data/MG/MG20230416_1505_mothsAndFree-002_processed_DM_with_functional_networks.nwb'
    pkl_infile = '/project/nicho/projects/dalton/data/MG/MG20230416_1505_mothsAndFree-002_processed_DM_alpha_pt00001_removedUnits_181_440_fixedMUA_745_796_encoding_models_30ms_shift_v4.pkl'
    plot_storage = '/project/nicho/projects/dalton/plots/MG20230416/Fig1'
    area_order=['6dc', 'M1', '3a']
    reaches_to_plot=[42, 43, 44, 45, 46, 47]

lead_lag_key = 'lead_100_lag_300'

# color1     = (159/255, 206/255, 239/255)
color1     = (  0/255, 141/255, 208/255)
color2     = (  0/255, 141/255, 208/255)
spontColor = (183/255, 219/255, 165/255)

fig_mode='pres'

if fig_mode == 'pres':
    plot_storage = plot_storage.replace('plots', 'defense_plots')
    # reaches_to_plot = [77, 78, 79, 80, 81]
    reaches_to_plot = [3]  #[23, 24, 25, 26]

    
    
os.makedirs(plot_storage, exist_ok=True)

class plot_params:
    
    fps = 150
    
    if fig_mode == 'paper':
        axis_fontsize = 11
        dpi = 300
        axis_linewidth = 2
        tick_length = 1.5
        tick_width = 1
        tick_fontsize = 8
        
        spksamp_markersize = 4
        vel_markersize = 2
        
        raster_ticksize = 0.01
        raster_figsize = (10, 2)
        raster_pretime = 1
        raster_posttime= 7
        
        traj_pos_sample_figsize = (1.75, 1.75)
        traj_vel_sample_figsize = (1.5  ,   1.5)
        traj_linewidth = 1
        traj_leadlag_linewidth = 2
        
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

def generate_event_raster(units, units_res, reaches, preTime=1, postTime=1, 
                          figsize=(5, 2), marker='.', msize=0.1, 
                          reaches_to_plot= [10, 11, 12], sort_by = 'traj_avgPos_auc', ascend = False, 
                          units_to_plot=None, color1='green', color2='blue', alpha=.2, rng_seed=10):
    reach_start_times = [reach.start_time for idx, reach in reaches.iterrows()]
    reach_end_times   = [reach.stop_time for idx, reach in reaches.iterrows()]
    reach_peak_times  = [float(reach.peak_extension_times.split(',')[0]) for idx, reach in reaches.iterrows() if len(reach.peak_extension_times)>0]
        
    if units_to_plot is None:
        units_to_plot = [row for row, unit in units.iterrows()]
    elif type(units_to_plot) == int:
        sorted_units_res = units_res.sort_values(by=sort_by, ascending=ascend)
        tmp_units = []
        yticks = []
        ylabels = []
        prevBound = 0
        for area in area_order:
            area_res = sorted_units_res[sorted_units_res['cortical_area'] == area]
            nUnits = round(area_res.shape[0] * units_to_plot/175)
            mask = [True if idx < nUnits else False for idx in range(area_res.shape[0])]
            tmp_units = tmp_units + area_res.loc[mask, 'unit_name'].to_list() 
            
            lowerBound = prevBound
            upperBound = prevBound + nUnits - 1
            yticks.extend([(lowerBound + upperBound)/2, upperBound])
            ylabels.extend([area, ''])
            prevBound += nUnits
        units_to_plot = tmp_units
    
    ylabels[-1] = upperBound+1
    
    reach_starts_to_plot = [start for idx, start in enumerate(reach_start_times) if idx in reaches_to_plot]
    reach_ends_to_plot   = [stop  for idx, stop  in enumerate(reach_end_times)   if idx in reaches_to_plot]
     
    fig_label  = f'rasterplot_reaches_{reaches_to_plot[0]}_thru_{reaches_to_plot[-1]}.png'
    peth_label = f'raster_and_peth_plot_reaches_{reaches_to_plot[0]}_thru_{reaches_to_plot[-1]}.png'  
    
    spiketrains = []
    for unit_name in units_to_plot:# in units.iterrows():
                
        spike_times = units[units.unit_name==unit_name].iloc[0].spike_times
        
        segment_start = reach_starts_to_plot[0] - preTime
        segment_end   = reach_ends_to_plot[-1] + postTime
        
        segment_spike_times = [spk for spk in spike_times if spk > segment_start and spk < segment_end]
        
        st = neo.spiketrain.SpikeTrain(segment_spike_times*s, t_start=segment_start*s, t_stop = segment_end*s)
    
        spiketrains.append(st)

    fig0, (ax0, ax1) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [5, 1]}, dpi=300)    
    rasterplot(spiketrains, axes=(ax0, ax1), color='black', s=msize, histogram_bins = 50, marker=marker)

    for start, end in zip(reach_starts_to_plot, reach_ends_to_plot): 
        events = neo.Event([start, end] * s, labels=['Reach Onset', 'Reach End'])
        add_event((ax0, ax1), event=events)

    # raster_kin_times = [1 + reach_starts_to_plot[0], 2.5 + reach_starts_to_plot[0]]
    # events = neo.Event(raster_kin_times * s, labels=['', ''])
    # add_event((ax0, ax1), event=events)
    
    fig0.savefig(os.path.join(plot_storage, peth_label), dpi=300)
   
    

    fig, ax = plt.subplots(figsize=figsize, dpi=300)    
    rasterplot(spiketrains, axes=ax, color='black', s=msize, marker=marker)

    tmp = list(range(len(reach_starts_to_plot)))
    reachset1 = np.random.default_rng(rng_seed).choice(tmp, int(len(tmp)/2))   
    for idx, (start, end) in enumerate(zip(reach_starts_to_plot, reach_ends_to_plot)): 
        # events = neo.Event([start, end] * s, labels=['Reach Onset', 'Reach End'])
        # add_event(ax, event=events)
        if idx in reachset1:
            color = color1
        else:
            color = color2
        ax.axvspan(start, end, color=color, alpha=alpha)
            
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
        
    sns.despine(ax=ax, top=True, bottom=True, left=True, right=True)
        
    fig.savefig(os.path.join(plot_storage, fig_label), dpi=300)
        
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

def plot_fig1_trajectory_sampling(units_res, reaches, reachNum = 3, pos_color = 'black', lead_color = 'blue', lag_color = 'red'):
    
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
    
    reachNum= 3
    reach  = reaches.loc[reachNum, :]
    sample_info = results_dict[lead_lag_key]
    
    # (4.95, 4.5)
    

    fig4 = plt.figure(figsize = plot_params.traj_pos_sample_figsize, dpi=plot_params.dpi)
    ax4 = plt.axes(projection='3d')
    fig2, ax2 = plt.subplots(figsize = plot_params.traj_vel_sample_figsize, dpi=plot_params.dpi)
    fig3, ax3 = plt.subplots(figsize = plot_params.traj_vel_sample_figsize, dpi=plot_params.dpi)
    
    # get event data using container and ndx_pose names from segment_info table following form below:
    # nwb.processing['goal_directed_kinematics'].data_interfaces['moths_s_1_e_004_position']
    event_data      = kin_module.data_interfaces[reach.video_event] 
    
    wrist_kinematics    = event_data.pose_estimation_series[   wrist_label].data[reach.start_idx:reach.stop_idx+1].T
    shoulder_kinematics = event_data.pose_estimation_series[shoulder_label].data[reach.start_idx:reach.stop_idx+1].T
    timestamps          = event_data.pose_estimation_series[   wrist_label].timestamps[reach.start_idx:reach.stop_idx+1]

    wrist_vel, _ = compute_derivatives(marker_pos=wrist_kinematics, marker_vel=None, smooth = True)

    tmp_timestamps = timestamps - timestamps[0]
    sample_t0 = np.where(np.isclose(tmp_timestamps, 0.1))[0][0]
    subsample_idxs  = np.linspace(0, 60, 16, dtype=int) 
    subsample_times = tmp_timestamps[subsample_idxs] - .1 
    lead_samples = np.hstack((subsample_idxs[subsample_idxs <= 15], [15]))
    lag_samples  = np.hstack(([15], subsample_idxs[subsample_idxs > 16]))
    short_lag_samples = subsample_idxs[[8, 9]]

    
    for sampleNum in range(6):
        
        time_jump = 0.12 * sampleNum
        sample_jump = int(0.12 * plot_params.fps * sampleNum) 
        
        sample_t0 = np.where(np.isclose(tmp_timestamps, 0.1, rtol=1e-02))[0][0] + sample_jump 
        subsample_idxs  = np.linspace(0 + sample_jump, 60 + sample_jump, 16, dtype=int) 
        subsample_times = tmp_timestamps[subsample_idxs] - .1 
        lead_samples = np.hstack((subsample_idxs[subsample_idxs <= 15 + sample_jump], [15 + sample_jump]))
        lag_samples  = np.hstack(([15 + sample_jump], subsample_idxs[subsample_idxs > 16 + sample_jump]))
        short_lag_samples = subsample_idxs[[8, 9]]
        
        fig0 = plt.figure(figsize = plot_params.traj_pos_sample_figsize, dpi=plot_params.dpi)
        ax0 = plt.axes(projection='3d')
        fig1 = plt.figure(figsize = plot_params.traj_pos_sample_figsize, dpi=plot_params.dpi)
        ax1 = plt.axes(projection='3d')
        
        ax0.plot3D(wrist_kinematics[0] , wrist_kinematics[1], wrist_kinematics[2], linewidth=plot_params.traj_linewidth, color=pos_color, linestyle='-')
        ax0.plot3D(wrist_kinematics[0, lead_samples] , wrist_kinematics[1, lead_samples], wrist_kinematics[2, lead_samples], linewidth=plot_params.traj_leadlag_linewidth, color=lead_color, linestyle='-')
        ax0.plot3D(wrist_kinematics[0, lag_samples] , wrist_kinematics[1, lag_samples], wrist_kinematics[2, lag_samples], linewidth=plot_params.traj_leadlag_linewidth, color=lag_color, linestyle='-')
        ax0.plot3D(wrist_kinematics[0, sample_t0], wrist_kinematics[1, sample_t0], wrist_kinematics[2, sample_t0], 
                   marker='o', markersize=plot_params.spksamp_markersize, markeredgecolor='black', markerfacecolor='white')
    
        ax1.plot3D(wrist_kinematics[0], wrist_kinematics[1], wrist_kinematics[2], linewidth=plot_params.traj_linewidth, color=pos_color, linestyle='-')
        ax1.plot3D(wrist_kinematics[0, short_lag_samples] , wrist_kinematics[1, short_lag_samples], wrist_kinematics[2, short_lag_samples], linewidth=plot_params.traj_leadlag_linewidth, color=lag_color, linestyle='-')
        ax1.plot3D(wrist_kinematics[0, sample_t0], wrist_kinematics[1, sample_t0], wrist_kinematics[2, sample_t0], 
                   marker='o', markersize=plot_params.spksamp_markersize, markeredgecolor='black', markerfacecolor='white')
        
        for ax in [ax0, ax1]:
            ax.set_xlabel('', fontsize = plot_params.axis_fontsize)
            ax.set_ylabel('', fontsize = plot_params.axis_fontsize)
            ax.set_zlabel('', fontsize = plot_params.axis_fontsize)
            # ax.set_xlabel('x (cm)', fontsize = plot_params.axis_fontsize)
            # ax.set_ylabel('y (cm)', fontsize = plot_params.axis_fontsize)
            # ax.set_zlabel('z (cm)', fontsize = plot_params.axis_fontsize)
            ax.set_xticks([], fontsize = plot_params.axis_fontsize)
            ax.set_yticks([], fontsize = plot_params.axis_fontsize)
            ax.set_zticks([], fontsize = plot_params.axis_fontsize)
            # ax.legend(['lead', 'lag'], loc='upper right', bbox_to_anchor=(1, 1), fontsize = 14, shadow=False)
            ax.w_xaxis.line.set_color('black')
            ax.w_yaxis.line.set_color('black')
            ax.w_zaxis.line.set_color('black')
            ax.view_init(28, 148)
            
        fig0.savefig(os.path.join(plot_storage, f'{marmcode}_trajectory_sampling_pos_{sampleNum}.png'), bbox_inches='tight', dpi=plot_params.dpi)
        fig1.savefig(os.path.join(plot_storage, f'{marmcode}_velocity_sampling_pos_{sampleNum}.png'), bbox_inches='tight', dpi=plot_params.dpi)

        fig2, ax2 = plt.subplots(figsize = plot_params.traj_vel_sample_figsize, dpi=plot_params.dpi)
        fig3, ax3 = plt.subplots(figsize = plot_params.traj_vel_sample_figsize, dpi=plot_params.dpi)

        ax2.plot(subsample_times[subsample_idxs < 15 + sample_jump], wrist_vel[0, lead_samples[:-1]] + 40, marker='o', linestyle ='', color=lead_color, markersize = plot_params.vel_markersize)
        ax2.plot(subsample_times[subsample_idxs < 15 + sample_jump], wrist_vel[1, lead_samples[:-1]]     , marker='o', linestyle ='', color=lead_color, markersize = plot_params.vel_markersize)
        ax2.plot(subsample_times[subsample_idxs < 15 + sample_jump], wrist_vel[2, lead_samples[:-1]] - 50, marker='o', linestyle ='', color=lead_color, markersize = plot_params.vel_markersize)    
        ax2.plot(subsample_times[subsample_idxs > 15 + sample_jump], wrist_vel[0, lag_samples] + 40, marker='o', linestyle ='', color=lag_color, markersize = plot_params.vel_markersize)
        ax2.plot(subsample_times[subsample_idxs > 15 + sample_jump], wrist_vel[1, lag_samples]     , marker='o', linestyle ='', color=lag_color, markersize = plot_params.vel_markersize)
        ax2.plot(subsample_times[subsample_idxs > 15 + sample_jump], wrist_vel[2, lag_samples] - 50, marker='o', linestyle ='', color=lag_color, markersize = plot_params.vel_markersize)
        
        ax2.vlines(0 + time_jump, wrist_vel[2].min() - 50, wrist_vel[0].max() + 40, linestyles='--', color = 'black')
        # ax2.vlines(.1 + time_jump, wrist_vel[2].min() - 40, wrist_vel[0].max() + 40, linestyles='--', color = 'black')
        
        ax3.plot(subsample_times[[8, 9]], wrist_vel[0, short_lag_samples] + 40, marker='o', linestyle ='', color=lag_color, markersize = plot_params.vel_markersize)
        ax3.plot(subsample_times[[8, 9]], wrist_vel[1, short_lag_samples]     , marker='o', linestyle ='', color=lag_color, markersize = plot_params.vel_markersize)
        ax3.plot(subsample_times[[8, 9]], wrist_vel[2, short_lag_samples] - 50, marker='o', linestyle ='', color=lag_color, markersize = plot_params.vel_markersize)
        
        ax3.vlines(0 + time_jump, wrist_vel[2].min() - 50, wrist_vel[0].max() + 40, linestyles='--', color = 'black')
        
        for ax in [ax2, ax3]:
            ax.set_xlabel('Time (ms)', fontsize = plot_params.axis_fontsize)
            ax.set_ylabel('Velocity (cm/s)', fontsize = plot_params.axis_fontsize)
            ax.set_xlim(-.1 + time_jump, 0.3 + time_jump)
            ax.set_ylim(-80, 70)
            # ax.set_xticks([params.lead[0], params.lead[0] + .1, params.lead[0] + .15])
            ax.set_yticks([0, 25])
            ax.set_yticklabels(['', ''], fontsize = plot_params.vel_sample_tickfontsize)
            sns.despine(ax=ax)
            ax.spines['bottom'].set_linewidth(plot_params.axis_linewidth)
            ax.spines['left'  ].set_linewidth(plot_params.axis_linewidth)

        ax2.set_xticks([-.1 + time_jump, 0 + time_jump, .1 + time_jump, .3 + time_jump])
        ax2.set_xticklabels([-100, 0, 100, 300], fontsize = plot_params.vel_sample_tickfontsize, rotation=45, ha="right", rotation_mode="anchor")
        ax3.set_xticks(     [-0.1 + time_jump, 0 + time_jump,  .1 + time_jump, .15 + time_jump, .3 + time_jump])
        ax3.set_xticklabels([-100, 0, 100, 150, 300], fontsize = plot_params.vel_sample_tickfontsize, rotation=45, ha="right", rotation_mode="anchor")
        
        fig2.savefig(os.path.join(plot_storage, f'{marmcode}_trajectory_sampling_vel_{sampleNum}.png'), bbox_inches='tight', dpi=plot_params.dpi)
        fig3.savefig(os.path.join(plot_storage, f'{marmcode}_velocity_sampling_vel_{sampleNum}.png'), bbox_inches='tight', dpi=plot_params.dpi)
  
        

    mod_t0, mod_t1 = np.where(np.isclose(tmp_timestamps, 1, rtol=1e-03))[0][0], np.where(np.isclose(tmp_timestamps, 2.5, rtol=1e-03))[0][0]
    # mod_line_t0, mod_line_t1 = mod_t0 + timestamps[0], mod_t1 + timestamps[0]
    ax4.plot3D(wrist_kinematics[0], wrist_kinematics[1], wrist_kinematics[2], linewidth=plot_params.traj_linewidth, color=pos_color, linestyle='-')
    ax4.plot3D(wrist_kinematics[0, mod_t0], wrist_kinematics[1, mod_t0], wrist_kinematics[2, mod_t0], 
               marker='o', markersize=plot_params.spksamp_markersize, markeredgecolor='black', markerfacecolor='white')
    ax4.plot3D(wrist_kinematics[0, mod_t1], wrist_kinematics[1, mod_t1], wrist_kinematics[2, mod_t1], 
               marker='o', markersize=plot_params.spksamp_markersize, markeredgecolor='black', markerfacecolor='white')
    
    # ax2.plot(subsample_times[subsample_idxs < 15], wrist_vel[0, lead_samples[:-1]] + 40, marker='o', linestyle ='', color=lead_color, markersize = plot_params.vel_markersize)
    # ax2.plot(subsample_times[subsample_idxs < 15], wrist_vel[1, lead_samples[:-1]]     , marker='o', linestyle ='', color=lead_color, markersize = plot_params.vel_markersize)
    # ax2.plot(subsample_times[subsample_idxs < 15], wrist_vel[2, lead_samples[:-1]] - 50, marker='o', linestyle ='', color=lead_color, markersize = plot_params.vel_markersize)    
    # ax2.plot(subsample_times[subsample_idxs > 15], wrist_vel[0, lag_samples] + 40, marker='o', linestyle ='', color=lag_color, markersize = plot_params.vel_markersize)
    # ax2.plot(subsample_times[subsample_idxs > 15], wrist_vel[1, lag_samples]     , marker='o', linestyle ='', color=lag_color, markersize = plot_params.vel_markersize)
    # ax2.plot(subsample_times[subsample_idxs > 15], wrist_vel[2, lag_samples] - 50, marker='o', linestyle ='', color=lag_color, markersize = plot_params.vel_markersize)
    
    # ax2.vlines(0 , wrist_vel[2].min() - 50, wrist_vel[0].max() + 40, linestyles='--', color = 'black')
    # ax2.vlines(.1, wrist_vel[2].min() - 50, wrist_vel[0].max() + 40, linestyles='--', color = 'black')
    
    # ax3.plot(subsample_times[[8, 9]], wrist_vel[0, short_lag_samples] + 40, marker='o', linestyle ='', color=lag_color, markersize = plot_params.vel_markersize)
    # ax3.plot(subsample_times[[8, 9]], wrist_vel[1, short_lag_samples]     , marker='o', linestyle ='', color=lag_color, markersize = plot_params.vel_markersize)
    # ax3.plot(subsample_times[[8, 9]], wrist_vel[2, short_lag_samples] - 50, marker='o', linestyle ='', color=lag_color, markersize = plot_params.vel_markersize)
    
    # ax3.vlines(0 , wrist_vel[2].min() - 50, wrist_vel[0].max() + 40, linestyles='--', color = 'black')
    
    
    # ax2.plot(0, wrist_vel[0, sample_t0] + 40, marker='o', markersize=plot_params.vel_markersize, markeredgecolor='black', markerfacecolor='white')
    # ax2.plot(0, wrist_vel[1, sample_t0]     , marker='o', markersize=plot_params.vel_markersize, markeredgecolor='black', markerfacecolor='white')
    # ax2.plot(0, wrist_vel[2, sample_t0] - 50, marker='o', markersize=plot_params.vel_markersize, markeredgecolor='black', markerfacecolor='white')

            # ax.set_title(title, fontsize = 16, fontweight = 'bold')
        # ax.set_xlim(min_xyz[0], max_xyz[0])
        # ax.set_ylim(min_xyz[1], max_xyz[1])
        # ax.set_zlim(min_xyz[2], max_xyz[2])
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # ax.set_zticklabels([])
        # ax.set_xlabel('x', fontsize = plot_params.axis_fontsize)
        # ax.set_ylabel('y', fontsize = plot_params.axis_fontsize)
        # ax.set_zlabel('z', fontsize = plot_params.axis_fontsize)
    
    for ax in [ax4]:
        ax.set_xlabel('', fontsize = plot_params.axis_fontsize)
        ax.set_ylabel('', fontsize = plot_params.axis_fontsize)
        ax.set_zlabel('', fontsize = plot_params.axis_fontsize)
        # ax.set_xlabel('x (cm)', fontsize = plot_params.axis_fontsize)
        # ax.set_ylabel('y (cm)', fontsize = plot_params.axis_fontsize)
        # ax.set_zlabel('z (cm)', fontsize = plot_params.axis_fontsize)
        ax.set_xticks([], fontsize = plot_params.axis_fontsize)
        ax.set_yticks([], fontsize = plot_params.axis_fontsize)
        ax.set_zticks([], fontsize = plot_params.axis_fontsize)
        # ax.legend(['lead', 'lag'], loc='upper right', bbox_to_anchor=(1, 1), fontsize = 14, shadow=False)
        ax.w_xaxis.line.set_color('black')
        ax.w_yaxis.line.set_color('black')
        ax.w_zaxis.line.set_color('black')
        ax.view_init(28, 148)
    
    fig4.savefig(os.path.join(plot_storage, f'{marmcode}_pos_with_modulation.png'), bbox_inches='tight', dpi=plot_params.dpi)

    
    # for ax in [ax2, ax3]:
    #     ax.set_xlabel('Time (ms)', fontsize = plot_params.axis_fontsize)
    #     ax.set_ylabel('Velocity (cm/s)', fontsize = plot_params.axis_fontsize)
    #     ax.set_xlim(-.1, 0.3)
    #     ax.set_ylim(-70, 70)
    #     # ax.set_xticks([params.lead[0], params.lead[0] + .1, params.lead[0] + .15])
    #     ax.set_yticks([0, 25])
    #     ax.set_yticklabels(ax.get_yticks(), fontsize = plot_params.tick_fontsize)
    #     sns.despine(ax=ax)
    #     ax.spines['bottom'].set_linewidth(plot_params.axis_linewidth)
    #     ax.spines['left'  ].set_linewidth(plot_params.axis_linewidth)

    # ax2.set_xticks([-.1, 0, .1, .3])
    # ax2.set_xticklabels([-100, 0, 100, 300], fontsize = plot_params.tick_fontsize)
    # ax3.set_xticks(     [-0.1, 0,  .1, .15, .3])
    # ax3.set_xticklabels([-100, 0, 100, 150, 300], fontsize = plot_params.tick_fontsize)
        
    plt.show()
        
    # fig0.savefig(os.path.join(plot_storage, f'{marmcode}_trajectory_sampling_pos.png'), bbox_inches='tight', dpi=plot_params.dpi)
    # fig1.savefig(os.path.join(plot_storage, f'{marmcode}_velocity_sampling_pos.png'), bbox_inches='tight', dpi=plot_params.dpi)
    # fig2.savefig(os.path.join(plot_storage, f'{marmcode}_trajectory_sampling_vel.png'), bbox_inches='tight', dpi=plot_params.dpi)
    # fig3.savefig(os.path.join(plot_storage, f'{marmcode}_velocity_sampling_vel.png'), bbox_inches='tight', dpi=plot_params.dpi)
    
    # return [mod_line_t0, mod_line_t1]

if __name__ == '__main__':
    # io_acq = NWBHDF5IO(nwb_acquisition_file, mode='r')
    # nwb_acq = io_acq.read()
    
    with open(pkl_infile, 'rb') as f:
        results_dict = dill.load(f)
    units_res = results_dict[lead_lag_key]['all_models_summary_results']
    
    with NWBHDF5IO(nwb_analysis_file, 'r') as io:
        nwb = io.read()

        FN = nwb.scratch['split_reach_FNs'].data[:] 
        units_res = add_in_weight_to_units_df(units_res, FN.copy())

        reaches_key = [key for key in nwb.intervals.keys() if 'reaching_segments' in key][0]
        
        units, reaches, kin_module = get_sorted_units_and_apparatus_kinematics_with_metadata(nwb, reaches_key, plot=False)
    
        plot_fig1_trajectory_sampling(units_res, reaches, reachNum = 3, pos_color = 'black', lead_color = 'blue', lag_color = 'red')    
    
        generate_event_raster(units, units_res, reaches, preTime=plot_params.raster_pretime, postTime=plot_params.raster_posttime, 
                              figsize=plot_params.raster_figsize, marker='.', msize = plot_params.raster_ticksize, 
                              reaches_to_plot= reaches_to_plot, sort_by = 'W_in', ascend = False, 
                              units_to_plot=175, color1=color1, color2=color2, alpha=.5, rng_seed=10)
        

    # reaches_to_plot = [29, 30, 31] [10, 11, 12] [76, 77, 78, 79, 80, 81]
    #sort_by = 'traj_avgPos_auc', ascend = False
    
