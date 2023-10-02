#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 07:39:59 2023

@author: daltonm
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 31 10:09:38 2022

@author: Dalton
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pickle
import dill
import os
import glob
import math
import re
import seaborn as sns
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc
import statsmodels.api as sm
from scipy.stats import binomtest, ttest_rel, ttest_ind, mannwhitneyu, f_oneway, pearsonr
from scipy.integrate import cumtrapz
from scipy.integrate import simps
from mpl_toolkits import mplot3d 
from scipy.ndimage import median_filter, gaussian_filter
from importlib import sys, reload
from scipy.spatial.transform import Rotation as R
from pynwb import NWBHDF5IO
import ndx_pose
from matplotlib.animation import FuncAnimation, FFMpegWriter
import ffmpeg

sys.path.insert(0, '/project/nicho/projects/marmosets/code_database/data_processing/nwb_tools/hatlab_nwb_tools/')
from hatlab_nwb_functions import remove_duplicate_spikes_from_good_single_units, get_sorted_units_and_apparatus_kinematics_with_metadata   

sys.path.insert(0, '/project/nicho/projects/marmosets/code_database/analysis/trajectory_encoding_model/')
from utils import get_interelectrode_distances_by_unit, choose_units_for_model


marmcode='TY'
other_marm = 'MG'
FN_computed = True

fig_mode='pres'

# nwb_infile = '/project/nicho/projects/dalton/data/TY20210211_freeAndMoths-003_DM.nwb' 
# pkl_infile = '/project/nicho/projects/dalton/data/TY20210211_freeAndMoths-003_DM_encoding_model_regularized_results_30ms_shift_v4.pkl' #'/project/nicho/projects/dalton/data/TY20210211_freeAndMoths-003_DM_encoding_model_30ms_shift_results_v2.pkl'

if marmcode=='TY':
    if FN_computed:
        nwb_infile = '/project/nicho/projects/dalton/data/TY/TY20210211_freeAndMoths-003_resorted_20230612_DM_with_functional_networks.nwb' 
    else:
        nwb_infile = '/project/nicho/projects/dalton/data/TY/TY20210211_freeAndMoths-003_resorted_20230612_DM.nwb' 
    pkl_infile = '/project/nicho/projects/dalton/data/TY/TY20210211_freeAndMoths-003_resorted_20230612_DM_alpha_pt00001_encoding_models_30ms_shift_v2.pkl' 
    # pkl_infile = '/project/nicho/projects/dalton/data/TY/TY20210211_freeAndMoths-003_resorted_20230612_DM_FINAL_trajectory_shuffled_encoding_models_30ms_shift_v2.pkl' #'/project/nicho/projects/dalton/data/TY20210211_freeAndMoths-003_resorted_20230612_DM_encoding_model_sorting_corrected_30ms_shift_v4.pkl'
    
    # pkl_infile = '/project/nicho/projects/dalton/data/TY/TY20210211_freeAndMoths-003_resorted_20230612_DM_trajectory_shuffled_tortuosity_split_encoding_models_30ms_shift_v2.pkl' #'/project/nicho/projects/dalton/data/TY20210211_freeAndMoths-003_resorted_20230612_DM_encoding_model_sorting_corrected_30ms_shift_v4.pkl'
    bad_units_list = None
    mua_to_fix = []
    units_to_plot = [1, 2, 3]
    
    unit_axlims = (np.array([-0.009687  , -0.00955038, -0.01675681]),
                   np.array([0.02150172 , 0.02333975 , 0.01376333]))
    
    # reaches_to_plot=[[76, 78, 79], [77, 80, 81]]
    # reaches_to_plot=[[76], [77]]

    # reaches_to_plot=[[3], []]
    # vid_name_mod = 'event008' 

    reaches_to_plot = [[23, 24, 25, 26], []]
    vid_name_mod = 'event049' 
    
    ani_interval_ms = 1/150 * 1e3

elif marmcode=='MG':
    if FN_computed:
        nwb_infile   = '/project/nicho/projects/dalton/data/MG/MG20230416_1505_mothsAndFree-002_processed_DM_with_functional_networks.nwb'
        # nwb_infile   = '/project/nicho/projects/dalton/data/MG/MG20230416_1505_mothsAndFree-002_processed_DM_with_functional_networks_noBadUnitsList.nwb'
    else:
        nwb_infile   = '/project/nicho/projects/dalton/data/MG/MG20230416_1505_mothsAndFree-002_processed_DM.nwb'
    pkl_infile = '/project/nicho/projects/dalton/data/MG/MG20230416_1505_mothsAndFree-002_processed_DM_alpha_pt00001_removedUnits_181_440_fixedMUA_745_796_encoding_models_30ms_shift_v2.pkl'
    # pkl_infile = '/project/nicho/projects/dalton/data/MG/MG20230416_1505_mothsAndFree-002_processed_DM_dlcIter5_noBadUnitsList_trajectory_shuffled_encoding_models_30ms_shift_v2.pkl'
    # pkl_infile = '/project/nicho/projects/dalton/data/MG/MG20230416_1505_mothsAndFree-002_processed_DM_dlcIter5_resortedUnits_trajectory_shuffled_encoding_models_30ms_shift_v2.pkl'

    bad_units_list = [181, 440] # []
    mua_to_fix = [745, 796]
    units_to_plot = [0, 2, 3]
    
    unit_axlims = (np.array([-0.009687  , -0.00955038, -0.01675681]),
                   np.array([0.02150172 , 0.02333975 , 0.01376333]))

    reaches_to_plot=[[3, 4, 5], [6, 7, 11]]

    ani_interval_ms = 1/200 * 1e3


    # reaches_to_plot=[[42, 44, 45], [43, 46, 47]]

split_pattern = '_shift_v' # '_results_v'
base, ext = os.path.splitext(pkl_infile)
base, in_version = base.split(split_pattern)
out_version = str(int(in_version) + 1)  
pkl_outfile = base + split_pattern + out_version + ext

dataset_code = os.path.basename(pkl_infile)[:10] 
# plots = os.path.join(os.path.dirname(os.path.dirname(pkl_infile)), 'plots', dataset_code)
if fig_mode == 'paper':
    plots = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(pkl_infile))), 'plots', dataset_code)
elif fig_mode == 'pres':
    plots = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(pkl_infile))), 'defense_plots', dataset_code)

shift_set = int(pkl_infile.split('ms_shift')[0][-2:])
  
# color1     = (  0/255, 141/255, 208/255)
# color2     = (159/255, 206/255, 239/255)
  
color1     = (0, 0, 0)
color2     = (0, 0, 0)
spontColor = (183/255, 219/255, 165/255)

class params:
    lead = 'all' #[0.5, 0.4, 0.4, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0  , 0  , 0  ] #[0.2, 0.15, 0.1, 0.05, 0] #[0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0] #[0.05] # 100ms lead time
    lag  = 'all' #[0  , 0  , 0.1, 0  , 0.1, 0.2, 0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.3, 0.4, 0.5] #[0.2, 0.25, 0.3, 0.35, 0.4] #[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4] #[0.35] # 300ms lag time
    
    if marmcode=='TY':
        # reorder = [0, 1, 3, 2, 4, 5, 6, 8, 13, 12, 14, 15, 16, 17, 18, 11,  7,  9, 10]
        # reorder = [0, 1, 3, 2, 4, 5, 6, 12, 13, 16, 8,  14, 15, 11,  7,  9, 10]
        best_lead_lag_key = 'lead_100_lag_300' #None

        cortical_boundaries = {'x_coord'      : [   0,          400,  800, 1200,         1600, 2000, 2400, 2800, 3200, 3600],
                               'y_bound'      : [None,         1600, None, None,         1200, None, None, None, None, None],
                               'areas'        : ['3b', ['3a', '3b'], '3a', '3a', ['M1', '3a'], 'M1', 'M1', 'M1', 'M1', 'M1'],
                               'unique_areas' : ['3b', '3a', 'M1']}
    elif marmcode=='MG':
        # reorder = [0]
        # reorder = [0, 1, 3, 2, 4, 5, 6, 12, 13, 16, 8,  14, 15, 11,  7,  9, 10]
        best_lead_lag_key = 'lead_100_lag_300' #None

        cortical_boundaries = {'x_coord'      : [    0,   400,  800, 1200, 1600, 2000, 2400, 2800, 3200, 3600],
                               'y_bound'      : [ None,  None, None, None, None, None, None, None, None, None],
                               'areas'        : ['6dc', '6dc', 'M1', 'M1', 'M1', 'M1', 'M1', '3a', '3a', '3a'],
                               'unique_areas' : ['6dc', 'M1', '3a']}
        
    mua_to_fix=mua_to_fix
    # reorder = [0, 1, 3, 2, 4, 5, 6, 7, 9, 8, 10, 11, 12, 13, 14, 15]
    # reorder = [0, 1, 3, 2, 4, 5, 6, 7, 8 , 9 , 10 ]

    FN_key = 'split_reach_FNs'
    frate_thresh = 2
    snr_thresh = 3
    significant_proportion_thresh = 0.95
    nUnits_percentile = 60
    primary_traj_model = 'traj_avgPos'
    shuffle_to_test = 'shuffled_traj'

    apparatus_dimensions = [14, 12.5, 7]#[14, 12.5, 13]

    # cortical_boundaries = {'x_coord'      : [   0,          400,  800, 1200,         1600, 2000, 2400, 2800,          3200,          3600],
    #                         'y_bound'      : [None,         1200, None, None,         1200, None, None, None,           800,          2000],
    #                         'areas'        : ['3b', ['3a', '3b'], '3a', '3a', ['M1', '3a'], 'M1', 'M1', 'M1', ['6Dc', 'M1'], ['6Dc', 'M1']],
    #                         'unique_areas' : ['3b', '3a', 'M1', '6Dc']}
class plot_params:
    # axis_fontsize = 24
    # dpi = 300
    # axis_linewidth = 2
    # tick_length = 2
    # tick_width = 1
    # map_figSize = (6, 8)
    # tick_fontsize = 18
    # aucScatter_figSize = (7, 7)
    
    figures_list = ['Fig1', 'Fig2', 'Fig3', 'Fig4', 'Fig5', 'Fig6', 'Fig7', 'FigS1',  'FigS2',  'FigS3',  'FigS4', 'FigS5', 'unknown']

    if fig_mode == 'paper':
        axis_fontsize = 8
        dpi = 300
        axis_linewidth = 1
        tick_length = 1.75
        tick_width = 0.5
        tick_fontsize = 8
        
        spksamp_markersize = 4
        vel_markersize = 2
        traj_length_markersize = 6
        scatter_markersize = 8
        stripplot_markersize = 2
        feature_corr_markersize = 8
        wji_vs_trajcorr_markersize = 2
        reach_sample_markersize = 4
        
        corr_marker_color = 'gray'
        
        traj_pos_sample_figsize = (1.75, 1.75)
        traj_vel_sample_figsize = (1.5  ,   1.5)
        traj_linewidth = 1
        traj_leadlag_linewidth = 2
        reach_sample_linewidth = 1
        
        preferred_traj_linewidth = .5
        distplot_linewidth = 1
        preferred_traj_figsize = (1.75, 1.75)
        
        weights_by_distance_figsize = (2.5, 1.5)
        aucScatter_figSize = (1.75, 1.75)
        FN_figsize = (3, 3)
        feature_corr_figSize = (1.75, 1.75)
        trajlength_figsize = (1.75, 1.75)
        pearsonr_histsize = (1.5, 1.5)
        distplot_figsize = (1.5, 1)
        shuffle_figsize = (8, 3)
        stripplot_figsize = (5, 2)
        scatter_figsize = (1.75, 1.75)
        reach_sample_figsize = (2.5, 2.25)

    elif fig_mode == 'pres':
        axis_fontsize = 14
        dpi = 300
        axis_linewidth = 2
        tick_length = 2
        tick_width = 1
        tick_fontsize = 12
    
        map_figSize = (6, 8)
        FN_figsize = (5, 5)
        weights_by_distance_figsize = (6, 4)
        aucScatter_figSize = (6, 6)
        feature_corr_figSize = (4, 4)   
        
        reach_sample_figsize = (3.5, 3.25)
        reach_sample_linewidth = 3
        reach_sample_markersize = 10

# class plot_params:
#     # axis_fontsize = 24
#     # dpi = 300
#     # axis_linewidth = 2
#     # tick_length = 2
#     # tick_width = 1
#     # map_figSize = (6, 8)
#     # tick_fontsize = 18
#     # aucScatter_figSize = (7, 7)
    
#     figures_list = ['Fig1', 'Fig2', 'Fig3', 'Fig4', 'Fig5', 'Fig6', 'Fig7', 'FigS1',  'FigS2',  'FigS3',  'FigS4', 'FigS5', 'unknown']

#     if fig_mode == 'paper':
#         axis_fontsize = 8
#         dpi = 300
#         axis_linewidth = 2
#         tick_length = 1.5
#         tick_width = 1
#         tick_fontsize = 8
        
#         spksamp_markersize = 4
#         vel_markersize = 2
#         traj_length_markersize = 6
#         scatter_markersize = 8
        
#         traj_pos_sample_figsize = (1.75, 1.75)
#         traj_vel_sample_figsize = (1.5  ,   1.5)
#         traj_linewidth = 1
#         traj_leadlag_linewidth = 2
        
#         preferred_traj_linewidth = .5
#         preferred_traj_figsize = (1.75, 1.75)
        
#         weights_by_distance_figsize = (2.5, 1.5)
#         aucScatter_figSize = (1.75, 1.75)
#         FN_figsize = (3, 3)
#         feature_corr_figSize = (3, 3)
#         trajlength_figsize = (1.75, 1.75)
#         pearsonr_histsize = (1.5, 1.5)


#     elif fig_mode == 'pres':
#         axis_fontsize = 20
#         dpi = 300
#         axis_linewidth = 2
#         tick_length = 2
#         tick_width = 1
#         tick_fontsize = 18
    
#         map_figSize = (6, 8)
#         FN_figsize = (5, 5)
#         weights_by_distance_figsize = (6, 4)
#         aucScatter_figSize = (6, 6)
#         feature_corr_figSize = (4, 4)

plt.rcParams['figure.dpi'] = plot_params.dpi
plt.rcParams['savefig.dpi'] = plot_params.dpi
plt.rcParams["font.family"] = "Arial"
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = 'Arial'
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

# plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path
    

for fig_name in plot_params.figures_list:
    os.makedirs(os.path.join(plots, fig_name), exist_ok=True)
    if fig_name == 'unknown':
        os.makedirs(os.path.join(plots, fig_name, 'network'), exist_ok=True)
        os.makedirs(os.path.join(plots, fig_name, 'kinematics'), exist_ok=True)
        os.makedirs(os.path.join(plots, fig_name, 'auc_comparison'), exist_ok=True)


# def get_reach_samples(nReaches = 3, reachset1 = None, reachset2 = None, color1 = 'blue', color2='green'):
    
#     # linestyles= ['solid', 'solid', 'solid']
    
#     combined_reaches = sorted(reachset1 + reachset2)
    
#     first_event_key = [key for idx, key in enumerate(kin_module.data_interfaces.keys()) if idx == 0][0]
#     camPeriod = np.mean(np.diff(kin_module.data_interfaces[first_event_key].pose_estimation_series['origin'].timestamps[:]))
#     dlc_scorer = kin_module.data_interfaces[first_event_key].scorer 
    
#     if 'simple_joints_model' in dlc_scorer:
#         wrist_label = 'hand'
#         shoulder_label = 'shoulder'
#     elif 'marmoset_model' in dlc_scorer and nwb.subject.subject_id == 'TY':
#         wrist_label = 'l-wrist'
#         shoulder_label = 'l-shoulder'
#     elif 'marmoset_model' in dlc_scorer and nwb.subject.subject_id == 'MG':
#         wrist_label = 'r-wrist'
#         shoulder_label = 'r-shoulder'
    
#     if reachset1 is None:
#         reachset1 = reach_set_df.loc[reach_set_df['FN_reach_set']==1, 'reach_num'].to_list()
#     if reachset2 is None:
#         reachset2 = reach_set_df.loc[reach_set_df['FN_reach_set']==2, 'reach_num'].to_list()

#     reach_list = []
#     color_list = []
#     for rIdx, reach in reaches.iterrows():
        
#         # get event data using container and ndx_pose names from segment_info table following form below:
#         # nwb.processing['goal_directed_kinematics'].data_interfaces['moths_s_1_e_004_position']
#         event_data      = kin_module.data_interfaces[reach.video_event] 
        
#         wrist_kinematics    = event_data.pose_estimation_series[   wrist_label].data[reach.start_idx:reach.stop_idx+1]
#         shoulder_kinematics = event_data.pose_estimation_series[shoulder_label].data[reach.start_idx:reach.stop_idx+1]
#         timestamps          = event_data.pose_estimation_series[   wrist_label].timestamps[reach.start_idx:reach.stop_idx+1]
            
#         if rIdx in reachset1:
#             reach_list.append(wrist_kinematics)
#             color_list.append(color1)
#         elif rIdx in reachset2:
#             color_list.append(color2)
#             reach_list.append(wrist_kinematics)
#         else:
#             continue
        
#     return reach_list, color_list
        
#     # fig0.savefig(os.path.join(plots, 'Fig1', f'{marmcode}_reach_set1_reaches.png'), bbox_inches='tight', dpi=plot_params.dpi)
#     # fig1.savefig(os.path.join(plots, 'Fig1', f'{marmcode}_reach_set2_reaches.png'), bbox_inches='tight', dpi=plot_params.dpi)

def get_reach_samples(nReaches = 3, reachset1 = None, reachset2 = None, color1 = 'blue', color2='green'):
    
    # linestyles= ['solid', 'solid', 'solid']
    combined_reaches = sorted(reachset1 + reachset2) 
    
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
    
    if reachset1 is None:
        reachset1 = reach_set_df.loc[reach_set_df['FN_reach_set']==1, 'reach_num'].to_list()
    if reachset2 is None:
        reachset2 = reach_set_df.loc[reach_set_df['FN_reach_set']==2, 'reach_num'].to_list()

    color_list = []
    kinIdx_list = []
    kIdx = 0
    last_time=0
    for rIdx, reach in reaches.iterrows():
        
        # get event data using container and ndx_pose names from segment_info table following form below:
        # nwb.processing['goal_directed_kinematics'].data_interfaces['moths_s_1_e_004_position']
        event_data      = kin_module.data_interfaces[reach.video_event] 
        
        wrist_kinematics    = event_data.pose_estimation_series[   wrist_label].data[reach.start_idx:reach.stop_idx+1]
        shoulder_kinematics = event_data.pose_estimation_series[shoulder_label].data[reach.start_idx:reach.stop_idx+1]
        timestamps          = event_data.pose_estimation_series[   wrist_label].timestamps[reach.start_idx:reach.stop_idx+1]
        
        if rIdx == combined_reaches[0]:
            period_in_sec = np.mean(np.diff(timestamps))
            print(period_in_sec)
            pre_frames = int(.5 / period_in_sec)
            reach_kin = np.full((pre_frames, 3), np.nan)
            color_list.extend([color1 for i in range(pre_frames)])
            kinIdx_list.extend([kIdx for i in range(pre_frames)])
        
        elif rIdx > combined_reaches[0] and rIdx in combined_reaches:
            pre_frames = int((timestamps[0] - last_time) / period_in_sec)
            reach_kin = np.vstack((reach_kin, np.full((pre_frames, 3), np.nan)))
            color_list.extend([color1 for i in range(pre_frames)])        
            kinIdx_list.extend([kIdx for i in range(pre_frames)])

        if rIdx in reachset1:
            reach_kin = np.vstack((reach_kin, wrist_kinematics))
            # reach_list.append(wrist_kinematics)
            color_list.extend([color1 for i in range(wrist_kinematics.shape[0])])
            kinIdx_list.extend([kIdx for i in range(wrist_kinematics.shape[0])])

            last_time = timestamps[-1]
            kIdx+=1
        elif rIdx in reachset2:
            reach_kin = np.vstack((reach_kin, wrist_kinematics))
            # color_list.append(color2)
            color_list.extend([color2 for i in range(wrist_kinematics.shape[0])])
            kinIdx_list.extend([kIdx for i in range(wrist_kinematics.shape[0])])
            last_time = timestamps[-1]
            kIdx+=1
            # reach_list.append(wrist_kinematics)
        else:
            continue
        
        if rIdx > combined_reaches[-1]:
            post_frames = int(7 / period_in_sec)
            reach_kin = np.full((post_frames, 3), np.nan)
            color_list.extend([color1 for i in range(post_frames)])
            kinIdx_list.extend([kIdx for i in range(post_frames)])
            
        
    return reach_kin, color_list, period_in_sec, kinIdx_list
        
    # fig0.savefig(os.path.join(plots, 'Fig1', f'{marmcode}_reach_set1_reaches.png'), bbox_inches='tight', dpi=plot_params.dpi)
    # fig1.savefig(os.path.join(plots, 'Fig1', f'{marmcode}_reach_set2_reaches.png'), bbox_inches='tight', dpi=plot_params.dpi)

def animate_data(i, reach_kin:np.ndarray, color_list:list, kinIdx_list:list):

    kIdx = kinIdx_list[i]    

    x[kIdx].append(reach_kin[i, 0])
    y[kIdx].append(reach_kin[i, 1])
    z[kIdx].append(reach_kin[i, 2])
    
    colors = color_list[i]
    
    ax.clear()
    
    ax.plot3D(x[kIdx], y[kIdx], z[kIdx], linewidth=plot_params.reach_sample_linewidth, color=colors, linestyle='solid')
    ax.plot3D(x[kIdx][-1], y[kIdx][-1], z[kIdx][-1], marker='o', color=colors, markersize=plot_params.reach_sample_markersize)
    
    ax.set_xlabel('x (cm)', fontsize = plot_params.axis_fontsize)
    ax.set_ylabel('y (cm)', fontsize = plot_params.axis_fontsize)
    ax.set_zlabel('z (cm)', fontsize = plot_params.axis_fontsize)
    # ax.legend(['lead', 'lag'], loc='upper right', bbox_to_anchor=(1, 1), fontsize = 14, shadow=False)
    # ax.w_xaxis.line.set_color('black')
    # ax.w_yaxis.line.set_color('black')
    # ax.w_zaxis.line.set_color('black')
    ax.view_init(28, 148)
    ax.set_xlim(0, params.apparatus_dimensions[0]),
    ax.set_ylim(0, params.apparatus_dimensions[1])
    ax.set_zlim(0, params.apparatus_dimensions[2])
    ax.set_xticks([0, params.apparatus_dimensions[0]]),
    ax.set_yticks([0, params.apparatus_dimensions[1]])
    ax.set_zticks([0, params.apparatus_dimensions[2]])

    
        
if __name__ == "__main__":
    
    with open(pkl_infile, 'rb') as f:
        results_dict = dill.load(f)
        
    with NWBHDF5IO(nwb_infile, 'r') as io:
        nwb = io.read()

        reaches_key = [key for key in nwb.intervals.keys() if 'reaching_segments' in key][0]
        
        units, reaches, kin_module = get_sorted_units_and_apparatus_kinematics_with_metadata(nwb, reaches_key, mua_to_fix=params.mua_to_fix, plot=False) 
        
        units = choose_units_for_model(units, quality_key = 'snr', quality_thresh = params.snr_thresh, frate_thresh = params.frate_thresh, bad_units_list=bad_units_list)
        # units = choose_units_for_model(units, quality_key='amp', quality_thresh=5, frate_thresh=params.frate_thresh)
        
        if FN_computed:
            spontaneous_FN = nwb.scratch['spontaneous_FN'].data[:]
            reach_FN = nwb.scratch[params.FN_key].data[:] 
            reach_set_df = nwb.scratch['split_FNs_reach_sets'].to_dataframe()
            
            reach_kin, color_list, period_in_sec, kinIdx_list = get_reach_samples(nReaches = None, reachset1 = reaches_to_plot[0], reachset2 = reaches_to_plot[1], 
                                                                                  color1 = color1, color2=color2)
            
            
    fig = plt.figure(figsize = plot_params.reach_sample_figsize)
    ax = plt.axes(projection='3d')
    
    # for reach_kin, col in zip(reach_list, color_list):
    
    x=[[] for i in range(len(reaches_to_plot[0]+reaches_to_plot[1]))]
    y=[[] for i in range(len(reaches_to_plot[0]+reaches_to_plot[1]))]
    z=[[] for i in range(len(reaches_to_plot[0]+reaches_to_plot[1]))]
    colors = []
    # ani = FuncAnimation(fig, animate_data, frames=None, interval=ani_interval_ms, fargs = (reach_kin, col), repeat=False)
    ani = FuncAnimation(fig, animate_data, frames=reach_kin.shape[0], interval=period_in_sec*1e3, fargs = (reach_kin, color_list, kinIdx_list), repeat=False)
    
    f = os.path.join(plots, 'Fig1', f'{marmcode}_data_video_with_marker_{vid_name_mod}.mp4') 
    writervideo = FFMpegWriter(fps=round(1/period_in_sec)) 
    ani.save(f, writer=writervideo)
    
    # plt.show()


        
