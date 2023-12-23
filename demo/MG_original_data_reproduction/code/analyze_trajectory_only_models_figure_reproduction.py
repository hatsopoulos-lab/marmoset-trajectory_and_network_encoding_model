# -*- coding: utf-8 -*-
"""
Created on Tue May 31 10:09:38 2022

@author: Dalton
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import colormaps
import dill
import os
import re
import seaborn as sns
from scipy.stats import binomtest, ttest_rel, ttest_ind, mannwhitneyu, f_oneway, pearsonr, median_test
from scipy.integrate import cumtrapz
from scipy.integrate import simps
from mpl_toolkits import mplot3d 
from importlib import sys
from pynwb import NWBHDF5IO
import ndx_pose
from pathlib import Path

script_directory = Path(os.path.dirname(os.path.abspath(sys.argv[0])))
code_path = script_directory.parent.parent.parent / 'clean_final_analysis/'
data_path = script_directory.parent.parent / 'data' / 'original'

sys.path.insert(0, str(code_path))
from hatlab_nwb_functions import get_sorted_units_and_apparatus_kinematics_with_metadata   
from utils import choose_units_for_model, get_interelectrode_distances_by_unit, load_dict_from_hdf5, save_dict_to_hdf5

marmcode='MG'
demo = True
other_marm = 'TY' #None #'MG' #None
FN_computed = True
fig_mode='paper'
save_kinModels_pkl = False

pkl_in_tag  = 'kinematic_models_created'
pkl_out_tag = 'kinematic_models_summarized' 

if marmcode=='TY':
    if FN_computed:
        nwb_infile   = data_path / 'TY' / 'TY20210211_freeAndMoths-003_resorted_20230612_DM_with_functional_networks.nwb' 
    else:
        nwb_infile   = data_path / 'TY' / 'TY20210211_freeAndMoths-003_resorted_20230612_DM.nwb'
    other_marm_pkl_infile = data_path / 'MG' / f'MG20230416_1505_mothsAndFree-002_processed_DM_{pkl_out_tag}.pkl'
    bad_units_list = None
    mua_to_fix = []
    units_to_plot = [1, 7, 8, 13]
    pathlet_subset = 'context_invariant'
    # all_units_plot_subset = [  0,   5,   6,   9,  10,  13,  25,  31,  32,  34,  38,  44,  45,  47,
    #         48,  54,  63,  64,  65,  67,  73,  74,  75,  76,  79,  80,  89, 101,
    #        121, 122, 128, 130, 131, 148, 153, 154, 155, 159, 163, 172] 
    all_units_plot_subset = [1,2,3,4,7,8,11,12,14,15,16,17,18,19,20,21,22,23,24,26,27,28,29,30,33,35,36,37,39,40,41,42,43,46,49,
      50,51,52,53,55,56,57,58,59,60,61,62,66,68,69,70,71,72,77,78,81,82,83,84,85,86,87,88,
      90,91,92,93,94,95,96,97,98,99,100,102,103,104,105,106,107,108,109,110,111,112,113,
      114,115,116,117,118,119,120,123,124,125,126,127,129,132,133,134,135,136,137,138,139,
      140,141,142,143,144,145,146,147,149,150,151,152,156,157,158,160,161,162,164,165,166,
      167,168,169,170,171,173,174]
    # unit_axlims = (np.array([-0.009687  , -0.00955038, -0.01675681]),
    #                np.array([0.02150172 , 0.02333975 , 0.01376333]))
    unit_axlims = (np.array([-0.00955038  , -0.00955038, -0.00955038]),
                   np.array([0.02333975 , 0.02333975 , 0.02333975]))
    reaches_to_plot=[[4, 76, 79], [3, 78, 81]]

    # reaches_to_plot=[[76, 78, 79], [77, 80, 81]]

elif marmcode=='MG':
    if FN_computed:
        nwb_infile   = data_path / 'MG' / 'MG20230416_1505_mothsAndFree-002_processed_DM_with_functional_networks.nwb'
    else:
        nwb_infile   = data_path / 'MG' / 'MG20230416_1505_mothsAndFree-002_processed_DM.nwb'
    if demo:
        other_marm_pkl_infile = data_path / 'MG' / f'MG20230416_1505_mothsAndFree-002_processed_DM_{pkl_out_tag}.pkl'
    else:
        other_marm_pkl_infile = data_path / 'TY' / f'TY20210211_freeAndMoths-003_resorted_20230612_DM_{pkl_out_tag}.pkl'
    bad_units_list = [181, 440]
    mua_to_fix = [745, 796]
    units_to_plot = [0, 3, 8, 9]
    
    pathlet_subset = 'context_invariant'
    # all_units_plot_subset = [2, 6, 16, 19, 20, 21, 45, 60, 70]
    all_units_plot_subset = [ 0,  1,  3,  4,  5,  7,  8,  9, 10, 11, 12, 13, 14, 15, 17, 18, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
            42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61,
            62, 63, 64, 65, 66, 67, 68, 69, 71, 72]
    unit_axlims = (np.array([-0.009687  , -0.00955038, -0.01675681]),
                   np.array([0.02150172 , 0.02333975 , 0.01376333]))
    # reaches_to_plot=[[3, 4, 5], [6, 7, 11]]
    reaches_to_plot=[[3, 7, 37], [6, 14, 16]]

    
pkl_infile   = nwb_infile.parent / f'{nwb_infile.stem.split("_with_functional_networks")[0]}_{pkl_in_tag}.pkl'
pkl_outfile  = nwb_infile.parent / f'{nwb_infile.stem.split("_with_functional_networks")[0]}_{pkl_out_tag}.pkl'

dataset_code = pkl_infile.stem.split('_')[0]
if fig_mode == 'paper':
    plots = script_directory.parent / 'plots' / dataset_code
elif fig_mode == 'pres':
    plots = script_directory.parent / 'presentation_plots' / dataset_code
  
color1     = (  0/255, 141/255, 208/255)
color2     = (159/255, 206/255, 239/255)
spontColor = (183/255, 219/255, 165/255)

class params:
    lead = 'all' #[0.5, 0.4, 0.4, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0  , 0  , 0  ] #[0.2, 0.15, 0.1, 0.05, 0] #[0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0] #[0.05] # 100ms lead time
    lag  = 'all' #[0  , 0  , 0.1, 0  , 0.1, 0.2, 0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.3, 0.4, 0.5] #[0.2, 0.25, 0.3, 0.35, 0.4] #[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4] #[0.35] # 300ms lag time
    
    if marmcode=='TY':
        # reorder = [0, 1, 3, 2, 4, 5, 6, 8, 13, 12, 14, 15, 16, 17, 18, 11,  7,  9, 10]
        # reorder = [0, 1, 3, 2, 4, 5, 6, 12, 13, 16, 8,  14, 15, 11,  7,  9, 10]
        best_lead_lag_key = 'lead_100_lag_300'

        cortical_boundaries = {'x_coord'      : [        0,          400,       800,      1200,                 1600,    2000,    2400,    2800,    3200,    3600],
                               'y_bound'      : [     None,         None,      None,      None,                 1200,    None,    None,    None,    None,    None],
                               'areas'        : ['Sensory',    'Sensory', 'Sensory', 'Sensory', ['Motor', 'Sensory'], 'Motor', 'Motor', 'Motor', 'Motor', 'Motor'],
                               'unique_areas' : ['Sensory', 'Motor']}
        view_angle = (28, 148)#(14, 146)
        apparatus_min_dims = [0, 0, 0]
        apparatus_max_dims = [14, 12.5, 7]
    elif marmcode=='MG':
        best_lead_lag_key = 'lead_100_lag_300'

        cortical_boundaries = {'x_coord'      : [      0,     400,     800,    1200,    1600,    2000,    2400,      2800,      3200,      3600],
                               'y_bound'      : [   None,    None,    None,    None,    None,    None,    None,      None,      None,      None],
                               'areas'        : ['Motor', 'Motor', 'Motor', 'Motor', 'Motor', 'Motor', 'Motor', 'Sensory', 'Sensory', 'Sensory'],
                               'unique_areas' : ['Motor', 'Sensory']}
        view_angle = (28, 11)
        apparatus_min_dims = [6, -3, 0]
        apparatus_max_dims = [14, 10, 5]
        
    FN_key = 'split_reach_FNs'
    frate_thresh = 2
    snr_thresh = 3
    significant_proportion_thresh = 0.95
    nUnits_percentile = 60
    primary_traj_model = 'traj_avgPos'
    shuffle_keys = ['shuffled_spikes', 'shuffled_traj']

    # apparatus_dimensions = [14, 12.5, 7]#[14, 12.5, 13]

class plot_params:
    
    figures_list = ['Fig1', 'Fig2', 'Fig3', 'Fig4', 'Fig5', 'Fig6', 'FigS1',  'FigS2',  'FigS3',  'FigS4', 'FigS5', 'FigS6', 'FigS7', 'unknown']

    if fig_mode == 'paper':
        axis_fontsize = 8
        dpi = 300
        axis_linewidth = 1
        tick_length = 1.75
        tick_width = 0.5
        tick_fontsize = 8
        
        spksamp_markersize = 4
        vel_markersize = 2
        scatter_markersize = 8
        stripplot_markersize = 2
        feature_corr_markersize = 8
        wji_vs_trajcorr_markersize = 2
        reach_sample_markersize = 4
        
        corr_marker_color = 'gray'
        
        traj_pos_sample_figsize = (2.25, 2.25)
        traj_vel_sample_figsize = (1.5 ,  1.5)
        traj_linewidth = 1
        traj_leadlag_linewidth = 2
        reach_sample_linewidth = 1
        
        preferred_traj_linewidth = .5
        distplot_linewidth = 1
        preferred_traj_figsize = (2.25, 2.25)
        
        weights_by_distance_figsize = (2.5, 1.5)
        aucScatter_figSize = (1.9, 1.9)
        FN_figsize = (3, 3)
        feature_corr_figSize = (1.75, 1.75)
        pearsonr_histsize = (1.9, 1.9)
        distplot_figsize = (1.5, 1)
        shuffle_figsize = (8, 3)
        stripplot_figsize = (5, 2)
        scatter_figsize = (1.75, 1.75)
        reach_sample_figsize = (2.5, 2.25)
        
        boxplot_figsize = (4, 1.75)
        boxplot_boxwidth = .5

        trajlength_figsize = (1.9, 1.9)
        trajlength_markersize = 6
        trajlength_linewidth = 1

    elif fig_mode == 'pres':
        axis_fontsize = 20
        dpi = 300
        axis_linewidth = 2
        tick_length = 2
        tick_width = 1
        tick_fontsize = 18
          
        spksamp_markersize = 4
        vel_markersize = 2
        traj_length_markersize = 6
        scatter_markersize = 8
        stripplot_markersize = 2
        feature_corr_markersize = 8
        wji_vs_trajcorr_markersize = 2
        reach_sample_markersize = 4
                
        traj_pos_sample_figsize = (4.5, 4.5)
        traj_vel_sample_figsize = (4, 4)
        traj_linewidth = 2
        traj_leadlag_linewidth = 3
        reach_sample_linewidth = 3
        
        preferred_traj_linewidth = 2
        distplot_linewidth = 3
        preferred_traj_figsize = (4.5, 4.5)
        
        weights_by_distance_figsize = (6, 4)
        aucScatter_figSize = (6, 6)
        FN_figsize = (5, 5)
        feature_corr_figSize = (4, 4)
        pearsonr_histsize = (3, 3)
        distplot_figsize = (3, 2)
        shuffle_figsize = (12, 3)
        stripplot_figsize = (6, 3)
        scatter_figsize = (5, 5)
        reach_sample_figsize = (7.5, 7.25)
        
        boxplot_figsize = (7, 3.5)
        boxplot_boxwidth = .5

        trajlength_figsize = (4, 4)
        trajlength_markersize = 12
        trajlength_linewidth = 2
        
        corr_marker_color = 'gray'

plt.rcParams['figure.dpi'] = plot_params.dpi
plt.rcParams['savefig.dpi'] = plot_params.dpi
# plt.rcParams["font.family"] = "Arial"
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

    
for fig_name in plot_params.figures_list:
    os.makedirs(os.path.join(plots, fig_name), exist_ok=True)
    if fig_name == 'unknown':
        os.makedirs(os.path.join(plots, fig_name, 'network'), exist_ok=True)
        os.makedirs(os.path.join(plots, fig_name, 'kinematics'), exist_ok=True)
        os.makedirs(os.path.join(plots, fig_name, 'auc_comparison'), exist_ok=True)

def add_cortical_area_to_units_results_df(units_res, cortical_bounds):
    
    cortical_area = []
    for row, unit in units_res.iterrows():
        bound_idx = [idx for idx, x_coord in enumerate(cortical_bounds['x_coord']) if x_coord == unit.x][0]
        y_bound = cortical_bounds['y_bound'][bound_idx]
        if y_bound is None:
            cortical_area.append(cortical_bounds['areas'][bound_idx])
        else:
            if unit.y >= y_bound:
                cortical_area.append(cortical_bounds['areas'][bound_idx][-1])
            else:
                cortical_area.append(cortical_bounds['areas'][bound_idx][0])
                
    units_res['cortical_area'] = cortical_area
    
    return units_res

def trajectory_vs_shuffle_sign_test(traj_res, shuf_res, units_res):
    
    ttest_p = [0]*traj_res.shape[0]
    mwu_p   = [0]*traj_res.shape[0]
    for idx, (unit_traj_auc, unit_shuf_auc) in enumerate(zip(traj_res, shuf_res)):
        
        tmp, ttest_p[idx] = ttest_ind   (unit_traj_auc, unit_shuf_auc, alternative='greater')
        tmp,   mwu_p[idx] = mannwhitneyu(unit_traj_auc, unit_shuf_auc, alternative='greater')
    
    units_res['ttest_p'] = ttest_p
    units_res['mwu_p'  ] = mwu_p
    
    return units_res

def plot_reach_samples(nReaches = 3, reachset1 = None, reachset2 = None, color1 = 'blue', color2='green', paperFig='unknown'):
        
    first_event_key = [key for idx, key in enumerate(kin_module.data_interfaces.keys()) if idx == 0][0]
    dlc_scorer = kin_module.data_interfaces[first_event_key].scorer 
    
    if 'simple_joints_model' in dlc_scorer:
        wrist_label = 'hand'
    elif 'marmoset_model' in dlc_scorer and nwb.subject.subject_id == 'TY':
        wrist_label = 'l-wrist'
    elif 'marmoset_model' in dlc_scorer and nwb.subject.subject_id == 'MG':
        wrist_label = 'r-wrist'
    
    if reachset1 is None:
        reachset1 = reach_set_df.loc[reach_set_df['FN_reach_set']==1, 'reach_num'].to_list()
    if reachset2 is None:
        reachset2 = reach_set_df.loc[reach_set_df['FN_reach_set']==2, 'reach_num'].to_list()
    
    fig0 = plt.figure(figsize = plot_params.reach_sample_figsize)
    ax0 = plt.axes(projection='3d')
    fig1 = plt.figure(figsize = plot_params.reach_sample_figsize)
    ax1 = plt.axes(projection='3d')
    r1_count, r2_count = 0, 0
    for rIdx, reach in reaches.iterrows():
        
        # get event data using container and ndx_pose names from segment_info table following form below:
        # nwb.processing['goal_directed_kinematics'].data_interfaces['moths_s_1_e_004_position']
        event_data      = kin_module.data_interfaces[reach.video_event] 
        
        wrist_kinematics    = event_data.pose_estimation_series[   wrist_label].data[reach.start_idx:reach.stop_idx+1].T

        if rIdx in reachset1:
            if nReaches is not None and r1_count > nReaches:
                continue
            lstyle = 'solid'
            r1_count += 1
            ax = ax0
            color = color1
        elif rIdx in reachset2:
            if nReaches is not None and r2_count > nReaches:
                continue
            lstyle = 'solid'
            r2_count += 1
            ax = ax1
            color = color2
        else:
            continue

        ax.plot3D(wrist_kinematics[0] , wrist_kinematics[1], wrist_kinematics[2], 
                  linewidth=plot_params.reach_sample_linewidth, color=color, linestyle=lstyle)
        ax.plot3D(wrist_kinematics[0,0], wrist_kinematics[1,0], wrist_kinematics[2,0], 
                  linewidth=plot_params.reach_sample_linewidth, color='black', marker='o', markersize=plot_params.reach_sample_markersize)
    
    for ax in [ax0, ax1]:
        ax.set_xlabel('x (cm)', fontsize = plot_params.axis_fontsize)
        ax.set_ylabel('y (cm)', fontsize = plot_params.axis_fontsize)
        ax.set_zlabel('z (cm)', fontsize = plot_params.axis_fontsize)
        ax.view_init(params.view_angle[0], params.view_angle[1])
        ax.set_xlim(params.apparatus_min_dims[0], params.apparatus_max_dims[0]),
        ax.set_ylim(params.apparatus_min_dims[1], params.apparatus_max_dims[1])
        ax.set_zlim(params.apparatus_min_dims[2], params.apparatus_max_dims[2])
        ax.set_xticks([params.apparatus_min_dims[0], params.apparatus_max_dims[0]]),
        ax.set_yticks([params.apparatus_min_dims[1], params.apparatus_max_dims[1]])
        ax.set_zticks([params.apparatus_min_dims[2], params.apparatus_max_dims[2]])
    plt.show()
        
    fig0.savefig(os.path.join(plots, paperFig, f'{marmcode}_reach_set1_reaches.png'), bbox_inches='tight', dpi=plot_params.dpi)
    fig1.savefig(os.path.join(plots, paperFig, f'{marmcode}_reach_set2_reaches.png'), bbox_inches='tight', dpi=plot_params.dpi)

def plot_boxplot_of_trajectory_model_auc(units_res, other_marm = False, filter_other_marm = False, model_list = None, label_list = None, paperFig = 'unknown'):
    
    if not other_marm:
        print('Edit the value of "other_marm" at top of script to produce a boxplot')
        return
    
    # with open(other_marm_pkl_infile, 'rb') as f:
    #     other_res_dict = dill.load(f)
    
    other_res_dict = load_dict_from_hdf5(other_marm_pkl_infile.with_suffix('.h5'))
    
    other_units_res = other_res_dict[params.best_lead_lag_key]['all_models_summary_results'].copy()
    
    if filter_other_marm:
        # other_units_res = other_units_res.loc[other_units_res['shuffled_traj_proportion_sign']>=0.5, :]
        other_units_res = other_units_res.loc[other_units_res['shuffled_traj_pval']<0.01/units_res.shape[0], :]
    
    if model_list is None:
        model_list = [col.split('_auc')[0] for col in units_res.columns if '_auc' in col] 
        label_list = model_list
    
    box_df = pd.DataFrame()
    for marm, res in zip([marmcode, other_marm], [units_res, other_units_res]):
        for model, label in zip(model_list, label_list):
            auc_key = f'{model}_auc'
            
            # auc_vals = res.loc[res['proportion_sign'] > 0.5, auc_key]
            auc_vals = res[auc_key]
            model_name_list = [label for i in range(auc_vals.shape[0])]
            marm_list = [marm for i in range(auc_vals.shape[0])]
            
            box_df = pd.concat((box_df, pd.DataFrame(data=zip(auc_vals, model_name_list, marm_list),
                                                     columns=['AUC', 'Model', 'Monkey'])),
                               axis=0, ignore_index=True)
    
    
    fig, axes = plt.subplots(1, 2, figsize = plot_params.boxplot_figsize, dpi = plot_params.dpi)
    sns.boxplot(ax=axes[0], data=box_df.loc[box_df['Monkey'] == 'TY', :], x='Model', y='AUC', whis=(2.5, 97.5), 
                color=(34/255, 131/255, 67/255), width=plot_params.boxplot_boxwidth, flierprops={"marker": ".", "fillstyle": 'full'})
    sns.boxplot(ax=axes[1], data=box_df.loc[box_df['Monkey'] == 'MG', :], x='Model', y='AUC',  whis=(2.5, 97.5),
                color=(34/255, 131/255, 67/255), width=plot_params.boxplot_boxwidth, flierprops={"marker": ".", "fillstyle": 'full'})
    
    # fig = sns.catplot(data=box_df, x='Model', y='AUC', col='Monkey', kind='box', color=(34/255, 131/255, 67/255))
    
    # axes[1].set_yticklabels(axes[1].get_yticklabels(), fontsize=plot_params.tick_fontsize)
    for ax in axes:
        ax.set_xlabel('', fontsize = plot_params.axis_fontsize)
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.set_xticks(ax.get_xticks(), labels = ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        sns.despine(ax=ax)
        # for axis in ['bottom', 'left']: 
        #     ax.spines[axis].set_linewidth(plot_params.axis_linewidth)
        #     ax.spines[axis].set_color('black')
        # ax.tick_params(width=plot_params.tick_width, length = plot_params.tick_length*2, labelsize = plot_params.tick_fontsize)
        ax.set_ylim(0.45, 0.75)

    axes[0].set_ylabel('AUC', fontsize = plot_params.axis_fontsize)    
    # axes[1].set_yticklabels(axes[1].get_yticklabels(), fontsize=plot_params.tick_fontsize)

    axes[1].set_ylabel('', fontsize = plot_params.axis_fontsize)    
    axes[1].set_yticklabels('', fontsize=plot_params.tick_fontsize)
    
    fig.savefig(os.path.join(plots, paperFig, f'trajectory_model_auc_results.png'), bbox_inches='tight', dpi=plot_params.dpi)

    #### Plot scatter of tortuosity and mean speed
    sample_info_marm1 = results_dict[params.best_lead_lag_key]['sampled_data']['sample_info'].copy()
    sample_info_marm2 = other_res_dict[params.best_lead_lag_key]['sampled_data']['sample_info'].copy()
    
    sample_info_marm1['Monkey'] = marmcode
    sample_info_marm2['Monkey'] = other_marm
    
    sample_info_combo_df = pd.concat((sample_info_marm1, sample_info_marm2), axis=0, ignore_index=True)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=plot_params.dpi)
    sns.scatterplot(ax=ax, data=sample_info_combo_df, x='mean_tortuosity', y='mean_speed', hue='Monkey', s = 2)
    ax.set_xlim(1, 15)
    ax.set_ylim(0, 70)

    jfig = sns.jointplot(data=sample_info_combo_df, x='mean_tortuosity', y='mean_speed', hue='Monkey', s = 2)
    jfig.fig.axes[0].set_xlim(1, 15)
    jfig.fig.axes[0].set_ylim(0, 70)
    jfig.fig.set_dpi(300)
    jfig.fig.set_figheight(5)
    jfig.fig.set_figwidth(5)
    # jfig.savefig(os.path.join(plots, paperFig, f'mean_speed_tortuosity_joint_plot.png'), bbox_inches='tight', dpi=plot_params.dpi)


def compute_AUC_distribution_statistics(model_keys, unit_idxs, lead_lag_key, plot=False):
        
    if unit_idxs is None:
        unit_idxs = range(results_dict[lead_lag_key]['all_models_summary_results'].shape[0])

    spike_samples = results_dict[lead_lag_key]['sampled_data']['spike_samples']
    nSpikeSamples = spike_samples.shape[1]

    p_ttest       = []
    p_signtest    = []
    prop_signtest = []
    ci_signtest_lower   = []
    p_mediantest = []
    
    for unit in unit_idxs:
        unit_AUCs    = []
        model_labels  = []
        for model_key in model_keys:
            unit_AUCs.extend(results_dict[lead_lag_key]['model_results'][model_key]['AUC'][unit])
            model_labels.extend([model_key]*results_dict[lead_lag_key]['model_results'][model_key]['AUC'].shape[1])
            if 'shuffle' in model_key:
                shuffle_key = model_key
        
        auc_df = pd.DataFrame(data = zip(unit_AUCs, model_labels), columns = ['AUC', 'Model'])
    
        t_stats = ttest_rel(auc_df.loc[auc_df['Model'] != shuffle_key, 'AUC'], 
                            auc_df.loc[auc_df['Model'] == shuffle_key, 'AUC'], 
                            alternative='greater')
    
        nTrajGreater = np.sum(auc_df.loc[auc_df['Model'] != shuffle_key, 'AUC'].values > 
                              auc_df.loc[auc_df['Model'] == shuffle_key, 'AUC'].values)
        nSamples     = len(auc_df.loc[auc_df['Model'] != shuffle_key, 'AUC'])
        sign_test    = binomtest(nTrajGreater, nSamples, p = 0.5, alternative='greater')
        
        med_out = median_test(auc_df.loc[auc_df['Model'] != shuffle_key, 'AUC'].values, 
                              auc_df.loc[auc_df['Model'] == shuffle_key, 'AUC'].values,
                              nan_policy='omit')
    
        if plot:
            if sign_test.proportion_estimate == 0.99 or sign_test.proportion_estimate == 0.66:
                fig, ax = plt.subplots()
                sns.kdeplot(data=auc_df, ax=ax, x='AUC', hue='Model',
                              log_scale=False, fill=False,
                              cumulative=False, common_norm=False, bw_adjust=.5)
                ax.set_xlabel('% AUC Loss')
                ax.set_title('Unit %d, prop=%.2f, p-val=%.3f, auc=%.2f, spkProp=%.3f' % (unit, 
                                                                                         sign_test.proportion_estimate, 
                                                                                         sign_test.pvalue,
                                                                                         auc_df.loc[auc_df['Model'] != shuffle_key, 'AUC'].mean(),
                                                                                         np.sum(spike_samples[unit] >= 1, axis=0)[0] / nSpikeSamples))      
                ax.set_xlim(0.4, 1)
            
                plt.show()
        
        p_ttest.append(t_stats.pvalue)
        p_signtest.append(sign_test.pvalue)
        prop_signtest.append(sign_test.proportion_estimate)
        ci_signtest_lower.append(sign_test.proportion_ci(confidence_level=.99)[0])
        p_mediantest.append(med_out[1])
    
    stats_df = pd.DataFrame(data = zip(p_ttest, p_signtest, prop_signtest, ci_signtest_lower, p_mediantest),
                            columns = ['pval_t', 'pval_sign', 'proportion_sign', 'CI_lower', 'pval_med'])
    
    return stats_df

def add_tuning_class_to_df(all_units_res, stats_df, stat_key, pval_key, thresh, direction='greater'):
    
    # tuning = ['untuned']*all_units_res.shape[0]
    # for idx, stat in enumerate(stats_df[stat_key]):
    #     if (direction.lower() == 'greater' and stat >= thresh) or (direction.lower() == 'less' and stat <= thresh):
    #         tuning[idx] = 'tuned'

    # all_units_res['tuning'] = tuning
    all_units_res[stat_key]                 = stats_df['proportion_sign'].values 
    all_units_res[pval_key]                 = stats_df['pval_sign'].values 
    all_units_res[f'{pval_key}_mediantest'] = stats_df['pval_med'].values
    
    return all_units_res

def determine_trajectory_significance(lead_lag_keys, plot = False):
    
    for lead_lag_key in lead_lag_keys:
        all_units_res = results_dict[lead_lag_key]['all_models_summary_results']    
        
        
        for shuffle_key in params.shuffle_keys:
            shuffle_model = [key for key in results_dict[lead_lag_key]['model_results'].keys() if shuffle_key in key][0]
            stats_df = compute_AUC_distribution_statistics(model_keys=[params.primary_traj_model, shuffle_model], 
                                                           unit_idxs=None, 
                                                           lead_lag_key=lead_lag_key,
                                                           plot=False)
            
            sorted_idx = stats_df.sort_values(by = 'proportion_sign', ascending = False).index.to_list()
            
            if plot:
                if lead_lag_key == params.best_lead_lag_key:
                    _ = compute_AUC_distribution_statistics(model_keys=[params.primary_traj_model, shuffle_model],
                                                            unit_idxs=sorted_idx, 
                                                            lead_lag_key=lead_lag_key,
                                                            plot=True)
                
            all_units_res = add_tuning_class_to_df(all_units_res, stats_df, f'{shuffle_key}_proportion_sign', f'{shuffle_key}_pval', thresh = params.significant_proportion_thresh, direction='greater') 
        
        results_dict[lead_lag_key]['all_models_summary_results'] = all_units_res

            
def organize_results_by_model_for_all_lags(fig_mode, per=None, prop=None, paperFig = 'unknown'):
    
    tmp_lead_lag_key = list(results_dict.keys())[0]
    model_keys = [key for key in results_dict[tmp_lead_lag_key]['all_models_summary_results'].columns if 'auc' in key.lower() and 'shuffle' not in key]
    corrected_model_names = [name.split('_auc')[0] for name in model_keys]
    
    results       = []
    model_names   = []
    sign_prop_df = pd.DataFrame()
    for idx, (mod_key, name) in enumerate(zip(model_keys, corrected_model_names)):
        model_names.append(name)
        tmp_results = pd.DataFrame()
        for lead_lag_key in results_dict.keys():
            all_units_res = results_dict[lead_lag_key]['all_models_summary_results']
            tmp_results[lead_lag_key] = all_units_res[mod_key] 
            cortical_areas = all_units_res['cortical_area']
            if name==params.primary_traj_model:
                # sign_prop_df[lead_lag_key] = all_units_res[f'{params.shuffle_keys[0]}_proportion_sign']
                sign_prop_df[lead_lag_key] = all_units_res['shuffled_traj_pval']
            
        if 'traj_avgPos' == name:
            plt_title = name
            fig_df = pd.DataFrame()
            for idx, col in enumerate(tmp_results.columns):
                tmp_data = tmp_results[col]
                tmp_prop = sign_prop_df[col]
                if fig_mode == 'percentile':
                    tmp_data = tmp_data[tmp_data > np.percentile(tmp_data, per)]
                    if idx == 0:
                        plt_title += f', Top {per}%'
                elif fig_mode == 'tuning_prop':
                    # tmp_data = tmp_data[tmp_prop > prop]
                    tmp_data = tmp_data[tmp_prop < prop]

                fig_df = pd.concat((fig_df, 
                                    pd.DataFrame(data=zip(tmp_data, 
                                                          np.repeat(col, tmp_data.shape[0]),
                                                          cortical_areas),
                                                 columns=['AUC', 'lead_lag_key', 'cortical_area'])),
                                   axis=0, ignore_index=True)
            
            # test for significant differences in performance compared to best model
            if prop is None and per is None:  
                best_leadlag_idx = np.argmax(fig_df.groupby('lead_lag_key')['AUC'].mean())            
                best_leadlag = np.unique(fig_df['lead_lag_key'])[best_leadlag_idx]
                for tmp_leadlag in np.unique(fig_df['lead_lag_key']):
                    nBest = np.sum(fig_df.loc[fig_df['lead_lag_key'] == best_leadlag, 'AUC'].values > fig_df.loc[fig_df['lead_lag_key'] == tmp_leadlag, 'AUC'].values)
                    nPossible = fig_df.loc[fig_df['lead_lag_key'] == best_leadlag, 'AUC'].size
                    sign_test = binomtest(nBest, nPossible, p = 0.5, alternative='greater')   
                    print(f'lead_lag = {tmp_leadlag}: prop = {np.round(sign_test.proportion_estimate, 2)}, p = {np.round(sign_test.pvalue, 5)}')
                # ttest_paired = ttest_rel(fig_df.loc[fig_df['lead_lag_key'] == best_leadlag, 'AUC'].values, fig_df.loc[fig_df['lead_lag_key'] == tmp_leadlag, 'AUC'].values, alternative='greater')
                # print(f'marm = {marm}, alpha = {tmp_leadlag}: p = {np.round(ttest_paired.pvalue, 3)}')


                                   
            leads = [re.findall(re.compile('lead_[0-9]{1,3}'), lead_lag_key)[0].split('_')[-1] for lead_lag_key in fig_df['lead_lag_key']] 
            lags  = [re.findall(re.compile('lag_[0-9]{1,3}' ), lead_lag_key)[0].split('_')[-1] for lead_lag_key in fig_df['lead_lag_key']]
            fig_df['Trajectory center (ms)'] = [(-int(lead)+int(lag)) / 2 for lead, lag in zip(leads, lags)]        
            fig_df['Trajectory length (ms)'] = [( int(lead)+int(lag))     for lead, lag in zip(leads, lags)]
            
            try: 
                plt.get_cmap('ylgn_dark')
            except:
                cmap_orig = plt.get_cmap("YlGn")
                colors_tophalf = cmap_orig(np.arange(cmap_orig.N, cmap_orig.N/2, -1, dtype=int))
                cmap = ListedColormap(colors_tophalf)
                colormaps.register(cmap, name = "ylgn_dark")
            
            fig, ax = plt.subplots(figsize=plot_params.trajlength_figsize, dpi = plot_params.dpi)
            sns.lineplot(ax=ax, data=fig_df, x = 'Trajectory center (ms)', y='AUC', hue = 'Trajectory length (ms)', 
                         legend=True, linestyle='-', err_style='bars', errorbar=("se", 1), linewidth=plot_params.trajlength_linewidth, 
                         marker='o', markersize=plot_params.trajlength_markersize, palette='ylgn_dark')
            ax.set_title('')

            if prop is None and per is None:
                ax.set_ylim(0.54, 0.59)
                ax.set_yticks(np.linspace(0.54, 0.585, 4))
                ax.set_yticklabels(ax.get_yticks(), fontsize=plot_params.tick_fontsize)
                ax.set_xticks([-250, -100, 0, 100, 250])
                ax.set_xticklabels(ax.get_xticks(), fontsize=plot_params.tick_fontsize)
            else:
                # ax.set_ylim(0.55, 0.595)
                # ax.set_yticks(np.linspace(0.55, 0.595, 4))
                # ax.set_yticklabels(ax.get_yticks(), fontsize=plot_params.tick_fontsize)
                ax.set_xticks([-250, -100, 0, 100, 250])
                ax.set_xticklabels(ax.get_xticks(), fontsize=plot_params.tick_fontsize)
            sns.despine(ax=ax)
            ax.set_xlabel('Trajectory center (ms)', fontsize=plot_params.axis_fontsize)
            ax.set_ylabel('Full kinematics AUC', fontsize=plot_params.axis_fontsize)
            
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            
            plt.show()
            
            if fig_mode == 'all':
                fig.savefig(os.path.join(plots, paperFig, f'{name}_auc_leadlagsRange_unfiltered_WITH_LEGEND.png'), bbox_inches='tight', dpi=plot_params.dpi)
            else:
                filtThresh = per if per is not None else int(prop*100)
                figname = f'{name}_auc_leadlagsRange_filtered_by_{fig_mode}_{filtThresh}.png'
                if prop is not None:
                    fig.savefig(os.path.join(plots, paperFig, figname), bbox_inches='tight', dpi=plot_params.dpi)
                else:
                    fig.savefig(os.path.join(plots, paperFig, figname), bbox_inches='tight', dpi=plot_params.dpi)
                
            rel = sns.relplot(data=fig_df, x = 'Trajectory center (ms)', y='AUC', hue = 'Trajectory length (ms)', col='cortical_area', 
                              linestyle='-.', kind='line', err_style='bars', errorbar=("se", 1), marker='o', markersize=10, palette='tab10')
            rel.fig.subplots_adjust(top=0.875) # adjust the Figure in rp
            rel.fig.suptitle(plt_title)
            plt.show()
            
            if fig_mode == 'all':
                rel.savefig(os.path.join(plots, paperFig, f'{name}_auc_leadlagsRange_unfiltered_sepByArea.png'), bbox_inches='tight', dpi=plot_params.dpi)
            else:
                rel.savefig(os.path.join(plots, paperFig, f'{figname.split(".png")[0]}_sepByArea.png'), bbox_inches='tight', dpi=plot_params.dpi)
                
        results.append(tmp_results)
    
    model_results_across_lags = {'model_name'    : model_names,
                                 'model_results' : results,
                                 'signtest_prop' : sign_prop_df}     
    
    return model_results_across_lags      

def compute_mean_model_performance(model_results_across_lags, percent = 0, percentile_mode='per_lag_set'):
    
    model_results_across_lags['mean_performance_by_lead_lag_all']     = [0 for i in range(len(model_results_across_lags['model_name']))]
    model_results_across_lags['mean_performance_by_lead_lag_untuned'] = [0 for i in range(len(model_results_across_lags['model_name']))]
    model_results_across_lags['mean_performance_by_lead_lag_tuned']   = [0 for i in range(len(model_results_across_lags['model_name']))]
    model_results_across_lags['mean_performance_by_lead_lag_percentile'] = [0 for i in range(len(model_results_across_lags['model_name']))]
    for idx, results in enumerate(model_results_across_lags['model_results']):
        
        # compute means and SE using AUC percentile filter
        tmp_results = results.copy()
        if percentile_mode == 'across_lag_sets':
            tmp_results = tmp_results[tmp_results >= np.percentile(tmp_results, percent)]
        elif percentile_mode == 'per_lag_set':
            for col in tmp_results.columns:
                print(tmp_results[col].shape[0] - np.sum(tmp_results[col] < np.percentile(tmp_results[col], percent)))
                tmp_results.loc[tmp_results[col] < np.percentile(tmp_results[col], percent), col] = np.nan
                
        model_results_across_lags['mean_performance_by_lead_lag_percentile'][idx] = pd.DataFrame(data=zip(tmp_results.mean().to_numpy(), tmp_results.sem().to_numpy()),
                                                                                                 index=results.columns,
                                                                                                 columns=['auc', 'sem'])    
        # compute means and SE for all, no filter
        tmp_results = results.copy()
        model_results_across_lags['mean_performance_by_lead_lag_all'][idx] = pd.DataFrame(data=zip(tmp_results.mean().to_numpy(), tmp_results.sem().to_numpy()),
                                                                                          index=results.columns,
                                                                                          columns=['auc', 'sem'])    
        # compute means and SE for tuned units
        tmp_results = results.copy()
        tmp_results = tmp_results[model_results_across_lags['signtest_prop'] >= params.significant_proportion_thresh]
        model_results_across_lags['mean_performance_by_lead_lag_tuned'][idx] = pd.DataFrame(data=zip(tmp_results.mean().to_numpy(), tmp_results.sem().to_numpy()),
                                                                                          index=results.columns,
                                                                                          columns=['auc', 'sem']) 

        # compute means and SE for untuned units
        tmp_results = results.copy()
        tmp_results = tmp_results[model_results_across_lags['signtest_prop'] < params.significant_proportion_thresh]
        model_results_across_lags['mean_performance_by_lead_lag_untuned'][idx] = pd.DataFrame(data=zip(tmp_results.mean().to_numpy(), tmp_results.sem().to_numpy()),
                                                                                          index=results.columns,
                                                                                          columns=['auc', 'sem'])   
        
    return model_results_across_lags

def plot_mean_traj_center_by_area(units_res, weighted_mean = True, weightsKey='%s_auc' % params.primary_traj_model):
    
    mean_traj_centers = []
    sem_traj_centers = []

    cortical_areas = params.cortical_boundaries['unique_areas']
    
    for area in cortical_areas:
        mask = units_res['cortical_area'] == area
        traj_centers = units_res.loc[mask, 'traj_center']
        if weighted_mean:
            weights = units_res.loc[mask, weightsKey]
            idxs = traj_centers.index[~np.isnan(traj_centers)]
            weights = weights[idxs]
            traj_centers = traj_centers[idxs]
            weighted_average = np.average(a = traj_centers, weights = weights)
            mean_traj_centers.append(weighted_average)    
        else:
            mean_traj_centers.append(traj_centers.mean())
        sem_traj_centers.append(traj_centers.sem())    
        
    
    fig, ax = plt.subplots(figsize=(6, 4), dpi =plot_params.dpi)
    
    ax.errorbar(x = range(len(cortical_areas)), 
                y = mean_traj_centers, 
                yerr = sem_traj_centers, 
                linewidth=0,
                elinewidth=2,
                marker='o',
                markersize=6,
                color='black')

    ax.set_xticks(range(len(cortical_areas)))
    ax.set_xticklabels(cortical_areas, fontsize = plot_params.tick_fontsize)
    ax.set_ylim([-5, 75])
    ax.set_yticks([0, 40, 75], fontsize = plot_params.tick_fontsize)
    ax.set_yticklabels([0, 40, 75], fontsize = plot_params.tick_fontsize)
    
    ax.set_xlabel('Cortical Area', fontsize = plot_params.axis_fontsize)
    ax.set_ylabel('Trajectory Center (ms)', fontsize = plot_params.axis_fontsize)
    
    plt.show()
    
    fig.savefig(os.path.join(plots, 'brain_area_optimal_lead_lag_final_output.png'), bbox_inches='tight', dpi=plot_params.dpi) 
    

        
def plot_sweep_over_lead_lag(model_results_across_lags, filter_key):

    # reorder = params.reorder
    reorder = [0, 1, 3, 2, 4, 5, 6, 12, 13, 16, 8,  14, 15, 11,  7,  9, 10]

    model_idx = [idx for idx, model_name in enumerate(model_results_across_lags['model_name']) if model_name == params.primary_traj_model][0]

    if filter_key is None:
        traj_mean_performance = model_results_across_lags['mean_performance_by_lead_lag_all'][model_idx].copy()
        nUnits = [model_results_across_lags['signtest_prop'].shape[0]]*len(reorder)
    else:
        traj_mean_performance = model_results_across_lags['mean_performance_by_lead_lag_%s' % filter_key][model_idx].copy()
        if 'tuned' in filter_key:
            mask = np.where(model_results_across_lags['signtest_prop'] >= params.significant_proportion_thresh)
            nUnits = [sum(mask[1] == model) for model in np.unique(mask[1])]
        else:
            nUnits = [model_results_across_lags['signtest_prop'].shape[0]]*len(reorder)
            
    leads = [re.findall(re.compile('lead_[0-9]{1,3}'), lead_lag_key)[0].split('_')[-1] for lead_lag_key in traj_mean_performance.index] 
    lags  = [re.findall(re.compile('lag_[0-9]{1,3}' ), lead_lag_key)[0].split('_')[-1] for lead_lag_key in traj_mean_performance.index]
    # lead_lag_ordering = [(-int(lead)+int(lag)) / 2 + (int(lead)+int(lag)) / 1000 for lead, lag in zip(leads, lags)]        
    traj_center = [(-int(lead)+int(lag)) / 2 for lead, lag in zip(leads, lags)]        
    traj_length = [( int(lead)+int(lag)) / 2 for lead, lag in zip(leads, lags)]
    
    traj_mean_performance['nUnits'] = nUnits
    traj_mean_performance['Trajectory Center (ms)'] = traj_center
    traj_mean_performance['Trajectory Length (ms)'] = traj_length
    # traj_mean_performance['reorder'] = lead_lag_ordering #reorder
    # traj_mean_performance.sort_values(by='reorder', inplace=True)
    
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi = plot_params.dpi)
    sns.pointplot(data=traj_mean_performance, ax=ax, x = 'Trajectory Center (ms)', hue='Trajectory Length (ms)')
    plt.show()
    
    best_ll_idx = np.where(traj_mean_performance.index == params.best_lead_lag_key)[0]
    leads = [re.findall(re.compile('lead_[0-9]{1,3}'), lead_lag_key)[0].split('_')[-1] for lead_lag_key in traj_mean_performance.index] 
    lags  = [re.findall(re.compile('lag_[0-9]{1,3}' ), lead_lag_key)[0].split('_')[-1] for lead_lag_key in traj_mean_performance.index] 
    xticklabels = ['-%s --> +%s' % (lead.zfill(3), lag.zfill(3)) for lead, lag in zip(leads, lags)]
    # fig, ax = plt.subplots()
    # ax.errorbar(traj_mean_performance['reorder'], traj_mean_performance['auc'], yerr=traj_mean_performance['SE'])
    # ax.set_xticks(traj_mean_performance['reorder'])
    # ax.set_xticklabels(traj_mean_performance.index, rotation=45)
    # plt.show()
    
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi = plot_params.dpi)
    ax.errorbar(traj_mean_performance['reorder'], 
                traj_mean_performance['auc'], 
                yerr=traj_mean_performance['sem'], 
                linewidth=0,
                elinewidth=3,
                marker='o',
                markersize=10,
                color='black')
    ax.errorbar(best_ll_idx,
                traj_mean_performance['auc'].iloc[best_ll_idx],
                yerr=traj_mean_performance['sem'].iloc[best_ll_idx], 
                linewidth=0,
                elinewidth=3,
                marker='o',
                markersize=10,
                color='green')
    
    if filter_key is not None:
        if 'tuned' in filter_key:
            y = np.max(traj_mean_performance['auc']) + np.max(traj_mean_performance['sem']) 
            
            for x, count in zip(traj_mean_performance['reorder'], traj_mean_performance['nUnits']):
                ax.text(x-.25, y, str(count))
        
    ax.set_xlabel('Trajectory Center (ms)', fontsize = plot_params.axis_fontsize)
    ax.set_ylabel('AUC (Mean %s sem)' % '\u00B1', fontsize = plot_params.axis_fontsize)
    ax.set_xticks([])
    ax.set_xticklabels([])
    if filter_key == 'all':
        ax.set_yticks([0.54, 0.57])
    elif filter_key == 'tuned':
        ax.set_yticks([0.58, 0.61])
    # ax.tick_params(width=2, length = 4, labelsize = plot_params.tick_fontsize)
    # ax.set_xticks(traj_mean_performance['reorder'])
    # for tick in ax.get_xticklabels():
    #     tick.set_fontsize(plot_params.tick_fontsize)
    # for tick in ax.get_yticklabels():
    #     tick.set_fontsize(plot_params.tick_fontsize)
    ax.set_xticks(traj_mean_performance['reorder'])
    ax.set_xticklabels(xticklabels, rotation=90)

    sns.despine(ax=ax)
    # ax.spines['bottom'].set_linewidth(plot_params.axis_linewidth)
    # ax.spines['left'  ].set_linewidth(plot_params.axis_linewidth)
    
    plt.show()
    
    if filter_key is None:
        fig.savefig(os.path.join(plots, 'model_auc_over_leadlags_unfiltered.png'), bbox_inches='tight', dpi=plot_params.dpi)
    else:
        fig.savefig(os.path.join(plots, 'model_auc_over_leadlags_filtered_by_%s.png' % filter_key), bbox_inches='tight', dpi=plot_params.dpi)
        
def summarize_model_results(units, lead_lag_keys):  
    
    for lead_lag_key in lead_lag_keys:
        
        if 'all_models_summary_results' in results_dict[lead_lag_key].keys():
            all_units_res = results_dict[lead_lag_key]['all_models_summary_results']
        else:
            all_units_res = units.copy()
            
        try:
            all_units_res.drop(columns=['spike_times', 'n_spikes'], inplace=True)
        except:
            pass
        
        for model_key in results_dict[lead_lag_key]['model_results'].keys():
            if model_key == params.primary_traj_model:
                col_names   = [f'{model_key}_auc', f'{model_key}_train_auc']
                metric_keys = [             'AUC',               'trainAUC'] 
            else:
                col_names   = [f'{model_key}_auc']
                metric_keys = [             'AUC'] 
                
            for col_name, metric_key in zip(col_names, metric_keys):
                if col_name not in all_units_res.columns and metric_key in results_dict[lead_lag_key]['model_results'][model_key].keys(): 
                    all_units_res[col_name] = results_dict[lead_lag_key]['model_results'][model_key][metric_key].mean(axis=-1)    
                else:
                    print('This model (%s, %s) has already been summarized in the all_models_summary_results dataframe' % (lead_lag_key, model_key))
        
        all_units_res = add_cortical_area_to_units_results_df(all_units_res, cortical_bounds=params.cortical_boundaries)

        results_dict[lead_lag_key]['all_models_summary_results'] = all_units_res
        
def plot_model_auc_comparison(units_res, x_key, y_key, minauc = 0.5, maxauc=1.0, targets=None, palette=None, paperFig = 'unknown'):
    
    if x_key[-4:] != '_auc':
        x_key = x_key + '_auc'
    if y_key[-4:] != '_auc':
        y_key = y_key + '_auc'
    
    xlabel = [f'{lab} AUC' for lab, key in zip(['Trajectory', 'Full kinematics', 'Velocity', 'Short kinematics', 'Kinematics and reachFN', 'Kinematics and spontaneous FN generalization'], 
                                               ['traj', 'traj_avgPos', 'shortTraj', 'shortTraj_avgPos', 'traj_avgPos_reach_FN', 'traj_avgPos_spont_train_reach_test_FN']) if f'{key}_auc' == x_key][0]
    
    ylabel = [f'{lab} AUC' for lab, key in zip(['Trajectory', 'Full kinematics', 'Velocity', 'Short kinematics', 'Kinematics and reachFN', 'Kinematics and spontaneous FN generalization'], 
                                               ['traj', 'traj_avgPos', 'shortTraj', 'shortTraj_avgPos', 'traj_avgPos_reach_FN', 'traj_avgPos_spont_train_reach_test_FN']) if f'{key}_auc' == y_key][0]
    
    
    units_res_plots = units_res.copy()     
   
    plot_title = 'Targets: All units'
    plot_name = 'area_under_curve_%s_%s.png' % (x_key, y_key)

    fig, ax = plt.subplots(figsize = plot_params.aucScatter_figSize, dpi = plot_params.dpi)
    sns.scatterplot(ax = ax, data = units_res_plots, x = x_key, y = y_key, 
                    hue = "snr", s = plot_params.scatter_markersize, legend=False, palette=palette)    
    ax.plot(np.arange(minauc, maxauc, 0.05), np.arange(minauc, maxauc, 0.05), '--k', linewidth = plot_params.traj_linewidth)
    ax.set_xlim(minauc, maxauc)
    ax.set_ylim(minauc, maxauc)
    ax.set_xlabel(xlabel, fontsize = plot_params.axis_fontsize)
    ax.set_ylabel(ylabel, fontsize = plot_params.axis_fontsize)
    ax.set_title('')

    ax.grid(False)
    plt.show()
    
    # fig.savefig(os.path.join(plots, paperFig, plot_name), bbox_inches='tight', dpi=plot_params.dpi)

def compute_and_analyze_pathlets(lead_lag_key, model, numplots, unitsToPlot=None, all_units_plot_subset = None, axlims=None):
    
    lead = float(re.findall(re.compile('lead_[0-9]{1,3}'), lead_lag_key)[0].split('_')[-1]) * 1e-3
    lag  = float(re.findall(re.compile('lag_[0-9]{1,3}' ), lead_lag_key)[0].split('_')[-1]) * 1e-3
    
    coefs = results_dict[lead_lag_key]['model_results'][model]['coefs']
    
    if 'traj' in model:
        if 'avgPos' in model:
            velTraj_mean = np.mean(coefs, axis = -1)[1:-3, :]
            avgPos_mean  = np.mean(coefs, axis=-1)[-3:, :]
            avgPos_mean  = np.swapaxes(avgPos_mean.reshape((3, int(np.shape(avgPos_mean)[0] / 3), np.shape(avgPos_mean)[-1])), 0, 1)
        elif model == 'traj_pca':
            comps = results_dict[lead_lag_key]['model_features']['traj_PCA_components']
            beta_mean  = np.mean(coefs, axis = -1)[1:np.shape(comps)[-1]+1, :]
            velTraj_mean = comps.T @ beta_mean 
            avgPos_mean = None
        else:
            avgPos_mean = None
            velTraj_mean = np.mean(coefs, axis = -1)[1:, :]
            
        velTraj_mean = np.swapaxes(velTraj_mean.reshape((3, int(np.shape(velTraj_mean)[0] / 3), np.shape(velTraj_mean)[-1])), 0, 1)
        posTraj_mean = cumtrapz(velTraj_mean, dx = (lead + lag) / np.shape(velTraj_mean)[0], axis = 0, initial = 0)
        dist = simps(np.linalg.norm(velTraj_mean, axis = 1), dx = (lead + lag) / np.shape(velTraj_mean)[0], axis = 0)
        
    elif model == 'position':
        posTraj_mean = np.mean(coefs, axis = -1)[1:, :]
        posTraj_mean = np.swapaxes(posTraj_mean.reshape((3, int(np.shape(posTraj_mean)[0] / 3), np.shape(posTraj_mean)[-1])), 0, 1)
    
    pathDivergence = np.empty(np.shape(coefs[0, ...].transpose()))
    velTraj_samples = []
    posTraj_samples = []
    for samp in range(np.shape(coefs)[-1]):
        if 'traj' in model:
            if 'avgPos' in model:
                velTraj_samp = coefs[1:-3, :, samp]
            elif model == 'traj_pca':
                beta_samp = coefs[1:np.shape(comps)[-1] +1, :, samp]
                velTraj_samp = comps.T @ beta_samp
            else:
                velTraj_samp = coefs[1:, :, samp]
            velTraj_samp = np.swapaxes(velTraj_samp.reshape((3, int(np.shape(velTraj_samp)[0] / 3), np.shape(velTraj_samp)[-1])), 0, 1)
            posTraj_samp = cumtrapz(velTraj_samp, dx = (lead + lag) / np.shape(velTraj_samp)[0], axis = 0, initial = 0)
        
        elif model == 'position':
            posTraj_samp = coefs[1:, :, samp]
            posTraj_samp = np.swapaxes(posTraj_samp.reshape((3, int(np.shape(posTraj_samp)[0] / 3), np.shape(posTraj_samp)[-1])), 0, 1)
        
        pathDivergence[samp, :] = np.sum(np.linalg.norm(posTraj_mean - posTraj_samp, axis = 1), axis = 0)
                    
        # divShuffle = np.empty((np.shape(pathDivergence)[0], np.shape(pathDivergence)[1], 100))
        # for shuffle in range(100):
        #     idx = np.random.choice(np.arange(np.shape(pathDivergence)[1]), size = np.shape(pathDivergence)[1], replace = 0)
        #     while np.sum(idx == np.arange(np.shape(pathDivergence)[1])) > 0:
        #         idx = np.random.choice(np.arange(np.shape(pathDivergence)[1]), size = np.shape(pathDivergence)[1], replace = 0)
    
        #     divShuffle[samp, :, shuffle] = np.sum(np.linalg.norm(posTraj[..., idx] - posTraj_samp, axis = 1), axis = 0)
            
        posTraj_samples.append(posTraj_samp)
        if 'velTraj_samp' in locals():
            velTraj_samples.append(velTraj_samp)
    
    # axlims_best  = plot_pathlet(posTraj, sample_pathlets, unit_info, unit_selector = 'max', numToPlot = 1, unitsToPlot = None)
    # axlims_worst = plot_pathlet(posTraj, sample_pathlets, unit_info, unit_selector = 'min', numToPlot = 1, unitsToPlot = None, axlims = axlims_best)
    axlims_good = plot_pathlet(posTraj_mean, posTraj_samples, lead_lag_key, model, avgPos_mean = avgPos_mean, unit_selector = 'max', numToPlot = numplots, unitsToPlot = unitsToPlot, all_units_plot_subset = all_units_plot_subset, axlims = axlims)
    _           = plot_pathlet(posTraj_mean, posTraj_samples, lead_lag_key, model, unit_selector = 'min', numToPlot = numplots, unitsToPlot = unitsToPlot, all_units_plot_subset = all_units_plot_subset, axlims = axlims_good)
    # axlims_good = plot_pathlet(velTraj_mean, velTraj_samples, lead_lag_key, model, unit_selector = 'max', numToPlot = 20, unitsToPlot = None, axlims = None)
    # _           = plot_pathlet(velTraj_mean, velTraj_samples, lead_lag_key, model, unit_selector = 'min', numToPlot =  5, unitsToPlot = None, axlims = axlims_good)

        
    pathDivergence_mean = np.mean(pathDivergence, axis = 0)
    # shuffledPathDivergence_mean = np.mean(np.mean(divShuffle, axis = -1), axis = 0)
    
    
    if 'velTraj_mean' not in locals():
        velTraj_mean = []

    return posTraj_mean, velTraj_mean, posTraj_samples, velTraj_samples, axlims_good  
    
def plot_pathlet(posTraj_mean, posTraj_samples, lead_lag_key, model, avgPos_mean = None, unit_selector = 'max', numToPlot = 5, unitsToPlot = None, all_units_plot_subset = None, axlims = None):
    
    all_units_res = results_dict[lead_lag_key]['all_models_summary_results']  
    traj_auc = all_units_res['%s_auc' % model].to_numpy()
    
    lead = float(re.findall(re.compile('lead_[0-9]{1,3}'), lead_lag_key)[0].split('_')[-1]) * 1e-3
    lag  = float(re.findall(re.compile('lag_[0-9]{1,3}' ), lead_lag_key)[0].split('_')[-1]) * 1e-3
    
    if unitsToPlot is None:
        if unit_selector == 'max':
            units = np.argpartition(traj_auc, -1*numToPlot)[-1*numToPlot:]
            units = units[np.argsort(traj_auc[units])]
            units = units[::-1]
        elif unit_selector == 'min':
            units = np.argpartition(traj_auc, numToPlot)[:numToPlot]
            units = units[np.argsort(traj_auc[units])]
    elif unitsToPlot is not None and numToPlot is not None:
        if unit_selector == 'max':
            units = np.argpartition(traj_auc, -1*numToPlot)[-1*numToPlot:]
            units = units[np.argsort(traj_auc[units])]
            units = units[::-1]
        elif unit_selector == 'min':
            units = np.argpartition(traj_auc, numToPlot)[:numToPlot]
            units = units[np.argsort(traj_auc[units])]
    else:
        units = unitsToPlot
    
    if axlims is None:
        pathlets_min_xyz = np.empty((numToPlot, 3))
        pathlets_max_xyz = np.empty((numToPlot, 3))
        for plotIdx, unit in enumerate(units):
            pathlets_min_xyz[plotIdx] = np.min(posTraj_mean[..., unit], axis = 0)
            pathlets_max_xyz[plotIdx] = np.max(posTraj_mean[..., unit], axis = 0)
        
        min_xyz = np.min(pathlets_min_xyz, axis = 0)
        max_xyz = np.max(pathlets_max_xyz, axis = 0)
    else:
        min_xyz = axlims[0]
        max_xyz = axlims[1]
    
    if numToPlot > 0:
        fig = plt.figure(figsize = plot_params.preferred_traj_figsize)
        ax = plt.axes(projection='3d')
        figname = 'units_'
        for idx, unit in enumerate(units):
            
            if numToPlot is not None and unitsToPlot is not None: 
                if idx not in unitsToPlot:
                    continue
            
            # title = '(%s) Unit %d' %(unit_selector, unit) 
                
            figname += f'{unit}_'
            leadSamp = round(lead / (lead + lag) * posTraj_mean.shape[0])
            for sampPath in posTraj_samples:
                ax.plot3D(sampPath[:leadSamp + 1, 0, unit], sampPath[:leadSamp + 1, 1, unit], sampPath[:leadSamp + 1, 2, unit], 'blue', linewidth=plot_params.preferred_traj_linewidth)
                ax.plot3D(sampPath[leadSamp:    , 0, unit], sampPath[leadSamp:    , 1, unit], sampPath[leadSamp:    , 2, unit], 'red', linewidth=plot_params.preferred_traj_linewidth)
            ax.plot3D(posTraj_mean[:leadSamp + 1, 0, unit], posTraj_mean[:leadSamp + 1, 1, unit], posTraj_mean[:leadSamp + 1, 2, unit], 'black', linewidth=plot_params.preferred_traj_linewidth*2)
            ax.plot3D(posTraj_mean[leadSamp:, 0, unit], posTraj_mean[leadSamp:, 1, unit], posTraj_mean[leadSamp:, 2, unit], 'black', linewidth=plot_params.preferred_traj_linewidth*2)
        
            if fig_mode == 'pres':
                ax.set_xlim(min_xyz[0], max_xyz[0])
                ax.set_ylim(min_xyz[1], max_xyz[1])
                ax.set_zlim(min_xyz[2], max_xyz[2])
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_zticklabels([])
                ax.set_xlabel('', fontsize = plot_params.axis_fontsize)
                ax.set_ylabel('', fontsize = plot_params.axis_fontsize)
                ax.set_zlabel('', fontsize = plot_params.axis_fontsize)
                
                ax.view_init(params.view_angle[0], params.view_angle[1])
                plt.show()                
                fig.savefig(os.path.join(plots, 'Fig2', f'{figname}pathlets.png' % unit), bbox_inches='tight', dpi=plot_params.dpi)
                
                fig = plt.figure(figsize = plot_params.preferred_traj_figsize)
                ax = plt.axes(projection='3d')
        
        if fig_mode == 'pres':
            return (min_xyz, max_xyz)    
                    
            
            # ax.set_title(title, fontsize = 16, fontweight = 'bold')
        ax.set_xlim(min_xyz[0], max_xyz[0])
        ax.set_ylim(min_xyz[1], max_xyz[1])
        ax.set_zlim(min_xyz[2], max_xyz[2])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        # ax.set_xlabel('x', fontsize = plot_params.axis_fontsize)
        # ax.set_ylabel('y', fontsize = plot_params.axis_fontsize)
        # ax.set_zlabel('z', fontsize = plot_params.axis_fontsize)
        ax.set_xlabel('', fontsize = plot_params.axis_fontsize)
        ax.set_ylabel('', fontsize = plot_params.axis_fontsize)
        ax.set_zlabel('', fontsize = plot_params.axis_fontsize)
        # ax.legend(['lead', 'lag'], loc='upper right', bbox_to_anchor=(1, 1), fontsize = 14, shadow=False)
        # ax.w_xaxis.line.set_color('black')
        # ax.w_yaxis.line.set_color('black')
        # ax.w_zaxis.line.set_color('black')
        # ax.w_xaxis.line.set_linewidth(plot_params.axis_linewidth)
        # ax.w_yaxis.line.set_linewidth(plot_params.axis_linewidth)
        # ax.w_zaxis.line.set_linewidth(plot_params.axis_linewidth)
    
        ax.view_init(params.view_angle[0], params.view_angle[1])
        plt.show()
        
        fig.savefig(os.path.join(plots, 'Fig2', f'{figname}pathlets.png'), bbox_inches='tight', dpi=plot_params.dpi)
    
    if unit_selector == 'max':
        
        fig1 = plt.figure(figsize = (4, 3.6))
        ax1  = plt.axes(projection='3d')
        fig2 = plt.figure(figsize = (4, 3.6))
        ax2  = plt.axes(projection='3d')
        axis_mins = np.empty((3, posTraj_mean.shape[-1]))
        axis_maxs = np.empty_like(axis_mins)
        for unit in range(posTraj_mean.shape[-1]):
            if all_units_plot_subset is not None and unit not in all_units_plot_subset:
                continue
            # title = '(%s) Unit %d' %(unit_selector, unit) 
            
            posTraj_plus_avgPos = posTraj_mean[..., unit]
            posTraj_plus_avgPos = posTraj_plus_avgPos - posTraj_plus_avgPos.mean(axis=0) + avgPos_mean[0, :, unit] 
            
            leadSamp = round(lead / (lead + lag) * posTraj_mean.shape[0])
            ax1.plot3D(posTraj_mean[:leadSamp + 1, 0, unit], posTraj_mean[:leadSamp + 1, 1, unit], posTraj_mean[:leadSamp + 1, 2, unit], 'blue', linewidth=1)
            ax1.plot3D(posTraj_mean[leadSamp:    , 0, unit], posTraj_mean[leadSamp:    , 1, unit], posTraj_mean[leadSamp:    , 2, unit], 'red' , linewidth=1)
            ax2.plot3D(posTraj_plus_avgPos[:leadSamp + 1, 0], posTraj_plus_avgPos[:leadSamp + 1, 1], posTraj_plus_avgPos[:leadSamp + 1, 2], 'blue', linewidth=1)
            ax2.plot3D(posTraj_plus_avgPos[leadSamp:    , 0], posTraj_plus_avgPos[leadSamp:    , 1], posTraj_plus_avgPos[leadSamp:    , 2], 'red' , linewidth=1)
            
            axis_mins[:, unit] = np.array([posTraj_plus_avgPos[:, dim].min() for dim in range(3)]) 
            axis_maxs[:, unit] = np.array([posTraj_plus_avgPos[:, dim].max() for dim in range(3)]) 
            
        # ax.set_title(title, fontsize = 16, fontweight = 'bold')
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # ax.set_zticklabels([])
        ax1.set_xlabel('x', fontsize = plot_params.axis_fontsize)
        ax1.set_ylabel('y', fontsize = plot_params.axis_fontsize)
        ax1.set_zlabel('z', fontsize = plot_params.axis_fontsize)
        ax2.set_xlabel('x', fontsize = plot_params.axis_fontsize)
        ax2.set_ylabel('y', fontsize = plot_params.axis_fontsize)
        ax2.set_zlabel('z', fontsize = plot_params.axis_fontsize)   
        ax1.set_xlim(min_xyz[0], max_xyz[0])
        ax1.set_ylim(min_xyz[1], max_xyz[1])
        ax1.set_zlim(min_xyz[2], max_xyz[2])
         # ax.set_xlabel('', fontsize = plot_params.axis_fontsize)
        # ax.set_ylabel('', fontsize = plot_params.axis_fontsize)
        # ax.set_zlabel('', fontsize = plot_params.axis_fontsize)
        # ax.legend(['lead', 'lag'], loc='upper right', bbox_to_anchor=(1, 1), fontsize = 14, shadow=False)
        
        axis_mins = np.percentile(axis_mins,  1, axis=0)
        axis_maxs = np.percentile(axis_maxs,  99, axis=0)
        # ax2.set_xlim(axis_mins[0], axis_maxs[0])
        # ax2.set_ylim(axis_mins[1], axis_maxs[1])
        # ax2.set_zlim(axis_mins[2], axis_maxs[2])
        # ax2.set_xlim(-0.25, 0.25)
        # ax2.set_ylim(-0.3, 0.2)
        # ax2.set_zlim(-0.25, 0.25)
        
        # ax1.w_xaxis.line.set_color('black')
        # ax1.w_yaxis.line.set_color('black')
        # ax1.w_zaxis.line.set_color('black')
        ax1.view_init(params.view_angle[0], params.view_angle[1])
        
        # ax2.w_xaxis.line.set_color('black')
        # ax2.w_yaxis.line.set_color('black')
        # ax2.w_zaxis.line.set_color('black')
        ax2.view_init(params.view_angle[0], params.view_angle[1])
        
        plt.show() 
        
        figname_mod = 'all' if all_units_plot_subset is None else pathlet_subset 
        fig1.savefig(os.path.join(plots, 'FigS2', f'{figname_mod}_units_pathlets_noPos.png'), bbox_inches='tight', dpi=plot_params.dpi)
        # fig2.savefig(os.path.join(plots, 'all_units_pathlets_withPos.png'), bbox_inches='tight', dpi=plot_params.dpi)


    if unitsToPlot is not None: 
        print(traj_auc[unitsToPlot[0]])
    
    return (min_xyz, max_xyz)

def compute_and_analyze_trajectory_correlations(units_res, posTraj_mean, velTraj_mean, 
                                                electrode_distances, lead_lag_key, 
                                                FN=None, mode = 'concat', nplots=5):
    
    lead = float(re.findall(re.compile('lead_[0-9]{1,3}'), lead_lag_key)[0].split('_')[-1]) * 1e-3
    lag  = float(re.findall(re.compile('lag_[0-9]{1,3}' ), lead_lag_key)[0].split('_')[-1]) * 1e-3
    
    pos_corr = np.full_like(electrode_distances, np.nan)
    vel_corr = np.full_like(electrode_distances, np.nan)
    connect  = np.full((electrode_distances.shape[0], electrode_distances.shape[0]), 'xx-xx')
    x1, y1 = np.full_like(electrode_distances, np.nan), np.full_like(electrode_distances, np.nan)
    x2, y2 = np.full_like(electrode_distances, np.nan), np.full_like(electrode_distances, np.nan)

    for i in range(posTraj_mean.shape[-1]):
        for j in range(posTraj_mean.shape[-1]):
            if i == j:
                    continue
            if mode == 'concat':
                p_i = posTraj_mean[..., i].transpose().flatten()     
                p_j = posTraj_mean[..., j].transpose().flatten()     
                v_i = velTraj_mean[..., i].transpose().flatten()     
                v_j = velTraj_mean[..., j].transpose().flatten()
                pos_rval, _ = pearsonr(p_i, p_j)
                vel_rval, _ = pearsonr(v_i, v_j)
            elif mode == 'average':
                pos_rval = 0
                vel_rval = 0
                for dim in [0, 1, 2]: 
                    pos_rval_tmp, _ = pearsonr(posTraj_mean[:, dim, i], posTraj_mean[:, dim, j])
                    vel_rval_tmp, _ = pearsonr(velTraj_mean[:, dim, i], velTraj_mean[:, dim, j])
                    pos_rval += pos_rval_tmp
                    vel_rval += vel_rval_tmp
                
                pos_rval = pos_rval / 3
                vel_rval = vel_rval / 3
                
            pos_corr[i, j] = pos_rval
            vel_corr[i, j] = vel_rval
            
            areas_pair = sorted([units_res["cortical_area"].iloc[i], units_res["cortical_area"].iloc[j]])
            pairs_idx = [i, j] if areas_pair[0] == units_res["cortical_area"].iloc[i] else [j, i]             
            x1[i, j] = units_res['x'].iloc[pairs_idx[0]]
            x2[i, j] = units_res['x'].iloc[pairs_idx[1]]
            y1[i, j] = units_res['y'].iloc[pairs_idx[0]]
            y2[i, j] = units_res['y'].iloc[pairs_idx[1]]
            connect [i, j] = f'{areas_pair[0]}-{areas_pair[1]}'
    
    if nplots is not None:
        max_corrs = np.sort(pos_corr[~np.isnan(pos_corr)].flatten())[-nplots*2::2]
        min_corrs = np.sort(pos_corr[~np.isnan(pos_corr)].flatten())[:nplots*2:2]
        med_corrs = np.sort(pos_corr[~np.isnan(pos_corr)].flatten())[round(pos_corr.size/2-nplots) : round(pos_corr.size/2+nplots):2]
        for corrs in [max_corrs, min_corrs, med_corrs]:
            unit_pairs = [np.where(pos_corr == corr)[0] for corr in corrs]
            for pair, corr in zip(unit_pairs, corrs):                
                leadSamp = round(lead / (lead + lag) * posTraj_mean.shape[0])
                fig = plt.figure(figsize = (4.95, 4.5))
                ax = plt.axes(projection='3d')
                for unit, colors in zip(pair, [('blue', 'red'), ('cyan', 'magenta')]):
                    ax.plot3D(posTraj_mean[:leadSamp + 1, 0, unit], posTraj_mean[:leadSamp + 1, 1, unit], posTraj_mean[:leadSamp + 1, 2, unit], colors[0], linewidth=3)
                    ax.plot3D(posTraj_mean[leadSamp:    , 0, unit], posTraj_mean[leadSamp:    , 1, unit], posTraj_mean[leadSamp:    , 2, unit], colors[1], linewidth=3)
                # ax.set_title(title, fontsize = 16, fontweight = 'bold')

                # ax.set_xticklabels([])
                # ax.set_yticklabels([])
                # ax.set_zticklabels([])
                ax.set_xlabel('x', fontsize = plot_params.axis_fontsize)
                ax.set_ylabel('y', fontsize = plot_params.axis_fontsize)
                ax.set_zlabel('z', fontsize = plot_params.axis_fontsize)
                # ax.set_xlabel('', fontsize = plot_params.axis_fontsize)
                # ax.set_ylabel('', fontsize = plot_params.axis_fontsize)
                # ax.set_zlabel('', fontsize = plot_params.axis_fontsize)
                # ax.legend(['lead', 'lag'], loc='upper right', bbox_to_anchor=(1, 1), fontsize = 14, shadow=False)
                # ax.w_xaxis.line.set_color('black')
                # ax.w_yaxis.line.set_color('black')
                # ax.w_zaxis.line.set_color('black')
                ax.view_init(params.view_angle[0], params.view_angle[1])
                ax.set_title(f'Units {pair[0]} and {pair[1]}, r = {round(corr, 2)}')
                plt.show()
                
                # fig.savefig(os.path.join(plots, 'unknown', f'corr_pair_pathlets_{pair[0]}_{pair[1]}.png'), bbox_inches='tight', dpi=plot_params.dpi)

    
    if FN is None:
        print('Load computed FNs to see correlations of pathlets vs other variables')
        return
    
    if FN.ndim == 3:
        FN = np.mean(FN, axis = 0)
    
    nUnits = pos_corr.shape[0]
    df = pd.DataFrame(data = zip(pos_corr[np.triu_indices(nUnits, k=1)],
                                  electrode_distances[np.triu_indices(nUnits, k=1)],
                                  FN[np.triu_indices(nUnits, k=1)],
                                  connect[np.triu_indices(nUnits, k=1)],
                                  x1[np.triu_indices(nUnits, k=1)],
                                  x2[np.triu_indices(nUnits, k=1)],
                                  y1[np.triu_indices(nUnits, k=1)],
                                  y2[np.triu_indices(nUnits, k=1)]),
                      columns = ['Pearson_corr', 'Distance', 'Wji', 'Connection', 'x1', 'x2', 'y1', 'y2'])
    # df = pd.DataFrame(data = zip(np.abs(pos_corr.flatten()),
    #                              electrode_distances.flatten(),
    #                              FN.flatten()),
    #                   columns = ['Pearson_Corr', 'Distance', 'Wji'])
    # df = pd.DataFrame(data = zip(pos_corr.flatten(),
    #                              vel_corr.flatten(),
    #                              electrode_distances.flatten(),
    #                              FN.flatten()),
                      # columns = ['Pearson_Corr', 'VelTraj_corr', 'Distance', 'Wji'])
    df['r_squared'] = df['Pearson_corr']**2
    
    df = df.loc[~np.isnan(df['Distance']), :]
    df.sort_values(by='Pearson_corr', ascending=False, inplace=True)
    df['rank'] = np.arange(df.shape[0]+1, 1, -1) / 2
    df.sort_index(inplace=True)

    fig, ax = plt.subplots(figsize = (4, 4), dpi = plot_params.dpi)
    sns.scatterplot(ax = ax, data = df, x = 'Pearson_corr', y = 'Wji', s = 20, legend=True) 
    plt.show()
    # fig.savefig(os.path.join(plots, 'Fig4', 'wji_vs_pearson_r.png'), bbox_inches='tight', dpi=plot_params.dpi)
    
    fig, ax = plt.subplots(figsize = (4, 4), dpi = plot_params.dpi)
    sns.scatterplot(ax = ax, data = df, x = 'r_squared', y = 'Wji', s = 20, legend=True) 
    plt.show()

    fig, ax = plt.subplots(figsize = (4, 4), dpi = plot_params.dpi)
    sns.pointplot(ax=ax, data = df, x = 'Connection', y = 'Pearson_corr', color='black', errorbar=('ci', 99))
    plt.show()
    # fig.savefig(os.path.join(plots, 'unknown', 'pearson_r_vs_connection.png'), bbox_inches='tight', dpi=plot_params.dpi)

    fig, ax = plt.subplots(figsize = (4, 4), dpi = plot_params.dpi)
    sns.pointplot(ax=ax, data = df, x = 'Connection', y = 'r_squared', color='black', errorbar='se')
    plt.show()
    # fig.savefig(os.path.join(plots, 'unknown', 'pearson_rsquare_vs_connection.png'), bbox_inches='tight', dpi=plot_params.dpi)

    fig, ax = plt.subplots(figsize = (4, 4), dpi = plot_params.dpi)
    sns.pointplot(ax=ax, data = df, x = 'Connection', y = 'Wji', color='black', errorbar='se')
    plt.show()
    # fig.savefig(os.path.join(plots, 'unknown', 'wji_vs_connection.png'), bbox_inches='tight', dpi=plot_params.dpi)

    fig, ax = plt.subplots(figsize = plot_params.pearsonr_histsize, dpi = plot_params.dpi)
    sns.histplot(ax=ax, data = df, x = 'Pearson_corr', color='black', kde=True)
    sns.despine(ax=ax, left=True)
    # for axis in ['bottom','left']:
    #     ax.spines[axis].set_linewidth(plot_params.axis_linewidth)
    #     ax.spines[axis].set_color('black')
    # ax.tick_params(width=plot_params.tick_width, length = plot_params.tick_length*2, labelsize = plot_params.tick_fontsize)
    ax.set_xlabel('Preferred trajectory \n correlation', fontsize = plot_params.axis_fontsize, multialignment='center')
    ax.set_ylabel('', fontsize = plot_params.axis_fontsize)
    ax.set_yticks([])
    ax.set_yticklabels('', fontsize = plot_params.axis_fontsize)
    ax.set_ylim(0, 1450)
    
    plt.show()
    fig.savefig(os.path.join(plots, 'Fig2', 'pearson_r_histogram.png'), bbox_inches='tight', dpi=plot_params.dpi)
    
    return df
            
def evaluate_lead_lag_by_model_coefficients(lead_lag_key = 'lead_200_lag_200', kin_type = 'traj_avgPos', mode='average', proportion_thresh=0.99):
    coefs = results_dict[lead_lag_key]['model_results'][kin_type]['coefs']
    feature_sample_times = results_dict[lead_lag_key]['model_features']['subsample_times']
    feature_sample_times = np.round(feature_sample_times, 2)
    all_units_res = results_dict[lead_lag_key]['all_models_summary_results']
    first_lag_idx = np.where(feature_sample_times>0)[0][0]

    norms_list  = []
    label_list  = []
    unit_list          = []
    cortical_area_list = []
    for unit_idx in range(coefs.shape[1]):
        if all_units_res['proportion_sign'].iloc[unit_idx] < proportion_thresh:
            print(unit_idx)
            continue
        unit_coefs = coefs[:, unit_idx]
        for sample in range(coefs.shape[2]):
            if kin_type == 'traj_avgPos':
                sample_coefs = unit_coefs[1:-3, sample]
            elif kin_type in ['position', 'traj']:
                sample_coefs = unit_coefs[1:, sample]
                
            sample_coefs = np.swapaxes(sample_coefs.reshape((3, int(np.shape(sample_coefs)[0] / 3))), 0, 1)

            if mode == 'average':
                label_list.append('lead')
                norms_list.append(np.linalg.norm(sample_coefs[:first_lag_idx, :].flatten()) / sample_coefs[:first_lag_idx, :].shape[0])
                unit_list.append(unit_idx)
                cortical_area_list.append(all_units_res['cortical_area'].iloc[unit_idx])            
                
                label_list.append('lag')
                norms_list.append (np.linalg.norm(sample_coefs[first_lag_idx:, :].flatten()) / sample_coefs[first_lag_idx:, :].shape[0])
                unit_list.append(unit_idx)
                cortical_area_list.append(all_units_res['cortical_area'].iloc[unit_idx])  
            elif mode == 'each_lag':
                for ll_idx, ll_time in enumerate(feature_sample_times):
                    label_list.append(ll_time)
                    unit_list.append(unit_idx)
                    cortical_area_list.append(all_units_res['cortical_area'].iloc[unit_idx])
                    norms_list.append(np.linalg.norm(sample_coefs[ll_idx, :]))   
    
    lead_lag_norms_df = pd.DataFrame(data = zip(unit_list, norms_list, label_list, cortical_area_list),
                                      columns = ['unit', 'norm', 'label', 'cortical_area'])
    
    significant_diff_df = pd.DataFrame(columns = lead_lag_norms_df.columns) 
    diff_magnitude_df = pd.DataFrame(data = np.full((np.unique(lead_lag_norms_df.unit).size, 2), np.nan), columns = ['diff', 'cortical_area'])
    for unit_idx, unit in enumerate(np.unique(lead_lag_norms_df.unit)):
        unit_df = lead_lag_norms_df.loc[lead_lag_norms_df['unit'] == unit, :] 
        if mode == 'average':
            _, ttest_pval = ttest_ind(unit_df.loc[unit_df['label'] == 'lead', 'norm'], unit_df.loc[unit_df['label'] == 'lag', 'norm'], alternative='two-sided')
            if ttest_pval < 0.05:
                significant_diff_df = pd.concat((significant_diff_df, unit_df), axis = 0)
            diff_magnitude_df.iloc[unit_idx] = [unit_df.loc[unit_df['label'] == 'lag', 'norm'].mean() - unit_df.loc[unit_df['label'] == 'lead', 'norm'].mean(),
                                                unit_df.cortical_area.iloc[0]]
            
        elif mode == 'each_lag':
            f, pval =  f_oneway(unit_df.loc[unit_df['label'] == feature_sample_times[0], 'norm'],
                                unit_df.loc[unit_df['label'] == feature_sample_times[1], 'norm'],
                                unit_df.loc[unit_df['label'] == feature_sample_times[2], 'norm'],
                                unit_df.loc[unit_df['label'] == feature_sample_times[3], 'norm'],
                                unit_df.loc[unit_df['label'] == feature_sample_times[4], 'norm'],
                                unit_df.loc[unit_df['label'] == feature_sample_times[5], 'norm'],
                                unit_df.loc[unit_df['label'] == feature_sample_times[6], 'norm'],
                                unit_df.loc[unit_df['label'] == feature_sample_times[7], 'norm'],
                                unit_df.loc[unit_df['label'] == feature_sample_times[8], 'norm'],
                                unit_df.loc[unit_df['label'] == feature_sample_times[9], 'norm'],
                                unit_df.loc[unit_df['label'] == feature_sample_times[10], 'norm'],
                                unit_df.loc[unit_df['label'] == feature_sample_times[11], 'norm'],
                                unit_df.loc[unit_df['label'] == feature_sample_times[12], 'norm'],
                                unit_df.loc[unit_df['label'] == feature_sample_times[13], 'norm'],
                                unit_df.loc[unit_df['label'] == feature_sample_times[14], 'norm'],
                                unit_df.loc[unit_df['label'] == feature_sample_times[15], 'norm'])
            if pval<1:
                significant_diff_df = pd.concat((significant_diff_df, unit_df), axis = 0)
    
    if mode == 'average':
        fig, ax = plt.subplots(3, 1, sharex=True)
        sns.histplot(ax = ax[0], data = diff_magnitude_df.loc[diff_magnitude_df['cortical_area'] == 'M1'], x='diff', hue='cortical_area', palette={'M1': 'b'}, bins = np.linspace(-0.025, 0.025, 25))
        sns.histplot(ax = ax[1], data = diff_magnitude_df.loc[diff_magnitude_df['cortical_area'] == '3a'], x='diff', hue='cortical_area', palette={'3a': 'g'}, bins = np.linspace(-0.025, 0.025, 25))
        sns.histplot(ax = ax[2], data = diff_magnitude_df.loc[diff_magnitude_df['cortical_area'] == '3b'], x='diff', hue='cortical_area', palette={'3b': 'm'}, bins = np.linspace(-0.025, 0.025, 25))
        ax[0].set_xlim(-0.025, 0.025)
        plt.show()
    
    fig = sns.catplot (data = significant_diff_df, x='label', y='norm', hue='unit', col='cortical_area', kind='point', legend=False, errorbar=('ci', 99))
    for ax, area in zip(fig.axes[0], ['M1', '3a', '3b']): 
        zordered_lines = ax.lines
        zordered_collections = ax.collections
        plt.setp(zordered_lines, zorder=1)
        plt.setp(zordered_collections, zorder=1)
        sns.pointplot(ax=ax, data = significant_diff_df.loc[significant_diff_df['cortical_area'] == area], x='label', y='norm', color='black', errorbar=('ci', 99))
        for l in ax.lines:
            if l not in zordered_lines:
                plt.setp(l, zorder=2)
                zordered_lines.append(l)
        for c in ax.collections:
            if c not in zordered_collections:
                plt.setp(c, zorder=2)
                zordered_collections.append(c) 
                
def sig_tests(unit_info, x_key, y_key, alternative='greater', unit_info_reduced = None):
    
    if x_key[-4:] != '_auc':
        x_key = x_key + '_auc'
    if y_key[-4:] != '_auc':
        y_key = y_key + '_auc'
    
    if unit_info_reduced is None:
        nFull = np.sum(unit_info[y_key] > unit_info[x_key])
        nUnits = np.shape(unit_info)[0]
        
        sign_test = binomtest(nFull, nUnits, p = 0.5, alternative=alternative)
        
        ttest_paired = ttest_rel(unit_info[y_key], unit_info[x_key], alternative=alternative)

    else:
        nPathlet = np.sum(unit_info.pathlet_AUC > unit_info_reduced.pathlet_AUC)
        nUnits = np.shape(unit_info)[0]
        sign_test = binomtest(nPathlet, nUnits, p = 0.5, alternative=alternative)
        ttest_paired = ttest_rel(unit_info.pathlet_AUC, unit_info_reduced.pathlet_AUC, alternative=alternative)

    return sign_test, ttest_paired

        
if __name__ == "__main__":
    
    # with open(pkl_infile, 'rb') as f:
    #     results_dict = dill.load(f)
    
    results_dict = load_dict_from_hdf5(pkl_infile.with_suffix('.h5'), top_level_list=False, convert_4d_array_to_list = True)
    
    with NWBHDF5IO(nwb_infile, 'r') as io:
        nwb = io.read()

        reaches_key = [key for key in nwb.intervals.keys() if 'reaching_segments' in key][0]
        
        units, reaches, kin_module = get_sorted_units_and_apparatus_kinematics_with_metadata(nwb, reaches_key, mua_to_fix=mua_to_fix, plot=False) 
        
        units = choose_units_for_model(units, quality_key = 'snr', quality_thresh = params.snr_thresh, frate_thresh = params.frate_thresh, bad_units_list=bad_units_list)
        # units = choose_units_for_model(units, quality_key='amp', quality_thresh=5, frate_thresh=params.frate_thresh)
        
        if FN_computed:
            spontaneous_FN = nwb.scratch['spontaneous_FN'].data[:]
            reach_FN = nwb.scratch[params.FN_key].data[:] 
            reach_set_df = nwb.scratch['split_FNs_reach_sets'].to_dataframe()
            
            plot_reach_samples(nReaches = None, reachset1 = reaches_to_plot[0], reachset2 = reaches_to_plot[1], 
                                color1 = color1, color2=color2, paperFig='Fig1')
        
    if params.lead == 'all' and params.lag == 'all':
        lead_lag_keys = list(results_dict.keys())
    else:
        lead_lag_keys = ['lead_%d_lag_%d' % (int(lead*1e3), int(lag*1e3)) for lead, lag in zip(params.lead, params.lag)]
    
    lead_lag_keep_keys = []
    for ll_idx, lead_lag_key in enumerate(lead_lag_keys):
        if 'model_results' not in results_dict[lead_lag_key].keys():
            del results_dict[lead_lag_key]
        else:
            lead_lag_keep_keys.append(lead_lag_key)
    
    lead_lag_keys = lead_lag_keep_keys
    
    summarize_model_results(units, lead_lag_keys)  
        
    determine_trajectory_significance(lead_lag_keys, plot=False)
        
    units_res = results_dict[params.best_lead_lag_key]['all_models_summary_results'].copy()
    electrode_distances = get_interelectrode_distances_by_unit(units_res, array_type='utah')
    # units_res_completely_untuned_units_filtered = units_res.loc[units_res['shuffled_traj_proportion_sign']>=0.5, :]
    units_res_completely_untuned_units_filtered = units_res.loc[units_res['shuffled_traj_pval']<0.01/units_res.shape[0], :]
    
    # model_results_across_lags = organize_results_by_model_for_all_lags(fig_mode='percentile', per=60)
    # model_results_across_lags = organize_results_by_model_for_all_lags(fig_mode='tuning_prop', prop=0.5)
    # model_results_across_lags = organize_results_by_model_for_all_lags(fig_mode='tuning_prop', prop=0.01/units_res.shape[0], paperFig='Fig2')
    model_results_across_lags = organize_results_by_model_for_all_lags(fig_mode='all', paperFig='Fig2')
    
    # model_results_across_lags = compute_mean_model_performance(model_results_across_lags, percent = params.nUnits_percentile, percentile_mode = 'per_lag_set')
    
    '''Plotting with no cutoff for proportion sign'''     
    plot_model_auc_comparison(units_res, 'traj', 'traj_avgPos', targets = None, minauc=0.45, maxauc=0.8, paperFig='Fig2') 
    sign_test, ttest = sig_tests(units_res, 'traj', 'traj_avgPos', alternative='greater')
    print(('traj', 'traj_avgPos', sign_test))

    plot_model_auc_comparison(units_res, 'shortTraj', 'traj', targets = None, minauc=0.45, maxauc=0.8, paperFig='Fig2') 
    sign_test, ttest = sig_tests(units_res, 'shortTraj', 'traj', alternative='greater')
    print(('shortTraj', 'traj', sign_test))

    plot_model_auc_comparison(units_res, 'shortTraj_avgPos', 'traj_avgPos', targets = None, minauc=0.45, maxauc=0.8, paperFig='Fig2') 
    sign_test, ttest = sig_tests(units_res, 'shortTraj_avgPos', 'traj_avgPos', alternative='greater')
    print(('shortTraj_avgPos', 'traj_avgPos', sign_test))

    '''Plotting with 0.5 cutoff for proportion sign'''
    plot_model_auc_comparison(units_res_completely_untuned_units_filtered, 'shortTraj', 'traj', targets = None, minauc=0.45, maxauc=0.8, paperFig='Fig2') 
    sign_test, ttest = sig_tests(units_res_completely_untuned_units_filtered, 'shortTraj', 'traj', alternative='greater')
    print(('shortTraj', 'traj', sign_test))

    plot_model_auc_comparison(units_res_completely_untuned_units_filtered, 'shortTraj_avgPos', 'traj_avgPos', targets = None, minauc=0.45, maxauc=0.8, paperFig='Fig2') 
    sign_test, ttest = sig_tests(units_res_completely_untuned_units_filtered, 'shortTraj_avgPos', 'traj_avgPos', alternative='greater')
    print(('shortTraj_avgPos', 'traj_avgPos', sign_test))

    # labels = ['Trajectory', 'Full Kinematics', 'Velocity', 'Short Kinematics', 'Kinematics and reachFN', 'Kinematics and Spontaneous FN Generalization']
    labels = ['Total shuffle', 'Trajectory shuffle', 'Velocity', 'Trajectory', 'Short kinematics', 'Full kinematics']
    plot_boxplot_of_trajectory_model_auc(units_res, other_marm = other_marm, filter_other_marm = False,
                                         model_list = ['traj_avgPos_shuffled_spikes', 
                                                       'traj_avgPos_shuffled_traj', 'shortTraj','traj','shortTraj_avgPos',
                                                       'traj_avgPos', 'traj_avgPos_shuffled_spikes', 
                                                       'traj_avgPos_shuffled_traj'],
                                         label_list = labels,
                                         paperFig = 'Fig2')
    
    plot_boxplot_of_trajectory_model_auc(units_res_completely_untuned_units_filtered, other_marm = other_marm, filter_other_marm = True,
                                         model_list = ['traj_avgPos_shuffled_spikes', 
                                                       'traj_avgPos_shuffled_traj', 'shortTraj','traj','shortTraj_avgPos',
                                                       'traj_avgPos'],
                                         label_list = labels,
                                         paperFig = 'Fig2')

    posTraj_mean, velTraj_mean, posTraj_samples, velTraj_samples, axlims = compute_and_analyze_pathlets(params.best_lead_lag_key, 
                                                                                                        'traj_avgPos', 
                                                                                                        numplots = 0, 
                                                                                                        unitsToPlot=None,
                                                                                                        all_units_plot_subset = None,
                                                                                                        axlims=unit_axlims)


    posTraj_mean, velTraj_mean, posTraj_samples, velTraj_samples, axlims = compute_and_analyze_pathlets(params.best_lead_lag_key, 
                                                                                                        'traj_avgPos', 
                                                                                                        numplots = 15, 
                                                                                                        unitsToPlot=units_to_plot,
                                                                                                        all_units_plot_subset = all_units_plot_subset,
                                                                                                        axlims=unit_axlims)

    df = compute_and_analyze_trajectory_correlations(units_res, posTraj_mean, velTraj_mean, electrode_distances, params.best_lead_lag_key, FN = reach_FN, mode='concat', nplots=5)
    
    pval=0.01
    print(f'At p = {pval} with bonferroni correction: {(units_res["shuffled_spikes_pval"]<pval/units_res.shape[0]).sum()}/{units_res.shape[0]} tuned to full kinematics, {(units_res["shuffled_traj_pval"]<pval/units_res.shape[0]).sum()}/{units_res.shape[0]} tuned to trajectory details')
    pval=0.05
    print(f'At p = {pval} with bonferroni correction: {(units_res["shuffled_spikes_pval"]<pval/units_res.shape[0]).sum()}/{units_res.shape[0]} tuned to full kinematics, {(units_res["shuffled_traj_pval"]<pval/units_res.shape[0]).sum()}/{units_res.shape[0]} tuned to trajectory details')
    
   
    if save_kinModels_pkl:
        with open(pkl_outfile, 'wb') as f:
            dill.dump(results_dict, f, recurse=True) 


