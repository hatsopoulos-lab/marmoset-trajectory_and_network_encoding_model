# -*- coding: utf-8 -*-
"""
Created on Tue May 31 10:09:38 2022

@author: Dalton
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import colormaps
import dill
import os
import math
import re
import h5py
import seaborn as sns
from itertools import product
from sklearn.metrics.pairwise import cosine_similarity
import scipy.io
from scipy.spatial.distance import correlation, euclidean, cosine, seuclidean
from scipy.stats import binomtest, ttest_rel, ttest_ind, mannwhitneyu, linregress, pearsonr, median_test
from scipy.integrate import cumtrapz
from scipy.integrate import simps
from scipy.ndimage import gaussian_filter
from mpl_toolkits import mplot3d 
from scipy.ndimage import median_filter
from importlib import sys
from pynwb import NWBHDF5IO
import ndx_pose
from pathlib import Path

data_path = Path('/project/nicho/projects/dalton/network_encoding_paper/clean_final_analysis/data')
code_path = Path('/project/nicho/projects/marmosets/code_database/analysis/trajectory_encoding_model/clean_final_analysis/')

sys.path.insert(0, str(code_path))
from utils import get_interelectrode_distances_by_unit, load_dict_from_hdf5, save_dict_to_hdf5
from hatlab_nwb_functions import get_sorted_units_and_apparatus_kinematics_with_metadata, remove_duplicate_spikes_from_good_single_units   

marmcode='MG'
other_marm = None #'MG' #None
fig_mode='paper'
save_kinModels_pkl = False
skip_network_permutations = True

pkl_in_tag  = 'network_models_created'
pkl_out_tag = 'network_models_summarized' 
pkl_add_stats_tag = 'kinematic_models_summarized' 
gen_file_tag = 'generalization_experiments'

if marmcode=='TY':
    nwb_infile   = data_path / 'TY' / 'TY20210211_freeAndMoths-003_resorted_20230612_DM_with_functional_networks.nwb' 
    annotated_nwbfile =  nwb_infile.parent / f'{nwb_infile.stem}_annotated_with_reach_segments{nwb_infile.suffix}'
    single_reach_nwbfile = nwb_infile.parent / f'{nwb_infile.stem}_single_reaches{nwb_infile.suffix}'
    # reach_seg_nwbfile =  nwb_infile.parent / f'{nwb_infile.stem}_reach_segments{nwb_infile.suffix}'
    modulation_base = nwb_infile.parent / nwb_infile.stem.split('_with_functional_networks')[0]
    fps = 150
    filtered_good_units_idxs = [88, 92, 123]

elif marmcode=='MG':
    nwb_infile   = data_path / 'MG' / 'MG20230416_1505_mothsAndFree-002_processed_DM_with_functional_networks.nwb'
    annotated_nwbfile =  nwb_infile.parent / f'{nwb_infile.stem}_annotated_with_reach_segments{nwb_infile.suffix}'
    single_reach_nwbfile = nwb_infile.parent / f'{nwb_infile.stem}_single_reaches{nwb_infile.suffix}'
    modulation_base = nwb_infile.parent / nwb_infile.stem.split('_with_functional_networks')[0]
    filtered_good_units_idxs = [7, 9, 16]

    fps = 200
    
pkl_infile       = nwb_infile.parent / f'{nwb_infile.stem.split("_with_functional_networks")[0]}_{pkl_in_tag}.pkl'
pkl_outfile      = nwb_infile.parent / f'{nwb_infile.stem.split("_with_functional_networks")[0]}_{pkl_out_tag}.pkl'
pkl_addstatsfile = nwb_infile.parent / f'{nwb_infile.stem.split("_with_functional_networks")[0]}_{pkl_add_stats_tag}.pkl'
gen_results_file = nwb_infile.parent / f'{nwb_infile.stem.split("_with_functional_networks")[0]}_{gen_file_tag}.h5'

dataset_code = pkl_infile.stem.split('_')[0]
if fig_mode == 'paper':
    plots = nwb_infile.parent.parent.parent / 'plots' / dataset_code
elif fig_mode == 'pres':
    plots = nwb_infile.parent.parent.parent / 'presentation_plots' / dataset_code


try: 
    plt.get_cmap('FN_palette')
except:
    cmap_orig = plt.get_cmap("Paired")
    FN_colors = cmap_orig([1, 0, 2])
    cmap = colors.ListedColormap(FN_colors)
    colormaps.register(cmap, name="FN_palette")
    FN_5_colors_annotated      = np.vstack((FN_colors, np.array([[99,99,99, 256], [251,154,153,256]])/256)) 
    FN_5_colors_extend_retract = np.vstack((FN_colors, np.array([[51,160,44, 256], [128,0,128,256]])/256)) 
    frate_colors               = np.vstack((FN_5_colors_annotated[[0, 2, 3, 4]], FN_5_colors_extend_retract[[3, 4]])) 
    colormaps.register(colors.ListedColormap((FN_5_colors_annotated)), name='annot_FN_palette')
    colormaps.register(colors.ListedColormap((FN_5_colors_extend_retract)), name='ext_ret_FN_palette')
    colormaps.register(colors.ListedColormap((frate_colors)), name='frate_palette')    
try: 
    plt.get_cmap('ylgn_dark_fig4')
except:
    cmap_orig = plt.get_cmap("YlGn")
    colors_tophalf = cmap_orig(np.arange(cmap_orig.N, cmap_orig.N/10, -1, dtype=int))
    cmap = colors.ListedColormap(colors_tophalf)
    colormaps.register(cmap, name = "ylgn_dark_fig4")

fig6_palette = 'Dark2'

class params:
    FN_key = 'split_reach_FNs'#'split_reach_FNs'
    significant_proportion_thresh = 0.99
 
    primary_traj_model = 'traj_avgPos'
    
    reach_specific_thresh = dict()
    test_behavior = None

    if marmcode == 'TY':
        best_lead_lag_key = 'lead_100_lag_300' #None
        cortical_boundaries = {'x_coord'      : [        0,          400,       800,      1200,                 1600,    2000,    2400,    2800,    3200,    3600],
                               'y_bound'      : [     None,         None,      None,      None,                 1200,    None,    None,    None,    None,    None],
                               'areas'        : ['Sensory',    'Sensory', 'Sensory', 'Sensory', ['Motor', 'Sensory'], 'Motor', 'Motor', 'Motor', 'Motor', 'Motor'],
                               'unique_areas' : ['Sensory', 'Motor']}
        motor_bound = 1800
        motor_bound_names = ['S', 'M']
        flip_area_order = False
        kin_only_model='traj_avgPos'
        
    elif marmcode == 'MG':
        best_lead_lag_key = 'lead_100_lag_300' #None
        cortical_boundaries = {'x_coord'      : [      0,     400,     800,    1200,    1600,    2000,    2400,      2800,      3200,      3600],
                               'y_bound'      : [   None,    None,    None,    None,    None,    None,    None,      None,      None,      None],
                               'areas'        : ['Motor', 'Motor', 'Motor', 'Motor', 'Motor', 'Motor', 'Motor', 'Sensory', 'Sensory', 'Sensory'],
                               'unique_areas' : ['Motor', 'Sensory']}
        motor_bound = 2600
        motor_bound_names = ['M', 'S']
        flip_order_area = True
        kin_only_model='traj_avgPos'
    
class plot_params:
    # axis_fontsize = 24
    # dpi = 300
    # axis_linewidth = 2
    # tick_length = 2
    # tick_width = 1
    # map_figSize = (6, 8)
    # tick_fontsize = 18
    # aucScatter_figSize = (7, 7)
    
    figures_list = ['Fig1', 'Fig2', 'Fig3', 'Fig4', 'Fig5', 'Fig6', 
                    'FigS1',  'FigS2',  'FigS3',  'FigS4', 'FigS5', 'FigS6', 'FigS7', 'FigS8_and_9', 'FigS10_and_11', 
                    'unknown', 'Exploratory_Spont_FNs', 'extension_retraction_FNs', 'Revision_Plots', 'Response_Only',]

    mostUnits_FN = 175
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
        traj_length_markersize = 6
        scatter_markersize = 8
        stripplot_markersize = 2
        feature_corr_markersize = 8
        wji_vs_trajcorr_markersize = 2
        
        shuffle_errwidth = 1.25
        shuffle_sigmarkersize = 1.5
        shuffle_figsize = (8, 1.5)
        shuffle_markersize = 5
        shuffle_linewidth = 1
        
        corr_marker_color = 'gray'
        
        traj_pos_sample_figsize = (1.75, 1.75)
        traj_vel_sample_figsize = (1.5  ,   1.5)
        traj_linewidth = 1
        traj_leadlag_linewidth = 2
        
        preferred_traj_linewidth = .5
        distplot_linewidth = 1
        lineplot_linewidth = 1
        lineplot_markersize = 5
        preferred_traj_figsize = (1.75, 1.75)
        
        weights_by_distance_figsize = (2.25, 2)
        aucScatter_figSize = (1.9, 1.9)
        FN_figsize = (2.75, 2.75)
        feature_corr_figSize = (1.9, 1.9)
        trajlength_figsize = (1.75, 1.75)
        pearsonr_histsize = (1.5, 1.5)
        distplot_figsize = (1.5, 1.0)
        stripplot_figsize = (5, 2)
        scatter_figsize = (1.75, 1.75)
        map_figSize = (3, 3)
        classifier_figSize = (3, 2)
        gas_figSize = (4,4)
        frate_figsize = weights_by_distance_figsize


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
        scatter_markersize = 30
        stripplot_markersize = 2
        feature_corr_markersize = 8
        wji_vs_trajcorr_markersize = 2
        
        shuffle_errwidth = 3
        shuffle_sigmarkersize = 3
        shuffle_figsize = (12, 3)
        
        traj_pos_sample_figsize = (4.5, 4.5)
        traj_vel_sample_figsize = (4, 4)
        traj_linewidth = 2
        traj_leadlag_linewidth = 3
        
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
        map_figSize = (6, 6)
        classifier_figSize = (2, 2.5)

        corr_marker_color = 'gray'
        

plt.rcParams['figure.dpi'] = plot_params.dpi
plt.rcParams['savefig.dpi'] = plot_params.dpi
plt.rcParams['font.family'] = 'Dejavu Sans'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.titlesize'] = 12
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
 
color1     = (  0/255, 141/255, 208/255)
color2     = (159/255, 206/255, 239/255)
spontColor = (183/255, 219/255, 165/255)        

# def standardize_plots(ax, figsize):
    
 
def add_icms_results_to_units_results_df(units_res, icms_res):

    thresh_list = [0 for idx in range(units_res.shape[0])]
    part_list   = [0 for idx in range(units_res.shape[0])]
    for idx, (x, y) in enumerate(zip(units_res['x'], units_res['y'])):
        chan_df = icms_res.loc[(icms_res['x'] == x) & (icms_res['y'] == y), :]
        thresh_list[idx] = chan_df['icms_threshold'].values[0]
        part_list  [idx] = chan_df['icms_bodypart'].values[0]   
    
    units_res['icms_threshold'] = thresh_list
    units_res['icms_bodypart']  = part_list
    
    return units_res

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

def summarize_model_results(units, lead_lag_keys):  
    
    if type(lead_lag_keys) != list:
        lead_lag_keys = [lead_lag_keys]
    
    for lead_lag_key in lead_lag_keys:
        
        if 'all_models_summary_results' in results_dict[lead_lag_key].keys():
            all_units_res = results_dict[lead_lag_key]['all_models_summary_results']
        else:
            all_units_res = units.copy()
            
        try:
            all_units_res.drop(columns=['spike_times', 'n_spikes'], inplace=True)
        except:
            pass
        
        try:
            units_res_add_stats = results_dict_add_stats[lead_lag_key]['all_models_summary_results']
            for col in units_res_add_stats.columns:
                if col not in all_units_res.columns:
                    all_units_res[col] = units_res_add_stats[col]
        except:
            pass
        
        
        for model_key in results_dict[lead_lag_key]['model_results'].keys():
            print(model_key)
            if 'shuffled_weights_FN' in model_key or 'shuffled_topology_FN' in model_key:
                col_names = ['%s_train_auc' % model_key]  
                results_keys = ['trainAUC']
            elif '%s_full_FN' % params.primary_traj_model == model_key or model_key == params.primary_traj_model:
                col_names = ['%s_auc' % model_key, '%s_train_auc' % model_key]  
                results_keys = ['AUC', 'trainAUC']   
            else:
                col_names = ['%s_auc' % model_key]  
                results_keys = ['AUC']  

            for col_name, results_key in zip(col_names, results_keys):                
                if col_name not in all_units_res.columns and results_key in results_dict[lead_lag_key]['model_results'][model_key].keys(): 
                    all_units_res[col_name] = results_dict[lead_lag_key]['model_results'][model_key][results_key].mean(axis=-1)
                else:
                    print('This model (%s, %s) has already been summarized in the all_models_summary_results dataframe' % (lead_lag_key, model_key))                    

        if 'cortical_area' not in all_units_res.keys():
            all_units_res = add_cortical_area_to_units_results_df(all_units_res, cortical_bounds=params.cortical_boundaries)

        results_dict[lead_lag_key]['all_models_summary_results'] = all_units_res.copy()
        
def plot_result_on_channel_map(units_res, jitter_radius = .15, hue_key='W_in', 
                               size_key = 'traj_avgPos_AUC',  title_type = 'size', style_key=None,
                               paperFig='unknown', hue_order = None, palette=None, sizes=None, s=None, gen_test_behavior=None):
    
    if hue_key[-4:] != '_auc' and f'{hue_key}_auc' in units_res.columns:
        hue_key = f'{hue_key}_auc'
    
    if size_key is not None:
        if size_key[-4:] != '_auc':
            size_key = size_key + '_auc'

    title_key = hue_key if title_type == 'hue' else size_key

    try:
        title = [f'{lab} AUC' for lab, key in zip(['Trajectory', 'Full_Kinematics', 'Velocity', 'Short_Kinematics', 'Kinematics_plus_reachFN', 'Kinematics_plus_spontaneousFN_Generalization'], 
                                                   ['traj', 'traj_avgPos', 'shortTraj', 'shortTraj_avgPos', 'traj_avgPos_reach_FN', 'traj_avgPos_spont_train_reach_test_FN']) if f'{key}_auc' == title_key][0]
    except:
        title = size_key
    
    distances = []
    for u1, unit1 in units_res.iterrows():
        for u2, unit2 in units_res.iterrows():
            distances.append( math.dist((unit1.x, unit1.y), (unit2.x, unit2.y)) )
    distance_mult = np.min([dist for dist in distances if dist != 0])
    
    jitter_radius = jitter_radius * distance_mult
    
    scatter_units_res = units_res.copy()
    scatter_units_res['scatter_x'] = np.full((scatter_units_res.shape[0],), np.nan)
    scatter_units_res['scatter_y'] = np.full((scatter_units_res.shape[0],), np.nan)
    for ch in np.unique(scatter_units_res.channel_index):
        chan_mask = scatter_units_res.channel_index == ch
        chan_units = scatter_units_res.loc[chan_mask, :]
        if len(chan_units) == 1:
            jitters = [(0, 0)]
        else:
            jitters = [(np.round(jitter_radius * np.cos(n*2*np.pi / len(chan_units)), 3), 
                        np.round(jitter_radius * np.sin(n*2*np.pi / len(chan_units)), 3)) for n in range(len(chan_units))]
        base_pos = chan_units.loc[:, ['x', 'y']]
        base_pos = np.array([base_pos['x'].values[0], base_pos['y'].values[0]])        
               
        scatter_units_res.loc[chan_mask, 'scatter_x'] = [jitter[0] + base_pos[0] for jitter in jitters]
        scatter_units_res.loc[chan_mask, 'scatter_y'] = [jitter[1] + base_pos[1] for jitter in jitters]
    
    fig, ax = plt.subplots(figsize=plot_params.map_figSize)
    sns.scatterplot(ax = ax, data = scatter_units_res, x = 'scatter_x', y = 'scatter_y', 
                    size = size_key, hue = hue_key, style = style_key, hue_order = hue_order,
                    palette=palette, sizes = sizes, s = s, legend=False)
    ax.vlines(np.arange(-0.5* distance_mult, 10.5* distance_mult, 1* distance_mult), -0.5* distance_mult, 9.5* distance_mult, colors='black')
    ax.hlines(np.arange(-0.5* distance_mult, 10.5* distance_mult, 1* distance_mult), -0.5* distance_mult, 9.5* distance_mult, colors='black')
    ax.set_xlim(-0.5* distance_mult, 9.5* distance_mult)
    ax.set_ylim(-0.5* distance_mult, 9.5* distance_mult)
    # for axis in ['bottom','left']:
    #     ax.spines[axis].set_linewidth(1)
    #     ax.spines[axis].set_color('black')
    # for axis in ['top','right']:
    #     ax.spines[axis].set_linewidth(0)
    # ax.tick_params(width=0, length = 0, labelsize = 0)
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_xlabel('Anterior'  , fontsize = plot_params.axis_fontsize, fontweight = 'bold')
    # ax.set_ylabel('Lateral', fontsize = plot_params.axis_fontsize, fontweight = 'bold')
    # ax.legend(bbox_to_anchor=(-.25, 1), loc='upper right', borderaxespad=0)
    # ax.get_legend().remove()
    ax.set_title('')

    ax.grid(False)

    # for txt, x, y, scat_x, scat_y in zip(scatter_units_res['ns6_elec_id'], scatter_units_res['center_x'], scatter_units_res['center_y'],
    #                      scatter_units_res['scatter_x'], scatter_units_res['scatter_y']):
    #     print((txt, x, y))
    #     ax.annotate('%d' % txt, (x, y))
    plt.show()
    
    fig_base = os.path.join(plots, paperFig)
    gen_test_label = '' if gen_test_behavior is None else gen_test_behavior
    fig.savefig(os.path.join(fig_base, f'{title}_{hue_key}_on_array_map_{gen_test_label}.png'), bbox_inches='tight', dpi=plot_params.dpi)    
              
def plot_model_auc_comparison(units_res, x_key, y_key, minauc = 0.5, maxauc = 1.0, hue_key='W_in', 
                              style_key='cortical_area', targets=None, col_key=None, hue_order=None, 
                              col_order=None, style_order=None, paperFig='unknown', asterisk='', 
                              palette=None, gen_test_behavior = None, extra_label='', full_test_behavior=None):
    
    if x_key[-4:] != '_auc':
        x_key = x_key + '_auc'
    if y_key[-4:] != '_auc':
        y_key = y_key + '_auc'
    
    try:
        xlabel = [f'{lab} AUC' for lab, key in zip(['Trajectory', 'Full kinematics', 'Velocity', 
                                                    'Short kinematics', 'Kinematics + reachFN', 
                                                    'Kinematics + spontaneousFN \ngeneralization',
                                                    'Kinematics + reachFN \nreverse generalization',
                                                    f'Kinematics + {full_test_behavior.lower()} FN \ngeneralization',
                                                    ], 
                                                   ['traj', 'traj_avgPos', 'shortTraj', 'shortTraj_avgPos', 
                                                    'traj_avgPos_reach_FN', 'traj_avgPos_spont_train_reach_test_FN',
                                                    'traj_avgPos_reach_train_spont_test_FN',
                                                    f'traj_avgPos_{gen_test_behavior}_train_reach_test_FN',
                                                    ]) if f'{key}_auc' == x_key][0]
    except:
        xlabel = 'Kinematics + spontaneousFN \ngeneralization' if x_key == 'traj_avgPos_spont_train_reach_test_FN_auc' else x_key
    
    try:
        ylabel = [f'{lab} AUC' for lab, key in zip(['Trajectory', 'Full kinematics', 'Velocity', 
                                                    'Short kinematics', 'Kinematics + reachFN', 
                                                    'Kinematics + spontaneousFN \ngeneralization',
                                                    'Kinematics + spontaneousFN'
                                                    ], 
                                                   ['traj', 'traj_avgPos', 'shortTraj', 'shortTraj_avgPos', 
                                                    'traj_avgPos_reach_FN', 'traj_avgPos_spont_train_reach_test_FN',
                                                    'traj_avgPos_spontaneous_FN'
                                                    ]) if f'{key}_auc' == y_key][0]
    except:
        ylabel = y_key
    
    sign_test, ttest = sig_tests(units_res, f'{y_key}', f'{x_key}', alternative='greater')
    
    units_res_plots = units_res.copy()
        
    if targets is not None:
        if type(targets) != list:
            targets = [targets]
        units_res_plots = isolate_target_units_for_plots(units_res_plots, targets)        
        plot_title = 'Targets:'
        plot_name = 'auc_{x_key}_{y_key}_targetUnits' % (x_key, y_key)
        for targ in targets:
            plot_title = plot_title + ' {targ},'
            plot_name = plot_name + '_{targ}'
    else:
        plot_title = 'Targets: All units'
        plot_name = f'auc_{x_key}_{y_key}'
    
    if hue_key is not None:
        plot_name += f'_hueKey_{hue_key}'

    if style_key is not None:
        plot_name += f'_styleKey_{style_key}'

    if col_key is not None:
        plot_name += f'_colKey_{col_key}'
        
        fig = sns.relplot(data = units_res_plots, x=x_key, y=y_key, hue=hue_key, 
                          col=col_key, style = style_key, kind='scatter', legend=True,
                          hue_order=hue_order, col_order=col_order, style_order=style_order)
        for ax, area in zip(fig.axes[0], ['M1', '3a', '3b']):
            ax.set_xlim(minauc, maxauc)
            ax.set_ylim(minauc, maxauc)
            ax.set_xticks(np.arange(np.ceil(minauc*10)/10, maxauc+.1, 0.1))
            ax.plot(np.arange(minauc, maxauc, 0.05), np.arange(minauc, maxauc, 0.05), '--k')
            # zordered_lines = ax.lines
            # zordered_collections = ax.collections
            # plt.setp(zordered_lines, zorder=1)
            # plt.setp(zordered_collections, zorder=1)
            # for l in ax.lines:
            #     if l not in zordered_lines:
            #         plt.setp(l, zorder=2)
            #         zordered_lines.append(l)
            # for c in ax.collections:
            #     if c not in zordered_collections:
            #         plt.setp(c, zorder=2)
            #         zordered_collections.append(c) 
            # for axis in ['bottom','left']:
            #     ax.spines[axis].set_linewidth(2)
            #     ax.spines[axis].set_color('black')
            # for axis in ['top','right']:
            #     ax.spines[axis].set_linewidth(0)
            # ax.tick_params(width=2, length = 4, labelsize = plot_params.tick_fontsize)
            ax.set_xlabel('ROC area (%s)' % x_key[:-4], fontsize = plot_params.axis_fontsize)
            ax.set_ylabel('ROC area (%s)' % y_key[:-4], fontsize = plot_params.axis_fontsize)
            # ax.set_xlabel('')
            # ax.set_ylabel('')
            ax.grid(False)
    else:
        fig, ax = plt.subplots(figsize = plot_params.aucScatter_figSize, dpi = plot_params.dpi)
        sns.scatterplot(ax = ax, data = units_res_plots, x = x_key, y = y_key, 
                        hue = hue_key, style = style_key, s = plot_params.scatter_markersize, 
                        legend=False, palette=palette,
                        hue_order=hue_order, style_order=style_order)     

        ax.plot(np.arange(minauc, maxauc, 0.05), np.arange(minauc, maxauc, 0.05), '--k', linewidth = plot_params.traj_linewidth)
        # ax.scatter(units_res_plots[x_key].to_numpy()[44] , units_res_plots[y_key].to_numpy()[44] , s = 60, c ='red', marker='x')
        # ax.scatter(units_res_plots[x_key].to_numpy()[107], units_res_plots[y_key].to_numpy()[107], s = 60,  c ='red', marker='o')
        ax.set_xlim(minauc, maxauc)
        ax.set_ylim(minauc, maxauc)
        # for axis in ['bottom','left']:
        #     ax.spines[axis].set_linewidth(plot_params.axis_linewidth)
        #     ax.spines[axis].set_color('black')
        # for axis in ['top','right']:
        #     ax.spines[axis].set_linewidth(0)
        # ax.tick_params(width=plot_params.tick_width, length = plot_params.tick_length*2, labelsize = plot_params.tick_fontsize)
        ax.set_xlabel(xlabel, fontsize = plot_params.axis_fontsize)
        ax.set_ylabel(ylabel, fontsize = plot_params.axis_fontsize)
        # ax.set_xlabel('')
        # ax.set_ylabel('')
        # ax.set_title(plot_title)
        ax.set_title('')
        
        if gen_test_behavior in params.reach_specific_thresh.keys():
            ax.plot(np.arange(minauc, maxauc, 0.05), np.arange(minauc, maxauc, 0.05) + params.reach_specific_thresh[gen_test_behavior]*np.sqrt(2), 
                    color = 'black', linestyle='dotted', linewidth = plot_params.traj_linewidth)
    
        if sign_test.pvalue<0.01:
            text = f'p < 0.01{asterisk}'
        else:
            text = f'p = {np.round(sign_test.pvalue, 4)}{asterisk}'
        if y_key == 'traj_avgPos_reach_FN_auc':
            text_y = 1.0*maxauc 
        else:
            text_y = 0.95*maxauc
        ax.text(minauc+(maxauc-minauc)*0.5, text_y, text, horizontalalignment='center', fontsize = plot_params.tick_fontsize)
    # ax.legend(bbox_to_anchor=(1.5, 1.5), loc='center left', borderaxespad=0)
    plt.show()
    
    if paperFig == 'unknown':
        fig_base = os.path.join(plots, paperFig, 'auc_comparison')
    else:
        fig_base = os.path.join(plots, paperFig)
    
    if asterisk != '':
        plot_name += '_filtered'
    if gen_test_behavior.lower()=='spont':
        plot_name += f'_coloredBy_{params.test_behavior}'

    fig.savefig(Path(fig_base) / f'{plot_name}{extra_label}.png', bbox_inches='tight', dpi=plot_params.dpi)

    return sign_test

def isolate_target_units_for_plots(units_res, targets):
    
    for targ in targets:
        if targ.lower()=='motor':
            units_res = units_res.loc[(units_res['cortical_area'] == 'M1') | (units_res['cortical_area'] == '6Dc'), :]
        elif targ.lower()=='sensory':
            units_res = units_res.loc[(units_res['cortical_area'] == '3a') | (units_res['cortical_area'] == '3b'), :]
        elif targ.lower() in ['3a', '3b']:
            units_res = units_res.loc[units_res['cortical_area'] == targ, :]
        elif targ.lower() == 'tuned':
            units_res = units_res.loc[units_res['proportion_sign'] >= params.significant_proportion_thresh, :]
        elif targ.lower() == 'untuned':
            num_tuned_units = np.sum(units_res['proportion_sign'] >= params.significant_proportion_thresh)
            units_res = units_res.sort_values(by='proportion_sign', ascending = True)
            units_res = units_res.iloc[:num_tuned_units, :]
            
    return units_res

def sort_units_by_area_for_FN_plots(units_res):
    ticks  = []
    labels = []
    units_sorted = pd.DataFrame() 
    for area in ['3b', '3a', 'M1', '6dc']: 
        tmp_df = units_res.copy().loc[units_res['cortical_area']==area, :]
        tmp_df.sort_values(by='W_in', inplace=True)
        units_sorted = pd.concat((units_sorted, tmp_df), axis=0, ignore_index=False)
        if tmp_df.size > 0:
            area_idxs = np.where(units_sorted['cortical_area']==area)
            ticks.extend ([np.mean(area_idxs), np.max(area_idxs)])
            labels.extend([area, ''])
            
    ticks  = ticks[:-1]
    labels = labels[:-1]
    
    return units_sorted, ticks, labels

def sort_units_by_column_for_FN_plots(units_res):
    ticks  = []
    labels = []
    units_sorted = pd.DataFrame() 
    
    tmp_df = units_res.copy().loc[units_res['x']<params.motor_bound, :]
    tmp_df.sort_values(by='W_in', inplace=True)
    units_sorted = pd.concat((units_sorted, tmp_df), axis=0, ignore_index=False)
    col_idxs = np.where(units_sorted['x']<params.motor_bound)
    ticks.extend ([np.mean(col_idxs), np.max(col_idxs)+1])
    labels.extend([params.motor_bound_names[0], ''])
    
    tmp_df = units_res.copy().loc[units_res['x']>params.motor_bound, :]
    tmp_df.sort_values(by='W_in', inplace=True)
    units_sorted = pd.concat((units_sorted, tmp_df), axis=0, ignore_index=False)
    col_idxs = np.where(units_sorted['x']>params.motor_bound)
    ticks.extend ([np.mean(col_idxs), np.max(col_idxs)+1])
    labels.extend([params.motor_bound_names[1], ''])
    
    ticks  = ticks[:-1]
    labels = labels[:-1]
    
    # if params.flip_order_area:
    #     tmp_df0 = units_sorted.loc[units_sorted['x']>params.motor_bound].iloc[::-1]
    #     tmp_df1 = units_sorted.loc[units_sorted['x']<params.motor_bound].iloc[::-1]
    #     units_sorted = pd.concat((tmp_df1, tmp_df0), axis=0, ignore_index=False)
        
    #     ticks = [units_sorted.shape[0]-1 - tick for tick in ticks]
    
    return units_sorted, ticks, labels

def plot_covariability_speed_and_FNs(single_reaches_FN_dict, kin_df, units_res, reach_specific_units, non_specific_units, paperFig=None):
    df = pd.DataFrame()
    FG_list = []
    for key, FN in single_reaches_FN_dict.items():
        if 'half' in key:
            continue
        reachIdx = int(key.split('reach')[-1])
        
        csFN = FN[np.ix_(reach_specific_units, reach_specific_units)]
        ciFN = FN[np.ix_(  non_specific_units,   non_specific_units)]
        reach_kin_df = kin_df.loc[kin_df['reach'] == reachIdx, 'speed']
        
        for tmp_FN, fnKey in zip([FN, csFN, ciFN], ['Full', 'Context-Specific', 'Context-Invariant']):
            tmp_df = pd.DataFrame(data=np.array([[tmp_FN.mean(),
                                                 np.percentile(tmp_FN, 25),
                                                 np.percentile(tmp_FN, 50),
                                                 np.percentile(tmp_FN, 75),
                                                 np.percentile(tmp_FN, 90),
                                                 np.percentile(reach_kin_df, 50),
                                                 reachIdx,
                                                 ]]),
                                  columns = ['Wmean', 'W25', 'W50', 'W75', 'W90', 'medSpeed', 'reachNum'])
            df = pd.concat((df, tmp_df))
            FG_list.append(fnKey)
    
    df['FG'] = FG_list
    
    fig, ax = plt.subplots(figsize = (4, 4), dpi=plot_params.dpi)
    sns.scatterplot(data=df, ax=ax, y='W90', x='medSpeed', hue='FG', s=5)
                

def plot_functional_networks(FN, units_res, FN_key = 'split_reach_FNs', 
                             cmin=None, cmax=None, subset_idxs = None, subset_type='both', 
                             paperFig='unknown', gen_test_behavior=None, kin_df = None):
    
    # units_sorted = units_res.copy()    
    # units_sorted.sort_values(by='cortical_area', inplace=True, ignore_index=False)
    # units_sorted, ticks, labels = sort_units_by_area_for_FN_plots(units_res.copy())
    units_sorted, ticks, labels = sort_units_by_column_for_FN_plots(units_res.copy())
                          
    if subset_idxs is not None:
        units_res_subset = units_res.copy().loc[subset_idxs, :]
        # units_sorted_subset, ticks_subset, labels_subset = sort_units_by_area_for_FN_plots(units_res_subset.copy()) 
        units_sorted_subset, ticks_subset, labels_subset = sort_units_by_column_for_FN_plots(units_res_subset.copy()) 

    if FN.ndim < 3:
        FN = np.expand_dims(FN, axis = 0)
    
    if cmin is None:
        net_min = []
        net_max = []
        for network in FN:
            net_min.append(np.nanmin(network))
            net_max.append(np.nanmax(network))
        cmin = min(net_min)
        cmax = max(net_max)
    
    if FN_key == 'split_reach_FNs':
        titles = ['reachFN1', 'reachFN2']
    elif FN_key == 'spontaneous_FN':
        titles = ['spontaneousFN']
    else:
        titles = [FN_key]
    
    FNs_to_save = []
    for network, title in zip(FN, titles):
        network_copy = network.copy()
        
        fsize = plot_params.FN_figsize
        
        
        if subset_idxs is not None:
            # if subset_idxs.size > FN.shape[-1]/2:
            #     title += ' Non'
                # fsize = (fsize[0]*network_copy.shape[1]/network.shape[1], fsize[1]*network_copy.shape[0]/network.shape[0])
            # else:
                # fsize = (fsize[0]*np.log(network_copy.shape[1])/np.log(plot_params.mostUnits_FN), fsize[1]*np.log(network_copy.shape[0])/np.log(plot_params.mostUnits_FN))  
        
            # size_mod = np.log(len(subset_idxs))/np.log(plot_params.mostUnits_FN) 
            size_mod = (len(subset_idxs))**(1/3)/(plot_params.mostUnits_FN)**(1/3)
            fsize = (fsize[0]*size_mod, fsize[1]*size_mod)  
            
            if subset_type == 'both':
                if subset_idxs.size > FN.shape[-1]/2:
                    title = 'Context-invariant\n' + title
                else:
                    title = 'Context-specific\n' + title
                target_idx, source_idx = units_sorted_subset.index.values, units_sorted_subset.index.values 
                xticks = ticks_subset
                yticks = ticks_subset
                xlabels = labels_subset
                ylabels = labels_subset
            elif subset_type == 'target':
                title += ' Reach Specific Targets'
                target_idx, source_idx = units_sorted_subset.index.values, units_sorted.index.values  
                xticks = ticks
                yticks = ticks_subset
                xlabels = labels
                ylabels = labels_subset
            elif subset_type == 'source':
                title += f' Reach Specific Sources'
                target_idx, source_idx = units_sorted.index.values, units_sorted_subset.index.values  
                xticks = ticks_subset
                yticks = ticks
                xlabels = labels_subset
                ylabels = labels
        else:
            target_idx, source_idx = units_sorted.index.values, units_sorted.index.values  
            xticks = ticks
            yticks = ticks
            xlabels = labels
            ylabels = labels
            # fsize = (fsize[0]*np.log(network_copy.shape[1])/np.log(plot_params.mostUnits_FN), fsize[1]*np.log(network_copy.shape[0])/np.log(plot_params.mostUnits_FN))
            fsize = (fsize[0]*network_copy.shape[1]/plot_params.mostUnits_FN, fsize[1]*network_copy.shape[0]/plot_params.mostUnits_FN)


        network_copy = network_copy[np.ix_(target_idx, source_idx)]
        
        if kin_df is not None:
            fig, (ax, ax1) = plt.subplots(2, 1, figsize=(2, 4), dpi = plot_params.dpi, gridspec_kw={'height_ratios': [4, 1]})
            
            sns.heatmap(network_copy,ax=ax,cmap= 'viridis',square=True, norm=colors.PowerNorm(0.5, vmin=cmin, vmax=cmax)) # norm=colors.LogNorm(vmin=cmin, vmax=cmax)
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.set_xticklabels(xlabels)
            ax.set_yticklabels(ylabels)
            ax.set_title(title, fontsize=plot_params.axis_fontsize)
            ax.set_ylabel('Target unit', fontsize=plot_params.axis_fontsize)
            ax.set_xlabel('Source unit' , fontsize=plot_params.axis_fontsize)
            
            # key = kin_df.name
            # kin_df = pd.DataFrame(data=zip(kin_df.values, kin_df.index), columns=['speed', 'frame'])
            ax1.plot(np.arange(0, kin_df.size) / plot_params.fps *1e3, kin_df.values)
            # sns.kdeplot(data=kin_df, ax=ax1, legend = False, linewidth = 2, common_norm=False, bw_adjust=0.5)
            # if key in ['vx', 'vy', 'vz',]:
            #     ax1.set_xlim(-75, 75)
            # elif key in ['x', 'y', 'z',]:
            #     ax1.set_xlim(-10, 10)
            # elif key == 'speed':
            #     ax1.set_xlim(0, 100)
            # ax1.set_yticks([])
            # ax1.set_xlabel(key, fontsize=plot_params.axis_fontsize)
            title = title.replace('\n', '_')
            ax1.set_ylabel('Speed (cm/s)')
            ax1.set_xlabel('Time (ms)')
            ax1.set_ylim(0, 100)
            
            fig.savefig(os.path.join(plots, paperFig, 
                                     f'{marmcode}_functional_network_{title.replace(" ", "_").replace("-", "_")}.png'), bbox_inches='tight', dpi=plot_params.dpi)
            
            plt.show()
            
        else:
            fig, ax = plt.subplots(figsize=fsize, dpi = plot_params.dpi)
            
            sns.heatmap(network_copy,ax=ax,cmap= 'viridis',square=True, norm=colors.PowerNorm(0.5, vmin=cmin, vmax=cmax)) # norm=colors.LogNorm(vmin=cmin, vmax=cmax)
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.set_xticklabels(xlabels)
            ax.set_yticklabels(ylabels)
            ax.set_title(title, fontsize=plot_params.axis_fontsize)
            ax.set_ylabel('Target unit', fontsize=plot_params.axis_fontsize)
            ax.set_xlabel('Source unit' , fontsize=plot_params.axis_fontsize)
            plt.show()
            
            title = title.replace('\n', '_')
            gen_test_label = '' if gen_test_behavior is None else gen_test_behavior
            fig.savefig(os.path.join(plots, paperFig, 
                                     f'{marmcode}_functional_network_{gen_test_label}_{title.replace(" ", "_").replace("-", "_")}.png'), bbox_inches='tight', dpi=plot_params.dpi)
            
            plt.show()
    
        # plt.hist(network_copy.flatten(), bins = 30)
        # plt.show()
        
        FNs_to_save.append(network_copy)
        
    return FNs_to_save, cmin, cmax
    # np.correlate(FN[0].flatten(), FN[1].flatten())

def sig_tests(unit_info, y_key, x_key, alternative='greater', unit_info_reduced = None):
    
    if unit_info_reduced is None:
        nY = np.sum(unit_info[y_key] > unit_info[x_key])
        nUnits = np.shape(unit_info)[0]
        
        sign_test = binomtest(nY, nUnits, p = 0.5, alternative=alternative)
        
        ttest_paired = ttest_rel(unit_info[y_key], unit_info[x_key], alternative=alternative)

    else:
        nPathlet = np.sum(unit_info.pathlet_AUC > unit_info_reduced.pathlet_AUC)
        nUnits = np.shape(unit_info)[0]
        sign_test = binomtest(nPathlet, nUnits, p = 0.5, alternative=alternative)
        ttest_paired = ttest_rel(unit_info.pathlet_AUC, unit_info_reduced.pathlet_AUC, alternative=alternative)

    return sign_test, ttest_paired

def plot_weights_versus_interelectrode_distances(FN, spontaneous_FN, electrode_distances, annotated_FN_dict=None, extend_retract_FNs=None,
                                                 ymin=None, ymax=None, paperFig='Fig1', palette=None):
    
    fig, ax = plt.subplots(figsize=plot_params.weights_by_distance_figsize, dpi=plot_params.dpi)

    reach_labels = ['reachFN1', 'reachFN2']
    weights_df = pd.DataFrame()
    for weights, reach_label in zip(FN, reach_labels):  
    
        tmp_df = pd.DataFrame(data = zip(weights.flatten(), electrode_distances.flatten(), [reach_label]*weights.size), 
                              columns = ['Wji', 'Distance', 'Reaches'])
        weights_df = pd.concat((weights_df, tmp_df), axis=0, ignore_index=True)
    
    tmp_df = pd.DataFrame(data = zip(spontaneous_FN.flatten(), electrode_distances.flatten(), ['spontFN']*spontaneous_FN.size), 
                          columns = ['Wji', 'Distance', 'Reaches'])
    weights_df = pd.concat((weights_df, tmp_df), axis=0, ignore_index=True)
    
    if type(annotated_FN_dict) == dict:
        behavior_df = weights_df.copy()
        ext_ret_df = weights_df.copy()
        for behavior, behavior_FN in annotated_FN_dict.items():
            # if behavior not in ['Unknown', 'Apparatus']:
            if behavior in ['Rest', 'Locomotion']:    

                tmp_df = pd.DataFrame(data = zip(behavior_FN.flatten(), electrode_distances.flatten(), [behavior]*behavior_FN.size),
                                      columns = ['Wji', 'Distance', 'Reaches'])
                behavior_df = pd.concat((behavior_df, tmp_df), axis=0, ignore_index=True)

            if behavior in ['Extension', 'Retraction']:
                tmp_df = pd.DataFrame(data = zip(behavior_FN.flatten(), electrode_distances.flatten(), [behavior]*behavior_FN.size),
                                      columns = ['Wji', 'Distance', 'Reaches'])
                ext_ret_df = pd.concat((ext_ret_df, tmp_df), axis=0, ignore_index=True)

        fig2, ax2 = plt.subplots(figsize=plot_params.weights_by_distance_figsize, dpi=plot_params.dpi)                
        sns.lineplot(ax = ax2, data=behavior_df, x='Distance', y='Wji', hue='Reaches', 
                     err_style="bars", errorbar=("se", 1), linewidth=plot_params.lineplot_linewidth,
                     marker='o', markersize=plot_params.lineplot_markersize, palette=palette['spont'], legend=True,
                     hue_order=['reachFN1', 'reachFN2', 'spontFN', 'Rest', 'Locomotion'])
        sns.move_legend(ax2,  "upper left", bbox_to_anchor=(1, 1))
        ax2.set_ylabel('$W_{{ji}}$ (mean \u00B1 sem)')
        ax2.set_xlabel('Inter-unit distance (\u03bcm)')            

        if ymin:
            ax2.set_ylim(ymin, ymax)
        else:
            ymin, ymax = ax2.get_ylim()
        plt.show()
        
        fig2.savefig(os.path.join(plots, paperFig, f'{marmcode}_weights_by_distance_Rest_Locomotion.png'), bbox_inches='tight', dpi=plot_params.dpi)

        fig3, ax3 = plt.subplots(figsize=plot_params.weights_by_distance_figsize, dpi=plot_params.dpi)
        sns.lineplot(ax = ax3, data=ext_ret_df, x='Distance', y='Wji', hue='Reaches', 
                     err_style="bars", errorbar=("se", 1), linewidth=plot_params.lineplot_linewidth,
                     marker='o', markersize=plot_params.lineplot_markersize, palette=palette['ext_ret'], legend=True,
                     hue_order=['reachFN1', 'reachFN2', 'spontFN', 'Retraction', 'Extension'])
        sns.move_legend(ax3,  "upper left", bbox_to_anchor=(1, 1))
        ax3.set_ylabel('$W_{{ji}}$ (mean \u00B1 sem)')
        ax3.set_xlabel('Inter-unit distance (\u03bcm)')            
        if ymin:
            ax3.set_ylim(ymin, ymax)
        else:
            ymin, ymax = ax3.get_ylim()
        plt.show()
        fig3.savefig(os.path.join(plots, paperFig, f'{marmcode}_weights_by_distance_Extension_Retraction.png'), bbox_inches='tight', dpi=plot_params.dpi)


    # elif extend_retract_FNs:
    #     ext_ret_df = weights_df.copy()
    #     for label, seg_FN in zip(['Extension', 'Retraction'], extend_retract_FNs):
    #         tmp_df = pd.DataFrame(data = zip(seg_FN.flatten(), electrode_distances.flatten(), [label]*seg_FN.size),
    #                               columns = ['Wji', 'Distance', 'Reaches'])
    #         ext_ret_df = pd.concat((ext_ret_df, tmp_df), axis=0, ignore_index=True)
            
    #     fig2, ax2 = plt.subplots(figsize=plot_params.weights_by_distance_figsize, dpi=plot_params.dpi)
    #     sns.lineplot(ax = ax2, data=ext_ret_df, x='Distance', y='Wji', hue='Reaches', 
    #                  err_style="bars", errorbar=("se", 1), linewidth=plot_params.lineplot_linewidth,
    #                  marker='o', markersize=plot_params.lineplot_markersize, palette=palette, legend=True)
    #     sns.move_legend(ax2,  "upper left", bbox_to_anchor=(1, 1))
    #     ax.set_ylabel('$W_{{ji}}$ (mean \u00B1 sem)')
    #     ax.set_xlabel('Inter-unit distance (\u03bcm)')            
    #     if ymin:
    #         ax.set_ylim(ymin, ymax)
    #     else:
    #         ymin, ymax = ax2.get_ylim()
    #     sns.despine(ax=ax2)
    #     plt.show()
    #     fig2.savefig(os.path.join(plots, paperFig, f'{marmcode}_weights_by_distance_{label}.png'), bbox_inches='tight', dpi=plot_params.dpi)

        
    else:
        sns.lineplot(ax = ax, data=weights_df, x='Distance', y='Wji', hue='Reaches', 
                     err_style="bars", errorbar=("se", 1), linewidth=plot_params.lineplot_linewidth,
                     marker='o', markersize=plot_params.lineplot_markersize, palette=palette, legend=False)
        # ax.set_ylabel(f'Wji (mean %s sem)' % '\u00B1', fontsize = plot_params.axis_fontsize)
        ax.set_ylabel('$W_{{ji}}$ (mean \u00B1 sem)')
        # ax.set_xlabel('Inter-Unit Distance (%sm)' % '\u03bc', fontsize = plot_params.axis_fontsize)
        ax.set_xlabel('Inter-unit distance (\u03bcm)')
    
        if ymin:
            ax.set_ylim(ymin, ymax)
        else:
            ymin, ymax = ax.get_ylim()
    
        # ax.legend(bbox_to_anchor=(1.1, 0.5), loc='center left', 
        #           labels  = weights_df['Reaches'].unique(), 
        #           title_fontsize = plot_params.axis_fontsize,
        #           fontsize = plot_params.tick_fontsize)
        sns.despine(ax=ax)
        plt.show()
        
        fig.savefig(os.path.join(plots, paperFig, f'{marmcode}_weights_by_distance.png'), bbox_inches='tight', dpi=plot_params.dpi)
    
    return ymin, ymax          

def add_in_weight_to_units_df(units_res, FN):
    
    units_res = units_res.copy()
    
    idx_motor   = np.where(units_res['cortical_area'] == 'Motor')[0]
    idx_sensory = np.where(units_res['cortical_area'] == 'Sensory')[0]
    
    in_weights         = []
    in_weights_motor   = []
    in_weights_sensory = []
    out_weights        = []
    for unit_idx in units_res.index:
        if FN.ndim == 3:
            w_in  = (FN[0, unit_idx].sum() + FN[1, unit_idx].sum())/2 / FN.shape[-1]
            w_out = (FN[0, :, unit_idx].sum() + FN[1, :, unit_idx].sum())/2 / FN.shape[-1]            
            w_in_motor   = (FN[0, unit_idx, idx_motor  ].sum() + FN[1, unit_idx, idx_motor  ].sum())/2 / len(idx_motor)
            w_in_sensory = (FN[0, unit_idx, idx_sensory].sum() + FN[1, unit_idx, idx_sensory].sum())/2 / len(idx_sensory)
        else:
            w_in  = FN[unit_idx].sum()
            w_out = FN[:, unit_idx].sum()
        in_weights.append(w_in)
        in_weights_motor.append  (w_in_motor)
        in_weights_sensory.append(w_in_sensory)
        out_weights.append(w_out)
    
    tmp_df = pd.DataFrame(data = zip(in_weights, out_weights, in_weights_motor, in_weights_sensory),
                          columns = ['W_in',     'W_out',     'W_in_motor',     'W_in_sensory'],
                          index = units_res.index)

    units_res = pd.concat((units_res, tmp_df), axis = 1)
    
    return units_res

def add_generalization_experiments_to_units_df(units_res, gen_res):
    
    for model_key in gen_res['model_results'].keys():
        if 'reach_test_FN' in model_key:
            units_res[f'{model_key}_auc'] = gen_res['model_results'][model_key]['AUC'].mean(axis=-1)
    
    return units_res                                               

def add_modulation_data_to_units_df(units_res, paperFig, behavior_name_map):

    with open(f'{modulation_base}_modulationData.pkl', 'rb') as f:
        modulation_df = dill.load(f)     
    mask = [True if int(uName) in units_res.unit_name.astype(int).values else False for uName in modulation_df.unit_name.values]
    modulation_df = modulation_df.loc[mask, :]
    modulation_df.reset_index(drop=True, inplace=True)

    
    try:
        average_rates_df = pd.read_hdf(nwb_infile.parent / f'{nwb_infile.stem}_combined_average_firing_rates.h5', 'frates') 
    except:
        with open(f'{modulation_base}_average_firing_rates.pkl', 'rb') as f:
            average_rates_df = dill.load(f)     

    for met in modulation_df.columns[:6]:
        units_res[met] = modulation_df[met]

    for col in average_rates_df.columns:
        behavior = behavior_name_map[col] if col in behavior_name_map.keys() else col 
        units_res[f'{behavior}_frate'] = average_rates_df[col]
    # units_res['reach_frate'] = average_rates_df['Reach']
    # units_res['spont_frate'] = average_rates_df['Spontaneous']
    units_res['percent_frate_increase'] = average_rates_df['Reach'] / average_rates_df['Spontaneous']

    plot_average_frates_simple(average_rates_df, 'FigS8_and_9', 
                               to_plot=['Reach', 'Spontaneous', 'Rest', 'Locomotion', 'Retraction', 'Extension',],
                               behavior_name_map = behavior_name_map,
                               palette='frate_palette')

    return units_res    

def evaluate_effect_of_network_shuffles(lead_lag_key, comparison_model, 
                                        kin_only_model=None, all_samples=False, 
                                        targets = None, ylim=(0,50), paperFig='unknown',
                                        plot_difference=True, alpha = 0.01, palette=None):
    
    percentPattern       = re.compile('[0-9]{1,3}_percent')
    shuffleModePattern   = re.compile('shuffled_[a-zA-Z]*')
    shuffleMetricPattern = re.compile('by_[a-zA-Z]*') 
    
    results_key = 'trainAUC'
    
    comparison_all_units_auc = results_dict[lead_lag_key]['model_results'][comparison_model][results_key].copy()
    units_res_tmp = results_dict[params.best_lead_lag_key]['all_models_summary_results']

    if kin_only_model is not None and results_key in results_dict[lead_lag_key]['model_results'][kin_only_model].keys():
        kin_only_all_units_auc   = results_dict[lead_lag_key]['model_results'][kin_only_model][results_key].copy()

    if targets is not None:
        if type(targets) != list:
            targets = [targets]
        units_res_tmp = isolate_target_units_for_plots(units_res_tmp, targets)        
        target_string = targets[0]
        for targ in targets[1:]:
            target_string = f'{target_string}_and_{targ}' 
    else:
        target_string = 'All_Units'
    
    target_idxs = units_res_tmp.index.values
    
    train_auc_df = pd.DataFrame()
    full_minus_kin_loss = np.array([])
    for model_key in results_dict[lead_lag_key]['model_results'].keys():
        if 'shuffled' in model_key and comparison_model == model_key.split('_shuffled')[0]:
            results_key = 'trainAUC'
            percent     = int(re.findall(percentPattern, model_key)[0].split('_percent')[0])
            shuf_mode   = re.findall(shuffleModePattern, model_key)[0].split('shuffled_')[-1]
            shuf_metric = re.findall(shuffleMetricPattern, model_key)[0].split('by_')[-1]
            
            if percent in [1, 100]:
                continue
            
            shuffle_auc = results_dict[lead_lag_key]['model_results'][model_key][results_key].copy()
            
            
            comparison_auc = comparison_all_units_auc[target_idxs, :shuffle_auc.shape[1]]
            shuffle_auc    = shuffle_auc             [target_idxs, :]
            if kin_only_model is not None and results_key in results_dict[lead_lag_key]['model_results'][kin_only_model].keys():
                kin_only_auc = kin_only_all_units_auc  [target_idxs, :shuffle_auc.shape[1]]                
                # auc_loss     = np.divide(comparison_auc - shuffle_auc, comparison_auc - kin_only_auc) * 100
                auc_loss     = np.divide(comparison_auc - shuffle_auc, comparison_auc - 0.5) * 100
                
                if shuf_mode == 'weights':
                    full_minus_kin = np.divide(comparison_auc - kin_only_auc, comparison_auc - 0.5) * 100
                    full_minus_kin_loss = np.hstack((full_minus_kin_loss, np.nanmean(full_minus_kin, axis=1)))
            else:
                auc_loss    = np.divide(comparison_auc - shuffle_auc, comparison_auc - 0.5) * 100
            
            unit_list    = []
            percent_list = []
            mode_list    = []
            metric_list  = []
            
            if all_samples:
                sample_list  = []
                for unit in range(auc_loss.shape[0]):
                    for sample in range(auc_loss.shape[1]):
                        unit_list.append(unit)
                        sample_list.append(sample)
                        percent_list.append(percent)
                        mode_list.append(shuf_mode)
                        metric_list.append(shuf_metric)
                        
                tmp_df = pd.DataFrame(data = zip(auc_loss.flatten(), unit_list, sample_list, percent_list, mode_list, metric_list, full_minus_kin_loss.flatten()), 
                                      columns = ['auc_loss (%)', 'unit', 'sample', 'percent', 'mode', 'metric', 'full_minus_kin_auc_loss'])
            else:
                for unit in range(auc_loss.shape[0]):
                    unit_list.append(unit)
                    percent_list.append(percent)
                    mode_list.append(shuf_mode)
                    metric_list.append(shuf_metric)

                tmp_df = pd.DataFrame(data = zip(auc_loss.mean(axis=1), unit_list, percent_list, mode_list, metric_list), 
                                      columns = ['auc_loss (%)', 'unit', 'percent', 'mode', 'metric'])
            
            train_auc_df = pd.concat((train_auc_df, tmp_df), axis = 0, ignore_index=True)
            
    tmp_df_strength = train_auc_df.loc[train_auc_df['metric'] == 'strength', :]            
    tmp_df_random = train_auc_df.loc[train_auc_df['metric'] == 'random', :]

    tmp_df_diff = tmp_df_random.copy()  
    tmp_df_diff['auc_loss (%)'] = tmp_df_strength['auc_loss (%)'].to_numpy() - tmp_df_random['auc_loss (%)'].to_numpy()
    tmp_df_diff['metric'] = np.full((tmp_df_diff.shape[0],), 'difference')      

    if plot_difference:
        train_auc_df = pd.concat((train_auc_df, tmp_df_diff), axis = 0, ignore_index=True)    

    tmp_df2 = tmp_df_random.copy()  
    tmp_df2['auc_loss (%)'] = full_minus_kin_loss
    tmp_df2['metric'] = np.full((tmp_df_diff.shape[0],), 'fullKinFN_minus_fullKin')
    percent_vals = np.unique(train_auc_df['percent'])
    # tmp_df2['percent'] = [np.where(percent_vals == per)[0][0] for per in tmp_df2['percent']]
    # train_auc_df = pd.concat((train_auc_df, tmp_df_diff), axis = 0, ignore_index=True)    

    # fig = sns.catplot(data = train_auc_df, x='percent', y='auc_loss (%)', col='mode', hue='metric', kind='point', legend=True, errorbar='se')
    
    sig_strength_v_random = dict(weights=[], topology=[])
    for mode in train_auc_df['mode'].unique(): 
        for idx, percent in enumerate(train_auc_df['percent'].unique()):    
            stats_df = train_auc_df.loc[(train_auc_df['mode'] == mode) & (train_auc_df['percent'] == percent), :]
            nStrength = np.sum(stats_df.loc[stats_df['metric'] == 'strength', 'auc_loss (%)'].values > 
                               stats_df.loc[stats_df['metric'] == 'random', 'auc_loss (%)'].values)
            nUnits = np.sum(stats_df['metric'] == 'strength')
            
            sign_test = binomtest(nStrength, nUnits, p = 0.5, alternative='greater')
            print(f'Strength != Random, {percent}%, {mode}:  p={np.round(sign_test.pvalue, 4)}, nStrength={sign_test.k}, nUnits={sign_test.n}, prop={np.round(sign_test.proportion_estimate, 2)}')

            if sign_test.pvalue < alpha:
                sig_strength_v_random[mode].append(percent)

    sig_noFN_v_strength = dict(weights=[], topology=[])
    for mode in train_auc_df['mode'].unique(): 
        for idx, percent in enumerate(train_auc_df['percent'].unique()):    
            stats_df = train_auc_df.loc[(train_auc_df['mode'] == mode) & (train_auc_df['percent'] == percent), :]
            kin_comp_stats_df = tmp_df2.loc[(tmp_df2['mode'] == mode) & (tmp_df2['percent'] == 5), :]
            nStrength = np.sum(stats_df.loc[stats_df['metric'] == 'strength', 'auc_loss (%)'].values > kin_comp_stats_df['auc_loss (%)'].values)
            nUnits = np.sum(stats_df['metric'] == 'strength')
            
            sign_test = binomtest(nStrength, nUnits, p = 0.5, alternative='two-sided')
            print(f'Strength != No-FN, {percent}%, {mode}:  p={np.round(sign_test.pvalue, 4)}, nStrength={sign_test.k}, nUnits={sign_test.n}, prop={np.round(sign_test.proportion_estimate, 2)}')

            if sign_test.pvalue < alpha:
                sig_noFN_v_strength[mode].append(percent)

    sig_noFN_v_random = dict(weights=[], topology=[])
    for mode in train_auc_df['mode'].unique(): 
        for idx, percent in enumerate(train_auc_df['percent'].unique()):   
            stats_df = train_auc_df.loc[(train_auc_df['mode'] == mode) & (train_auc_df['percent'] == percent), :]
            kin_comp_stats_df = tmp_df2.loc[(tmp_df2['mode'] == mode) & (tmp_df2['percent'] == 5), :]
            nStrength = np.sum(stats_df.loc[stats_df['metric'] == 'random', 'auc_loss (%)'].values > kin_comp_stats_df['auc_loss (%)'].values)
            nUnits = np.sum(stats_df['metric'] == 'random')
            
            sign_test = binomtest(nStrength, nUnits, p = 0.5, alternative='two-sided')
            print(f'Random != No-FN, {percent}%, {mode}:  p={np.round(sign_test.pvalue, 4)}, nRandom={sign_test.k}, nUnits={sign_test.n}, prop={np.round(sign_test.proportion_estimate, 2)}')
    
            if sign_test.pvalue < alpha:
                sig_noFN_v_random[mode].append(percent)

    sig_strength_v_fullShuffle = dict(weights=[], topology=[])
    for mode in train_auc_df['mode'].unique(): 
        for idx, percent in enumerate(train_auc_df['percent'].unique()):    
            stats_df       = train_auc_df.loc[(train_auc_df['metric'] == 'strength') & (train_auc_df['mode'] == mode) & (train_auc_df['percent'] == percent), :]
            fullShuffle_df = train_auc_df.loc[(train_auc_df['metric'] == 'strength') & (train_auc_df['mode'] == mode) & (train_auc_df['percent'] == 90), :]
            nFullShuffle = np.sum(fullShuffle_df['auc_loss (%)'].values > stats_df['auc_loss (%)'].values)
            nUnits = stats_df.shape[0]
            
            sign_test = binomtest(nFullShuffle, nUnits, p = 0.5, alternative='greater')
            print(f'fullShuffle > partialShuffle, {percent}%, {mode}:  p={np.round(sign_test.pvalue, 6)}, nStrength={sign_test.k}, nUnits={sign_test.n}, prop={np.round(sign_test.proportion_estimate, 2)}')

            # if sign_test.proportion_estimate > 0.77:
            #     sig_strength_v_fullShuffle[mode].append(percent)
                
    if plot_difference:
        nonsig_diff = dict(weights=[], topology=[])
        for mode in train_auc_df['mode'].unique(): 
            mean_diff_by_percent = train_auc_df.loc[(train_auc_df['mode'] == mode) &
                                                    (train_auc_df['metric'] == 'difference'), 
                                                    ['percent', 'auc_loss (%)']].groupby('percent').mean()
            percent_max_diff = mean_diff_by_percent.index[mean_diff_by_percent['auc_loss (%)'] == mean_diff_by_percent['auc_loss (%)'].max()].values[0]
            stats_df_max_diff = train_auc_df.loc[(train_auc_df['mode'] == mode) & (train_auc_df['percent'] == percent_max_diff), :]
            for idx, percent in enumerate(train_auc_df['percent'].unique()):    
                stats_df = train_auc_df.loc[(train_auc_df['mode'] == mode) & (train_auc_df['percent'] == percent), :]
                
                nMaxDiff = np.sum(stats_df_max_diff.loc[stats_df_max_diff['metric'] == 'difference', 'auc_loss (%)'].values > 
                                  stats_df.loc[stats_df['metric'] == 'difference', 'auc_loss (%)'].values)
                nUnits = np.sum(stats_df['metric'] == 'difference')
                
                sign_test = binomtest(nMaxDiff, nUnits, p = 0.5, alternative='greater')
                print(f'percentMaxDiff > percentDiff, {percent}%, {mode}:  p={np.round(sign_test.pvalue, 4)}, nStrength={sign_test.k}, nUnits={sign_test.n}, prop={np.round(sign_test.proportion_estimate, 2)}')
    
                if sign_test.pvalue >= alpha:
                    nonsig_diff[mode].append(idx)
    
    fig, (ax0, ax1) = plt.subplots(1,2, figsize = plot_params.shuffle_figsize, sharey=True, dpi=plot_params.dpi)
    sns.lineplot(ax=ax0, data = train_auc_df.loc[train_auc_df['mode'] == 'weights', :], x='percent', 
                  y='auc_loss (%)', hue='metric', errorbar='se', err_style='bars', marker='o', markersize=plot_params.shuffle_markersize,
                  linewidth = plot_params.shuffle_linewidth, err_kws={'linewidth': plot_params.shuffle_errwidth}, palette=palette)
    sns.lineplot(ax=ax1, data = train_auc_df.loc[train_auc_df['mode'] == 'topology', :], x='percent', 
                  y='auc_loss (%)', hue='metric', errorbar='se', err_style='bars', marker='o', markersize=plot_params.shuffle_markersize,
                  linewidth = plot_params.shuffle_linewidth, err_kws={'linewidth': plot_params.shuffle_errwidth}, palette=palette)

    # sns.pointplot(ax=ax0, data = train_auc_df.loc[train_auc_df['mode'] == 'weights', :], x='percent', 
    #               y='auc_loss (%)', hue='metric', errorbar='se', scale=plot_params.shuffle_markerscale, 
    #               err_kws={'linewidth': plot_params.shuffle_errwidth}, palette=palette)
    # sns.pointplot(ax=ax1, data = train_auc_df.loc[train_auc_df['mode'] == 'topology', :], x='percent', 
    #               y='auc_loss (%)', hue='metric', errorbar='se', scale=plot_params.shuffle_markerscale, errwidth=plot_params.shuffle_errwidth, palette=palette)
    sns.lineplot (ax=ax0, data = tmp_df2, x='percent', y='auc_loss (%)', color='black', errorbar='se')
    sns.lineplot (ax=ax1, data = tmp_df2, x='percent', y='auc_loss (%)', color='black', errorbar='se')
 
    if plot_difference:
        ax0.plot(nonsig_diff['weights' ], np.repeat(-1, len(nonsig_diff['weights' ])), color='green', marker='o', linestyle='None', markersize=plot_params.shuffle_sigmarkersize)   
        ax1.plot(nonsig_diff['topology'], np.repeat(-1, len(nonsig_diff['topology'])), color='green', marker='o', linestyle='None', markersize=plot_params.shuffle_sigmarkersize)   
        legend_labels = ['Strength', 'Random', 'Difference', 'No FN']
    else:
        legend_labels = ['Strength', 'Random', 'No FN']
        
    ax0.plot(sig_strength_v_random['weights' ], np.repeat(ylim[1]-2.5, len(sig_strength_v_random['weights' ])), color=plt.get_cmap(palette).colors[100], marker='o', linestyle='None', markersize=plot_params.shuffle_sigmarkersize)   
    ax1.plot(sig_strength_v_random['topology'], np.repeat(ylim[1]-2.5, len(sig_strength_v_random['topology'])), color=plt.get_cmap(palette).colors[100], marker='o', linestyle='None', markersize=plot_params.shuffle_sigmarkersize)   
    ax0.plot(sig_noFN_v_strength['weights' ], np.repeat(ylim[1]-1, len(sig_noFN_v_strength['weights' ])), color='black', marker='o', linestyle='None', markersize=plot_params.shuffle_sigmarkersize)   
    ax1.plot(sig_noFN_v_strength['topology'], np.repeat(ylim[1]-1, len(sig_noFN_v_strength['topology'])), color='black', marker='o', linestyle='None', markersize=plot_params.shuffle_sigmarkersize)   
    ax0.plot(sig_strength_v_fullShuffle['weights' ], np.repeat(ylim[1]-0.5, len(sig_strength_v_fullShuffle['weights' ])), color='orange', marker='o', linestyle='None', markersize=plot_params.shuffle_sigmarkersize)   
    ax1.plot(sig_strength_v_fullShuffle['topology'], np.repeat(ylim[1]-0.5, len(sig_strength_v_fullShuffle['topology'])), color='orange', marker='o', linestyle='None', markersize=plot_params.shuffle_sigmarkersize)   

    
    xticklabels = [lab.get_text() for lab in ax0.get_xticklabels()]
    xticklabels = [lab if int(lab)%10==0 else '' for lab in xticklabels]
    # xticklabels[-1] = '100'
    for ax in [ax0, ax1]:
        ax.set_xlabel('Percent of FN permuted')
        ax.set_ylim(ylim)
        ax.set_xticks(np.arange(10, 91, 10))
        # ax.set_xticklabels(xticklabels)

    ax0.set_ylabel('AUC percent loss')    
    ax1.set_ylabel('')    
    ax0.set_title('')
    ax1.set_title('')
    ax0.legend().remove()
    noFN_line_idx = [idx for idx, ob in enumerate(ax1.get_children()) if type(ob) == matplotlib.collections.PolyCollection][0] - 1
    ax1.legend(bbox_to_anchor=(1.1, 0.5), loc='center left', 
               handles = ax1.get_legend_handles_labels()[0] + [ax1.get_children()[noFN_line_idx]],
               labels  = legend_labels)
    sns.despine(fig)
    
    # [ax1.get_children()[66]]

    # fig.set_titles('Shuffled {col_name}:' +  f' {comparison_model}, {target_string}')
    # fig.set(ylim=ylim)
    # fig.set_axis_labels('Percent of Weights Shuffled', 'AUC Percent Loss')
    fig.savefig(os.path.join(plots, paperFig, f'{marmcode}_shuffled_network_auc_loss_summary_figure_{comparison_model}_{target_string}_alpha_pt0{alpha*100}.png'), bbox_inches='tight', dpi=plot_params.dpi)    

    return train_auc_df

def plot_modulation_for_subsets(auc_df, paperFig = 'unknown', figname_mod = '', hue_order_FN=None, 
                                palette=None):    

    plot_save_dir = os.path.join(plots, paperFig) 
    os.makedirs(plot_save_dir, exist_ok=True)

    for metric in auc_df.columns:
        if 'mod' in metric or 'dev' in metric or 'frate' in metric: 
            
            if metric not in ['mod_RO', 'frate_percent_increase']:
                continue
            
            xlabel = [axis_label for axis_label, met_label in zip(['Modulation around \n reach onset (spikes/s)', 'Firing rate ratio \n (reach/spontaneous) '], 
                                                                  ['mod_RO', 'frate_percent_increase']) if met_label == metric][0]
            
            tmp_df = auc_df.loc[~np.isnan(auc_df[metric]), [metric, 'Units Subset', 'quality']]
            
            if 'nomua' in figname_mod.lower():
                tmp_df = tmp_df.loc[tmp_df['quality'] == 'good', :]
            
            med_out = median_test(tmp_df.loc[tmp_df['Units Subset'] == 'Reach-Specific', metric], 
                                  tmp_df.loc[tmp_df['Units Subset'] == 'Non-Specific', metric],
                                  nan_policy='omit')
            print(f'{metric}_{figname_mod}: reach v non, p={np.round(med_out[1], 4)}, {med_out[-1][0,0]}a-{med_out[-1][1,0]}b, {med_out[-1][0,1]}a-{med_out[-1][1,1]}b')
            pval = np.round(med_out[1], 2)
            
            med_out = median_test(tmp_df.loc[tmp_df['Units Subset'] == 'Reach-Specific', metric], 
                                  tmp_df.loc[tmp_df['Units Subset'] == 'Full', metric],
                                  nan_policy='omit')
            pval_full = np.round(med_out[1], 3) 
            print(f'{metric}_{figname_mod}: reach v full, p={np.round(med_out[1], 4)}, {med_out[-1][0,0]}a-{med_out[-1][1,0]}b, {med_out[-1][0,1]}a-{med_out[-1][1,1]}b')
            med_out = median_test(tmp_df.loc[tmp_df['Units Subset'] == 'Non-Specific', metric], 
                                  tmp_df.loc[tmp_df['Units Subset'] == 'Full', metric],
                                  nan_policy='omit')
            print(f'{metric}_{figname_mod}: non v full, p={np.round(med_out[1], 4)}, {med_out[-1][0,0]}a-{med_out[-1][1,0]}b, {med_out[-1][0,1]}a-{med_out[-1][1,1]}b')
            
            subset_colors = plt.get_cmap('Dark2').colors
            full_index = [idx for idx, subset in enumerate(hue_order_FN) if 'full' in subset.lower()][0]
            rs_index   = [idx for idx, subset in enumerate(hue_order_FN) if 'reach' in subset.lower()][0]
            ns_index   = [idx for idx, subset in enumerate(hue_order_FN) if 'non' in subset.lower()][0]


            # fig, ax = plt.subplots(figsize=plot_params.distplot_figsize)
            # sns.kdeplot(ax=ax, data=tmp_df, palette=palette, linewidth=plot_params.distplot_linewidth,
            #             x=metric, hue='Units Subset', common_norm=False, bw_adjust=0.4, cumulative=True, hue_order=hue_order_FN)
            # ax.legend().remove()
            # sns.despine(ax=ax)
            # ax.set_xlabel(metric, fontsize=plot_params.axis_fontsize)
            # ax.text(tmp_df[metric].max()*0.9, ax.get_ylim()[-1]*0.25, f'p={pval}', horizontalalignment='center', fontsize = 12)
            # plt.show()        
            # fig.savefig(os.path.join(plot_save_dir, f'{marmcode}_distribution_{metric}_huekey_UnitsSubset{figname_mod}.png'), bbox_inches='tight', dpi=plot_params.dpi)    

            fig, ax = plt.subplots(figsize=plot_params.distplot_figsize)
            sns.kdeplot(ax=ax, data=tmp_df.loc[tmp_df['Units Subset'] == 'Non-Specific', :], linewidth=plot_params.distplot_linewidth,
                        x=metric, common_norm=False, bw_adjust=0.4, cumulative=False, color=subset_colors[ns_index])
            kde_x, kde_y = ax.get_children()[0].get_data()
            for uIdx, unit_data in tmp_df.iterrows():
                if unit_data['Units Subset'] == 'Reach-Specific':
                    idxs = np.where(np.isclose(unit_data[metric], kde_x, rtol = 1e-1))[0]
                    idx = idxs.mean().astype(int)
                    if idx < 0:
                        continue
                    ax.vlines(kde_x[idx], 0, kde_y[idx], color=subset_colors[rs_index], linewidth=0.5)
            ax.legend().remove()
            sns.despine(ax=ax)
            ax.set_xlabel(xlabel, fontsize=plot_params.axis_fontsize)
            ax.text(tmp_df[metric].max()*0.75, ax.get_ylim()[-1]*0.6, f'p={pval}', horizontalalignment='left', fontsize = plot_params.tick_fontsize)
            plt.show()        
            fig.savefig(os.path.join(plot_save_dir, f'{figname_mod[1:]}_{marmcode}_distribution_{metric}_histogram_highlighted_with_reachspecific{figname_mod}.png'), bbox_inches='tight', dpi=plot_params.dpi)    

def add_neuron_classifications(units_res):
    if marmcode == 'TY':
        class_data = h5py.File(nwb_infile.parent / 'GoodUnitClasses_TY20210211_freeAndMoths-003_processed_new_sorting_corrected.mat')
        neuron_classes = class_data['classes'][:]
    elif marmcode == 'MG':
        neuron_classes = scipy.io.loadmat(nwb_infile.parent / 'GoodUnitClasses_MG20230416.mat')['classes'].squeeze()
    
    neuron_type_strings = ['', 'WS', 'NS', 'unclassified']
    neuron_classes = [neuron_type_strings[int(cl)] for idx, cl in enumerate(neuron_classes) if idx not in filtered_good_units_idxs]
    
    units_res['neuron_type'] = np.full((units_res.shape[0],), 'mua')
    units_res.loc[units_res['quality'] == 'good','neuron_type'] = neuron_classes
    
    return units_res

def plot_average_frates_simple(average_rates_df, paperFig, to_plot=None, behavior_name_map=None, palette=None):
    label = []
    rate = []
    for col in average_rates_df.columns:
        behavior = behavior_name_map[col] if col in behavior_name_map.keys() else col 
        if to_plot is None or behavior in to_plot:
            rate.extend(average_rates_df[col])
            label.extend([behavior]*average_rates_df.shape[0])
    
    frates_sns = pd.DataFrame(data = zip(rate, label),
                              columns = ['Rate', 'Behavior'])
    
    fig, ax = plt.subplots(figsize = plot_params.frate_figsize)
    sns.kdeplot(data=frates_sns, linewidth=plot_params.traj_linewidth,
                x='Rate', hue='Behavior', hue_order=to_plot, legend=True,
                common_norm=False, bw_adjust=0.4, cumulative=True, palette=palette,)
    sns.move_legend(ax,  "upper left", bbox_to_anchor=(1, 1))
    ax.set_xlabel('Firing rate (spikes/s)')
    ax.set_xlim([0, 150])
    ax.set_yticks([])
    fig.savefig(os.path.join(plots, paperFig, f'{marmcode}_behavior_firing_rates_cumulative.png'), bbox_inches='tight', dpi=plot_params.dpi)    
    
    fig1, ax1 = plt.subplots(figsize = plot_params.frate_figsize)
    sns.kdeplot(data=frates_sns, linewidth=plot_params.traj_linewidth,
                x='Rate', hue='Behavior', hue_order=to_plot, legend=True,
                common_norm=False, bw_adjust=0.4, cumulative=False, palette=palette,)
    sns.move_legend(ax1,  "upper left", bbox_to_anchor=(1, 1))
    ax1.set_xlim([0, 150])
    ax1.set_yticks([])
    ax1.set_xlabel('Firing rate (spikes/s)')
    fig1.savefig(os.path.join(plots, paperFig, f'{marmcode}_behavior_firing_rates.png'), bbox_inches='tight', dpi=plot_params.dpi)    

    fig2, ax2 = plt.subplots(figsize = plot_params.frate_figsize)
    sns.kdeplot(data=frates_sns.loc[(frates_sns['Behavior'] == 'Extension') | (frates_sns['Behavior'] == 'Retraction') | (frates_sns['Behavior'] == 'Reach')], 
                linewidth=plot_params.traj_linewidth,
                x='Rate', hue='Behavior', common_norm=False, bw_adjust=0.4, cumulative=False)
    sns.move_legend(ax1, 'upper right')
    # fig2.savefig(os.path.join(plots, paperFig, f'{marmcode}_extension_retraction_firing_rates.png'), bbox_inches='tight', dpi=plot_params.dpi)    

    plt.show()
    
    cosine_sim_df = pd.DataFrame(index   = frates_sns['Behavior'].unique(), 
                                 columns = frates_sns['Behavior'].unique(), 
                                 dtype='float')
    for beh1, beh2 in product(frates_sns['Behavior'].unique(), frates_sns['Behavior'].unique()):
        print(beh1, beh2)
        if beh1 != beh2:
            # cosine_sim_df.loc[beh1, beh2] = cosine_similarity(np.expand_dims(frates_sns.loc[frates_sns['Behavior'] == beh1, 'Rate'], 0),
                                                              # np.expand_dims(frates_sns.loc[frates_sns['Behavior'] == beh2, 'Rate'], 0))
            cosine_sim_df.loc[beh1, beh2] = 1- correlation(frates_sns.loc[frates_sns['Behavior'] == beh1, 'Rate'],
                                                      frates_sns.loc[frates_sns['Behavior'] == beh2, 'Rate'])
    
    # fig, ax = plt.subplots(figsize=(11, 9))
    # # sns.heatmap(ax=ax, data=cosine_sim_df, annot=True, mask=None, cmap='cividis',
    # #             vmin=0.9, vmax=1.0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    # sns.heatmap(ax=ax, data=cosine_sim_df, annot=True, mask=None, cmap='cividis',
    #             square=True, linewidths=.5, cbar_kws={"shrink": .5})
    # ax.set_title(f'Cosine Similarity: Average Firing Rates')
    # fig.savefig(os.path.join(plots, paperFig, f'{marmcode}_GAS_{sub_basis}.png'), bbox_inches='tight', dpi=plot_params.dpi)    

    plt.show()
    
def plot_wji_distributions_for_subsets(weights_df, paperFig = 'unknown', figname_mod = '', hue_order_FN=None, palette=None):
    
    
    for sub_basis in np.unique(weights_df['Units Subset']):
        fig, ax = plt.subplots(figsize=plot_params.distplot_figsize)
        sns.kdeplot(ax=ax, data=weights_df.loc[(weights_df['Units Subset'] == sub_basis), :], linewidth=plot_params.distplot_linewidth, 
                    x='Wji', hue='FN_key', common_norm=False, bw_adjust=0.4, cumulative=True, palette="FN_palette")
        ax.set_title(sub_basis)
        ax.legend(bbox_to_anchor=(1.1, 0.5), loc='center left', 
                  handles = ax.lines[::-1],
                  labels  = ['reachFN1', 'reachFN2', 'spontaneousFN'], 
                  title_fontsize = plot_params.axis_fontsize,
                  fontsize = plot_params.tick_fontsize)
        ax.set_xlabel('$W_{{ji}}$')
        ax.set_yticks([0, 1])
        sns.despine(ax=ax)
        plt.show()
        fig.savefig(os.path.join(plots, paperFig, f'{marmcode}_cumDist_Wji_{sub_basis}_huekey_FNkey{figname_mod}.png'), bbox_inches='tight', dpi=plot_params.dpi)    

        med_out = median_test(weights_df.loc[(weights_df['Units Subset'] == sub_basis) & (weights_df['FN_key'] == 'reachFN1') , 'Wji'], 
                              weights_df.loc[(weights_df['Units Subset'] == sub_basis) & (weights_df['FN_key'] == 'spontaneousFN' ) , 'Wji'])
        print(f'Wji_{sub_basis}{figname_mod}: reachFN1 v spont, p={np.round(med_out[1], 4)}, {med_out[-1][0,0]}a-{med_out[-1][1,0]}b, {med_out[-1][0,1]}a-{med_out[-1][1,1]}b')
        med_out = median_test(weights_df.loc[(weights_df['Units Subset'] == sub_basis) & (weights_df['FN_key'] == 'reachFN2') , 'Wji'], 
                              weights_df.loc[(weights_df['Units Subset'] == sub_basis) & (weights_df['FN_key'] == 'spontaneousFN' ) , 'Wji'])
        print(f'Wji_{sub_basis}{figname_mod}: reachFN1 v spont, p={np.round(med_out[1], 4)}, {med_out[-1][0,0]}a-{med_out[-1][1,0]}b, {med_out[-1][0,1]}a-{med_out[-1][1,1]}b')
                

    for FN_key in np.unique(weights_df['FN_key']):
        fig, ax = plt.subplots(figsize=plot_params.distplot_figsize)
        sns.kdeplot(ax=ax, data=weights_df.loc[(weights_df['FN_key'] == FN_key), :], palette=palette, linewidth=plot_params.distplot_linewidth,
                    x='Wji', hue='Units Subset', common_norm=False, bw_adjust=0.4, cumulative=True, hue_order=hue_order_FN)
        ax.set_title(FN_key)
        ax.legend(bbox_to_anchor=(1.1, 0.5), loc='center left', 
                  handles = ax.lines[::-1],
                  labels  = hue_order_FN, 
                  title_fontsize = plot_params.axis_fontsize,
                  fontsize = plot_params.tick_fontsize)
        ax.set_xlabel('$W_{{ji}}$')
        sns.despine(ax=ax)
        ax.set_yticks([0, 1])
        plt.show()
        fig.savefig(os.path.join(plots, paperFig, f'{marmcode}cumDist_Wji_{FN_key}_huekey_UnitsSubset{figname_mod}.png'), bbox_inches='tight', dpi=plot_params.dpi)    

        med_out = median_test(weights_df.loc[(weights_df['Units Subset'] == 'Reach-Specific') & (weights_df['FN_key'] == FN_key) , 'Wji'], 
                              weights_df.loc[(weights_df['Units Subset'] == 'Non-Specific') & (weights_df['FN_key'] == FN_key ) , 'Wji'])
        print(f'Wji_{FN_key}{figname_mod}: reach-spec v non-spec, p={np.round(med_out[1], 4)}, {med_out[-1][0,0]}a-{med_out[-1][1,0]}b, {med_out[-1][0,1]}a-{med_out[-1][1,1]}b')
       
        med_out = median_test(weights_df.loc[(weights_df['Units Subset'] == 'Reach-Specific') & (weights_df['FN_key'] == FN_key) , 'Wji'], 
                              weights_df.loc[(weights_df['Units Subset'] == 'Full') & (weights_df['FN_key'] == FN_key ) , 'Wji'])
        print(f'Wji_{FN_key}{figname_mod}: reach-spec v original, p={np.round(med_out[1], 4)}, {med_out[-1][0,0]}a-{med_out[-1][1,0]}b, {med_out[-1][0,1]}a-{med_out[-1][1,1]}b')

        med_out = median_test(weights_df.loc[(weights_df['Units Subset'] == 'Non-Specific') & (weights_df['FN_key'] == FN_key) , 'Wji'], 
                              weights_df.loc[(weights_df['Units Subset'] == 'Full') & (weights_df['FN_key'] == FN_key ) , 'Wji'])
        print(f'Wji_{FN_key}{figname_mod}: non_reach-spec v original, p={np.round(med_out[1], 4)}, {med_out[-1][0,0]}a-{med_out[-1][1,0]}b, {med_out[-1][0,1]}a-{med_out[-1][1,1]}b')
    
        if 'reachFN' in FN_key:    
            for w_diff_key, w_diff_label in zip(['Wji_diff', 'absolute_Wji_diff'], ['$W_{{ji}}$ (reachFN) - $W_{{ji}}$ (spontaneousFN)', '|$W_{{ji}}$ (reachFN) - $W_{{ji}}$ (spontaneousFN)|']):  
                fig, ax = plt.subplots(figsize = plot_params.distplot_figsize)
                sns.kdeplot(data=weights_df[weights_df['FN_key'] == FN_key], ax=ax, x=w_diff_key, 
                            hue='Units Subset', hue_order = hue_order_FN, palette=palette,
                            cumulative=True, common_norm=False, bw_adjust=0.05, linewidth=plot_params.distplot_linewidth)
                ax.legend(bbox_to_anchor=(1.1, 0.5), loc='center left',
                          handles = ax.lines[::-1],
                          labels  = hue_order_FN, 
                          title_fontsize = plot_params.axis_fontsize,
                          fontsize = plot_params.tick_fontsize)
                sns.despine(ax=ax)
                ax.set_yticks([0, 1])
                ax.set_xlabel(w_diff_label)
                plt.show()
                fig.savefig(os.path.join(plots, paperFig, f'{marmcode}_{w_diff_key}_cumulative_dist_{FN_key}_minus_spont{figname_mod}.png'), bbox_inches='tight', dpi=plot_params.dpi)
                med_out = median_test(weights_df.loc[(weights_df['Units Subset'] == 'Reach-Specific') & (weights_df['FN_key'] == FN_key), w_diff_key], 
                                      weights_df.loc[(weights_df['Units Subset'] == 'Non-Specific')   & (weights_df['FN_key'] == FN_key), w_diff_key])
                print(f'{w_diff_key}_{FN_key}{figname_mod}:  reach-spec vs non-spec, p={np.round(med_out[1], 4)}, {med_out[-1][0,0]}a-{med_out[-1][1,0]}b, {med_out[-1][0,1]}a-{med_out[-1][1,1]}b')
                med_out = median_test(weights_df.loc[(weights_df['Units Subset'] == 'Reach-Specific') & (weights_df['FN_key'] == FN_key), w_diff_key], 
                                      weights_df.loc[(weights_df['Units Subset'] == 'Full')   & (weights_df['FN_key'] == FN_key), w_diff_key])
                print(f'{w_diff_key}_{FN_key}{figname_mod}:  reach-spec vs full, p={np.round(med_out[1], 4)}, {med_out[-1][0,0]}a-{med_out[-1][1,0]}b, {med_out[-1][0,1]}a-{med_out[-1][1,1]}b')
                med_out = median_test(weights_df.loc[(weights_df['Units Subset'] == 'Non-Specific') & (weights_df['FN_key'] == FN_key), w_diff_key], 
                                      weights_df.loc[(weights_df['Units Subset'] == 'Full')   & (weights_df['FN_key'] == FN_key), w_diff_key])
                print(f'{w_diff_key}_{FN_key}{figname_mod}:  non-spec vs full, p={np.round(med_out[1], 4)}, {med_out[-1][0,0]}a-{med_out[-1][1,0]}b, {med_out[-1][0,1]}a-{med_out[-1][1,1]}b')
                                                                                     
    # fig, ax = plt.subplots(figsize=plot_params.stripplot_figsize, dpi=plot_params.dpi)
    # sns.stripplot(ax=ax, data=weights_df, x='Units Subset', y='Wji', hue='FN_key', 
    #               dodge=True, order=hue_order_FN, palette='FN_palette', s=plot_params.stripplot_markersize)
    # ax.legend(bbox_to_anchor=(1.1, 0.5), loc='center left', 
    #           labels  = ['reachFN1', 'reachFN2', 'spontaneousFN'], 
    #           title_fontsize = plot_params.axis_fontsize,
    #           fontsize = plot_params.tick_fontsize)
    # ax.set_ylabel('$W_{{ji}}$')
    # sns.despine(ax=ax)
    # fig.savefig(os.path.join(plots, 'FigS3', f'{marmcode}_striplot_Wji_groupedBy_UnitsSubset_huekey_FNkey{figname_mod}.png'), bbox_inches='tight', dpi=plot_params.dpi)       

    # fig = sns.relplot(data=weights_df, x='pearson_r', y='Wji', col='Units Subset', kind='scatter')
    # plt.show() 
    # fig.savefig(os.path.join(plots, 'unknown', f'{marmcode}_subnetwork_pearson_r_squared_vs_wji_colkey_UnitsSubset{figname_mod}.png'), bbox_inches='tight', dpi=plot_params.dpi)    

def graph_alignment_score(FN_comps):
    gas = 2*FN_comps.min(axis=1).sum() / FN_comps.sum(axis=1).sum(axis=0) 
    return gas

def make_FN_similarity_plots(FN_df, sub_basis, paperFig, gen_test_behavior):
    # corr = FN_df.corr()
    # np.fill_diagonal(corr.values, np.nan)    
    # mask = np.triu(np.ones_like(corr, dtype=bool))
    # f, ax = plt.subplots(figsize=(11, 9))
    # sns.heatmap(ax=ax, data=corr, annot=True, mask=None, cmap='cividis',
    #             square=True, linewidths=.5, cbar_kws={"shrink": .5})
    # ax.set_title(f'Correlation: {sub_basis}')
    # plt.show()
    
    # cosineSim_df = pd.DataFrame(index = FN_df.columns, columns=FN_df.columns, dtype='float')
    # for col1, col2 in product(FN_df.columns, FN_df.columns):
    #     if col1 != col2:
    #         cosineSim_df.loc[col1, col2] = cosine_similarity(np.expand_dims(FN_df[col1], 0), np.expand_dims(FN_df[col2], 0))[0]
    
    # f, ax = plt.subplots(figsize=(11, 9))
    # sns.heatmap(ax=ax, data=cosineSim_df, annot=True, mask=None, cmap='cividis',
    #             vmin=0.2, vmax=1, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    # ax.set_title(f'Cosine Similarity: {sub_basis}')
    # plt.show()
    
    gas_df = pd.DataFrame(index = FN_df.columns, columns=FN_df.columns, dtype='float')
    for col1, col2 in product(FN_df.columns, FN_df.columns):
        if col1 != col2:
            gas_df.loc[col1, col2] = graph_alignment_score(FN_df[[col1, col2]])
    mask = np.triu(np.ones_like(gas_df, dtype=bool))
    fig, ax = plt.subplots(figsize=(plot_params.gas_figSize))
    sns.heatmap(ax=ax, data=gas_df, annot=True, mask=mask, cmap='cividis',
                vmin=0.1, vmax=0.85, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot_kws={"size": plot_params.tick_fontsize-2})
    ax.set_title(f'GAS: {sub_basis}')
    fig.savefig(os.path.join(plots, paperFig, f'{marmcode}_GAS_{gen_test_behavior}_{sub_basis}.png'), bbox_inches='tight', dpi=plot_params.dpi)    

    plt.show()
    
def plot_trajectory_correlation_and_auc_distributions_for_subsets(auc_df, weights_df, paperFig = 'unknown', figname_mod = '', hue_order_FN=None, palette=None):
    
    fig, ax = plt.subplots(figsize=plot_params.distplot_figsize)
    sns.kdeplot(ax=ax, data=auc_df, palette=palette, linewidth=plot_params.distplot_linewidth,
                x='Kinematics_AUC', hue='Units Subset', common_norm=False, bw_adjust=0.4, cumulative=True, hue_order=hue_order_FN)
    ax.legend(bbox_to_anchor=(1.1, 0.5), loc='center left', 
              handles = ax.lines[::-1],
              labels  = hue_order_FN, 
              title_fontsize = plot_params.axis_fontsize,
              fontsize = plot_params.tick_fontsize)
    ax.set_yticks([0, 1])
    sns.despine(ax=ax)
    ax.set_xlabel('Full kinematics AUC', fontsize=plot_params.axis_fontsize)
    plt.show()        
    fig.savefig(os.path.join(plots, paperFig, f'{marmcode}_cumDist_KinAUC_huekey_UnitsSubset{figname_mod}.png'), bbox_inches='tight', dpi=plot_params.dpi)    

    #---------------------------------------------------------------------------------
    med_out = median_test(auc_df.loc[auc_df['Units Subset'] == 'Reach-Specific', 'Kinematics_AUC'], 
                          auc_df.loc[auc_df['Units Subset'] == 'Non-Specific', 'Kinematics_AUC'])
    print(f'Kin_AUC{figname_mod}: reach v non, p={np.round(med_out[1], 4)}, {med_out[-1][0,0]}a-{med_out[-1][1,0]}b, {med_out[-1][0,1]}a-{med_out[-1][1,1]}b')
       
    med_out = median_test(auc_df.loc[auc_df['Units Subset'] == 'Reach-Specific', 'Kinematics_AUC'], 
                          auc_df.loc[auc_df['Units Subset'] == 'Full', 'Kinematics_AUC'])
    print(f'Kin_AUC{figname_mod}: reach v full, p={np.round(med_out[1], 4)}, {med_out[-1][0,0]}a-{med_out[-1][1,0]}b, {med_out[-1][0,1]}a-{med_out[-1][1,1]}b')
    med_out = median_test(auc_df.loc[auc_df['Units Subset'] == 'Non-Specific', 'Kinematics_AUC'], 
                          auc_df.loc[auc_df['Units Subset'] == 'Full', 'Kinematics_AUC'])
    print(f'Kin_AUC{figname_mod}: non v full, p={np.round(med_out[1], 4)}, {med_out[-1][0,0]}a-{med_out[-1][1,0]}b, {med_out[-1][0,1]}a-{med_out[-1][1,1]}b')
    
    fig, ax = plt.subplots(figsize=plot_params.distplot_figsize)
    sns.kdeplot(ax=ax, data=auc_df, palette=palette, linewidth=plot_params.distplot_linewidth,
                x='Kinematics_AUC', hue='Units Subset', common_norm=False, bw_adjust=0.4, cumulative=False, hue_order=hue_order_FN)
    ax.legend().remove()
    sns.despine(ax=ax)
    ax.set_xlabel('Full kinematics AUC', fontsize=plot_params.axis_fontsize)
    plt.show()        
    fig.savefig(os.path.join(plots, paperFig, f'{marmcode}_distribution_KinAUC_huekey_UnitsSubset{figname_mod}.png'), bbox_inches='tight', dpi=plot_params.dpi)    

    #---------------------------------------------------------------------------------    

    med_out = median_test(weights_df.loc[weights_df['Units Subset'] == 'Reach-Specific', 'pearson_r'], 
                          weights_df.loc[weights_df['Units Subset'] == 'Non-Specific', 'pearson_r'])
    print(f'pearson_corr{figname_mod}: reach v non, p={np.round(med_out[1], 4)}, {med_out[-1][0,0]}a-{med_out[-1][1,0]}b, {med_out[-1][0,1]}a-{med_out[-1][1,1]}b')
    med_out = median_test(weights_df.loc[weights_df['Units Subset'] == 'Reach-Specific', 'pearson_r'], 
                          weights_df.loc[weights_df['Units Subset'] == 'Full', 'pearson_r'])
    print(f'pearson_corr{figname_mod}: reach v full, p={np.round(med_out[1], 4)}, {med_out[-1][0,0]}a-{med_out[-1][1,0]}b, {med_out[-1][0,1]}a-{med_out[-1][1,1]}b')
    med_out = median_test(weights_df.loc[weights_df['Units Subset'] == 'Non-Specific', 'pearson_r'], 
                          weights_df.loc[weights_df['Units Subset'] == 'Full', 'pearson_r'])
    print(f'pearson_corr{figname_mod}: non v full, p={np.round(med_out[1], 4)}, {med_out[-1][0,0]}a-{med_out[-1][1,0]}b, {med_out[-1][0,1]}a-{med_out[-1][1,1]}b')
        
    fig, ax = plt.subplots(figsize=plot_params.distplot_figsize)
    sns.kdeplot(ax=ax, data=weights_df, palette=palette, linewidth=plot_params.distplot_linewidth,
                x='pearson_r', hue='Units Subset', common_norm=False, bw_adjust=0.4, cumulative=False, hue_order=hue_order_FN)
    ax.legend().remove()
    ax.set_xlabel('Preferred trajectory\ncorrelation')
    sns.despine(ax=ax)
    plt.show()    
    fig.savefig(os.path.join(plots, paperFig, f'{marmcode}_distribution_pearson_r_huekey_UnitsSubset{figname_mod}.png'), bbox_inches='tight', dpi=plot_params.dpi)    


def NO_SUMS_plot_in_and_out_weights_of_functional_groups(units_res, annotated_FN_dict, outside_group_only = False, 
                                                 subset_idxs = None, subset_basis=['Reach-Specific'],
                                                 gen_test_behavior = None,
                                                 hue_order_FN=None, palette=None):
    
    weights_df = pd.DataFrame()
    completed_comps = []
    if subset_idxs is not None:
        for sub_basis in subset_basis: 
            if outside_group_only and sub_basis == 'Full':
                continue
            for (behavior, behavior_FN), (behaviorComp, behaviorComp_FN) in product(annotated_FN_dict.items(), annotated_FN_dict.items()):
                
                if not (behaviorComp.lower() == gen_test_behavior.lower() and behavior.lower() == 'reach1'):
                    continue
                
                # if not ((behaviorComp.lower() == gen_test_behavior.lower() and behavior.lower() == 'reach1') or 
                #        ( behavior.lower() == gen_test_behavior.lower() and behaviorComp.lower() == 'reach1')):
                #     continue
                
                # elif behavior == behaviorComp or f'{behaviorComp}_vs_{behavior}' in completed_comps:
                #     continue
                # else:
                #     completed_comps.append(f'{behavior}_vs_{behaviorComp}')
                
                sub_idxs = subset_idxs.copy()
                
                if 'Non-' in sub_basis:
                    sub_idxs = np.setdiff1d(np.array(range(units_res.shape[0])), sub_idxs)
                elif sub_basis == 'Full':
                    sub_idxs = np.array(range(units_res.shape[0]))                     
                
                for sub_type in ['target', 'source']:
                    if sub_type == 'target':
                        if outside_group_only:
                            targets, sources = sub_idxs, np.setdiff1d(np.arange(behavior_FN.shape[1], dtype=int), sub_idxs)
                        else:
                            targets, sources = sub_idxs, range(behavior_FN.shape[1])                            
                        sub_FN, sub_comp_FN = behavior_FN[np.ix_(targets, sources)], behaviorComp_FN[np.ix_(targets, sources)]
                        units_res_subset_units = units_res.loc[targets, :]
                    elif sub_type == 'source':
                        if outside_group_only:
                            targets, sources = np.setdiff1d(np.arange(behavior_FN.shape[0], dtype=int), sub_idxs), sub_idxs 
                        else:
                            targets, sources = range(behavior_FN.shape[0]), sub_idxs  
                        sub_FN, sub_comp_FN = behavior_FN[np.ix_(targets, sources)], behaviorComp_FN[np.ix_(targets, sources)]
                        units_res_subset_units = units_res.loc[sources, :]
                                
                    units_res_sources = units_res.loc[sources, :]
                    units_res_targets = units_res.loc[targets, :]

                    if sub_type == 'target':
                        wKey = 'In'
                        w_in      = sub_FN[sub_FN != 0]
                        w_in_comp = sub_comp_FN[sub_FN != 0]
                        tmp_df = pd.DataFrame(data=zip(w_in,
                                                       w_in_comp,
                                                       np.tile(units_res_targets['cortical_area'], sub_FN.shape[1]),
                                                       np.repeat(wKey, sub_FN.size),
                                                       np.repeat(sub_basis, sub_FN.size),
                                                       ),
                                              columns=['Wji', 'Wji_comp', 'input_area', 'In/Out', 'Units Subset'])
                    elif sub_type == 'source':
                        wKey = 'Out'
                        w_out      = sub_FN[sub_FN != 0]
                        w_out_comp = sub_comp_FN[sub_FN != 0]
                        tmp_df = pd.DataFrame(data=zip(w_out,
                                                       w_out_comp,
                                                       np.tile(units_res_sources['cortical_area'], sub_FN.shape[0]),
                                                       np.repeat(wKey, sub_FN.size),
                                                       np.repeat(sub_basis, sub_FN.size),
                                                       ),
                                              columns=['Wji', 'Wji_comp', 'input_area', 'In/Out', 'Units Subset'])
                    weights_df = pd.concat((weights_df, tmp_df))

    # plot_wji_distributions_for_subsets(weights_df, paperFig = 'Exploratory_Spont_FNs', figname_mod = '', hue_order_FN=hue_order_FN, palette=palette) 
    make_in_and_out_weight_distribution_plots(weights_df, behavior='reach1', comp_behavior=gen_test_behavior, hue_order_FN=hue_order_FN, palette=palette, paperFig = None)
    # plot_trajectory_correlation_and_auc_distributions_for_subsets(auc_df, weights_df, paperFig='Exploratory_Spont_FNs', figname_mod = '', hue_order_FN=hue_order_FN, palette=palette)
    # plot_modulation_for_subsets(auc_df, paperFig='Exploratory_Spont_FNs', figname_mod = '_all', hue_order_FN=hue_order_FN, palette=palette)

    return weights_df  

def make_in_and_out_weight_distribution_plots(weights_df, behavior, comp_behavior, hue_order_FN=None, palette=None, paperFig = None):
    for in_out_key, FG_member_key in product(weights_df['In/Out'].unique(), weights_df['FG Member'].unique()): 
        fig, ax = plt.subplots(figsize=plot_params.distplot_figsize)
        sns.kdeplot(ax=ax, data=weights_df.loc[(weights_df['In/Out'] == in_out_key) & (weights_df['FG Member'] == FG_member_key), :], palette=palette, linewidth=plot_params.distplot_linewidth,
                    x='Wji', hue='Units Subset', common_norm=False, bw_adjust=0.4, cumulative=True, hue_order=hue_order_FN)
        ax.set_title(f'{in_out_key} Weights: {FG_member_key}')
        ax.legend(bbox_to_anchor=(1.1, 0.5), loc='center left', 
                  handles = ax.lines[::-1],
                  labels  = hue_order_FN, 
                  title_fontsize = plot_params.axis_fontsize,
                  fontsize = plot_params.tick_fontsize)
        ax.set_xlabel(f'$W_{{{in_out_key}}}$')
        sns.despine(ax=ax)
        ax.set_yticks([0, 1])
        plt.show()

    for behave, FG_member_key in product([behavior, comp_behavior], weights_df['FG Member'].unique()): 
        fig, ax = plt.subplots(figsize=plot_params.distplot_figsize)
        behave = behave.capitalize() if behave != 'reach1' else behave
        sns.kdeplot(ax=ax, data=weights_df.loc[(weights_df['FG Member'] == FG_member_key) & \
                                               (weights_df['Behavior'] == behave), :], 
                    palette=palette, linewidth=plot_params.distplot_linewidth,
                    x='Wji', hue='Units Subset', common_norm=False, bw_adjust=0.4, cumulative=True, hue_order=hue_order_FN)
        ax.set_title(f'{behave}: {FG_member_key}')
        ax.legend(bbox_to_anchor=(1.1, 0.5), loc='center left', 
                  handles = ax.lines[::-1],
                  labels  = hue_order_FN, 
                  title_fontsize = plot_params.axis_fontsize,
                  fontsize = plot_params.tick_fontsize)
        ax.set_xlabel(f'$W_{{mean}}$')
        sns.despine(ax=ax)
        ax.set_xlim([0, .02])
        ax.set_yticks([0, 1])
        plt.show()

    for behave, subset in product([behavior, comp_behavior], weights_df['Units Subset'].unique()): 
        fig, ax = plt.subplots(figsize=plot_params.distplot_figsize)
        behave = behave.capitalize() if behave != 'reach1' else behave
        sns.kdeplot(ax=ax, data=weights_df.loc[(weights_df['Units Subset'] == subset) & \
                                               (weights_df['Behavior'] == behave), :], 
                    palette=palette, linewidth=plot_params.distplot_linewidth,
                    x='Wji', hue='FG Member', common_norm=False, bw_adjust=0.4, cumulative=True, hue_order=None)
        ax.set_title(f'{behave}: {subset}')
        ax.legend(bbox_to_anchor=(1.1, 0.5), loc='center left', 
                  handles = ax.lines[::-1],
                  labels  = ['Within FG', 'Outside FG'], 
                  title_fontsize = plot_params.axis_fontsize,
                  fontsize = plot_params.tick_fontsize)
        ax.set_xlabel(f'$W_{{mean}}$')
        sns.despine(ax=ax)
        ax.set_xlim([0, .02])
        ax.set_yticks([0, 1])
        
        med_out = median_test(weights_df.loc[(weights_df['Units Subset'] == subset) & \
                                             (weights_df['Behavior'] == behave)     & \
                                             (weights_df['FG Member'] == 'inFG'),        'Wji'], 
                              weights_df.loc[(weights_df['Units Subset'] == subset) & \
                                             (weights_df['Behavior'] == behave)     & \
                                             (weights_df['FG Member'] == 'outFG'),        'Wji']
                              )
        print(f'Wji, {behave}, {subset}: within vs outside FG, p={np.round(med_out[1], 4)}, {med_out[-1][0,0]}a-{med_out[-1][1,0]}b, {med_out[-1][0,1]}a-{med_out[-1][1,1]}b')
        
        
        plt.show()
        
        # fig, ax = plt.subplots(figsize=plot_params.distplot_figsize)
        # sns.kdeplot(ax=ax, data=weights_df.loc[(weights_df['In/Out'] == in_out_key) & (weights_df['FG Member'] == FG_member_key), :], palette=palette, linewidth=plot_params.distplot_linewidth,
        #             x='Wji_comp', hue='Units Subset', common_norm=False, bw_adjust=0.4, cumulative=True, hue_order=hue_order_FN)
        # ax.set_title(f'{in_out_key} Weights: {comp_behavior}')
        # ax.legend(bbox_to_anchor=(1.1, 0.5), loc='center left', 
        #           handles = ax.lines[::-1],
        #           labels  = hue_order_FN, 
        #           title_fontsize = plot_params.axis_fontsize,
        #           fontsize = plot_params.tick_fontsize)
        # ax.set_xlabel(f'$W_{{{in_out_key}}}$, {FG_member_key}')
        # sns.despine(ax=ax)
        # ax.set_yticks([0, 1])
        # plt.show()
    
    for subset, FG_member_key in product(weights_df['Units Subset'].unique(), weights_df['FG Member'].unique()): 
        fig, ax = plt.subplots(figsize=plot_params.distplot_figsize)
        sns.kdeplot(ax=ax, data=weights_df.loc[(weights_df['Units Subset'] == subset) & (weights_df['FG Member'] == FG_member_key), :], palette=palette, linewidth=plot_params.distplot_linewidth,
                    x='Wji', hue='Behavior', common_norm=False, bw_adjust=0.4, cumulative=True, hue_order=None)
        ax.set_title(f'{subset}: {FG_member_key}')
        ax.legend(bbox_to_anchor=(1.1, 0.5), loc='center left', 
                  handles = ax.lines[::-1], 
                  labels = [behavior, comp_behavior],
                  title_fontsize = plot_params.axis_fontsize,
                  fontsize = plot_params.tick_fontsize)
        ax.set_xlabel(f'$W_{{ji}}$')
        ax.set_xlim([0, .02])
        sns.despine(ax=ax)
        ax.set_yticks([0, 1])
        plt.show()
        
    

# def make_in_and_out_weight_distribution_plots(weights_df, behavior, comp_behavior, hue_order_FN=None, palette=None, paperFig = None):
#     for in_out_key in weights_df['In/Out'].unique(): 
#         fig, ax = plt.subplots(figsize=plot_params.distplot_figsize)
#         sns.kdeplot(ax=ax, data=weights_df.loc[(weights_df['In/Out'] == in_out_key), :], palette=palette, linewidth=plot_params.distplot_linewidth,
#                     x='Wji', hue='Units Subset', common_norm=False, bw_adjust=0.4, cumulative=True, hue_order=hue_order_FN)
#         ax.set_title(f'{in_out_key} Weights: {behavior}')
#         ax.legend(bbox_to_anchor=(1.1, 0.5), loc='center left', 
#                   handles = ax.lines[::-1],
#                   labels  = hue_order_FN, 
#                   title_fontsize = plot_params.axis_fontsize,
#                   fontsize = plot_params.tick_fontsize)
#         ax.set_xlabel('$W_{{ji}}$')
#         sns.despine(ax=ax)
#         ax.set_yticks([0, 1])
#         plt.show()
        
#         fig, ax = plt.subplots(figsize=plot_params.distplot_figsize)
#         sns.kdeplot(ax=ax, data=weights_df.loc[(weights_df['In/Out'] == in_out_key), :], palette=palette, linewidth=plot_params.distplot_linewidth,
#                     x='Wji_comp', hue='Units Subset', common_norm=False, bw_adjust=0.4, cumulative=True, hue_order=hue_order_FN)
#         ax.set_title(f'{in_out_key} Weights: {comp_behavior}')
#         ax.legend(bbox_to_anchor=(1.1, 0.5), loc='center left', 
#                   handles = ax.lines[::-1],
#                   labels  = hue_order_FN, 
#                   title_fontsize = plot_params.axis_fontsize,
#                   fontsize = plot_params.tick_fontsize)
#         ax.set_xlabel('$W_{{ji}}$')
#         sns.despine(ax=ax)
#         ax.set_yticks([0, 1])
#         plt.show()

def plot_in_and_out_weights_of_functional_groups(units_res, annotated_FN_dict, outside_group_only = False, 
                                                 subset_idxs = None, subset_basis=['Reach-Specific'],
                                                 gen_test_behavior = None,
                                                 hue_order_FN=None, palette=None):
    
    weights_df = pd.DataFrame()
    completed_comps = []
    if subset_idxs is not None:
        for sub_basis in subset_basis: 
            if outside_group_only and sub_basis == 'Full':
                continue
            for (behavior, behavior_FN), (behaviorComp, behaviorComp_FN) in product(annotated_FN_dict.items(), annotated_FN_dict.items()):
                print(behavior, behaviorComp)

                if not (behaviorComp.lower() == gen_test_behavior.lower() and behavior.lower() == 'reach1'):
                    continue
                                
                # if not ((behaviorComp.lower() == gen_test_behavior.lower() and behavior.lower() == 'reach1') or 
                #        ( behavior.lower() == gen_test_behavior.lower() and behaviorComp.lower() == 'reach1')):
                #     continue
                
                # elif behavior == behaviorComp or f'{behaviorComp}_vs_{behavior}' in completed_comps:
                #     continue
                # else:
                #     completed_comps.append(f'{behavior}_vs_{behaviorComp}')
                
                sub_idxs = subset_idxs.copy()
                
                if 'Non-' in sub_basis:
                    sub_idxs = np.setdiff1d(np.array(range(units_res.shape[0])), sub_idxs)
                elif sub_basis == 'Full':
                    sub_idxs = np.array(range(units_res.shape[0]))                     
                
                for wDir in ['In', 'Out']:
                    if wDir == 'In':
                        targets, sources = sub_idxs, range(behavior_FN.shape[1])                            
                        sub_FN, sub_comp_FN = behavior_FN[np.ix_(targets, sources)], behaviorComp_FN[np.ix_(targets, sources)]
                        units_res_subset_units = units_res.loc[targets, :]
                        
                        tmp_df = pd.DataFrame(data=zip(sub_FN     [:, sub_idxs].mean(axis=1),
                                                       units_res_subset_units['cortical_area'],
                                                       np.repeat(wDir, sub_FN.shape[0]),
                                                       np.repeat(sub_basis, sub_FN.shape[0]),
                                                       np.repeat('inFG', sub_FN.shape[0]),
                                                       np.repeat(behavior, sub_FN.shape[0])),
                                              columns=['Wji', 'input_area', 'In/Out', 'Units Subset', 'FG Member', 'Behavior'])
                        weights_df = pd.concat((weights_df, tmp_df))

                        tmp_df = pd.DataFrame(data=zip(sub_comp_FN[:, sub_idxs].mean(axis=1),
                                                       units_res_subset_units['cortical_area'],
                                                       np.repeat(wDir, sub_FN.shape[0]),
                                                       np.repeat(sub_basis, sub_FN.shape[0]),
                                                       np.repeat('inFG', sub_FN.shape[0]),
                                                       np.repeat(behaviorComp, sub_FN.shape[0])),
                                              columns=['Wji', 'input_area', 'In/Out', 'Units Subset', 'FG Member', 'Behavior'])
                        weights_df = pd.concat((weights_df, tmp_df))

                        tmp_df = pd.DataFrame(data=zip(sub_FN     [:, np.setdiff1d(np.arange(behavior_FN.shape[0]), sub_idxs)].mean(axis=1),
                                                       units_res_subset_units['cortical_area'],
                                                       np.repeat(wDir, sub_FN.shape[0]),
                                                       np.repeat(sub_basis, sub_FN.shape[0]),
                                                       np.repeat('outFG', sub_FN.shape[0]),
                                                       np.repeat(behavior, sub_FN.shape[0])),
                                              columns=['Wji', 'input_area', 'In/Out', 'Units Subset', 'FG Member', 'Behavior'])
                        weights_df = pd.concat((weights_df, tmp_df))

                        tmp_df = pd.DataFrame(data=zip(sub_comp_FN[:, np.setdiff1d(np.arange(behavior_FN.shape[0]), sub_idxs)].mean(axis=1),
                                                       units_res_subset_units['cortical_area'],
                                                       np.repeat(wDir, sub_FN.shape[0]),
                                                       np.repeat(sub_basis, sub_FN.shape[0]),
                                                       np.repeat('outFG', sub_FN.shape[0]),
                                                       np.repeat(behaviorComp, sub_FN.shape[0])),
                                              columns=['Wji', 'input_area', 'In/Out', 'Units Subset', 'FG Member', 'Behavior'])
                        weights_df = pd.concat((weights_df, tmp_df))
                        
                    elif wDir == 'Out':
                        targets, sources = range(behavior_FN.shape[0]), sub_idxs  
                        sub_FN, sub_comp_FN = behavior_FN[np.ix_(targets, sources)], behaviorComp_FN[np.ix_(targets, sources)]
                        units_res_subset_units = units_res.loc[sources, :]
                                
                        tmp_df = pd.DataFrame(data=zip(sub_FN     [sub_idxs].mean(axis=0),
                                                       units_res_subset_units['cortical_area'],
                                                       np.repeat(wDir, sub_FN.shape[0]),
                                                       np.repeat(sub_basis, sub_FN.shape[0]),
                                                       np.repeat('inFG', sub_FN.shape[0]),
                                                       np.repeat(behavior, sub_FN.shape[0])),
                                              columns=['Wji', 'input_area', 'In/Out', 'Units Subset', 'FG Member', 'Behavior'])
                        weights_df = pd.concat((weights_df, tmp_df))

                        tmp_df = pd.DataFrame(data=zip(sub_comp_FN[sub_idxs].mean(axis=0),
                                                       units_res_subset_units['cortical_area'],
                                                       np.repeat(wDir, sub_FN.shape[0]),
                                                       np.repeat(sub_basis, sub_FN.shape[0]),
                                                       np.repeat('inFG', sub_FN.shape[0]),
                                                       np.repeat(behaviorComp, sub_FN.shape[0])),
                                              columns=['Wji', 'input_area', 'In/Out', 'Units Subset', 'FG Member', 'Behavior'])
                        weights_df = pd.concat((weights_df, tmp_df))

                        tmp_df = pd.DataFrame(data=zip(sub_FN     [np.setdiff1d(np.arange(behavior_FN.shape[0]), sub_idxs)].mean(axis=0),
                                                       units_res_subset_units['cortical_area'],
                                                       np.repeat(wDir, sub_FN.shape[0]),
                                                       np.repeat(sub_basis, sub_FN.shape[0]),
                                                       np.repeat('outFG', sub_FN.shape[0]),
                                                       np.repeat(behavior, sub_FN.shape[0])),
                                              columns=['Wji', 'input_area', 'In/Out', 'Units Subset', 'FG Member', 'Behavior'])
                        weights_df = pd.concat((weights_df, tmp_df))

                        tmp_df = pd.DataFrame(data=zip(sub_comp_FN[np.setdiff1d(np.arange(behavior_FN.shape[0]), sub_idxs)].mean(axis=0),
                                                       units_res_subset_units['cortical_area'],
                                                       np.repeat(wDir, sub_FN.shape[0]),
                                                       np.repeat(sub_basis, sub_FN.shape[0]),
                                                       np.repeat('outFG', sub_FN.shape[0]),
                                                       np.repeat(behaviorComp, sub_FN.shape[0])),
                                              columns=['Wji', 'input_area', 'In/Out', 'Units Subset', 'FG Member', 'Behavior'])
                        weights_df = pd.concat((weights_df, tmp_df))

    # plot_wji_distributions_for_subsets(weights_df, paperFig = 'Exploratory_Spont_FNs', figname_mod = '', hue_order_FN=hue_order_FN, palette=palette) 
    make_in_and_out_weight_distribution_plots(weights_df, behavior='reach1', comp_behavior=gen_test_behavior, hue_order_FN=hue_order_FN, palette=palette, paperFig = 'Revision_Plots')
    # plot_trajectory_correlation_and_auc_distributions_for_subsets(auc_df, weights_df, paperFig='Exploratory_Spont_FNs', figname_mod = '', hue_order_FN=hue_order_FN, palette=palette)
    # plot_modulation_for_subsets(auc_df, paperFig='Exploratory_Spont_FNs', figname_mod = '_all', hue_order_FN=hue_order_FN, palette=palette)

    return weights_df

def examine_all_annotated_FNs(units_res, annotated_FN_dict, 
                              subset_idxs = None, sub_type = None, subset_basis=['Reach-Specific'],
                              gen_test_behavior = None, traj_corr_df = None,
                              hue_order_FN=None, palette=None, paperFig=None):
    
    weights_df = pd.DataFrame()
    auc_df = pd.DataFrame()
    FN_dataframes = dict()
    completed_comps = []
    if subset_idxs is not None:
        for sub_basis in subset_basis: 
            FN_df = pd.DataFrame()
            for (behavior, behavior_FN), (behaviorComp, behaviorComp_FN) in product(annotated_FN_dict.items(), annotated_FN_dict.items()):
                                
                sub_idxs = subset_idxs.copy()
                
                if 'Non-' in sub_basis:
                    sub_idxs = np.setdiff1d(np.array(range(units_res.shape[0])), sub_idxs)
                elif sub_basis == 'Full':
                    sub_idxs = np.array(range(units_res.shape[0]))
               
                if sub_type == 'both':
                    targets, sources, sub_FN, sub_comp_FN = sub_idxs, sub_idxs, behavior_FN[np.ix_(sub_idxs, sub_idxs)], behaviorComp_FN[np.ix_(sub_idxs, sub_idxs)]
                    units_res_subset_units = units_res.loc[sources, :]  
                                
                units_res_sources = units_res.loc[sources, :]
                units_res_targets = units_res.loc[targets, :]

                subset_unit_names = [int(unit_name) for unit_name in units_res_subset_units['unit_name'].values] 
                target_unit_names = [int(unit_name) for unit_name in units_res_targets['unit_name'].values]

                correlation_mask  = [True if (unit1 in subset_unit_names and unit2 in subset_unit_names) else False for unit1, unit2 in zip(traj_corr_df['unit1'], traj_corr_df['unit2'])]
                sub_correlations  = traj_corr_df.loc[correlation_mask, 'Pearson_corr'].values
                sub_corr_names    = traj_corr_df.loc[correlation_mask, ['unit1', 'unit2']]
            
                sub_corr_i = [np.where(np.array(target_unit_names) == unit1)[0][0] for unit1 in sub_corr_names.unit1] 
                sub_corr_j = [np.where(np.array(target_unit_names) == unit2)[0][0] for unit2 in sub_corr_names.unit2] 
                
                sub_corr_array  = np.full_like(sub_FN, 0)
                for i, j, corr in zip(sub_corr_i, sub_corr_j, sub_correlations):
                    sub_corr_array[i, j] = corr

                sub_corr_array += sub_corr_array.transpose()

                sub_corr_array   = sub_corr_array[sub_FN != 0]
                sub_FN_no_selfEdges      = sub_FN[sub_FN != 0]
                sub_comp_FN_no_selfEdges = sub_comp_FN[sub_FN != 0]
                FN_df[f'{behavior}'] = sub_FN_no_selfEdges
                if behavior == behaviorComp or f'{behaviorComp}_vs_{behavior}' in completed_comps:
                    continue
                else:
                    completed_comps.append(f'{behavior}_vs_{behaviorComp}')


                tmp_df = pd.DataFrame(data=zip(sub_FN_no_selfEdges - sub_comp_FN_no_selfEdges,
                                               np.tile(units_res_sources['cortical_area'], sub_FN.shape[0]),
                                               np.repeat(completed_comps[-1], sub_FN.size),
                                               np.repeat(sub_basis, sub_FN.size),
                                               sub_corr_array,
                                               ),
                                      columns=['Wji_diff', 'input_area', 'Comparison_key', 'Units Subset', 'pearson_r'])
                weights_df = pd.concat((weights_df, tmp_df))
                
                if (behaviorComp.lower() == gen_test_behavior.lower() and behavior.lower() == 'reach1') or \
                   (behavior.lower() == gen_test_behavior.lower() and behaviorComp.lower() == 'reach1'):
                    tmp_auc_df = pd.DataFrame(data=zip(units_res_subset_units['traj_avgPos_auc'],
                                                       units_res_subset_units['cortical_area'],
                                                       units_res_subset_units['modulation_RO'],
                                                       units_res_subset_units['Reach_frate'],
                                                       units_res_subset_units[f'{gen_test_behavior}_frate'],
                                                       units_res_subset_units['Reach_frate'] / units_res_subset_units[f'{gen_test_behavior}_frate'],
                                                       np.repeat(sub_basis, units_res_subset_units.shape[0]),
                                                       units_res_subset_units['quality']),
                                              columns=['Kinematics_AUC', 
                                                       'cortical_area', 
                                                       'mod_RO', 
                                                       'Reach_frate', 
                                                       f'{gen_test_behavior}_frate',
                                                       'frate_percent_increase',
                                                       'Units Subset', 
                                                       'quality'])
                    
                    auc_df = pd.concat((auc_df, tmp_auc_df))
                
            FN_dataframes[sub_basis] = FN_df[['Rest', 'Spontaneous', 'Locomotion', 
                                              'Retraction', 'Extension', 'Reach1', 
                                              'Reach2']]             
            # if marmcode == 'TY':
            #     # FN_dataframes[sub_basis] = FN_df[['Groomed', 'Rest', 'spontaneous', 'Locomotion', 
            #     #                                   'Arm_movements', 
            #     #                                   'extension', 'retraction', 'reach1', 'reach2']]    
            #     # FN_dataframes[sub_basis] = FN_df[['Groomed', 'Rest', 'spontaneous', 'Locomation', 
            #     #                                   'Climbing', 'Directed_arm_movement', 
            #     #                                   'extension', 'retraction', 'reach1', 'reach2']]    
            #     # FN_dataframes[sub_basis] = FN_df[['Groomed', 'Rest', 'spontaneous', 'Locomation', 
            #     #                                   'Climbing', 'extension', 'retraction', 'reach1', 
            #     #                                   'reach2']]
            #     FN_dataframes[sub_basis] = FN_df[['Rest', 'Spontaneous', 'Locomotion', 
            #                                       'Retraction', 'Extension', 'Reach1', 
            #                                       'Reach2']] 
            # elif marmcode == 'MG':
            #     # FN_dataframes[sub_basis] = FN_df[['Rest', 'spontaneous', 'Locomotion', 
            #     #                                   'Arm_movements', 
            #     #                                   'extension', 'retraction', 'reach1', 'reach2']] 
            #     # FN_dataframes[sub_basis] = FN_df[['Rest', 'spontaneous', 'Locomation', 
            #     #                                   'Climbing', 'Arm_movements', 
            #     #                                   'extension', 'retraction', 'reach1', 'reach2']]                 
            #     FN_dataframes[sub_basis] = FN_df[['Rest', 'spontaneous', 'Locomation', 
            #                                       'Climbing', 'extension', 'retraction', 
            #                                       'reach1', 'reach2']]  
            
            make_FN_similarity_plots(FN_dataframes[sub_basis], sub_basis, 'FigS8_and_9', gen_test_behavior)

    # plot_wji_distributions_for_subsets(weights_df, paperFig = 'Exploratory_Spont_FNs', figname_mod = '', hue_order_FN=hue_order_FN, palette=palette) 
    plot_trajectory_correlation_and_auc_distributions_for_subsets(auc_df, weights_df, paperFig=paperFig, figname_mod = f'_{gen_test_behavior}', hue_order_FN=hue_order_FN, palette=palette)
    plot_modulation_for_subsets(auc_df, paperFig=paperFig, figname_mod = f'_{gen_test_behavior}', hue_order_FN=hue_order_FN, palette=palette)

    return weights_df           

def compute_performance_difference_by_unit(units_res, model_1, model_2):
    
    if model_1[-4:] != '_auc':
        model_1 = model_1 + '_auc'
    if model_2[-4:] != '_auc':
        model_2 = model_2 + '_auc'
    
    cols = [col for col in units_res.columns if 'auc' not in col]
    diff_df = units_res.loc[:, cols]
    
    dist_from_unity = np.abs(-1*units_res[model_1] + 1*units_res[model_2]) / np.sqrt(1**2+1**2)
    
    model_diff = pd.DataFrame(data = units_res[model_2] - units_res[model_1],
                              columns = ['auc_diff'])
    
    model_dist = pd.DataFrame(data = dist_from_unity,
                              columns = ['dist_from_unity'])
    diff_df = pd.concat((diff_df, model_diff, model_dist), axis = 1)
    
    return diff_df

def find_reach_specific_group(diff_df, model_1, model_2, paperFig = 'FigS5', gen_test_behavior=None):
    
    try:
        full_res_model_1 = results_dict[params.best_lead_lag_key]['model_results'][model_1]['AUC']
    except:
        full_res_model_1 = generalization_results[params.best_lead_lag_key]['model_results'][model_1]['AUC']

    try:
        full_res_model_2 = results_dict[params.best_lead_lag_key]['model_results'][model_2]['AUC']
    except:
        full_res_model_2 = generalization_results[params.best_lead_lag_key]['model_results'][model_2]['AUC'] 
    
    nTests = np.min([full_res_model_1.shape[1], full_res_model_2.shape[1]])
    proportion = []
    pval = []
    for unit_model1, unit_model2 in zip(full_res_model_1, full_res_model_2):        
        n2over1 = (unit_model2[:nTests] > unit_model1[:nTests]).sum()
        
        sign_test = binomtest(n2over1, nTests, p = 0.5, alternative='greater')
        proportion.append(np.round(sign_test.proportion_estimate, 2))
        pval.append(np.round(sign_test.pvalue, 4))

    diff_df['generalization_proportion'] = proportion
    diff_df['pval'] = pval
    diff_df[model_1] = full_res_model_1.mean(axis=1)
    diff_df[model_2] = full_res_model_2.mean(axis=1)   

    sorted_diff_df = diff_df.sort_values(by='auc_diff', ascending=False)
    sorted_diff_df['dist_positive_grad'] = np.hstack((np.abs(np.diff(sorted_diff_df['dist_from_unity'])),
                                                      [np.nan]))     
    medFilt_grad = median_filter(sorted_diff_df['dist_positive_grad'], 8) #TODO 9
    lastUnit = np.where(medFilt_grad < 0.1  * np.nanmax(medFilt_grad))[0][0] #TODO 0.075
    top_value_cut = -12 if marmcode=='TY' else -3 
    tmp = sorted_diff_df['dist_positive_grad'].values
    tmp = tmp[~np.isnan(tmp)]
    lastUnit = np.where(medFilt_grad < 0.1 * np.median(np.sort(tmp)[top_value_cut:]))[0][0] #TODO 0.075
    if marmcode=='TY' and gen_test_behavior.lower()=='rest':
        lastUnit = 60 # TODO
    
    plot_mult = 10 if marmcode == 'TY' else 5
    fig, ax = plt.subplots(figsize = plot_params.classifier_figSize, dpi=plot_params.dpi)
    ax.plot(np.arange(diff_df.shape[0]), sorted_diff_df['auc_diff'], linewidth = plot_params.distplot_linewidth)
    ax.plot(np.arange(diff_df.shape[0]), sorted_diff_df['dist_positive_grad']*plot_mult, linewidth = plot_params.distplot_linewidth)
    ax.plot(np.arange(diff_df.shape[0]), medFilt_grad*plot_mult, linewidth = plot_params.distplot_linewidth)
    ax.vlines(lastUnit, 0, np.max([sorted_diff_df['auc_diff'].max(), sorted_diff_df['dist_positive_grad'].max()*2]), 
              'black', linewidth=0.5, linestyle = '--')
    ax.set_xticks([lastUnit, diff_df.shape[0]])
    ax.legend(labels=['AUC difference', 'd(AUC difference)/dUnit', 'Median filter of d/dUnit'],
              bbox_to_anchor=(1.1, 0.5), loc='center left', fontsize = plot_params.tick_fontsize)
    ax.set_xlabel('Units')
    
    sns.despine(ax=ax)
    
    plt.show()

    fig.savefig(os.path.join(plots, paperFig, f'{marmcode}_classifier_selection_{gen_test_behavior}.png'), bbox_inches='tight', dpi=plot_params.dpi)

    # reach_specific_units_byStats = diff_df.index[diff_df['generalization_proportion'] >= .95]    
    # non_specific_units_byStats = diff_df.index[diff_df['generalization_proportion'] < 0.95]    

    reach_specific_units_byStats = sorted_diff_df.index[:lastUnit] 
    non_specific_units_byStats   = sorted_diff_df.index[lastUnit:]
    
    params.reach_specific_thresh[gen_test_behavior] = sorted_diff_df['dist_from_unity'].iloc[lastUnit-1:lastUnit+1].mean()

    return reach_specific_units_byStats, non_specific_units_byStats     

def plot_FN_differences(FN1, FN2, CS_FN1, CS_FN2, CI_FN1, CI_FN2, 
                        paperFig, FN_labels, hue_order_FN=None, palette=None):
    
    set1_list = [FN1, CS_FN1, CI_FN1]
    set2_list = [FN2, CS_FN2, CI_FN2]
    labels = ['Full', 'Reach-Specific', 'Non-Specific']
    
    diff_df = pd.DataFrame()
    for f1, f2, label in zip(set1_list, set2_list, labels):
        diff_data = f1.flatten() - f2.flatten()
        label_list = [label for i in range(diff_data.shape[0])]
        
        tmp_df = pd.DataFrame(data = zip(diff_data, label_list),
                              columns = ['w_diff', 'Units Subset'])
        
        diff_df = pd.concat((diff_df, tmp_df), axis=0, ignore_index=True)
    
    #------------------------------------------------------------------------------------
    
    fig, ax = plt.subplots(figsize = plot_params.distplot_figsize)
    sns.kdeplot(data=diff_df, ax=ax, x='w_diff', 
                hue='Units Subset', hue_order = hue_order_FN, palette=palette,
                cumulative=True, common_norm=False, bw_adjust=0.05, linewidth=plot_params.distplot_linewidth)
    ax.legend(bbox_to_anchor=(1.1, 0.5), loc='center left',
              handles = ax.lines[::-1],
              labels  = hue_order_FN, 
              title_fontsize = plot_params.axis_fontsize,
              fontsize = plot_params.tick_fontsize)
    sns.despine(ax=ax)
    ax.set_yticks([0, 1])
    ax.set_xlabel(f'$W_{{ji}}$ ({FN_labels[0]}) - $W_{{ji}}$ ({FN_labels[1]})')
    plt.show()
    fig.savefig(os.path.join(plots, paperFig, f'{marmcode}_w_diff_cumulative_dist_reachFN1_minus_{FN_labels[1]}'), bbox_inches='tight', dpi=plot_params.dpi)
    med_out = median_test(diff_df.loc[(diff_df['Units Subset'] == 'Reach-Specific') , 'w_diff'], 
                          diff_df.loc[(diff_df['Units Subset'] == 'Non-Specific'), 'w_diff'])
    print(f'Wji-Difference_reachFN1:  reach-spec vs non-spec, p={np.round(med_out[1], 4)}, {med_out[-1][0,0]}a-{med_out[-1][1,0]}b, {med_out[-1][0,1]}a-{med_out[-1][1,1]}b')
    med_out = median_test(diff_df.loc[(diff_df['Units Subset'] == 'Reach-Specific')  , 'w_diff'], 
                          diff_df.loc[(diff_df['Units Subset'] == 'Full')    , 'w_diff'])
    print(f'Wji-Difference_reachFN1:  reach-spec vs full, p={np.round(med_out[1], 4)}, {med_out[-1][0,0]}a-{med_out[-1][1,0]}b, {med_out[-1][0,1]}a-{med_out[-1][1,1]}b')
    med_out = median_test(diff_df.loc[(diff_df['Units Subset'] == 'Non-Specific') , 'w_diff'], 
                          diff_df.loc[(diff_df['Units Subset'] == 'Full')  , 'w_diff'])
    print(f'Wji-Difference_reachFN1:  non-spec vs full, p={np.round(med_out[1], 4)}, {med_out[-1][0,0]}a-{med_out[-1][1,0]}b, {med_out[-1][0,1]}a-{med_out[-1][1,1]}b')
        
    return   

def compute_and_analyze_pathlets(lead_lag_key, model, numplots):
    
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
            
        posTraj_samples.append(posTraj_samp)
        if 'velTraj_samp' in locals():
            velTraj_samples.append(velTraj_samp)
    
    pathDivergence_mean = np.mean(pathDivergence, axis = 0)    
    
    if 'velTraj_mean' not in locals():
        velTraj_mean = []

    return posTraj_mean, velTraj_mean, posTraj_samples, velTraj_samples  

def compute_and_analyze_trajectory_correlations(units_res, posTraj_mean, velTraj_mean, 
                                                electrode_distances, lead_lag_key, FN=None, mode = 'concat', 
                                                reach_specific_units = None, nplots=5, paperFig='unknown'):
    
    lead = float(re.findall(re.compile('lead_[0-9]{1,3}'), lead_lag_key)[0].split('_')[-1]) * 1e-3
    lag  = float(re.findall(re.compile('lag_[0-9]{1,3}' ), lead_lag_key)[0].split('_')[-1]) * 1e-3
    
    pos_corr = np.full_like(electrode_distances, np.nan)
    vel_corr = np.full_like(electrode_distances, np.nan)
    connect  = np.full((electrode_distances.shape[0], electrode_distances.shape[0]), 'xx-xx')
    both_reach_FN_dep = np.full((electrode_distances.shape[0], electrode_distances.shape[0]), False)
    one_reach_FN_dep  = np.full((electrode_distances.shape[0], electrode_distances.shape[0]), False)
    source_reach_FN_dep  = np.full((electrode_distances.shape[0], electrode_distances.shape[0]), False)
    target_reach_FN_dep  = np.full((electrode_distances.shape[0], electrode_distances.shape[0]), False)
    x1, y1 = np.full_like(electrode_distances, np.nan), np.full_like(electrode_distances, np.nan)
    x2, y2 = np.full_like(electrode_distances, np.nan), np.full_like(electrode_distances, np.nan)
    unit1, unit2 = np.full_like(electrode_distances, np.nan), np.full_like(electrode_distances, np.nan)

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
            unit1[i, j] = units_res['unit_name'].iloc[pairs_idx[0]]
            unit2[i, j] = units_res['unit_name'].iloc[pairs_idx[1]]
            connect [i, j] = f'{areas_pair[0]}-{areas_pair[1]}'
            if reach_specific_units is not None:
                if i in reach_specific_units and j in reach_specific_units:
                    both_reach_FN_dep[i, j] = True
                elif i in reach_specific_units: 
                    # one_reach_FN_dep[i, j] = True
                    target_reach_FN_dep[i, j] = True
                elif j in reach_specific_units:
                    source_reach_FN_dep[i, j] = True
                    
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
                ax.w_xaxis.line.set_color('black')
                ax.w_yaxis.line.set_color('black')
                ax.w_zaxis.line.set_color('black')
                ax.view_init(28, 148)
                ax.set_title(f'Units {pair[0]} and {pair[1]}, r = {round(corr, 2)}')
                plt.show()
                
                fig.savefig(os.path.join(plots, 'unknown', f'{marmcode}_corr_pair_pathlets_{pair[0]}_{pair[1]}.png'), bbox_inches='tight', dpi=plot_params.dpi)

    
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
                                  y2[np.triu_indices(nUnits, k=1)],
                                  unit1[np.triu_indices(nUnits, k=1)],
                                  unit2[np.triu_indices(nUnits, k=1)],
                                  both_reach_FN_dep[np.triu_indices(nUnits, k=1)],
                                  target_reach_FN_dep[np.triu_indices(nUnits, k=1)],
                                  source_reach_FN_dep[np.triu_indices(nUnits, k=1)]),
                      columns = ['Pearson_corr', 'Distance', 'Wji', 'Connection', 'x1', 'x2', 'y1', 'y2', 
                                 'unit1', 'unit2', 'both_reach_FN_dependent', 
                                 'target_reach_FN_dependent', 'source_reach_FN_dependent'])

    df['r_squared'] = df['Pearson_corr']**2
    
    return df  

def add_all_FNs_to_annotated_dict(FN, spontaneous_FN, annotated_FN_dict=None, ext_FN=None, ret_FN=None):
    
    if not annotated_FN_dict:
        annotated_FN_dict = dict()
    
    annotated_FN_dict['Reach1'] = FN[0]
    annotated_FN_dict['Reach2'] = FN[1]
    annotated_FN_dict['Spontaneous'] = spontaneous_FN
    
    if ext_FN is not None:
        annotated_FN_dict['extension']  = ext_FN
    if ret_FN is not None:
        annotated_FN_dict['retraction'] = ret_FN    
    
    return annotated_FN_dict

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

def get_single_reach_kinematic_distributions(reaches, plot=False):
    
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
    for reachNum, reach in reaches.iterrows():      
                
        # get event data using container and ndx_pose names from segment_info table following form below:
        # nwb.processing['goal_directed_kinematics'].data_interfaces['moths_s_1_e_004_position']
        event_data      = kin_module.data_interfaces[reach.video_event] 
        
        wrist_kinematics    = event_data.pose_estimation_series[   wrist_label].data[reach.start_idx:reach.stop_idx+1].T
        shoulder_kinematics = event_data.pose_estimation_series[shoulder_label].data[reach.start_idx:reach.stop_idx+1].T
    
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
                
    kin_df = kin_df.loc[~np.isnan(kin_df['speed'])]
    
    return kin_df
    
if __name__ == "__main__":
    
    os.makedirs(plots, exist_ok=True) 
    
    results_dict = load_dict_from_hdf5(pkl_infile.with_suffix('.h5'), top_level_list=False, convert_4d_array_to_list = True)
    results_dict_add_stats = load_dict_from_hdf5(pkl_addstatsfile.with_suffix('.h5'), top_level_list=False, convert_4d_array_to_list = True)
    generalization_results = load_dict_from_hdf5(gen_results_file)
    
    with NWBHDF5IO(nwb_infile, 'r') as io:
        nwb = io.read()
        FN = nwb.scratch[params.FN_key].data[:] 
        spontaneous_FN = nwb.scratch['spontaneous_FN'].data[:]
        reaches_key = [key for key in nwb.intervals.keys() if 'reaching_segments' in key][0]
        units, reaches, kin_module = get_sorted_units_and_apparatus_kinematics_with_metadata(nwb, reaches_key, plot=False)
        kin_df = get_single_reach_kinematic_distributions(reaches)

    behavior_name_map = dict(Climbing='Climbing',
                             Locomation='Locomotion',
                             Locomotion='Locomotion',
                             Rest='Rest',
                             extension_FN='Extension',
                             retraction_FN='Retraction',
                             Directed_arm_movement = 'Arm Movements',
                             Arm_movements = 'Arm Movements',
                             Groomed = 'Groomed',
                             )

        
    with NWBHDF5IO(annotated_nwbfile, 'r') as io_ann:
        annotated_nwb = io_ann.read()
        annotated_FN_dict = dict() 
        for behavior in annotated_nwb.scratch.keys():
            if behavior not in ['all_reach_FN', 'split_reach_FNs', 'spontaneous_FN', 'spikes_chronological', 'split_FNs_reach_sets']:
                annotated_FN_dict[behavior_name_map[behavior]] = annotated_nwb.scratch[behavior].data[:]
        # ext_FN = annotated_nwb.scratch[ 'extension_FN'].data[:] 
        # ret_FN = annotated_nwb.scratch['retraction_FN'].data[:] 

    with NWBHDF5IO(single_reach_nwbfile, 'r') as io_single:
        single_nwb = io_single.read()
        single_reaches_FN_dict = dict() 
        for key in single_nwb.scratch.keys():
            if key not in ['all_reach_FN', 'split_reach_FNs', 'spontaneous_FN', 'spikes_chronological', 'split_FNs_reach_sets', 'extension_FN', 'retraction_FN']:
                single_reaches_FN_dict[key] = single_nwb.scratch[key].data[:]
    
    annotated_FN_dict = add_all_FNs_to_annotated_dict(FN, spontaneous_FN, annotated_FN_dict)

    summarize_model_results(units=None, lead_lag_keys = params.best_lead_lag_key)
    
    units_res = results_dict[params.best_lead_lag_key]['all_models_summary_results']
    units_res = add_in_weight_to_units_df(units_res, FN.copy())
    
    units_res = add_modulation_data_to_units_df(units_res, paperFig='Revision_Plots', behavior_name_map=behavior_name_map)
    units_res = add_generalization_experiments_to_units_df(units_res, generalization_results[params.best_lead_lag_key])
    
    units_res = add_neuron_classifications(units_res)
    
    # if marmcode == 'TY':
        # units_res = add_icms_results_to_units_results_df(units_res, params.icms_res)
    
    electrode_distances = get_interelectrode_distances_by_unit(units_res, array_type='utah')
        
    _, cmin, cmax = plot_functional_networks(FN, units_res, FN_key = params.FN_key, paperFig='Fig1')
    # _, _, _       = plot_functional_networks(spontaneous_FN, units_res, FN_key ='spontaneous_FN', paperFig='Fig1', cmin=cmin, cmax=cmax)
    # _, _, _       = plot_functional_networks(ext_FN, units_res, FN_key ='Extension' , paperFig='extension_retraction_FNs', cmin=cmin, cmax=cmax)
    # _, _, _       = plot_functional_networks(ret_FN, units_res, FN_key ='Retraction', paperFig='extension_retraction_FNs', cmin=cmin, cmax=cmax)
    for behavior, behavior_FN in annotated_FN_dict.items():
        _, _, _ = plot_functional_networks(behavior_FN, units_res, FN_key = behavior, paperFig='FigS8_and_9', cmin=cmin, cmax=cmax)
    for key, single_FN in single_reaches_FN_dict.items():
        if 'half' in key:
            _, _, _ = plot_functional_networks(single_FN, units_res, FN_key = key, paperFig='Revision_Plots', cmin=cmin, cmax=cmax)
    #     else:
    #         reach_kin_df = kin_df.loc[kin_df['reach'] == int(key.split('reach')[-1]), 'speed']
    #         _, _, _ = plot_functional_networks(single_FN, units_res, FN_key = key, paperFig='Exploratory_Spont_FNs', cmin=cmin, cmax=cmax, kin_df = reach_kin_df)

    ymin, ymax = plot_weights_versus_interelectrode_distances(FN, spontaneous_FN, 
                                                              electrode_distances, 
                                                              annotated_FN_dict=annotated_FN_dict,
                                                              paperFig='FigS8_and_9',
                                                              palette=dict(spont='annot_FN_palette',
                                                                           ext_ret = 'ext_ret_FN_palette'),
                                                              )
    

    full_gen_behavior_map = dict(armMove='Arm Movements', 
                                  spont='Spontaneous', 
                                  climbing='Climbing', 
                                  rest='Rest',
                                  locomotion='Locomotion')

    style_key = None
    # for gen_exp in units_res.columns:
    #     if 'reach_test_FN_auc' not in gen_exp or 'traj_avgPos' not in gen_exp:
    #         continue
    #     print(gen_exp)
    #     diff_df = compute_performance_difference_by_unit(units_res, gen_exp, 'traj_avgPos_reach_FN_auc')   
    #     # reach_specific_units = diff_df.index[(diff_df.auc_diff > 0) & (diff_df.dist_from_unity > params.reach_specific_thresh) & (diff_df.dist_from_unity < 0.04)]
    #     reach_specific_units, non_specific_units = find_reach_specific_group(diff_df,
    #                                                                          gen_exp.split('_auc')[0], 
    #                                                                          'traj_avgPos_reach_FN',
    #                                                                          paperFig = 'Exploratory_Spont_FNs',
    #                                                                          gen_test_behavior=gen_exp.split('avgPos_')[-1].split('_train')[0])
        
    #     reach_specific_units_res = units_res.loc[reach_specific_units, :]
    #     non_specific_units_res   = units_res.loc[non_specific_units, :]
        
    #     units_res['Units Subset'] = ['Non-Specific' for idx in range(units_res.shape[0])]
    #     units_res.loc[reach_specific_units, 'Units Subset'] = ['Reach-Specific' for idx in range(reach_specific_units.size)]

    #     if gen_exp.split('avgPos_')[-1].split('_train')[0] in full_gen_behavior_map.keys():
    #         gen_test_tmp = full_gen_behavior_map[gen_exp.split('avgPos_')[-1].split('_train')[0]]
    #     else:
    #         gen_test_tmp = None
            
    #     plot_model_auc_comparison   (units_res, gen_exp, 'traj_avgPos_reach_FN_auc', 
    #                                  minauc = 0.45, maxauc = 0.8, hue_key='Units Subset', hue_order=('Reach-Specific', 'Non-Specific'), 
    #                                  style_key=style_key, targets = None, paperFig='Exploratory_Spont_FNs', palette=fig6_palette,
    #                                  gen_test_behavior=gen_test_tmp)
        
    #     # plot_model_auc_comparison   (units_res, 'traj_avgPos_spont_train_reach_test_FN_auc', 'traj_avgPos_reach_FN_auc', 
    #     #                              minauc = 0.45, maxauc = 0.8, hue_key='Units Subset', hue_order=('Reach-Specific', 'Non-Specific'), 
    #     #                              style_key=style_key, targets = None, paperFig='Exploratory_Spont_FNs', palette=fig6_palette,
    #     #                              gen_test_behavior=full_gen_behavior_map['spont'])
        
        
    # diff_df = compute_performance_difference_by_unit(units_res, 'traj_avgPos_spont_train_reach_test_FN_auc', 'traj_avgPos_reach_FN_auc')   
    # # reach_specific_units = diff_df.index[(diff_df.auc_diff > 0) & (diff_df.dist_from_unity > params.reach_specific_thresh) & (diff_df.dist_from_unity < 0.04)]
    # reach_specific_units, non_specific_units = find_reach_specific_group(diff_df,
    #                                                                      'traj_avgPos_spont_train_reach_test_FN', 
    #                                                                      'traj_avgPos_reach_FN',
    #                                                                      paperFig = 'FigS6')

    gen_test_behavior = 'spont' 
    params.test_behavior='spont'
    full_test_behavior = full_gen_behavior_map[gen_test_behavior]
    diff_df = compute_performance_difference_by_unit(units_res, f'traj_avgPos_{gen_test_behavior}_train_reach_test_FN_auc', 'traj_avgPos_reach_FN_auc')   
    # reach_specific_units = diff_df.index[(diff_df.auc_diff > 0) & (diff_df.dist_from_unity > params.reach_specific_thresh) & (diff_df.dist_from_unity < 0.04)]
    reach_specific_units, non_specific_units = find_reach_specific_group(diff_df,
                                                                         f'traj_avgPos_{gen_test_behavior}_train_reach_test_FN', 
                                                                         'traj_avgPos_reach_FN',
                                                                         paperFig = 'FigS10_and_11',
                                                                         gen_test_behavior=gen_test_behavior)
    
    reach_specific_units_res = units_res.loc[reach_specific_units, :]
    non_specific_units_res   = units_res.loc[non_specific_units, :]
    
    units_res['Units Subset'] = ['Non-Specific' for idx in range(units_res.shape[0])]
    units_res.loc[reach_specific_units, 'Units Subset'] = ['Reach-Specific' for idx in range(reach_specific_units.size)]
    plot_model_auc_comparison   (units_res, 'traj_avgPos_reach_train_spont_test_FN_auc', 'traj_avgPos_spontaneous_FN_auc', 
                                  minauc = 0.45, maxauc = 0.8, hue_key='Units Subset', hue_order=('Reach-Specific', 'Non-Specific'), 
                                  style_key=style_key, targets = None, paperFig='Response_Only', palette=fig6_palette,
                                  gen_test_behavior='spont')

    plot_model_auc_comparison   (units_res, 'traj_avgPos_spont_train_reach_test_FN_auc', 'traj_avgPos_reach_FN_auc', 
                                  minauc = 0.45, maxauc = 0.8, hue_key='neuron_type', style_key='neuron_type',
                                  hue_order=('WS', 'NS', 'mua', 'unclassified'), style_order = ('WS', 'NS', 'mua', 'unclassified'),
                                  targets = None, paperFig='Response_Only', palette=None,
                                  gen_test_behavior='spont', extra_label='_NeuronClasses')

    # for gen_test_behavior in ['climbing']:
    #     params.test_behavior=gen_test_behavior
    #     full_test_behavior = full_gen_behavior_map[gen_test_behavior]
    #     diff_df = compute_performance_difference_by_unit(units_res, f'traj_avgPos_{gen_test_behavior}_train_reach_test_FN_auc', 'traj_avgPos_reach_FN_auc')   
    #     # reach_specific_units = diff_df.index[(diff_df.auc_diff > 0) & (diff_df.dist_from_unity > params.reach_specific_thresh) & (diff_df.dist_from_unity < 0.04)]
    #     reach_specific_units, non_specific_units = find_reach_specific_group(diff_df,
    #                                                                          f'traj_avgPos_{gen_test_behavior}_train_reach_test_FN', 
    #                                                                          'traj_avgPos_reach_FN',
    #                                                                          paperFig = 'FigS10_and_11',
    #                                                                          gen_test_behavior=gen_test_behavior)
        
    #     reach_specific_units_res = units_res.loc[reach_specific_units, :]
    #     non_specific_units_res   = units_res.loc[non_specific_units, :]
        
    #     units_res['Units Subset'] = ['Non-Specific' for idx in range(units_res.shape[0])]
    #     units_res.loc[reach_specific_units, 'Units Subset'] = ['Reach-Specific' for idx in range(reach_specific_units.size)]  
        
    #     plot_model_auc_comparison   (units_res, f'traj_avgPos_{gen_test_behavior}_train_reach_test_FN_auc', 'traj_avgPos_reach_FN_auc', 
    #                                   minauc = 0.45, maxauc = 0.8, hue_key='neuron_type', style_key='neuron_type',
    #                                   hue_order=('WS', 'NS', 'mua', 'unclassified'), style_order = ('WS', 'NS', 'mua', 'unclassified'),
    #                                   targets = None, paperFig='Response_Only', palette=None,
    #                                   gen_test_behavior='spont', extra_label='_NeuronClasses')

    for gen_test_behavior in ['rest', 'locomotion']: 
        params.test_behavior=gen_test_behavior
        full_test_behavior = full_gen_behavior_map[gen_test_behavior]
        diff_df = compute_performance_difference_by_unit(units_res, f'traj_avgPos_{gen_test_behavior}_train_reach_test_FN_auc', 'traj_avgPos_reach_FN_auc')   
        # reach_specific_units = diff_df.index[(diff_df.auc_diff > 0) & (diff_df.dist_from_unity > params.reach_specific_thresh) & (diff_df.dist_from_unity < 0.04)]
        reach_specific_units, non_specific_units = find_reach_specific_group(diff_df,
                                                                             f'traj_avgPos_{gen_test_behavior}_train_reach_test_FN', 
                                                                             'traj_avgPos_reach_FN',
                                                                             paperFig = 'FigS10_and_11',
                                                                             gen_test_behavior=gen_test_behavior)
        
        reach_specific_units_res = units_res.loc[reach_specific_units, :]
        non_specific_units_res   = units_res.loc[non_specific_units, :]
        
        units_res['Units Subset'] = ['Non-Specific' for idx in range(units_res.shape[0])]
        units_res.loc[reach_specific_units, 'Units Subset'] = ['Reach-Specific' for idx in range(reach_specific_units.size)]
        
        plot_model_auc_comparison   (units_res, f'traj_avgPos_{gen_test_behavior}_train_reach_test_FN_auc', 'traj_avgPos_reach_FN_auc', 
                                     minauc = 0.45, maxauc = 0.8, hue_key='Units Subset', hue_order=('Reach-Specific', 'Non-Specific'), 
                                     style_key=style_key, targets = None, paperFig='FigS10_and_11', palette=fig6_palette,
                                     gen_test_behavior=gen_test_behavior, full_test_behavior = full_test_behavior)
        
        plot_model_auc_comparison   (units_res, 'traj_avgPos_spont_train_reach_test_FN_auc', 'traj_avgPos_reach_FN_auc', 
                                      minauc = 0.45, maxauc = 0.8, hue_key='Units Subset', hue_order=('Reach-Specific', 'Non-Specific'), 
                                      style_key=style_key, targets = None, paperFig='FigS10_and_11', palette=fig6_palette,
                                      gen_test_behavior='spont')
    
    
        
        subset = 'both'
        # reach_specific_reachFNs, _, _ = plot_functional_networks(FN, units_res, FN_key = params.FN_key, cmin=cmin, cmax=cmax, 
        #                                                          subset_idxs = reach_specific_units, subset_type=subset, paperFig='Fig5', gen_test_behavior=gen_test_behavior)
        # reach_specific_spontFN , _, _ = plot_functional_networks(spontaneous_FN, units_res, FN_key ='spontaneous_FN', cmin=cmin, cmax=cmax, 
        #                                                          subset_idxs = reach_specific_units, subset_type=subset, paperFig='Fig5', gen_test_behavior=gen_test_behavior)
        # # reach_specific_extFN   , _, _ = plot_functional_networks(ext_FN, units_res, FN_key ='extension_FN', cmin=cmin, cmax=cmax, 
        # #                                                          subset_idxs = reach_specific_units, subset_type=subset, paperFig='extension_retraction_FNs', gen_test_behavior=gen_test_behavior)
        # # reach_specific_retFN   , _, _ = plot_functional_networks(ret_FN, units_res, FN_key ='retraction_FN', cmin=cmin, cmax=cmax, 
        # #                                                          subset_idxs = reach_specific_units, subset_type=subset, paperFig='extension_retraction_FNs', gen_test_behavior=gen_test_behavior)
        # for behavior, behavior_FN in annotated_FN_dict.items():
        #     _, _, _ = plot_functional_networks(behavior_FN, units_res, FN_key = behavior, paperFig='Revision_Plots', 
        #                                        subset_idxs = reach_specific_units, subset_type=subset, cmin=cmin, cmax=cmax, gen_test_behavior=gen_test_behavior)
    
        # non_specific_reachFNs, _, _ = plot_functional_networks(FN, units_res, FN_key = params.FN_key, cmin=cmin, cmax=cmax, 
        #                                                        subset_idxs = non_specific_units, subset_type=subset, paperFig='Fig5', gen_test_behavior=gen_test_behavior)
        # non_specific_spontFN , _, _ = plot_functional_networks(spontaneous_FN, units_res, FN_key ='spontaneous_FN', cmin=cmin, cmax=cmax, 
        #                                                        subset_idxs = non_specific_units, subset_type=subset, paperFig='Fig5', gen_test_behavior=gen_test_behavior)
        # # non_specific_extFN   , _, _ = plot_functional_networks(ext_FN, units_res, FN_key ='extension_FN', cmin=cmin, cmax=cmax, 
        # #                                                        subset_idxs = non_specific_units, subset_type=subset, paperFig='extension_retraction_FNs', gen_test_behavior=gen_test_behavior)
        # # non_specific_retFN   , _, _ = plot_functional_networks(ret_FN, units_res, FN_key ='retraction_FN', cmin=cmin, cmax=cmax, 
        # #                                                        subset_idxs = non_specific_units, subset_type=subset, paperFig='extension_retraction_FNs', gen_test_behavior=gen_test_behavior)
        # for behavior, behavior_FN in annotated_FN_dict.items():
        #     _, _, _ = plot_functional_networks(behavior_FN, units_res, FN_key = behavior, paperFig='Revision_Plots', 
        #                                        subset_idxs = non_specific_units, subset_type=subset, cmin=cmin, cmax=cmax, gen_test_behavior=gen_test_behavior)
    
        posTraj_mean, velTraj_mean, posTraj_samples, velTraj_samples = compute_and_analyze_pathlets(params.best_lead_lag_key, 'traj_avgPos', numplots = None)
        traj_corr_df = compute_and_analyze_trajectory_correlations(units_res, posTraj_mean, velTraj_mean, 
                                                                   electrode_distances, params.best_lead_lag_key, 
                                                                   FN = FN, mode='concat', paperFig='Revision_Plots',
                                                                   reach_specific_units = reach_specific_units, nplots=None)
        
        # plot_covariability_speed_and_FNs(single_reaches_FN_dict, kin_df, units_res, reach_specific_units, non_specific_units, paperFig='Revision_Plots')    
        
        # in_out_weights_df = plot_in_and_out_weights_of_functional_groups(units_res, 
        #                                                                  annotated_FN_dict,
        #                                                                  outside_group_only = True,
        #                                                                  subset_idxs = reach_specific_units, 
        #                                                                  subset_basis=['Reach-Specific', 'Non-Specific', 'Full'], 
        #                                                                  gen_test_behavior=full_test_behavior,
        #                                                                  hue_order_FN=['Reach-Specific', 'Non-Specific', 'Full'], 
        #                                                                  palette=fig6_palette)
    
        weights_df = examine_all_annotated_FNs(units_res, 
                                               annotated_FN_dict,
                                               traj_corr_df = traj_corr_df,
                                               subset_idxs = reach_specific_units, 
                                               sub_type = 'both', 
                                               subset_basis=['Reach-Specific', 'Non-Specific', 'Full'], 
                                               gen_test_behavior=full_test_behavior,
                                               hue_order_FN=['Reach-Specific', 'Non-Specific', 'Full'], palette=fig6_palette,
                                               paperFig = 'FigS10_and_11')
    
    
        plot_result_on_channel_map(units_res, jitter_radius = .2, hue_key='Units Subset', 
                                   size_key = None,  title_type = 'hue', style_key=style_key,
                                   paperFig='Revision_Plots', hue_order = ['Reach-Specific', 'Non-Specific'], 
                                   palette=fig6_palette, sizes=None, s=13, gen_test_behavior=gen_test_behavior)
    
        
        completed_comps = []
        for (behavior1, FN1), (behavior2, FN2) in product(annotated_FN_dict.items(), annotated_FN_dict.items()):
            # if behavior1 == behavior2 or f'{behavior2}_vs_{behavior1}' in completed_comps:
            #     continue
            # else:
            #     completed_comps.append(f'{behavior1}_vs_{behavior2}')
            if not (behavior1.lower() == 'reach1' and behavior2.lower() == full_test_behavior.lower()):
                continue

            
            CS_FN1, _, _ = plot_functional_networks(FN1, units_res, FN_key = behavior1, paperFig='FigS10_and_11', gen_test_behavior=behavior2,
                                                    subset_idxs = reach_specific_units, subset_type=subset, cmin=cmin, cmax=cmax)
            CS_FN2, _, _ = plot_functional_networks(FN2, units_res, FN_key = behavior2, paperFig='FigS10_and_11', gen_test_behavior=behavior2,
                                                    subset_idxs = reach_specific_units, subset_type=subset, cmin=cmin, cmax=cmax)
            CI_FN1, _, _ = plot_functional_networks(FN1, units_res, FN_key = behavior1, paperFig='FigS10_and_11', gen_test_behavior=behavior2,
                                                    subset_idxs = non_specific_units, subset_type=subset, cmin=cmin, cmax=cmax)
            CI_FN2, _, _ = plot_functional_networks(FN2, units_res, FN_key = behavior2, paperFig='FigS10_and_11', gen_test_behavior=behavior2,
                                                    subset_idxs = non_specific_units, subset_type=subset, cmin=cmin, cmax=cmax)
            
            plot_FN_differences(FN1, FN2, CS_FN1[0], CS_FN2[0], CI_FN1[0], CI_FN2[0], 
                                paperFig='FigS10_and_11', FN_labels=[behavior1, behavior2], 
                                hue_order_FN = ['Reach-Specific', 'Non-Specific', 'Full'], palette=fig6_palette)

                                    

