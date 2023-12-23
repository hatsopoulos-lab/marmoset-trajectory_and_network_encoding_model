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
import seaborn as sns
from itertools import product
from scipy.stats import binomtest, ttest_rel, ttest_ind, mannwhitneyu, linregress, pearsonr, median_test
from scipy.integrate import cumtrapz
from scipy.integrate import simps
from mpl_toolkits import mplot3d 
from scipy.ndimage import median_filter
from importlib import sys
from pynwb import NWBHDF5IO
import ndx_pose
from pathlib import Path

script_directory = Path(os.path.dirname(os.path.abspath(sys.argv[0])))
code_path = script_directory.parent.parent.parent / 'clean_final_analysis/'
data_path = script_directory.parent.parent / 'data' / 'demo'

sys.path.insert(0, str(code_path))
from utils import get_interelectrode_distances_by_unit, load_dict_from_hdf5, save_dict_to_hdf5

marmcode='MG'
other_marm = None #'MG' #None
fig_mode='paper'
save_kinModels_pkl = False
skip_network_permutations = False
demo = True
show_plots=False

pkl_in_tag  = 'network_models_created'
pkl_out_tag = 'network_models_summarized' 
pkl_add_stats_tag = 'kinematic_models_summarized' 

if marmcode=='TY':
    nwb_infile   = data_path / 'TY' / 'TY20210211_freeAndMoths-003_resorted_20230612_DM_with_functional_networks.nwb' 
    modulation_base = nwb_infile.parent / nwb_infile.stem.split('_with_functional_networks')[0]
elif marmcode=='MG':
    nwb_infile   = data_path / 'MG' / 'MG20230416_1505_mothsAndFree-002_processed_DM_with_functional_networks.nwb'
    modulation_base = data_path / 'MG' / nwb_infile.stem.split('_with_functional_networks')[0]
    
pkl_infile       = nwb_infile.parent / f'{nwb_infile.stem.split("_with_functional_networks")[0]}_{pkl_in_tag}.pkl'
pkl_outfile      = nwb_infile.parent / f'{nwb_infile.stem.split("_with_functional_networks")[0]}_{pkl_out_tag}.pkl'
pkl_addstatsfile = nwb_infile.parent / f'{nwb_infile.stem.split("_with_functional_networks")[0]}_{pkl_add_stats_tag}.pkl'

dataset_code = pkl_infile.stem.split('_')[0]
if fig_mode == 'paper':
    plots = script_directory.parent / 'plots' / dataset_code
elif fig_mode == 'pres':
    plots = script_directory.parent / 'presentation_plots' / dataset_code


try: 
    plt.get_cmap('FN_palette')
except:
    cmap_orig = plt.get_cmap("Paired")
    FN_colors = cmap_orig([1, 0, 2])
    cmap = colors.ListedColormap(FN_colors)
    colormaps.register(cmap, name="FN_palette")

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
    
    figures_list = ['Fig1', 'Fig2', 'Fig3', 'Fig4', 'Fig5', 'Fig6', 'FigS1',  'FigS2',  'FigS3',  'FigS4', 'FigS5', 'FigS6', 'FigS7', 'unknown']

    mostUnits_FN = 175

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
        classifier_figSize = (3, 3)

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
                               paperFig='unknown', hue_order = None, palette=None, sizes=None, s=None):
    
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
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    fig_base = os.path.join(plots, paperFig)
    fig.savefig(os.path.join(fig_base, f'{title}_{hue_key}_on_array_map.png'), bbox_inches='tight', dpi=plot_params.dpi)    
              
def plot_model_auc_comparison(units_res, x_key, y_key, minauc = 0.5, maxauc = 1.0, hue_key='W_in', 
                              style_key='cortical_area', targets=None, col_key=None, hue_order=None, 
                              col_order=None, style_order=None, paperFig='unknown', asterisk='', palette=None):
    
    if x_key[-4:] != '_auc':
        x_key = x_key + '_auc'
    if y_key[-4:] != '_auc':
        y_key = y_key + '_auc'
    
    try:
        xlabel = [f'{lab} AUC' for lab, key in zip(['Trajectory', 'Full kinematics', 'Velocity', 'Short kinematics', 'Kinematics + reachFN', 'Kinematics + spontaneousFN \ngeneralization'], 
                                                   ['traj', 'traj_avgPos', 'shortTraj', 'shortTraj_avgPos', 'traj_avgPos_reach_FN', 'traj_avgPos_spont_train_reach_test_FN']) if f'{key}_auc' == x_key][0]
    except:
        xlabel = x_key
    
    try:
        ylabel = [f'{lab} AUC' for lab, key in zip(['Trajectory', 'Full kinematics', 'Velocity', 'Short kinematics', 'Kinematics + reachFN', 'Kinematics + spontaneousFN \ngeneralization'], 
                                                   ['traj', 'traj_avgPos', 'shortTraj', 'shortTraj_avgPos', 'traj_avgPos_reach_FN', 'traj_avgPos_spont_train_reach_test_FN']) if f'{key}_auc' == y_key][0]
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
        
        if 'spont_train_reach_test_FN' in x_key:
            ax.plot(np.arange(minauc, maxauc, 0.05), np.arange(minauc, maxauc, 0.05) + params.reach_specific_thresh*np.sqrt(2), 
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
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    if paperFig == 'unknown':
        fig_base = os.path.join(plots, paperFig, 'auc_comparison')
    else:
        fig_base = os.path.join(plots, paperFig)
    
    if asterisk != '':
        plot_name += '_filtered'

    fig.savefig(os.path.join(fig_base, plot_name + '.png'), bbox_inches='tight', dpi=plot_params.dpi)

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


def plot_functional_networks(FN, units_res, FN_key = 'split_reach_FNs', cmin=None, cmax=None, subset_idxs = None, subset_type='both', paperFig='unknown'):
    
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
        
        fig, ax = plt.subplots(figsize=fsize, dpi = plot_params.dpi)
        
        sns.heatmap(network_copy,ax=ax,cmap= 'viridis',square=True, norm=colors.PowerNorm(0.5, vmin=cmin, vmax=cmax)) # norm=colors.LogNorm(vmin=cmin, vmax=cmax)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels(xlabels)
        ax.set_yticklabels(ylabels)
        ax.set_title(title, fontsize=plot_params.axis_fontsize)
        ax.set_ylabel('Target unit', fontsize=plot_params.axis_fontsize)
        ax.set_xlabel('Source unit' , fontsize=plot_params.axis_fontsize)
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        title = title.replace('\n', '_')
        fig.savefig(os.path.join(plots, paperFig, 
                                 f'{marmcode}_functional_network_{title.replace(" ", "_").replace("-", "_")}.png'), bbox_inches='tight', dpi=plot_params.dpi)
    
        plt.hist(network_copy.flatten(), bins = 30)
        if show_plots:
            plt.show()
        else:
            plt.close()
        
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

def plot_weights_versus_interelectrode_distances(FN, spontaneous_FN, electrode_distances, 
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
    if show_plots:
        plt.show()
    else:
        plt.close()
        
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

def add_modulation_data_to_units_df(units_res):

    modulation_df = pd.read_hdf(f'{modulation_base}_modulationData.h5')
    mask = [True if int(uName) in units_res.unit_name.astype(int).values else False for uName in modulation_df.unit_name.values]
    modulation_df = modulation_df.loc[mask, :]
    modulation_df.reset_index(drop=True, inplace=True)

    average_rates_df = pd.read_hdf(f'{modulation_base}_average_firing_rates.h5')  

    for met in modulation_df.columns[:6]:
        units_res[met] = modulation_df[met]

    units_res['reach_frate'] = average_rates_df['Reach']
    units_res['spont_frate'] = average_rates_df['Spontaneous']
    units_res['percent_frate_increase'] = average_rates_df['Reach'] / average_rates_df['Spontaneous']

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
    fig.savefig(os.path.join(plots, paperFig, f'{marmcode}_shuffled_network_auc_loss_summary_figure_{comparison_model}_{target_string}_alpha_pt0{int(alpha*100)}.png'), bbox_inches='tight', dpi=plot_params.dpi)    

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
            if show_plots:
                plt.show()
            else:
                plt.close()        
            fig.savefig(os.path.join(plot_save_dir, f'{figname_mod[1:]}_{marmcode}_distribution_{metric}_histogram_highlighted_with_reachspecific{figname_mod}.png'), bbox_inches='tight', dpi=plot_params.dpi)    


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
    if show_plots:
        plt.show()
    else:
        plt.close()        
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
    if show_plots:
        plt.show()
    else:
        plt.close()        
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
    if show_plots:
        plt.show()
    else:
        plt.close()    
    fig.savefig(os.path.join(plots, paperFig, f'{marmcode}_distribution_pearson_r_huekey_UnitsSubset{figname_mod}.png'), bbox_inches='tight', dpi=plot_params.dpi)    
    
    #---------------------------------------------------------------------------------
    
    med_out = median_test(weights_df.loc[weights_df['Units Subset'] == 'Reach-Specific', 'pearson_r_squared'], 
                          weights_df.loc[weights_df['Units Subset'] == 'Non-Specific', 'pearson_r_squared'])
    print(f'pearson_r_squared{figname_mod}: reach v non, p={np.round(med_out[1], 4)}, {med_out[-1][0,0]}a-{med_out[-1][1,0]}b, {med_out[-1][0,1]}a-{med_out[-1][1,1]}b')
    med_out = median_test(weights_df.loc[weights_df['Units Subset'] == 'Reach-Specific', 'pearson_r_squared'], 
                          weights_df.loc[weights_df['Units Subset'] == 'Full', 'pearson_r_squared'])
    print(f'pearson_r_squared{figname_mod}: reach v full, p={np.round(med_out[1], 4)}, {med_out[-1][0,0]}a-{med_out[-1][1,0]}b, {med_out[-1][0,1]}a-{med_out[-1][1,1]}b')
    med_out = median_test(weights_df.loc[weights_df['Units Subset'] == 'Non-Specific', 'pearson_r_squared'], 
                          weights_df.loc[weights_df['Units Subset'] == 'Full', 'pearson_r_squared'])
    print(f'pearson_r_squared{figname_mod}: non v full, p={np.round(med_out[1], 4)}, {med_out[-1][0,0]}a-{med_out[-1][1,0]}b, {med_out[-1][0,1]}a-{med_out[-1][1,1]}b')
    
        
    fig, ax = plt.subplots(figsize=plot_params.distplot_figsize)
    sns.kdeplot(ax=ax, data=weights_df, palette=palette, linewidth=plot_params.distplot_linewidth,
                x='pearson_r_squared', hue='Units Subset', common_norm=False, bw_adjust=0.4, cumulative=False, hue_order=hue_order_FN)
    ax.legend().remove()
    sns.despine(ax=ax)
    if show_plots:
        plt.show()
    else:
        plt.close()    
    fig.savefig(os.path.join(plots, paperFig, f'{marmcode}_distribution_pearson_r_squared_huekey_UnitsSubset{figname_mod}.png'), bbox_inches='tight', dpi=plot_params.dpi)    

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
        if show_plots:
            plt.show()
        else:
            plt.close()
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
        if show_plots:
            plt.show()
        else:
            plt.close()
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
                if show_plots:
                    plt.show()
                else:
                    plt.close()
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
    

def plot_distributions_after_source_props(units_res, electrode_distances, 
                                          traj_corr_df, FN_sets = [], subset_idxs = None, 
                                          sub_type='both', subset_basis=['Reach-Specific'], 
                                          good_only=False, plot_auc_matched = True,
                                          hue_order_FN=None, palette=None):
    
    spontFN_for_diff = FN_sets[1][1]
    weights_df = pd.DataFrame()
    auc_df = pd.DataFrame()
    for sub_basis, FN_set in product(subset_basis, FN_sets):
        
        sub_idxs = subset_idxs.copy()
        
        if 'Non-' in sub_basis:
            sub_idxs = np.setdiff1d(np.array(range(units_res.shape[0])), sub_idxs)
        elif sub_basis == 'Full':
            sub_idxs = np.array(range(units_res.shape[0]))
        
        FN_key = FN_set[0]
        FN_tmp = FN_set[1]
        
        # FN_mean = FN_tmp.mean(axis=0) if FN_tmp.ndim == 3 else FN_tmp.copy()
        if np.ndim(FN_tmp) < 3:
            FN_tmp = np.expand_dims(FN_tmp, 0)
            FN_names = ['spontaneousFN']
        else:
            FN_names = ['reachFN1', 'reachFN2']
            
        for FN, FN_name in zip(FN_tmp, FN_names):
            if sub_type == 'both':
                targets, sources, sub_FN, sub_spontFN = sub_idxs, sub_idxs, FN[np.ix_(sub_idxs, sub_idxs)], spontFN_for_diff[np.ix_(sub_idxs, sub_idxs)]
                sub_dist = electrode_distances[np.ix_(sub_idxs, sub_idxs)]
                units_res_subset_units = units_res.loc[sources, :]  
            elif sub_type == 'target':
                targets, sources, sub_FN, sub_spontFN = sub_idxs, np.arange(FN.shape[1]), FN[np.ix_(sub_idxs, range(FN.shape[1]))], spontFN_for_diff[np.ix_(sub_idxs, range(FN.shape[1]))]
                sub_dist = electrode_distances[np.ix_(sub_idxs, range(FN.shape[1]))]
                units_res_subset_units = units_res.loc[targets, :]
            elif sub_type == 'source':
                targets, sources, sub_FN, sub_spontFN = np.arange(FN.shape[0]), sub_idxs, FN[np.ix_(range(FN.shape[0]), sub_idxs)], spontFN_for_diff[np.ix_(range(FN.shape[0]), sub_idxs)]
                sub_dist = electrode_distances[np.ix_(range(FN.shape[0]), sub_idxs)]
                units_res_subset_units = units_res.loc[sources, :]
        
            units_res_targets = units_res.loc[targets, :]
            units_res_sources = units_res.loc[sources, :]
            
            subset_unit_names = [int(unit_name) for unit_name in units_res_subset_units['unit_name'].values] 
            target_unit_names = [int(unit_name) for unit_name in units_res_targets['unit_name'].values]
            source_unit_names = [int(unit_name) for unit_name in units_res_sources['unit_name'].values]
    
            correlation_mask  = [True if (unit1 in subset_unit_names and unit2 in subset_unit_names) else False for unit1, unit2 in zip(traj_corr_df['unit1'], traj_corr_df['unit2'])]
            sub_correlations  = traj_corr_df.loc[correlation_mask, 'Pearson_corr'].values
            sub_corr_names    = traj_corr_df.loc[correlation_mask, ['unit1', 'unit2']]
        
            sub_corr_i = [np.where(np.array(target_unit_names) == unit1)[0][0] for unit1 in sub_corr_names.unit1] 
            sub_corr_j = [np.where(np.array(target_unit_names) == unit2)[0][0] for unit2 in sub_corr_names.unit2] 
            
            sub_corr_array  = np.full_like(sub_FN, 0)
            sub_higher_auc  = np.full_like(sub_corr_array, 0)
            sub_lower_auc   = np.full_like(sub_corr_array, 0)
            sub_average_auc = np.full_like(sub_corr_array, 0)
            for i, j, corr, unit1, unit2 in zip(sub_corr_i, sub_corr_j, sub_correlations, sub_corr_names['unit1'], sub_corr_names['unit2']):
                sub_corr_array[i, j] = corr
                
                unit_mask = [True if int(unit_name) in [int(unit1), int(unit2)] else False for unit_name in units_res['unit_name'].values]
                units_pair_res = units_res.loc[unit_mask, f'{params.primary_traj_model}_auc']
                
                sub_higher_auc [i, j] = units_pair_res.max()
                sub_lower_auc  [i, j] = units_pair_res.min()
                sub_average_auc[i, j] = units_pair_res.mean()
                
            sub_corr_array += sub_corr_array.transpose()
            sub_higher_auc += sub_higher_auc.transpose()
            sub_lower_auc += sub_lower_auc.transpose()
            sub_average_auc += sub_average_auc.transpose()        
            
            if good_only:
                units_res_subset_units = units_res_subset_units.loc[units_res_subset_units['quality'] == 'good', :]  
                good_targets = np.where(units_res_targets['quality'] == 'good')[0]
                good_sources = np.where(units_res_sources['quality'] == 'good')[0]
                units_res_targets = units_res_targets.iloc[good_targets, :]
                units_res_sources = units_res_sources.iloc[good_sources, :]
                sub_FN   = sub_FN  [good_targets, good_sources]
                sub_dist = sub_dist[good_targets, good_sources]
                sub_corr_array = sub_corr_array[good_targets, good_sources]
                sub_higher_auc = sub_higher_auc[good_targets, good_sources]
                sub_lower_auc = sub_lower_auc[good_targets, good_sources]
                sub_average_auc = sub_average_auc[good_targets, good_sources]
    
            
            tmp_df = pd.DataFrame(data=zip(sub_FN.flatten(),
                                           sub_FN.flatten() - sub_spontFN.flatten(),
                                           np.abs(sub_FN.flatten() - sub_spontFN.flatten()),
                                           np.tile(units_res_sources['cortical_area'], sub_FN.shape[0]),
                                           np.repeat(FN_name, sub_FN.size),
                                           np.repeat(sub_basis, sub_FN.size),
                                           sub_corr_array.flatten(),
                                           sub_higher_auc.flatten(),
                                           sub_lower_auc.flatten(),
                                           sub_average_auc.flatten(),
                                           sub_dist.flatten()), 
                                  columns=['Wji', 'Wji_diff', 'absolute_Wji_diff', 'input_area', 'FN_key', 'Units Subset', 
                                           'pearson_r', 'high_auc', 'low_auc', 'avg_auc', 'Distance'])
            tmp_df = tmp_df.loc[tmp_df['Wji'] != 0, :]
            tmp_df['pearson_r_squared'] = tmp_df['pearson_r']**2
            weights_df = pd.concat((weights_df, tmp_df))
    
            if FN_name == 'reachFN1':
                tmp_auc_df = pd.DataFrame(data=zip(units_res_subset_units['traj_avgPos_auc'],
                                                   units_res_subset_units['cortical_area'],
                                                   units_res_subset_units['percent_frate_increase'],
                                                   units_res_subset_units['modulation_RO'],
                                                   units_res_subset_units['modulation_RP'],
                                                   units_res_subset_units['modulation_RE'],
                                                   units_res_subset_units['maxDev_RO'],
                                                   units_res_subset_units['maxDev_RP'],
                                                   units_res_subset_units['maxDev_RE'],
                                                   units_res_subset_units['reach_frate'],
                                                   units_res_subset_units['spont_frate'],
                                                   np.repeat(sub_basis, units_res_subset_units.shape[0]),
                                                   units_res_subset_units['quality']),
                                          columns=['Kinematics_AUC', 'cortical_area', 'frate_percent_increase', 
                                                   'mod_RO', 'mod_RP', 'mod_RE', 'dev_RO', 'dev_RP', 'dev_RE', 
                                                   'reach_frate', 'spont_frate', 'Units Subset', 'quality'])
                auc_df = pd.concat((auc_df, tmp_auc_df))
    if plot_auc_matched:
        auc_df_auc_matched     = auc_df.loc[auc_df['Kinematics_AUC'] >= auc_df.loc[auc_df['Units Subset'] == 'Reach-Specific', 'Kinematics_AUC'].min(), :]
        weights_df_auc_matched = weights_df.loc[weights_df['low_auc'] >= auc_df.loc[auc_df['Units Subset'] == 'Reach-Specific', 'Kinematics_AUC'].min(), :]

        plot_wji_distributions_for_subsets(weights_df_auc_matched, paperFig = 'FigS7', figname_mod = '_AUCmatched', hue_order_FN=hue_order_FN, palette=palette)
        plot_trajectory_correlation_and_auc_distributions_for_subsets(auc_df_auc_matched, weights_df_auc_matched, paperFig='FigS7', figname_mod = '_AUCmatched', hue_order_FN=hue_order_FN, palette=palette)
        plot_modulation_for_subsets(auc_df_auc_matched, paperFig='Fig6', figname_mod = '_AUCmatched', hue_order_FN=hue_order_FN, palette=palette)
        # plot_modulation_for_subsets(auc_df_auc_matched, paperFig='modulation', figname_mod = '_AUCmatched_noMUA', hue_order_FN=hue_order_FN, palette=palette)

    else:   
        auc_df_auc_matched = None
        weights_df_auc_matched = None

    plot_wji_distributions_for_subsets(weights_df, paperFig = 'Fig5', figname_mod = '', hue_order_FN=hue_order_FN, palette=palette) 
    plot_trajectory_correlation_and_auc_distributions_for_subsets(auc_df, weights_df, paperFig='Fig6', figname_mod = '', hue_order_FN=hue_order_FN, palette=palette)
    plot_modulation_for_subsets(auc_df, paperFig='Fig6', figname_mod = '_all', hue_order_FN=hue_order_FN, palette=palette)
    # plot_modulation_for_subsets(auc_df, paperFig='modulation', figname_mod = '_noMUA', hue_order_FN=hue_order_FN, palette=palette)


    return weights_df, auc_df, weights_df_auc_matched, auc_df_auc_matched
        
def feature_correlation_plot(units_res, x_key, y_key, hue_key=None, col_key=None, paperFig='unknown'):
    
    if 'W_in' in x_key and marmcode=='TY':
        xmin, xmax = 0, 0.12
        xrange = units_res[x_key].max() - units_res[x_key].min() 
        xmin, xmax = units_res[x_key].min() - xrange*.1, units_res[x_key].max() + xrange*.1

    else:
        xrange = units_res[x_key].max() - units_res[x_key].min() 
        xmin, xmax = units_res[x_key].min() - xrange*.1, units_res[x_key].max() + xrange*.1    
    yrange = units_res[y_key].max() - units_res[y_key].min() 
    ymin, ymax = units_res[y_key].min()-yrange*.1, units_res[y_key].max()+yrange*.1

    try:
        xlabel = [f'{lab}' for lab, key in zip(['Trajectory AUC', 'Full kinematics AUC', 'Velocity AUC', 'Short kinematics AUC', 
                                                'Kinematics + reachFN AUC', 'Kinematics + spontaneousFN \ngeneralization AUC',
                                                '$W_{{ji}}$', 'Average in-weight', 'Squared pearson correlation', 
                                                'AUC added by FN', 'Average AUC'], 
                                               ['traj', 'traj_avgPos', 'shortTraj', 'shortTraj_avgPos', 
                                                'traj_avgPos_reach_FN', 'traj_avgPos_spont_train_reach_test_FN',
                                                'Wji', 'W_in', 'pearson_r_squared', 'FN_auc_improvement', 
                                                'avg_auc']) if x_key in [key, f'{key}_auc']][0]
    except:
        xlabel = x_key
    
    try:
        ylabel = [f'{lab}' for lab, key in zip(['Trajectory AUC', 'Full kinematics AUC', 'Velocity AUC', 'Short kinematics AUC', 
                                                'Kinematics + reachFN AUC', 'Kinematics + spontaneousFN \ngeneralization AUC',
                                                '$W_{{ji}}$', 'Average in-weight', 'Squared pearson correlation', 
                                                'AUC added by FN', 'Average AUC'], 
                                               ['traj', 'traj_avgPos', 'shortTraj', 'shortTraj_avgPos', 
                                                'traj_avgPos_reach_FN', 'traj_avgPos_spont_train_reach_test_FN',
                                                'Wji', 'W_in', 'pearson_r_squared', 'FN_auc_improvement', 
                                                'avg_auc']) if y_key in [key, f'{key}_auc']][0]
    except:
        ylabel = y_key
    
    if col_key is not None:
        fig = sns.relplot(data = units_res, x=x_key, y=y_key, 
                          col=col_key, kind='scatter', legend=True, 
                          height = plot_params.feature_corr_figSize[1], 
                          aspect = plot_params.feature_corr_figSize[0]/plot_params.feature_corr_figSize[1])
        for ax in fig.axes[0]:
            col_value = ax.title.get_text().split(' = ')[-1]
            tmp_units_res = units_res.loc[units_res[col_key] == col_value, [x_key, y_key]]
            slope, intercept, r, p, stderr = linregress(tmp_units_res[x_key], tmp_units_res[y_key])
            line = f'r = {r:.2f}'
            ax.text((xmax+xmin)/2, ymax, line, horizontalalignment='center', fontsize = plot_params.tick_fontsize)
            ax.plot(tmp_units_res[x_key], intercept + slope * tmp_units_res[x_key], label=line, 
                    linestyle='solid', color='black', linewidth=plot_params.traj_linewidth)
            # ax.legend(loc='upper center')
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
        plot_name = f'{marmcode}_feature_correlation_{x_key}_{y_key}_columns_{col_key}.png'
    else:
        fig, ax = plt.subplots(figsize = plot_params.feature_corr_figSize, dpi = plot_params.dpi)
        try:
            sns.scatterplot(ax = ax, data = units_res, x = x_key, y = y_key, color=plot_params.corr_marker_color,
                            hue = hue_key, s = plot_params.feature_corr_markersize, legend=False)
        except:
            sns.scatterplot(ax = ax, data = units_res, x = x_key, y = y_key, color=plot_params.corr_marker_color,
                            s = plot_params.feature_corr_markersize, legend=False)        
        slope, intercept, r, p, stderr = linregress(units_res[x_key], units_res[y_key])
        line = f'r = {r:.2f}'
        ax.text((xmax+xmin)/2, ymax, line, horizontalalignment='center', fontsize = plot_params.tick_fontsize)
        ax.plot(units_res[x_key], intercept + slope * units_res[x_key], label=line,
                linestyle='solid', color='black', linewidth=plot_params.traj_linewidth)
        # ax.legend(bbox_to_anchor=(1.1, 0.5), loc='center left', 
        #           labels  = [line], 
        #           title_fontsize = plot_params.axis_fontsize,
        #           fontsize = plot_params.tick_fontsize)
        sns.despine(ax=ax)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        plot_name = f'{marmcode}_feature_correlation_{x_key}_{y_key}.png'
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    fig.savefig(os.path.join(plots, paperFig, plot_name), bbox_inches='tight', dpi=plot_params.dpi)

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

def find_reach_specific_group(diff_df, model_1, model_2, paperFig = 'FigS5'):
    full_res_model_1 = results_dict[params.best_lead_lag_key]['model_results'][model_1]['AUC']
    full_res_model_2 = results_dict[params.best_lead_lag_key]['model_results'][model_2]['AUC']
    
    nTests = full_res_model_1.shape[1]
    proportion = []
    pval = []
    for unit_model1, unit_model2 in zip(full_res_model_1, full_res_model_2):        
        n2over1 = (unit_model2 > unit_model1).sum()
        
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
    medFilt_grad = median_filter(sorted_diff_df['dist_positive_grad'], 9)
    lastUnit = np.where(medFilt_grad < 0.075 * np.nanmax(medFilt_grad))[0][0]
    
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
    
    if show_plots:
        plt.show()
    else:
        plt.close()

    fig.savefig(os.path.join(plots, paperFig, f'{marmcode}_classifier_selection.png'), bbox_inches='tight', dpi=plot_params.dpi)

    # reach_specific_units_byStats = diff_df.index[diff_df['generalization_proportion'] >= .95]    
    # non_specific_units_byStats = diff_df.index[diff_df['generalization_proportion'] < 0.95]    

    reach_specific_units_byStats = sorted_diff_df.index[:lastUnit] 
    non_specific_units_byStats   = sorted_diff_df.index[lastUnit:]
    
    params.reach_specific_thresh = sorted_diff_df['dist_from_unity'].iloc[lastUnit-1:lastUnit+1].mean()

    return reach_specific_units_byStats, non_specific_units_byStats     

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
    
    if numplots is not None:

        axlims_good = plot_pathlet(posTraj_mean, posTraj_samples, lead_lag_key, model, avgPos_mean = avgPos_mean, unit_selector = 'max', numToPlot = numplots, unitsToPlot = None, axlims = None)
        _           = plot_pathlet(posTraj_mean, posTraj_samples, lead_lag_key, model, unit_selector = 'min', numToPlot = numplots, unitsToPlot = None, axlims = axlims_good)

    pathDivergence_mean = np.mean(pathDivergence, axis = 0)    
    
    if 'velTraj_mean' not in locals():
        velTraj_mean = []

    return posTraj_mean, velTraj_mean, posTraj_samples, velTraj_samples  
    
def plot_pathlet(posTraj_mean, posTraj_samples, lead_lag_key, model, avgPos_mean = None, unit_selector = 'max', numToPlot = 5, unitsToPlot = None, axlims = None, paperFig='unknown'):
    
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
    
    for unit in units:
        # title = '(%s) Unit %d' %(unit_selector, unit) 
        
        leadSamp = round(lead / (lead + lag) * posTraj_mean.shape[0])
        fig = plt.figure(figsize = (4.95, 4.5))
        ax = plt.axes(projection='3d')
        for sampPath in posTraj_samples:
            ax.plot3D(sampPath[:leadSamp + 1, 0, unit], sampPath[:leadSamp + 1, 1, unit], sampPath[:leadSamp + 1, 2, unit], 'blue')
            ax.plot3D(sampPath[leadSamp:    , 0, unit], sampPath[leadSamp:    , 1, unit], sampPath[leadSamp:    , 2, unit], 'red')
        ax.plot3D(posTraj_mean[:leadSamp + 1, 0, unit], posTraj_mean[:leadSamp + 1, 1, unit], posTraj_mean[:leadSamp + 1, 2, unit], 'black', linewidth=3)
        ax.plot3D(posTraj_mean[leadSamp:, 0, unit], posTraj_mean[leadSamp:, 1, unit], posTraj_mean[leadSamp:, 2, unit], 'black', linewidth=3)
        # ax.set_title(title, fontsize = 16, fontweight = 'bold')
        ax.set_xlim(min_xyz[0], max_xyz[0])
        ax.set_ylim(min_xyz[1], max_xyz[1])
        ax.set_zlim(min_xyz[2], max_xyz[2])
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # ax.set_zticklabels([])
        # ax.set_xlabel('x', fontsize = plot_params.axis_fontsize)
        # ax.set_ylabel('y', fontsize = plot_params.axis_fontsize)
        # ax.set_zlabel('z', fontsize = plot_params.axis_fontsize)
        ax.set_xlabel('', fontsize = plot_params.axis_fontsize)
        ax.set_ylabel('', fontsize = plot_params.axis_fontsize)
        ax.set_zlabel('', fontsize = plot_params.axis_fontsize)
        # ax.legend(['lead', 'lag'], loc='upper right', bbox_to_anchor=(1, 1), fontsize = 14, shadow=False)
        ax.w_xaxis.line.set_color('black')
        ax.w_yaxis.line.set_color('black')
        ax.w_zaxis.line.set_color('black')
        ax.view_init(28, 148)
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        fig.savefig(os.path.join(plots, paperFig, f'{marmcode}_unit_%d_pathlet.png' % unit), bbox_inches='tight', dpi=plot_params.dpi)

    
    if unit_selector == 'max':
        
        fig1 = plt.figure(figsize = (4.95, 4.5))
        ax1  = plt.axes(projection='3d')
        fig2 = plt.figure(figsize = (4.95, 4.5))
        ax2  = plt.axes(projection='3d')
        for unit in range(posTraj_mean.shape[-1]):
            # title = '(%s) Unit %d' %(unit_selector, unit) 
            
            posTraj_plus_avgPos = posTraj_mean[..., unit]
            posTraj_plus_avgPos = posTraj_plus_avgPos - posTraj_plus_avgPos.mean(axis=0) + avgPos_mean[0, :, unit] 
            
            leadSamp = round(lead / (lead + lag) * posTraj_mean.shape[0])
            ax1.plot3D(posTraj_mean[:leadSamp + 1, 0, unit], posTraj_mean[:leadSamp + 1, 1, unit], posTraj_mean[:leadSamp + 1, 2, unit], 'blue', linewidth=1)
            ax1.plot3D(posTraj_mean[leadSamp:    , 0, unit], posTraj_mean[leadSamp:    , 1, unit], posTraj_mean[leadSamp:    , 2, unit], 'red' , linewidth=1)
            ax2.plot3D(posTraj_plus_avgPos[:leadSamp + 1, 0], posTraj_plus_avgPos[:leadSamp + 1, 1], posTraj_plus_avgPos[:leadSamp + 1, 2], 'blue', linewidth=1)
            ax2.plot3D(posTraj_plus_avgPos[leadSamp:    , 0], posTraj_plus_avgPos[leadSamp:    , 1], posTraj_plus_avgPos[leadSamp:    , 2], 'red' , linewidth=1)
            
            # ax.set_title(title, fontsize = 16, fontweight = 'bold')
            # ax.set_xlim(min_xyz[0], max_xyz[0])
            # ax.set_ylim(min_xyz[1], max_xyz[1])
            # ax.set_zlim(min_xyz[2], max_xyz[2])
            # ax.set_xticklabels([])
            # ax.set_yticklabels([])
            # ax.set_zticklabels([])
            ax1.set_xlabel('x', fontsize = plot_params.axis_fontsize)
            ax1.set_ylabel('y', fontsize = plot_params.axis_fontsize)
            ax1.set_zlabel('z', fontsize = plot_params.axis_fontsize)
            ax2.set_xlabel('x', fontsize = plot_params.axis_fontsize)
            ax2.set_ylabel('y', fontsize = plot_params.axis_fontsize)
            ax2.set_zlabel('z', fontsize = plot_params.axis_fontsize)            # ax.set_xlabel('', fontsize = plot_params.axis_fontsize)
            # ax.set_ylabel('', fontsize = plot_params.axis_fontsize)
            # ax.set_zlabel('', fontsize = plot_params.axis_fontsize)
            # ax.legend(['lead', 'lag'], loc='upper right', bbox_to_anchor=(1, 1), fontsize = 14, shadow=False)
            ax1.w_xaxis.line.set_color('black')
            ax1.w_yaxis.line.set_color('black')
            ax1.w_zaxis.line.set_color('black')
            ax1.view_init(28, 148)
            
            ax2.w_xaxis.line.set_color('black')
            ax2.w_yaxis.line.set_color('black')
            ax2.w_zaxis.line.set_color('black')
            ax2.view_init(28, 148)
        if show_plots:
            plt.show()
        else:
            plt.close() 
        
        # fig1.savefig(os.path.join(plots, 'all_units_pathlets_noPos.png'), bbox_inches='tight', dpi=plot_params.dpi)
        # fig2.savefig(os.path.join(plots, 'all_units_pathlets_withPos.png'), bbox_inches='tight', dpi=plot_params.dpi)


    if unitsToPlot is not None: 
        print(traj_auc[unitsToPlot[0]])
    
    return (min_xyz, max_xyz)

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
                if show_plots:
                    plt.show()
                else:
                    plt.close()
                
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

    fig, ax = plt.subplots(figsize = plot_params.feature_corr_figSize, dpi = plot_params.dpi)
    sns.scatterplot(ax = ax, data = df, x = 'Pearson_corr', y = 'Wji', s = plot_params.feature_corr_markersize, 
                    color=plot_params.corr_marker_color, legend=False) 
    ax.set_xlabel('Preferred trajectory\ncorrelation')
    ax.set_ylabel('$W_{{ji}}$')  
    # ax.tick_params(width=plot_params.tick_width, length = plot_params.tick_length*2, labelsize = plot_params.tick_fontsize)
    sns.despine(ax=ax)
    if show_plots:
        plt.show()
    else:
        plt.close()
    fig.savefig(os.path.join(plots, paperFig, f'{marmcode}_wji_vs_pearson_r.png'), bbox_inches='tight', dpi=plot_params.dpi)
    
    return df

def plot_FN_differences(FN, spontaneous_FN, reach_specific_reachFNs, reach_specific_spontFN, 
                        non_specific_reachFNs, non_specific_spontFN, paperFig, hue_order_FN=None, palette=None):
    reach_set1_list = [         FN[0], reach_specific_reachFNs[0], non_specific_reachFNs[0]]
    reach_set2_list = [         FN[1], reach_specific_reachFNs[1], non_specific_reachFNs[1]]
    spont_list      = [spontaneous_FN, reach_specific_spontFN[0] , non_specific_spontFN[0]] 
    labels = ['Full', 'Reach-Specific', 'Non-Specific']
    
    diff_df = pd.DataFrame()
    for rFN1, rFN2, sFN, label in zip(reach_set1_list, reach_set2_list, spont_list, labels):
        diff_data = np.concatenate((np.abs(rFN1.flatten() - sFN.flatten()),
                                    np.abs(rFN2.flatten() - sFN.flatten())),
                                   axis=0)
        label_list = [label for i in range(diff_data.shape[0])]
        set_list   = ['reachFN1' for i in range(rFN1.size)] + ['reachFN2' for i in range(rFN2.size)]
        
        tmp_df = pd.DataFrame(data = zip(diff_data, label_list, set_list),
                              columns = ['w_diff', 'Units Subset', 'Reach Set'])
        
        diff_df = pd.concat((diff_df, tmp_df), axis=0, ignore_index=True)
    
    #------------------------------------------------------------------------------------
    
    fig, ax = plt.subplots(figsize = plot_params.distplot_figsize)
    sns.kdeplot(data=diff_df[diff_df['Reach Set'] == 'reachFN1'], ax=ax, x='w_diff', 
                hue='Units Subset', hue_order = hue_order_FN, palette=palette,
                cumulative=True, common_norm=False, bw_adjust=0.05, linewidth=plot_params.distplot_linewidth)
    ax.legend(bbox_to_anchor=(1.1, 0.5), loc='center left',
              handles = ax.lines[::-1],
              labels  = hue_order_FN, 
              title_fontsize = plot_params.axis_fontsize,
              fontsize = plot_params.tick_fontsize)
    sns.despine(ax=ax)
    ax.set_yticks([0, 1])
    ax.set_xlabel('|$W_{{ji}}$ (reachFN) - $W_{{ji}}$ (spontaneousFN)|')
    if show_plots:
        plt.show()
    else:
        plt.close()
    fig.savefig(os.path.join(plots, paperFig, f'{marmcode}_w_diff_cumulative_dist_reachFN1_minus_spont'), bbox_inches='tight', dpi=plot_params.dpi)
    med_out = median_test(diff_df.loc[(diff_df['Units Subset'] == 'Reach-Specific') & (diff_df['Reach Set'] == 'reachFN1') , 'w_diff'], 
                          diff_df.loc[(diff_df['Units Subset'] == 'Non-Specific')   & (diff_df['Reach Set'] == 'reachFN1') , 'w_diff'])
    print(f'Wji-Difference_reachFN1:  reach-spec vs non-spec, p={np.round(med_out[1], 4)}, {med_out[-1][0,0]}a-{med_out[-1][1,0]}b, {med_out[-1][0,1]}a-{med_out[-1][1,1]}b')
    med_out = median_test(diff_df.loc[(diff_df['Units Subset'] == 'Reach-Specific') & (diff_df['Reach Set'] == 'reachFN1') , 'w_diff'], 
                          diff_df.loc[(diff_df['Units Subset'] == 'Full')   & (diff_df['Reach Set'] == 'reachFN1') , 'w_diff'])
    print(f'Wji-Difference_reachFN1:  reach-spec vs full, p={np.round(med_out[1], 4)}, {med_out[-1][0,0]}a-{med_out[-1][1,0]}b, {med_out[-1][0,1]}a-{med_out[-1][1,1]}b')
    med_out = median_test(diff_df.loc[(diff_df['Units Subset'] == 'Non-Specific') & (diff_df['Reach Set'] == 'reachFN1') , 'w_diff'], 
                          diff_df.loc[(diff_df['Units Subset'] == 'Full')   & (diff_df['Reach Set'] == 'reachFN1') , 'w_diff'])
    print(f'Wji-Difference_reachFN1:  non-spec vs full, p={np.round(med_out[1], 4)}, {med_out[-1][0,0]}a-{med_out[-1][1,0]}b, {med_out[-1][0,1]}a-{med_out[-1][1,1]}b')
        
    #------------------------------------------------------------------------------------   
    
    fig, ax = plt.subplots(figsize = plot_params.distplot_figsize)
    sns.kdeplot(data=diff_df[diff_df['Reach Set'] == 'reachFN2'], ax=ax, x='w_diff', 
                hue='Units Subset', hue_order = hue_order_FN, palette=palette, 
                cumulative=True, common_norm=False, bw_adjust=0.05, linewidth=plot_params.distplot_linewidth)
    ax.legend(bbox_to_anchor=(1.1, 0.5), loc='center left', 
              handles = ax.lines[::-1],
              labels  = hue_order_FN, 
              title_fontsize = plot_params.axis_fontsize,
              fontsize = plot_params.tick_fontsize)
    ax.set_yticks([0, 1])
    ax.set_xlabel('|$W_{{ji}}$ (reachFN) - $W_{{ji}}$ (spontaneousFN)|')
    sns.despine(ax=ax)
    if show_plots:
        plt.show()
    else:
        plt.close()
    fig.savefig(os.path.join(plots, paperFig, f'{marmcode}_w_diff_cumulative_dist_reachFN2_minus_spont'), bbox_inches='tight', dpi=plot_params.dpi)
    med_out = median_test(diff_df.loc[(diff_df['Units Subset'] == 'Reach-Specific') & (diff_df['Reach Set'] == 'reachFN2') , 'w_diff'], 
                          diff_df.loc[(diff_df['Units Subset'] == 'Non-Specific')   & (diff_df['Reach Set'] == 'reachFN2') , 'w_diff'])
    print(f'Wji-Difference_reachFN2:  reach-spec vs non-spec, p={np.round(med_out[1], 4)}, {med_out[-1][0,0]}a-{med_out[-1][1,0]}b, {med_out[-1][0,1]}a-{med_out[-1][1,1]}b')
    med_out = median_test(diff_df.loc[(diff_df['Units Subset'] == 'Reach-Specific') & (diff_df['Reach Set'] == 'reachFN2') , 'w_diff'], 
                          diff_df.loc[(diff_df['Units Subset'] == 'Full')   & (diff_df['Reach Set'] == 'reachFN2') , 'w_diff'])
    print(f'Wji-Difference_reachFN2:  reach-spec vs full, p={np.round(med_out[1], 4)}, {med_out[-1][0,0]}a-{med_out[-1][1,0]}b, {med_out[-1][0,1]}a-{med_out[-1][1,1]}b')
    med_out = median_test(diff_df.loc[(diff_df['Units Subset'] == 'Non-Specific') & (diff_df['Reach Set'] == 'reachFN2') , 'w_diff'], 
                          diff_df.loc[(diff_df['Units Subset'] == 'Full')   & (diff_df['Reach Set'] == 'reachFN2') , 'w_diff'])
    print(f'Wji-Difference_reachFN2:  non-spec vs full, p={np.round(med_out[1], 4)}, {med_out[-1][0,0]}a-{med_out[-1][1,0]}b, {med_out[-1][0,1]}a-{med_out[-1][1,1]}b')
        
    #------------------------------------------------------------------------------------

    return           

if __name__ == "__main__":
    
    os.makedirs(plots, exist_ok=True)
    
    # with open(pkl_infile, 'rb') as f:
    #     results_dict = dill.load(f)  

    # with open(pkl_addstatsfile, 'rb') as f:
    #     results_dict_add_stats = dill.load(f)  
    
    results_dict = load_dict_from_hdf5(pkl_infile.with_suffix('.h5'), top_level_list=False, convert_4d_array_to_list = True)
    results_dict_add_stats = load_dict_from_hdf5(pkl_addstatsfile.with_suffix('.h5'), top_level_list=False, convert_4d_array_to_list = True)
        
    with NWBHDF5IO(nwb_infile, 'r') as io:
        nwb = io.read()

        FN = nwb.scratch[params.FN_key].data[:] 
        spontaneous_FN = nwb.scratch['spontaneous_FN'].data[:]

    summarize_model_results(units=None, lead_lag_keys = params.best_lead_lag_key)
    
    units_res = results_dict[params.best_lead_lag_key]['all_models_summary_results']
    units_res = add_in_weight_to_units_df(units_res, FN.copy())
    
    units_res = add_modulation_data_to_units_df(units_res)
    
    # if marmcode == 'TY':
        # units_res = add_icms_results_to_units_results_df(units_res, params.icms_res)
    
    electrode_distances = get_interelectrode_distances_by_unit(units_res, array_type='utah')

    if not skip_network_permutations:
        train_auc_df = evaluate_effect_of_network_shuffles(lead_lag_key = params.best_lead_lag_key, 
                                                           comparison_model = 'traj_avgPos_reach_FN', 
                                                           kin_only_model=params.kin_only_model, all_samples=False, 
                                                           targets=None, ylim=(0,33), paperFig = 'Fig4', 
                                                           plot_difference = False, alpha=0.01, palette='ylgn_dark_fig4')
        
    _, cmin, cmax = plot_functional_networks(FN, units_res, FN_key = params.FN_key, paperFig='Fig1')
    _, _, _       = plot_functional_networks(spontaneous_FN, units_res, FN_key ='spontaneous_FN', paperFig='Fig1', cmin=cmin, cmax=cmax)
    # _, cmin, cmax =plot_functional_networks(spontaneous_FN, units_res, FN_key ='spontaneous_FN')
    # _, cmin, cmax = plot_functional_networks(FN, units_res, FN_key = params.FN_key, cmin=cmin, cmax=cmax)

    ymin, ymax = plot_weights_versus_interelectrode_distances(FN, spontaneous_FN, 
                                                              electrode_distances, paperFig='Fig1',
                                                              palette='FN_palette')
    
    style_key = None
    plotted_on_map_record = []
    model_list_x = [       'traj',    'position',          'traj_avgPos']
    model_list_y = ['traj_avgPos', 'traj_avgPos', 'traj_avgPos_reach_FN']
    fig_list     = [       'FigS3',    'unknown',                 'FigS3']
    for model_x, model_y, fignum in zip(model_list_x, model_list_y, fig_list):    
        sign_test = plot_model_auc_comparison(units_res, model_x, model_y, 
                                              minauc = 0.45, maxauc = 0.8, hue_key='W_in', style_key=style_key, 
                                              targets = None, col_key = None, paperFig=fignum, asterisk='')
        print(f'{model_y} v {model_x}, NO_tuning_filter: p={np.round(sign_test.pvalue, 4)}, nY={sign_test.k}, nUnits={sign_test.n}, prop={np.round(sign_test.proportion_estimate, 2)}')
        if model_y not in plotted_on_map_record:
            plot_result_on_channel_map(units_res, jitter_radius = .2, hue_key='W_in', 
                                       size_key = model_y,  title_type = 'size', style_key=style_key,
                                       paperFig=fignum, palette=None, sizes=(6, 20), hue_order=None)
            plotted_on_map_record.append(model_y)
    
    if not demo:
        units_res_completely_untuned_units_filtered = units_res.loc[units_res['shuffled_traj_pval']<0.01/units_res.shape[0], :]
        model_list_x = ['shortTraj', 'shortTraj_avgPos']
        model_list_y = [     'traj',      'traj_avgPos']
        fig_list     = [     'Fig2',             'Fig2']
        for model_x, model_y, fignum in zip(model_list_x, model_list_y, fig_list):    
            sign_test = plot_model_auc_comparison(units_res_completely_untuned_units_filtered, model_x, model_y, 
                                                  minauc = 0.45, maxauc = 0.8, hue_key='W_in', style_key=style_key, 
                                                  targets = None, col_key = None, paperFig=fignum, asterisk='*')
            print(f'{model_y} v {model_x}, YES_tuning_filter: p={np.round(sign_test.pvalue, 4)}, nY={sign_test.k}, nUnits={sign_test.n}, prop={np.round(sign_test.proportion_estimate, 2)}')

    diff_df = compute_performance_difference_by_unit(units_res, 'traj_avgPos_spont_train_reach_test_FN_auc', 'traj_avgPos_reach_FN_auc')   
    # reach_specific_units = diff_df.index[(diff_df.auc_diff > 0) & (diff_df.dist_from_unity > params.reach_specific_thresh) & (diff_df.dist_from_unity < 0.04)]
    reach_specific_units, non_specific_units = find_reach_specific_group(diff_df,
                                                                         'traj_avgPos_spont_train_reach_test_FN', 
                                                                         'traj_avgPos_reach_FN',
                                                                         paperFig = 'FigS6')
    
    posTraj_mean, velTraj_mean, posTraj_samples, velTraj_samples = compute_and_analyze_pathlets(params.best_lead_lag_key, 'traj_avgPos', numplots = None)
    traj_corr_df = compute_and_analyze_trajectory_correlations(units_res, posTraj_mean, velTraj_mean, 
                                                               electrode_distances, params.best_lead_lag_key, 
                                                               FN = FN, mode='concat', paperFig='Fig3',
                                                               reach_specific_units = reach_specific_units, nplots=None)
    
    subset = 'both'
    reach_specific_reachFNs, _, _ = plot_functional_networks(FN, units_res, FN_key = params.FN_key, cmin=cmin, cmax=cmax, 
                                                             subset_idxs = reach_specific_units, subset_type=subset, paperFig='Fig5')
    reach_specific_spontFN , _, _ = plot_functional_networks(spontaneous_FN, units_res, FN_key ='spontaneous_FN', cmin=cmin, cmax=cmax, 
                                                             subset_idxs = reach_specific_units, subset_type=subset, paperFig='Fig5')

    non_specific_reachFNs, _, _ = plot_functional_networks(FN, units_res, FN_key = params.FN_key, cmin=cmin, cmax=cmax, 
                                                           subset_idxs = non_specific_units, subset_type=subset, paperFig='Fig5')
    non_specific_spontFN , _, _ = plot_functional_networks(spontaneous_FN, units_res, FN_key ='spontaneous_FN', cmin=cmin, cmax=cmax, 
                                                           subset_idxs = non_specific_units, subset_type=subset, paperFig='Fig5')


    reach_specific_units_res = units_res.loc[reach_specific_units, :]
    non_specific_units_res   = units_res.loc[non_specific_units, :]

    weights_df, auc_df, weights_df_auc_matched, auc_df_auc_matched =plot_distributions_after_source_props(units_res, 
                                                                                                          electrode_distances,
                                                                                                          traj_corr_df,
                                                                                                          FN_sets = [('split_reach_FNs', FN), ('spontaneous_FN', spontaneous_FN)], 
                                                                                                          subset_idxs = reach_specific_units, 
                                                                                                          sub_type='both', 
                                                                                                          subset_basis=['Reach-Specific', 'Non-Specific', 'Full'],
                                                                                                          good_only=False,
                                                                                                          plot_auc_matched=True,
                                                                                                          hue_order_FN = ['Reach-Specific', 'Non-Specific', 'Full'],
                                                                                                          palette = fig6_palette)
    
    
    units_res['Units Subset'] = ['Non-Specific' for idx in range(units_res.shape[0])]
    units_res.loc[reach_specific_units, 'Units Subset'] = ['Reach-Specific' for idx in range(reach_specific_units.size)]

    plot_result_on_channel_map(units_res, jitter_radius = .2, hue_key='Units Subset', 
                               size_key = None,  title_type = 'hue', style_key=style_key,
                               paperFig='Fig6', hue_order = ['Reach-Specific', 'Non-Specific'], 
                               palette=fig6_palette, sizes=None, s=13)

    # plot_FN_differences(FN, spontaneous_FN, reach_specific_reachFNs, reach_specific_spontFN, 
    #                     non_specific_reachFNs, non_specific_spontFN, paperFig='Fig5', 
    #                     hue_order_FN = ['Reach-Specific', 'Non-Specific', 'Full'], palette=fig6_palette)
                                      
    plot_model_auc_comparison   (units_res, 'traj_avgPos_auc', 'traj_avgPos_reach_FN_auc', 
                                 minauc = 0.45, maxauc = 0.8, hue_key='Units Subset', hue_order=('Reach-Specific', 'Non-Specific'), 
                                 style_key=style_key, targets = None, paperFig='FigS7', palette=fig6_palette)
    plot_model_auc_comparison   (units_res, 'spontaneous_FN_auc', 'reach_FN_auc', 
                                 minauc = 0.45, maxauc = 0.8, hue_key='Units Subset', hue_order=('Reach-Specific', 'Non-Specific'), 
                                 style_key=style_key, targets = None, paperFig='unknown', palette=fig6_palette)
    plot_model_auc_comparison   (units_res, 'traj_avgPos_spont_train_reach_test_FN_auc', 'traj_avgPos_reach_FN_auc', 
                                 minauc = 0.45, maxauc = 0.8, hue_key='Units Subset', hue_order=('Reach-Specific', 'Non-Specific'), 
                                 style_key=style_key, targets = None, paperFig='Fig5', palette=fig6_palette)

    plot_model_auc_comparison   (units_res, 'shortTraj_avgPos', 'traj_avgPos', minauc = 0.45, maxauc = 0.8, hue_key='Units Subset', 
                                 hue_order=('Reach-Specific', 'Non-Specific'), style_key=style_key, targets = None, paperFig='unknown', palette=fig6_palette)
    plot_model_auc_comparison   (units_res, 'shortTraj', 'traj', minauc = 0.45, maxauc = 0.8, hue_key='Units Subset', 
                                 hue_order=('Reach-Specific', 'Non-Specific'), style_key=style_key, targets = None, paperFig='unknown', palette=fig6_palette)
    
    units_res['FN_auc_improvement']     = units_res['traj_avgPos_reach_FN_auc'] - units_res['traj_avgPos_auc']
    units_res['FN_percent_improvement'] = (units_res['traj_avgPos_reach_FN_auc'] - units_res['traj_avgPos_auc']) / units_res['traj_avgPos_auc']
    
    feature_correlation_plot(units_res, 'W_in', 'traj_avgPos_auc', paperFig='Fig3')
    feature_correlation_plot(units_res, 'snr', 'W_in', col_key=None, paperFig = 'FigS4')    
    feature_correlation_plot(units_res, 'fr', 'W_in', col_key=None, paperFig = 'FigS4')    
    feature_correlation_plot(units_res, 'snr', 'traj_avgPos_reach_FN_auc', col_key=None, paperFig = 'FigS4')    
    feature_correlation_plot(units_res, 'fr', 'traj_avgPos_reach_FN_auc', col_key=None, paperFig = 'FigS4')    
    feature_correlation_plot(units_res, 'fr', 'traj_avgPos_auc', col_key=None, paperFig = 'FigS4')    
    feature_correlation_plot(units_res, 'snr', 'traj_avgPos_auc', col_key=None, paperFig = 'FigS4')    

    # feature_correlation_plot(units_res.loc[units_res['quality'] == 'good'], 'fr', 'traj_avgPos_auc', col_key=None)
    # feature_correlation_plot(units_res.loc[units_res['quality'] == 'good'], 'fr', 'traj_avgPos_auc', col_key='cortical_area')
    # feature_correlation_plot(units_res.loc[units_res['quality'] == 'good'], 'fr', 'traj_avgPos_reach_FN_auc', col_key='cortical_area')
    feature_correlation_plot(weights_df.loc[(weights_df['FN_key']=='reachFN1'), :], 'avg_auc', 'pearson_r_squared', col_key=None, paperFig = 'FigS3')    
    # feature_correlation_plot(weights_df.loc[(weights_df['FN_key']=='reachFN1'), :], 'Distance', 'pearson_r', col_key=None, paperFig = 'unknown')    
    feature_correlation_plot(units_res, 'W_in', 'FN_auc_improvement', paperFig='FigS3')
    feature_correlation_plot(units_res, 'traj_avgPos_auc', 'FN_auc_improvement', paperFig='FigS3')

    # feature_correlation_plot(units_res, 'traj_avgPos_auc', 'modulation_RO', paperFig='unknown')
    # feature_correlation_plot(units_res, 'traj_avgPos_auc', 'percent_frate_increase', paperFig='unknown')

    # feature_correlation_plot(units_res, 'W_in', 'FN_percent_improvement', paperFig='unknown')
    # feature_correlation_plot(units_res, 'traj_avgPos_auc', 'FN_percent_improvement', paperFig='unknown')
    
    if not demo:
        # stats for figure 2 summary plot
        model_list_x = [  'shortTraj',        'traj', 'shortTraj_avgPos', 'traj_avgPos_shuffled_spikes', 'traj_avgPos_shuffled_traj']
        model_list_y = ['traj_avgPos', 'traj_avgPos',      'traj_avgPos',                 'traj_avgPos',               'traj_avgPos']
        fig_list     = [    'unknown',     'unknown',          'unknown',                     'unknown',                   'unknown']
        for model_x, model_y, fignum in zip(model_list_x, model_list_y, fig_list):    
            sign_test = plot_model_auc_comparison(units_res_completely_untuned_units_filtered, model_x, model_y, 
                                                  minauc = 0.45, maxauc = 0.8, hue_key='W_in', style_key=style_key, 
                                                  targets = None, col_key = None, paperFig=fignum, asterisk='')
            print(f'{model_y} v {model_x}, YES_tuning_filter: p={np.round(sign_test.pvalue, 4)}, nY={sign_test.k}, nUnits={sign_test.n}, prop={np.round(sign_test.proportion_estimate, 2)}')

    save_dict_to_hdf5(results_dict, pkl_outfile.with_suffix('.h5'))
    # if save_kinModels_pkl:
    #     with open(pkl_outfile, 'wb') as f:
    #         dill.dump(results_dict, f, recurse=True) 