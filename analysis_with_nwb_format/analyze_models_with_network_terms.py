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
from matplotlib.colors import ListedColormap
import pickle
import dill
import os
import glob
import math
import re
import seaborn as sns
import math
import h5py
from itertools import product
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc
import statsmodels.api as sm
from scipy.stats import binomtest, ttest_rel, ttest_ind, mannwhitneyu, linregress, pearsonr, median_test
from scipy.integrate import cumtrapz
from scipy.integrate import simps
from mpl_toolkits import mplot3d 
from scipy.ndimage import median_filter, gaussian_filter
from importlib import sys, reload
from scipy.spatial.transform import Rotation as R
from pynwb import NWBHDF5IO
import ndx_pose

sys.path.insert(0, '/project/nicho/projects/marmosets/code_database/data_processing/nwb_tools/hatlab_nwb_tools/')
from hatlab_nwb_functions import remove_duplicate_spikes_from_good_single_units, get_sorted_units_and_apparatus_kinematics_with_metadata   

sys.path.insert(0, '/project/nicho/projects/marmosets/code_database/analysis/trajectory_encoding_model/')
from utils import get_interelectrode_distances_by_unit, choose_units_for_model

marmcode='TY'
fig_mode = 'pres' # 'paper'

if marmcode=='TY':
    nwb_infile = '/beagle3/nicho/projects/dalton/data/TY/TY20210211_freeAndMoths-003_resorted_20230612_DM_with_functional_networks.nwb' 
    pkl_infile = '/beagle3/nicho/projects/dalton/data/TY/TY20210211_freeAndMoths-003_resorted_20230612_DM_alpha_pt00001_encoding_models_30ms_shift_v4.pkl' 
    modulation_base = '/beagle3/nicho/projects/dalton/data/TY/TY20210211_freeAndMoths-003_resorted_20230612_DM'

    # pkl_infile = '/project/nicho/projects/dalton/data/TY/TY20210211_freeAndMoths-003_resorted_20230612_DM_FINAL_trajectory_shuffled_encoding_models_30ms_shift_v4.pkl'
    
    
    # pkl_infile = '/project/nicho/projects/dalton/data/TY/TY20210211_freeAndMoths-003_resorted_20230612_DM_encoding_model_sorting_corrected_30mscortical_networks_shift_v6.pkl'
    # pkl_infile = '/project/nicho/projects/dalton/data/TY/TY20210211_freeAndMoths-003_resorted_20230612_DM_FINAL_tinyAlpha_trajectory_shuffled_encoding_models_30ms_shift_v4.pkl'

    filtered_good_units_idxs = [88, 92, 123]
elif marmcode=='MG':
    nwb_infile   = '/beagle3/nicho/projects/dalton/data/MG/MG20230416_1505_mothsAndFree-002_processed_DM_with_functional_networks.nwb'
    pkl_infile = '/beagle3/nicho/projects/dalton/data/MG/MG20230416_1505_mothsAndFree-002_processed_DM_alpha_pt00001_removedUnits_181_440_fixedMUA_745_796_encoding_models_30ms_shift_v4.pkl'
    modulation_base = '/beagle3/nicho/projects/dalton/data/MG/MG20230416_1505_mothsAndFree-002_processed_DM'


    # pkl_infile = '/project/nicho/projects/dalton/data/MG/MG20230416_1505_mothsAndFree-002_processed_DM_dlcIter5_resortedUnits_trajectory_shuffled_encoding_models_30ms_shift_v4.pkl'
    # pkl_infile = '/project/nicho/projects/dalton/data/MG/MG20230416_1505_mothsAndFree-002_processed_DM_dlcIter5_tinyAlpha_resortedUnits_trajectory_shuffled_encoding_models_30ms_shift_v4.pkl'
    
    
    
    # pkl_infile = '/project/nicho/projects/dalton/data/MG/MG20230416_1505_mothsAndFree-002_processed_DM_dlcIter5_ALPHA_E_neg4_resortedUnits_trajectory_shuffled_encoding_models_30ms_shift_v4.pkl'
    # nwb_infile   = '/project/nicho/projects/dalton/data/MG/MG20230416_1505_mothsAndFree-002_processed_DM_with_functional_networks_noBadUnitsList.nwb'
    # pkl_infile = '/project/nicho/projects/dalton/data/MG/MG20230416_1505_mothsAndFree-002_processed_DM_dlcIter5_noBadUnitsList_trajectory_shuffled_encoding_models_30ms_shift_v4.pkl'
    # nwb_infile   = '/project/nicho/projects/dalton/data/MG/MG20230416_1505_mothsAndFree-002_processed_DM_with_functional_networks_WITH_UNMODULATED_UNITS_REMAINING.nwb'
    # pkl_infile = '/project/nicho/projects/dalton/data/MG/MG20230416_1505_mothsAndFree-002_processed_DM_dlcIter5_reachMethod6_trajectory_shuffled_encoding_models_30ms_shift_v4.pkl'


    
split_pattern = '_shift_v' # '_results_v'
base, ext = os.path.splitext(pkl_infile)
base, in_version = base.split(split_pattern)
out_version = str(int(in_version) + 1)  
pkl_outfile = base + split_pattern + out_version + ext

dataset_code = os.path.basename(pkl_infile)[:10] 
# plots = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(pkl_infile))), 'plots', dataset_code, 'network')

if fig_mode == 'paper':
    plots = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(pkl_infile))), 'plots', dataset_code)
elif fig_mode == 'pres':
    plots = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(pkl_infile))), 'defense_plots', dataset_code)

cmap_orig = plt.get_cmap("Paired")
FN_colors = cmap_orig([1, 0, 2])
# colors_tophalf = cmap_orig(np.arange(0, cmap_orig.N*3/4, dtype=int))
cmap = ListedColormap(FN_colors)
plt.cm.register_cmap("FN_palette", cmap=cmap)
fig6_palette = 'Dark2'

cmap_orig = plt.get_cmap("YlGn")
colors_tophalf = cmap_orig(np.arange(cmap_orig.N, cmap_orig.N/10, -1, dtype=int))
# colors_tophalf = cmap_orig(np.arange(0, cmap_orig.N*3/4, dtype=int))
cmap = ListedColormap(colors_tophalf)
plt.cm.register_cmap("ylgn_dark", cmap=cmap)

class params:
    FN_key = 'split_reach_FNs'#'split_reach_FNs'
    significant_proportion_thresh = 0.99
    tuned_auc_thresh = 0.6
 
    primary_traj_model = 'traj_avgPos'
    # cortical_boundaries = {'x_coord'      : [   0,          400,  800, 1200,         1600, 2000, 2400, 2800,          3200,          3600],
    #                         'y_bound'      : [None,         1200, None, None,         1200, None, None, None,           800,          2000],
    #                         'areas'        : ['3b', ['3a', '3b'], '3a', '3a', ['M1', '3a'], 'M1', 'M1', 'M1', ['6Dc', 'M1'], ['6Dc', 'M1']],
    #                         'unique_areas' : ['3b', '3a', 'M1', '6Dc']}

    if marmcode == 'TY':
        best_lead_lag_key = 'lead_100_lag_300' #None

        # reach_specific_thresh = 0.0075
        reach_specific_thresh = 0.025
        cortical_boundaries = {'x_coord'      : [   0,          400,  800, 1200,         1600, 2000, 2400, 2800, 3200, 3600],
                                'y_bound'      : [None,         1200, None, None,         1200, None, None, None, None, None],
                                'areas'        : ['3b', ['3a', '3b'], '3a', '3a', ['M1', '3a'], 'M1', 'M1', 'M1', 'M1', 'M1'],
                                'unique_areas' : ['3b', '3a', 'M1']}
        x_coord = np.hstack((np.repeat(0, 8), np.repeat(np.linspace(400, 3200, 8), 10), np.repeat(3600, 8)))
        y_coord = np.hstack((np.linspace(400, 3200, 8), np.tile(np.linspace(0, 3600, 10), (8,)), np.linspace(400, 3200, 8)))
        thresh = [    15,  7,     12, np.nan,     15, np.nan, np.nan, 25,
                  35,  7,  5,      7,      7,     15,     15,     10, 45, np.nan,
                   5,  5,  4,      5,      5,      8,      7,     20, 15, 25,
                   5,  5,  5,      3,      5,      5,      5,     15, 10, 20,
                   7,  3,  5, np.nan,     10,      7,      6,     12, 10,  5,
                   5,  5,  5,      5,      3,      7,      5,      5, 15, 10,
                   5,  3,  5,      3,      5,     10,      4,     10, 15, 10,
                   5,  5,  5,      5,     10,      5,      5,     10,  3,  7,
                  15, 10,  5,      5,      3,      5,      5,     12, 15, np.nan,
                      15,  7,     10,     10,      7,     15,     15, 10]
        bodypart = [ 'tmp' for idx in range(len(thresh)) ]
        icms_res = pd.DataFrame(data=zip(x_coord, y_coord, thresh, bodypart),
                                columns=['x', 'y', 'icms_threshold', 'icms_bodypart'])
        kin_only_model='traj_avgPos'
        
    elif marmcode == 'MG':
        best_lead_lag_key = 'lead_100_lag_300' #None

        reach_specific_thresh = 0.025
        cortical_boundaries = {'x_coord'      : [    0,   400,  800, 1200, 1600, 2000, 2400, 2800, 3200, 3600],
                               'y_bound'      : [ None,  None, None, None, None, None, None, None, None, None],
                               'areas'        : ['6dc', '6dc', 'M1', 'M1', 'M1', 'M1', 'M1', '3a', '3a', '3a'],
                               'unique_areas' : ['6dc', 'M1', '3a']}
        kin_only_model='traj_avgPos'
    
# class plot_params:
    
#     if fig_mode == 'paper':
#         axis_fontsize = 11
#         dpi = 300
#         axis_linewidth = 2
#         tick_length = 1.5
#         tick_width = 1
#         tick_fontsize = 8
    
#         map_figSize = (6, 8)
#         weights_by_distance_figsize = (6, 4)
#         aucScatter_figSize = (6, 6)
#         FN_figsize = (3, 3)
#         feature_corr_figSize = (4, 4)
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
        
#     figures_list = ['Fig1', 'Fig2', 'Fig3', 'Fig4', 'Fig5', 'Fig6', 'Fig7', 'FigS1',  'FigS2',  'FigS3',  'FigS4', 'FigS5', 'unknown']
        
class plot_params:
    # axis_fontsize = 24
    # dpi = 300
    # axis_linewidth = 2
    # tick_length = 2
    # tick_width = 1
    # map_figSize = (6, 8)
    # tick_fontsize = 18
    # aucScatter_figSize = (7, 7)
    
    figures_list = ['Fig1', 'Fig2', 'Fig3', 'Fig4', 'Fig5', 'Fig6', 'Fig7', 'FigS1',  'FigS2',  'FigS3',  'FigS4', 'FigS5', 'unknown', 'chapter4']

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
        
        shuffle_markerscale = 0.45
        shuffle_errwidth = 1.25
        shuffle_sigmarkersize = 1.5
        shuffle_figsize = (8, 1.5)

        
        corr_marker_color = 'gray'
        
        traj_pos_sample_figsize = (1.75, 1.75)
        traj_vel_sample_figsize = (1.5  ,   1.5)
        traj_linewidth = 1
        traj_leadlag_linewidth = 2
        
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
        stripplot_figsize = (5, 2)
        scatter_figsize = (1.75, 1.75)


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
        
        shuffle_markerscale = 1
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

        corr_marker_color = 'gray'

plt.rcParams['figure.dpi'] = plot_params.dpi
plt.rcParams['savefig.dpi'] = plot_params.dpi
plt.rcParams['font.family'] = 'sans-serif'
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

def trajectory_vs_shuffle_sign_test(traj_res, shuf_res, units_res):
    
    ttest_p = [0]*traj_res.shape[0]
    mwu_p   = [0]*traj_res.shape[0]
    for idx, (unit_traj_auc, unit_shuf_auc) in enumerate(zip(traj_res, shuf_res)):
        
        tmp, ttest_p[idx] = ttest_ind   (unit_traj_auc, unit_shuf_auc, alternative='greater')
        tmp,   mwu_p[idx] = mannwhitneyu(unit_traj_auc, unit_shuf_auc, alternative='greater')
    
    units_res['ttest_p'] = ttest_p
    units_res['mwu_p'  ] = mwu_p
    
    return units_res

def compute_AUC_distribution_statistics(model_keys, unit_idxs, lead_lag_key, plot=False):
        
    if unit_idxs is None:
        unit_idxs = range(results_dict[lead_lag_key]['all_models_summary_results'].shape[0])

    spike_samples = results_dict[lead_lag_key]['sampled_data']['spike_samples']
    nSpikeSamples = spike_samples.shape[1]

    p_ttest       = []
    p_signtest    = []
    prop_signtest = []
    ci_signtest_lower   = []
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
        
        if plot:
            fig, ax = plt.subplots(figsize=plot_params.distplot_figsize)
            sns.kdeplot(data=auc_df, ax=ax, x='AUC', hue='Model',
                          log_scale=False, fill=False, linewidth=plot_params.distplot_linewidth,
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
    
    stats_df = pd.DataFrame(data = zip(p_ttest, p_signtest, prop_signtest, ci_signtest_lower),
                            columns = ['pval_t', 'pval_sign', 'proportion_sign', 'CI_lower'])
    
    return stats_df
            
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

        results_dict[lead_lag_key]['all_models_summary_results'] = all_units_res
        
def prune_for_neurons_with_same_channel_connections(units_res):
    
    units_res_pruned = units_res.copy()
    
    for elec_label in np.unique(units_res['electrode_label']):
        if (units_res['electrode_label'] == elec_label).sum() < 2:
            units_res_pruned.loc[units_res['electrode_label'] == elec_label, :] = np.nan
    
    return units_res_pruned
        
def plot_model_auc_comparison(units_res, x_key, y_key, minauc = 0.5, maxauc = 1.0, hue_key='W_in', 
                              style_key='cortical_area', targets=None, col_key=None, hue_order=None, 
                              col_order=None, style_order=None, paperFig='unknown', asterisk='', palette=None):
    
    if x_key[-4:] != '_auc':
        x_key = x_key + '_auc'
    if y_key[-4:] != '_auc':
        y_key = y_key + '_auc'
    
    try:
        xlabel = [f'{lab} AUC' for lab, key in zip(['Trajectory', 'Full Kinematics', 'Velocity', 'Short Kinematics', 'Kinematics + reachFN', 'Kinematics + spontaneousFN \nGeneralization'], 
                                                   ['traj', 'traj_avgPos', 'shortTraj', 'shortTraj_avgPos', 'traj_avgPos_reach_FN', 'traj_avgPos_spont_train_reach_test_FN']) if f'{key}_auc' == x_key][0]
    except:
        xlabel = x_key
    
    try:
        ylabel = [f'{lab} AUC' for lab, key in zip(['Trajectory', 'Full Kinematics', 'Velocity', 'Short Kinematics', 'Kinematics + reachFN', 'Kinematics + spontaneousFN \nGeneralization'], 
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
            text_y = 1.05*maxauc 
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

def get_training_metric_distributions_and_means(units_res, model_keys, lead_lag_key, metric = 'logLikelihood'):
    
    model_results_dict = results_dict[lead_lag_key]['model_results']
    
    metric_distributions = []
    for key in model_keys:
        metric_values = model_results_dict[key][metric]
        metric_distributions.append(metric_values)
        units_res['%s_%s' % (key, metric)] = np.nanmean(metric_values, axis = 1)
    
    return units_res, metric_distributions
    
def plot_model_training_performance_comparison(units_res, x_key, y_key, lead_lag_key, metric='logLikelihood', targets=None, paperFig='unknown'):
    
    units_res, metric_distributions = get_training_metric_distributions_and_means(units_res, [x_key, y_key], lead_lag_key, metric = metric)
    
    units_res_plots = units_res.copy()
        
    if targets is not None:
        if type(targets) != list:
            targets = [targets]
        units_res_plots = isolate_target_units_for_plots(units_res_plots, targets)        
        plot_title = 'Targets:'
        plot_name = f'{marmcode}_{metric}_{x_key}_{y_key}_targetUnits'
        for targ in targets:
            plot_title = plot_title + f' {targ},'
            plot_name = plot_name + '_{targ}'
        plot_name = plot_name + '.png'
    else:
        plot_title = 'Targets: All units'
        plot_name = f'{marmcode}_{metric}_{x_key}_{y_key}.png'

    
    fig, ax = plt.subplots(figsize = plot_params.aucScatter_figSize, dpi=plot_params.dpi)
    # sns.scatterplot(ax = ax, data = units_res, x = x_key, y = y_key, 
    #                 hue = "fr", style = "group")
    sns.scatterplot(ax = ax, data = units_res_plots, x = '%s_%s' % (x_key, metric), y = '%s_%s' % (y_key, metric), 
                    style = "quality", s = 60, legend=False)
    
    if 'auc' in metric.lower():
        metric_min = 0.5
        metric_max = 1.0
    else:
        metric_min = np.min(units_res_plots['%s_%s' % (x_key, metric)].min(), units_res_plots['%s_%s' % (y_key, metric)].min())
        metric_max = np.max(units_res_plots['%s_%s' % (x_key, metric)].max(), units_res_plots['%s_%s' % (y_key, metric)].max())

    ax.plot(np.linspace(metric_min, metric_max, 100), np.linspace(metric_min, metric_max, 100), '--k')
    ax.set_xlim(metric_min, metric_max)
    ax.set_ylim(metric_min, metric_max)
    # ax.scatter(units_res_plots[x_key].to_numpy()[44] , units_res_plots[y_key].to_numpy()[44] , s = 60, c ='red', marker='x')
    # ax.scatter(units_res_plots[x_key].to_numpy()[107], units_res_plots[y_key].to_numpy()[107], s = 60,  c ='red', marker='o')
    # ax.set_xlim(minauc, 1)
    # ax.set_ylim(minauc, 1)
    # for axis in ['bottom','left']:
    #     ax.spines[axis].set_linewidth(2)
    #     ax.spines[axis].set_color('black')
    # for axis in ['top','right']:
    #     ax.spines[axis].set_linewidth(0)
    # ax.tick_params(width=2, length = 4, labelsize = plot_params.tick_fontsize)
    ax.set_xlabel('%s_%s' % (x_key, metric), fontsize = plot_params.axis_fontsize)
    ax.set_ylabel('%s_%s' % (y_key, metric), fontsize = plot_params.axis_fontsize)
    # ax.set_xlabel('')
    # ax.set_ylabel('')
    ax.grid(False)
    ax.set_title(plot_title)
    # ax.legend(bbox_to_anchor=(1.5, 1.5), loc='center left', borderaxespad=0)
    plt.show()
    
    if paperFig == 'unknown':
        fig_base = os.path.join(plots, paperFig, 'network')
    else:
        fig_base = os.path.join(plots, paperFig)
        
    fig.savefig(os.path.join(fig_base, plot_name + '.png'), bbox_inches='tight', dpi=plot_params.dpi)
    
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

def plot_functional_networks(FN, units_res, FN_key = 'split_reach_FNs', cmin=None, cmax=None, subset_idxs = None, subset_type='both', paperFig='unknown'):
    
    # units_sorted = units_res.copy()    
    # units_sorted.sort_values(by='cortical_area', inplace=True, ignore_index=False)
    units_sorted, ticks, labels = sort_units_by_area_for_FN_plots(units_res.copy())
                          
    if subset_idxs is not None:
        units_res_subset = units_res.copy().loc[subset_idxs, :]
        units_sorted_subset, ticks_subset, labels_subset = sort_units_by_area_for_FN_plots(units_res_subset.copy()) 

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
        
            fsize = (fsize[0]*np.log(len(subset_idxs))/np.log(plot_params.mostUnits_FN), fsize[1]*np.log(len(subset_idxs))/np.log(plot_params.mostUnits_FN))  

            
            if subset_type == 'both':
                if subset_idxs.size > FN.shape[-1]/2:
                    title = 'Non-Specific\n' + title
                else:
                    title = 'Reach-Specific\n' + title
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
            fsize = (fsize[0]*np.log(network_copy.shape[1])/np.log(plot_params.mostUnits_FN), fsize[1]*np.log(network_copy.shape[0])/np.log(plot_params.mostUnits_FN))
            # fsize = (fsize[0]*network_copy.shape[1]/network.shape[1], fsize[1]*network_copy.shape[0]/network.shape[0])


        network_copy = network_copy[np.ix_(target_idx, source_idx)]
        
        fig, ax = plt.subplots(figsize=fsize, dpi = plot_params.dpi)
        
        sns.heatmap(network_copy,ax=ax,cmap= 'viridis',square=True, norm=colors.PowerNorm(0.5, vmin=cmin, vmax=cmax)) # norm=colors.LogNorm(vmin=cmin, vmax=cmax)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels(xlabels)
        ax.set_yticklabels(ylabels)
        ax.set_title(title, fontsize=plot_params.axis_fontsize)
        ax.set_ylabel('Target Unit', fontsize=plot_params.axis_fontsize)
        ax.set_xlabel('Source Unit' , fontsize=plot_params.axis_fontsize)
        plt.show()
        
        title = title.replace('\n', '_')
        fig.savefig(os.path.join(plots, paperFig, 
                                 f'{marmcode}_functional_network_{title.replace(" ", "_").replace("-", "_")}.png'), bbox_inches='tight', dpi=plot_params.dpi)
    
        plt.hist(network_copy.flatten(), bins = 30)
        plt.show()
        
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
                 err_style="bars", errorbar='se', linewidth=plot_params.distplot_linewidth,
                 palette=palette)
    # ax.set_ylabel(f'Wji (mean %s sem)' % '\u00B1', fontsize = plot_params.axis_fontsize)
    ax.set_ylabel('$W_{{ji}}$ (mean \u00B1 sem)', fontsize = plot_params.axis_fontsize)
    # ax.set_xlabel('Inter-Unit Distance (%sm)' % '\u03bc', fontsize = plot_params.axis_fontsize)
    ax.set_xlabel('Inter-Unit Distance (\u03bcm)', fontsize = plot_params.axis_fontsize)

    if ymin:
        ax.set_ylim(ymin, ymax)
    else:
        ymin, ymax = ax.get_ylim()

    ax.legend(bbox_to_anchor=(1.1, 0.5), loc='center left', 
              labels  = weights_df['Reaches'].unique(), 
              title_fontsize = plot_params.axis_fontsize,
              fontsize = plot_params.tick_fontsize)
    sns.despine(ax=ax)
    plt.show()
        
    fig.savefig(os.path.join(plots, paperFig, f'{marmcode}_weights_by_distance.png'), bbox_inches='tight', dpi=plot_params.dpi)
    
    return ymin, ymax

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
            elif kin_type == 'traj_avgPos_full_FN':
                sample_coefs = unit_coefs[1:-5, sample]                
                
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
            significant_diff_df = pd.concat((significant_diff_df, unit_df), axis = 0)
            # f, pval =  f_oneway(unit_df.loc[unit_df['label'] == feature_sample_times[0], 'norm'],
            #                     unit_df.loc[unit_df['label'] == feature_sample_times[1], 'norm'],
            #                     unit_df.loc[unit_df['label'] == feature_sample_times[2], 'norm'],
            #                     unit_df.loc[unit_df['label'] == feature_sample_times[3], 'norm'],
            #                     unit_df.loc[unit_df['label'] == feature_sample_times[4], 'norm'],
            #                     unit_df.loc[unit_df['label'] == feature_sample_times[5], 'norm'],
            #                     unit_df.loc[unit_df['label'] == feature_sample_times[6], 'norm'],
            #                     unit_df.loc[unit_df['label'] == feature_sample_times[7], 'norm'],
            #                     unit_df.loc[unit_df['label'] == feature_sample_times[8], 'norm'],
            #                     unit_df.loc[unit_df['label'] == feature_sample_times[9], 'norm'],
            #                     unit_df.loc[unit_df['label'] == feature_sample_times[10], 'norm'],
            #                     unit_df.loc[unit_df['label'] == feature_sample_times[11], 'norm'],
            #                     unit_df.loc[unit_df['label'] == feature_sample_times[12], 'norm'],
            #                     unit_df.loc[unit_df['label'] == feature_sample_times[13], 'norm'],
            #                     unit_df.loc[unit_df['label'] == feature_sample_times[14], 'norm'],
            #                     unit_df.loc[unit_df['label'] == feature_sample_times[15], 'norm'])
            # if pval<1:
            #     significant_diff_df = pd.concat((significant_diff_df, unit_df), axis = 0)
    
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

def add_in_weight_to_units_df(units_res, FN):
    
    units_res = units_res.copy()
    
    idx_m1  = np.where(units_res['cortical_area'] == 'M1')[0]
    idx_3a  = np.where(units_res['cortical_area'] == '3a')[0]
    idx_3b  = np.where(units_res['cortical_area'] == '3b')[0]
    idx_6dc = np.where(units_res['cortical_area'] == '6dc')[0]
    
    in_weights     = []
    in_weights_m1  = []
    in_weights_3a  = []
    in_weights_3b  = []
    in_weights_6dc = []
    out_weights    = []
    for unit_idx in units_res.index:
        if FN.ndim == 3:
            w_in  = (FN[0, unit_idx].sum() + FN[1, unit_idx].sum())/2 / FN.shape[-1]
            w_out = (FN[0, :, unit_idx].sum() + FN[1, :, unit_idx].sum())/2 / FN.shape[-1]
            
            w_in_m1  = (FN[0, unit_idx, idx_m1 ].sum() + FN[1, unit_idx, idx_m1 ].sum())/2 / len(idx_m1)
            w_in_3a  = (FN[0, unit_idx, idx_3a ].sum() + FN[1, unit_idx, idx_3a ].sum())/2 / len(idx_3a)
            w_in_3b  = (FN[0, unit_idx, idx_3b ].sum() + FN[1, unit_idx, idx_3b ].sum())/2 / len(idx_3b)
            w_in_6dc = (FN[0, unit_idx, idx_6dc].sum() + FN[1, unit_idx, idx_6dc].sum())/2 / len(idx_6dc)
        else:
            w_in  = FN[unit_idx].sum()
            w_out = FN[:, unit_idx].sum()
        in_weights.append(w_in)
        in_weights_m1.append (w_in_m1)
        in_weights_3a.append (w_in_3a)
        in_weights_3b.append (w_in_3b)
        in_weights_6dc.append(w_in_6dc)
        
        out_weights.append(w_out)
    
    tmp_df = pd.DataFrame(data = zip(in_weights, out_weights, in_weights_m1, in_weights_3a, in_weights_3b, in_weights_6dc),
                          columns = ['W_in',     'W_out',     'W_in_m1',     'W_in_3a',     'W_in_3b', 'W_in_6dc'],
                          index = units_res.index)

    
    units_res = pd.concat((units_res, tmp_df), axis = 1)
    
    return units_res

def add_modulation_data_to_units_df(units_res):

    with open(f'{modulation_base}_modulationData.pkl', 'rb') as f:
        modulation_df = dill.load(f)     
    mask = [True if int(uName) in units_res.unit_name.astype(int).values else False for uName in modulation_df.unit_name.values]
    modulation_df = modulation_df.loc[mask, :]

    with open(f'{modulation_base}_average_firing_rates.pkl', 'rb') as f:
        average_rates_df = dill.load(f)     

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
            
            # or percent%10!=0 and percent != 99
            if percent in [1]:
                continue
            
            shuffle_auc = results_dict[lead_lag_key]['model_results'][model_key][results_key].copy()
            
            
            comparison_auc = comparison_all_units_auc[target_idxs, :]
            shuffle_auc    = shuffle_auc             [target_idxs, :]
            if kin_only_model is not None and results_key in results_dict[lead_lag_key]['model_results'][kin_only_model].keys():
                kin_only_auc = kin_only_all_units_auc  [target_idxs, :]                
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
    tmp_df2['percent'] = [np.where(percent_vals == per)[0][0] for per in tmp_df2['percent']]
    # train_auc_df = pd.concat((train_auc_df, tmp_df_diff), axis = 0, ignore_index=True)    

    # fig = sns.catplot(data = train_auc_df, x='percent', y='auc_loss (%)', col='mode', hue='metric', kind='point', legend=True, errorbar='se')
    
    sig_strength_v_random = dict(weights=[], topology=[])
    for mode in train_auc_df['mode'].unique(): 
        for idx, percent in enumerate(train_auc_df['percent'].unique()):    
            stats_df = train_auc_df.loc[(train_auc_df['mode'] == mode) & (train_auc_df['percent'] == percent), :]
            nStrength = np.sum(stats_df.loc[stats_df['metric'] == 'strength', 'auc_loss (%)'].values > 
                               stats_df.loc[stats_df['metric'] == 'random', 'auc_loss (%)'].values)
            nUnits = np.sum(stats_df['metric'] == 'strength')
            
            sign_test = binomtest(nStrength, nUnits, p = 0.5, alternative='two-sided')
            print(f'Strength != Random, {percent}%, {mode}:  p={np.round(sign_test.pvalue, 4)}, nStrength={sign_test.k}, nUnits={sign_test.n}, prop={np.round(sign_test.proportion_estimate, 2)}')

            if sign_test.pvalue < alpha:
                sig_strength_v_random[mode].append(idx)

    sig_noFN_v_strength = dict(weights=[], topology=[])
    for mode in train_auc_df['mode'].unique(): 
        for idx, percent in enumerate(train_auc_df['percent'].unique()):    
            stats_df = train_auc_df.loc[(train_auc_df['mode'] == mode) & (train_auc_df['percent'] == percent), :]
            kin_comp_stats_df = tmp_df2.loc[(tmp_df2['mode'] == mode) & (tmp_df2['percent'] == 0), :]
            nStrength = np.sum(stats_df.loc[stats_df['metric'] == 'strength', 'auc_loss (%)'].values > kin_comp_stats_df['auc_loss (%)'].values)
            nUnits = np.sum(stats_df['metric'] == 'strength')
            
            sign_test = binomtest(nStrength, nUnits, p = 0.5, alternative='two-sided')
            print(f'Strength != No-FN, {percent}%, {mode}:  p={np.round(sign_test.pvalue, 4)}, nStrength={sign_test.k}, nUnits={sign_test.n}, prop={np.round(sign_test.proportion_estimate, 2)}')

            if sign_test.pvalue < alpha:
                sig_noFN_v_strength[mode].append(idx)

    sig_noFN_v_random = dict(weights=[], topology=[])
    for mode in train_auc_df['mode'].unique(): 
        for idx, percent in enumerate(train_auc_df['percent'].unique()):   
            stats_df = train_auc_df.loc[(train_auc_df['mode'] == mode) & (train_auc_df['percent'] == percent), :]
            kin_comp_stats_df = tmp_df2.loc[(tmp_df2['mode'] == mode) & (tmp_df2['percent'] == 0), :]
            nStrength = np.sum(stats_df.loc[stats_df['metric'] == 'random', 'auc_loss (%)'].values > kin_comp_stats_df['auc_loss (%)'].values)
            nUnits = np.sum(stats_df['metric'] == 'random')
            
            sign_test = binomtest(nStrength, nUnits, p = 0.5, alternative='two-sided')
            print(f'Random != No-FN, {percent}%, {mode}:  p={np.round(sign_test.pvalue, 4)}, nRandom={sign_test.k}, nUnits={sign_test.n}, prop={np.round(sign_test.proportion_estimate, 2)}')
    
            if sign_test.pvalue < alpha:
                sig_noFN_v_random[mode].append(idx)
                
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
    sns.pointplot(ax=ax0, data = train_auc_df.loc[train_auc_df['mode'] == 'weights', :], x='percent', 
                  y='auc_loss (%)', hue='metric', errorbar='se', scale=plot_params.shuffle_markerscale, errwidth=plot_params.shuffle_errwidth, palette=palette)
    sns.pointplot(ax=ax1, data = train_auc_df.loc[train_auc_df['mode'] == 'topology', :], x='percent', 
                  y='auc_loss (%)', hue='metric', errorbar='se', scale=plot_params.shuffle_markerscale, errwidth=plot_params.shuffle_errwidth, palette=palette)
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

    
    xticklabels = [lab.get_text() for lab in ax0.get_xticklabels()]
    xticklabels = [lab if int(lab)%10==0 else '' for lab in xticklabels]
    xticklabels[-1] = '100'
    for ax in [ax0, ax1]:
        ax.set_xlabel('Percent of FN Permuted')
        ax.set_ylim(ylim)
        ax.set_xticklabels(xticklabels)

    ax0.set_ylabel('AUC Percent Loss')    
    ax1.set_ylabel('')    
    ax0.set_title('Permuted Weights')
    ax1.set_title('Permuted Edges')
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

def identify_connections_by_strength(weights, percent):
    shuffle_idxs = np.where(weights > np.percentile(weights, 100-percent))

    weights_at_idxs = np.full_like(shuffle_idxs[0], np.nan, dtype=weights.dtype)
    for idx, (target_unit, source_unit) in enumerate(zip(shuffle_idxs[0], shuffle_idxs[1])):
        weights_at_idxs[idx] = weights[target_unit, source_unit]        
    
    shuffle_set = (shuffle_idxs[0], shuffle_idxs[1], weights_at_idxs)
    
    return shuffle_set

def identify_shuffle_set_by_strength(weights, percent, target_idxs = None, source_idxs = None):
    
    if target_idxs is not None:
        subgraph_weights = weights.copy()[np.ix_(target_idxs, source_idxs)]
        subgraph_shuffle_set  = identify_connections_by_strength(subgraph_weights, percent)
        shuffle_set = (subgraph_shuffle_set[0].copy(), subgraph_shuffle_set[1].copy(), subgraph_shuffle_set[2].copy())
        for idx, (target_unit, source_unit) in enumerate(zip(subgraph_shuffle_set[0], subgraph_shuffle_set[1])):
            shuffle_set[0][idx] = target_idxs[target_unit]
            shuffle_set[1][idx] = source_idxs[source_unit]
    else:
        shuffle_set  = identify_connections_by_strength(weights, percent)

    return shuffle_set

def grab_cortical_area_FN_idxs(units_res):
    
    cortical_area_idxs= dict()
    for regions, set_name in zip([['M1'], ['3a', '3b'], ['3a'], ['3b'], ['M1', '3b'], ['3a', 'M1']], 
                                  ['motor', 'sensory', '3a', '3b', '3b_and_motor', '3a_and_motor']):  
    # for regions, set_name in zip([['M1'], ['3a'], ['3b']], 
    #                              ['motor', '3a', '3b']):         
        area_units = units_res.copy()
        mask = area_units['cortical_area'] == 0
        for reg in regions:
            mask = mask | (area_units['cortical_area'] == reg)
            
        area_units = area_units.loc[mask, 'cortical_area']
        
        # if network.shape[0] < units_res.shape[0]:
        #     area_FN = network_copy[np.ix_(list(range(network.shape[0])), area_units.index.values, area_units.index.values)]
        # else:
        #     area_FN = network_copy[np.ix_(area_units.index.values, area_units.index.values)]

        cortical_area_idxs[set_name] = area_units.index.values
            
    return cortical_area_idxs

def add_neuron_classifications(units_res):
    class_file = h5py.File(glob.glob(os.path.join(os.path.dirname(nwb_infile), 'Good*.mat'))[0])
    neuron_classes = class_file['classes'][:]
    
    neuron_type_strings = ['', 'exc', 'inh', 'unclassified']
    neuron_classes = [neuron_type_strings[int(cl)] for idx, cl in enumerate(neuron_classes) if idx not in filtered_good_units_idxs]
    
    units_res['neuron_type'] = np.full((units_res.shape[0],), 'mua')
    units_res.loc[units_res['quality'] == 'good','neuron_type'] = neuron_classes
    
    return units_res

def parse_properties_of_shuffled_sources(units_res, percent, shuffle_sets, electrode_distances, source_props=None):

    if source_props is None:
        source_props = pd.DataFrame()

    for idx, shuffle_set in enumerate(shuffle_sets):
        for target in np.unique(shuffle_set[0]):
            sources = shuffle_set[1][shuffle_set[0] == target]
            weights = shuffle_set[2][shuffle_set[0] == target]
            
            units_res_sources = units_res.iloc[sources, :]
            
            target_tuning = 'tuned' if units_res['proportion_sign'].iloc[target] >= params.significant_proportion_thresh else 'untuned'
            target_features = dict(target_cortical_area = units_res['cortical_area'].iloc[target],
                                   target_neuron_type   = units_res['neuron_type'].iloc[target],
                                   target_tuning        = target_tuning)
            
            n_tuned_motor_inputs   = np.sum((units_res_sources['proportion_sign'] >= params.significant_proportion_thresh) &
                                            (units_res_sources['cortical_area'] == 'M1'))
            n_tuned_sensory_inputs = np.sum((units_res_sources['proportion_sign'] >= params.significant_proportion_thresh) & 
                                            ((units_res_sources['cortical_area'] == '3a') | (units_res_sources['cortical_area'] == '3b')))
            n_untuned_motor_inputs = np.sum((units_res_sources['proportion_sign'] < params.significant_proportion_thresh) &
                                            (units_res_sources['cortical_area'] == 'M1'))
            n_untuned_sensory_inputs = np.sum((units_res_sources['proportion_sign'] < params.significant_proportion_thresh) &
                                              ((units_res_sources['cortical_area'] == '3a') | (units_res_sources['cortical_area'] == '3b')))
            source_features = dict(n_source_edges   = sources.size,
                                   mean_edge_weight = np.round(weights.mean(), 5),
                                   n_m1_inputs      = np.sum(units_res_sources['cortical_area'] == 'M1'),
                                   n_3a_inputs      = np.sum(units_res_sources['cortical_area'] == '3a'),
                                   n_3b_inputs      = np.sum(units_res_sources['cortical_area'] == '3b'),
                                   mean_input_dist  = np.round(np.mean(electrode_distances[target, sources]), 0),
                                   n_tuned_inputs   = np.sum(units_res_sources['proportion_sign'] >= params.significant_proportion_thresh),
                                   n_untuned_inputs = np.sum(units_res_sources['proportion_sign'] < params.significant_proportion_thresh),
                                   n_tuned_motor_inputs     = n_tuned_motor_inputs,
                                   n_tuned_sensory_inputs   = n_tuned_sensory_inputs,
                                   n_untuned_motor_inputs   = n_untuned_motor_inputs,
                                   n_untuned_sensory_inputs = n_untuned_sensory_inputs,
                                   n_excitatory_inputs = np.sum(units_res_sources['neuron_type'] == 'exc'),
                                   n_inhibitory_inputs = np.sum(units_res_sources['neuron_type'] == 'inh'))
            
            tmp_df = pd.DataFrame() 
            for key, value in target_features.items():
                tmp_df[key] = [value]
            for key, value in source_features.items():
                tmp_df[key] = [value]
            tmp_df['percent'] = [percent]
            tmp_df['FN_idx'] = [idx]
            
            source_props = pd.concat((source_props, tmp_df), axis=0, ignore_index=True)
    
    return source_props

def parse_properties_of_FN_subsets(units_res, FN, electrode_distances, FN_key = 'split_reach_FNs', subset_idxs = None, subset_types=None, subset_basis=['Reach-Specific'], tune = ('traj_avgPos_auc', 0.6), source_props=None):
    
    # if FN.ndim < 3:
        # FN = np.expand_dims(FN, axis = 0)
    FN_mean = FN.mean(axis=0) if FN.ndim == 3 else FN.copy()
    
    if source_props is None:
        source_props = pd.DataFrame()
    
    for sub_type, sub_basis in product(subset_types, subset_basis):
        
        if 'Non-' in sub_basis:
            subset_idxs = np.setdiff1d(np.array(range(units_res.shape[0])), subset_idxs)
        
        if sub_type == 'both':
            targets, sources, sub_FN = subset_idxs, subset_idxs, FN_mean[np.ix_(subset_idxs, subset_idxs)]
        elif sub_type == 'target':
            targets, sources, sub_FN = subset_idxs, np.arange(FN_mean.shape[1]), FN_mean[np.ix_(subset_idxs, range(FN_mean.shape[1]))]
        elif sub_type == 'source':
            targets, sources, sub_FN = np.arange(FN_mean.shape[0]), subset_idxs, FN_mean[np.ix_(range(FN_mean.shape[0]), subset_idxs)]
    
        # units_res_targets = units_res.loc[targets, :]
        units_res_sources = units_res.loc[sources, :]
    
        for tIdx, target in enumerate(targets):
            weights = sub_FN[tIdx]
               
            target_tuning = 'tuned' if units_res[tune[0]].loc[target] >= tune[1] else 'untuned'
            target_features = dict(target_cortical_area = units_res['cortical_area'].loc[target],
                                   target_neuron_type   = units_res['neuron_type'].loc[target],
                                   target_tuning        = target_tuning)
            
            n_tuned_motor_inputs   = np.sum((units_res_sources[tune[0]] >= tune[1]) &
                                            (units_res_sources['cortical_area'] == 'M1'))
            n_tuned_sensory_inputs = np.sum((units_res_sources[tune[0]] >= tune[1]) & 
                                            ((units_res_sources['cortical_area'] == '3a') | (units_res_sources['cortical_area'] == '3b')))
            n_untuned_motor_inputs = np.sum((units_res_sources[tune[0]] < tune[1]) &
                                            (units_res_sources['cortical_area'] == 'M1'))
            n_untuned_sensory_inputs = np.sum((units_res_sources[tune[0]] < tune[1]) &
                                              ((units_res_sources['cortical_area'] == '3a') | (units_res_sources['cortical_area'] == '3b')))
            source_features = dict(n_source_edges   = sources.size,
                                   mean_Win    = np.round(weights.mean(), 5),
                                   mean_Win_m1 = np.round(weights[units_res_sources.cortical_area=='M1'].mean(), 5),
                                   mean_Win_3a = np.round(weights[units_res_sources.cortical_area=='3a'].mean(), 5),
                                   mean_Win_3b = np.round(weights[units_res_sources.cortical_area=='3b'].mean(), 5),
                                   input_share_m1      = np.round(np.sum(units_res_sources['cortical_area'] == 'M1') / units_res_sources.shape[0], 3),
                                   input_share_3a      = np.round(np.sum(units_res_sources['cortical_area'] == '3a') / units_res_sources.shape[0], 3),
                                   input_share_3b      = np.round(np.sum(units_res_sources['cortical_area'] == '3b') / units_res_sources.shape[0], 3),
                                   input_share_tuned   = np.round(np.sum(units_res_sources[tune[0]] >= tune[1])      / units_res_sources.shape[0], 3),
                                   input_share_untuned = np.round(np.sum(units_res_sources[tune[0]] <  tune[1])      / units_res_sources.shape[0], 3),
                                   input_share_exc     = np.round(np.sum(units_res_sources['neuron_type'] == 'exc')  / units_res_sources.shape[0], 3),
                                   input_share_inh     = np.round(np.sum(units_res_sources['neuron_type'] == 'inh')  / units_res_sources.shape[0], 3))


                                   # input_share_m1      = np.sum(units_res_sources['cortical_area'] == 'M1') / np.sum(units_res['cortical_area'] == 'M1'),
                                   # input_share_3a      = np.sum(units_res_sources['cortical_area'] == '3a') / np.sum(units_res['cortical_area'] == '3a'),
                                   # input_share_3b      = np.sum(units_res_sources['cortical_area'] == '3b') / np.sum(units_res['cortical_area'] == '3b'),
                                   # input_share_tuned   = np.sum(units_res_sources[tune[0]] >= tune[1]) / np.sum(units_res[tune[0]] >= tune[1]),
                                   # input_share_untuned = np.sum(units_res_sources[tune[0]] <  tune[1]) / np.sum(units_res[tune[0]] <  tune[1]),
                                   # percent_exc_inputs     = np.sum(units_res_sources['neuron_type'] == 'exc') / np.sum(units_res['neuron_type'] == 'exc'),
                                   # percent_inh_inputs     = np.sum(units_res_sources['neuron_type'] == 'inh') / np.sum(units_res['neuron_type'] == 'inh')            
                                   # n_tuned_motor_inputs     = n_tuned_motor_inputs,
                                   # n_tuned_sensory_inputs   = n_tuned_sensory_inputs,
                                   # n_untuned_motor_inputs   = n_untuned_motor_inputs,
                                   # n_untuned_sensory_inputs = n_untuned_sensory_inputs,
            
            tmp_df = pd.DataFrame() 
            for key, value in target_features.items():
                tmp_df[key] = [value]
            for key, value in source_features.items():
                tmp_df[key] = [value]
            tmp_df['sub_type']  = sub_type
            tmp_df['FN_key']    = FN_key
            tmp_df['Units_Subset'] = sub_basis
            
            tmp_df = tmp_df.copy()
            
            source_props = pd.concat((source_props, tmp_df), axis=0, ignore_index=True)
    
    return source_props

def plot_modulation_for_subsets(auc_df, paperFig = 'unknown', figname_mod = '', hue_order_FN=None, palette=None):    

    plot_save_dir = os.path.join(plots, paperFig, figname_mod[1:]) 
    os.makedirs(plot_save_dir, exist_ok=True)

    for metric in auc_df.columns:
        if 'mod' in metric or 'dev' in metric or 'frate' in metric: 
            
            tmp_df = auc_df.loc[~np.isnan(auc_df[metric]), [metric, 'Units Subset', 'quality']]
            
            if 'nomua' in figname_mod.lower():
                tmp_df = tmp_df.loc[tmp_df['quality'] == 'good', :]
            
            med_out = median_test(tmp_df.loc[tmp_df['Units Subset'] == 'Reach-Specific', metric], 
                                  tmp_df.loc[tmp_df['Units Subset'] == 'Non-Specific', metric],
                                  nan_policy='omit')
            print(f'{metric}_{figname_mod}: reach v non, p={np.round(med_out[1], 4)}, {med_out[-1][0,0]}a-{med_out[-1][1,0]}b, {med_out[-1][0,1]}a-{med_out[-1][1,1]}b')
            pval = np.round(med_out[1], 4)
            
            med_out = median_test(tmp_df.loc[tmp_df['Units Subset'] == 'Reach-Specific', metric], 
                                  tmp_df.loc[tmp_df['Units Subset'] == 'Full', metric],
                                  nan_policy='omit')
            print(f'{metric}_{figname_mod}: reach v full, p={np.round(med_out[1], 4)}, {med_out[-1][0,0]}a-{med_out[-1][1,0]}b, {med_out[-1][0,1]}a-{med_out[-1][1,1]}b')
            med_out = median_test(tmp_df.loc[tmp_df['Units Subset'] == 'Non-Specific', metric], 
                                  tmp_df.loc[tmp_df['Units Subset'] == 'Full', metric],
                                  nan_policy='omit')
            print(f'{metric}_{figname_mod}: non v full, p={np.round(med_out[1], 4)}, {med_out[-1][0,0]}a-{med_out[-1][1,0]}b, {med_out[-1][0,1]}a-{med_out[-1][1,1]}b')
            
            fig, ax = plt.subplots(figsize=plot_params.distplot_figsize)
            sns.kdeplot(ax=ax, data=tmp_df, palette=palette, linewidth=plot_params.distplot_linewidth,
                        x=metric, hue='Units Subset', common_norm=False, bw_adjust=0.4, cumulative=True, hue_order=hue_order_FN)
            ax.legend().remove()
            sns.despine(ax=ax)
            ax.set_xlabel(metric, fontsize=plot_params.axis_fontsize)
            ax.text(tmp_df[metric].max()*0.9, ax.get_ylim()[-1]*0.25, f'p={pval}', horizontalalignment='center', fontsize = 12)
            plt.show()        
            fig.savefig(os.path.join(plot_save_dir, f'{marmcode}_distribution_{metric}_huekey_UnitsSubset{figname_mod}.png'), bbox_inches='tight', dpi=plot_params.dpi)    

            fig, ax = plt.subplots(figsize=plot_params.distplot_figsize)
            sns.kdeplot(ax=ax, data=tmp_df, linewidth=plot_params.distplot_linewidth,
                        x=metric, common_norm=False, bw_adjust=0.4, cumulative=False, color='black')
            kde_x, kde_y = ax.get_children()[0].get_data()
            for uIdx, unit_data in tmp_df.iterrows():
                if unit_data['Units Subset'] == 'Reach-Specific':
                    idxs = np.where(np.isclose(unit_data[metric], kde_x, rtol = 1e-1))[0]
                    idx = idxs.mean().astype(int)
                    if idx < 0:
                        continue
                    ax.vlines(kde_x[idx], 0, kde_y[idx], color='blue', linewidth=0.5)
            ax.legend().remove()
            sns.despine(ax=ax)
            ax.set_xlabel(metric, fontsize=plot_params.axis_fontsize)
            ax.text(tmp_df[metric].max()*0.9, ax.get_ylim()[-1]*0.25, f'p={pval}', horizontalalignment='center', fontsize = 12)
            plt.show()        
            fig.savefig(os.path.join(plot_save_dir, f'{marmcode}_distribution_{metric}_histogram_highlighted_with_reachspecific{figname_mod}.png'), bbox_inches='tight', dpi=plot_params.dpi)    

            

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
    ax.set_xlabel('Full Kinematics AUC', fontsize=plot_params.axis_fontsize)
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
    ax.set_xlabel('Full Kinematics AUC', fontsize=plot_params.axis_fontsize)
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
    ax.set_xlabel('Preferred Trajectory\nCorrelation')
    sns.despine(ax=ax)
    plt.show()    
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
    plt.show()    
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
               
        
    fig, ax = plt.subplots(figsize=plot_params.stripplot_figsize, dpi=plot_params.dpi)
    sns.stripplot(ax=ax, data=weights_df, x='Units Subset', y='Wji', hue='FN_key', 
                  dodge=True, order=hue_order_FN, palette='FN_palette', s=plot_params.stripplot_markersize)
    ax.legend(bbox_to_anchor=(1.1, 0.5), loc='center left', 
              labels  = ['reachFN1', 'reachFN2', 'spontaneousFN'], 
              title_fontsize = plot_params.axis_fontsize,
              fontsize = plot_params.tick_fontsize)
    ax.set_ylabel('$W_{{ji}}$')
    sns.despine(ax=ax)
    fig.savefig(os.path.join(plots, 'FigS3', f'{marmcode}_striplot_Wji_groupedBy_UnitsSubset_huekey_FNkey{figname_mod}.png'), bbox_inches='tight', dpi=plot_params.dpi)       

    fig = sns.relplot(data=weights_df, x='pearson_r', y='Wji', col='Units Subset', kind='scatter')
    plt.show() 
    fig.savefig(os.path.join(plots, 'unknown', f'{marmcode}_subnetwork_pearson_r_squared_vs_wji_colkey_UnitsSubset{figname_mod}.png'), bbox_inches='tight', dpi=plot_params.dpi)    
    
        
# def plot_distributions_after_source_props(units_res, electrode_distances, 
#                                           traj_corr_df, FN_sets = [], subset_idxs = None, 
#                                           sub_type='both', subset_basis=['Reach-Specific'], 
#                                           good_only=False, plot_auc_matched = True,
#                                           hue_order_FN=None, palette=None):
    
#     weights_df = pd.DataFrame()
#     auc_df = pd.DataFrame()
#     for sub_basis, FN_set in product(subset_basis, FN_sets):
        
#         sub_idxs = subset_idxs.copy()
        
#         if 'Non-' in sub_basis:
#             sub_idxs = np.setdiff1d(np.array(range(units_res.shape[0])), sub_idxs)
#         elif sub_basis == 'Full':
#             sub_idxs = np.array(range(units_res.shape[0]))
        
#         FN_key = FN_set[0]
#         FN_tmp = FN_set[1]
        
#         FN_mean = FN_tmp.mean(axis=0) if FN_tmp.ndim == 3 else FN_tmp.copy()
        
#         if sub_type == 'both':
#             targets, sources, sub_FN = sub_idxs, sub_idxs, FN_mean[np.ix_(sub_idxs, sub_idxs)]
#             units_res_subset_units = units_res.loc[sources, :]  
#         elif sub_type == 'target':
#             targets, sources, sub_FN = sub_idxs, np.arange(FN_mean.shape[1]), FN_mean[np.ix_(sub_idxs, range(FN_mean.shape[1]))]
#             units_res_subset_units = units_res.loc[targets, :]
#         elif sub_type == 'source':
#             targets, sources, sub_FN = np.arange(FN_mean.shape[0]), sub_idxs, FN_mean[np.ix_(range(FN_mean.shape[0]), sub_idxs)]
#             units_res_subset_units = units_res.loc[sources, :]
        
#         units_res_targets = units_res.loc[targets, :]
#         units_res_sources = units_res.loc[sources, :]
        
#         subset_unit_names = [int(unit_name) for unit_name in units_res_subset_units['unit_name'].values] 
#         target_unit_names = [int(unit_name) for unit_name in units_res_targets['unit_name'].values]
#         source_unit_names = [int(unit_name) for unit_name in units_res_sources['unit_name'].values]

#         correlation_mask  = [True if (unit1 in subset_unit_names and unit2 in subset_unit_names) else False for unit1, unit2 in zip(traj_corr_df['unit1'], traj_corr_df['unit2'])]
#         sub_correlations  = traj_corr_df.loc[correlation_mask, 'Pearson_corr'].values
#         sub_corr_names    = traj_corr_df.loc[correlation_mask, ['unit1', 'unit2']]
    
#         sub_corr_i = [np.where(np.array(target_unit_names) == unit1)[0][0] for unit1 in sub_corr_names.unit1] 
#         sub_corr_j = [np.where(np.array(target_unit_names) == unit2)[0][0] for unit2 in sub_corr_names.unit2] 
        
#         sub_corr_array  = np.full_like(sub_FN, 0)
#         sub_higher_auc  = np.full_like(sub_corr_array, 0)
#         sub_lower_auc   = np.full_like(sub_corr_array, 0)
#         sub_average_auc = np.full_like(sub_corr_array, 0)
#         for i, j, corr, unit1, unit2 in zip(sub_corr_i, sub_corr_j, sub_correlations, sub_corr_names['unit1'], sub_corr_names['unit2']):
#             sub_corr_array[i, j] = corr
            
#             unit_mask = [True if int(unit_name) in [int(unit1), int(unit2)] else False for unit_name in units_res['unit_name'].values]
#             units_pair_res = units_res.loc[unit_mask, f'{params.primary_traj_model}_auc']
            
#             sub_higher_auc [i, j] = units_pair_res.max()
#             sub_lower_auc  [i, j] = units_pair_res.min()
#             sub_average_auc[i, j] = units_pair_res.mean()
            
#         sub_corr_array += sub_corr_array.transpose()
#         sub_higher_auc += sub_higher_auc.transpose()
#         sub_lower_auc += sub_lower_auc.transpose()
#         sub_average_auc += sub_average_auc.transpose()        
        
#         if good_only:
#             units_res_subset_units = units_res_subset_units.loc[units_res_subset_units['quality'] == 'good', :]  
#             good_targets = np.where(units_res_targets['quality'] == 'good')[0]
#             good_sources = np.where(units_res_sources['quality'] == 'good')[0]
#             units_res_targets = units_res_targets.iloc[good_targets, :]
#             units_res_sources = units_res_sources.iloc[good_sources, :]
#             sub_FN = sub_FN[good_targets, good_sources]
#             sub_corr_array = sub_corr_array[good_targets, good_sources]
#             sub_higher_auc = sub_higher_auc[good_targets, good_sources]
#             sub_lower_auc = sub_lower_auc[good_targets, good_sources]
#             sub_average_auc = sub_average_auc[good_targets, good_sources]

            
#         tmp_df = pd.DataFrame(data=zip(sub_FN.flatten(), 
#                                        np.tile(units_res_sources['cortical_area'], sub_FN.shape[0]),
#                                        np.repeat(FN_key, sub_FN.size),
#                                        np.repeat(sub_basis, sub_FN.size),
#                                        sub_corr_array.flatten(),
#                                        sub_higher_auc.flatten(),
#                                        sub_lower_auc.flatten(),
#                                        sub_average_auc.flatten()), 
#                               columns=['Wji', 'input_area', 'FN_key', 'Units Subset', 'pearson_r', 'high_auc', 'low_auc', 'avg_auc'])
#         tmp_df = tmp_df.loc[tmp_df['Wji'] != 0, :]
#         tmp_df['pearson_r_squared'] = tmp_df['pearson_r']**2
#         weights_df = pd.concat((weights_df, tmp_df))

#         if FN_key == FN_sets[0][0]:
#             tmp_auc_df = pd.DataFrame(data=zip(units_res_subset_units['traj_avgPos_auc'],
#                                                units_res_subset_units['cortical_area'],
#                                                np.repeat(sub_basis, units_res_subset_units.shape[0])),
#                                       columns=['Kinematics_AUC', 'cortical_area', 'Units Subset'])
#             auc_df = pd.concat((auc_df, tmp_auc_df))
#     if plot_auc_matched:
#         auc_df_auc_matched     = auc_df.loc[auc_df['Kinematics_AUC'] >= auc_df.loc[auc_df['Units Subset'] == 'Reach-Specific', 'Kinematics_AUC'].min(), :]
#         weights_df_auc_matched = weights_df.loc[weights_df['low_auc'] >= auc_df.loc[auc_df['Units Subset'] == 'Reach-Specific', 'Kinematics_AUC'].min(), :]

#         plot_wji_distributions_for_subsets(weights_df_auc_matched, paperFig = 'unknown', figname_mod = '_AUCmatched', hue_order_FN=hue_order_FN, palette=palette)
#         plot_trajectory_correlation_and_auc_distributions_for_subsets(auc_df_auc_matched, weights_df_auc_matched, paperFig='Fig7', figname_mod = '_AUCmatched', hue_order_FN=hue_order_FN, palette=palette)

#     else:   
#         auc_df_auc_matched = None
#         weights_df_auc_matched = None

#     plot_wji_distributions_for_subsets(weights_df, paperFig = 'unknown', figname_mod = '', hue_order_FN=hue_order_FN, palette=palette) 
#     plot_trajectory_correlation_and_auc_distributions_for_subsets(auc_df, weights_df, paperFig='Fig7', figname_mod = '', hue_order_FN=hue_order_FN, palette=palette)

#     return weights_df, auc_df, weights_df_auc_matched, auc_df_auc_matched

def plot_distributions_after_source_props(units_res, electrode_distances, 
                                          traj_corr_df, FN_sets = [], subset_idxs = None, 
                                          sub_type='both', subset_basis=['Reach-Specific'], 
                                          good_only=False, plot_auc_matched = True,
                                          hue_order_FN=None, palette=None):
    
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
                targets, sources, sub_FN = sub_idxs, sub_idxs, FN[np.ix_(sub_idxs, sub_idxs)]
                units_res_subset_units = units_res.loc[sources, :]  
            elif sub_type == 'target':
                targets, sources, sub_FN = sub_idxs, np.arange(FN.shape[1]), FN[np.ix_(sub_idxs, range(FN.shape[1]))]
                units_res_subset_units = units_res.loc[targets, :]
            elif sub_type == 'source':
                targets, sources, sub_FN = np.arange(FN.shape[0]), sub_idxs, FN[np.ix_(range(FN.shape[0]), sub_idxs)]
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
                sub_FN = sub_FN[good_targets, good_sources]
                sub_corr_array = sub_corr_array[good_targets, good_sources]
                sub_higher_auc = sub_higher_auc[good_targets, good_sources]
                sub_lower_auc = sub_lower_auc[good_targets, good_sources]
                sub_average_auc = sub_average_auc[good_targets, good_sources]
    
                
            tmp_df = pd.DataFrame(data=zip(sub_FN.flatten(), 
                                           np.tile(units_res_sources['cortical_area'], sub_FN.shape[0]),
                                           np.repeat(FN_name, sub_FN.size),
                                           np.repeat(sub_basis, sub_FN.size),
                                           sub_corr_array.flatten(),
                                           sub_higher_auc.flatten(),
                                           sub_lower_auc.flatten(),
                                           sub_average_auc.flatten()), 
                                  columns=['Wji', 'input_area', 'FN_key', 'Units Subset', 'pearson_r', 'high_auc', 'low_auc', 'avg_auc'])
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

        plot_wji_distributions_for_subsets(weights_df_auc_matched, paperFig = 'unknown', figname_mod = '_AUCmatched', hue_order_FN=hue_order_FN, palette=palette)
        plot_trajectory_correlation_and_auc_distributions_for_subsets(auc_df_auc_matched, weights_df_auc_matched, paperFig='Fig7', figname_mod = '_AUCmatched', hue_order_FN=hue_order_FN, palette=palette)
        plot_modulation_for_subsets(auc_df_auc_matched, paperFig='modulation', figname_mod = '_AUCmatched', hue_order_FN=hue_order_FN, palette=palette)
        plot_modulation_for_subsets(auc_df_auc_matched, paperFig='modulation', figname_mod = '_AUCmatched_noMUA', hue_order_FN=hue_order_FN, palette=palette)

    else:   
        auc_df_auc_matched = None
        weights_df_auc_matched = None

    plot_wji_distributions_for_subsets(weights_df, paperFig = 'unknown', figname_mod = '', hue_order_FN=hue_order_FN, palette=palette) 
    plot_trajectory_correlation_and_auc_distributions_for_subsets(auc_df, weights_df, paperFig='Fig7', figname_mod = '', hue_order_FN=hue_order_FN, palette=palette)
    plot_modulation_for_subsets(auc_df, paperFig='modulation', figname_mod = '_all', hue_order_FN=hue_order_FN, palette=palette)
    plot_modulation_for_subsets(auc_df, paperFig='modulation', figname_mod = '_noMUA', hue_order_FN=hue_order_FN, palette=palette)


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
        xlabel = [f'{lab}' for lab, key in zip(['Trajectory AUC', 'Full Kinematics AUC', 'Velocity AUC', 'Short Kinematics AUC', 
                                                'Kinematics + reachFN AUC', 'Kinematics + spontaneousFN \nGeneralization AUC',
                                                '$W_{{ji}}$', 'Average In-Weight'], 
                                               ['traj', 'traj_avgPos', 'shortTraj', 'shortTraj_avgPos', 
                                                'traj_avgPos_reach_FN', 'traj_avgPos_spont_train_reach_test_FN',
                                                'Wji', 'W_in']) if x_key in [key, f'{key}_auc']][0]
    except:
        xlabel = x_key
    
    try:
        ylabel = [f'{lab}' for lab, key in zip(['Trajectory AUC', 'Full Kinematics AUC', 'Velocity AUC', 'Short Kinematics AUC', 
                                                'Kinematics + reachFN AUC', 'Kinematics + spontaneousFN \nGeneralization AUC',
                                                '$W_{{ji}}$', 'Average In-Weight'], 
                                               ['traj', 'traj_avgPos', 'shortTraj', 'shortTraj_avgPos', 
                                                'traj_avgPos_reach_FN', 'traj_avgPos_spont_train_reach_test_FN',
                                                'Wji', 'W_in']) if y_key in [key, f'{key}_auc']][0]
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
    
    # sns.despine(fig=fig)
    plt.show()
    fig.savefig(os.path.join(plots, paperFig, plot_name), bbox_inches='tight', dpi=plot_params.dpi)
    
def get_grouped_means(units_res):
    
    features_list = ['W_in', 'W_in_m1', 'W_in_3a', 'W_in_3b', 'W_out', 'traj_avgPos_auc', 'traj_avgPos_reach_FN_auc', 'reach_FN_auc']
    
    by_class = units_res.groupby(['neuron_type']).mean().loc[:, features_list]
    by_area = units_res.groupby(['cortical_area']).mean().loc[:, features_list]
    
    by_area_and_class = units_res.groupby(['cortical_area', 'neuron_type']).mean()
    by_area_and_class = by_area_and_class.loc[by_area_and_class.index.get_level_values('neuron_type') != 'unclassified',
                                              features_list]
    
    return by_class, by_area, by_area_and_class

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
                    
        # divShuffle = np.empty((np.shape(pathDivergence)[0], np.shape(pathDivergence)[1], 100))
        # for shuffle in range(100):
        #     idx = np.random.choice(np.arange(np.shape(pathDivergence)[1]), size = np.shape(pathDivergence)[1], replace = 0)
        #     while np.sum(idx == np.arange(np.shape(pathDivergence)[1])) > 0:
        #         idx = np.random.choice(np.arange(np.shape(pathDivergence)[1]), size = np.shape(pathDivergence)[1], replace = 0)
    
        #     divShuffle[samp, :, shuffle] = np.sum(np.linalg.norm(posTraj[..., idx] - posTraj_samp, axis = 1), axis = 0)
            
        posTraj_samples.append(posTraj_samp)
        if 'velTraj_samp' in locals():
            velTraj_samples.append(velTraj_samp)
    
    if numplots is not None:
        # axlims_best  = plot_pathlet(posTraj, sample_pathlets, unit_info, unit_selector = 'max', numToPlot = 1, unitsToPlot = None)
        # axlims_worst = plot_pathlet(posTraj, sample_pathlets, unit_info, unit_selector = 'min', numToPlot = 1, unitsToPlot = None, axlims = axlims_best)
        axlims_good = plot_pathlet(posTraj_mean, posTraj_samples, lead_lag_key, model, avgPos_mean = avgPos_mean, unit_selector = 'max', numToPlot = numplots, unitsToPlot = None, axlims = None)
        _           = plot_pathlet(posTraj_mean, posTraj_samples, lead_lag_key, model, unit_selector = 'min', numToPlot = numplots, unitsToPlot = None, axlims = axlims_good)
        # axlims_good = plot_pathlet(velTraj_mean, velTraj_samples, lead_lag_key, model, unit_selector = 'max', numToPlot = 20, unitsToPlot = None, axlims = None)
        # _           = plot_pathlet(velTraj_mean, velTraj_samples, lead_lag_key, model, unit_selector = 'min', numToPlot =  5, unitsToPlot = None, axlims = axlims_good)
             
    pathDivergence_mean = np.mean(pathDivergence, axis = 0)
    # shuffledPathDivergence_mean = np.mean(np.mean(divShuffle, axis = -1), axis = 0)
    
    
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
        plt.show()
        
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
        plt.show() 
        
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
    # df = pd.DataFrame(data = zip(pos_corr.flatten(),
    #                               electrode_distances.flatten(),
    #                               FN.flatten(),
    #                               connect.flatten(),
    #                               x1.flatten(),
    #                               x2.flatten(),
    #                               y1.flatten(),
    #                               y2.flatten(),
    #                               both_reach_FN_dep.flatten(),
    #                               target_reach_FN_dep.flatten(),
    #                               source_reach_FN_dep.flatten()),
    #                   columns = ['Pearson_corr', 'Distance', 'Wji', 'Connection', 'x1', 'x2', 'y1', 'y2', 
    #                              'both_reach_FN_dependent', 'target_reach_FN_dependent', 'source_reach_FN_dependent'])

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
    
    # df = df.loc[~np.isnan(df['Distance']), :]
    # df.sort_values(by='Pearson_corr', ascending=False, inplace=True)
    # df['rank'] = np.arange(df.shape[0]+1, 1, -1) / 2
    # df.sort_index(inplace=True)
    
    # nbin = 15
    # bins = np.quantile(df['Distance'], np.linspace(0, 1,nbin+1))[:-1]
    # df['bin'], bins = pd.qcut(df['Distance'], nbin, labels=False, retbins = True)
    # bin_centers = np.convolve(bins, np.ones(2), 'valid') / 2
    # df['dist_bin_center'] = np.round(bin_centers[df['bin'].to_numpy(dtype=np.int8)], 0)
    
    # dist_counts = corr_df['dist_bin_center'].value_counts().sort_index()

    fig, ax = plt.subplots(figsize = plot_params.scatter_figsize, dpi = plot_params.dpi)
    sns.scatterplot(ax = ax, data = df, x = 'Pearson_corr', y = 'Wji', s = plot_params.feature_corr_markersize, 
                    color=plot_params.corr_marker_color, legend=False) 
    ax.set_xlabel('Preferred Trajectory\nCorrelation', fontsize=plot_params.axis_fontsize)
    ax.set_ylabel('$W_{{ji}}$', fontsize=plot_params.axis_fontsize)  
    # ax.tick_params(width=plot_params.tick_width, length = plot_params.tick_length*2, labelsize = plot_params.tick_fontsize)
    sns.despine(ax=ax)
    plt.show()
    fig.savefig(os.path.join(plots, paperFig, f'{marmcode}_wji_vs_pearson_r.png'), bbox_inches='tight', dpi=plot_params.dpi)
    
    # fig, ax = plt.subplots(figsize = (4, 4), dpi = plot_params.dpi)
    # sns.scatterplot(ax = ax, data = df, x = 'r_squared', y = 'Wji', s = 20, legend=True) 
    # plt.show()
    # fig.savefig(os.path.join(plots, paperFig 'wji_vs_pearson_rsquare.png'), bbox_inches='tight', dpi=plot_params.dpi)
    # # fig, ax = plt.subplots(figsize = (4, 4), dpi = plot_params.dpi)
    # # sns.scatterplot(ax = ax, data = df, x = 'VelTraj_corr', y = 'Wji', s = 20, legend=True) 
    # # plt.show()
    # # fig, ax = plt.subplots(figsize = (4, 4), dpi = plot_params.dpi)
    # # sns.pointplot(ax=ax, data = df, x = 'connect', y = 'r_squared', color='black', errorbar=('ci', 99))
    # # plt.show()

    # # fig, ax = plt.subplots(figsize = (4, 4), dpi = plot_params.dpi)
    # # sns.pointplot(ax=ax, data = df, x = 'dist_bin_center', y = 'Pearson_Corr', color='black', errorbar=('ci', 99))
    # # plt.show()
    # fig, ax = plt.subplots(figsize = (4, 4), dpi = plot_params.dpi)
    # sns.pointplot(ax=ax, data = df, x = 'Connection', y = 'Pearson_corr', color='black', errorbar=('ci', 99))
    # plt.show()
    # fig.savefig(os.path.join(plots, paperFig 'pearson_r_vs_connection.png'), bbox_inches='tight', dpi=plot_params.dpi)

    # fig, ax = plt.subplots(figsize = (4, 4), dpi = plot_params.dpi)
    # sns.pointplot(ax=ax, data = df, x = 'Connection', y = 'r_squared', color='black', errorbar='se')
    # plt.show()
    # fig.savefig(os.path.join(plots, paperFig 'pearson_rsquare_vs_connection.png'), bbox_inches='tight', dpi=plot_params.dpi)

    # fig, ax = plt.subplots(figsize = (4, 4), dpi = plot_params.dpi)
    # sns.pointplot(ax=ax, data = df, x = 'Connection', y = 'Wji', color='black', errorbar='se')
    # plt.show()
    # fig.savefig(os.path.join(plots, paperFig 'wji_vs_connection.png'), bbox_inches='tight', dpi=plot_params.dpi)
    
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
    plt.show()
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
    plt.show()
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

    fig, ax = plt.subplots()
    sns.violinplot(data=diff_df, ax=ax, x='Units Subset', y='w_diff', hue='Reach Set',
                order = hue_order_FN, split=True, scale='area', palette=palette)
    plt.show()
    fig.savefig(os.path.join(plots, paperFig, f'{marmcode}_w_diff_violinplot_reach_minus_spont'), bbox_inches='tight', dpi=plot_params.dpi)
    
    #------------------------------------------------------------------------------------
    
    # sns.kdeplot(data=diff_df[diff_df['Reach Set'] == 'reachFN2'], ax=axs[1], x='w_diff', 
    #             hue='Units Subset', hue_order = ['Full', 'Reach-Specific', 'Non-Specific'], 
    #             cumulative=True, common_norm=False, bw_adjust=0.05)
    
    # sns.displot(data=diff_df, x='w_diff', hue = 'Units Subset', col='Reach Set',
    #             hue_order = ['Full', 'Reach-Specific', 'Non-Specific'], kind='kde', common_norm=False)
    

    
    # sns.swarmplot(data=diff_df, ax=ax, x='Units Subset', y='w_diff', hue='Reach Set',
    #             order = ['Full', 'Reach-Specific', 'Non-Specific'])
    
    # sns.catplot(data=diff_df, x='Units Subset', y='w_diff', col='Reach Set',
    #             order = ['Full', 'Reach-Specific', 'Non-Specific'], kind='violin')            
 
def plot_supplemental_distribution_plots(units_res):
    
    fig, ax = plt.subplots(figsize=plot_params.distplot_figsize, dpi = plot_params.dpi)
    sns.kdeplot(ax=ax, data=units_res.loc[units_res['neuron_type'] != 'unclassified'], 
                x='fr', hue='neuron_type', bw_adjust=.4, common_norm=False, linewidth=plot_params.distplot_linewidth)
    ax.legend(bbox_to_anchor=(1.1, 0.5), loc='center left', 
              handles = ax.lines[::-1],
              labels  = ['Inh', 'Exc', 'mua'], 
              title_fontsize = plot_params.axis_fontsize,
              fontsize = plot_params.tick_fontsize)
    plt.show()
    fig.savefig(os.path.join(plots, 'FigS5', f'{marmcode}_fr_kdeplot_split_by_class'), bbox_inches='tight', dpi=plot_params.dpi)

    fig, ax = plt.subplots(figsize=plot_params.distplot_figsize, dpi = plot_params.dpi)
    sns.kdeplot(ax=ax, data=units_res, x='fr', hue='cortical_area', bw_adjust=.4, common_norm=False, linewidth=plot_params.distplot_linewidth)
    ax.legend(bbox_to_anchor=(1.1, 0.5), loc='center left', 
              handles = ax.lines[::-1],
              labels  = [lab for lab in units_res.cortical_area.unique()], 
              title_fontsize = plot_params.axis_fontsize,
              fontsize = plot_params.tick_fontsize)
    plt.show()
    fig.savefig(os.path.join(plots, 'FigS5', f'{marmcode}_fr_kdeplot_split_by_area'), bbox_inches='tight', dpi=plot_params.dpi)

    fig, ax = plt.subplots(figsize=plot_params.distplot_figsize, dpi = plot_params.dpi)
    sns.kdeplot(ax=ax, data=units_res, x='snr', hue='cortical_area', bw_adjust=.4, common_norm=False, linewidth=plot_params.distplot_linewidth)
    ax.legend(bbox_to_anchor=(1.1, 0.5), loc='center left', 
              handles = ax.lines[::-1],
              labels  = [lab for lab in units_res.cortical_area.unique()], 
              title_fontsize = plot_params.axis_fontsize,
              fontsize = plot_params.tick_fontsize)
    plt.show()
    fig.savefig(os.path.join(plots, 'FigS5', f'{marmcode}_snr_kdeplot_split_by_area'), bbox_inches='tight', dpi=plot_params.dpi)

    fig, ax = plt.subplots(figsize=plot_params.distplot_figsize, dpi = plot_params.dpi)
    sns.kdeplot(ax=ax, data=units_res, x='traj_avgPos_reach_FN_auc', hue='neuron_type', bw_adjust=.4, common_norm=True, linewidth=plot_params.distplot_linewidth)
    ax.legend(bbox_to_anchor=(1.1, 0.5), loc='center left', 
              handles = ax.lines[::-1],
              labels  = ['Inh', 'Exc', 'mua'], 
              title_fontsize = plot_params.axis_fontsize,
              fontsize = plot_params.tick_fontsize)
    plt.show()
    fig.savefig(os.path.join(plots, 'FigS5', f'{marmcode}_traj_avgPos_reach_FN_auc_kdeplot_split_by_class'), bbox_inches='tight', dpi=plot_params.dpi)

    fig, ax = plt.subplots(figsize=plot_params.distplot_figsize, dpi = plot_params.dpi)
    sns.kdeplot(ax=ax, data=units_res, x='traj_avgPos_reach_FN_auc', hue='cortical_area', bw_adjust=.4, 
                common_norm=False, linewidth=plot_params.distplot_linewidth)
    ax.legend(bbox_to_anchor=(1.1, 0.5), loc='center left', 
              handles = ax.lines[::-1],
              labels  = [lab for lab in units_res.cortical_area.unique()], 
              title_fontsize = plot_params.axis_fontsize,
              fontsize = plot_params.tick_fontsize)
    plt.show()
    fig.savefig(os.path.join(plots, 'FigS5', f'{marmcode}_traj_avgPos_reach_FN_auc_kdeplot_split_by_area'), bbox_inches='tight', dpi=plot_params.dpi)


    fig, ax = plt.subplots(figsize=plot_params.distplot_figsize, dpi = plot_params.dpi)
    sns.kdeplot(ax=ax, data=units_res.loc[units_res['neuron_type'] != 'unclassified'], x='snr', 
                hue='neuron_type', bw_adjust=.4, common_norm=True, linewidth=plot_params.distplot_linewidth)
    ax.legend(bbox_to_anchor=(1.1, 0.5), loc='center left', 
              handles = ax.lines[::-1],
              labels  = ['Inh', 'Exc', 'mua'], 
              title_fontsize = plot_params.axis_fontsize,
              fontsize = plot_params.tick_fontsize)
    plt.show()
    fig.savefig(os.path.join(plots, 'FigS5', f'{marmcode}_snr_kdeplot_split_by_class'), bbox_inches='tight', dpi=plot_params.dpi)
    
 
if __name__ == "__main__":
    
    os.makedirs(plots, exist_ok=True)
    
    with open(pkl_infile, 'rb') as f:
        results_dict = dill.load(f)  
        
    with NWBHDF5IO(nwb_infile, 'r') as io:
        nwb = io.read()

        FN = nwb.scratch[params.FN_key].data[:] 
        spontaneous_FN = nwb.scratch['spontaneous_FN'].data[:]

    summarize_model_results(units=None, lead_lag_keys = params.best_lead_lag_key)
    
    units_res = results_dict[params.best_lead_lag_key]['all_models_summary_results']
    units_res = add_in_weight_to_units_df(units_res, FN.copy())
    
    units_res = add_modulation_data_to_units_df(units_res)
    
    if marmcode == 'TY':
        units_res = add_neuron_classifications(units_res)
        units_res = add_icms_results_to_units_results_df(units_res, params.icms_res)
    
        by_class, by_area, by_class_and_area = get_grouped_means(units_res)
    
    electrode_distances = get_interelectrode_distances_by_unit(units_res, array_type='utah')

    train_auc_df = evaluate_effect_of_network_shuffles(lead_lag_key = params.best_lead_lag_key, 
                                                       comparison_model = 'traj_avgPos_reach_FN', 
                                                       kin_only_model=params.kin_only_model, all_samples=False, 
                                                       targets=None, ylim=(0,33), paperFig = 'Fig4', 
                                                       plot_difference = False, alpha=0.01, palette='ylgn_dark')
   
    if marmcode == 'TY':
        plot_supplemental_distribution_plots(units_res)

    cortical_area_idxs = grab_cortical_area_FN_idxs(units_res)
    # target_idxs = cortical_area_idxs['motor']
    # source_idxs = cortical_area_idxs['motor']
    target_idxs = None
    source_idxs = None

    percent = 25
    shuffle_set_0 = identify_shuffle_set_by_strength(FN[0], percent, target_idxs = target_idxs, source_idxs = source_idxs)
    shuffle_set_1 = identify_shuffle_set_by_strength(FN[1], percent, target_idxs = target_idxs, source_idxs = source_idxs)
    
    shuffle_sets = [shuffle_set_0, shuffle_set_1]
    if marmcode == 'TY':
        source_props = parse_properties_of_shuffled_sources(units_res, percent, shuffle_sets, electrode_distances, source_props = None)

    # for area in [None, 'motor', '3b', '3a']:
    # for area in [None, 'motor']:
    #     evaluate_effect_of_network_shuffles(lead_lag_key = params.best_lead_lag_key, comparison_model = 'traj_avgPos_inter_reach_FN', all_samples=False, targets=area, ylim=(0,30))
    # for area in [None, 'motor', '3b', '3a']:
    # for area in [None, 'motor']:
    #     evaluate_effect_of_network_shuffles(lead_lag_key = params.best_lead_lag_key, comparison_model = 'traj_avgPos_intra_reach_FN', all_samples=False, targets=area, ylim=(0,30))

    # # for area in [None, 'motor', '3b', '3a']:    
    # for area in [None, 'motor']:
    #     evaluate_effect_of_network_shuffles(lead_lag_key = params.best_lead_lag_key, comparison_model = 'traj_avgPos_reach_FN', all_samples=False, targets=area, ylim=(0,30))
    # for area in [None, 'motor', '3b', '3a']:
    #     evaluate_effect_of_network_shuffles(lead_lag_key = params.best_lead_lag_key, comparison_model = 'traj_avgPos_spontaneous_FN', all_samples=False, targets=area, ylim=(0,30))
    # for area in [None, 'motor', '3b', '3a']:
    #     evaluate_effect_of_network_shuffles(lead_lag_key = params.best_lead_lag_key, comparison_model = 'reach_FN', all_samples=False, targets=area, ylim=(0,200))
    # # for area in [None, 'motor', '3b', '3a']:
    # #     evaluate_effect_of_network_shuffles(lead_lag_key = params.best_lead_lag_key, comparison_model = 'spontaneous_FN', all_samples=False, targets=area, ylim=(0,200))
    # for area in [None, 'motor', '3b', '3a']:
    #     evaluate_effect_of_network_shuffles(lead_lag_key = params.best_lead_lag_key, comparison_model = 'inter_reach_FN', all_samples=False, targets=area, ylim=(0,200))
    # for area in [None, 'motor', '3b', '3a']:
    #     evaluate_effect_of_network_shuffles(lead_lag_key = params.best_lead_lag_key, comparison_model = 'intra_reach_FN', all_samples=False, targets=area, ylim=(0,200))
    
    _, cmin, cmax = plot_functional_networks(FN, units_res, FN_key = params.FN_key, paperFig='Fig1')
    _, _, _       = plot_functional_networks(spontaneous_FN, units_res, FN_key ='spontaneous_FN', paperFig='Fig1', cmin=cmin, cmax=cmax)
    # _, cmin, cmax =plot_functional_networks(spontaneous_FN, units_res, FN_key ='spontaneous_FN')
    # _, cmin, cmax = plot_functional_networks(FN, units_res, FN_key = params.FN_key, cmin=cmin, cmax=cmax)


    ymin, ymax = plot_weights_versus_interelectrode_distances(FN, spontaneous_FN, 
                                                              electrode_distances, paperFig='Fig1',
                                                              palette='FN_palette')

    if marmcode == 'TY':
        # style_key = 'neuron_type'
        style_key = None
    else:
        style_key = None
    
    model_list_x = [       'traj',    'position',          'traj_avgPos', 'traj_avgPos_spont_train_reach_test_FN']
    model_list_y = ['traj_avgPos', 'traj_avgPos', 'traj_avgPos_reach_FN',                  'traj_avgPos_reach_FN']
    fig_list     = [       'Fig2',       'FigS1',                 'Fig3',                               'unknown']
    for model_x, model_y, fignum in zip(model_list_x, model_list_y, fig_list):    
        sign_test = plot_model_auc_comparison(units_res, model_x, model_y, 
                                              minauc = 0.45, maxauc = 0.8, hue_key='W_in', style_key=style_key, 
                                              targets = None, col_key = None, paperFig=fignum, asterisk='')
        print(f'{model_y} v {model_x}, NO_tuning_filter: p={np.round(sign_test.pvalue, 4)}, nY={sign_test.k}, nUnits={sign_test.n}, prop={np.round(sign_test.proportion_estimate, 2)}')
    
    units_res_completely_untuned_units_filtered = units_res.loc[units_res['proportion_sign']>=0.5, :]
    model_list_x = ['shortTraj', 'shortTraj_avgPos', 'shortTraj_low_tortuosity', 'shortTraj_avgPos_low_tortuosity', 'shortTraj_high_tortuosity', 'shortTraj_avgPos_high_tortuosity']
    model_list_y = [     'traj',      'traj_avgPos',      'traj_low_tortuosity',      'traj_avgPos_low_tortuosity',      'traj_high_tortuosity',      'traj_avgPos_high_tortuosity']
    fig_list     = [     'Fig2',             'Fig2',                    'FigS1',                           'FigS1',                     'FigS1',                            'FigS1']
    for model_x, model_y, fignum in zip(model_list_x, model_list_y, fig_list):    
        sign_test = plot_model_auc_comparison(units_res_completely_untuned_units_filtered, model_x, model_y, 
                                              minauc = 0.45, maxauc = 0.8, hue_key='W_in', style_key=style_key, 
                                              targets = None, col_key = None, paperFig=fignum, asterisk='*')
        print(f'{model_y} v {model_x}, YES_tuning_filter: p={np.round(sign_test.pvalue, 4)}, nY={sign_test.k}, nUnits={sign_test.n}, prop={np.round(sign_test.proportion_estimate, 2)}')
    


        
    # plot_model_auc_comparison   (units_res, 'traj_avgPos_auc', 'traj_avgPos_reach_FN_auc', minauc = 0.45, hue_key='W_in', style_key=style_key, 
    #                              col_key = 'cortical_area', targets = None, paperFig='FigS3')

    # plot_model_auc_comparison   (units_res, 'traj_avgPos_auc', 'traj_avgPos_reach_FN_auc', minauc = 0.45, hue_key='W_in', style_key=style_key, 
    #                              targets = None, col_key = None, paperFig='Fig4')
    # sign_test, ttest = sig_tests(units_res, 'traj_avgPos_auc', 'traj_avgPos_reach_FN_auc', alternative='greater')
    # print(sign_test)
    # plot_model_auc_comparison   (units_res, 'reach_FN_auc', 'traj_avgPos_reach_FN_auc', minauc = 0.45, hue_key='W_in', style_key='cortical_area', targets = None)
    # sign_test, ttest = sig_tests(units_res, 'reach_FN_auc', 'traj_avgPos_reach_FN_auc', alternative='greater')
    # print(sign_test)
    # plot_model_auc_comparison   (units_res, 'reach_FN_auc', 'traj_avgPos_auc', minauc = 0.45, hue_key='W_in', style_key='cortical_area', targets = None)
    # sign_test, ttest = sig_tests(units_res, 'reach_FN_auc', 'traj_avgPos_auc', alternative='greater')
    # print(sign_test)
    # plot_model_auc_comparison   (units_res, 'traj_avgPos_spontaneous_FN_auc', 'traj_avgPos_reach_FN_auc', minauc = 0.45, hue_key='W_in', 
    #                              style_key='cortical_area', targets = None, paperFig='unknown')
    # sign_test, ttest = sig_tests(units_res, 'traj_avgPos_spontaneous_FN_auc', 'traj_avgPos_reach_FN_auc', alternative='less')
    # print(sign_test)
    # plot_model_auc_comparison   (units_res, 'spontaneous_FN_auc', 'reach_FN_auc', minauc = 0.45, hue_key='W_in', style_key='cortical_area', 
    #                              targets = None, paperFig='unknown')
    # sign_test, ttest = sig_tests(units_res, 'spontaneous_FN_auc', 'reach_FN_auc', alternative='two-sided')
    # print(sign_test)

    # plot_model_auc_comparison   (units_res, 'reach_train_spont_test_FN_auc', 'reach_FN_auc', minauc = 0.45, hue_key='W_in', style_key=style_key, targets = None)
    # plot_model_auc_comparison   (units_res, 'spont_train_reach_test_FN_auc', 'reach_FN_auc', minauc = 0.45, hue_key='W_in', style_key=style_key, 
    #                              targets = None, paperFig='unknown')
    # plot_model_auc_comparison   (units_res, 'reach_train_spont_test_FN_auc', 'spontaneous_FN_auc', minauc = 0.45, hue_key='W_in', style_key=style_key, targets = None)

    # plot_model_auc_comparison   (units_res, 'traj_avgPos_reach_train_spont_test_FN_auc', 'traj_avgPos_reach_FN_auc', minauc = 0.45, hue_key='W_in', style_key=style_key, targets = None)
    # plot_model_auc_comparison   (units_res, 'traj_avgPos_reach_train_spont_test_FN_auc', 'traj_avgPos_spontaneous_FN_auc', minauc = 0.45, hue_key='W_in', style_key=style_key, targets = None)


    # for area in [None, 'motor', 'sensory']:
    #     plot_model_auc_comparison   (units_res, 'traj_avgPos_spont_train_reach_test_FN_auc', 'traj_avgPos_spontaneous_FN_auc', minauc = 0.45,targets = area)
    #     plot_model_auc_comparison   (units_res, 'traj_avgPos_reach_train_spont_test_FN_auc', 'traj_avgPos_reach_FN_auc', minauc = 0.45,targets = area)
    #     plot_model_auc_comparison   (units_res, 'traj_avgPos_spont_train_reach_test_FN_auc', 'traj_avgPos_reach_train_spont_test_FN_auc', minauc = 0.45,targets = area)

    # plot_model_auc_comparison   (units_res, 'traj_avgPos_intra_reach_FN', 'traj_avgPos_reach_FN', minauc = 0.45, hue_key='W_in', style_key=style_key, col_key = 'cortical_area', targets = None)
    # plot_model_auc_comparison   (units_res, 'traj_avgPos_intra_reach_FN', 'traj_avgPos_inter_reach_FN', minauc = 0.45, hue_key='W_in', style_key=style_key, col_key = 'cortical_area', targets = None)
    # plot_model_auc_comparison   (units_res, 'intra_reach_FN', 'inter_reach_FN', minauc = 0.45, hue_key='W_in', style_key=style_key, col_key = 'cortical_area', targets = None)

    diff_df = compute_performance_difference_by_unit(units_res, 'traj_avgPos_spont_train_reach_test_FN_auc', 'traj_avgPos_reach_FN_auc')   
    # reach_specific_units = diff_df.index[(diff_df.auc_diff > 0) & (diff_df.dist_from_unity > params.reach_specific_thresh) & (diff_df.dist_from_unity < 0.04)]
    reach_specific_units = diff_df.index[(diff_df.auc_diff > 0) & (diff_df.dist_from_unity > params.reach_specific_thresh)]
    non_specific_units = np.setdiff1d(units_res.index, reach_specific_units)
    
    # diff_df = compute_performance_difference_by_unit(units_res, 'spont_train_reach_test_FN_auc', 'reach_FN_auc')   
    # reach_specific_units_FN_only = diff_df.index[(diff_df.auc_diff > 0) & (diff_df.dist_from_unity > 0.04)]
    
    posTraj_mean, velTraj_mean, posTraj_samples, velTraj_samples = compute_and_analyze_pathlets(params.best_lead_lag_key, 'traj_avgPos', numplots = None)
    traj_corr_df = compute_and_analyze_trajectory_correlations(units_res, posTraj_mean, velTraj_mean, 
                                                               electrode_distances, params.best_lead_lag_key, 
                                                               FN = FN, mode='concat', paperFig='Fig3',
                                                               reach_specific_units = reach_specific_units, nplots=None)
    
    
    
    # sns.kdeplot(data = traj_corr_df, x='r_squared', hue='both_reach_FN_dependent', bw_adjust=.4, common_norm=False)
    # plt.show()
    # # plt.gcf().savefig(os.path.join(plots, 'FigS5', f'{marmcode}_fr_kdeplot_split_by_class'), bbox_inches='tight', dpi=plot_params.dpi)
    # sns.kdeplot(data = traj_corr_df, x='r_squared', hue='target_reach_FN_dependent', bw_adjust=.4, common_norm=False)
    # plt.show()
    # sns.kdeplot(data = traj_corr_df, x='r_squared', hue='source_reach_FN_dependent', bw_adjust=.4, common_norm=False)
    # plt.show()
    
    subset = 'both'

    reach_specific_reachFNs, _, _ = plot_functional_networks(FN, units_res, FN_key = params.FN_key, cmin=cmin, cmax=cmax, 
                                                             subset_idxs = reach_specific_units, subset_type=subset, paperFig='Fig6')
    reach_specific_spontFN , _, _ = plot_functional_networks(spontaneous_FN, units_res, FN_key ='spontaneous_FN', cmin=cmin, cmax=cmax, 
                                                             subset_idxs = reach_specific_units, subset_type=subset, paperFig='Fig6')

    non_specific_reachFNs, _, _ = plot_functional_networks(FN, units_res, FN_key = params.FN_key, cmin=cmin, cmax=cmax, 
                                                           subset_idxs = non_specific_units, subset_type=subset, paperFig='Fig6')
    non_specific_spontFN , _, _ = plot_functional_networks(spontaneous_FN, units_res, FN_key ='spontaneous_FN', cmin=cmin, cmax=cmax, 
                                                           subset_idxs = non_specific_units, subset_type=subset, paperFig='Fig6')


    if marmcode == 'TY':    
        reach_specific_FN_props = parse_properties_of_FN_subsets(units_res, FN, electrode_distances, 
                                                                 FN_key = params.FN_key, 
                                                                 subset_idxs = reach_specific_units, 
                                                                 subset_types=['both', 'target', 'source'], 
                                                                 subset_basis = ['Reach-Specific', 'Non-Specific'], 
                                                                 tune = (f'{params.primary_traj_model}_auc', params.tuned_auc_thresh),
                                                                 source_props=None)
    
        reach_specific_FN_props = parse_properties_of_FN_subsets(units_res, spontaneous_FN, electrode_distances, 
                                                                 FN_key = 'spontaneous', 
                                                                 subset_idxs = reach_specific_units, 
                                                                 subset_types=['both', 'target', 'source'], 
                                                                 subset_basis = ['Reach-Specific', 'Non-Specific'], 
                                                                 tune = (f'{params.primary_traj_model}_auc', params.tuned_auc_thresh),
                                                                 source_props=reach_specific_FN_props)
    
    reach_specific_units_res = units_res.loc[reach_specific_units, :]
    non_specific_units_res = units_res.loc[non_specific_units, :]
    # units_res_for_distributions = units_res.loc[units_res['traj_avgPos_auc'] >= reach_specific_units_res['traj_avgPos_auc'].min(), :]    

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

    plot_FN_differences(FN, spontaneous_FN, reach_specific_reachFNs, reach_specific_spontFN, 
                        non_specific_reachFNs, non_specific_spontFN, paperFig='Fig6', 
                        hue_order_FN = ['Reach-Specific', 'Non-Specific', 'Full'], palette=fig6_palette)

    if marmcode == 'TY':    
        fig, ax = plt.subplots(figsize=plot_params.distplot_figsize, dpi = plot_params.dpi)
        sns.kdeplot(ax=ax, data=units_res, x='icms_threshold', hue='Units Subset', 
                    hue_order=('Reach-Specific', 'Non-Specific'), bw_adjust=.4, 
                    common_norm=False, palette=fig6_palette, linewidth=plot_params.distplot_linewidth)
        ax.legend(bbox_to_anchor=(1.1, 0.5), loc='center left', 
                  handles = ax.lines[::-1],
                  labels  = ['Reach-Specific', 'Non-Specific'], 
                  title_fontsize = plot_params.axis_fontsize,
                  fontsize = plot_params.tick_fontsize)
        plt.show()
        fig.savefig(os.path.join(plots, 'FigS5', f'{marmcode}_icms_threshold_kdeplot_split_by_units_subset'), bbox_inches='tight', dpi=plot_params.dpi)
        med_out = median_test(units_res.loc[units_res['Units Subset'] == 'Reach-Specific', 'icms_threshold'], 
                              units_res.loc[units_res['Units Subset'] == 'Non-Specific'  , 'icms_threshold'],
                              nan_policy='omit')
        print(f'icms_thresh: reach v non, p={np.round(med_out[1], 4)}, {med_out[-1][0,0]}a-{med_out[-1][1,0]}b, {med_out[-1][0,1]}a-{med_out[-1][1,1]}b')

        fig, ax = plt.subplots(figsize=plot_params.distplot_figsize, dpi = plot_params.dpi)
        sns.kdeplot(ax=ax, data=units_res.loc[units_res['neuron_type'] != 'unclassified', :], x='traj_avgPos_auc', hue='neuron_type', bw_adjust=.4, 
                    common_norm=False, cumulative=True, linewidth=plot_params.distplot_linewidth)
        ax.legend(bbox_to_anchor=(1.1, 0.5), loc='center left', 
                  handles = ax.lines[::-1],
                  labels  = ['NS', 'WS', 'mua'], 
                  title_fontsize = plot_params.axis_fontsize,
                  fontsize = plot_params.tick_fontsize)
        ax.set_xlabel('Full Kinematics AUC')
        ax.set_yticks([0, 1])
        plt.show()
        fig.savefig(os.path.join(plots, 'chapter4', f'{marmcode}_fullKin_AUC_kdeplot_split_by_neuron_type'), bbox_inches='tight', dpi=plot_params.dpi)
        med_out = median_test(units_res.loc[units_res['neuron_type'] == 'inh', 'traj_avgPos_auc'], 
                              units_res.loc[units_res['neuron_type'] == 'exc', 'traj_avgPos_auc'],
                              nan_policy='omit')
        print(f'neuron_type: inh v exc, p={np.round(med_out[1], 4)}, {med_out[-1][0,0]}a-{med_out[-1][1,0]}b, {med_out[-1][0,1]}a-{med_out[-1][1,1]}b')
        med_out = median_test(units_res.loc[units_res['neuron_type'] == 'inh', 'traj_avgPos_auc'], 
                              units_res.loc[units_res['neuron_type'] == 'mua', 'traj_avgPos_auc'],
                              nan_policy='omit')
        print(f'neuron_type: Exc v mua, p={np.round(med_out[1], 4)}, {med_out[-1][0,0]}a-{med_out[-1][1,0]}b, {med_out[-1][0,1]}a-{med_out[-1][1,1]}b')
        med_out = median_test(units_res.loc[units_res['neuron_type'] == 'exc', 'traj_avgPos_auc'], 
                              units_res.loc[units_res['neuron_type'] == 'mua', 'traj_avgPos_auc'],
                              nan_policy='omit')
        print(f'neuron_type: inh v mua, p={np.round(med_out[1], 4)}, {med_out[-1][0,0]}a-{med_out[-1][1,0]}b, {med_out[-1][0,1]}a-{med_out[-1][1,1]}b')

        fig, ax = plt.subplots(figsize=plot_params.distplot_figsize, dpi = plot_params.dpi)
        sns.kdeplot(ax=ax, data=units_res.loc[units_res['neuron_type'] != 'unclassified', :], x='traj_avgPos_reach_FN_auc', hue='neuron_type', bw_adjust=.4, 
                    common_norm=False, cumulative=True, linewidth=plot_params.distplot_linewidth)
        ax.legend(bbox_to_anchor=(1.1, 0.5), loc='center left', 
                  handles = ax.lines[::-1],
                  labels  = ['NS', 'WS', 'mua'], 
                  title_fontsize = plot_params.axis_fontsize,
                  fontsize = plot_params.tick_fontsize)
        ax.set_xlabel('Kinematics + ReachFN AUC')
        ax.set_yticks([0, 1])
        plt.show()
        fig.savefig(os.path.join(plots, 'chapter4', f'{marmcode}_kinFN_AUC_kdeplot_split_by_neuron_type'), bbox_inches='tight', dpi=plot_params.dpi)
        med_out = median_test(units_res.loc[units_res['neuron_type'] == 'inh', 'traj_avgPos_reach_FN_auc'], 
                              units_res.loc[units_res['neuron_type'] == 'exc', 'traj_avgPos_reach_FN_auc'],
                              nan_policy='omit')
        print(f'neuron_type: inh v exc, p={np.round(med_out[1], 4)}, {med_out[-1][0,0]}a-{med_out[-1][1,0]}b, {med_out[-1][0,1]}a-{med_out[-1][1,1]}b')
        med_out = median_test(units_res.loc[units_res['neuron_type'] == 'inh', 'traj_avgPos_reach_FN_auc'], 
                              units_res.loc[units_res['neuron_type'] == 'mua', 'traj_avgPos_reach_FN_auc'],
                              nan_policy='omit')
        print(f'neuron_type: Exc v mua, p={np.round(med_out[1], 4)}, {med_out[-1][0,0]}a-{med_out[-1][1,0]}b, {med_out[-1][0,1]}a-{med_out[-1][1,1]}b')
        med_out = median_test(units_res.loc[units_res['neuron_type'] == 'exc', 'traj_avgPos_reach_FN_auc'], 
                              units_res.loc[units_res['neuron_type'] == 'mua', 'traj_avgPos_reach_FN_auc'],
                              nan_policy='omit')
        print(f'neuron_type: inh v mua, p={np.round(med_out[1], 4)}, {med_out[-1][0,0]}a-{med_out[-1][1,0]}b, {med_out[-1][0,1]}a-{med_out[-1][1,1]}b')


        fig, ax = plt.subplots(figsize=plot_params.distplot_figsize, dpi = plot_params.dpi)
        sns.kdeplot(ax=ax, data=units_res.loc[units_res['neuron_type'] != 'unclassified', :], x='snr', hue='neuron_type', bw_adjust=.4, 
                    common_norm=False, cumulative=True, linewidth=plot_params.distplot_linewidth)
        ax.legend(bbox_to_anchor=(1.1, 0.5), loc='center left', 
                  handles = ax.lines[::-1],
                  labels  = ['NS', 'WS', 'mua'], 
                  title_fontsize = plot_params.axis_fontsize,
                  fontsize = plot_params.tick_fontsize)
        ax.set_xlabel('Signal-to-Noise Ratio')
        ax.set_yticks([0, 1])
        plt.show()
        fig.savefig(os.path.join(plots, 'chapter4', f'{marmcode}_snr_kdeplot_split_by_neuron_type'), bbox_inches='tight', dpi=plot_params.dpi)
        med_out = median_test(units_res.loc[units_res['neuron_type'] == 'inh', 'snr'], 
                              units_res.loc[units_res['neuron_type'] == 'exc', 'snr'],
                              nan_policy='omit')
        print(f'neuron_type: inh v exc, p={np.round(med_out[1], 4)}, {med_out[-1][0,0]}a-{med_out[-1][1,0]}b, {med_out[-1][0,1]}a-{med_out[-1][1,1]}b')
        med_out = median_test(units_res.loc[units_res['neuron_type'] == 'inh', 'snr'], 
                              units_res.loc[units_res['neuron_type'] == 'mua', 'snr'],
                              nan_policy='omit')
        print(f'neuron_type: Exc v mua, p={np.round(med_out[1], 4)}, {med_out[-1][0,0]}a-{med_out[-1][1,0]}b, {med_out[-1][0,1]}a-{med_out[-1][1,1]}b')
        med_out = median_test(units_res.loc[units_res['neuron_type'] == 'exc', 'snr'], 
                              units_res.loc[units_res['neuron_type'] == 'mua', 'snr'],
                              nan_policy='omit')
        print(f'neuron_type: inh v mua, p={np.round(med_out[1], 4)}, {med_out[-1][0,0]}a-{med_out[-1][1,0]}b, {med_out[-1][0,1]}a-{med_out[-1][1,1]}b')
                                      

    plot_model_auc_comparison   (units_res, 'traj_avgPos_auc', 'traj_avgPos_reach_FN_auc', 
                                 minauc = 0.45, maxauc = 0.8, hue_key='Units Subset', hue_order=('Reach-Specific', 'Non-Specific'), 
                                 style_key=style_key, targets = None, paperFig='FigS4', palette=fig6_palette)
    plot_model_auc_comparison   (units_res, 'spontaneous_FN_auc', 'reach_FN_auc', 
                                 minauc = 0.45, maxauc = 0.8, hue_key='Units Subset', hue_order=('Reach-Specific', 'Non-Specific'), 
                                 style_key=style_key, targets = None, paperFig='unknown', palette=fig6_palette)
    plot_model_auc_comparison   (units_res, 'traj_avgPos_spont_train_reach_test_FN_auc', 'traj_avgPos_reach_FN_auc', 
                                 minauc = 0.45, maxauc = 0.8, hue_key='Units Subset', hue_order=('Reach-Specific', 'Non-Specific'), 
                                 style_key=style_key, targets = None, paperFig='Fig6', palette=fig6_palette)

    plot_model_auc_comparison   (units_res, 'shortTraj_avgPos', 'traj_avgPos', minauc = 0.45, maxauc = 0.8, hue_key='Units Subset', 
                                 hue_order=('Reach-Specific', 'Non-Specific'), style_key=style_key, targets = None, paperFig='FigS4', palette=fig6_palette)
    plot_model_auc_comparison   (units_res, 'shortTraj', 'traj', minauc = 0.45, maxauc = 0.8, hue_key='Units Subset', 
                                 hue_order=('Reach-Specific', 'Non-Specific'), style_key=style_key, targets = None, paperFig='FigS4', palette=fig6_palette)

    
    if marmcode == 'TY':
        feature_correlation_plot(units_res, 'W_in_3a', 'traj_avgPos_auc', col_key=None)    
        feature_correlation_plot(units_res, 'W_in_3b', 'traj_avgPos_auc', col_key=None)    
        feature_correlation_plot(units_res, 'W_in_m1', 'traj_avgPos_auc', col_key=None)    
        feature_correlation_plot(units_res, 'W_in_3a', 'traj_avgPos_auc', col_key='cortical_area')
        feature_correlation_plot(units_res, 'W_in_3b', 'traj_avgPos_auc', col_key='cortical_area')
        feature_correlation_plot(units_res, 'W_in_m1', 'traj_avgPos_auc', col_key='cortical_area')
        feature_correlation_plot(units_res, 'W_in', 'traj_avgPos_auc', col_key='cortical_area')
    
    feature_correlation_plot(units_res, 'W_in', 'traj_avgPos_auc', paperFig='Fig3')


    feature_correlation_plot(units_res, 'snr', 'W_in', col_key=None, paperFig = 'FigS3')    
    feature_correlation_plot(units_res, 'fr', 'W_in', col_key=None, paperFig = 'FigS3')    
    feature_correlation_plot(units_res, 'snr', 'traj_avgPos_reach_FN_auc', col_key=None, paperFig = 'FigS3')    
    feature_correlation_plot(units_res, 'fr', 'traj_avgPos_reach_FN_auc', col_key=None, paperFig = 'FigS3')    
    feature_correlation_plot(units_res, 'fr', 'traj_avgPos_auc', col_key=None, paperFig = 'FigS3')    
    feature_correlation_plot(units_res, 'snr', 'traj_avgPos_auc', col_key=None, paperFig = 'FigS3')    

    feature_correlation_plot(units_res.loc[units_res['quality'] == 'good'], 'fr', 'traj_avgPos_auc', col_key=None)

    feature_correlation_plot(units_res.loc[units_res['quality'] == 'good'], 'fr', 'traj_avgPos_auc', col_key='cortical_area')
    feature_correlation_plot(units_res.loc[units_res['quality'] == 'good'], 'fr', 'traj_avgPos_reach_FN_auc', col_key='cortical_area')
    
    feature_correlation_plot(weights_df.loc[(weights_df['FN_key']=='reachFN1'), :], 'avg_auc', 'pearson_r_squared', col_key=None, paperFig = 'unknown')    
    
    units_res['FN_auc_improvement']     = units_res['traj_avgPos_reach_FN_auc'] - units_res['traj_avgPos_auc']
    units_res['FN_percent_improvement'] = (units_res['traj_avgPos_reach_FN_auc'] - units_res['traj_avgPos_auc']) / units_res['traj_avgPos_auc']

    feature_correlation_plot(units_res, 'W_in', 'FN_auc_improvement', paperFig='unknown')
    feature_correlation_plot(units_res, 'traj_avgPos_auc', 'FN_auc_improvement', paperFig='unknown')

    feature_correlation_plot(units_res, 'W_in', 'FN_percent_improvement', paperFig='unknown')
    feature_correlation_plot(units_res, 'traj_avgPos_auc', 'FN_percent_improvement', paperFig='unknown')
    
    
    # stats for figure 2 summary plot
    model_list_x = [  'shortTraj',        'traj', 'shortTraj_avgPos', 'traj_avgPos_shuffled_spikes', 'traj_avgPos_shuffled_traj']
    model_list_y = ['traj_avgPos', 'traj_avgPos',      'traj_avgPos',                 'traj_avgPos',               'traj_avgPos']
    fig_list     = [    'unknown',     'unknown',          'unknown',                     'unknown',                   'unknown']
    for model_x, model_y, fignum in zip(model_list_x, model_list_y, fig_list):    
        sign_test = plot_model_auc_comparison(units_res_completely_untuned_units_filtered, model_x, model_y, 
                                              minauc = 0.45, maxauc = 0.8, hue_key='W_in', style_key=style_key, 
                                              targets = None, col_key = None, paperFig=fignum, asterisk='')
        print(f'{model_y} v {model_x}, YES_tuning_filter: p={np.round(sign_test.pvalue, 4)}, nY={sign_test.k}, nUnits={sign_test.n}, prop={np.round(sign_test.proportion_estimate, 2)}')

    # dissertation Chapter 4 Fig 1
    model_list_x = ['traj_avgPos', 'traj_avgPos_spont_train_reach_test_FN']   
    model_list_y = ['traj_avgPos_reach_FN', 'traj_avgPos_reach_FN']
    fig_list     = ['chapter4', 'chapter4']
    for model_x, model_y, fignum in zip(model_list_x, model_list_y, fig_list):    
        sign_test = plot_model_auc_comparison(units_res, model_x, model_y, 
                                              minauc = 0.45, maxauc = 0.8, hue_key='neuron_type', style_key='neuron_type', 
                                              targets = None, col_key = None, paperFig=fignum, asterisk='')
        print(f'{model_y} v {model_x}, NO_tuning_filter: p={np.round(sign_test.pvalue, 4)}, nY={sign_test.k}, nUnits={sign_test.n}, prop={np.round(sign_test.proportion_estimate, 2)}')
        
    
    
    # for area in [None, 'motor', '3b', '3a']:
    #     plot_model_auc_comparison(units_res, 'traj_avgPos_intra_reach_FN', 'traj_avgPos_reach_FN', targets = area)
    #     plot_model_auc_comparison(units_res, 'traj_avgPos_intra_reach_FN', 'traj_avgPos_inter_reach_FN', targets = area)

    # plot_model_auc_comparison   (units_res, 'traj_avgPos_shuffled_topology_FN_55_percent_by_strength_train_auc', 'traj_avgPos_shuffled_weights_FN_55_percent_by_random_train_auc', minauc = 0.5,targets = None)
    
    # # evaluate_lead_lag_by_model_coefficients(lead_lag_key = 'lead_200_lag_300', kin_type = 'traj_avgPos_full_FN', mode='average', proportion_thresh=0.99)
    
    # # evaluate_effect_of_network_shuffles(lead_lag_key = params.best_lead_lag_key, comparison_model = 'reach_FN', all_samples=False)
    # # evaluate_effect_of_network_shuffles(lead_lag_key = params.best_lead_lag_key, comparison_model = 'spontaneous_FN', all_samples=False)

    # # evaluate_effect_of_network_shuffles(lead_lag_key = params.best_lead_lag_key, comparison_model = '%s_reach_FN' % params.primary_traj_model, all_samples=False)
    # # plot_model_auc_comparison(units_res, 'traj_and_avgPos_auc', 'traj_and_avgPos_full_FN_auc', targets = 'tuned')
    # # plot_model_auc_comparison(units_res, 'traj_and_avgPos_auc', 'traj_and_avgPos_full_FN_auc', targets = 'untuned')
    # # plot_model_auc_comparison(units_res, 'traj_and_avgPos_auc', 'traj_and_avgPos_full_FN_auc', targets = 'motor')
    # # plot_model_auc_comparison(units_res, 'traj_and_avgPos_auc', 'traj_and_avgPos_full_FN_auc', targets = 'sensory')

    # # plot_model_auc_comparison(units_res, 'traj_and_avgPos_auc', 'full_FN_auc', targets = None)
    # # plot_model_auc_comparison(units_res, 'traj_and_avgPos_auc', 'full_FN_auc', targets = 'tuned')
    # # plot_model_auc_comparison(units_res, 'traj_and_avgPos_auc', 'full_FN_auc', targets = 'untuned')
    # # plot_model_auc_comparison(units_res, 'traj_and_avgPos_auc', 'full_FN_auc', targets = 'motor')
    # # plot_model_auc_comparison(units_res, 'traj_and_avgPos_auc', 'full_FN_auc', targets = 'sensory')

    # # # plot nonzero vs zero distance connections    
    # # units_res_pruned = prune_for_neurons_with_same_channel_connections(units_res)
    # # plot_model_auc_comparison(units_res_pruned, x_key = 'traj_and_avgPos_zero_dist_FN', y_key = 'traj_and_avgPos_full_FN', targets=None)
    # # plot_model_auc_comparison(units_res_pruned, x_key = 'traj_and_avgPos_nonzero_dist_FN', y_key = 'traj_and_avgPos_full_FN', targets=None)
    # # x_keys = ['traj_and_avgPos_zero_dist_FN', 'traj_and_avgPos_nonzero_dist_FN', 'traj_and_avgPos_zero_dist_FN', 'traj_and_avgPos_nonzero_dist_FN']
    # # y_keys = ['traj_and_avgPos_full_FN', 'traj_and_avgPos_full_FN', 'traj_and_avgPos_full_FN', 'traj_and_avgPos_full_FN']
    # # targets_list = ['tuned', 'tuned', 'untuned', 'untuned']
    # # for x_key, y_key, targets in zip(x_keys, y_keys, targets_list):
    # #     plot_model_auc_comparison(units_res_pruned, x_key = x_key, y_key = y_key, targets=targets)  

    # # plot_model_training_performance_comparison(units_res_pruned, x_key = 'traj_and_avgPos_zero_dist_FN', y_key = 'traj_and_avgPos_full_FN', lead_lag_key=params.best_lead_lag_key, metric='trainAUC', targets=None)
    # # plot_model_training_performance_comparison(units_res_pruned, x_key = 'traj_and_avgPos_nonzero_dist_FN', y_key = 'traj_and_avgPos_full_FN', lead_lag_key=params.best_lead_lag_key, metric='trainAUC', targets=None)
    # # x_keys = ['traj_and_avgPos_zero_dist_FN', 'traj_and_avgPos_nonzero_dist_FN', 'traj_and_avgPos_zero_dist_FN', 'traj_and_avgPos_nonzero_dist_FN']
    # # y_keys = ['traj_and_avgPos_full_FN', 'traj_and_avgPos_full_FN', 'traj_and_avgPos_full_FN', 'traj_and_avgPos_full_FN']
    # # targets_list = ['tuned', 'tuned', 'untuned', 'untuned']
    # # for x_key, y_key, targets in zip(x_keys, y_keys, targets_list):
    # #     plot_model_training_performance_comparison(units_res_pruned, x_key = x_key, y_key = y_key, 
    # #                                                lead_lag_key=params.best_lead_lag_key, 
    # #                                                metric='trainAUC', targets=targets) 

    
    # # # results of tuned vs untuned inputs to tuned vs untuned targets
    # # x_keys = ['traj_and_avgPos_tuned_inputs_FN', 'traj_and_avgPos_untuned_inputs_FN', 'traj_and_avgPos_tuned_inputs_FN', 'traj_and_avgPos_untuned_inputs_FN']
    # # y_keys = ['traj_and_avgPos_full_FN', 'traj_and_avgPos_full_FN', 'traj_and_avgPos_full_FN', 'traj_and_avgPos_full_FN']
    # # targets_list = ['tuned', 'tuned', 'untuned', 'untuned']
    # # for x_key, y_key, targets in zip(x_keys, y_keys, targets_list):
    # #     plot_model_training_performance_comparison(units_res, x_key = x_key, y_key = y_key, 
    # #                                                lead_lag_key=params.best_lead_lag_key, 
    # #                                                metric='trainAUC', targets=targets)
        
    # # # results of tuned vs untuned inputs to motor vs sensory targets
    # # x_keys = ['traj_and_avgPos_tuned_inputs_FN', 'traj_and_avgPos_untuned_inputs_FN', 'traj_and_avgPos_tuned_inputs_FN', 'traj_and_avgPos_untuned_inputs_FN']
    # # y_keys = ['traj_and_avgPos_full_FN', 'traj_and_avgPos_full_FN', 'traj_and_avgPos_full_FN', 'traj_and_avgPos_full_FN']
    # # targets_list = ['motor', 'motor', 'sensory', 'sensory']
    # # for x_key, y_key, targets in zip(x_keys, y_keys, targets_list):
    # #     plot_model_training_performance_comparison(units_res, x_key = x_key, y_key = y_key, 
    # #                                                lead_lag_key=params.best_lead_lag_key, 
    # #                                                metric='trainAUC', targets=targets)
        
    # # # results of motor, sensory inputs to motor vs sensory targets
    # # x_keys = ['traj_and_avgPos_permuted_motor_inputs_FN', 'traj_and_avgPos_permuted_sensory_inputs_FN', 'traj_and_avgPos_permuted_motor_inputs_FN', 'traj_and_avgPos_permuted_sensory_inputs_FN']
    # # y_keys = ['traj_and_avgPos_full_FN', 'traj_and_avgPos_full_FN', 'traj_and_avgPos_full_FN', 'traj_and_avgPos_full_FN']
    # # targets_list = ['motor', 'motor', 'sensory', 'sensory']
    # # for x_key, y_key, targets in zip(x_keys, y_keys, targets_list):
    # #     plot_model_training_performance_comparison(units_res, x_key = x_key, y_key = y_key, 
    # #                                                lead_lag_key=params.best_lead_lag_key, 
    # #                                                metric='trainAUC', targets=targets)        

    # # # results of motor, sensory inputs to motor vs sensory targets (tuned targets only)
    # # x_keys = ['traj_and_avgPos_permuted_motor_inputs_FN', 'traj_and_avgPos_permuted_sensory_inputs_FN', 'traj_and_avgPos_permuted_motor_inputs_FN', 'traj_and_avgPos_permuted_sensory_inputs_FN']
    # # y_keys = ['traj_and_avgPos_full_FN', 'traj_and_avgPos_full_FN', 'traj_and_avgPos_full_FN', 'traj_and_avgPos_full_FN']
    # # targets_list = [['motor', 'tuned'], ['motor', 'tuned'], ['sensory', 'tuned'], ['sensory', 'tuned']]
    # # for x_key, y_key, targets in zip(x_keys, y_keys, targets_list):
    # #     plot_model_training_performance_comparison(units_res, x_key = x_key, y_key = y_key, 
    # #                                                lead_lag_key=params.best_lead_lag_key, 
    # #                                                metric='trainAUC', targets=targets)  
        
    # # sign_test, ttest = sig_tests(units_res, 'traj_and_avgPos_auc', 'traj_and_avgPos_full_FN_auc')
    # # sign_test, ttest = sig_tests(units_res, 'full_FN_auc', 'traj_and_avgPos_full_FN_auc')
    # # sign_test, ttest = sig_tests(units_res, 'full_FN_auc', 'traj_and_avgPos_auc')
    # # sign_test, ttest = sig_tests(units_res, 'traj_auc', 'traj_and_avgPos_auc')
    # # sign_test, ttest = sig_tests(units_res, 'short_traj_and_avgPos_auc', 'traj_and_avgPos_auc')
    # # sign_test, ttest = sig_tests(units_res, 'traj_and_avgPos_tuned_inputs_FN_auc', 'traj_and_avgPos_full_FN_auc')
    # # sign_test, ttest = sig_tests(units_res, 'traj_and_avgPos_untuned_inputs_FN_auc', 'traj_and_avgPos_full_FN_auc')



    # # with open(pkl_outfile, 'wb') as f:
    # #     dill.dump(results_dict, f, recurse=True) 