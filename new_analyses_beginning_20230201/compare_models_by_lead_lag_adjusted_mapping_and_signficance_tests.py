# -*- coding: utf-8 -*-
"""
Created on Tue May 31 10:09:38 2022

@author: Dalton
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import dill
import os
import glob
import math
import re
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc
import statsmodels.api as sm
from scipy.stats import binomtest, ttest_rel, ttest_ind, mannwhitneyu
from scipy.integrate import cumtrapz
from scipy.integrate import simps
from mpl_toolkits import mplot3d 
from scipy.ndimage import median_filter, gaussian_filter
from importlib import sys, reload
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, '/project/nicho/projects/marmosets/code_database/analysis/trajectory_encoding_model/')
from utils_before_nwb import *

class path:
    # storage = '/home/daltonm/Documents/tmp_analysis_folder/processed_datasets' # /project2/nicho/dalton/processed_datasets'
    # intermediate_save_path = '/home/daltonm/Documents/tmp_analysis_folder/analysis/encoding_model/intermediate_variable_storage'#'/project2/nicho/dalton/analysis/encoding_model/intermediate_variable_storage'
    # new_save_path = '/home/daltonm/Documents/tmp_analysis_folder/analysis/encoding_model/intermediate_variable_storage/10pt0_ms_bins/data_updated_february_2023' #'/project2/nicho/dalton/analysis/encoding_model/intermediate_variable_storage/10pt0_ms_bins/data_updated_february_2023'
    # plots = '/home/daltonm/Documents/tmp_analysis_folder/analysis/encoding_model/plots' #'/project2/nicho/dalton/analysis/encoding_model/plots'
    # date = '20210211'
    storage = '/project2/nicho/dalton/processed_datasets'
    intermediate_save_path = '/project2/nicho/dalton/analysis/encoding_model/intermediate_variable_storage'
    new_save_path = '/project2/nicho/dalton/analysis/encoding_model/intermediate_variable_storage/10pt0_ms_bins/data_updated_february_2023'
    plots = '/project2/nicho/dalton/analysis/encoding_model/plots/new_analysis_february_2023'
    date = '20210211'    
class params:
    spkSampWin = 0.01
    trajShift = 0.05 #sample every 50ms
    lead = [0.5, 0.4, 0.4, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0  , 0  , 0  ] #[0.2, 0.15, 0.1, 0.05, 0] #[0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0] #[0.05] # 100ms lead time
    lag  = [0  , 0  , 0.1, 0  , 0.1, 0.2, 0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.3, 0.4, 0.5] #[0.2, 0.25, 0.3, 0.35, 0.4] #[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4] #[0.35] # 300ms lag time
    spkRatioCheck = 'off'
    normalize = 'off' # NEED TO FIX NORMALIZATION SO THAT IT NORMALIZES ENTIRE TRAJECTORY SET PRIOR TO PCA, not individual chunks during collection!!
    numThresh = 1000
    trainRatio = 0.9
    numIters = 100
    minSpikeRatio = .005
    nDims = 3
    nShuffles = 1
    starting_eventNum = None 
    frate_thresh = 2
    fps = 150
    
    pca_var_thresh = 0.9# MAKE SURE that the idx being pulled is for the hand in run_pca_on_trajectories()
    include_avg_speed = False
    include_avg_pos = False
    network = 'on'
    FN_source = 'split_reach_FNs'
    hand_traj_idx = 0
    compute_shortTraj = False
    
    networkSampleBins = 3
    networkFeatureBins = 2 
    
class plot_params:
    axis_fontsize = 24
    dpi = 300
    axis_linewidth = 2
    tick_length = 2
    tick_width = 1
    map_figSize = (6, 8)
    tick_fontsize = 18

def trajectory_vs_shuffle_sign_test(traj_res, shuf_res, unit_info):
    
    ttest_p = [0]*traj_res.shape[0]
    mwu_p   = [0]*traj_res.shape[0]
    for idx, (unit_traj_auc, unit_shuf_auc) in enumerate(zip(traj_res, shuf_res)):
        
        tmp, ttest_p[idx] = ttest_ind   (unit_traj_auc, unit_shuf_auc, alternative='greater')
        tmp,   mwu_p[idx] = mannwhitneyu(unit_traj_auc, unit_shuf_auc, alternative='greater')
    
    unit_info['ttest_p'] = ttest_p
    unit_info['mwu_p'  ] = mwu_p
    
    return unit_info

def determine_trajectory_significance(all_models_data):
    for unit_info, model_details in zip(all_models_data['unit_info'], all_models_data['model_details']):
        traj_idx = [idx for idx, name in enumerate(model_details['model_names']) if name == 'trajectory'][0]
        shuf_idx = [idx for idx, name in enumerate(model_details['model_names']) if name == 'shuffle'][0]

        traj_AUC_results = model_details['model_results'][traj_idx]['AUC']
        shuf_AUC_results = model_details['model_results'][shuf_idx]['AUC']
        
        tmp_res = pd.DataFrame(data = zip(np.hstack((traj_AUC_results[0], shuf_AUC_results[0])), ['traj']*100+['shuf']*100), 
                               columns = ['auc', 'model'])
        # fig, ax = plt.subplots()
        # sns.histplot(data = tmp_res, ax=ax, x = 'auc', hue = 'model', bins = 10)
        # plt.show()
        
        unit_info = trajectory_vs_shuffle_sign_test(traj_AUC_results, shuf_AUC_results, unit_info)
            
    return all_models_data

def organize_results_by_model_for_all_lags(all_models_data):
    
    tmp_unit_info = all_models_data['unit_info'][0]
    model_keys = [key for key in tmp_unit_info.columns if 'AUC' in key and 'shuffled' not in key and 'brief' not in key]
    corrected_model_names = [name.replace('pathlet', 'trajectory').split('_AUC')[0] for name in model_keys]
    
    # model_results_across_lags = {'model_name'    : [0]*len(model_keys),
    #                              'model_results' : [pd.DataFrame()]*len(model_keys)}
    results       = []
    model_names   = []
    tuning_df    = pd.DataFrame()
    sign_prop_df = pd.DataFrame()
    for idx, (key, name) in enumerate(zip(model_keys, corrected_model_names)):
        model_names.append(name)
        tmp_results = pd.DataFrame()
        for unit_info, lead_lag in zip(all_models_data['unit_info'], all_models_data['lead_lag']): 
            tmp_results['lead_%d_lag_%d' % (lead_lag[0], lead_lag[1])] = unit_info[key] 
            if name=='trajectory':
                tuning_df   ['lead_%d_lag_%d' % (lead_lag[0], lead_lag[1])] = unit_info['tuning']
                sign_prop_df['lead_%d_lag_%d' % (lead_lag[0], lead_lag[1])] = unit_info['proportion_sign']
                
        results.append(tmp_results)
    
    model_results_across_lags = {'model_name'    : model_names,
                                 'model_results' : results,
                                 'tuning'        : tuning_df,
                                 'signtest_prop' : sign_prop_df}     
    
    return model_results_across_lags

def find_optimal_lag_for_each_unit(model_results_across_lags, all_models_data, only_tuned = True):
    modelIdx = [idx for idx, name in enumerate(model_results_across_lags['model_name']) if name == 'trajectory'][0]
    traj_results = model_results_across_lags['model_results'][modelIdx]
    
    if only_tuned:
        traj_results = traj_results[model_results_across_lags['tuning'] == 'tuned']
    
    optimal_lead_lag = traj_results.idxmax(axis = 1)
    lead_pattern = re.compile('lead_\d{1,3}')
    lag_pattern  = re.compile('lag_\d{1,3}')
    optimal_lead_lag = [(int(re.findall(lead_pattern, leadlag)[0].split('lead_')[-1]), 
                         int(re.findall(lag_pattern, leadlag)[0].split('lag_')[-1])) 
                        if type(leadlag) == str 
                        else (np.nan, np.nan)
                        for leadlag in optimal_lead_lag]
    optimal_traj_center = [(-1*ll[0] + ll[1]) // 2 for ll in optimal_lead_lag]

    unit_info_optimal_lag = all_models_data['unit_info'][0].copy()
    unit_info_optimal_lag[['lead', 'lag', 'traj_center']] = np.full((unit_info_optimal_lag.shape[0], 3), np.nan)
    for idx, (opt_ll, opt_center) in enumerate(zip(optimal_lead_lag, optimal_traj_center)):
        try:
            tmp_ll_idx = [ll_idx for ll_idx, ll in enumerate(all_models_data['lead_lag']) if ll == opt_ll][0]
            tmp_unit_info = all_models_data['unit_info'][tmp_ll_idx]
            columns_with_auc_or_tuning = [idx for idx, col in enumerate(tmp_unit_info.columns) if 'AUC' in col or 'tuning' in col]
            columns_in_unit_info = [idx for idx, col in enumerate(unit_info_optimal_lag.columns) if 'AUC' in col or 'tuning' in col]
            unit_info_optimal_lag.iloc[idx, columns_in_unit_info] = tmp_unit_info.iloc[idx, columns_with_auc_or_tuning]
        except:
            pass
        unit_info_optimal_lag.at[unit_info_optimal_lag.index[idx], 'lead']    = opt_ll[0]
        unit_info_optimal_lag.at[unit_info_optimal_lag.index[idx], 'lag' ]    = opt_ll[1]
        unit_info_optimal_lag.at[unit_info_optimal_lag.index[idx], 'traj_center'] = opt_center
        
    return unit_info_optimal_lag

def compute_mean_model_performance(model_results_across_lags, percent = 0, percentile_mode='per_lag_set'):
    
    model_results_across_lags['mean_performance_by_lead_lag_all']     = [0]*len(model_results_across_lags['model_name'])
    model_results_across_lags['mean_performance_by_lead_lag_untuned'] = [0]*len(model_results_across_lags['model_name'])
    model_results_across_lags['mean_performance_by_lead_lag_tuned']   = [0]*len(model_results_across_lags['model_name'])
    model_results_across_lags['mean_performance_by_lead_lag_filtered_by_percentile'] = [0]*len(model_results_across_lags['model_name'])
    for idx, results in enumerate(model_results_across_lags['model_results']):
        
        # compute means and SE using AUC percentile filter
        tmp_results = results.copy()
        if percentile_mode == 'across_lag_sets':
            tmp_results = tmp_results[tmp_results >= np.percentile(tmp_results, percent)]
        elif percentile_mode == 'per_lag_set':
            for col in tmp_results.columns:
                tmp_results.loc[tmp_results[col] < np.percentile(tmp_results[col], percent), col] = np.nan
        model_results_across_lags['mean_performance_by_lead_lag_filtered_by_percentile'][idx] = pd.DataFrame(data=zip(tmp_results.mean().to_numpy(), tmp_results.sem().to_numpy()),
                                                                                                             index=results.columns,
                                                                                                             columns=['auc', 'sem'])    
        # compute means and SE for all, no filter
        tmp_results = results.copy()
        model_results_across_lags['mean_performance_by_lead_lag_all'][idx] = pd.DataFrame(data=zip(tmp_results.mean().to_numpy(), tmp_results.sem().to_numpy()),
                                                                                          index=results.columns,
                                                                                          columns=['auc', 'sem'])    
        # compute means and SE for tuned units
        tmp_results = results.copy()
        tmp_results = tmp_results[model_results_across_lags['tuning'] == 'tuned']
        model_results_across_lags['mean_performance_by_lead_lag_tuned'][idx] = pd.DataFrame(data=zip(tmp_results.mean().to_numpy(), tmp_results.sem().to_numpy()),
                                                                                          index=results.columns,
                                                                                          columns=['auc', 'sem']) 

        # compute means and SE for untuned units
        tmp_results = results.copy()
        tmp_results = tmp_results[model_results_across_lags['tuning'] == 'untuned']
        model_results_across_lags['mean_performance_by_lead_lag_untuned'][idx] = pd.DataFrame(data=zip(tmp_results.mean().to_numpy(), tmp_results.sem().to_numpy()),
                                                                                          index=results.columns,
                                                                                          columns=['auc', 'sem'])   
        
    return model_results_across_lags

def plot_optimal_lag_on_channel_map(unit_info, chan_map_df, jitter_radius = .15, 
                                    distance_mult = 400, hueKey = 'traj_center', 
                                    sizeKey = 'pathlet_AUC', weighted_mean = False,
                                    weightsKey = None):
    
    jitter_radius = jitter_radius * distance_mult
    
    scatter_unit_info = unit_info.copy()
    scatter_unit_info['scatter_x'] = np.full((scatter_unit_info.shape[0],), np.nan)
    scatter_unit_info['scatter_y'] = np.full((scatter_unit_info.shape[0],), np.nan)
    for ch in np.unique(scatter_unit_info.ns6_elec_id):
        chan_mask = scatter_unit_info.ns6_elec_id == ch
        chan_clusters = scatter_unit_info.loc[chan_mask, 'cluster_id']
        if len(chan_clusters) == 1:
            jitters = [(0, 0)]
        else:
            jitters = [(np.round(jitter_radius * np.cos(n*2*np.pi / len(chan_clusters)), 3), 
                        np.round(jitter_radius * np.sin(n*2*np.pi / len(chan_clusters)), 3)) for n in range(len(chan_clusters))]
        base_pos = scatter_unit_info.loc[chan_mask, ['x', 'y']]
        base_pos = np.array([base_pos['x'].values[0], base_pos['y'].values[0]])        
               
        scatter_unit_info.loc[chan_mask, 'scatter_x'] = [jitter[0] + base_pos[0] for jitter in jitters]
        scatter_unit_info.loc[chan_mask, 'scatter_y'] = [jitter[1] + base_pos[1] for jitter in jitters]
    
    x_vals = np.unique(scatter_unit_info.x)
    mean_traj_centers = []
    sem_traj_centers = []
    if weightsKey is None:
        weightsKey = sizeKey 
    for x in x_vals:
        traj_centers = scatter_unit_info.loc[scatter_unit_info['x'] == x, 'traj_center']
        if weighted_mean:
            weights = scatter_unit_info.loc[scatter_unit_info['x'] == x, weightsKey]
            idxs = traj_centers.index[~np.isnan(traj_centers)]
            weights = weights[idxs]
            traj_centers = traj_centers[idxs]
            weighted_average = np.average(a = traj_centers, weights = weights)
            mean_traj_centers.append(weighted_average)    
        else:
            mean_traj_centers.append(traj_centers.mean())
        sem_traj_centers.append(traj_centers.sem())
    
    fig, (ax_top, ax) = plt.subplots(2, 1, figsize=plot_params.map_figSize, gridspec_kw={'height_ratios': [1.25, 4]})
    sns.scatterplot(ax = ax, data = scatter_unit_info, x = 'scatter_x', y = 'scatter_y', 
                    size = sizeKey, hue = hueKey, style = "group", palette='seismic',
                    edgecolor="black")
    ax.vlines(np.arange(-0.5* distance_mult, 10.5* distance_mult, 1* distance_mult), -0.5* distance_mult, 9.5* distance_mult, colors='black')
    ax.hlines(np.arange(-0.5* distance_mult, 10.5* distance_mult, 1* distance_mult), -0.5* distance_mult, 9.5* distance_mult, colors='black')
    ax.set_xlim(-0.5* distance_mult, 9.5* distance_mult)
    ax.set_ylim(-0.5* distance_mult, 9.5* distance_mult)
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color('black')
    for axis in ['top','right']:
        ax.spines[axis].set_linewidth(0)
    ax.tick_params(width=0, length = 0, labelsize = 0)
    ax.set_ylabel('')
    ax.set_xlabel('')
    # ax.set_xlabel('Anterior'  , fontsize = plot_params.axis_fontsize, fontweight = 'bold')
    # ax.set_ylabel('Lateral', fontsize = plot_params.axis_fontsize, fontweight = 'bold')
    # ax.legend(bbox_to_anchor=(-.25, 1), loc='upper right', borderaxespad=0)
    ax.get_legend().remove()
    ax.set_title(hueKey)

    ax.grid(False)
    
    ax_top.errorbar(x = x_vals, 
                    y = mean_traj_centers, 
                    yerr = sem_traj_centers, 
                    linewidth=0,
                    elinewidth=3,
                    marker='o',
                    markersize=10,
                    color='black')

    ax_top.set_xticks([])
    ax_top.set_xticklabels([])
    ax_top.set_ylim([-125, 125])
    ax_top.set_yticks([-100, 0, 100])
    
    
    # for txt, x, y, scat_x, scat_y in zip(scatter_unit_info['ns6_elec_id'], scatter_unit_info['center_x'], scatter_unit_info['center_y'],
    #                      scatter_unit_info['scatter_x'], scatter_unit_info['scatter_y']):
    #     print((txt, x, y))
    #     ax.annotate('%d' % txt, (x, y))
    plt.show()

    fig.savefig(os.path.join(path.plots, '%s_map.png' % hueKey), bbox_inches='tight', dpi=plot_params.dpi)
            
    # fig.savefig('C:/Users/Dalton/Documents/lab_files/analysis_encoding_model/plots/map_%s' % key, bbox_inches='tight', dpi=plot_params.dpi)

def plot_sweep_over_lead_lag(model_results_across_lags, filter_key):
    reorder = [12, 13, 14, 8, 10, 11, 6, 7, 9, 2, 4, 5, 1, 3, 0]
    
    traj_mean_performance = model_results_across_lags['mean_performance_by_lead_lag_%s' % filter_key][0].copy()
    if 'tuned' in filter_key:
        tuning = model_results_across_lags['tuning']
        mask = np.where(tuning == filter_key)
        nUnits = [sum(mask[1] == model) for model in np.unique(mask[1])]
    else:
        nUnits = [model_results_across_lags['tuning'].shape[0]]*len(reorder)
    traj_mean_performance['nUnits'] = nUnits
    traj_mean_performance['reorder'] = reorder
    traj_mean_performance.sort_values(by='reorder', inplace=True)
    
    # fig, ax = plt.subplots()
    # ax.errorbar(traj_mean_performance['reorder'], traj_mean_performance['auc'], yerr=traj_mean_performance['SE'])
    # ax.set_xticks(traj_mean_performance['reorder'])
    # ax.set_xticklabels(traj_mean_performance.index, rotation=45)
    # plt.show()
    
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.errorbar(traj_mean_performance['reorder'], 
                traj_mean_performance['auc'], 
                yerr=traj_mean_performance['sem'], 
                linewidth=0,
                elinewidth=3,
                marker='o',
                markersize=10,
                color='black')
    ax.errorbar(9,
                traj_mean_performance['auc'].iloc[9],
                yerr=traj_mean_performance['sem'].iloc[9], 
                linewidth=0,
                elinewidth=3,
                marker='o',
                markersize=10,
                color='green')
    
    if 'tuned' in filter_key:
        y = np.max(traj_mean_performance['auc']) + np.max(traj_mean_performance['sem']) 
        
        for x, count in zip(traj_mean_performance['reorder'], traj_mean_performance['nUnits']):
            ax.text(x-.25, y, str(count))
        
    # ax.set_xlabel('Top Percent of Weights Shuffled', fontsize = plot_params.axis_fontsize)
    # ax.set_ylabel('Percent AUC Loss', fontsize = plot_params.axis_fontsize)
    # ax.set_ylabel('')
    # ax.set_xlabel('')
    ax.set_xlabel('Trajectory Center (ms)', fontsize = plot_params.axis_fontsize)
    ax.set_ylabel('AUC (Mean %s sem)' % '\u00B1', fontsize = plot_params.axis_fontsize)
    ax.set_xticks([])
    ax.set_xticklabels([])
    if filter_key == 'all':
        ax.set_yticks([0.54, 0.57])
    elif filter_key == 'tuned':
        ax.set_yticks([0.58, 0.61])
    ax.tick_params(width=2, length = 4, labelsize = plot_params.tick_fontsize)
    # ax.set_xticks(traj_mean_performance['reorder'])
    # for tick in ax.get_xticklabels():
    #     tick.set_fontsize(plot_params.tick_fontsize)
    # for tick in ax.get_yticklabels():
    #     tick.set_fontsize(plot_params.tick_fontsize)
    ax.set_xticks(traj_mean_performance['reorder'])
    ax.set_xticklabels([-250, -200, -150, -150, -100, -50, -50, 0, 50, 50, 100, 150, 150, 200, 250], rotation=45)

    sns.despine(ax=ax)
    ax.spines['bottom'].set_linewidth(plot_params.axis_linewidth)
    ax.spines['left'  ].set_linewidth(plot_params.axis_linewidth)
    
    
    
    plt.show()
    
    if filter_key is None:
        fig.savefig(os.path.join(path.plots, 'model_auc_over_leadlags_unfiltered.png'), bbox_inches='tight', dpi=plot_params.dpi)
    else:
        fig.savefig(os.path.join(path.plots, 'model_auc_over_leadlags_filtered_by_%s.png' % filter_key), bbox_inches='tight', dpi=plot_params.dpi)
        
    # fig.savefig('C:/Users/Dalton/Documents/lab_files/AREADNE/plots/model_auc_over_leadlags.png', bbox_inches='tight', dpi=plot_params.dpi)


if __name__ == "__main__":
    
    all_models_data = load_all_models_dict(os.path.join(path.new_save_path, 'all_models_data_dict.pkl'))

    chan_map_df = load_channel_map_from_prb(marm = 'Tony')
    
    # all_models_data = determine_trajectory_significance(all_models_data)
    
    model_results_across_lags = organize_results_by_model_for_all_lags(all_models_data)
    
    model_results_across_lags = compute_mean_model_performance(model_results_across_lags, percent = 25, percentile_mode = 'per_lag_set')
    
    unit_info_optimal_lag = find_optimal_lag_for_each_unit(model_results_across_lags, all_models_data, only_tuned=False)
    
    unit_info_optimal_lag = fix_unit_info_elec_labels(unit_info_optimal_lag, chan_map_df)
    
    plot_optimal_lag_on_channel_map(unit_info_optimal_lag, chan_map_df, 
                                    jitter_radius = .15, distance_mult = 400, 
                                    hueKey = 'traj_center', sizeKey = 'pathlet_AUC',
                                    weighted_mean = True, weightsKey='proportion_sign')
        
    plot_sweep_over_lead_lag(model_results_across_lags, filter_key = 'all')