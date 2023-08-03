#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 08:00:51 2020

@author: daltonm
"""
#%matplotlib notebook
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import dill
import os
import glob
import math
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc
import statsmodels.api as sm
from scipy.stats import binomtest, ttest_rel
from scipy.integrate import cumtrapz
from scipy.integrate import simps
from mpl_toolkits import mplot3d 
from scipy.ndimage import median_filter, gaussian_filter
from importlib import sys, reload


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
    
class plot_params:
    palette_path = os.path.join(os.path.dirname(path.plots), '0-1/Linear_L_0-1.csv')
    axis_fontsize = 24
    dpi = 300
    axis_linewidth = 2
    tick_length = 2
    tick_width = 1
    tick_fontsize = 18
    boxplot_figSize = (5.5, 5.5)
    aucScatter_figSize = (7, 7)
    FN_figSize = (7,7)
    map_figSize = (7, 7)
    plot_linewidth = 3
    
class params:
    spkSampWin = 0.01
    trajShift = 0.05 #sample every 50ms
    lead = [0.2] #[0.2, 0.15, 0.1, 0.05, 0] #[0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0] #[0.05] # 100ms lead time
    lag = [0.3] #[0.2, 0.25, 0.3, 0.35, 0.4] #[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4] #[0.35] # 300ms lag time
    lead_to_analyze = [0.2]
    lag_to_analyze  = [0.3]
    spkRatioCheck = 'off'
    normalize = 'off'
    numThresh = 100
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
    hand_traj_idx = 0 
    FN_source = 'split_reach_FNs'
    
    networkSampleBins = 3
    networkFeatureBins = 2 

def plot_model_auc_comparison(unit_info, x_key, y_key, minauc = 0.5, targets=None):
    fig, ax = plt.subplots(figsize = plot_params.aucScatter_figSize)
    # sns.scatterplot(ax = ax, data = unit_info, x = x_key, y = y_key, 
    #                 hue = "fr", style = "group")
    sns.scatterplot(ax = ax, data = unit_info, x = x_key, y = y_key, 
                    style = "group", s = 60, legend=False)
    ax.plot(np.arange(minauc, 1.0, 0.05), np.arange(minauc, 1.0, 0.05), '--k')
    # ax.scatter(unit_info[x_key].to_numpy()[44] , unit_info[y_key].to_numpy()[44] , s = 60, c ='red', marker='x')
    # ax.scatter(unit_info[x_key].to_numpy()[107], unit_info[y_key].to_numpy()[107], s = 60,  c ='red', marker='o')
    ax.set_xlim(minauc, 1)
    ax.set_ylim(minauc, 1)
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color('black')
    for axis in ['top','right']:
        ax.spines[axis].set_linewidth(0)
    ax.tick_params(width=2, length = 4, labelsize = plot_params.tick_fontsize)
    ax.set_xlabel('ROC area (%s)' % x_key[:-4], fontsize = plot_params.axis_fontsize)
    ax.set_ylabel('ROC area (%s)' % y_key[:-4], fontsize = plot_params.axis_fontsize)
    # ax.set_xlabel('')
    # ax.set_ylabel('')
    ax.grid(False)
    # ax.legend(bbox_to_anchor=(1.5, 1.5), loc='upper left', borderaxespad=0)
    plt.show()
    
    if targets is None:
        fig.savefig(os.path.join(path.plots, 'area_under_curve_%s_%s.png' % (x_key, y_key)), bbox_inches='tight', dpi=plot_params.dpi)
    else:
        fig.savefig(os.path.join(path.plots, 'area_under_curve_%s_%s_%s_targetUnits.png' % (x_key, y_key, targets)), bbox_inches='tight', dpi=plot_params.dpi)

def prune_for_neurons_with_same_channel_connections(unit_info):
    
    unit_info_pruned = unit_info.copy()
    
    for elec_id in np.unique(unit_info['ns6_elec_id']):
        if (unit_info['ns6_elec_id'] == elec_id).sum() < 2:
            unit_info_pruned.loc[unit_info['ns6_elec_id'] == elec_id, :] = np.nan
    
    return unit_info_pruned

if __name__ == "__main__":
    
    models_dict_path = os.path.join(path.new_save_path, 'all_models_data_dict.pkl')
    all_models_data = load_all_models_dict(models_dict_path) 
    
    single_lead_lag_models, ll_idx = get_single_lead_lag_models(all_models_data, params.lead[0], params.lag[0])
    unit_info = single_lead_lag_models['unit_info']
    chan_map_df = load_channel_map_from_prb(marm = 'Tony')   
    unit_info = fix_unit_info_elec_labels(unit_info, chan_map_df)

    thresh = 0.90
    plot_model_auc_comparison(unit_info.loc[unit_info.proportion_sign >= thresh, :], 'tuned_inputs_FN_AUC', 'untuned_inputs_FN_AUC')
    unit_info_sorted = unit_info.sort_values(by='proportion_sign', ascending = False)
    num_units = sum(unit_info.proportion_sign >= thresh)
    plot_model_auc_comparison(unit_info_sorted.iloc[-num_units:, :], 'tuned_inputs_FN_AUC', 'untuned_inputs_FN_AUC')
    
    plot_model_auc_comparison(unit_info.loc[unit_info.proportion_sign >= thresh, :], 'tuned_inputs_FN_AUC', 'full_AUC', targets='tuned')
    plot_model_auc_comparison(unit_info.loc[unit_info.proportion_sign >= thresh, :], 'untuned_inputs_FN_AUC', 'full_AUC', targets='tuned')

    plot_model_auc_comparison(unit_info_sorted.iloc[-num_units:, :], 'tuned_inputs_FN_AUC', 'full_AUC', targets='untuned')
    plot_model_auc_comparison(unit_info_sorted.iloc[-num_units:, :], 'untuned_inputs_FN_AUC', 'full_AUC', targets='untuned')


    
    unit_info_pruned = prune_for_neurons_with_same_channel_connections(unit_info)

    plot_model_auc_comparison(unit_info_pruned, 'zero_dist_FN_AUC', 'full_AUC')
    plot_model_auc_comparison(unit_info_pruned, 'nonzero_dist_FN_AUC', 'full_AUC')
    plot_model_auc_comparison(unit_info_pruned, 'pathlet_AUC', 'zero_dist_FN_AUC')
    plot_model_auc_comparison(unit_info_pruned, 'pathlet_AUC', 'nonzero_dist_FN_AUC')
    
    plot_model_auc_comparison(unit_info_pruned, 'full_AUC', 'transposed_FN_AUC')

    ''' 
        Note: get idxs of units that have zero_distance inputs, compare full vs zero vs nonzero for these units
    '''

    
    
