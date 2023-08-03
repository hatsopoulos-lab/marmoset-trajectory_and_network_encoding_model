# -*- coding: utf-8 -*-
"""
Created on Tue May 31 10:09:38 2022

@author: Dalton
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
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
from scipy.stats import binomtest, ttest_rel, ttest_ind, mannwhitneyu, linregress, pearsonr
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

if marmcode=='TY':
    nwb_infile = '/project/nicho/projects/dalton/data/TY/TY20210211_freeAndMoths-003_resorted_20230612_DM_with_functional_networks.nwb' 
    pkl_infile = '/project/nicho/projects/dalton/data/TY/TY20210211_freeAndMoths-003_resorted_20230612_DM_encoding_model_sorting_corrected_30mscortical_networks_shift_v6.pkl'
    filtered_good_units_idxs = [88, 92, 123]
elif marmcode=='MG':
    nwb_infile = ''
    pkl_infile = ''
    
split_pattern = '_shift_v' # '_results_v'
base, ext = os.path.splitext(pkl_infile)
base, in_version = base.split(split_pattern)
out_version = str(int(in_version) + 1)  
pkl_outfile = base + split_pattern + out_version + ext

dataset_code = os.path.basename(pkl_infile)[:10] 
plots = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(pkl_infile))), 'plots', dataset_code, 'network')

class params:
    best_lead_lag_key = 'lead_200_lag_300' #None
    FN_key = 'split_reach_FNs'#'split_reach_FNs'
    significant_proportion_thresh = 0.99
    tuned_auc_thresh = 0.6
 
    primary_traj_model = 'traj_avgPos'
    # cortical_boundaries = {'x_coord'      : [   0,          400,  800, 1200,         1600, 2000, 2400, 2800,          3200,          3600],
    #                         'y_bound'      : [None,         1200, None, None,         1200, None, None, None,           800,          2000],
    #                         'areas'        : ['3b', ['3a', '3b'], '3a', '3a', ['M1', '3a'], 'M1', 'M1', 'M1', ['6Dc', 'M1'], ['6Dc', 'M1']],
    #                         'unique_areas' : ['3b', '3a', 'M1', '6Dc']}

    cortical_boundaries = {'x_coord'      : [   0,          400,  800, 1200,         1600, 2000, 2400, 2800, 3200, 3600],
                            'y_bound'      : [None,         1200, None, None,         1200, None, None, None, None, None],
                            'areas'        : ['3b', ['3a', '3b'], '3a', '3a', ['M1', '3a'], 'M1', 'M1', 'M1', 'M1', 'M1'],
                            'unique_areas' : ['3b', '3a', 'M1']}
        
class plot_params:
    axis_fontsize = 20
    dpi = 300
    axis_linewidth = 2
    tick_length = 2
    tick_width = 1
    tick_fontsize = 18

    map_figSize = (6, 8)
    weights_by_distance_figsize = (6, 4)
    aucScatter_figSize = (6, 6)
    feature_corr_figSize = (4, 4)
    
plt.rcParams['figure.dpi'] = plot_params.dpi
plt.rcParams['savefig.dpi'] = plot_params.dpi

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
            elif '%s_full_FN' % params.primary_traj_model == model_key:
                col_names = ['%s_auc' % model_key, '%s_train_auc' % model_key]  
                results_keys = ['AUC', 'trainAUC']   
            else:
                col_names = ['%s_auc' % model_key]  
                results_keys = ['AUC']  

            for col_name, results_key in zip(col_names, results_keys):                
                if col_name not in all_units_res.columns: 
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
        
def plot_model_auc_comparison(units_res, x_key, y_key, minauc = 0.5, hue_key='W_in', style_key='cortical_area', targets=None, col_key=None):
    
    if x_key[-4:] != '_auc':
        x_key = x_key + '_auc'
    if y_key[-4:] != '_auc':
        y_key = y_key + '_auc'
    
    units_res_plots = units_res.copy()
        
    if targets is not None:
        if type(targets) != list:
            targets = [targets]
        units_res_plots = isolate_target_units_for_plots(units_res_plots, targets)        
        plot_title = 'Targets:'
        plot_name = 'area_under_curve_%s_%s_targetUnits' % (x_key, y_key)
        for targ in targets:
            plot_title = plot_title + ' %s,' % targ
            plot_name = plot_name + '_%s' % targ
    else:
        plot_title = 'Targets: All units'
        plot_name = f'area_under_curve_{x_key}_{y_key}'
    
    if hue_key is not None:
        plot_name += f'_hueKey_{hue_key}'

    if style_key is not None:
        plot_name += f'_styleKey_{style_key}'

    if col_key is not None:
        plot_name += f'_colKey_{col_key}'
        
        fig = sns.relplot(data = units_res_plots, x=x_key, y=y_key, hue=hue_key, 
                          col=col_key, style = style_key, kind='scatter', legend=True)
        for ax, area in zip(fig.axes[0], ['M1', '3a', '3b']):
            ax.set_xlim(minauc, 1)
            ax.set_ylim(minauc, 1)
            ax.set_xticks(np.arange(np.ceil(minauc*10)/10, 1.01, 0.1))
            ax.plot(np.arange(minauc, 1.0, 0.05), np.arange(minauc, 1.0, 0.05), '--k')
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
    else:
        fig, ax = plt.subplots(figsize = plot_params.aucScatter_figSize, dpi = plot_params.dpi)
        sns.scatterplot(ax = ax, data = units_res_plots, x = x_key, y = y_key, 
                        hue = hue_key, style = style_key, s = 60, legend=True)     

        ax.plot(np.arange(minauc, 1.0, 0.05), np.arange(minauc, 1.0, 0.05), '--k')
        # ax.scatter(units_res_plots[x_key].to_numpy()[44] , units_res_plots[y_key].to_numpy()[44] , s = 60, c ='red', marker='x')
        # ax.scatter(units_res_plots[x_key].to_numpy()[107], units_res_plots[y_key].to_numpy()[107], s = 60,  c ='red', marker='o')
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
        ax.set_title(plot_title)
        ax.grid(False)
        
        if 'spont_train_reach_test_FN' in x_key:
            ax.plot(np.arange(minauc, 1.0, 0.05), np.arange(minauc, 1.0, 0.05) + 0.04*np.sqrt(2), '-.r')
    
    # ax.legend(bbox_to_anchor=(1.5, 1.5), loc='upper left', borderaxespad=0)
    plt.show()
    
    fig.savefig(os.path.join(plots, plot_name + '.png'), bbox_inches='tight', dpi=plot_params.dpi)


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
    
def plot_model_training_performance_comparison(units_res, x_key, y_key, lead_lag_key, metric='logLikelihood', targets=None):
    
    units_res, metric_distributions = get_training_metric_distributions_and_means(units_res, [x_key, y_key], lead_lag_key, metric = metric)
    
    units_res_plots = units_res.copy()
        
    if targets is not None:
        if type(targets) != list:
            targets = [targets]
        units_res_plots = isolate_target_units_for_plots(units_res_plots, targets)        
        plot_title = 'Targets:'
        plot_name = '%s_%s_%s_targetUnits' % (metric, x_key, y_key)
        for targ in targets:
            plot_title = plot_title + ' %s,' % targ
            plot_name = plot_name + '_%s' % targ
        plot_name = plot_name + '.png'
    else:
        plot_title = 'Targets: All units'
        plot_name = '%s_%s_%s.png' % (metric, x_key, y_key)

    
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
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color('black')
    for axis in ['top','right']:
        ax.spines[axis].set_linewidth(0)
    ax.tick_params(width=2, length = 4, labelsize = plot_params.tick_fontsize)
    ax.set_xlabel('%s_%s' % (x_key, metric), fontsize = plot_params.axis_fontsize)
    ax.set_ylabel('%s_%s' % (y_key, metric), fontsize = plot_params.axis_fontsize)
    # ax.set_xlabel('')
    # ax.set_ylabel('')
    ax.grid(False)
    ax.set_title(plot_title)
    # ax.legend(bbox_to_anchor=(1.5, 1.5), loc='upper left', borderaxespad=0)
    plt.show()
    
    fig.savefig(os.path.join(plots, plot_name), bbox_inches='tight', dpi=plot_params.dpi)

def plot_functional_networks(FN, units_res, FN_key = 'split_reach_FNs', cmin=None, cmax=None, subset_idxs = None, subset_type='both'):
    
    units_sorted = units_res.copy()    
    units_sorted.sort_values(by='cortical_area', inplace=True, ignore_index=False)

    if subset_idxs is not None:
        units_sorted_subset = units_res.copy().loc[subset_idxs, :]
        units_sorted_subset.sort_values(by='cortical_area', inplace=True, ignore_index=False) 
        
        subset_tick_3a = np.sum(units_sorted_subset['cortical_area'] == '3a')
        subset_tick_3b = subset_tick_3a + np.sum(units_sorted_subset['cortical_area'] == '3b') 
        subset_tick_m1 = subset_tick_3b + np.sum(units_sorted_subset['cortical_area'] == 'M1')
    
    tick_3a = np.sum(units_sorted['cortical_area'] == '3a')
    tick_3b = tick_3a + np.sum(units_sorted['cortical_area'] == '3b') 
    tick_m1 = tick_3b + np.sum(units_sorted['cortical_area'] == 'M1')
    
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
        titles = ['Reach Set 1', 'Reach Set 2']
    elif FN_key == 'spontaneous_FN':
        titles = ['Spontaneous']
    
    for network, title in zip(FN, titles):
        fig, ax = plt.subplots(figsize=(6,6), dpi = plot_params.dpi)
        network_copy = network.copy()
        
        if subset_idxs is not None:
            if subset_idxs.size > FN.shape[-1]/2:
                title += ' Non'
            if subset_type == 'both':
                title += ' Reach Specific'
                target_idx, source_idx = units_sorted_subset.index.values, units_sorted_subset.index.values 
                xtick_3a, xtick_3b, xtick_m1 = subset_tick_3a, subset_tick_3b, subset_tick_m1
                ytick_3a, ytick_3b, ytick_m1 = subset_tick_3a, subset_tick_3b, subset_tick_m1               
            elif subset_type == 'target':
                title += ' Reach Specific Targets'
                target_idx, source_idx = units_sorted_subset.index.values, units_sorted.index.values  
                xtick_3a, xtick_3b, xtick_m1 =        tick_3a,        tick_3b,        tick_m1
                ytick_3a, ytick_3b, ytick_m1 = subset_tick_3a, subset_tick_3b, subset_tick_m1  
            elif subset_type == 'source':
                title += f' Reach Specific Sources'
                target_idx, source_idx = units_sorted.index.values, units_sorted_subset.index.values  
                xtick_3a, xtick_3b, xtick_m1 = subset_tick_3a, subset_tick_3a, subset_tick_3a
                ytick_3a, ytick_3b, ytick_m1 =        tick_3a,        tick_3b,        tick_m1  
        else:
            target_idx, source_idx = units_sorted.index.values, units_sorted.index.values  
            xtick_3a, xtick_3b, xtick_m1 = tick_3a, tick_3b, tick_m1
            ytick_3a, ytick_3b, ytick_m1 = tick_3a, tick_3b, tick_m1  

        network_copy = network_copy[np.ix_(target_idx, source_idx)]
        sns.heatmap(network_copy,ax=ax,cmap= 'viridis',square=True, norm=colors.PowerNorm(0.5, vmin=cmin, vmax=cmax)) # norm=colors.LogNorm(vmin=cmin, vmax=cmax)
        ax.set_xticks([np.mean([0, xtick_3a]), xtick_3a, np.mean([xtick_3a, xtick_3b]), xtick_3b, np.mean([xtick_3b, xtick_m1])])
        ax.set_yticks([np.mean([0, ytick_3a]), ytick_3a, np.mean([ytick_3a, ytick_3b]), tick_3b, np.mean([ytick_3b, ytick_m1])])
        ax.set_xticklabels(['3a', '', '3b', '', 'Motor'])
        ax.set_yticklabels(['3a', '', '3b', '', 'Motor'])
        ax.set_title(title, fontsize=plot_params.axis_fontsize)
        ax.set_ylabel('Target Unit', fontsize=plot_params.axis_fontsize)
        ax.set_xlabel('Input Unit' , fontsize=plot_params.axis_fontsize)
        plt.show()
        
        fig.savefig(os.path.join(plots, f'functional_network_{title.replace(" ", "_")}.png'), bbox_inches='tight', dpi=plot_params.dpi)
    
        plt.hist(network_copy.flatten(), bins = 30)
        plt.show()
        
    return cmin, cmax
    # np.correlate(FN[0].flatten(), FN[1].flatten())

def sig_tests(unit_info, key_test, key_full, alternative='greater', unit_info_reduced = None):
    
    if unit_info_reduced is None:
        nFull = np.sum(unit_info[key_full] > unit_info[key_test])
        nUnits = np.shape(unit_info)[0]
        
        sign_test = binomtest(nFull, nUnits, p = 0.5, alternative=alternative)
        
        ttest_paired = ttest_rel(unit_info[key_full], unit_info[key_test], alternative=alternative)

    else:
        nPathlet = np.sum(unit_info.pathlet_AUC > unit_info_reduced.pathlet_AUC)
        nUnits = np.shape(unit_info)[0]
        sign_test = binomtest(nPathlet, nUnits, p = 0.5, alternative=alternative)
        ttest_paired = ttest_rel(unit_info.pathlet_AUC, unit_info_reduced.pathlet_AUC, alternative=alternative)

    return sign_test, ttest_paired

def plot_weights_versus_interelectrode_distances(FN, electrode_distances, FN_key = 'split_reach_FNs', ymin=None, ymax=None):
    
    fig, ax = plt.subplots(figsize=plot_params.weights_by_distance_figsize, dpi=plot_params.dpi)

    if FN_key == 'split_reach_FNs': 
        reach_labels = ['Even', 'Odd']
        weights_df = pd.DataFrame()
        for weights, reach_label in zip(FN, reach_labels):  
        
            tmp_df = pd.DataFrame(data = zip(weights.flatten(), electrode_distances.flatten(), [reach_label]*weights.size), 
                                  columns = ['Wji', 'Distance', 'Reaches'])
            weights_df = pd.concat((weights_df, tmp_df), axis=0, ignore_index=True)
            
        # sns.lineplot(ax = ax, data=df, x='Distance', y='Wij', err_style="bars", errorbar=('ci', 95))
        sns.lineplot(ax = ax, data=weights_df, x='Distance', y='Wji', hue='Reaches', err_style="bars", errorbar='se')
        ax.set_ylabel('Wji %s sem' % '\u00B1', fontsize = plot_params.axis_fontsize)
        ax.set_xlabel('Inter-Unit Distance (%sm)' % '\u03bc', fontsize = plot_params.axis_fontsize)

        if ymin:
            ax.set_ylim(ymin, ymax)
        else:
            ymin, ymax = ax.get_ylim()

        plt.show()

    elif FN_key == 'spontaneous_FN':
        reach_label = 'Spontaneous'
        weights = FN
        weights_df = pd.DataFrame()
        
        weights_df = pd.DataFrame(data = zip(weights.flatten(), electrode_distances.flatten(), [reach_label]*weights.size), 
                                  columns = ['Wji', 'Distance', 'Reaches'])
            
        # sns.lineplot(ax = ax, data=df, x='Distance', y='Wij', err_style="bars", errorbar=('ci', 95))
        sns.lineplot(ax = ax, data=weights_df, x='Distance', y='Wji', hue='Reaches', err_style="bars", errorbar='se')
        ax.set_ylabel('Wji %s sem' % '\u00B1', fontsize = plot_params.axis_fontsize)
        ax.set_xlabel('Inter-Unit Distance (%sm)' % '\u03bc', fontsize = plot_params.axis_fontsize)

        if ymin:
            ax.set_ylim(ymin, ymax)
        else:
            ymin, ymax = ax.get_ylim()

        plt.show()    
        
    fig.savefig(os.path.join(plots, 'weights_by_distance_for_%s.png' % FN_key), bbox_inches='tight', dpi=plot_params.dpi)
    
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
    
    idx_m1 = np.where(units_res['cortical_area'] == 'M1')[0]
    idx_3a = np.where(units_res['cortical_area'] == '3a')[0]
    idx_3b = np.where(units_res['cortical_area'] == '3b')[0]
    
    in_weights    = []
    in_weights_m1 = []
    in_weights_3a = []
    in_weights_3b = []
    out_weights   = []
    for unit_idx in units_res.index:
        if FN.ndim == 3:
            w_in  = (FN[0, unit_idx].sum() + FN[1, unit_idx].sum())/2 / FN.shape[-1]
            w_out = (FN[0, :, unit_idx].sum() + FN[1, :, unit_idx].sum())/2 / FN.shape[-1]
            
            w_in_m1 = (FN[0, unit_idx, idx_m1].sum() + FN[1, unit_idx, idx_m1].sum())/2 / len(idx_m1)
            w_in_3a = (FN[0, unit_idx, idx_3a].sum() + FN[1, unit_idx, idx_3a].sum())/2 / len(idx_3a)
            w_in_3b = (FN[0, unit_idx, idx_3b].sum() + FN[1, unit_idx, idx_3b].sum())/2 / len(idx_3b)
        else:
            w_in  = FN[unit_idx].sum()
            w_out = FN[:, unit_idx].sum()
        in_weights.append(w_in)
        in_weights_m1.append(w_in_m1)
        in_weights_3a.append(w_in_3a)
        in_weights_3b.append(w_in_3b)
        
        out_weights.append(w_out)
    
    tmp_df = pd.DataFrame(data = zip(in_weights, out_weights, in_weights_m1, in_weights_3a, in_weights_3b),
                          columns = ['W_in',     'W_out',     'W_in_m1',     'W_in_3a',     'W_in_3b'],
                          index = units_res.index)

    
    units_res = pd.concat((units_res, tmp_df), axis = 1)
    
    return units_res

def evaluate_effect_of_network_shuffles(lead_lag_key, comparison_model, kin_only_model=None, all_samples=False, targets = None, ylim=(0,50)):
    
    percentPattern       = re.compile('[0-9]{1,3}_percent')
    shuffleModePattern   = re.compile('shuffled_[a-zA-Z]*')
    shuffleMetricPattern = re.compile('by_[a-zA-Z]*') 
    
    results_key = 'trainAUC'
    
    comparison_all_units_auc = results_dict[lead_lag_key]['model_results'][comparison_model][results_key].copy()
    units_res_tmp = results_dict[params.best_lead_lag_key]['all_models_summary_results']

    if kin_only_model is not None:
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
    for model_key in results_dict[lead_lag_key]['model_results'].keys():
        if 'shuffled' in model_key and comparison_model == model_key.split('_shuffled')[0]:
            results_key = 'trainAUC'
            percent     = int(re.findall(percentPattern, model_key)[0].split('_percent')[0])
            shuf_mode   = re.findall(shuffleModePattern, model_key)[0].split('shuffled_')[-1]
            shuf_metric = re.findall(shuffleMetricPattern, model_key)[0].split('by_')[-1]
            
            if percent in [1]:
                continue
            
            shuffle_auc = results_dict[lead_lag_key]['model_results'][model_key][results_key].copy()
            
            
            comparison_auc = comparison_all_units_auc[target_idxs, :]
            shuffle_auc    = shuffle_auc             [target_idxs, :]
            if kin_only_model is not None:
                kin_only_auc   = kin_only_all_units_auc  [target_idxs, :]                
                auc_loss    = np.divide(comparison_auc - shuffle_auc, comparison_auc - kin_only_auc) * 100
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
                        
                tmp_df = pd.DataFrame(data = zip(auc_loss.flatten(), unit_list, sample_list, percent_list, mode_list, metric_list), 
                                      columns = ['auc_loss (%)', 'unit', 'sample', 'percent', 'mode', 'metric'])
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

    train_auc_df = pd.concat((train_auc_df, tmp_df_diff), axis = 0, ignore_index=True)    

    fig = sns.catplot(data = train_auc_df, x='percent', y='auc_loss (%)', col='mode', hue='metric', kind='point', legend=True, errorbar='se')
    fig.set_titles('Shuffled {col_name}:' +  f' {comparison_model}, {target_string}')
    fig.set(ylim=ylim)
    fig.set_axis_labels('Percent of Weights Shuffled', 'AUC Percent Loss')
    fig.savefig(os.path.join(plots, f'shuffled_network_auc_loss_summary_figure_{comparison_model}_{target_string}.png'), bbox_inches='tight', dpi=plot_params.dpi)    

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

def parse_properties_of_FN_subsets(units_res, FN, electrode_distances, FN_key = 'split_reach_FNs', subset_idxs = None, subset_types=None, subset_basis=['reach_specific'], tune = ('traj_avgPos_auc', 0.6), source_props=None):
    
    # if FN.ndim < 3:
        # FN = np.expand_dims(FN, axis = 0)
    FN_mean = FN.mean(axis=0) if FN.ndim == 3 else FN.copy()
    
    if source_props is None:
        source_props = pd.DataFrame()
    
    for sub_type, sub_basis in product(subset_types, subset_basis):
        
        if 'non_' in sub_basis:
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

def plot_distributions_after_source_props(units_res, electrode_distances, 
                                          traj_corr_df, FN_sets = [], subset_idxs = None, 
                                          sub_type='both', subset_basis=['reach_specific'], good_only=False):
    
    weights_df = pd.DataFrame()
    auc_df = pd.DataFrame()
    for sub_basis, FN_set in product(subset_basis, FN_sets):
        
        sub_idxs = subset_idxs.copy()
        
        if 'non_' in sub_basis:
            sub_idxs = np.setdiff1d(np.array(range(units_res.shape[0])), sub_idxs)
        elif sub_basis == 'original':
            sub_idxs = np.array(range(units_res.shape[0]))
        
        FN_key = FN_set[0]
        FN_tmp = FN_set[1]
        
        FN_mean = FN_tmp.mean(axis=0) if FN_tmp.ndim == 3 else FN_tmp.copy()
        
        if sub_type == 'both':
            targets, sources, sub_FN = sub_idxs, sub_idxs, FN_mean[np.ix_(sub_idxs, sub_idxs)]
            units_res_subset_units = units_res.loc[sources, :]  
        elif sub_type == 'target':
            targets, sources, sub_FN = sub_idxs, np.arange(FN_mean.shape[1]), FN_mean[np.ix_(sub_idxs, range(FN_mean.shape[1]))]
            units_res_subset_units = units_res.loc[targets, :]
        elif sub_type == 'source':
            targets, sources, sub_FN = np.arange(FN_mean.shape[0]), sub_idxs, FN_mean[np.ix_(range(FN_mean.shape[0]), sub_idxs)]
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
        
        sub_corr_array = np.full_like(sub_FN, 0)
        for i, j, corr in zip(sub_corr_i, sub_corr_j, sub_correlations):
            sub_corr_array[i, j] = corr
        sub_corr_array += sub_corr_array.transpose()
        
        if good_only:
            units_res_subset_units = units_res_subset_units.loc[units_res_subset_units['quality'] == 'good', :]  
            good_targets = np.where(units_res_targets['quality'] == 'good')[0]
            good_sources = np.where(units_res_sources['quality'] == 'good')[0]
            units_res_targets = units_res_targets.iloc[good_targets, :]
            units_res_sources = units_res_sources.iloc[good_sources, :]
            sub_FN = sub_FN[good_targets, good_sources]
            sub_corr_array = sub_corr_array[good_targets, good_sources]
            
        tmp_df = pd.DataFrame(data=zip(sub_FN.flatten(), 
                                       np.tile(units_res_sources['cortical_area'], sub_FN.shape[0]),
                                       np.repeat(FN_key, sub_FN.size),
                                       np.repeat(sub_basis, sub_FN.size),
                                       sub_corr_array.flatten()), 
                              columns=['Wji', 'input_area', 'FN_key', 'Units Subset', 'pearson_r'])
        tmp_df = tmp_df.loc[tmp_df['Wji'] != 0, :]
        tmp_df['pearson_r_squared'] = tmp_df['pearson_r']**2
        weights_df = pd.concat((weights_df, tmp_df))

        if FN_key == FN_sets[0][0]:
            tmp_auc_df = pd.DataFrame(data=zip(units_res_subset_units['traj_avgPos_auc'],
                                               units_res_subset_units['cortical_area'],
                                               np.repeat(sub_basis, units_res_subset_units.shape[0])),
                                      columns=['Kinematics_AUC', 'cortical_area', 'Units Subset'])
            auc_df = pd.concat((auc_df, tmp_auc_df))
    
    for sub_basis in np.unique(weights_df['Units Subset']):
        fig, ax = plt.subplots()
        sns.kdeplot(ax=ax, data=weights_df.loc[(weights_df['Units Subset'] == sub_basis), :], 
                    x='Wji', hue='FN_key', common_norm=False, bw_adjust=0.4, cumulative=True)
        ax.set_title(sub_basis)
        plt.show()
        fig.savefig(os.path.join(plots, f'cumDist_Wji_{sub_basis}_huekey_FNkey_.png'), bbox_inches='tight', dpi=plot_params.dpi)    

    for FN_key in np.unique(weights_df['FN_key']):
        fig, ax = plt.subplots()
        sns.kdeplot(ax=ax, data=weights_df.loc[(weights_df['FN_key'] == FN_key), :], 
                    x='Wji', hue='Units Subset', common_norm=False, bw_adjust=0.4, cumulative=True)
        ax.set_title(FN_key)
        plt.show()
        fig.savefig(os.path.join(plots, f'cumDist_Wji_{FN_key}_huekey_UnitsSubset.png'), bbox_inches='tight', dpi=plot_params.dpi)    
        
    fig, ax = plt.subplots()
    sns.kdeplot(ax=ax, data=auc_df, 
                x='Kinematics_AUC', hue='Units Subset', common_norm=False, bw_adjust=0.4, cumulative=True)
    # ax.set_title()
    plt.show()        
    fig.savefig(os.path.join(plots, 'cumDist_KinAUC_huekey_UnitsSubset.png'), bbox_inches='tight', dpi=plot_params.dpi)    


    fig, ax = plt.subplots()
    sns.kdeplot(ax=ax, data=auc_df, 
                x='Kinematics_AUC', hue='Units Subset', common_norm=False, bw_adjust=0.4, cumulative=False)
    # ax.set_title()
    plt.show()        

    fig, ax = plt.subplots()
    sns.kdeplot(ax=ax, data=weights_df, 
                x='pearson_r', hue='Units Subset', common_norm=False, bw_adjust=0.4, cumulative=False)
    # ax.set_title()
    plt.show()    
    fig.savefig(os.path.join(plots, 'distribution_pearson_r_huekey_UnitsSubset.png'), bbox_inches='tight', dpi=plot_params.dpi)    
    
    fig, ax = plt.subplots()
    sns.kdeplot(ax=ax, data=weights_df, 
                x='pearson_r_squared', hue='Units Subset', common_norm=False, bw_adjust=0.4, cumulative=False)
    # ax.set_title()
    plt.show()    
    fig.savefig(os.path.join(plots, 'distribution_pearson_r_squared_huekey_UnitsSubset.png'), bbox_inches='tight', dpi=plot_params.dpi)    

    fig = sns.relplot(data=weights_df, x='pearson_r', y='Wji', col='Units Subset', kind='scatter')
    plt.show() 
    fig.savefig(os.path.join(plots, 'subnetwork_pearson_r_squared_vs_wji_colkey_UnitsSubset.png'), bbox_inches='tight', dpi=plot_params.dpi)    
    
    
    return weights_df, auc_df

    # fig, ax = plt.subplots()
    # sns.kdeplot(ax=ax, data=weights_df.loc[(weights_df['Units Subset'] == 'non_reach_specific'), :], 
    #             x='Wji', hue='FN_key', common_norm=False, bw_adjust=0.4)
    # ax.set_title('non_reach_specific')
    # plt.show()

    # fig, ax = plt.subplots()
    # sns.kdeplot(ax=ax, data=weights_df.loc[(weights_df['Units Subset'] == 'normal'), :], 
    #             x='Wji', hue='FN_key', common_norm=False, bw_adjust=0.4)
    # ax.set_title('non_reach_specific')
    # plt.show()
    
    # constant_pair = ('FN_key', 'split_reach_FNs')
    # for col in weights_df.columns[(weights_df.columns != 'Wji') & (weights_df.columns != constant_pair[0])]:
    #     fig, ax = plt.subplots()
    #     sns.kdeplot(ax=ax, data=weights_df.loc[weights_df[constant_pair[0]] == constant_pair[1], :], x='Wji', hue=col, common_norm=False, bw_adjust=0.4)
    #     ax.set_title(constant_pair[1])
    #     plt.show()
    
    # constant_pair = ('FN_key', 'spontaneous_FN')
    # for col in weights_df.columns[(weights_df.columns != 'Wji') & (weights_df.columns != constant_pair[0])]:
    #     fig, ax = plt.subplots()
    #     sns.kdeplot(ax=ax, data=weights_df.loc[weights_df[constant_pair[0]] == constant_pair[1], :], x='Wji', hue=col, common_norm=False, bw_adjust=0.4)
    #     ax.set_title(constant_pair[1])
    #     plt.show()
        
def feature_correlation_plot(units_res, x_key, y_key, hue_key=None, col_key=None):
    
    if 'W_in' in x_key and marmcode=='TY':
        xmin, xmax = 0, 0.12
        xrange = units_res[x_key].max() - units_res[x_key].min() 
        xmin, xmax = units_res[x_key].min() - xrange*.1, units_res[x_key].max() + xrange*.1

    else:
        xrange = units_res[x_key].max() - units_res[x_key].min() 
        xmin, xmax = units_res[x_key].min() - xrange*.1, units_res[x_key].max() + xrange*.1    
    yrange = units_res[y_key].max() - units_res[y_key].min() 
    ymin, ymax = units_res[y_key].min()-yrange*.1, units_res[y_key].max()+yrange*.1

    
    if col_key is not None:
        fig = sns.relplot(data = units_res, x=x_key, y=y_key, 
                          col=col_key, kind='scatter', legend=True, 
                          height = plot_params.feature_corr_figSize[1], 
                          aspect = plot_params.feature_corr_figSize[0]/plot_params.feature_corr_figSize[1])
        for ax in fig.axes[0]:
            col_value = ax.title.get_text().split(' = ')[-1]
            tmp_units_res = units_res.loc[units_res[col_key] == col_value, [x_key, y_key]]
            slope, intercept, r, p, stderr = linregress(tmp_units_res[x_key], tmp_units_res[y_key])
            line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'
            ax.plot(tmp_units_res[x_key], intercept + slope * tmp_units_res[x_key], label=line)
            ax.legend(loc='upper center')
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
        plot_name = f'feature_correlation_{x_key}_{y_key}_columns_{col_key}.png'
    else:
        fig, ax = plt.subplots(figsize = plot_params.feature_corr_figSize, dpi = plot_params.dpi)
        try:
            sns.scatterplot(ax = ax, data = units_res, x = x_key, y = y_key, 
                            hue = hue_key, s = 20, legend=True)
        except:
            sns.scatterplot(ax = ax, data = units_res, x = x_key, y = y_key, 
                            s = 20, legend=False)        
        slope, intercept, r, p, stderr = linregress(units_res[x_key], units_res[y_key])
        line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'
        ax.plot(units_res[x_key], intercept + slope * units_res[x_key], label=line)
        ax.legend(bbox_to_anchor=(-0.1, 1.1), loc='upper left', borderaxespad=0)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        plot_name = f'feature_correlation_{x_key}_{y_key}.png'
    
    # sns.despine(fig=fig)
    plt.show()
    fig.savefig(os.path.join(plots, plot_name), bbox_inches='tight', dpi=plot_params.dpi)
    
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
    
def plot_pathlet(posTraj_mean, posTraj_samples, lead_lag_key, model, avgPos_mean = None, unit_selector = 'max', numToPlot = 5, unitsToPlot = None, axlims = None):
    
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
        
        fig.savefig(os.path.join(plots, 'unit_%d_pathlet.png' % unit), bbox_inches='tight', dpi=plot_params.dpi)

    
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
                                                reach_specific_units = None, nplots=5):
    
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
                
                fig.savefig(os.path.join(plots, f'corr_pair_pathlets_{pair[0]}_{pair[1]}.png'), bbox_inches='tight', dpi=plot_params.dpi)

    
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

    fig, ax = plt.subplots(figsize = (4, 4), dpi = plot_params.dpi)
    sns.scatterplot(ax = ax, data = df, x = 'Pearson_corr', y = 'Wji', s = 20, legend=True) 
    plt.show()
    # fig.savefig(os.path.join(plots, 'wji_vs_pearson_r.png'), bbox_inches='tight', dpi=plot_params.dpi)
    
    # fig, ax = plt.subplots(figsize = (4, 4), dpi = plot_params.dpi)
    # sns.scatterplot(ax = ax, data = df, x = 'r_squared', y = 'Wji', s = 20, legend=True) 
    # plt.show()
    # fig.savefig(os.path.join(plots, 'wji_vs_pearson_rsquare.png'), bbox_inches='tight', dpi=plot_params.dpi)
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
    # fig.savefig(os.path.join(plots, 'pearson_r_vs_connection.png'), bbox_inches='tight', dpi=plot_params.dpi)

    # fig, ax = plt.subplots(figsize = (4, 4), dpi = plot_params.dpi)
    # sns.pointplot(ax=ax, data = df, x = 'Connection', y = 'r_squared', color='black', errorbar='se')
    # plt.show()
    # fig.savefig(os.path.join(plots, 'pearson_rsquare_vs_connection.png'), bbox_inches='tight', dpi=plot_params.dpi)

    # fig, ax = plt.subplots(figsize = (4, 4), dpi = plot_params.dpi)
    # sns.pointplot(ax=ax, data = df, x = 'Connection', y = 'Wji', color='black', errorbar='se')
    # plt.show()
    # fig.savefig(os.path.join(plots, 'wji_vs_connection.png'), bbox_inches='tight', dpi=plot_params.dpi)
    
    return df
    
if __name__ == "__main__":
    
    with open(pkl_infile, 'rb') as f:
        results_dict = dill.load(f)  
        
    with NWBHDF5IO(nwb_infile, 'r') as io:
        nwb = io.read()

        FN = nwb.scratch[params.FN_key].data[:] 
        spontaneous_FN = nwb.scratch['spontaneous_FN'].data[:]

            
    summarize_model_results(units=None, lead_lag_keys = params.best_lead_lag_key)
    
    units_res = results_dict[params.best_lead_lag_key]['all_models_summary_results']
    units_res = add_in_weight_to_units_df(units_res, FN.copy())
    units_res = add_neuron_classifications(units_res)
    
    by_class, by_area, by_class_and_area = get_grouped_means(units_res)
    
    electrode_distances = get_interelectrode_distances_by_unit(units_res, array_type='utah')

    feature_correlation_plot(units_res, 'W_in_3a', 'traj_avgPos_auc', col_key=None)    
    feature_correlation_plot(units_res, 'W_in_3b', 'traj_avgPos_auc', col_key=None)    
    feature_correlation_plot(units_res, 'W_in_m1', 'traj_avgPos_auc', col_key=None)    
    feature_correlation_plot(units_res, 'W_in_3a', 'traj_avgPos_auc', col_key='cortical_area')
    feature_correlation_plot(units_res, 'W_in_3b', 'traj_avgPos_auc', col_key='cortical_area')
    feature_correlation_plot(units_res, 'W_in_m1', 'traj_avgPos_auc', col_key='cortical_area')

    feature_correlation_plot(units_res, 'W_in', 'traj_avgPos_auc', col_key='cortical_area')
    feature_correlation_plot(units_res, 'W_in', 'traj_avgPos_auc')


    feature_correlation_plot(units_res, 'snr', 'W_in', col_key=None)    
    feature_correlation_plot(units_res, 'fr', 'W_in', col_key=None)    
    feature_correlation_plot(units_res, 'snr', 'traj_avgPos_reach_FN_auc', col_key=None)    
    feature_correlation_plot(units_res, 'fr', 'traj_avgPos_reach_FN_auc', col_key=None)    
    feature_correlation_plot(units_res, 'fr', 'traj_avgPos_auc', col_key=None)    
    feature_correlation_plot(units_res, 'snr', 'traj_avgPos_auc', col_key=None)    

    feature_correlation_plot(units_res.loc[units_res['quality'] == 'good'], 'fr', 'traj_avgPos_auc', col_key=None)

    feature_correlation_plot(units_res.loc[units_res['quality'] == 'good'], 'fr', 'traj_avgPos_auc', col_key='cortical_area')
    feature_correlation_plot(units_res.loc[units_res['quality'] == 'good'], 'fr', 'traj_avgPos_reach_FN_auc', col_key='cortical_area')
    
    fig, ax = plt.subplots(figsize=(4,2), dpi = plot_params.dpi)
    sns.kdeplot(ax=ax, data=units_res.loc[units_res['neuron_type'] != 'unclassified'], x='fr', hue='neuron_type', bw_adjust=.4, common_norm=False)
    plt.show()
    fig.savefig(os.path.join(plots, 'fr_kdeplot_split_by_class'), bbox_inches='tight', dpi=plot_params.dpi)

    fig, ax = plt.subplots(figsize=(4,2), dpi = plot_params.dpi)
    sns.kdeplot(ax=ax, data=units_res, x='fr', hue='cortical_area', bw_adjust=.4, common_norm=False)
    plt.show()
    fig.savefig(os.path.join(plots, 'fr_kdeplot_split_by_area'), bbox_inches='tight', dpi=plot_params.dpi)

    fig, ax = plt.subplots(figsize=(4,2), dpi = plot_params.dpi)
    sns.kdeplot(ax=ax, data=units_res, x='snr', hue='cortical_area', bw_adjust=.4, common_norm=False)
    plt.show()
    fig.savefig(os.path.join(plots, 'snr_kdeplot_split_by_area'), bbox_inches='tight', dpi=plot_params.dpi)
    
    fig, ax = plt.subplots(figsize=(4,2), dpi = plot_params.dpi)
    sns.kdeplot(ax=ax, data=units_res, x='traj_avgPos_reach_FN_auc', hue='neuron_type', bw_adjust=.4, common_norm=True)
    plt.show()
    fig.savefig(os.path.join(plots, 'traj_avgPos_reach_FN_auc_kdeplot_split_by_class'), bbox_inches='tight', dpi=plot_params.dpi)

    fig, ax = plt.subplots(figsize=(4,2), dpi = plot_params.dpi)
    sns.kdeplot(ax=ax, data=units_res, x='traj_avgPos_reach_FN_auc', hue='cortical_area', bw_adjust=.4, common_norm=False)
    plt.show()
    fig.savefig(os.path.join(plots, 'traj_avgPos_reach_FN_auc_kdeplot_split_by_area'), bbox_inches='tight', dpi=plot_params.dpi)
    
    
    fig, ax = plt.subplots(figsize=(4,2), dpi = plot_params.dpi)
    sns.kdeplot(ax=ax, data=units_res.loc[units_res['neuron_type'] != 'unclassified'], x='snr', hue='neuron_type', bw_adjust=.4, common_norm=True)
    plt.show()
    fig.savefig(os.path.join(plots, 'snr_kdeplot_split_by_class'), bbox_inches='tight', dpi=plot_params.dpi)
    
    cortical_area_idxs = grab_cortical_area_FN_idxs(units_res)
    # target_idxs = cortical_area_idxs['motor']
    # source_idxs = cortical_area_idxs['motor']
    target_idxs = None
    source_idxs = None

    percent = 25
    shuffle_set_0 = identify_shuffle_set_by_strength(FN[0], percent, target_idxs = target_idxs, source_idxs = source_idxs)
    shuffle_set_1 = identify_shuffle_set_by_strength(FN[1], percent, target_idxs = target_idxs, source_idxs = source_idxs)
    
    shuffle_sets = [shuffle_set_0, shuffle_set_1]
    source_props = parse_properties_of_shuffled_sources(units_res, percent, shuffle_sets, electrode_distances, source_props = None)

    evaluate_effect_of_network_shuffles(lead_lag_key = params.best_lead_lag_key, comparison_model = 'traj_avgPos_reach_FN', kin_only_model=None, all_samples=False, targets=None, ylim=(0,30))

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
    
    cmin, cmax = plot_functional_networks(FN, units_res, FN_key = params.FN_key)
    plot_functional_networks(spontaneous_FN, units_res, FN_key ='spontaneous_FN', cmin=cmin, cmax=cmax)

    # ymin, ymax = plot_weights_versus_interelectrode_distances(FN, electrode_distances, FN_key = params.FN_key)
    # plot_weights_versus_interelectrode_distances(spontaneous_FN, electrode_distances, FN_key ='spontaneous_FN', ymin=ymin, ymax=ymax)

    plot_model_auc_comparison   (units_res, 'traj_avgPos_auc', 'traj_avgPos_reach_FN_auc', minauc = 0.45, hue_key='W_in', style_key='neuron_type', col_key = 'cortical_area', targets = None)

    plot_model_auc_comparison   (units_res, 'traj_avgPos_auc', 'traj_avgPos_reach_FN_auc', minauc = 0.45, hue_key='W_in', style_key='neuron_type', targets = None, col_key = None)
    # sign_test, ttest = sig_tests(units_res, 'traj_avgPos_auc', 'traj_avgPos_reach_FN_auc', alternative='greater')
    # print(sign_test)
    # plot_model_auc_comparison   (units_res, 'reach_FN_auc', 'traj_avgPos_reach_FN_auc', minauc = 0.45, hue_key='W_in', style_key='cortical_area', targets = None)
    # sign_test, ttest = sig_tests(units_res, 'reach_FN_auc', 'traj_avgPos_reach_FN_auc', alternative='greater')
    # print(sign_test)
    # plot_model_auc_comparison   (units_res, 'reach_FN_auc', 'traj_avgPos_auc', minauc = 0.45, hue_key='W_in', style_key='cortical_area', targets = None)
    # sign_test, ttest = sig_tests(units_res, 'reach_FN_auc', 'traj_avgPos_auc', alternative='greater')
    # print(sign_test)
    # plot_model_auc_comparison   (units_res, 'traj_avgPos_spontaneous_FN_auc', 'traj_avgPos_reach_FN_auc', minauc = 0.45, hue_key='W_in', style_key='cortical_area', targets = None)
    # sign_test, ttest = sig_tests(units_res, 'traj_avgPos_spontaneous_FN_auc', 'traj_avgPos_reach_FN_auc', alternative='less')
    # print(sign_test)
    # plot_model_auc_comparison   (units_res, 'spontaneous_FN_auc', 'reach_FN_auc', minauc = 0.45, hue_key='W_in', style_key='cortical_area', targets = None)
    # sign_test, ttest = sig_tests(units_res, 'spontaneous_FN_auc', 'reach_FN_auc', alternative='two-sided')
    # print(sign_test)

    # plot_model_auc_comparison   (units_res, 'spont_train_reach_test_FN_auc', 'spontaneous_FN_auc', minauc = 0.45, style_key='cortical_area', targets = None)
    # plot_model_auc_comparison   (units_res, 'reach_train_spont_test_FN_auc', 'reach_FN_auc', minauc = 0.45, style_key='cortical_area', targets = None)
    # plot_model_auc_comparison   (units_res, 'spont_train_reach_test_FN_auc', 'reach_train_spont_test_FN_auc', minauc = 0.45, style_key='cortical_area',targets = None)
    # plot_model_auc_comparison   (units_res, 'reach_train_spont_test_FN_auc', 'reach_FN_auc', minauc = 0.45, hue_key='W_in', style_key='neuron_type', col_key = 'cortical_area', targets = None)
    # plot_model_auc_comparison   (units_res, 'spont_train_reach_test_FN_auc', 'reach_FN_auc', minauc = 0.45, hue_key='W_in', style_key='neuron_type', col_key = 'cortical_area', targets = None)
    # plot_model_auc_comparison   (units_res, 'reach_train_spont_test_FN_auc', 'spontaneous_FN_auc', minauc = 0.45, hue_key='W_in', style_key='neuron_type', col_key = 'cortical_area', targets = None)

    # plot_model_auc_comparison   (units_res, 'reach_train_spont_test_FN_auc', 'reach_FN_auc', minauc = 0.45, hue_key='W_in', style_key='neuron_type', targets = None)
    plot_model_auc_comparison   (units_res, 'spont_train_reach_test_FN_auc', 'reach_FN_auc', minauc = 0.45, hue_key='W_in', style_key='neuron_type', targets = None)
    # plot_model_auc_comparison   (units_res, 'reach_train_spont_test_FN_auc', 'spontaneous_FN_auc', minauc = 0.45, hue_key='W_in', style_key='neuron_type', targets = None)

    # plot_model_auc_comparison   (units_res, 'traj_avgPos_reach_train_spont_test_FN_auc', 'traj_avgPos_reach_FN_auc', minauc = 0.45, hue_key='W_in', style_key='neuron_type', targets = None)
    plot_model_auc_comparison   (units_res, 'traj_avgPos_spont_train_reach_test_FN_auc', 'traj_avgPos_reach_FN_auc', minauc = 0.45, hue_key='W_in', style_key='neuron_type', targets = None)
    # plot_model_auc_comparison   (units_res, 'traj_avgPos_reach_train_spont_test_FN_auc', 'traj_avgPos_spontaneous_FN_auc', minauc = 0.45, hue_key='W_in', style_key='neuron_type', targets = None)


    # for area in [None, 'motor', 'sensory']:
    #     plot_model_auc_comparison   (units_res, 'traj_avgPos_spont_train_reach_test_FN_auc', 'traj_avgPos_spontaneous_FN_auc', minauc = 0.45,targets = area)
    #     plot_model_auc_comparison   (units_res, 'traj_avgPos_reach_train_spont_test_FN_auc', 'traj_avgPos_reach_FN_auc', minauc = 0.45,targets = area)
    #     plot_model_auc_comparison   (units_res, 'traj_avgPos_spont_train_reach_test_FN_auc', 'traj_avgPos_reach_train_spont_test_FN_auc', minauc = 0.45,targets = area)

    # plot_model_auc_comparison   (units_res, 'traj_avgPos_intra_reach_FN', 'traj_avgPos_reach_FN', minauc = 0.45, hue_key='W_in', style_key='neuron_type', col_key = 'cortical_area', targets = None)
    # plot_model_auc_comparison   (units_res, 'traj_avgPos_intra_reach_FN', 'traj_avgPos_inter_reach_FN', minauc = 0.45, hue_key='W_in', style_key='neuron_type', col_key = 'cortical_area', targets = None)
    # plot_model_auc_comparison   (units_res, 'intra_reach_FN', 'inter_reach_FN', minauc = 0.45, hue_key='W_in', style_key='neuron_type', col_key = 'cortical_area', targets = None)

    diff_df = compute_performance_difference_by_unit(units_res, 'traj_avgPos_spont_train_reach_test_FN_auc', 'traj_avgPos_reach_FN_auc')   
    reach_specific_units = diff_df.index[(diff_df.auc_diff > 0) & (diff_df.dist_from_unity > 0.04)]
    non_specific_units = np.setdiff1d(units_res.index, reach_specific_units)
    
    # diff_df = compute_performance_difference_by_unit(units_res, 'spont_train_reach_test_FN_auc', 'reach_FN_auc')   
    # reach_specific_units_FN_only = diff_df.index[(diff_df.auc_diff > 0) & (diff_df.dist_from_unity > 0.04)]
    
    posTraj_mean, velTraj_mean, posTraj_samples, velTraj_samples = compute_and_analyze_pathlets(params.best_lead_lag_key, 'traj_avgPos', numplots = None)
    traj_corr_df = compute_and_analyze_trajectory_correlations(units_res, posTraj_mean, velTraj_mean, 
                                                               electrode_distances, params.best_lead_lag_key, 
                                                               FN = FN, mode='concat', 
                                                               reach_specific_units = reach_specific_units, nplots=None)
    
    
    
    sns.kdeplot(data = traj_corr_df, x='r_squared', hue='both_reach_FN_dependent', bw_adjust=.4, common_norm=False)
    plt.show()
    plt.gcf().savefig(os.path.join(plots, 'fr_kdeplot_split_by_class'), bbox_inches='tight', dpi=plot_params.dpi)
    sns.kdeplot(data = traj_corr_df, x='r_squared', hue='target_reach_FN_dependent', bw_adjust=.4, common_norm=False)
    plt.show()
    sns.kdeplot(data = traj_corr_df, x='r_squared', hue='source_reach_FN_dependent', bw_adjust=.4, common_norm=False)
    plt.show()
    
    subset = 'both'
    plot_functional_networks(FN, units_res, FN_key = params.FN_key, cmin=cmin, cmax=cmax, subset_idxs = reach_specific_units, subset_type=subset)
    plot_functional_networks(spontaneous_FN, units_res, FN_key ='spontaneous_FN', cmin=cmin, cmax=cmax, subset_idxs = reach_specific_units, subset_type=subset)

    plot_functional_networks(FN, units_res, FN_key = params.FN_key, cmin=cmin, cmax=cmax, subset_idxs = non_specific_units, subset_type=subset)
    plot_functional_networks(spontaneous_FN, units_res, FN_key ='spontaneous_FN', cmin=cmin, cmax=cmax, subset_idxs = non_specific_units, subset_type=subset)

    reach_specific_FN_props = parse_properties_of_FN_subsets(units_res, FN, electrode_distances, 
                                                             FN_key = params.FN_key, 
                                                             subset_idxs = reach_specific_units, 
                                                             subset_types=['both', 'target', 'source'], 
                                                             subset_basis = ['reach_specific', 'non_reach_specific'], 
                                                             tune = (f'{params.primary_traj_model}_auc', params.tuned_auc_thresh),
                                                             source_props=None)

    reach_specific_FN_props = parse_properties_of_FN_subsets(units_res, spontaneous_FN, electrode_distances, 
                                                             FN_key = 'spontaneous', 
                                                             subset_idxs = reach_specific_units, 
                                                             subset_types=['both', 'target', 'source'], 
                                                             subset_basis = ['reach_specific', 'non_reach_specific'], 
                                                             tune = (f'{params.primary_traj_model}_auc', params.tuned_auc_thresh),
                                                             source_props=reach_specific_FN_props)
    
    weights_df, auc_df =plot_distributions_after_source_props(units_res, 
                                                              electrode_distances,
                                                              traj_corr_df,
                                                              FN_sets = [('split_reach_FNs', FN), ('spontaneous_FN', spontaneous_FN)], 
                                                              subset_idxs = reach_specific_units, 
                                                              sub_type='both', 
                                                              subset_basis=['reach_specific', 'non_reach_specific', 'original'],
                                                              good_only=False)

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