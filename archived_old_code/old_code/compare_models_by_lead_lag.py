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


class path:
    storage = r'Z:/marmosets/processed_datasets'
    # intermediate_save_path = r'C:\Users\daltonm\Documents\Lab_Files\encoding_model\intermediate_variable_storage'
    intermediate_save_path = r'C:\Users\Dalton\Documents\lab_files\analysis_encoding_model\intermediate_variable_storage'
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
    
    axis_fontsize = 12
    dpi = 300
    axis_linewidth = 2
    tick_length = 2
    tick_width = 1
    map_figSize = (7, 7)
    tick_fontsize = 18

     
def load_data():
    
    print('loading original spike and position/reach data, analog camera signals, and functional networks')
    
    spike_path = glob.glob(os.path.join(path.storage, 'formatted_spike_dir', '%s*.pkl' % path.date))
    spike_path = [f for f in spike_path if 'sleep' not in f][0]
    with open(spike_path, 'rb') as fp:
        spike_data = dill.load(fp)
    
    kin_path = glob.glob(os.path.join(path.storage, 'reach_and_trajectory_information', '%s*.pkl' % path.date))[0]
    with open(kin_path, 'rb') as fp:
        kinematics = dill.load(fp)

    analog_path = glob.glob(os.path.join(path.storage, 'analog_signal_and_video_frame_information/pickle_files', '%s*.pkl' % path.date))[0]
    with open(analog_path, 'rb') as fp:
        analog_and_video = dill.load(fp)

    FN_path = os.path.join(path.intermediate_save_path, 'FN_%s_fMI_10ms_bins_dict.pkl' % path.date)
    with open(FN_path, 'rb') as fp:
        # raster_list, FN = dill.load(fp)
        FN = dill.load(fp)

    return spike_data, kinematics, analog_and_video, FN[0]

def load_models():
    print('loading results from all models')
    
    spikeBins = math.modf(params.spkSampWin * 1e3)
    models_path = os.path.join(path.intermediate_save_path, '%dpt%d_ms_bins' % (spikeBins[1], spikeBins[0]*10))    
    results_paths = sorted(glob.glob(os.path.join(models_path, '%s_encoding_model_results_*%s*_shift_%d_PCAthresh_%d_norm_%s.pkl' % (path.date, 
                                                                                                                                     params.FN_source, 
                                                                                                                                     params.trajShift*1e3, 
                                                                                                                                     params.pca_var_thresh*100,
                                                                                                                                     params.normalize))))
    feature_paths = sorted(glob.glob(os.path.join(models_path, '%s_model_features_and_components_*%s*_shift_%d_PCAthresh_%d_norm_%s.pkl' % (path.date, 
                                                                                                                                            params.FN_source, 
                                                                                                                                            params.trajShift*1e3, 
                                                                                                                                            params.pca_var_thresh*100,
                                                                                                                                            params.normalize))))
    sampled_data_paths = sorted(glob.glob(os.path.join(models_path, '%s_model_trajectories_and_spikes*_shift_%d_with_network_lags.pkl' % (path.date, 
                                                                                                                                          params.trajShift*1e3))))

    results_list          = []
    unit_info_list        = []
    traj_features_list    = []
    network_features_list = []
    short_features_list   = []
    components_list       = []
    # sampled_data_list     = []
    lead_lag_list         = []
    for r_path, f_path, s_path in zip(results_paths, feature_paths, sampled_data_paths):    
        with open(r_path, 'rb') as f:
            models_results, unit_info = dill.load(f)

        # all_model_results = {'model_results' : [full_model_results, traj_model_results, 
        #                                         network_model_results, short_model_results, 
        #                                         shuffle_model_results],
        #                      'model_names'   : ['full', 'trajectory', 'network', 
        #                                         'velocity', 'shuffle']}
        # with open(r_path.split('_separatedResults')[0] + '.pkl', 'wb') as f:
        #     dill.dump([all_model_results, unit_info], f, recurse=True)  
        with open(f_path, 'rb') as f:
            traj_features, network_features, short_features, compsOut = dill.load(f)

        lead = int(re.findall(re.compile('lead_\d{1,3}'), r_path)[0].split('lead_')[-1])
        lag  = int(re.findall(re.compile('lag_\d{1,3}' ), r_path)[0].split('lag_' )[-1])
        results_list.append(models_results)
        unit_info_list.append(unit_info)
        traj_features_list.append(traj_features)
        network_features_list.append(network_features)
        short_features_list.append(short_features)
        components_list.append(compsOut)
        lead_lag_list.append((lead, lag))
    
    all_models_data = {'model_details'    : results_list,
                       'unit_info'        : unit_info_list,
                       'traj_features'    : traj_features_list,
                       'network_features' : network_features_list,
                       'short_features'   : short_features_list,
                       'components'       : components_list,
                       'lead_lag'         : lead_lag_list}
    
    return all_models_data

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
    results     = []
    model_names = []
    for idx, (key, name) in enumerate(zip(model_keys, corrected_model_names)):
        model_names.append(name)
        tmp_results = pd.DataFrame()
        for unit_info, lead_lag in zip(all_models_data['unit_info'], all_models_data['lead_lag']): 
            tmp_results['lead_%d_lag_%d' % (lead_lag[0], lead_lag[1])] = unit_info[key] 
        results.append(tmp_results)
    
    model_results_across_lags = {'model_name'    : model_names,
                                 'model_results' : results}     
    
    return model_results_across_lags

def find_optimal_lag_for_each_unit(model_results_across_lags, all_models_data):
    modelIdx = [idx for idx, name in enumerate(model_results_across_lags['model_name']) if name == 'trajectory'][0]
    traj_results = model_results_across_lags['model_results'][modelIdx]
    
    optimal_lead_lag = traj_results.idxmax(axis = 1)
    lead_pattern = re.compile('lead_\d{1,3}')
    lag_pattern  = re.compile('lag_\d{1,3}')
    optimal_lead_lag = [(int(re.findall(lead_pattern, leadlag)[0].split('lead_')[-1]), int(re.findall(lag_pattern, leadlag)[0].split('lag_')[-1])) 
                        for leadlag in optimal_lead_lag]
    optimal_traj_center = [(-1*ll[0] + ll[1]) // 2 for ll in optimal_lead_lag]
    
    
    #######################
    lead_lags = [(int(re.findall(lead_pattern, leadlag)[0].split('lead_')[-1]), int(re.findall(lag_pattern, leadlag)[0].split('lag_')[-1])) 
                        for leadlag in traj_results.columns]
    traj_lag_percent = [ll[1] / (ll[0] + ll[1]) for ll in lead_lags]
    numComps = [data.shape[0] for data in all_models_data['components']]
    for idx in range(len(traj_lag_percent)):
        if traj_lag_percent[idx] == 0:
            traj_lag_percent[idx] = -1*lead_lags[idx][0] / 3000
        elif traj_lag_percent[idx] == 1:
            traj_lag_percent[idx] = 1 + lead_lags[idx][1] / 3000

    # record std for filtering out poorly tuned units
    passed = []
    for plot_idx in range(traj_results.shape[0]):
        tmp_data = pd.DataFrame(data = zip(traj_lag_percent, 
                                           numComps, 
                                           traj_results.iloc[plot_idx, :].to_numpy(),
                                           range(len(traj_lag_percent))), 
                                columns = ['percent', 'nComps', 'auc', 'order'])
        tmp_data.sort_values(by = 'percent', inplace=True)
        tmp_data['plot_idx'] = range(tmp_data.shape[0])
    
        # std = np.std(tmp_data.auc)
        ind = np.argpartition(tmp_data.auc.to_numpy(), -5)[-5:]
        ind = ind[np.argsort(tmp_data.auc.to_numpy()[ind])]
        ind = ind[::-1]
        median_auc = np.percentile(tmp_data.auc, 50)
        max_diff = np.max(tmp_data.auc - median_auc)
        total_diff = np.sum(tmp_data.auc.to_numpy()[ind] - median_auc)
        # if max_diff > 0.012 and total_diff > 0.035 and np.max(tmp_data.auc) > 0.545 and (np.mean(abs(ind[:3] - np.mean(ind[:3]))) < 2.5 or abs(np.diff(ind[:2])) == 1) and abs(np.diff(ind[:2])) <= 4:
        if max_diff > 0.012 and total_diff > 0.035 and (np.mean(abs(ind[:3] - np.mean(ind[:3]))) < 2.5 or abs(np.diff(ind[:2])) == 1) and abs(np.diff(ind[:2])) <= 4:
            passed.append(True)
            continue
            fig, ax = plt.subplots()
            ax.plot(tmp_data.plot_idx, tmp_data.auc, 'ob')
            ax.plot(tmp_data.plot_idx, tmp_data.auc, '-r')
            # ax.plot(tmp_data.plot_idx, median_filter(tmp_data.auc, 3), '-r')
            ax.plot(tmp_data.plot_idx, np.repeat(median_auc, len(tmp_data.auc)), '--k')
            ax.set_ylim(0.5, 0.7)
            ax.set_title('%.3f, %.3f, %s, %.3f, %.3f' % (total_diff, max_diff, str(ind), np.mean(abs(ind[:3] - np.mean(ind[:3]))), abs(np.diff(ind[:2]))))
            plt.show()
        else:
            passed.append(False)
            continue
            fig, ax = plt.subplots()
            ax.plot(tmp_data.plot_idx, tmp_data.auc, '-ok')
            # ax.plot(tmp_data.plot_idx, median_filter(tmp_data.auc, 3), '-k')
            ax.plot(tmp_data.plot_idx, np.repeat(median_auc, len(tmp_data.auc)), '--k')
            ax.set_ylim(0.5, 0.7)
            ax.set_title('%.3f, %.3f, %s, %.3f, %.3f' % (total_diff, max_diff, str(ind), np.mean(abs(ind[:3] - np.mean(ind[:3]))), abs(np.diff(ind[:2]))))
            plt.show()
    ####################
    # comps_vs_auc = pd.DataFrame(data=zip(np.tile(numComps, (traj_results.shape[0],)), traj_results.to_numpy().flatten()),
    #                             columns = ['numComps', 'auc'])
    # fig,ax = plt.subplots()
    # sns.boxplot(data = comps_vs_auc, x = 'numComps', y='auc')
    
    
    
    
    ##################
    
    # columns_with_auc_or_pval = [idx for idx, col in enumerate(all_models_data['unit_info'][0].columns) if 'AUC' in col or '_p' in col]
    unit_info_optimal_lag = all_models_data['unit_info'][0].copy()
    unit_info_optimal_lag[['lead', 'lag', 'traj_center']] = np.full((unit_info_optimal_lag.shape[0], 3), np.nan)
    for idx, (opt_ll, opt_center, p) in enumerate(zip(optimal_lead_lag, optimal_traj_center, passed)):
        tmp_ll_idx = [ll_idx for ll_idx, ll in enumerate(all_models_data['lead_lag']) if ll == opt_ll][0]
        tmp_unit_info = all_models_data['unit_info'][tmp_ll_idx]
        columns_with_auc_or_pval = [idx for idx, col in enumerate(tmp_unit_info.columns) if 'AUC' in col or '_p' in col]
        unit_info_optimal_lag.iloc[idx, columns_with_auc_or_pval] = tmp_unit_info.iloc[idx, columns_with_auc_or_pval]
        unit_info_optimal_lag.at[unit_info_optimal_lag.index[idx], 'lead']    = opt_ll[0]
        unit_info_optimal_lag.at[unit_info_optimal_lag.index[idx], 'lag' ]    = opt_ll[1]
        unit_info_optimal_lag.at[unit_info_optimal_lag.index[idx], 'traj_center'] = opt_center
    
    unit_info_optimal_lag['passed_filters'] = passed
    
    return unit_info_optimal_lag

def compute_mean_model_performance(model_results_across_lags, percent = 0, percentile_mode = 'per_lag_set'):
    
    model_results_across_lags['mean_performance_by_lead_lag'] = [0]*len(model_results_across_lags['model_name'])
    for idx, results in enumerate(model_results_across_lags['model_results']):
        tmp_results = results.to_numpy().copy()
        if percentile_mode == 'across_lag_sets':
            tmp_results[tmp_results < np.percentile(tmp_results, percent)] = np.nan
        elif percentile_mode == 'per_lag_set':
            for col in range(tmp_results.shape[-1]):
                tmp_results[tmp_results[:, col] < np.percentile(tmp_results[:, col], percent), col] = np.nan
        model_results_across_lags['mean_performance_by_lead_lag'][idx] = pd.DataFrame(data=zip(np.nanmean(tmp_results, axis=0), np.nanstd(tmp_results, axis=0) / np.sqrt(tmp_results.shape[0])),
                                                                                      index=results.columns,
                                                                                      columns=['auc', 'SE'])    
    return model_results_across_lags

def plot_optimal_lag_on_channel_map(unit_info, spike_data, jitter_radius = .15, hueKey = 'traj_center', sizeKey = 'pathlet_AUC', minAUC = .55):
    
    rotated_map = spike_data['chan_map_ns6'].copy()
    # for j in range(rotated_map.shape[1]):
    #     rotated_map[:, j] = rotated_map[::-1, j]
    
    scatter_unit_info = unit_info.copy()
    scatter_unit_info = scatter_unit_info.loc[(scatter_unit_info.ttest_p < 0.05) & (scatter_unit_info.pathlet_AUC > minAUC) & (scatter_unit_info.passed_filters), :]
    scatter_unit_info['scatter_x'] = np.full((scatter_unit_info.shape[0],), np.nan)
    scatter_unit_info['scatter_y'] = np.full((scatter_unit_info.shape[0],), np.nan)
    for ch in np.unique(scatter_unit_info.ns6_elec_id):
        chan_clusters = scatter_unit_info.loc[scatter_unit_info.ns6_elec_id == ch, 'cluster_id']
        if len(chan_clusters) == 1:
            jitters = [(0, 0)]
        else:
            jitters = [(np.round(jitter_radius * np.cos(n*2*np.pi / len(chan_clusters)), 3), 
                        np.round(jitter_radius * np.sin(n*2*np.pi / len(chan_clusters)), 3)) for n in range(len(chan_clusters))]
        base_pos = np.where(rotated_map == ch)
        
        base_pos = [int(base_pos[1]), int(base_pos[0])]
        
        scatter_unit_info.loc[scatter_unit_info.ns6_elec_id == ch, 'scatter_x'] = [jitter[0] + base_pos[0] for jitter in jitters]
        scatter_unit_info.loc[scatter_unit_info.ns6_elec_id == ch, 'scatter_y'] = [jitter[1] + base_pos[1] for jitter in jitters]

        scatter_unit_info.loc[scatter_unit_info.ns6_elec_id == ch, 'center_x'] = base_pos[0]
        scatter_unit_info.loc[scatter_unit_info.ns6_elec_id == ch, 'center_y'] = base_pos[1]
    
    fig, ax = plt.subplots(figsize=params.map_figSize)
    sns.scatterplot(ax = ax, data = scatter_unit_info, x = 'scatter_x', y = 'scatter_y', 
                    size = sizeKey, hue = hueKey, style = "group", palette='seismic',
                    edgecolor="black")
    ax.vlines(np.arange(-0.5, 10.5, 1), -0.5, 9.5, colors='black')
    ax.hlines(np.arange(-0.5, 10.5, 1), -0.5, 9.5, colors='black')
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-0.5, 9.5)
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color('black')
    for axis in ['top','right']:
        ax.spines[axis].set_linewidth(0)
    ax.tick_params(width=0, length = 0, labelsize = 0)
    ax.set_xlabel('Lateral'  , fontsize = params.axis_fontsize, fontweight = 'bold')
    ax.set_ylabel('Posterior', fontsize = params.axis_fontsize, fontweight = 'bold')
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0)
    ax.set_title(hueKey)

    ax.grid(False)
    # for txt, x, y, scat_x, scat_y in zip(scatter_unit_info['ns6_elec_id'], scatter_unit_info['center_x'], scatter_unit_info['center_y'],
    #                      scatter_unit_info['scatter_x'], scatter_unit_info['scatter_y']):
    #     print((txt, x, y))
    #     ax.annotate('%d' % txt, (x, y))
    plt.show()

    # fig.savefig('C:/Users/Dalton/Documents/lab_files/analysis_encoding_model/plots/map_%s' % key, bbox_inches='tight', dpi=params.dpi)

def plot_sweep_over_lead_lag(model_results_across_lags):
    traj_mean_performance = model_results_across_lags['mean_performance_by_lead_lag'][0].copy()
    traj_mean_performance['reorder'] = [12, 13, 14, 8, 10, 11, 6, 7, 9, 2, 4, 5, 1, 3, 0]
    traj_mean_performance.sort_values(by='reorder', inplace=True)
    
    # fig, ax = plt.subplots()
    # ax.errorbar(traj_mean_performance['reorder'], traj_mean_performance['auc'], yerr=traj_mean_performance['SE'])
    # ax.set_xticks(traj_mean_performance['reorder'])
    # ax.set_xticklabels(traj_mean_performance.index, rotation=45)
    # plt.show()
    
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.errorbar(traj_mean_performance['reorder'], 
                traj_mean_performance['auc'], 
                yerr=traj_mean_performance['SE'], 
                linewidth=0,
                elinewidth=3,
                marker='o',
                markersize=10)
    ax.errorbar(9,
                traj_mean_performance['auc'].iloc[9],
                yerr=traj_mean_performance['SE'].iloc[9], 
                linewidth=0,
                elinewidth=3,
                marker='o',
                markersize=10,
                color='red')
        
    # ax.set_xlabel('Top Percent of Weights Shuffled', fontsize = params.axis_fontsize)
    # ax.set_ylabel('Percent AUC Loss', fontsize = params.axis_fontsize)
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([0.56, 0.59])
    ax.tick_params(width=2, length = 4, labelsize = params.tick_fontsize)
    # ax.set_xticks(traj_mean_performance['reorder'])
    # for tick in ax.get_xticklabels():
    #     tick.set_fontsize(params.tick_fontsize)
    # for tick in ax.get_yticklabels():
    #     tick.set_fontsize(params.tick_fontsize)
    sns.despine(ax=ax)
    ax.spines['bottom'].set_linewidth(params.axis_linewidth)
    ax.spines['left'  ].set_linewidth(params.axis_linewidth)
    plt.show()
    
    fig.savefig('C:/Users/Dalton/Documents/lab_files/AREADNE/plots/model_auc_over_leadlags.png', bbox_inches='tight', dpi=params.dpi)


if __name__ == "__main__":

    spike_data, kinematics, analog_and_video, FN = load_data() 
    
    all_models_data = load_models()
    
    all_models_data = determine_trajectory_significance(all_models_data)
    
    model_results_across_lags = organize_results_by_model_for_all_lags(all_models_data)
    
    model_results_across_lags = compute_mean_model_performance(model_results_across_lags, percent = 25, percentile_mode = 'per_lag_set')
    
    unit_info_optimal_lag = find_optimal_lag_for_each_unit(model_results_across_lags, all_models_data)
    
    plot_optimal_lag_on_channel_map(unit_info_optimal_lag, spike_data, jitter_radius = .15, hueKey = 'traj_center', sizeKey = 'pathlet_AUC', minAUC = 0.5)
    
    plot_sweep_over_lead_lag(model_results_across_lags)