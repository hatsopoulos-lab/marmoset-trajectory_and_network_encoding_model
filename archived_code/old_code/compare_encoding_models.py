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
import matplotlib.colors
import pickle
import dill
import os
import glob
import math
import re
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc, average_precision_score
import statsmodels.api as sm
from scipy.stats import binomtest, ttest_rel, ttest_ind
from scipy.integrate import cumtrapz
from scipy.integrate import simps
from mpl_toolkits import mplot3d 
from scipy.ndimage import median_filter, gaussian_filter

class path:
    # storage = '/project2/nicho/dalton/processed_datasets'
    # intermediate_save_path = '/project2/nicho/dalton/analysis/encoding_model/intermediate_variable_storage'
    # plots = '/project2/nicho/dalton/analysis/encoding_model/plots'
    # date = '20210211'
    
    storage = '/project2/nicho/dalton/processed_datasets'
    intermediate_save_path = '/project2/nicho/dalton/analysis/encoding_model/intermediate_variable_storage'
    new_save_path = '/project2/nicho/dalton/analysis/encoding_model/intermediate_variable_storage/10pt0_ms_bins/data_updated_february_2023'
    plots = '/project2/nicho/dalton/analysis/encoding_model/plots/new_analysis_february_2023'
    date = '20210211'  
    
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
    
    channel_sep_horizontal = 0.4 # in mm

    
def load_data():
    
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

    # FN_path = os.path.join(path.intermediate_save_path, 'FN_%s_fMI_10ms_bins_dict.pkl' % path.date)
    # with open(FN_path, 'rb') as fp:
    #     raster_list, FN = dill.load(fp)
    FN_path = os.path.join(path.intermediate_save_path, 'FN_%s_fMI_10ms_bins_dict.pkl' % path.date)
    with open(FN_path, 'rb') as fp:
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

    results_list            = []
    unit_info_list          = []
    traj_features_list      = []
    network_features_list   = []
    short_features_list     = []
    components_list         = []
    sampled_spikes_list     = []
    lead_lag_list           = []
    sample_info_list        = []
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
        with open(s_path, 'rb') as f:
            trajectoryList, shortTrajectoryList, avgPos, avgSpeed, sampledSpikes, reachSpikes, sample_info, tmp = dill.load(f)

        lead = int(re.findall(re.compile('lead_\d{1,3}'), r_path)[0].split('lead_')[-1])
        lag  = int(re.findall(re.compile('lag_\d{1,3}' ), r_path)[0].split('lag_' )[-1])
        results_list.append(models_results)
        unit_info_list.append(unit_info)
        traj_features_list.append(traj_features)
        network_features_list.append(network_features)
        short_features_list.append(short_features)
        components_list.append(compsOut)
        lead_lag_list.append((lead, lag))
        sampled_spikes_list.append(sampledSpikes)
        sample_info_list.append(sample_info)
    
    all_models_data = {'model_details'    : results_list,
                       'unit_info'        : unit_info_list,
                       'traj_features'    : traj_features_list,
                       'network_features' : network_features_list,
                       'short_features'   : short_features_list,
                       'components'       : components_list,
                       'lead_lag'         : lead_lag_list,
                       'sampled_spikes'   : sampled_spikes_list,
                       'sample_info'      : sample_info_list}
    
    return all_models_data

def load_color_palette():
    LinL = np.loadtxt(os.path.join(path.plots, '0-1/Linear_L_0-1.csv'), delimiter=',')
    
    b3=LinL[:,2] # value of blue at sample n
    b2=LinL[:,2] # value of blue at sample n
    b1=np.linspace(0,1,len(b2)) # position of sample n - ranges from 0 to 1
    
    # setting up columns for list
    g3=LinL[:,1]
    g2=LinL[:,1]
    g1=np.linspace(0,1,len(g2))
    
    r3=LinL[:,0]
    r2=LinL[:,0]
    r1=np.linspace(0,1,len(r2))
    
    # creating list
    R=zip(r1,r2,r3)
    G=zip(g1,g2,g3)
    B=zip(b1,b2,b3)
    
    # transposing list
    RGB=zip(R,G,B)
    rgb=zip(*RGB)
    # print rgb
    
    # creating dictionary
    k=['red', 'green', 'blue']
    LinearL=dict(zip(k,rgb)) # makes a dictionary from 2 lists
    my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',LinearL)
    
    return my_cmap

def test_model_significance(trueAUC_means, shuffleAUC_means):
    
    p_val = np.empty((np.shape(trueAUC_means)[0], np.shape(trueAUC_means)[1]))
    for unit, (trueMean, shuffleMeans) in enumerate(zip(trueAUC_means, shuffleAUC_means)):
        p_val[unit, 0] = np.sum(shuffleMeans[0, :] > trueMean[0]) / np.shape(shuffleMeans)[-1]     
        p_val[unit, 1] = np.sum(shuffleMeans[1, :] > trueMean[1]) / np.shape(shuffleMeans)[-1]     
    
    return p_val

def plot_model_auc_comparison(unit_info, x_key, y_key, minauc = 0.5):
    fig, ax = plt.subplots(figsize = params.aucScatter_figSize)
    # sns.scatterplot(ax = ax, data = unit_info, x = x_key, y = y_key, 
    #                 hue = "fr", style = "group")
    sns.scatterplot(ax = ax, data = unit_info, x = x_key, y = y_key, 
                    style = "group", s = 60, legend=False)
    ax.plot(np.arange(minauc, 1.0, 0.05), np.arange(minauc, 1.0, 0.05), '--k')
    ax.scatter(unit_info[x_key].to_numpy()[44] , unit_info[y_key].to_numpy()[44] , s = 60, c ='red', marker='x')
    ax.scatter(unit_info[x_key].to_numpy()[107], unit_info[y_key].to_numpy()[107], s = 60,  c ='red', marker='o')
    ax.set_xlim(minauc, 1)
    ax.set_ylim(minauc, 1)
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color('black')
    for axis in ['top','right']:
        ax.spines[axis].set_linewidth(0)
    ax.tick_params(width=2, length = 4, labelsize = params.tick_fontsize)
    # ax.set_xlabel('ROC area (%s)' % x_key[:-4], fontsize = params.axis_fontsize)
    # ax.set_ylabel('ROC area (%s)' % y_key[:-4], fontsize = params.axis_fontsize)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.grid(False)
    # ax.legend(bbox_to_anchor=(1.5, 1.5), loc='upper left', borderaxespad=0)
    plt.show()
    
    fig.savefig(os.path.join(path.plots, 'area_under_curve_%s_%s.png' % (x_key, y_key)), bbox_inches='tight', dpi=params.dpi)


def sig_tests(unit_info, key_test, key_full,  unit_info_reduced = None):
    
    if unit_info_reduced is None:
        nFull = np.sum(unit_info[key_full] > unit_info[key_test])
        nUnits = np.shape(unit_info)[0]
        
        sign_test = binomtest(nFull, nUnits, p = 0.5, alternative='greater')
        
        ttest_paired = ttest_rel(unit_info[key_full], unit_info[key_test], alternative='greater')

    else:
        nPathlet = np.sum(unit_info.pathlet_AUC > unit_info_reduced.pathlet_AUC)
        nUnits = np.shape(unit_info)[0]
        sign_test = binomtest(nPathlet, nUnits, p = 0.5, alternative='greater')
        ttest_paired = ttest_rel(unit_info.pathlet_AUC, unit_info_reduced.pathlet_AUC, alternative='greater')

    return sign_test, ttest_paired

def plot_scatter_of_result_on_channel_map(unit_info, spike_data, jitter_radius = .15, key = 'full_AUC', min_thresh = None):
    
    rotated_map = spike_data['chan_map_ns6'].copy()
    # for j in range(rotated_map.shape[1]):
    #     rotated_map[:, j] = rotated_map[::-1, j]
    
    scatter_unit_info = unit_info.copy()
    scatter_unit_info['scatter_x'] = np.full((unit_info.shape[0],), np.nan)
    scatter_unit_info['scatter_y'] = np.full((unit_info.shape[0],), np.nan)
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

    
    if min_thresh is not None:
        scatter_unit_info.loc[scatter_unit_info[key] < min_thresh, key] = np.nan
    
    # fig, ax = plt.subplots(figsize=params.map_figSize)
    fig, ax = plt.subplots(figsize=(5.5,5.5))
    sns.scatterplot(ax = ax, data = scatter_unit_info, x = 'scatter_x', y = 'scatter_y', 
                    size = key, sizes = (40, 100), hue = key, style = "group")
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
    ax.set_xlabel('Lateral'  , fontsize = params.axis_fontsize)
    ax.set_ylabel('Posterior', fontsize = params.axis_fontsize)
    ax.legend(bbox_to_anchor=(1.1, 1.1), loc='upper left', borderaxespad=0)
    ax.set_title(key)

    ax.grid(False)
    # for txt, x, y, scat_x, scat_y in zip(scatter_unit_info['ns6_elec_id'], scatter_unit_info['center_x'], scatter_unit_info['center_y'],
    #                      scatter_unit_info['scatter_x'], scatter_unit_info['scatter_y']):
    #     print((txt, x, y))
    #     ax.annotate('%d' % txt, (x, y))
    plt.show()

    fig.savefig(os.path.join(path.plots, 'map_%s' % key), bbox_inches='tight', dpi=params.dpi)
    
def compute_AUC_loss(unit_info):
    model_names = [name for name in unit_info.columns if '_AUC' in name 
                   and not any(match in name for match in ['full', 'brief', 'shuffled', 'network_partial_traj'])]
    for key in model_names:
        unit_info['%s_loss' % key[:-4]] = (unit_info['full_AUC'] - unit_info[key]) / unit_info['full_AUC'] 
    
    return unit_info

def plot_loss_distributions(unit_info):
    loss_keys = [key for key in unit_info.columns if 'loss' in key]
    
    loss_vals   = []
    loss_labels = []
    for key in loss_keys:
        loss_vals.extend  (unit_info[key].to_list())
        loss_labels.extend([key.split('_loss')[0]]*unit_info.shape[0])
    
    loss_vals = [100*val for val in loss_vals]
    
    loss_df = pd.DataFrame(data = zip(loss_vals, loss_labels), columns = ['Loss', 'Model'])
    fig, ax = plt.subplots()
    sns.histplot(data=loss_df, ax = ax, x='Loss', hue='Model',
                 log_scale=False, element="poly", fill=False,
                 cumulative=True, common_norm=False, bins=25)
    ax.set_xlabel('% AUC Loss')
    plt.show()
    
    fig, ax = plt.subplots()
    sns.kdeplot(data=loss_df, ax=ax, x='Loss', hue='Model',
                 log_scale=False, fill=False,
                 cumulative=False, common_norm=False, bw_adjust=.75)
    ax.set_xlabel('% AUC Loss')

    plt.show()

def plot_AUC_box_whisker(unit_info, mode = 'general'):
    
    if mode == 'general':
        # model_names = [name for name in unit_info.columns if '_AUC' in name
        #                and not any(match in name for match in ['network_partial_traj', 
        #                                                        'pathlet_two_comps',
        #                                                        'brief'])]
        model_names = [name for name in unit_info.columns if '_AUC' in name
                       and any(match in name for match in ['shuffle', 
                                                           'pathlet',
                                                           'full',
                                                           'network'])]
        # model_names = [model_names[idx] for idx in [3, 2, 1, 0, 5, 6, 7]] 
        model_names = [model_names[idx] for idx in [3, 2, 1, 0]] 

        auc_vals = []
        auc_lbls = []
        for key in model_names:
            auc_vals.extend(unit_info[key].to_list())
            auc_lbls.extend([key.split('_AUC')[0]]*unit_info.shape[0])
            
        auc_df = pd.DataFrame(data = zip(auc_vals, auc_lbls), columns = ['AUC', 'Model'])
        # fig, ax = plt.subplots(figsize=params.boxplot_figSize)
        fig, ax = plt.subplots(figsize=(5, 6))
        sns.boxplot(data=auc_df, ax=ax, x='Model', y='AUC', palette = sns.color_palette('Dark2'))
        # ax.set_xticklabels(['Full', 'Network', 'Kinematics', 'Shuffled\nData', 
        #                     'Permuted\nNetwork\nWeights', 'Permuted\nNetwork\nWeights\nTop 25%',
        #                     'Permuted\nNetwork\nTopology\nTop 25%'], fontsize=params.axis_fontsize)
        ax.set_xlabel('')
        # ax.set_ylabel('Area Under ROC-Curve', fontsize=params.axis_fontsize)
        ax.set_ylabel('')
        ax.set_xticklabels([])
        ax.tick_params(width=2, length = 4, labelsize = params.tick_fontsize)
        sns.despine(ax=ax)
        ax.spines['bottom'].set_linewidth(params.axis_linewidth)
        ax.spines['left'  ].set_linewidth(params.axis_linewidth)
        plt.show()
        
        fig.savefig(os.path.join(path.plots, 'boxplot.png'), bbox_inches='tight', dpi=params.dpi)

    
    elif mode == 'pathlet_network_subsets':
        model_names = [name for name in unit_info.columns if '_AUC' in name
                       and any(match in name for match in ['full', 'network', 'network_partial_traj', 'pathlet_two_comps'])]
        # model_names = [model_names[idx] for idx in [2, 0, 3, 1]]
        auc_vals = []
        auc_lbls = []
        for key in model_names:
            auc_vals.extend(unit_info[key].to_list())
            auc_lbls.extend([key.split('_AUC')[0]]*unit_info.shape[0])
            
        auc_df = pd.DataFrame(data = zip(auc_vals, auc_lbls), columns = ['AUC', 'Model'])
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.boxplot(data=auc_df, ax=ax, x='Model', y='AUC')
        # ax.set_xticklabels(['Full', 'Network', 'Top\nTwo\nKinematic\nComps', 
        #                     'Full\nMinus\nTwo\nComps'], fontsize=params.axis_fontsize)
        ax.set_xlabel('')
        ax.set_ylabel('Area Under ROC-Curve', fontsize=params.axis_fontsize)
        sns.despine(ax=ax)
        ax.spines['bottom'].set_linewidth(params.axis_linewidth)
        ax.spines['left'  ].set_linewidth(params.axis_linewidth)
        plt.show()
        
        fig.savefig(os.path.join(path.plots, 'boxplot_kin_net_subsets.png'), bbox_inches='tight', dpi=params.dpi)

    elif mode == 'all':
        model_names = [name for name in unit_info.columns if '_AUC' in name]
        model_names = [model_names[idx] for idx in [5, 3, 2, 1, 0, 4, 6]]      
        auc_vals = []
        auc_lbls = []
        for key in model_names:
            auc_vals.extend(unit_info[key].to_list())
            auc_lbls.extend([key.split('_AUC')[0]]*unit_info.shape[0])
            
        auc_df = pd.DataFrame(data = zip(auc_vals, auc_lbls), columns = ['AUC', 'Model'])
        fig, ax = plt.subplots()
        sns.boxplot(data=auc_df, ax=ax, x='Model', y='AUC')
        
        plt.show() 

def compute_loss_statistics(unit_info):
    
    unit_info = compute_AUC_loss(unit_info)
    plot_loss_distributions(unit_info)
    plot_AUC_box_whisker(unit_info, mode = 'general')
    plot_AUC_box_whisker(unit_info, mode = 'pathlet_network_subsets')
    
    loss_keys = [key for key in unit_info.columns if 'loss' in key]
    
    loss_statistics = pd.DataFrame(data = np.full((math.comb(len(loss_keys), 2), 6), np.nan),
                                   columns = ['pair', 'median1', 'IQR1', 'median2', 'IQR2', 'pval'])
    ct = 0
    for key1 in loss_keys:
        for key2 in loss_keys:
            if key1 != key2:
                pair = '%s_vs_%s' % (key1.split('_loss')[0], key2.split('_loss')[0])
                print(pair)
                reverse_pair = '%s_vs_%s' % (key2.split('_loss')[0], key1.split('_loss')[0])  
                if reverse_pair not in loss_statistics.pair.to_list():
                    loss_statistics.at[loss_statistics.index[ct], 'pair']    = pair
                    loss_statistics.at[loss_statistics.index[ct], 'median1'] = unit_info[key1].median() * 100
                    loss_statistics.at[loss_statistics.index[ct], 'median2'] = unit_info[key2].median() * 100
                    
                    q75, q25 = np.percentile(unit_info[key1], [75 ,25])
                    loss_statistics.at[loss_statistics.index[ct], 'IQR1'] = (q75-q25) * 100
                    q75, q25 = np.percentile(unit_info[key2], [75 ,25])
                    loss_statistics.at[loss_statistics.index[ct], 'IQR2'] = (q75-q25) * 100
                    
                    ttest_paired = ttest_rel(unit_info[key1], unit_info[key2], alternative='two-sided')
                    loss_statistics.at[loss_statistics.index[ct], 'pval'] = ttest_paired[1]
            
                    ct =+ 1      
    
    return unit_info, loss_statistics

def plot_sample_kinematics(kinematics, event = 8):
    
    vel_figsize = (3.5, 3)
    pos_figsize = (4.95, 4.5)
    
    tmp_kin = [kin for kin in kinematics if kin['event'] == event][0]
    ypos = tmp_kin['position'][3, 1]
    reachIdx = []
    velIdx = []
    for start, stop in zip(tmp_kin['starts'], tmp_kin['stops']):
        reachIdx.extend(list(range(start, stop)))
        velIdx.extend(list(range(start, stop-1)))
    fig = plt.figure(figsize = (5.5,5))
    ax3d = fig.add_subplot(111, projection='3d')
    ax3d.plot(tmp_kin['position'][3, 0, reachIdx], tmp_kin['position'][3, 1, reachIdx], tmp_kin['position'][3, 2, reachIdx], '-k', linewidth=2)
    ax3d.plot(tmp_kin['position'][3, 0, reachIdx[0]], tmp_kin['position'][3, 1, reachIdx[0]], tmp_kin['position'][3, 2, reachIdx[0]], 'ko', markersize=10)
    ax3d.set_xlabel('x (cm)', fontsize = params.axis_fontsize)
    ax3d.set_ylabel('y (cm)', fontsize = params.axis_fontsize)
    ax3d.set_zlabel('z (cm)', fontsize = params.axis_fontsize)
    ax3d.set_xlim(2.5, 13)
    ax3d.set_ylim(-2, 10)
    ax3d.set_zlim(0, 6)
    ax3d.view_init(28, 148)
    ax3d.tick_params(axis='x', which='major', pad=-1, labelsize = params.tick_fontsize)
    ax3d.tick_params(axis='y', which='major', pad=-1, labelsize = params.tick_fontsize)
    ax3d.tick_params(axis='z', which='major', pad=-1, labelsize = params.tick_fontsize)
    # ax3d.grid(False)
    fig.tight_layout()
    plt.show()
    
    fig.savefig(os.path.join(path.plots, 'event_%d_no_traj.png' % event), bbox_inches='tight', dpi=params.dpi)


    time = np.arange(0, len(ypos)-1) / params.fps
    # fig, ax = plt.subplots(figsize = (6, 5))
    # ax.plot(time[velIdx] - time[velIdx[0]], tmp_kin['velocity'][3, 0, velIdx] + 40, linewidth = 2)
    # ax.plot(time[velIdx] - time[velIdx[0]], tmp_kin['velocity'][3, 1, velIdx], linewidth = 2)
    # ax.plot(time[velIdx] - time[velIdx[0]], tmp_kin['velocity'][3, 2, velIdx] - 50, linewidth = 2)
    # ax.set_xlabel('Time', fontsize = params.axis_fontsize)
    # ax.set_ylabel('Velocity (cm/s)', fontsize = params.axis_fontsize)
    # ax.set_xticks([0, 3.5])
    # ax.set_xticklabels([0, 3.5], fontsize = params.axis_fontsize)
    # ax.set_yticks([0, 25])
    # ax.set_yticklabels(ax.get_yticks(), fontsize = params.axis_fontsize)
    # sns.despine(ax=ax)
    # ax.spines['bottom'].set_linewidth(params.axis_linewidth)
    # ax.spines['left'  ].set_linewidth(params.axis_linewidth)
    # plt.show()
    
    # time = np.arange(0, len(ypos)-1) / params.fps
    # fig, ax = plt.subplots(figsize = (6, 5))
    # ax.plot(time[velIdx] - time[velIdx[0]], tmp_kin['velocity'][3, 0, velIdx] + 40, linewidth = 2)
    # ax.plot(time[velIdx] - time[velIdx[0]], tmp_kin['velocity'][3, 1, velIdx], linewidth = 2)
    # ax.plot(time[velIdx] - time[velIdx[0]], tmp_kin['velocity'][3, 2, velIdx] - 50, linewidth = 2)
    # ax.set_xlabel('Time', fontsize = params.axis_fontsize)
    # ax.set_ylabel('Velocity (cm/s)', fontsize = params.axis_fontsize)
    # ax.set_xticks([0, 3.5])
    # ax.set_xticklabels([0, 3.5], fontsize = params.axis_fontsize)
    # ax.set_yticks([0, 25])
    # ax.set_yticklabels(ax.get_yticks(), fontsize = params.axis_fontsize)
    # sns.despine(ax=ax)
    # ax.spines['bottom'].set_linewidth(params.axis_linewidth)
    # ax.spines['left'  ].set_linewidth(params.axis_linewidth)
    # plt.show()
    
    vel_marksize = 4
    
    leadIdx = np.where(time <= params.lead[0])[0]
    lagIdx  = np.where((time < params.lead[0] + params.lag[0]) & (time >= params.lead[0]))[0]
    velIdx = np.array(velIdx)
    leadIdx = velIdx[leadIdx]
    lagIdx = velIdx[lagIdx]
    fig, ax = plt.subplots(figsize = vel_figsize)
    ax.plot(time[leadIdx] - time[leadIdx[0]], tmp_kin['velocity'][3, 0, leadIdx] + 40, 'bo', linewidth = 2, markersize = vel_marksize)
    ax.plot(time[leadIdx] - time[leadIdx[0]], tmp_kin['velocity'][3, 1, leadIdx], 'bo', linewidth = 2, markersize = vel_marksize)
    ax.plot(time[leadIdx] - time[leadIdx[0]], tmp_kin['velocity'][3, 2, leadIdx] - 50, 'bo', linewidth = 2, markersize = vel_marksize)
    ax.plot(time[lagIdx] - time[leadIdx[0]], tmp_kin['velocity'][3, 0, lagIdx] + 40, 'ro', linewidth = 2, markersize = vel_marksize)
    ax.plot(time[lagIdx] - time[leadIdx[0]], tmp_kin['velocity'][3, 1, lagIdx], 'ro', linewidth = 2, markersize = vel_marksize)
    ax.plot(time[lagIdx] - time[leadIdx[0]], tmp_kin['velocity'][3, 2, lagIdx] - 50, 'ro', linewidth = 2, markersize = vel_marksize)
    ax.plot(np.repeat(time[lagIdx[0]] - time[leadIdx[0]], 3), tmp_kin['velocity'][3, :, lagIdx[0]] + np.array([40, 0, -50]), 
            'wo', markeredgewidth=1.5, markeredgecolor='k', markersize = 10)
    ax.set_xlim(0, params.lead[0]+params.lag[0])
    ax.set_ylim(-70, 70)
    ax.set_xlabel('Time (ms)', fontsize = params.axis_fontsize)
    ax.set_ylabel('Velocity (cm/s)', fontsize = params.axis_fontsize)
    ax.set_xticks([0, params.lead[0], params.lead[0]+params.lag[0]])
    ax.set_xticklabels([int((-1*params.lead[0])*1e3), 0, int((params.lag[0]) * 1e3)], fontsize = params.tick_fontsize)
    ax.set_yticks([0, 25])
    ax.set_yticklabels(ax.get_yticks(), fontsize = params.tick_fontsize)
    sns.despine(ax=ax)
    ax.spines['bottom'].set_linewidth(params.axis_linewidth)
    ax.spines['left'  ].set_linewidth(params.axis_linewidth)
    plt.show()
    fig.savefig(os.path.join(path.plots, 'event_%d_traj_vel.png' % event), bbox_inches='tight', dpi=params.dpi)
    
    fig = plt.figure(figsize = pos_figsize)
    ax3d = fig.add_subplot(111, projection='3d')
    ax3d.plot(tmp_kin['position'][3, 0, reachIdx], tmp_kin['position'][3, 1, reachIdx], tmp_kin['position'][3, 2, reachIdx], 'k-')
    ax3d.plot(tmp_kin['position'][3, 0, reachIdx[0]], tmp_kin['position'][3, 1, reachIdx[0]], tmp_kin['position'][3, 2, reachIdx[0]], 'k-')
    
    ax3d.plot(tmp_kin['position'][3, 0, leadIdx], tmp_kin['position'][3, 1, leadIdx], tmp_kin['position'][3, 2, leadIdx], 'b-', linewidth = 4)
    ax3d.plot(tmp_kin['position'][3, 0, lagIdx], tmp_kin['position'][3, 1, lagIdx], tmp_kin['position'][3, 2, lagIdx], 'r-', linewidth = 4)
    ax3d.plot(tmp_kin['position'][3, 0, leadIdx[-1]], tmp_kin['position'][3, 1, leadIdx[-1]], tmp_kin['position'][3, 2, leadIdx[-1]], 
              'wo', markeredgewidth=1.5, markeredgecolor='k', markersize = 10)
    # ax3d.set_xlabel('x (cm)', fontsize = params.axis_fontsize)
    # ax3d.set_ylabel('y (cm)', fontsize = params.axis_fontsize)
    # ax3d.set_zlabel('z (cm)', fontsize = params.axis_fontsize)
    ax3d.set_xlabel('', fontsize = params.axis_fontsize)
    ax3d.set_ylabel('', fontsize = params.axis_fontsize)
    ax3d.set_zlabel('', fontsize = params.axis_fontsize)    
    ax3d.set_xlim(2.5, 13)
    ax3d.set_ylim(-2, 10)
    ax3d.set_zlim(0, 6)
    ax3d.set_xticklabels([])
    ax3d.set_yticklabels([])
    ax3d.set_zticklabels([])
    ax3d.tick_params(axis='x', which='major', pad=-1, labelsize = params.tick_fontsize)
    ax3d.tick_params(axis='y', which='major', pad=-1, labelsize = params.tick_fontsize)
    ax3d.tick_params(axis='z', which='major', pad=-1, labelsize = params.tick_fontsize)
    ax3d.view_init(28, 148)
    plt.show()
    fig.savefig(os.path.join(path.plots, 'event_%d_with_traj.png' % event), bbox_inches='tight', dpi=params.dpi)

    
    leadIdx = np.where(time <= params.lead[0])[0]
    lagIdx  = np.where((time < params.lead[0] + .15) & (time >= params.lead[0] + .1))[0]
    velIdx = np.array(velIdx)
    leadIdx = velIdx[leadIdx]
    lagIdx = velIdx[lagIdx]
    fig, ax = plt.subplots(figsize = vel_figsize)
    ax.plot(time[lagIdx] - time[leadIdx[0]], tmp_kin['velocity'][3, 0, lagIdx] + 40, 'ro', linewidth = 2, markersize = vel_marksize)
    ax.plot(time[lagIdx] - time[leadIdx[0]], tmp_kin['velocity'][3, 1, lagIdx], 'ro', linewidth = 2, markersize = vel_marksize)
    ax.plot(time[lagIdx] - time[leadIdx[0]], tmp_kin['velocity'][3, 2, lagIdx] - 50, 'ro', linewidth = 2, markersize = vel_marksize)
    ax.plot(np.repeat(time[leadIdx[-1]] - time[leadIdx[0]], 3), tmp_kin['velocity'][3, :, leadIdx[-1]] + np.array([40, 0, -50]), 
            'wo', markeredgewidth=1.5, markeredgecolor='k', markersize = vel_marksize)
    ax.set_xlabel('Time (ms)', fontsize = params.axis_fontsize)
    ax.set_ylabel('Velocity (cm/s)', fontsize = params.axis_fontsize)
    ax.set_xlim(0, params.lead[0] + params.lag[0])
    ax.set_ylim(-70, 70)
    # ax.set_xticks([params.lead[0], params.lead[0] + .1, params.lead[0] + .15])
    ax.set_xticks([params.lead[0], params.lead[0] + .1])
    ax.set_xticklabels([0, 100], fontsize = params.tick_fontsize)
    ax.set_yticks([0, 25])
    ax.set_yticklabels(ax.get_yticks(), fontsize = params.tick_fontsize)
    sns.despine(ax=ax)
    ax.spines['bottom'].set_linewidth(params.axis_linewidth)
    ax.spines['left'  ].set_linewidth(params.axis_linewidth)
    plt.show()
    fig.savefig(os.path.join(path.plots, 'event_%d_veltuning_vel.png' % event), bbox_inches='tight', dpi=params.dpi)
    
    fig = plt.figure(figsize = pos_figsize)
    ax3d = fig.add_subplot(111, projection='3d')
    ax3d.plot(tmp_kin['position'][3, 0, reachIdx], tmp_kin['position'][3, 1, reachIdx], tmp_kin['position'][3, 2, reachIdx], 'k-')
    ax3d.plot(tmp_kin['position'][3, 0, reachIdx[0]], tmp_kin['position'][3, 1, reachIdx[0]], tmp_kin['position'][3, 2, reachIdx[0]], 'k-')
    
    ax3d.plot(tmp_kin['position'][3, 0, lagIdx], tmp_kin['position'][3, 1, lagIdx], tmp_kin['position'][3, 2, lagIdx], 'r-', linewidth = 4)
    ax3d.plot(tmp_kin['position'][3, 0, leadIdx[-1]], tmp_kin['position'][3, 1, leadIdx[-1]], tmp_kin['position'][3, 2, leadIdx[-1]], 
              'wo', markeredgewidth=1.5, markeredgecolor='k', markersize = 10)
    # ax3d.set_xlabel('x (cm)', fontsize = params.axis_fontsize)
    # ax3d.set_ylabel('y (cm)', fontsize = params.axis_fontsize)
    # ax3d.set_zlabel('z (cm)', fontsize = params.axis_fontsize)
    ax3d.set_xlabel('', fontsize = params.axis_fontsize)
    ax3d.set_ylabel('', fontsize = params.axis_fontsize)
    ax3d.set_zlabel('', fontsize = params.axis_fontsize)    
    ax3d.set_xlim(2.5, 13)
    ax3d.set_ylim(-2, 10)
    ax3d.set_zlim(0, 6)
    ax3d.set_xticklabels([])
    ax3d.set_yticklabels([])
    ax3d.set_zticklabels([])

    ax3d.tick_params(axis='x', which='major', pad=-1, labelsize = params.tick_fontsize)
    ax3d.tick_params(axis='y', which='major', pad=-1, labelsize = params.tick_fontsize)
    ax3d.tick_params(axis='z', which='major', pad=-1, labelsize = params.tick_fontsize)
    ax3d.view_init(28, 148)
    plt.show()
    fig.savefig(os.path.join(path.plots, 'event_%d_with_vel.png' % event), bbox_inches='tight', dpi=params.dpi)


def plot_FN_heatmap(FN, name, cmax = None):
    
    FN_cmap = load_color_palette()
    
    # fig, ax = plt.subplots(figsize=params.FN_figSize)
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(data = FN, ax=ax, cmap= FN_cmap, square=True, cbar_kws={'label': 'Full Mutual Information', "orientation": "horizontal",
                                                                        'location' : 'top'})
    ax.set_xlabel('Target Unit', fontsize=params.axis_fontsize)
    ax.set_ylabel('Input Unit' , fontsize=params.axis_fontsize)
    ax.set_xticks([0.5, 168.5], labels=[1, 169])
    # ax.set_xticklabels([1, 169])
    ax.set_yticks([0.5, 168.5], labels=[1, 169])
    # ax.set_yticklabels([1, 169])
    sns.despine(ax=ax)
    ax.spines['bottom'].set_linewidth(params.axis_linewidth)
    ax.spines['left'  ].set_linewidth(params.axis_linewidth)
    cb = ax.collections[-1].colorbar
    # cb.set_ticks(cb.get_ticks()[:-1], labelsize = params.tick_fontsize)
    ax.tick_params(width=2, length = 4, labelsize = params.tick_fontsize)

    if cmax is None or cb.vmax > cmax:
        cmax_out = cb.vmax
    else:
        cb.vmax = cmax
        cmax_out = cmax 

    plt.show()

    fig.savefig(os.path.join(path.plots, 'FN_heatmap_%s.png' % name), bbox_inches='tight', dpi=params.dpi)

    return cmax_out
    

def get_single_lead_lag_data(all_models_data, lead, lag):
    ll_idx = [idx for idx, ll in enumerate(all_models_data['lead_lag']) if ll == (lead*1e3, lag*1e3)][0]
    unit_info     = all_models_data['unit_info'][ll_idx]
    model_details = all_models_data['model_details'][ll_idx]
    components    = all_models_data['components'][ll_idx]
    traj_features = all_models_data['traj_features'][ll_idx]
    sampledSpikes = all_models_data['sampled_spikes'][ll_idx]
    sample_info   = all_models_data['sample_info'][ll_idx]
    
    return unit_info, model_details, components, traj_features, sampledSpikes, sample_info

def compute_and_analyze_pathlets(model_details, PCAcomps, unit_info):
    traj_model_idx = [idx for idx, name in enumerate(model_details['model_names']) if name == 'trajectory'][0]
    coefs = model_details['model_results'][traj_model_idx]['param_coefs']
    comps = PCAcomps.transpose()
#    beta = coefs[1:np.shape(comps)[-1]+1, :, 10]    
    beta = np.mean(coefs, axis = -1)[1:np.shape(comps)[-1]+1, :]
    velTraj = comps @ beta
    velTraj = np.swapaxes(velTraj.reshape((params.nDims, int(np.shape(velTraj)[0] / params.nDims), np.shape(velTraj)[-1])), 0, 1)
    
#    posTraj = np.empty(np.shape(velTraj))
#    for unit in range(np.shape(velTraj)[-1]):
#        posTraj[..., unit] = cumtrapz(velTraj[..., unit], dx = (params.lag[0] + params.lead[0]) / np.shape(velTraj)[0], axis = 0, initial = 0)

    posTraj = cumtrapz(velTraj, dx = (params.lag_to_analyze[0] + params.lead_to_analyze[0]) / np.shape(velTraj)[0], axis = 0, initial = 0)
    dist = simps(np.linalg.norm(velTraj, axis = 1), dx = (params.lag_to_analyze[0] + params.lead_to_analyze[0]) / np.shape(velTraj)[0], axis = 0)
    
    pathDivergence = np.empty(np.shape(coefs[0, ...].transpose()))
    sample_pathlets = []
    for samp in range(np.shape(coefs)[-1]):
        beta_samp = coefs[1:np.shape(comps)[-1] +1, :, samp]
        velTraj_samp = comps @ beta_samp
        velTraj_samp = np.swapaxes(velTraj_samp.reshape((params.nDims, int(np.shape(velTraj_samp)[0] / params.nDims), np.shape(velTraj_samp)[-1])), 0, 1)
        posTraj_samp = cumtrapz(velTraj_samp, dx = (params.lag[0] + params.lead[0]) / np.shape(velTraj_samp)[0], axis = 0, initial = 0)
        sample_pathlets.append(posTraj_samp)
        pathDivergence[samp, :] = np.sum(np.linalg.norm(posTraj - posTraj_samp, axis = 1), axis = 0)
        
        divShuffle = np.empty((np.shape(pathDivergence)[0], np.shape(pathDivergence)[1], 100))
        for shuffle in range(100):
            idx = np.random.choice(np.arange(np.shape(pathDivergence)[1]), size = np.shape(pathDivergence)[1], replace = 0)
            while np.sum(idx == np.arange(np.shape(pathDivergence)[1])) > 0:
                idx = np.random.choice(np.arange(np.shape(pathDivergence)[1]), size = np.shape(pathDivergence)[1], replace = 0)
    
            divShuffle[samp, :, shuffle] = np.sum(np.linalg.norm(posTraj[..., idx] - posTraj_samp, axis = 1), axis = 0)
    
    # axlims_best  = plot_pathlet(posTraj, sample_pathlets, unit_info, unit_selector = 'max', numToPlot = 1, unitsToPlot = None)
    # axlims_worst = plot_pathlet(posTraj, sample_pathlets, unit_info, unit_selector = 'min', numToPlot = 1, unitsToPlot = None, axlims = axlims_best)
    axlims_good = plot_pathlet(posTraj, sample_pathlets, unit_info, unit_selector = 'min', numToPlot = 5, unitsToPlot = [107], axlims = None)
    axlims_bad  = plot_pathlet(posTraj, sample_pathlets, unit_info, unit_selector = 'min', numToPlot = 5, unitsToPlot = [44] , axlims = axlims_good)
        
    pathDivergence_mean = np.mean(pathDivergence, axis = 0)
    shuffledPathDivergence_mean = np.mean(np.mean(divShuffle, axis = -1), axis = 0)
    
    return velTraj, posTraj
    
def plot_pathlet(posTraj, sample_pathlets, unit_info, unit_selector = 'max', numToPlot = 5, unitsToPlot = None, axlims = None):
    
    traj_auc = unit_info.pathlet_AUC.to_numpy()
    
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
            pathlets_min_xyz[plotIdx] = np.min(posTraj[..., unit], axis = 0)
            pathlets_max_xyz[plotIdx] = np.max(posTraj[..., unit], axis = 0)
        
        min_xyz = np.min(pathlets_min_xyz, axis = 0)
        max_xyz = np.max(pathlets_max_xyz, axis = 0)
    else:
        min_xyz = axlims[0]
        max_xyz = axlims[1]
    
    for unit in units:
        # title = '(%s) Unit %d' %(unit_selector, unit) 
        
        leadSamp = round(params.lead_to_analyze[0] / (params.lead_to_analyze[0] + params.lag_to_analyze[0]) * posTraj.shape[0])
        fig = plt.figure(figsize = (4.95, 4.5))
        ax = plt.axes(projection='3d')
        for sampPath in sample_pathlets:
            ax.plot3D(sampPath[:leadSamp + 1, 0, unit], sampPath[:leadSamp + 1, 1, unit], sampPath[:leadSamp + 1, 2, unit], 'blue')
            ax.plot3D(sampPath[leadSamp:    , 0, unit], sampPath[leadSamp:    , 1, unit], sampPath[leadSamp:    , 2, unit], 'red')
        ax.plot3D(posTraj[:leadSamp + 1, 0, unit], posTraj[:leadSamp + 1, 1, unit], posTraj[:leadSamp + 1, 2, unit], 'black', linewidth=3)
        ax.plot3D(posTraj[leadSamp:, 0, unit], posTraj[leadSamp:, 1, unit], posTraj[leadSamp:, 2, unit], 'black', linewidth=3)
        # ax.set_title(title, fontsize = 16, fontweight = 'bold')
        ax.set_xlim(min_xyz[0], max_xyz[0])
        ax.set_ylim(min_xyz[1], max_xyz[1])
        ax.set_zlim(min_xyz[2], max_xyz[2])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        # ax.set_xlabel('x', fontsize = params.axis_fontsize)
        # ax.set_ylabel('y', fontsize = params.axis_fontsize)
        # ax.set_zlabel('z', fontsize = params.axis_fontsize)
        ax.set_xlabel('', fontsize = params.axis_fontsize)
        ax.set_ylabel('', fontsize = params.axis_fontsize)
        ax.set_zlabel('', fontsize = params.axis_fontsize)
        # ax.legend(['lead', 'lag'], loc='upper right', bbox_to_anchor=(1, 1), fontsize = 14, shadow=False)
        ax.w_xaxis.line.set_color('black')
        ax.w_yaxis.line.set_color('black')
        ax.w_zaxis.line.set_color('black')
        ax.view_init(28, 148)
        plt.show()
        
        fig.savefig(os.path.join(path.plots, 'unit_%d_pathlet.png' % unit), bbox_inches='tight', dpi=params.dpi)

    if unitsToPlot is not None: 
        print(traj_auc[unitsToPlot[0]])
    
    return (min_xyz, max_xyz)
    
def compute_trajectory_correlations(velTraj, spike_data, unit_info, minAUC = 0.55):
    chan_map = spike_data['chan_map_ns6'].copy()

    tmp_unit_info = unit_info.copy()
    tmp_unit_info['x'] = np.full((unit_info.shape[0],), np.nan)
    tmp_unit_info['y'] = np.full((unit_info.shape[0],), np.nan)
    for ch in np.unique(tmp_unit_info.ns6_elec_id):
        array_pos = np.where(chan_map == ch)
        array_pos = [int(array_pos[1]), int(array_pos[0])]    
    
        tmp_unit_info.loc[tmp_unit_info.ns6_elec_id == ch, 'x'] = array_pos[0]
        tmp_unit_info.loc[tmp_unit_info.ns6_elec_id == ch, 'y'] = array_pos[1]
    
    correlation = []
    distance    = []
    for unitIdx1, (auc1, x1, y1) in enumerate(zip(tmp_unit_info['pathlet_AUC'], tmp_unit_info['x'], tmp_unit_info['y'])):
        for unitIdx2, (auc2, x2, y2) in enumerate(zip(tmp_unit_info['pathlet_AUC'], tmp_unit_info['x'], tmp_unit_info['y'])):    
            if unitIdx1 != unitIdx2 and auc1 > minAUC and auc2 > minAUC:
                vel1 = velTraj[..., unitIdx1].flatten()
                vel2 = velTraj[..., unitIdx2].flatten()
                correlation.append(np.corrcoef(vel1, vel2)[0, 1])
                distance.append(math.dist([x1, y1], [x2, y2]) * params.channel_sep_horizontal)
    
    nbin = 19
    bins = np.quantile(distance, np.linspace(0, 1,nbin+1))[:-1]
    corr_df = pd.DataFrame(data = zip(correlation, distance), columns = ['corr', 'dist'])            
    # corr_df['bin'] = pd.cut(corr_df['dist'], bins=bins, labels=False, right=False)
    corr_df['bin'], bins = pd.qcut(corr_df['dist'], nbin, labels=False, retbins = True)
    bin_centers = np.convolve(bins, np.ones(2), 'valid') / 2
    corr_df['dist_bin_center'] = bin_centers[corr_df['bin'].to_numpy(dtype=np.int8)]
    
    dist_counts = corr_df['dist_bin_center'].value_counts().sort_index()

    fig, ax = plt.subplots()
    sns.lineplot(data=corr_df, x='dist_bin_center', y='corr', err_style="bars", ci=95)
    
    # fig, ax = plt.subplots(figsize=params.map_figSize)
    # for x, y in zip(tmp_unit_info['x'], tmp_unit_info['y']):
    #     ax.annotate('(%d, %d)' % (x, y), (x, y))    
    # ax.vlines(np.arange(-0.5, 10.5, 1), -0.5, 9.5, colors='black')
    # ax.hlines(np.arange(-0.5, 10.5, 1), -0.5, 9.5, colors='black')
    # ax.set_xlim(-0.5, 9.5)
    # ax.set_ylim(-0.5, 9.5)
    # for axis in ['bottom','left']:
    #     ax.spines[axis].set_linewidth(1)
    #     ax.spines[axis].set_color('black')
    # for axis in ['top','right']:
    #     ax.spines[axis].set_linewidth(0)
    # ax.tick_params(width=0, length = 0, labelsize = 0)
    # ax.set_xlabel('Lateral'  , fontsize = params.axis_fontsize, fontweight = 'bold')
    # ax.set_ylabel('Posterior', fontsize = params.axis_fontsize, fontweight = 'bold')

    # ax.grid(False)

    # plt.show()
    
    return corr_df

def shuffle_FN(weights, rng, percentile=None, mode = 'weights'):

    shuffled_FN = weights.copy()
    if percentile is None:
        if mode == 'weights':
            rng.shuffle(shuffled_FN, axis = 1)
        elif mode == 'topology':
            rng.shuffle(shuffled_FN, axis = 1)
            rng.shuffle(shuffled_FN, axis = 0)
            
    else:
        if mode == 'weights':
            percentIdx = np.where(weights > np.percentile(weights, percentile))
            for presyn in np.unique(percentIdx[0]):
                postsyn = percentIdx[1][percentIdx[0] == presyn]
                shuffled_FN[presyn, postsyn] = weights[presyn, rng.permutation(postsyn)]     
        elif mode == 'topology':
            percentIdx = np.where(weights > np.percentile(weights, percentile))
            shuffled_FN[percentIdx[0], percentIdx[1]] = rng.permutation(shuffled_FN[percentIdx[0], percentIdx[1]])
    return shuffled_FN
            
def shuffle_network_features(FN, sampledSpikes, sample_info, rng, percentile=None, mode='weights'):
    
    FN_tmp = FN[params.FN_source]
    shuffled_weights_network_features = np.empty((sampledSpikes.shape[0], sampledSpikes.shape[1], params.networkFeatureBins))
    if 'split' in params.FN_source:
        for sampleNum, kinIdx in enumerate(sample_info['kinIdx']):
            if kinIdx % 2 == 0:
                weights = FN_tmp[1]
            else:
                weights = FN_tmp[0]
            
            shuffled_FN = shuffle_FN(weights, rng, percentile = percentile, mode = mode)     
            for leadBin in range(params.networkFeatureBins):
                shuffled_weights_network_features[:, sampleNum, leadBin] = shuffled_FN @ sampledSpikes[:, sampleNum, (params.networkSampleBins-1) - leadBin]                 
    else:
        weights = FN_tmp
        shuffled_FN = shuffle_FN(weights, rng, percentile = percentile, mode = mode)     

        for leadBin in range(params.networkFeatureBins):
            shuffled_weights_network_features[..., leadBin] = shuffled_FN @ sampledSpikes[..., (params.networkSampleBins-1) - leadBin] 

    return shuffled_weights_network_features 
                                             
# def test_model_with_shuffled_network_weights(unit_info, model_details, components, traj_features, sampledSpikes, sample_info, FN, first_percent=25, second_percent=10):
    
#     RNGs = {'train_test_split' : [np.random.default_rng(n) for n in range(params.numIters)],
#             'partial_traj'     : [np.random.default_rng(n) for n in range(1000,  1000+params.numIters)],
#             'spike_shuffle'    : [np.random.default_rng(n) for n in range(5000,  5000+sampledSpikes.shape[0])],
#             'weight_shuffle'   : [np.random.default_rng(n) for n in range(10000, 10000+params.numIters)]}   
#     full_idx = [idx for idx, name in enumerate(model_details['model_names']) if name =='full'][0]    
#     param_coefs = model_details['model_results'][full_idx]['param_coefs']
    
#     areaUnderROC_all            = np.empty((sampledSpikes.shape[0], params.numIters))
#     areaUnderROC_first_percent  = np.empty((sampledSpikes.shape[0], params.numIters))
#     areaUnderROC_second_percent = np.empty((sampledSpikes.shape[0], params.numIters))
#     for n, (split_rng, traj_rng, FN_rng) in enumerate(zip(RNGs['train_test_split'], RNGs['partial_traj'], RNGs['weight_shuffle'])):

#         shuffled_weights_network_features_first_percent  = shuffle_network_features(FN, sampledSpikes, sample_info, FN_rng, percentile=100-first_percent , mode='weights')
#         shuffled_weights_network_features_second_percent = shuffle_network_features(FN, sampledSpikes, sample_info, FN_rng, percentile=100-second_percent, mode='weights')
#         shuffled_weights_network_features_all            = shuffle_network_features(FN, sampledSpikes, sample_info, FN_rng, percentile=None              , mode='weights')        

#         testSpikes                  = []
#         testFeatures_first_percent  = []
#         testFeatures_second_percent = []
#         testFeatures_all            = []
#         predictions_all             = []
#         predictions_first_percent   = []
#         predictions_second_percent  = []
#         for unit, spikes in enumerate(sampledSpikes[..., -1]):
                        
#             spikeIdxs   = np.where(spikes >= 1)[0]
#             noSpikeIdxs = np.where(spikes == 0)[0]
                
#             idxs = np.union1d(spikeIdxs, noSpikeIdxs)
#             trainIdx = np.hstack((split_rng.choice(spikeIdxs  , size = round(params.trainRatio*len(spikeIdxs  )), replace = False), 
#                                   split_rng.choice(noSpikeIdxs, size = round(params.trainRatio*len(noSpikeIdxs)), replace = False)))
#             testIdx  = np.setdiff1d(idxs, trainIdx)
                
#             testSpikes.append(spikes[testIdx])            
#             testFeatures_first_percent.append(np.hstack((np.full((len(testIdx), 1), 1),
#                                                          traj_features[testIdx], 
#                                                          shuffled_weights_network_features_first_percent[unit, testIdx]))) 
#             testFeatures_second_percent.append(np.hstack((np.full((len(testIdx), 1), 1),
#                                                           traj_features[testIdx], 
#                                                           shuffled_weights_network_features_second_percent[unit, testIdx]))) 
#             testFeatures_all.append(np.hstack((np.full((len(testIdx), 1), 1),
#                                                traj_features[testIdx], 
#                                                shuffled_weights_network_features_all[unit, testIdx])))   

#             coefs = np.expand_dims(param_coefs[:, unit, n], axis=-1) 
            
#             predictions_all           .append(np.exp((testFeatures_all           [-1] @ coefs).flatten()))
#             predictions_first_percent .append(np.exp((testFeatures_first_percent [-1] @ coefs).flatten()))
#             predictions_second_percent.append(np.exp((testFeatures_second_percent[-1] @ coefs).flatten()))

            
#         # Test GLM --> area under ROC
        
#         # allHitProbs = []
#         # allFalsePosProbs = []
#         for unit, (preds_all, preds_first, preds_second) in enumerate(zip(predictions_all, 
#                                                                           predictions_first_percent, 
#                                                                           predictions_second_percent)):
#             print((n, unit))
#             preds = preds_all
#             thresholds = np.linspace(preds_all.min(), preds.max(), params.numThresh)            
#             hitProb = np.empty((len(thresholds),))
#             falsePosProb = np.empty((len(thresholds),))
#             # testSpikes[unit][testSpikes[unit] >= 1] = 1
#             for t, thresh in enumerate(thresholds):    
#                 posIdx = np.where(preds > thresh)
#                 hitProb[t] = np.sum(testSpikes[unit][posIdx] >= 1) / np.sum(testSpikes[unit] >= 1)
#                 falsePosProb[t] = np.sum(testSpikes[unit][posIdx] == 0) / np.sum(testSpikes[unit] == 0)
            
#             areaUnderROC_all[unit, n] = auc(falsePosProb, hitProb)
            
#             preds = preds_first
            
#             # fig, ax = plt.subplots()
#             # ax.plot(preds, 'bo')
#             # ax.set_title('Unit' + str(unit))
#             # tmp = np.array(testSpikes[unit], dtype=np.float16)
#             # # tmp[tmp == 0] = np.nan
#             # # tmp[~np.isnan(tmp)] = preds[~np.isnan(tmp)]
#             # ax.plot(tmp, 'o', c = 'orange')
#             # plt.show()
            
#             thresholds = np.linspace(preds_all.min(), preds.max(), params.numThresh)            
#             hitProb = np.empty((len(thresholds),))
#             falsePosProb = np.empty((len(thresholds),))
#             # testSpikes[unit][testSpikes[unit] >= 1] = 1
#             for t, thresh in enumerate(thresholds):    
#                 posIdx = np.where(preds > thresh)
#                 hitProb[t] = np.sum(testSpikes[unit][posIdx] >= 1) / np.sum(testSpikes[unit] >= 1)
#                 falsePosProb[t] = np.sum(testSpikes[unit][posIdx] == 0) / np.sum(testSpikes[unit] == 0)
            
#             areaUnderROC_first_percent[unit, n] = auc(falsePosProb, hitProb)       
            
#             preds = preds_second
#             thresholds = np.linspace(preds_all.min(), preds.max(), params.numThresh)            
#             hitProb = np.empty((len(thresholds),))
#             falsePosProb = np.empty((len(thresholds),))
#             # testSpikes[unit][testSpikes[unit] >= 1] = 1
#             for t, thresh in enumerate(thresholds):    
#                 posIdx = np.where(preds > thresh)
#                 hitProb[t] = np.sum(testSpikes[unit][posIdx] >= 1) / np.sum(testSpikes[unit] >= 1)
#                 falsePosProb[t] = np.sum(testSpikes[unit][posIdx] == 0) / np.sum(testSpikes[unit] == 0)
            
#             areaUnderROC_second_percent[unit, n] = auc(falsePosProb, hitProb)
    
#     unit_info['FN_all_shuffled_AUC']                         = np.mean(areaUnderROC_all           , axis = -1)
#     unit_info['FN_%d_percent_shuffled_AUC' % first_percent ] = np.mean(areaUnderROC_first_percent , axis = -1)
#     unit_info['FN_%d_percent_shuffled_AUC' % second_percent] = np.mean(areaUnderROC_second_percent, axis = -1)

    
#     return unit_info

def test_model_with_shuffled_network_weights(unit_info, model_details, components, traj_features, sampledSpikes, sample_info, FN, percent=None, mode = 'weights'):
    
    RNGs = {'train_test_split' : [np.random.default_rng(n) for n in range(params.numIters)],
            'partial_traj'     : [np.random.default_rng(n) for n in range(1000,  1000+params.numIters)],
            'spike_shuffle'    : [np.random.default_rng(n) for n in range(5000,  5000+sampledSpikes.shape[0])],
            'weight_shuffle'   : [np.random.default_rng(n) for n in range(10000, 10000+params.numIters)]}   
    full_idx = [idx for idx, name in enumerate(model_details['model_names']) if name =='full'][0]    
    param_coefs = model_details['model_results'][full_idx]['param_coefs']
    
    areaUnderROC  = np.empty((sampledSpikes.shape[0], params.numIters))
    for n, (split_rng, traj_rng, FN_rng) in enumerate(zip(RNGs['train_test_split'], RNGs['partial_traj'], RNGs['weight_shuffle'])):

        print((percent, n))        

        if percent is None:
            shuffled_weights_network_features = shuffle_network_features(FN, sampledSpikes, sample_info, FN_rng, percentile=None, mode='weights')        
        else:    
            shuffled_weights_network_features = shuffle_network_features(FN, sampledSpikes, sample_info, FN_rng, percentile=100-percent , mode='weights')

        testSpikes   = []
        testFeatures = []
        predictions  = []
        for unit, spikes in enumerate(sampledSpikes[..., -1]):
                        
            spikeIdxs   = np.where(spikes >= 1)[0]
            noSpikeIdxs = np.where(spikes == 0)[0]
                
            idxs = np.union1d(spikeIdxs, noSpikeIdxs)
            trainIdx = np.hstack((split_rng.choice(spikeIdxs  , size = round(params.trainRatio*len(spikeIdxs  )), replace = False), 
                                  split_rng.choice(noSpikeIdxs, size = round(params.trainRatio*len(noSpikeIdxs)), replace = False)))
            testIdx  = np.setdiff1d(idxs, trainIdx)
                
            testSpikes.append(spikes[testIdx])            

            testFeatures.append(np.hstack((np.full((len(testIdx), 1), 1),
                                           traj_features[testIdx], 
                                           shuffled_weights_network_features[unit, testIdx])))   

            coefs = np.expand_dims(param_coefs[:, unit, n], axis=-1) 
            
            predictions.append(np.exp((testFeatures[-1] @ coefs).flatten()))

            
        # Test GLM --> area under ROC
        
        # allHitProbs = []
        # allFalsePosProbs = []
        for unit, (preds) in enumerate(predictions):
            thresholds = np.linspace(preds.min(), preds.max(), params.numThresh)            
            hitProb = np.empty((len(thresholds),))
            falsePosProb = np.empty((len(thresholds),))
            # testSpikes[unit][testSpikes[unit] >= 1] = 1
            for t, thresh in enumerate(thresholds):    
                posIdx = np.where(preds > thresh)
                hitProb[t] = np.sum(testSpikes[unit][posIdx] >= 1) / np.sum(testSpikes[unit] >= 1)
                falsePosProb[t] = np.sum(testSpikes[unit][posIdx] == 0) / np.sum(testSpikes[unit] == 0)
            
            areaUnderROC[unit, n] = auc(falsePosProb, hitProb)
                        
            # fig, ax = plt.subplots()
            # ax.plot(preds, 'bo')
            # ax.set_title('Unit' + str(unit))
            # tmp = np.array(testSpikes[unit], dtype=np.float16)
            # # tmp[tmp == 0] = np.nan
            # # tmp[~np.isnan(tmp)] = preds[~np.isnan(tmp)]
            # ax.plot(tmp, 'o', c = 'orange')
            # plt.show()
            
    if percent is None:
        unit_info['FN_all_shuffled_AUC'] = np.mean(areaUnderROC, axis = -1)
    else:
        unit_info['FN_%d_percent_shuffled_AUC' % percent ] = np.mean(areaUnderROC, axis = -1)
        
    return unit_info
    
def plot_shuffled_network_results(auc_results, percent_values, plot_type = 'diff'):
    
    if plot_type == 'diff':
        # auc_results = 100 / 0.5 *(np.tile(np.expand_dims(auc_results.to_numpy()[:, 0], axis = 1), (1, auc_results.shape[1]-1)) - auc_results.to_numpy()[:, 1:]) / np.mean(auc_results.to_numpy()[:,0])
        auc_results=auc_results.to_numpy()
        auc_results = auc_results - 0.5
        auc_results = 100*(np.tile(np.expand_dims(auc_results[:, 0], axis = 1), (1, auc_results.shape[1]-1)) - auc_results[:, 1:]) / np.mean(auc_results[:,0])
        # auc_results = 100*(np.tile(np.expand_dims(auc_results.to_numpy()[:, 0], axis = 1), (1, auc_results.shape[1]-1)) - auc_results.to_numpy()[:, 1:]) / np.mean(auc_results.to_numpy()[:,0])

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.errorbar(percent_values[1:-1], 
                    np.mean(auc_results, axis = 0)[1:], 
                    yerr = (np.std(auc_results, axis=0) / np.sqrt(auc_results.shape[0]))[1:], 
                    linewidth=0,
                    elinewidth=3,
                    marker='o',
                    markersize=10)

        ax.errorbar(percent_values[0], 
                    np.mean(auc_results, axis = 0)[0], 
                    yerr = (np.std(auc_results, axis=0) / np.sqrt(auc_results.shape[0]))[0], 
                    linewidth=0,
                    elinewidth=3,
                    ecolor='black',
                    marker='o',
                    markersize=10,
                    markerfacecolor='black',
                    markeredgecolor='black')
        
        # ax.set_xlabel('Top Percent of Weights Shuffled', fontsize = params.axis_fontsize)
        # ax.set_ylabel('Percent AUC Loss', fontsize = params.axis_fontsize)
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_xticks(percent_values[:-1])
        ax.tick_params(width=2, length = 4, labelsize = params.tick_fontsize)
        # for tick in ax.get_xticks():
        # for tick in ax.get_xticklabels():
        #     tick.set_fontsize(params.tick_fontsize)
        # for tick in ax.get_yticklabels():
        #     tick.set_fontsize(params.tick_fontsize)
        sns.despine(ax=ax)
        ax.spines['bottom'].set_linewidth(params.axis_linewidth)
        ax.spines['left'  ].set_linewidth(params.axis_linewidth)
        
        plt.show()
        
        pvals = [0]*(len(percent_values) - 1)
        for idx, col in enumerate(range(1, len(pvals))):
            t, pvals[idx] = ttest_ind(auc_results[:, col], auc_results[:, 0], alternative='less')
        
        network_shuffle_pvals = pd.DataFrame(data = zip(percent_values[1:-1], pvals, np.mean(auc_results, axis = 0)[1:]),
                                             columns = ['percent_shuffled', 'pval', 'auc_loss'])
        
        fig.savefig(os.path.join(path.plots, 'network_weights_auc_loss.png'), bbox_inches='tight', dpi=params.dpi)
        
        return network_shuffle_pvals
    
    else:
        auc_results = np.hstack((auc_results.to_numpy()[:, 1:], np.expand_dims(auc_results.to_numpy()[:, 0], axis=1))) 
        fig, ax = plt.subplots()
        ax.errorbar(percent_values, np.mean(auc_results, axis = 0), yerr = np.std(auc_results, axis=0) / np.sqrt(auc_results.shape[0]))

        ax.set_xlabel('Top Percent of Weights Shuffled', fontsize = params.axis_fontsize)
        ax.set_ylabel('Area under ROC', fontsize = params.axis_fontsize)
        plt.show()
        
        pvals = [0]*(len(percent_values) - 1)
        for idx, col in enumerate(range(1, len(percent_values))):
            t, pvals[idx] = ttest_ind(auc_results[:, col], auc_results[:, 0], alternative='greater')
        
        network_shuffle_pvals = pd.DataFrame(data = zip(percent_values[1:], pvals),
                                             columns = ['percent_shuffled', 'pval'])
        return network_shuffle_pvals
    
if __name__ == "__main__":

    spike_data, kinematics, analog_and_video, FN = load_data()        

    all_models_data = load_models()    

# with open(r'C:/Users/Dalton/Documents/lab_files/analysis_encoding_model/intermediate_variable_storage/20210211_encoding_model_results_network_on_lead_100_lag_300_shift_50_PCAthresh_90_norm_off_NETWORK_SHUFFLES.pkl', 'rb') as f:
#         all_model_results, unit_info = dill.load(f)    

    unit_info, model_details, components, traj_features, sampledSpikes, sample_info = get_single_lead_lag_data(all_models_data, 
                                                                                                               params.lead_to_analyze[0], 
                                                                                                               params.lag_to_analyze[0])

    # plot_model_auc_comparison(unit_info, 'brief_AUC', 'pathlet_AUC', minauc=0.48)
    # plot_model_auc_comparison(unit_info, 'network_AUC', 'full_AUC')
    # plot_model_auc_comparison(unit_info, 'pathlet_AUC', 'full_AUC', minauc=0.48)
    # plot_model_auc_comparison(unit_info, 'network_partial_traj_AUC', 'full_AUC')
    # plot_model_auc_comparison(unit_info, 'pathlet_AUC', 'network_AUC')
    # plot_model_auc_comparison(unit_info, 'pathlet_AUC', 'network_partial_traj_AUC')
    
    # unit_info, loss_stats = compute_loss_statistics(unit_info)

    # plot_scatter_of_result_on_channel_map(unit_info, spike_data, jitter_radius = .15, key = 'full_AUC',    min_thresh = None)
    # plot_scatter_of_result_on_channel_map(unit_info, spike_data, jitter_radius = .15, key = 'pathlet_AUC', min_thresh = None)
    # plot_scatter_of_result_on_channel_map(unit_info, spike_data, jitter_radius = .15, key = 'network_AUC', min_thresh = None)
    # plot_scatter_of_result_on_channel_map(unit_info, spike_data, jitter_radius = .15, key = 'fr',    min_thresh = None)

    # plot_sample_kinematics(kinematics, event = 8)

    # sign_test, ttest = sig_tests(unit_info, 'pathlet_AUC', 'network_AUC', unit_info_reduced = None) 
    
    velTraj, posTraj = compute_and_analyze_pathlets(model_details, components, unit_info)
    
    correlation_df = compute_trajectory_correlations(velTraj, spike_data, unit_info, minAUC = 0.55)
    
    # cmax = plot_FN_heatmap(FN[params.FN_source][1], 'odd' , cmax = None)
    # cmax = plot_FN_heatmap(FN[params.FN_source][0], 'even', cmax = cmax)
    # cmax = plot_FN_heatmap(FN[params.FN_source][0], 'even', cmax = None)
    # cmax = plot_FN_heatmap(FN[params.FN_source][1], 'odd' , cmax = cmax)

    
    # percent_list = [None, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    # # for percent in percent_list:
    # #     unit_info = test_model_with_shuffled_network_weights(unit_info, model_details, components, traj_features, sampledSpikes, sample_info, FN, percent=percent, mode='weights')
    # #     with open(os.path.join(path.intermediate_save_path, '10pt0_ms_bins', 'unit_info_with_shuffled_network.pkl'), 'wb') as f:
    # #         dill.dump(unit_info, f, recurse=True)
    
    # with open(os.path.join(path.intermediate_save_path, '10pt0_ms_bins', 'unit_info_with_shuffled_network.pkl'), 'rb') as f:
    #     unit_info = dill.load(f)

    # percent_list[0] = 100
    # percent_list.append(0)
    # network_shuffle_pvals = plot_shuffled_network_results(unit_info.iloc[:, -11:], percent_list)
    # plot_model_auc_comparison(unit_info, 'FN_30_percent_shuffled_AUC', 'full_AUC')

