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

sys.path.insert(0, '/project/nicho/projects/marmosets/code_database/data_processing/nwb_tools/hatlab_nwb_tools/')
from hatlab_nwb_functions import remove_duplicate_spikes_from_good_single_units, get_sorted_units_and_apparatus_kinematics_with_metadata   

sys.path.insert(0, '/project/nicho/projects/marmosets/code_database/analysis/trajectory_encoding_model/')
from utils import get_interelectrode_distances_by_unit, choose_units_for_model


marmcode='TY'
FN_computed = True

# nwb_infile = '/project/nicho/projects/dalton/data/TY20210211_freeAndMoths-003_DM.nwb' 
# pkl_infile = '/project/nicho/projects/dalton/data/TY20210211_freeAndMoths-003_DM_encoding_model_regularized_results_30ms_shift_v4.pkl' #'/project/nicho/projects/dalton/data/TY20210211_freeAndMoths-003_DM_encoding_model_30ms_shift_results_v2.pkl'

if marmcode=='TY':
    if FN_computed:
        nwb_infile = '/project/nicho/projects/dalton/data/TY/TY20210211_freeAndMoths-003_resorted_20230612_DM_with_functional_networks.nwb' 
    else:
        nwb_infile = '/project/nicho/projects/dalton/data/TY/TY20210211_freeAndMoths-003_resorted_20230612_DM.nwb' 
    pkl_infile = '/project/nicho/projects/dalton/data/TY/TY20210211_freeAndMoths-003_resorted_20230612_DM_trajectory_shuffled_tortuosity_split_encoding_models_30ms_shift_v2.pkl' #'/project/nicho/projects/dalton/data/TY20210211_freeAndMoths-003_resorted_20230612_DM_encoding_model_sorting_corrected_30ms_shift_v4.pkl'
elif marmcode=='MG':
    nwb_infile   = '/project/nicho/projects/dalton/data/MG/MG20230416_1505_mothsAndFree-002_processed_DM.nwb'
    pkl_infile = '/project/nicho/projects/dalton/data/MG/MG20230416_1505_mothsAndFree-002_processed_DM_traj_TEST_trajectory_shuffled_encoding_models_30ms_shift_v2.pkl'

split_pattern = '_shift_v' # '_results_v'
base, ext = os.path.splitext(pkl_infile)
base, in_version = base.split(split_pattern)
out_version = str(int(in_version) + 1)  
pkl_outfile = base + split_pattern + out_version + ext

dataset_code = os.path.basename(pkl_infile)[:10] 
# plots = os.path.join(os.path.dirname(os.path.dirname(pkl_infile)), 'plots', dataset_code)
plots = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(pkl_infile))), 'plots', dataset_code)

shift_set = int(pkl_infile.split('ms_shift')[0][-2:])
  
class params:
    lead = 'all' #[0.5, 0.4, 0.4, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0  , 0  , 0  ] #[0.2, 0.15, 0.1, 0.05, 0] #[0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0] #[0.05] # 100ms lead time
    lag  = 'all' #[0  , 0  , 0.1, 0  , 0.1, 0.2, 0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.3, 0.4, 0.5] #[0.2, 0.25, 0.3, 0.35, 0.4] #[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4] #[0.35] # 300ms lag time
    best_lead_lag_key = 'lead_200_lag_300' #None
    
    if marmcode=='TY':
        # reorder = [0, 1, 3, 2, 4, 5, 6, 8, 13, 12, 14, 15, 16, 17, 18, 11,  7,  9, 10]
        reorder = [0, 1, 3, 2, 4, 5, 6, 12, 13, 16, 8,  14, 15, 11,  7,  9, 10]
    elif marmcode=='MG':
        reorder = [0]
    # reorder = [0, 1, 3, 2, 4, 5, 6, 7, 9, 8, 10, 11, 12, 13, 14, 15]
    # reorder = [0, 1, 3, 2, 4, 5, 6, 7, 8 , 9 , 10 ]

    FN_key = 'split_reach_FNs'
    frate_thresh = 2
    snr_thresh = 3
    significant_proportion_thresh = 0.95
    nUnits_percentile = 60
    # primary_traj_model = 'traj_and_avgPos'
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
    axis_fontsize = 24
    dpi = 300
    axis_linewidth = 2
    tick_length = 2
    tick_width = 1
    map_figSize = (6, 8)
    tick_fontsize = 18
    aucScatter_figSize = (7, 7)

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
    
    stats_df = pd.DataFrame(data = zip(p_ttest, p_signtest, prop_signtest, ci_signtest_lower),
                            columns = ['pval_t', 'pval_sign', 'proportion_sign', 'CI_lower'])
    
    return stats_df

def add_tuning_class_to_df(all_units_res, stats_df, stat_key, thresh, direction='greater'):
    
    # tuning = ['untuned']*all_units_res.shape[0]
    # for idx, stat in enumerate(stats_df[stat_key]):
    #     if (direction.lower() == 'greater' and stat >= thresh) or (direction.lower() == 'less' and stat <= thresh):
    #         tuning[idx] = 'tuned'

    # all_units_res['tuning'] = tuning
    all_units_res[stat_key] = stats_df[stat_key].values 
    
    return all_units_res

def determine_trajectory_significance(lead_lag_keys, plot = False):
    
    for lead_lag_key in lead_lag_keys:
        all_units_res = results_dict[lead_lag_key]['all_models_summary_results']    
        
        
        #['traj', 'traj_with_shuffled_spike_samples']
        shuffle_model = [key for key in results_dict[lead_lag_key]['model_results'].keys() if 'shuffle' in key][0]
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
                
        all_units_res = add_tuning_class_to_df(all_units_res, stats_df, 'proportion_sign', thresh = params.significant_proportion_thresh, direction='greater') 
        
        results_dict[lead_lag_key]['all_models_summary_results'] = all_units_res

            
def organize_results_by_model_for_all_lags(fig_mode, per=None):
    
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
                sign_prop_df[lead_lag_key] = all_units_res['proportion_sign']
            
        if 'traj_avgPos' == name:
            plt_title = name
            fig_df = pd.DataFrame()
            for idx, col in enumerate(tmp_results.columns):
                tmp_data = tmp_results[col]
                if fig_mode == 'percentile':
                    tmp_data = tmp_data[tmp_data > np.percentile(tmp_data, per)]
                    if idx == 0:
                        plt_title += f', Top {per}%'
                fig_df = pd.concat((fig_df, 
                                    pd.DataFrame(data=zip(tmp_data, 
                                                          np.repeat(col, tmp_data.shape[0]),
                                                          cortical_areas),
                                                 columns=['AUC', 'lead_lag_key', 'cortical_area'])),
                                   axis=0, ignore_index=True)
            
            
                                   
            leads = [re.findall(re.compile('lead_[0-9]{1,3}'), lead_lag_key)[0].split('_')[-1] for lead_lag_key in fig_df['lead_lag_key']] 
            lags  = [re.findall(re.compile('lag_[0-9]{1,3}' ), lead_lag_key)[0].split('_')[-1] for lead_lag_key in fig_df['lead_lag_key']]
            # lead_lag_ordering = [(-int(lead)+int(lag)) / 2 + (int(lead)+int(lag)) / 1000 for lead, lag in zip(leads, lags)]        
            fig_df['Trajectory Center (ms)'] = [(-int(lead)+int(lag)) / 2 for lead, lag in zip(leads, lags)]        
            fig_df['Trajectory Length (ms)'] = [( int(lead)+int(lag))     for lead, lag in zip(leads, lags)]
            
            fig, ax = plt.subplots(figsize=(6.5, 5), dpi = plot_params.dpi)
            sns.lineplot(ax=ax, data=fig_df, x = 'Trajectory Center (ms)', y='AUC', hue = 'Trajectory Length (ms)', 
                         linestyle='-', err_style='bars', errorbar=("se", 1), marker='o', markersize=10, palette='tab10')
            ax.set_title(plt_title)
            plt.show()
            
            if fig_mode == 'all':
                fig.savefig(os.path.join(plots, f'{name}_auc_over_leadlags_unfiltered.png'), bbox_inches='tight', dpi=plot_params.dpi)
            else:
                fig.savefig(os.path.join(plots, f'{name}_auc_over_leadlags_filtered_by_%s.png' % fig_mode), bbox_inches='tight', dpi=plot_params.dpi)
            
            rel = sns.relplot(data=fig_df, x = 'Trajectory Center (ms)', y='AUC', hue = 'Trajectory Length (ms)', col='cortical_area', 
                              linestyle='-.', kind='line', err_style='bars', errorbar=("se", 1), marker='o', markersize=10, palette='tab10')
            rel.fig.subplots_adjust(top=0.875) # adjust the Figure in rp
            rel.fig.suptitle(plt_title)
            plt.show()
            
            if fig_mode == 'all':
                rel.savefig(os.path.join(plots, f'{name}_auc_over_leadlags_unfiltered_sepByArea.png'), bbox_inches='tight', dpi=plot_params.dpi)
            else:
                rel.savefig(os.path.join(plots, f'{name}_auc_over_leadlags_filtered_by_%s_sepByArea.png' % fig_mode), bbox_inches='tight', dpi=plot_params.dpi)
            
            
            # rel = sns.relplot(data=fig_df, x = 'Trajectory Center (ms)', y='AUC', hue = 'Trajectory Length (ms)', col='Trajectory Length (ms)', 
            #                   linestyle='-.', kind='line', err_style='bars', errorbar=("se", 1), marker='o', markersize=10, palette='tab10')
            # rel.fig.subplots_adjust(top=0.875) # adjust the Figure in rp
            # rel.fig.suptitle(plt_title)
            # plt.show()

            # rel = sns.displot(data=fig_df, x = 'AUC', col='lead_lag_key', col_wrap=6,
            #                   kind='hist', bins=np.linspace(0.45, 0.75, 30))
            # rel.fig.subplots_adjust(top=0.875) # adjust the Figure in rp
            # rel.fig.suptitle(plt_title)
            # plt.show()

            # sns.histplot(data=fig_df, x = 'AUC', hue='lead_lag_key',bins=np.linspace(0.45, 0.75, 30))
            # plt.show()
                
                
        results.append(tmp_results)
    
    model_results_across_lags = {'model_name'    : model_names,
                                 'model_results' : results,
                                 'signtest_prop' : sign_prop_df}     
    
    return model_results_across_lags

def identify_optimal_lead_lag_by_unit(unit, uIdx, plot=False, average_all_peaks = False):

    auc_dist_by_lead_lag = pd.DataFrame() 
    auc_shuf_dist_by_lead_lag = pd.DataFrame()
    for llIdx, lead_lag_key in enumerate(results_dict.keys()):
        auc_dist = results_dict[lead_lag_key]['model_results'][params.primary_traj_model]['AUC'][uIdx]
        auc_dist_by_lead_lag[lead_lag_key] = auc_dist

        shuffle_model = [key for key in results_dict[lead_lag_key]['model_results'].keys() if 'shuffle' in key][0]
        auc_shuf_dist = results_dict[lead_lag_key]['model_results'][shuffle_model]['AUC'][uIdx]
        auc_shuf_dist_by_lead_lag[lead_lag_key] = auc_shuf_dist

    firstPeak = unit.idxmax()
    similar_peaks = []
    tmp_unit=unit.copy()
    tmp_unit.loc[firstPeak] = np.nan
    
    nextPeak = tmp_unit.idxmax()
    if type(firstPeak) != str:
        return np.nan, (np.nan, np.nan)
    elif type(nextPeak) != str:
        all_peaks = similar_peaks.copy()
        all_peaks.append(firstPeak)
    else:
        tmp, ttest_peaks = ttest_ind(auc_dist_by_lead_lag[firstPeak], auc_dist_by_lead_lag[nextPeak], alternative='greater')
        while ttest_peaks > 0.05:
            similar_peaks.append(nextPeak)
            tmp_unit.loc[nextPeak] = np.nan
            nextPeak = tmp_unit.idxmax()
            if type(nextPeak) != str:
                break
            tmp, ttest_peaks = ttest_ind(auc_dist_by_lead_lag[firstPeak], auc_dist_by_lead_lag[nextPeak], alternative='greater')
        all_peaks = similar_peaks.copy()
        all_peaks.append(firstPeak)

    all_peaks_idxs = [np.where(auc_dist_by_lead_lag.columns == peak_ll)[0][0] for peak_ll in all_peaks]
    peak_diffs = np.diff(sorted(all_peaks_idxs))

    if np.any(peak_diffs > 1):
        if average_all_peaks:
            pass
        else:
            return np.nan, (np.nan, np.nan)
    elif np.all(np.isnan(tmp_unit)):
        return np.nan, (np.nan, np.nan)
    
    lead_pattern = re.compile('lead_\d{1,3}')
    lag_pattern  = re.compile('lag_\d{1,3}')
    peaks_ll = [(int(re.findall(lead_pattern, leadlag)[0].split('lead_')[-1]), 
                 int(re.findall(lag_pattern, leadlag)[0].split('lag_')[-1])) 
                if type(leadlag) == str 
                else (np.nan, np.nan)
                for leadlag in all_peaks]   
    peaks_traj_center = [(-1*ll[0] + ll[1]) // 2 for ll in peaks_ll]
    opt_traj_center = np.mean(peaks_traj_center)
    opt_lead_lag = peaks_ll[np.argmin(abs(peaks_traj_center - opt_traj_center))]
    
    if plot:
        # make sns stripplot
        model_list = []
        auc_list = []
        lead_lag_list = []
        for lead_lag_key, auc_vals in auc_dist_by_lead_lag.iteritems():
            auc_list.extend(auc_vals)
            model_list.extend([params.primary_traj_model for idx in range(auc_vals.shape[0])])
            lead_lag_list.extend([lead_lag_key for idx in range(auc_vals.shape[0])])
        for lead_lag_key, auc_vals in auc_shuf_dist_by_lead_lag.iteritems():
            auc_list.extend(auc_vals)
            model_list.extend([shuffle_model for idx in range(auc_vals.shape[0])])
            lead_lag_list.extend([lead_lag_key for idx in range(auc_vals.shape[0])])    
        
        auc_dist_df = pd.DataFrame(data    = zip(model_list, lead_lag_list, auc_list),
                                   columns = ['model', 'lead_lag', 'auc']) 
        
        # fig, ax = plt.subplots(figsize=(12, 6))
        # sns.stripplot(ax=ax, data=auc_dist_df, x='lead_lag', y='auc', hue='model', dodge=True)
        # ax.get_legend().remove()
        # ax.set_title('UnitIdx = %d' % uIdx)
        # plt.show()
        
        fig, ax = plt.subplots(figsize=(6, 6), dpi=plot_params.dpi)
        sns.pointplot(ax=ax, data=auc_dist_df, x='lead_lag', y='auc', hue='model', errorbar='se', dodge=True, scale=.75)
        ax.set_title('UnitIdx = %d' % uIdx)
        ax.plot([idx for idx, key in enumerate(auc_dist_by_lead_lag.columns) if key == firstPeak][0], 
                auc_dist_by_lead_lag[firstPeak].mean(),
                marker='o', mfc='black', mec='black', ms=6)
        # similar_idxs = [idx for idx, key in enumerate(auc_dist_by_lead_lag.columns) if key in similar_peaks]
        similar_idxs = [np.where(auc_dist_by_lead_lag.columns == peak_ll)[0] for peak_ll in similar_peaks]
    
        similar_aucs = [auc_dist_by_lead_lag.loc[:, peak_ll].mean() for peak_ll in similar_peaks]
        ax.errorbar(similar_idxs, 
                    similar_aucs, 
                    marker='o', mfc='red', mec='red', ms=6, linestyle='')
        ax.get_legend().remove()
        plt.show()
        
        fig.savefig(os.path.join(plots, 'optimal_lead_lag_single_unit_%s.png' % str(uIdx).zfill(3)), bbox_inches='tight', dpi=plot_params.dpi)
    
    return opt_traj_center, opt_lead_lag        

def get_normalized_auc_distributions_across_lags(traj_results, normalize=True, percentile=None):
    
    all_units_scaled_auc_dist_by_ll = pd.DataFrame()
    cortical_area = []
    
    for uIdx, unit_means in traj_results.iterrows():
        if np.all(np.isnan(unit_means)):
            print(uIdx)
            continue
        auc_dist_by_lead_lag = pd.DataFrame() 
        for llIdx, lead_lag_key in enumerate(results_dict.keys()):
            auc_dist = results_dict[lead_lag_key]['model_results'][params.primary_traj_model]['AUC'][uIdx]
            if normalize:
                auc_dist = (auc_dist - np.nanmin(auc_dist)) / (np.nanmax(auc_dist) - np.nanmin(auc_dist))
            auc_dist_by_lead_lag[lead_lag_key] = auc_dist    
        
        all_units_scaled_auc_dist_by_ll = pd.concat([all_units_scaled_auc_dist_by_ll, auc_dist_by_lead_lag], axis=0, ignore_index=True)
        unit_cortical_area = results_dict[lead_lag_key]['all_models_summary_results']['cortical_area'][uIdx]
        cortical_area.extend([unit_cortical_area for idx in range(auc_dist_by_lead_lag.size)])
    
    return all_units_scaled_auc_dist_by_ll, cortical_area


def find_optimal_lag_averaged_over_brain_area(model_results_across_lags, only_tuned = True, plot = False, normalize=True, percentile=None):
    modelIdx = [idx for idx, name in enumerate(model_results_across_lags['model_name']) if name == params.primary_traj_model][0]
    traj_results = model_results_across_lags['model_results'][modelIdx]
    
    if only_tuned and percentile:
        raise Exception('\n\nIn function "find_optimal_lag_averaged_over_brain_area", if only_tuned=True then you MUST set percentile=None. Both options being active is not accepted.\n\n')
    elif only_tuned:
        traj_results = traj_results[model_results_across_lags['signtest_prop'] >= params.significant_proportion_thresh]
    elif percentile is not None:
        nUnits = int(np.floor((100-percentile)/100 * traj_results.shape[0]))
        for lead_lag_key in traj_results.columns:
            ll_signtest_prop = model_results_across_lags['signtest_prop'][lead_lag_key].copy()
            ll_signtest_prop.loc[ll_signtest_prop < np.percentile(ll_signtest_prop, percentile)] = np.nan
            # np.sum(~np.isnan(ll_signtest_prop))
            if np.sum(~np.isnan(ll_signtest_prop)) > nUnits:
                ll_signtest_prop.sort_values(ascending=False, inplace=True)
                ll_signtest_prop.iloc[nUnits:] = np.nan
                ll_signtest_prop.sort_index(inplace=True)
            traj_results.loc[np.isnan(ll_signtest_prop), lead_lag_key] = np.nan
            print(np.sum(~np.isnan(traj_results[lead_lag_key])))
    
    all_units_scaled_auc_dist_by_ll, cortical_area = get_normalized_auc_distributions_across_lags(traj_results, normalize=normalize)
    
    
    if plot:
        auc_list = []
        lead_lag_list = []
        for lead_lag_key, auc_vals in all_units_scaled_auc_dist_by_ll.iteritems():
            auc_list.extend(auc_vals)
            lead_lag_list.extend([lead_lag_key for idx in range(auc_vals.shape[0])])   
        
        auc_dist_df = pd.DataFrame(data    = zip(lead_lag_list, auc_list, cortical_area),
                                   columns = ['lead_lag', 'auc', 'cortical_area']) 
        
        # fig, ax = plt.subplots(figsize=(12, 6))
        # sns.stripplot(ax=ax, data=auc_dist_df, x='lead_lag', y='auc', hue='cortical_area', dodge=True)
        # ax.set_title('')
        # plt.show()
        
        sns.catplot(data=auc_dist_df, x='lead_lag', y='auc', col ='cortical_area', errorbar='se', kind='point', scale=.75)
        # ax.get_legend().remove()
        fig = plt.gcf()
        plt.show()
    
        
        
        # for area in np.unique(auc_dist_df.cortical_area):
        #     tmp_auc_dist_df = auc_dist_df.loc[auc_dist_df['cortical_area'] == area, :]
        #     fig, ax = plt.subplots(figsize=(6, 6))
        #     sns.pointplot(ax=ax, data=tmp_auc_dist_df, x='lead_lag', y='auc', errorbar='se', dodge=True, scale=.75)
        #     ax.set_title(area)
        #     # ax.get_legend().remove()
        #     plt.show()

        if normalize:
            fig.savefig(os.path.join(plots, 'brain_area_lead_lag_grand_averages_normalized.png'), bbox_inches='tight', dpi=plot_params.dpi)
        else:
            fig.savefig(os.path.join(plots, 'brain_area_lead_lag_grand_averages.png'), bbox_inches='tight', dpi=plot_params.dpi)
    
def find_optimal_lag_for_each_unit(model_results_across_lags, only_tuned = True, plot=False, average_all_peaks = False):
    modelIdx = [idx for idx, name in enumerate(model_results_across_lags['model_name']) if name == params.primary_traj_model][0]
    traj_results = model_results_across_lags['model_results'][modelIdx]
    
    if only_tuned:
        traj_results = traj_results[model_results_across_lags['signtest_prop'] >= params.significant_proportion_thresh]
    
    optimal_traj_center = [None for idx in range(traj_results.shape[0])]
    optimal_lead_lag    = [None for idx in range(traj_results.shape[0])]
    unit_df_idx         = [None for idx in range(traj_results.shape[0])] 
    list_idx = 0
    for uIdx, unit in traj_results.iterrows():
        if uIdx in [5, 10, 15, 20]:
            optimal_traj_center[list_idx], optimal_lead_lag[list_idx] = identify_optimal_lead_lag_by_unit(unit, uIdx, plot=plot, average_all_peaks = average_all_peaks) 
        else:
            optimal_traj_center[list_idx], optimal_lead_lag[list_idx] = identify_optimal_lead_lag_by_unit(unit, uIdx, plot=False, average_all_peaks = average_all_peaks) 
            
        unit_df_idx[list_idx] = uIdx
        list_idx+=1
        
    # if peak and trough not sig diff, throw out of this analysis
    # check if peaks are close (maybe by measuring distirbution differences) and then cehck if variance around peak is significantly different
    
    # second, average allindividual unit  lag pltos by column or by area, find peak of average (normalize each to 0-1 range before computing average across units) 
    # unit = 9
    # plt.plot(traj_results.iloc[unit, :])
    # plt.show()
    
    # optimal_lead_lag = traj_results.idxmax(axis = 1)
    # tmp_traj_results = traj_results.copy()
    # for row, col in optimal_lead_lag.iteritems():
    #     tmp_traj_results.loc[row, col] = np.nan
    # second_lead_lag = tmp_traj_results.idxmax(axis = 1)
    # optimal_idx_num = [np.where(traj_results.columns == opt_ll)[0] for opt_ll in optimal_lead_lag.values] 
    # second_idx_num = [np.where(traj_results.columns == opt_ll)[0] for opt_ll in second_lead_lag.values]
    
    # keep_mask = pd.Series([True if opt.size>0 and sec.size>0 and np.abs(opt-sec) <= 2 else False for opt, sec in zip(optimal_idx_num, second_idx_num)])
    # # optimal_lead_lag.loc[~keep_mask] = np.nan

    # lead_pattern = re.compile('lead_\d{1,3}')
    # lag_pattern  = re.compile('lag_\d{1,3}')
    # optimal_lead_lag = [(int(re.findall(lead_pattern, leadlag)[0].split('lead_')[-1]), 
    #                      int(re.findall(lag_pattern, leadlag)[0].split('lag_')[-1])) 
    #                     if type(leadlag) == str 
    #                     else (np.nan, np.nan)
    #                     for leadlag in optimal_lead_lag]   
    # optimal_traj_center = [(-1*ll[0] + ll[1]) // 2 for ll in optimal_lead_lag]


    tmp_lead_lag_key = list(results_dict.keys())[0]
    units_res_optimal_lag = results_dict[tmp_lead_lag_key]['all_models_summary_results'].copy()
    units_res_optimal_lag[['lead', 'lag', 'traj_center']] = np.full((units_res_optimal_lag.shape[0], 3), np.nan)
    for idx, (opt_ll, opt_center) in enumerate(zip(optimal_lead_lag, optimal_traj_center)):
        try:
            tmp_lead_lag_key = 'lead_%d_lag_%d' % (opt_ll[0], opt_ll[1])
            tmp_units_res = results_dict[tmp_lead_lag_key]['all_models_summary_results']
            columns_with_auc_or_tuning = [idx for idx, col in enumerate(tmp_units_res.columns)         if 'auc' in col.lower() or 'proportion_sign' in col.lower()]
            columns_in_opt_ll_res      = [idx for idx, col in enumerate(units_res_optimal_lag.columns) if 'auc' in col.lower() or 'proportion_sign' in col.lower()]
            units_res_optimal_lag.iloc[idx, columns_in_opt_ll_res] = tmp_units_res.iloc[idx, columns_with_auc_or_tuning]
        except:
            pass
        units_res_optimal_lag.at[units_res_optimal_lag.index[idx], 'lead']    = opt_ll[0]
        units_res_optimal_lag.at[units_res_optimal_lag.index[idx], 'lag' ]    = opt_ll[1]
        units_res_optimal_lag.at[units_res_optimal_lag.index[idx], 'traj_center'] = opt_center
        
    return units_res_optimal_lag

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
    
def plot_optimal_lag_on_channel_map(units_res, jitter_radius = .15, hueKey = 'traj_center', 
                                    sizeKey = 'pathlet_AUC', weighted_mean = False,
                                    weightsKey = None, filtered=True):
    
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
    
    x_vals = np.unique(scatter_units_res.x)
    mean_traj_centers = []
    sem_traj_centers = []
    if weightsKey is None:
        weightsKey = sizeKey 
    for x in x_vals:
        traj_centers = scatter_units_res.loc[scatter_units_res['x'] == x, 'traj_center']
        if weighted_mean:
            weights = scatter_units_res.loc[scatter_units_res['x'] == x, weightsKey]
            idxs = traj_centers.index[~np.isnan(traj_centers)]
            weights = weights[idxs]
            traj_centers = traj_centers[idxs]
            weighted_average = np.average(a = traj_centers, weights = weights)
            mean_traj_centers.append(weighted_average)    
        else:
            mean_traj_centers.append(traj_centers.mean())
        sem_traj_centers.append(traj_centers.sem())
    
    fig, (ax_top, ax) = plt.subplots(2, 1, figsize=plot_params.map_figSize, gridspec_kw={'height_ratios': [1.25, 4]})
    sns.scatterplot(ax = ax, data = scatter_units_res, x = 'scatter_x', y = 'scatter_y', 
                    size = sizeKey, hue = hueKey, style = "quality", palette='seismic',
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
    ax_top.set_ylim([-50, 100])
    ax_top.set_yticks([-50, 0, 50, 100])
    ax_top.set_yticklabels([-50, 0, 50, 100], fontsize = plot_params.tick_fontsize)
    ax_top.set_ylabel('Traj Center (ms)', fontsize = plot_params.tick_fontsize)
    
    # for txt, x, y, scat_x, scat_y in zip(scatter_units_res['ns6_elec_id'], scatter_units_res['center_x'], scatter_units_res['center_y'],
    #                      scatter_units_res['scatter_x'], scatter_units_res['scatter_y']):
    #     print((txt, x, y))
    #     ax.annotate('%d' % txt, (x, y))
    plt.show()

    if filtered:
        fig.savefig(os.path.join(plots, '%s_%dms_shift_%d_sigThresh_%s_map.png' % (params.primary_traj_model, shift_set, int(params.significant_proportion_thresh*1e2), hueKey)), bbox_inches='tight', dpi=plot_params.dpi)
    else:
        fig.savefig(os.path.join(plots, '%s_%dms_shift_ALL_%s_map.png' % (params.primary_traj_model, shift_set, hueKey)), bbox_inches='tight', dpi=plot_params.dpi)

        
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
    ax.tick_params(width=2, length = 4, labelsize = plot_params.tick_fontsize)
    # ax.set_xticks(traj_mean_performance['reorder'])
    # for tick in ax.get_xticklabels():
    #     tick.set_fontsize(plot_params.tick_fontsize)
    # for tick in ax.get_yticklabels():
    #     tick.set_fontsize(plot_params.tick_fontsize)
    ax.set_xticks(traj_mean_performance['reorder'])
    ax.set_xticklabels(xticklabels, rotation=90)

    sns.despine(ax=ax)
    ax.spines['bottom'].set_linewidth(plot_params.axis_linewidth)
    ax.spines['left'  ].set_linewidth(plot_params.axis_linewidth)
    
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
            if model_key == 'shortTraj_high_tortuosity':
                stop = []
            col_name = '%s_auc' % model_key
            if col_name not in all_units_res.columns: 
                all_units_res[col_name] = results_dict[lead_lag_key]['model_results'][model_key]['AUC'].mean(axis=-1)
            else:
                print('This model (%s, %s) has already been summarized in the all_models_summary_results dataframe' % (lead_lag_key, model_key))
        
        all_units_res = add_cortical_area_to_units_results_df(all_units_res, cortical_bounds=params.cortical_boundaries)

        results_dict[lead_lag_key]['all_models_summary_results'] = all_units_res
        
def plot_model_auc_comparison(units_res, x_key, y_key, minauc = 0.5, targets=None):
    
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
        plot_name = plot_name + '.png'
    else:
        plot_title = 'Targets: All units'
        plot_name = 'area_under_curve_%s_%s.png' % (x_key, y_key)
    
    fig, ax = plt.subplots(figsize = plot_params.aucScatter_figSize, dpi = plot_params.dpi)
    # sns.scatterplot(ax = ax, data = units_res_plots, x = x_key, y = y_key, 
    #                 hue = "fr", style = "group")
    # sns.scatterplot(ax = ax, data = units_res_plots, x = x_key, y = y_key, 
    #                 style = "quality", s = 60, legend=False)
    sns.scatterplot(ax = ax, data = units_res_plots, x = x_key, y = y_key, 
                    hue = "snr", s = 60, legend=False)    
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
    # ax.legend(bbox_to_anchor=(1.5, 1.5), loc='upper left', borderaxespad=0)
    plt.show()
    
    fig.savefig(os.path.join(plots, plot_name), bbox_inches='tight', dpi=plot_params.dpi)


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

# def plot_model_auc_comparison(units_res, x_key, y_key, minauc = 0.5, prop_thresh=None, targets=None):
    
#     if prop_thresh is not None:
#         units_res = units_res.loc[units_res.proportion_sign >= prop_thresh, :]
    
#     fig, ax = plt.subplots(figsize = plot_params.aucScatter_figSize)
#     # sns.scatterplot(ax = ax, data = units_res, x = x_key, y = y_key, 
#     #                 hue = "fr", style = "group")
#     sns.scatterplot(ax = ax, data = units_res, x = x_key, y = y_key, 
#                     style = "quality", s = 60, legend=False)
#     ax.plot(np.arange(minauc, 1.0, 0.05), np.arange(minauc, 1.0, 0.05), '--k')
#     # ax.scatter(units_res[x_key].to_numpy()[44] , units_res[y_key].to_numpy()[44] , s = 60, c ='red', marker='x')
#     # ax.scatter(units_res[x_key].to_numpy()[107], units_res[y_key].to_numpy()[107], s = 60,  c ='red', marker='o')
#     ax.set_xlim(minauc, 1)
#     ax.set_ylim(minauc, 1)
#     for axis in ['bottom','left']:
#         ax.spines[axis].set_linewidth(2)
#         ax.spines[axis].set_color('black')
#     for axis in ['top','right']:
#         ax.spines[axis].set_linewidth(0)
#     ax.tick_params(width=2, length = 4, labelsize = plot_params.tick_fontsize)
#     ax.set_xlabel('ROC area (%s)' % x_key[:-4], fontsize = plot_params.axis_fontsize)
#     ax.set_ylabel('ROC area (%s)' % y_key[:-4], fontsize = plot_params.axis_fontsize)
#     # ax.set_xlabel('')
#     # ax.set_ylabel('')
#     ax.grid(False)
#     # ax.legend(bbox_to_anchor=(1.5, 1.5), loc='upper left', borderaxespad=0)
#     plt.show()
    
#     # if targets is None:
#     #     fig.savefig(os.path.join(path.plots, 'area_under_curve_%s_%s.png' % (x_key, y_key)), bbox_inches='tight', dpi=plot_params.dpi)
#     # else:
#     #     fig.savefig(os.path.join(path.plots, 'area_under_curve_%s_%s_%s_targetUnits.png' % (x_key, y_key, targets)), bbox_inches='tight', dpi=plot_params.dpi)

# def compute_and_analyze_pathlets_on_PCA_models(lead_lag_key):
#     coefs = results_dict[lead_lag_key]['model_results']['coefs']
#     comps = results_dict[lead_lag_key]['model_features']['traj_PCA_components']
#     beta = np.mean(coefs, axis = -1)[1:np.shape(comps)[-1]+1, :]
#     velTraj = comps @ beta
#     velTraj = np.swapaxes(velTraj.reshape((params.nDims, int(np.shape(velTraj)[0] / params.nDims), np.shape(velTraj)[-1])), 0, 1)

#     posTraj = cumtrapz(velTraj, dx = (params.lag_to_analyze[0] + params.lead_to_analyze[0]) / np.shape(velTraj)[0], axis = 0, initial = 0)
#     dist = simps(np.linalg.norm(velTraj, axis = 1), dx = (params.lag_to_analyze[0] + params.lead_to_analyze[0]) / np.shape(velTraj)[0], axis = 0)
    
#     pathDivergence = np.empty(np.shape(coefs[0, ...].transpose()))
#     sample_pathlets = []
#     for samp in range(np.shape(coefs)[-1]):
#         beta_samp = coefs[1:np.shape(comps)[-1] +1, :, samp]
#         velTraj_samp = comps @ beta_samp
#         velTraj_samp = np.swapaxes(velTraj_samp.reshape((params.nDims, int(np.shape(velTraj_samp)[0] / params.nDims), np.shape(velTraj_samp)[-1])), 0, 1)
#         posTraj_samp = cumtrapz(velTraj_samp, dx = (params.lag[0] + params.lead[0]) / np.shape(velTraj_samp)[0], axis = 0, initial = 0)
#         sample_pathlets.append(posTraj_samp)
#         pathDivergence[samp, :] = np.sum(np.linalg.norm(posTraj - posTraj_samp, axis = 1), axis = 0)
        
#         divShuffle = np.empty((np.shape(pathDivergence)[0], np.shape(pathDivergence)[1], 100))
#         for shuffle in range(100):
#             idx = np.random.choice(np.arange(np.shape(pathDivergence)[1]), size = np.shape(pathDivergence)[1], replace = 0)
#             while np.sum(idx == np.arange(np.shape(pathDivergence)[1])) > 0:
#                 idx = np.random.choice(np.arange(np.shape(pathDivergence)[1]), size = np.shape(pathDivergence)[1], replace = 0)
    
#             divShuffle[samp, :, shuffle] = np.sum(np.linalg.norm(posTraj[..., idx] - posTraj_samp, axis = 1), axis = 0)
    
#     # axlims_best  = plot_pathlet(posTraj, sample_pathlets, unit_info, unit_selector = 'max', numToPlot = 1, unitsToPlot = None)
#     # axlims_worst = plot_pathlet(posTraj, sample_pathlets, unit_info, unit_selector = 'min', numToPlot = 1, unitsToPlot = None, axlims = axlims_best)
#     axlims_good = plot_pathlet(posTraj, sample_pathlets, unit_info, unit_selector = 'min', numToPlot = 5, unitsToPlot = [107], axlims = None)
#     axlims_bad  = plot_pathlet(posTraj, sample_pathlets, unit_info, unit_selector = 'min', numToPlot = 5, unitsToPlot = [44] , axlims = axlims_good)
        
#     pathDivergence_mean = np.mean(pathDivergence, axis = 0)
#     shuffledPathDivergence_mean = np.mean(np.mean(divShuffle, axis = -1), axis = 0)
    
#     return velTraj, posTraj

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
        axis_mins = np.empty((3, posTraj_mean.shape[-1]))
        axis_maxs = np.empty_like(axis_mins)
        for unit in range(posTraj_mean.shape[-1]):
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
        ax2.set_zlabel('z', fontsize = plot_params.axis_fontsize)            # ax.set_xlabel('', fontsize = plot_params.axis_fontsize)
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
        
        ax1.w_xaxis.line.set_color('black')
        ax1.w_yaxis.line.set_color('black')
        ax1.w_zaxis.line.set_color('black')
        ax1.view_init(28, 148)
        
        ax2.w_xaxis.line.set_color('black')
        ax2.w_yaxis.line.set_color('black')
        ax2.w_zaxis.line.set_color('black')
        ax2.view_init(28, 148)
        
        plt.show() 
        
        fig1.savefig(os.path.join(plots, 'all_units_pathlets_noPos.png'), bbox_inches='tight', dpi=plot_params.dpi)
        fig2.savefig(os.path.join(plots, 'all_units_pathlets_withPos.png'), bbox_inches='tight', dpi=plot_params.dpi)


    if unitsToPlot is not None: 
        print(traj_auc[unitsToPlot[0]])
    
    return (min_xyz, max_xyz)

def compute_and_analyze_trajectory_correlations(units_res, posTraj_mean, velTraj_mean, electrode_distances, lead_lag_key, FN=None, mode = 'concat', nplots=5):
    
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
    
    # nbin = 15
    # bins = np.quantile(df['Distance'], np.linspace(0, 1,nbin+1))[:-1]
    # df['bin'], bins = pd.qcut(df['Distance'], nbin, labels=False, retbins = True)
    # bin_centers = np.convolve(bins, np.ones(2), 'valid') / 2
    # df['dist_bin_center'] = np.round(bin_centers[df['bin'].to_numpy(dtype=np.int8)], 0)
    
    # dist_counts = corr_df['dist_bin_center'].value_counts().sort_index()

    fig, ax = plt.subplots(figsize = (4, 4), dpi = plot_params.dpi)
    sns.scatterplot(ax = ax, data = df, x = 'Pearson_corr', y = 'Wji', s = 20, legend=True) 
    plt.show()
    fig.savefig(os.path.join(plots, 'wji_vs_pearson_r.png'), bbox_inches='tight', dpi=plot_params.dpi)
    
    fig, ax = plt.subplots(figsize = (4, 4), dpi = plot_params.dpi)
    sns.scatterplot(ax = ax, data = df, x = 'r_squared', y = 'Wji', s = 20, legend=True) 
    plt.show()
    fig.savefig(os.path.join(plots, 'wji_vs_pearson_rsquare.png'), bbox_inches='tight', dpi=plot_params.dpi)
    # fig, ax = plt.subplots(figsize = (4, 4), dpi = plot_params.dpi)
    # sns.scatterplot(ax = ax, data = df, x = 'VelTraj_corr', y = 'Wji', s = 20, legend=True) 
    # plt.show()
    # fig, ax = plt.subplots(figsize = (4, 4), dpi = plot_params.dpi)
    # sns.pointplot(ax=ax, data = df, x = 'connect', y = 'r_squared', color='black', errorbar=('ci', 99))
    # plt.show()

    # fig, ax = plt.subplots(figsize = (4, 4), dpi = plot_params.dpi)
    # sns.pointplot(ax=ax, data = df, x = 'dist_bin_center', y = 'Pearson_Corr', color='black', errorbar=('ci', 99))
    # plt.show()
    fig, ax = plt.subplots(figsize = (4, 4), dpi = plot_params.dpi)
    sns.pointplot(ax=ax, data = df, x = 'Connection', y = 'Pearson_corr', color='black', errorbar=('ci', 99))
    plt.show()
    fig.savefig(os.path.join(plots, 'pearson_r_vs_connection.png'), bbox_inches='tight', dpi=plot_params.dpi)

    fig, ax = plt.subplots(figsize = (4, 4), dpi = plot_params.dpi)
    sns.pointplot(ax=ax, data = df, x = 'Connection', y = 'r_squared', color='black', errorbar='se')
    plt.show()
    fig.savefig(os.path.join(plots, 'pearson_rsquare_vs_connection.png'), bbox_inches='tight', dpi=plot_params.dpi)

    fig, ax = plt.subplots(figsize = (4, 4), dpi = plot_params.dpi)
    sns.pointplot(ax=ax, data = df, x = 'Connection', y = 'Wji', color='black', errorbar='se')
    plt.show()
    fig.savefig(os.path.join(plots, 'wji_vs_connection.png'), bbox_inches='tight', dpi=plot_params.dpi)

    fig, ax = plt.subplots(figsize = (4, 4), dpi = plot_params.dpi)
    sns.histplot(ax=ax, data = df, x = 'Pearson_corr', color='black', kde=True)
    plt.show()
    fig.savefig(os.path.join(plots, 'pearson_r_histogram.png'), bbox_inches='tight', dpi=plot_params.dpi)
    
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
    
    with NWBHDF5IO(nwb_infile, 'r') as io:
        nwb = io.read()

        reaches_key = [key for key in nwb.intervals.keys() if 'reaching_segments' in key][0]
        
        units, reaches, kin_module = get_sorted_units_and_apparatus_kinematics_with_metadata(nwb, reaches_key, plot=False) 
        
        units = choose_units_for_model(units, quality_key = 'snr', quality_thresh = params.snr_thresh, frate_thresh = params.frate_thresh)
        # units = choose_units_for_model(units, quality_key='amp', quality_thresh=5, frate_thresh=params.frate_thresh)
        
        if FN_computed:
            spontaneous_FN = nwb.scratch['spontaneous_FN'].data[:]
            reach_FN = nwb.scratch[params.FN_key].data[:] 
        
    with open(pkl_infile, 'rb') as f:
        results_dict = dill.load(f)
        
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
        
    model_results_across_lags = organize_results_by_model_for_all_lags(fig_mode='percentile', per=60)
    # model_results_across_lags = organize_results_by_model_for_all_lags(fig_mode='all')
    
    model_results_across_lags = compute_mean_model_performance(model_results_across_lags, percent = params.nUnits_percentile, percentile_mode = 'per_lag_set')
    
    # plot_sweep_over_lead_lag(model_results_across_lags, filter_key = 'percentile')
    # plot_sweep_over_lead_lag(model_results_across_lags, filter_key = 'tuned')
    # plot_sweep_over_lead_lag(model_results_across_lags, filter_key = None)
    
    # units_res_optimal_lag = find_optimal_lag_for_each_unit(model_results_across_lags, only_tuned=True, plot=False, average_all_peaks = True)
    # plot_mean_traj_center_by_area(units_res_optimal_lag, weighted_mean = False, weightsKey='%s_auc' % params.primary_traj_model)
    
    # units_res_optimal_lag = results_dict[params.best_lead_lag_key]['all_models_summary_results'].copy()
    # units_res_optimal_lag['traj_center'] = [50] * units_res_optimal_lag.shape[0] 
    # plot_optimal_lag_on_channel_map(units_res_optimal_lag, 
    #                                 jitter_radius = .15, 
    #                                 hueKey = 'traj_center', sizeKey = '%s_auc' % params.primary_traj_model,
    #                                 weighted_mean = True, weightsKey='%s_auc' % params.primary_traj_model, filtered=True)#'proportion_sign')

    # units_res_optimal_lag = find_optimal_lag_for_each_unit(model_results_across_lags, only_tuned=False, plot=False, average_all_peaks = True)
    
    # plot_optimal_lag_on_channel_map(units_res_optimal_lag, 
    #                                 jitter_radius = .15, 
    #                                 hueKey = 'traj_center', sizeKey = '%s_auc' % params.primary_traj_model,
    #                                 weighted_mean = True, weightsKey='%s_auc' % params.primary_traj_model, filtered=False)#'proportion_sign')

    # percentile=20
    # only_tuned=False
    # find_optimal_lag_averaged_over_brain_area(model_results_across_lags, only_tuned = only_tuned, plot = True, normalize=True , percentile=percentile)
    # find_optimal_lag_averaged_over_brain_area(model_results_across_lags, only_tuned = only_tuned, plot = True, normalize=False, percentile=percentile)
    
    
    units_res = results_dict[params.best_lead_lag_key]['all_models_summary_results']
    electrode_distances = get_interelectrode_distances_by_unit(units_res, array_type='utah')
    
    
    # plot_model_auc_comparison(units_res, 'traj_auc', 'traj_pca_auc', targets = None)
    # plot_model_auc_comparison(units_res, 'traj_auc', 'position_auc', targets = None)   
    
    # evaluate_lead_lag_by_model_coefficients(lead_lag_key = 'lead_200_lag_200', kin_type = 'traj', mode='each_lag', proportion_thresh=0.95)

    # plot_model_auc_comparison(units_res, 'traj_avgPos', 'position', targets = None)
    plot_model_auc_comparison(units_res, 'traj_avgPos_shuffled_spike_samples', 'traj_avgPos', targets = None, minauc=0.45)   
    plot_model_auc_comparison(units_res, 'shortTraj_avgPos', 'traj_avgPos', targets = None, minauc=0.45) 
    plot_model_auc_comparison(units_res, 'shortTraj', 'traj', targets = None, minauc=0.45)   
    # plot_model_auc_comparison(units_res, 'shortTraj', 'traj', targets = None, minauc=0.45) 
    plot_model_auc_comparison(units_res, 'shortTraj', 'shortTraj_avgPos', targets = None, minauc=0.45) 
    plot_model_auc_comparison(units_res, 'traj', 'traj_avgPos', targets = None, minauc=0.45) 

    plot_model_auc_comparison(units_res, 'shortTraj_low_tortuosity', 'traj_low_tortuosity', targets = None, minauc=0.45) 
    sign_test, ttest = sig_tests(units_res, 'shortTraj_low_tortuosity', 'traj_low_tortuosity', alternative='greater')
    print(sign_test)
    plot_model_auc_comparison(units_res, 'shortTraj_high_tortuosity', 'traj_high_tortuosity', targets = None, minauc=0.45) 
    sign_test, ttest = sig_tests(units_res, 'shortTraj_high_tortuosity', 'traj_high_tortuosity', alternative='greater')
    print(sign_test)
    plot_model_auc_comparison(units_res, 'shortTraj_avgPos_low_tortuosity', 'traj_avgPos_low_tortuosity', targets = None, minauc=0.45) 
    sign_test, ttest = sig_tests(units_res, 'shortTraj_avgPos_low_tortuosity', 'traj_avgPos_low_tortuosity', alternative='greater')
    print(sign_test)
    plot_model_auc_comparison(units_res, 'shortTraj_avgPos_high_tortuosity', 'traj_avgPos_high_tortuosity', targets = None, minauc=0.45) 
    sign_test, ttest = sig_tests(units_res, 'shortTraj_avgPos_high_tortuosity', 'traj_avgPos_high_tortuosity', alternative='greater')
    print(sign_test)

    # plot_model_auc_comparison(units_res, 'shortTraj_avgPos_low_tortuosity', 'shortTraj_avgPos_high_tortuosity', targets = None, minauc=0.45) 
    # plot_model_auc_comparison(units_res, 'traj_avgPos_low_tortuosity', 'traj_avgPos_high_tortuosity', targets = None, minauc=0.45) 
    # plot_model_auc_comparison(units_res, 'shortTraj_avgPos_high_tortuosity', 'traj_avgPos_low_tortuosity', targets = None, minauc=0.45) 

    # plot_model_auc_comparison(units_res, 'shortTraj_avgPos_low_tortuosity', 'traj_avgPos', targets = None, minauc=0.45) 
    # plot_model_auc_comparison(units_res, 'shortTraj_avgPos_high_tortuosity', 'traj_avgPos', targets = None, minauc=0.45) 
    
    # evaluate_lead_lag_by_model_coefficients(lead_lag_key = 'lead_200_lag_200', kin_type = 'traj_avgPos', mode='average', proportion_thresh=0.9)

    posTraj_mean, velTraj_mean, posTraj_samples, velTraj_samples = compute_and_analyze_pathlets(params.best_lead_lag_key, 'traj_avgPos', numplots = 5)

    try:
        df = compute_and_analyze_trajectory_correlations(units_res, posTraj_mean, velTraj_mean, electrode_distances, params.best_lead_lag_key, FN = reach_FN, mode='concat', nplots=5)
    except:
        df = compute_and_analyze_trajectory_correlations(units_res, posTraj_mean, velTraj_mean, electrode_distances, params.best_lead_lag_key, nplots=5)    
        
    
    df.sort_values(by=['Connection', 'Pearson_corr'], inplace=True)
    
    percentile = 95
    
    tmp_df = df.loc[df['Connection'] == '3b-M1', :]
    fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, dpi=300)
    sns.pointplot(ax=ax1, data=tmp_df, x='x2', y='r_squared')
    sns.pointplot(ax=ax2, data=tmp_df, x='x2', y='Wji')
    ax1.set_title('3b-M1')
    ax2.set_xlabel('M1_x')
    plt.show()
    
    tmp_df = df.loc[(df['Connection'] == '3b-M1') & (df['Wji'] > np.percentile(df['Wji'], percentile)), :]
    fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, dpi=300)
    sns.stripplot(ax=ax1, data=tmp_df, x='x2', y='Pearson_corr', s = 2)
    sns.stripplot(ax=ax2, data=tmp_df, x='x2', y='Wji',  s = 2)
    ax2.set_ylim(0, 0.04)
    ax1.set_title('3b-M1')
    ax2.set_xlabel('M1_x')
    plt.show()

    tmp_df = df.loc[df['Connection'] == '3a-M1', :]
    fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, dpi=300)
    sns.pointplot(ax=ax1, data=tmp_df, x='x2', y='r_squared')
    sns.pointplot(ax=ax2, data=tmp_df, x='x2', y='Wji')
    ax1.set_title('3a-M1')
    ax2.set_xlabel('M1_x')
    plt.show()
    
    tmp_df = df.loc[(df['Connection'] == '3a-M1') & (df['Wji'] > np.percentile(df['Wji'], percentile)), :]
    fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, dpi=300)
    sns.stripplot(ax=ax1, data=tmp_df, x='x2', y='Pearson_corr', s = 2)
    sns.stripplot(ax=ax2, data=tmp_df, x='x2', y='Wji',  s = 2)
    ax2.set_ylim(0, 0.04)
    ax1.set_title('3a-M1')
    ax2.set_xlabel('M1_x')
    plt.show()
    
    sns.relplot(data = df, x = 'Pearson_corr', y = 'Wji', col='Connection', 
                kind='scatter', s = 20, col_wrap=3, legend=True) 
    fig=plt.gcf()
    plt.show()
    fig.savefig(os.path.join(plots, 'wji_vs_pearson_r_columns_by_connection.png'), bbox_inches='tight', dpi=plot_params.dpi) 
    
    # fig, ax = plt.subplots(figsize = (4, 4), dpi = plot_params.dpi)
    # g = sns.FacetGrid(data = df, col = 'Connection', col_wrap=3, sharex=False)
    # g.map(plt.hist, 'Pearson_corr', alpha=.4)
    # sns.catplot(ax = ax, data = df, x = 'Pearson_corr', col = 'Connection', legend=True, kind='count') 
    # plt.show()

    # sns.pointplot(ax=ax1, data=tmp_df, x='x2', y='r_squared')
    # sns.pointplot(ax=ax2, data=tmp_df, x='x2', y='Wji')

    # plot_model_auc_comparison(units_res, 'traj_auc', 'position_auc', targets = 'tuned')
    # plot_model_auc_comparison(units_res, 'traj_auc', 'position_auc', targets = 'untuned')
    # plot_model_auc_comparison(units_res, 'traj_auc', 'position_auc', targets = 'motor')
    # plot_model_auc_comparison(units_res, 'traj_auc', 'position_auc', targets = 'sensory')

    
    # plot_model_auc_comparison(units_res, 'traj_auc', 'traj_and_avgPos_auc', targets = None)
    # plot_model_auc_comparison(units_res, 'short_traj_and_avgPos_auc', 'traj_and_avgPos_auc', targets = None)
    # plot_model_auc_comparison(units_res, 'traj_and_avgSpeed_auc', 'traj_and_avgPos_auc', targets = None)
    # plot_model_auc_comparison(units_res, 'traj_auc', 'traj_and_avgPos_auc', targets = 'tuned')
    # plot_model_auc_comparison(units_res, 'short_traj_and_avgPos_auc', 'traj_and_avgPos_auc', targets = 'tuned')
    # plot_model_auc_comparison(units_res, 'traj_and_avgSpeed_auc', 'traj_and_avgPos_auc', targets = 'tuned')    
    # plot_model_auc_comparison(units_res, 'traj_auc', 'traj_and_avgPos_auc', targets = 'motor')
    # plot_model_auc_comparison(units_res, 'short_traj_and_avgPos_auc', 'traj_and_avgPos_auc', targets = 'motor')
    # plot_model_auc_comparison(units_res, 'traj_and_avgSpeed_auc', 'traj_and_avgPos_auc', targets = 'motor') 
    # plot_model_auc_comparison(units_res, 'traj_auc', 'traj_and_avgPos_auc', targets = 'sensory')
    # plot_model_auc_comparison(units_res, 'short_traj_and_avgPos_auc', 'traj_and_avgPos_auc', targets = 'sensory')
    # plot_model_auc_comparison(units_res, 'traj_and_avgSpeed_auc', 'traj_and_avgPos_auc', targets = 'sensory') 
    
    # posTraj_mean, velTraj_mean, posTraj_samples, velTraj_samples = compute_and_analyze_pathlets(params.best_lead_lag_key, 'traj_pca')

    # with open(pkl_outfile, 'wb') as f:
    #     dill.dump(results_dict, f, recurse=True) 
    
    # percentile=None#50#params.nUnits_percentile
    # only_tuned=False
    # find_optimal_lag_averaged_over_brain_area(model_results_across_lags.copy(), only_tuned = only_tuned, plot = True, normalize=True , percentile=percentile)
    # find_optimal_lag_averaged_over_brain_area(model_results_across_lags.copy(), only_tuned = only_tuned, plot = True, normalize=False, percentile=percentile)
    
'''
    Note: use a longer lead_lag model (perhaps -500 to +500), and use non-optimized and Lasso-optimized GLMs. 
    Then convert coefficients back to lagged xyz terms. Then sum coefficient magntiudes over leads or lags to identify preferred sensory/motor mode
'''
