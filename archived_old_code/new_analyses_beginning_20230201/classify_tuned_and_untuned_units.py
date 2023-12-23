#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 15:58:36 2023

@author: daltonm
"""



from importlib import reload, sys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import binomtest, ttest_rel, ttest_ind

sys.path.insert(0, '/project/nicho/projects/marmosets/code_database/analysis/trajectory_encoding_model/')
from utils import *

class path:
    storage = '/home/daltonm/Documents/tmp_analysis_folder/processed_datasets' # /project2/nicho/dalton/processed_datasets'
    intermediate_save_path = '/home/daltonm/Documents/tmp_analysis_folder/analysis/encoding_model/intermediate_variable_storage'#'/project2/nicho/dalton/analysis/encoding_model/intermediate_variable_storage'
    new_save_path = '/home/daltonm/Documents/tmp_analysis_folder/analysis/encoding_model/intermediate_variable_storage/10pt0_ms_bins/data_updated_february_2023' #'/project2/nicho/dalton/analysis/encoding_model/intermediate_variable_storage/10pt0_ms_bins/data_updated_february_2023'
    plots = '/home/daltonm/Documents/tmp_analysis_folder/analysis/encoding_model/plots' #'/project2/nicho/dalton/analysis/encoding_model/plots'
    date = '20210211'
    # storage = '/project2/nicho/dalton/processed_datasets'
    # intermediate_save_path = '/project2/nicho/dalton/analysis/encoding_model/intermediate_variable_storage'
    # new_save_path = '/project2/nicho/dalton/analysis/encoding_model/intermediate_variable_storage/10pt0_ms_bins/data_updated_february_2023'
    # plots = '/project2/nicho/dalton/analysis/encoding_model/plots'
    # date = '20210211'
    
class params:
    spkSampWin = 0.01
    trajShift = 0.05 #sample every 50ms
    lead = [0.2] #[0.5, 0.4, 0.4, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0  , 0  , 0  ] #[0.2] #[0.2, 0.15, 0.1, 0.05, 0] #[0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0] #[0.05] # 100ms lead time
    lag  = [0.3] # [0  , 0  , 0.1, 0  , 0.1, 0.2, 0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.3, 0.4, 0.5] #[0.3] #[0.2, 0.25, 0.3, 0.35, 0.4] #[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4] #[0.35] # 300ms lag time
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
    
def plot_AUC_distributions(models_dict, model_keys, units, plot=True):

    if units is None:
        units = range(models_dict['unit_info'].shape[0])
        
    sampled_spikes = models_dict['sampled_spikes']
    nSpikeSamples = sampled_spikes.shape[1]

    p_ttest       = []
    p_signtest    = []
    prop_signtest = []
    ci_signtest_lower   = []
    for unit in units:
        unit_AUCs    = []
        model_labels  = []
        for model_name, results in zip(models_dict['model_details']['model_names'],
                                       models_dict['model_details']['model_results']): 
            if model_name in model_keys:
                unit_AUCs.extend(results['AUC'][unit])
                model_labels.extend([model_name]*results['AUC'].shape[1])
        
        auc_df = pd.DataFrame(data = zip(unit_AUCs, model_labels), columns = ['AUC', 'Model'])
    
        t_stats = ttest_rel(auc_df.loc[auc_df['Model'] != 'shuffle', 'AUC'], 
                            auc_df.loc[auc_df['Model'] == 'shuffle', 'AUC'], 
                            alternative='greater')
    
        nTrajGreater = np.sum(auc_df.loc[auc_df['Model'] != 'shuffle', 'AUC'].values > 
                              auc_df.loc[auc_df['Model'] == 'shuffle', 'AUC'].values)
        nSamples     = len(auc_df.loc[auc_df['Model'] != 'shuffle', 'AUC'])
        sign_test    = binomtest(nTrajGreater, nSamples, p = 0.5, alternative='greater')
    
        # fig, ax = plt.subplots()
        # sns.histplot(data=auc_df, ax = ax, x='AUC', hue='Model',
        #              log_scale=False, element="poly", fill=False,
        #              cumulative=False, common_norm=False, bins=25)
        # ax.set_xlabel('AUC')
        # ax.set_title('Unit %d' % unit)
        # plt.show()
        
        if unit == 27:
            tmp = []
        
        if plot:
            fig, ax = plt.subplots()
            sns.kdeplot(data=auc_df, ax=ax, x='AUC', hue='Model',
                          log_scale=False, fill=False,
                          cumulative=False, common_norm=False, bw_adjust=.5)
            ax.set_xlabel('% AUC Loss')
            ax.set_title('Unit %d, prop=%.2f, p-val=%.3f, auc=%.2f, spkProp=%.3f' % (unit, 
                                                                                     sign_test.proportion_estimate, 
                                                                                     sign_test.pvalue,
                                                                                     auc_df.loc[auc_df['Model'] != 'shuffle', 'AUC'].mean(),
                                                                                     np.sum(sampled_spikes[unit] >= 1, axis=0)[0] / nSpikeSamples))      
            ax.set_xlim(0.4, 1)
        
            plt.show()
        
        p_ttest.append(t_stats.pvalue)
        p_signtest.append(sign_test.pvalue)
        prop_signtest.append(sign_test.proportion_estimate)
        ci_signtest_lower.append(sign_test.proportion_ci(confidence_level=.99)[0])
    
    stats_df = pd.DataFrame(data = zip(p_ttest, p_signtest, prop_signtest, ci_signtest_lower),
                            columns = ['pval_t', 'pval_sign', 'proportion_sign', 'CI_lower'])
    
    return stats_df

def add_tuning_class_to_df(unit_info, stats_df, stat_key, thresh, direction='greater'):
    
    tuning = ['untuned']*unit_info.shape[0]
    for idx, stat in enumerate(stats_df[stat_key]):
        if (direction.lower() == 'greater' and stat >= thresh) or (direction.lower() == 'less' and stat <= thresh):
            tuning[idx] = 'tuned'

    unit_info['tuning'] = tuning
    unit_info[stat_key] = stats_df[stat_key].values 
    
    return unit_info

if __name__ == "__main__":
    
    all_models_data = load_models(path, params)
    
    chan_map_df = load_channel_map_from_prb(marm = 'Tony')
    
    for lead, lag in zip(params.lead, params.lag):
        single_lead_lag_models, ll_idx = get_single_lead_lag_models(all_models_data, lead, lag)
        unit_info = single_lead_lag_models['unit_info']
        
        stats_df = plot_AUC_distributions(single_lead_lag_models, ['trajectory', 'shuffle'], None, plot=False)
        
        sorted_idx = stats_df.sort_values(by = 'proportion_sign', ascending = False).index.to_list()
        _ = plot_AUC_distributions(single_lead_lag_models, ['trajectory', 'shuffle'], sorted_idx, plot=True)
        
        unit_info = fix_unit_info_elec_labels(unit_info, chan_map_df)
        
        unit_info = add_tuning_class_to_df(unit_info, stats_df, 'proportion_sign', thresh = 0.9, direction='greater')

        all_models_data['unit_info'][ll_idx] = unit_info
    
    # with open(os.path.join(path.new_save_path, 'all_models_data_dict.pkl'), 'wb') as f:
    #     dill.dump(all_models_data, f, recurse=True)
    
    ### Now need to train new models with only tuned inputs or only untuned inputs, controlling for delta_x == 0 vs delta_x >= 400 um