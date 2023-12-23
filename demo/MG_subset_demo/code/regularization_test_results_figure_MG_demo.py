#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 18:43:14 2023

@author: daltonm
"""

import dill
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import os
from scipy.stats import binomtest, ttest_rel
from pathlib import Path
from importlib import sys

sys.path.insert(0, '/project/nicho/projects/marmosets/code_database/analysis/trajectory_encoding_model/clean_final_analysis/')
from utils import save_dict_to_hdf5, load_dict_from_hdf5

marmscode = 'MG'
filter_untuned = False

script_directory = Path(os.path.dirname(os.path.abspath(sys.argv[0])))
code_path = script_directory.parent.parent.parent / 'clean_final_analysis/'
data_path = script_directory.parent.parent / 'data' / 'demo'
fig_path = script_directory.parent / 'plots' 


if marmscode == 'TY':
    pkl_infiles = ['/project/nicho/projects/dalton/data/TY/TY20210211_freeAndMoths-003_resorted_20230612_DM_alpha_pt00001_encoding_models_30msREGTEST_V2_shift_v4.pkl']
    marms = ['TY']
elif marmscode == 'MG':
    pkl_infiles = [data_path / 'MG' / 'MG20230416_1505_mothsAndFree-002_processed_DM_regularization_test_results.h5']
    marms = ['MG']
    tuned_units = [ [0,  1,  2,  6,  8, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24,
                    25, 26, 27, 28, 29, 30, 32, 33, 34, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                    46, 47, 48, 49, 52, 54, 55, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
                    69, 70, 71, 72]]
elif marmscode == 'both':
    pkl_infiles = ['/project/nicho/projects/dalton/data/TY/TY20210211_freeAndMoths-003_resorted_20230612_DM_alpha_pt00001_encoding_models_30msREGTEST_V2_shift_v4.pkl',
                  '/project/nicho/projects/dalton/data/MG/MG20230416_1505_mothsAndFree-002_processed_DM_dlcIter5_resortedUnits_trajectory_shuffled_encoding_models_30msREGTEST_V2_shift_v4.pkl']
    marms = ['TY', 'MG']
    tuned_units = [[i for i in range(175) if i not in [3, 41, 51, 57, 58, 95, 120, 133, 137, 144, 167]],
                   [ 0,  1,  2,  6,  8, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24,
                                   25, 26, 27, 28, 29, 30, 32, 33, 34, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                                   46, 47, 48, 49, 52, 54, 55, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
                                   69, 70, 71, 72]]

lags = ['lead_100_lag_300', 'lead_200_lag_300']

regtest_df = pd.DataFrame()
for marm, pkl_file, tuned_idxs in zip(marms, pkl_infiles, tuned_units):    

    # with open(pkl_file, 'rb') as f:
    #     results_dict = dill.load(f)
    results_dict = load_dict_from_hdf5(pkl_file.with_suffix('.h5'), top_level_list=False, convert_4d_array_to_list = True)

    
    model_results = results_dict[lags[0]]['model_results']    
    # FN_only_keys = sorted([key for key in model_results.keys() if key[:8] == 'reach_FN'])
    kin_only_keys = sorted([key for key in model_results.keys() if 'traj_avgPos' in key and 'reach_FN' not in key])

    alpha = []
    l1    = []
    for key in kin_only_keys:
        alpha.append(model_results[key]['alpha'])
        l1.append(model_results[key]['l1'])
    
    for lead_lag_key in lags:
        
        model_results = results_dict[lead_lag_key]['model_results']    
        kin_only_keys = sorted([key for key in model_results.keys() if 'traj_avgPos' in key and len(key) == 15])
        kinFN_keys   = sorted([key for key in model_results.keys() if 'reach_FN' in key and key[:8] != 'reach_FN'])
    
        for model, keySet in zip(['Full Kinematics', 'Kinematics + reachFN'], [kin_only_keys, kinFN_keys]):
    
            for key in keySet:
                tmp_alpha = model_results[key]['alpha']
                tmp_l1    = model_results[key]['l1']
                tmp_auc   = model_results[key]['AUC']
                
                if filter_untuned:
                    tmp_auc = tmp_auc[tuned_idxs, :]

                nRows = tmp_auc.size                     
                tmp_df = pd.DataFrame(data = zip(np.repeat(tmp_alpha, nRows), 
                                                  np.repeat(tmp_l1, nRows), 
                                                  tmp_auc.flatten(), 
                                                  np.repeat(lead_lag_key, nRows), 
                                                  np.repeat(model, nRows),
                                                  np.repeat(marm, nRows)),
                                      columns = ['alpha', 'l1', 'auc', 'lead_lag', 'model', 'marm'])
                
                tmp_df = tmp_df.loc[tmp_df['auc'] > np.percentile(tmp_df['auc'], 20), :]
                        
                # nRows = tmp_auc.shape[0]                     
    
                # tmp_df = pd.DataFrame(data = zip(np.repeat(tmp_alpha, nRows), 
                #                                   np.repeat(tmp_l1, nRows), 
                #                                   tmp_auc.mean(axis=1), 
                #                                   np.repeat(lead_lag_key, nRows), 
                #                                   np.repeat(model, nRows),
                #                                   np.repeat(marm, nRows)),
                #                       columns = ['alpha', 'l1', 'auc', 'lead_lag', 'model', 'marm'])
                        
                regtest_df = pd.concat((regtest_df, tmp_df), ignore_index=True)
            
for model, marm in itertools.product(['Full Kinematics', 'Kinematics + reachFN'], marms): 

    tmp_df = regtest_df.loc[(regtest_df.model == model) & (regtest_df.marm == marm)]     

    bestAlpha_idx = np.argmax(tmp_df.groupby('alpha').mean(numeric_only=True)['auc'])            
    bestAlpha = np.unique(tmp_df.alpha)[bestAlpha_idx]
    auc_ticks = [0.55, 0.62] if model == 'Kinematics + reachFN' else [0.55, 0.585]
    # auc_ticks = [0.58, 0.62] if marm == 'TY' else [0.55, 0.61]
    
    print()
    for tmp_alpha in np.unique(tmp_df.alpha):
        nBest = np.sum(tmp_df.loc[tmp_df['alpha'] == bestAlpha, 'auc'].values > tmp_df.loc[tmp_df['alpha'] == tmp_alpha, 'auc'].values)
        nPossible = tmp_df.loc[tmp_df['alpha'] == bestAlpha, 'auc'].size
        sign_test = binomtest(nBest, nPossible, p = 0.5, alternative='greater')   
        print(f'model = {model}, marm = {marm}, alpha = {tmp_alpha}: prop = {np.round(sign_test.proportion_estimate, 2)}, p = {np.round(sign_test.pvalue, 6)}')
        # ttest_paired = ttest_rel(tmp_df.loc[tmp_df['alpha'] == bestAlpha, 'auc'].values, tmp_df.loc[tmp_df['alpha'] == tmp_alpha, 'auc'].values, alternative='greater')
        # print(f'marm = {marm}, alpha = {tmp_alpha}: p = {np.round(ttest_paired.pvalue, 3)}')



    fig, ax = plt.subplots(figsize=(3, 2.5), dpi = 300)
    sns.lineplot(ax=ax, data=tmp_df, 
                 x = 'alpha', y='auc', hue = 'lead_lag', legend=False, linestyle='-', 
                 err_style='bars', errorbar=("se", 1), linewidth=1, marker='o', 
                 markersize=6)   
    ax.set_xscale('log') 
    ax.set_xticks(np.unique(tmp_df.alpha)[::2])
    ax.set_yticks(auc_ticks) #([ax.get_yticks().min(), ax.get_yticks().max()])
    ax.set_ylabel(f'{model} AUC')
    
    ax.set_xlabel('Regularization Penalty (alpha)')
    sns.despine(ax=ax)
    
    pkl_file = [f for f in pkl_infiles if f'data/demo/{marm}/' in str(f)][0]
    dataset_code = str(pkl_file).split(f'data/demo/{marm}/')[-1][:10] 
    
    if fig_path.parent.stem != dataset_code:
        fig_path = fig_path / dataset_code / 'FigS5'
        os.makedirs(fig_path, exist_ok=True)
    
    fig.savefig(fig_path / f'{marm}_{model.replace(" ", "_")}_alpha_sweep.png', bbox_inches='tight', dpi=300)

    ax.set_title(marm)
    plt.show()

for model, marm in itertools.product(['Full Kinematics', 'Kinematics + reachFN'], marms): 

    tmp_df = regtest_df.loc[(regtest_df.model == model) & (regtest_df.marm == marm)]     
    
    print()
    for alpha1, alpha2 in zip(np.unique(tmp_df.alpha)[:-1], np.unique(tmp_df.alpha)[1:]):
        n1over2 = np.sum(tmp_df.loc[tmp_df['alpha'] == alpha1, 'auc'].values > tmp_df.loc[tmp_df['alpha'] == alpha2, 'auc'].values)
        nPossible = tmp_df.loc[tmp_df['alpha'] == alpha1, 'auc'].size
        sign_test = binomtest(n1over2, nPossible, p = 0.5, alternative='greater')   
        print(f'model = {model}, marm = {marm}, a={alpha1} > a={alpha2}: prop = {np.round(sign_test.proportion_estimate, 2)}, p = {np.round(sign_test.pvalue, 6)}')


# fig, ax = plt.subplots(figsize=(6,6), dpi = 300)
# sns.heatmap(average_auc,ax=ax,cmap='viridis', vmin=average_auc[average_auc>0].min(), vmax=average_auc.max()) 
# ax.set_ylabel('Alpha')
# ax.set_xlabel('L1 weight')
# ax.set_yticklabels(params.alpha_range)
# ax.set_xticklabels(params.l1_range)
# plt.show()