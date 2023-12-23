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

marmscode = 'both'

if marmscode == 'TY':
    pkl_infiles = ['/project/nicho/projects/dalton/data/TY/TY20210211_freeAndMoths-003_resorted_20230612_DM_alpha_pt00001_encoding_models_30msREGTEST_V2_shift_v4.pkl']
    marms = ['TY']
elif marmscode == 'MG':
    pkl_infiles = ['/project/nicho/projects/dalton/data/MG/MG20230416_1505_mothsAndFree-002_processed_DM_dlcIter5_resortedUnits_trajectory_shuffled_encoding_models_30msREGTEST_V2_shift_v4.pkl']
    marms = ['MG']
elif marmscode == 'both':
    pkl_infiles = ['/project/nicho/projects/dalton/data/TY/TY20210211_freeAndMoths-003_resorted_20230612_DM_alpha_pt00001_encoding_models_30msREGTEST_V2_shift_v4.pkl',
                  '/project/nicho/projects/dalton/data/MG/MG20230416_1505_mothsAndFree-002_processed_DM_dlcIter5_resortedUnits_trajectory_shuffled_encoding_models_30msREGTEST_V2_shift_v4.pkl']
    marms = ['TY', 'MG']

lags = ['lead_100_lag_300', 'lead_200_lag_300']

regtest_df = pd.DataFrame()
for marm, pkl_file in zip(marms, pkl_infiles):    

    with open(pkl_file, 'rb') as f:
        results_dict = dill.load(f)
    
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
    
                nRows = tmp_auc.size                     
                tmp_df = pd.DataFrame(data = zip(np.repeat(tmp_alpha, nRows), 
                                                  np.repeat(tmp_l1, nRows), 
                                                  tmp_auc.flatten(), 
                                                  np.repeat(lead_lag_key, nRows), 
                                                  np.repeat(model, nRows),
                                                  np.repeat(marm, nRows)),
                                      columns = ['alpha', 'l1', 'auc', 'lead_lag', 'model', 'marm'])
                        
                # nRows = tmp_auc.shape[0]                     
    
                # tmp_df = pd.DataFrame(data = zip(np.repeat(tmp_alpha, nRows), 
                #                                  np.repeat(tmp_l1, nRows), 
                #                                  tmp_auc.mean(axis=1), 
                #                                  np.repeat(lead_lag_key, nRows), 
                #                                  np.repeat(model, nRows),
                #                                  np.repeat(marm, nRows)),
                #                       columns = ['alpha', 'l1', 'auc', 'lead_lag', 'model', 'marm'])
                        
                regtest_df = pd.concat((regtest_df, tmp_df), ignore_index=True)
            
            
for model, marm in itertools.product(['Full Kinematics', 'Kinematics + reachFN'], marms): 

    tmp_df = regtest_df.loc[(regtest_df.model == model) & (regtest_df.marm == marm)]     

    bestAlpha_idx = np.argmax(tmp_df.groupby('alpha').mean()['auc'])            
    bestAlpha = np.unique(tmp_df.alpha)[bestAlpha_idx]
    auc_ticks = [0.55, 0.62] if model == 'Kinematics + reachFN' else [0.55, 0.585]
    # auc_ticks = [0.58, 0.62] if marm == 'TY' else [0.55, 0.61]

    for tmp_alpha in np.unique(tmp_df.alpha):
        nBest = np.sum(tmp_df.loc[tmp_df['alpha'] == bestAlpha, 'auc'].values > tmp_df.loc[tmp_df['alpha'] == tmp_alpha, 'auc'].values)
        nPossible = tmp_df.loc[tmp_df['alpha'] == bestAlpha, 'auc'].size
        sign_test = binomtest(nBest, nPossible, p = 0.5, alternative='greater')   
        print(f'marm = {marm}, alpha = {tmp_alpha}: prop = {np.round(sign_test.proportion_estimate, 2)}, p = {np.round(sign_test.pvalue, 3)}')
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
    
    pkl_file = [f for f in pkl_infiles if f'data/{marm}/' in f][0]
    dataset_code = pkl_file.split(f'data/{marm}/')[-1][:10] 
    fig_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(pkl_file))), 'plots', dataset_code, 'FigS4')
    os.makedirs(fig_path, exist_ok=True)
    
    fig.savefig(os.path.join(fig_path, f'{marm}_{model.replace(" ", "_")}_alpha_sweep.png'), bbox_inches='tight', dpi=300)

    ax.set_title(marm)
    plt.show()
    

# fig, ax = plt.subplots(figsize=(6,6), dpi = 300)
# sns.heatmap(average_auc,ax=ax,cmap='viridis', vmin=average_auc[average_auc>0].min(), vmax=average_auc.max()) 
# ax.set_ylabel('Alpha')
# ax.set_xlabel('L1 weight')
# ax.set_yticklabels(params.alpha_range)
# ax.set_xticklabels(params.l1_range)
# plt.show()