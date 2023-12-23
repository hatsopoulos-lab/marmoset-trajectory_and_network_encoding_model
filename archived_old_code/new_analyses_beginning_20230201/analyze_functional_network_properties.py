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
import h5py
from scipy.io import loadmat

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
    # plots = '/project2/nicho/dalton/analysis/encoding_model/plots'
    # date = '20210211'
    
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
    
def plot_weights_versus_interelectrode_distances(FN, electrode_distances, FN_type):
    
    weights = FN[FN_type]
    
    df = pd.DataFrame(data = zip(weights.flatten(), electrode_distances.flatten()), 
                      columns = ['Wij', 'Distance'])
    
    fig, ax = plt.subplots()
    sns.lineplot(data=df, x='Distance', y='Wij', err_style="bars", errorbar=('ci', 95))
    plt.show()


if __name__ == "__main__":
    
    spike_data, kinematics, analog_and_video, FN = load_data(path) 
        
    unit_info = filter_units(spike_data, params.frate_thresh)
    
    chan_map_df = load_channel_map_from_prb(marm = 'Tony')
    
    unit_info = fix_unit_info_elec_labels(unit_info, chan_map_df)
    
    electrode_distances = get_interelectrode_distances_by_unit(unit_info, chan_map_df, array_type='utah')
    
    plot_weights_versus_interelectrode_distances(FN, electrode_distances, 'reach_FN')
    
    