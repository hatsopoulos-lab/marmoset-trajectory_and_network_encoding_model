#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 15:56:34 2023

@author: daltonm
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pickle
import dill
import os
import glob
import math
import re
import seaborn as sns
from scipy.ndimage import median_filter, gaussian_filter
from importlib import sys, reload

sys.path.insert(0, '/project/nicho/projects/marmosets/code_database/data_processing/nwb_tools/hatlab_nwb_tools/')
from hatlab_nwb_functions import plot_prb, read_prb_hatlab 

def get_single_lead_lag_models(all_models_data, lead, lag):
    ll_idx = [idx for idx, ll in enumerate(all_models_data['lead_lag']) if ll == (lead*1e3, lag*1e3)][0]
    
    single_lead_lag_models = {}
    for key in all_models_data.keys(): 
        single_lead_lag_models[key] = all_models_data[key][ll_idx]
        
    return single_lead_lag_models, ll_idx

def choose_units_for_model(units, quality_key = 'snr', quality_thresh = 3, frate_thresh = 2):
    
    if quality_key == 'snr':
        quality = units.snr
    elif quality_key == 'amp':
        quality = units.amp
        quality_thresh = np.percentile(quality, quality_thresh)      
    
    units = units.loc[(quality > quality_thresh) | (units.quality == 'good'), :]
    units = units.loc[units.fr > frate_thresh, :]
    
    units.reset_index(inplace=True, drop=True)
    
    return units

def get_interelectrode_distances_by_unit(units_res, array_type ='utah'):
        
    electrode_distances = np.full((units_res.shape[0], units_res.shape[0]), np.nan)
    if array_type.lower() == 'utah':

        for i, (ix, iy) in enumerate(zip(units_res['x'], units_res['y'])):
            for j, (jx, jy) in enumerate(zip(units_res['x'], units_res['y'])):
                if i == j:
                    continue
                electrode_distances[i, j] = np.sqrt((ix - jx)**2 + (iy - jy)**2)

    return electrode_distances 

def fix_unit_info_elec_labels(unit_info, chan_map_df):
    if unit_info.index.to_list() != list(range(unit_info.shape[0])):
        unit_info.reset_index(drop=False, inplace=True)
    
    fixed_labels = [0]*unit_info.shape[0]
    x            = [0]*unit_info.shape[0]
    y            = [0]*unit_info.shape[0]
    z            = [-1]*unit_info.shape[0]
    for row, unit_row in unit_info.iterrows():
        fixed_labels[row] = chan_map_df.loc[unit_row.ch, 'shank_ids']
        x[row] = int(chan_map_df.loc[unit_row.ch, 'x'])
        y[row] = int(chan_map_df.loc[unit_row.ch, 'y'])
        try:
            z[row] = int(chan_map_df.loc[unit_row.ch, 'z'])
        except:
            pass
        
    unit_info['uncorrected_ns6_elec_id'] = unit_info['ns6_elec_id']
    unit_info['ns6_elec_id'] = fixed_labels    
    unit_info['x'] = x
    unit_info['y'] = y
    if not all([el==-1 for el in z]):
        unit_info['z'] = z
    
    return unit_info  
# def fix_unit_info_elec_labels(unit_info, chan_map_df):
#     if unit_info.index.to_list() != list(range(unit_info.shape[0])):
#         unit_info.reset_index(drop=False, inplace=True)
    
#     if 'x' not in unit_info.columns:    
#         fixed_labels = [0]*unit_info.shape[0]
#         x            = [0]*unit_info.shape[0]
#         y            = [0]*unit_info.shape[0]
#         z            = [-1]*unit_info.shape[0]
#         for row, unit_row in unit_info.iterrows():
#             fixed_labels[row] = int(chan_map_df.loc[unit_row.ch, 'shank_ids'])
#             x[row] = int(chan_map_df.loc[unit_row.ch, 'x'])
#             y[row] = int(chan_map_df.loc[unit_row.ch, 'y'])
#             try:
#                 z[row] = int(chan_map_df.loc[unit_row.ch, 'z'])
#             except:
#                 pass
            
#         unit_info['uncorrected_ns6_elec_id'] = unit_info['ns6_elec_id']
#         unit_info['ns6_elec_id'] = fixed_labels    
#         unit_info['x'] = x
#         unit_info['y'] = y
#         if not all([el==-1 for el in z]):
#             unit_info['z'] = z       
    
#     return unit_info     



def compute_derivatives(marker_pos, fps, smooth = True):
    marker_vel = np.diff(marker_pos, axis = -1) * fps
    if smooth:
        for dim in range(3):
            marker_vel[dim] = gaussian_filter(marker_vel[dim], sigma=1.5)

    marker_acc = np.diff(marker_vel, axis = -1) * fps
    if smooth:
        for dim in range(3):
            marker_acc[dim] = gaussian_filter(marker_acc[dim], sigma=1.5)
    
    return marker_vel, marker_acc


def load_channel_map_from_prb(marm = 'Tony'):
    if marm.lower() == 'tony':
        map_path = '/project/nicho/data/marmosets/prbfiles/TY_02.prb'
    elif marm.lower() == 'midge':
        map_path = '/project/nicho/data/marmosets/prbfiles/MG_01.prb'
        
    chan_map_probegroup, imp = read_prb_hatlab(map_path)
    plot_prb(chan_map_probegroup)
    chan_map_df = chan_map_probegroup.to_dataframe()
    
    return chan_map_df

def load_color_palette(palette_path):
    LinL = np.loadtxt(palette_path, delimiter=',')
    
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
