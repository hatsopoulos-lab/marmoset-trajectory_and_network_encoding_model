#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 15:56:34 2023

@author: daltonm
"""

import numpy as np
import pandas as pd
import matplotlib
import h5py
from scipy.ndimage import gaussian_filter
from functools import reduce  # forward compatibility for Python 3
import operator
from importlib import sys
from pathlib import Path

script_directory = Path(sys.argv[0]).resolve().parent

sys.path.insert(0, script_directory)
from hatlab_nwb_functions import plot_prb, read_prb_hatlab 

def get_single_lead_lag_models(all_models_data, lead, lag):
    ll_idx = [idx for idx, ll in enumerate(all_models_data['lead_lag']) if ll == (lead*1e3, lag*1e3)][0]
    
    single_lead_lag_models = {}
    for key in all_models_data.keys(): 
        single_lead_lag_models[key] = all_models_data[key][ll_idx]
        
    return single_lead_lag_models, ll_idx

def choose_units_for_model(units, quality_key = 'snr', quality_thresh = 3, frate_thresh = 2, bad_units_list = None):
    
    if quality_key == 'snr':
        quality = units.snr
    elif quality_key == 'amp':
        quality = units.amp
        quality_thresh = np.percentile(quality, quality_thresh)      
    
    units = units.loc[(quality > quality_thresh) | (units.quality == 'good'), :]
    units = units.loc[units.fr > frate_thresh, :]
    # tmp = units.loc[units.fr <= frate_thresh, :]
    # [idx for idx, name in enumerate(units.loc[units.quality == 'good', 'unit_name']) if name in tmp.unit_name.values]
    # tmp = units.loc[(quality <= quality_thresh) & (units.quality == 'good')]
    # [idx for idx, name in enumerate(units.loc[units.quality == 'good', 'unit_name']) if name in tmp.unit_name.values]

    if bad_units_list is not None:
        good_idx = [idx for idx, unit_info in units.iterrows() if int(unit_info.unit_name) not in bad_units_list]
        units = units.loc[good_idx, :]
    
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

def save_dict_to_hdf5(data, filename, top_level_list_namebase=None):
    """
    ....
    """
    df_keys_list, df_data_list = [], []
    with h5py.File(filename, 'w') as h5file:
        if type(data) == list:
            for idx, tmp_data in enumerate(data):
                df_keys_list, df_data_list = recursively_save_dict_contents_to_group(h5file, f'/{top_level_list_namebase}_{idx}/', tmp_data)
        elif type(data) == dict:
            df_keys_list, df_data_list = recursively_save_dict_contents_to_group(h5file, '/', data, df_keys_list, df_data_list)
        elif type(data) == pd.DataFrame:
            df_keys_list.append('df')
            df_data_list.append(data)
    
    for key, df in zip(df_keys_list, df_data_list):
        df.to_hdf(filename, key, mode='a')


def recursively_save_dict_contents_to_group(h5file, path, dic, df_keys_list = None, df_data_list = None):
    """
    ....
    """
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, int, float, np.integer, np.float32, np.float64, str, bytes)):
            h5file[path + key] = item
        elif isinstance(item, list):
            if len(item) > 0 and type(item[0]) == str:
                h5file.create_dataset(path + key, dtype=h5py.string_dtype(encoding='utf-8'), data=item)
            else:
                h5file[path + key] = np.array(item)
            
        elif isinstance(item, dict):
            df_keys_list, df_data_list = recursively_save_dict_contents_to_group(h5file, path + key + '/', item, df_keys_list, df_data_list)
        elif isinstance(item, pd.DataFrame):
            df_keys_list.extend([path + key])
            df_data_list.extend([item])
        else:
            raise ValueError('Cannot save %s type'%type(item))
     
    return df_keys_list, df_data_list       
     
def recursively_load_dict_contents_from_group(h5file, path, df_key_list, convert_4d_array_to_list = False):
    """
    ....
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            try:
                ans[key] = item[:]
            except:
                ans[key] = item[()]
            if convert_4d_array_to_list and isinstance(ans[key], np.ndarray) and ans[key].ndim == 4:
                ans[key] = [arr for arr in ans[key]]
        elif isinstance(item, h5py._hl.group.Group):
            if 'axis0' in item.keys() and 'axis1' in item.keys():
                df_key_list.extend([path + key])
            else:
                ans[key], df_key_list = recursively_load_dict_contents_from_group(h5file, path + key + '/', df_key_list)
    return ans, df_key_list

def load_dict_from_hdf5(filename, top_level_list=False, convert_4d_array_to_list = False):
    """
    ....
    """
    with h5py.File(filename, 'r') as h5file:
        if top_level_list:
            list_of_dicts = []
            for key in h5file.keys():
                df_key_list = []
                tmp_dict, df_key_list = recursively_load_dict_contents_from_group(h5file, key+'/', df_key_list, convert_4d_array_to_list)
                list_of_dicts.append(tmp_dict)
            loaded_data = list_of_dicts
        else:
            df_key_list = []
            tmp_dict, df_key_list = recursively_load_dict_contents_from_group(h5file, '/', df_key_list, convert_4d_array_to_list)
            loaded_data = tmp_dict
    
    if isinstance(loaded_data, dict):
        for df_key in df_key_list:
            key_tree = [part for part in df_key.split('/') if part != '']
            set_by_path(loaded_data, key_tree,  pd.read_hdf(filename, df_key) )

            # for branch in key_tree[:-1]:
            #     if branch in keys
            #     loaded_data.setdefault(branch, {})
            # loaded_data[key_tree[-1]] = pd.read_hdf(h5file, df_key) 
    
    # elif isinstance(loaded_data, list):
    #     tmp = [] # write code to grab list index, then dict path
    
    return loaded_data

def get_by_path(root, items):
    """Access a nested object in root by item sequence."""
    return reduce(operator.getitem, items, root)

def set_by_path(root, items, value):
    """Set a value in a nested object in root by item sequence."""
    get_by_path(root, items[:-1])[items[-1]] = value
