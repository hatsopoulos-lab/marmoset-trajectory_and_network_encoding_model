#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 12:22:46 2023

@author: daltonm
"""

import h5py
import dill
from pathlib import Path
from importlib import sys
import pandas as pd
import numpy as np

# sys.path.insert(0, '/project/nicho/projects/marmosets/code_database/analysis/trajectory_encoding_model/clean_final_analysis/')
# from utils import save_dict_to_hdf5, load_dict_from_hdf5

pklpath = Path('/project/nicho/projects/dalton/data/MG/MG20230416_1505_mothsAndFree-002_processed_DM_dlcIter5_resortedUnits_trajectory_shuffled_encoding_models_30msREGTEST_V2_shift_v4.pkl')
savepath = Path('/project/nicho/projects/dalton/data/MG/MG20230416_1505_mothsAndFree-002_processed_DM_alpha_parameter_sweep.h5')

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
     


with open(str(pklpath), 'rb') as f:
    results_dict = dill.load(f)
    
# save_dict_to_hdf5(results_dict, pklpath.with_suffix('.h5'))
save_dict_to_hdf5(results_dict, savepath)

# results_dict_loaded = load_dict_from_hdf5(pklpath.with_suffix('.h5'), top_level_list=False, convert_4d_array_to_list = True)