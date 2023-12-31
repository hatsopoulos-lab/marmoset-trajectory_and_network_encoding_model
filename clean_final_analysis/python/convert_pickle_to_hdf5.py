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

sys.path.insert(0, '/project/nicho/projects/marmosets/code_database/analysis/trajectory_encoding_model/clean_final_analysis/')
from utils import save_dict_to_hdf5, load_dict_from_hdf5

pklpath = Path('/project/nicho/projects/dalton/network_encoding_paper/clean_final_analysis/data/MG/MG20230416_1505_mothsAndFree-002_processed_DM_modulationData.pkl')
savepath = Path('/project/nicho/projects/marmosets/code_database/analysis/trajectory_encoding_model/demo/MG_original_data_reproduction/data/MG/MG20230416_1505_mothsAndFree-002_processed_DM_modulationData.h5')

with open(str(pklpath), 'rb') as f:
    results_dict = dill.load(f)
    
# save_dict_to_hdf5(results_dict, pklpath.with_suffix('.h5'))
save_dict_to_hdf5(results_dict, savepath)

# results_dict_loaded = load_dict_from_hdf5(pklpath.with_suffix('.h5'), top_level_list=False, convert_4d_array_to_list = True)