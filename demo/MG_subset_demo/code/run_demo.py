#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 00:16:27 2023

@author: daltonm
"""

import subprocess
from pathlib import Path
import os
import sys
import time

script_directory = Path(os.path.dirname(os.path.abspath(sys.argv[0])))

code_list = ['generate_functional_networks_MG_demo.py',
             'check_movement_peth_by_unit_MG_demo.py' 
             'collect_trajectory_samples_MG_demo.py',
             'create_trajectory_models_MG_demo.py', 
             'analyze_trajectory_only_models_MG_demo.py',
             'create_network_models_MG_demo.py',
             'analyze_models_with_network_terms_MG_demo.py',
             'reach_and_spontaneous_rasters_for_fig1_MG_demo.py',
             'regularization_sweep_MG_demo.py',
             'regularization_test_results_MG_demo.py']

for code in code_list:
    print(f'\n\nexecuting "{code}", time is {time.strftime("%c", time.localtime())}\n\n')
    subprocess.call(['python', str(script_directory / code)])
    