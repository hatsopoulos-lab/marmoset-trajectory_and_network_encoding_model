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

code_list = ['analyze_trajectory_only_models_figure_reproduction.py',
             'analyze_models_with_network_terms_figure_reproduction.py',
             'reach_and_spontaneous_rasters_for_fig1_reproduction.py',
             'regularization_test_results_figure_reproduction.py']

for code in code_list:
    print(f'executing "{code}", time is {time.strftime("%c", time.localtime())}')
    subprocess.call(['python', str(script_directory / code)])
    