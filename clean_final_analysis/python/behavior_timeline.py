#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 10:34:58 2024

@author: daltonm
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pathlib import Path

import dill
import re
import seaborn as sns
from scipy import sparse
from sklearn.metrics.cluster import normalized_mutual_info_score

from pynwb import NWBHDF5IO
import ndx_pose
from importlib import sys

data_path = Path('/project/nicho/projects/dalton/network_encoding_paper/clean_final_analysis/data')
code_path = Path('/project/nicho/projects/marmosets/code_database/analysis/trajectory_encoding_model/clean_final_analysis/')

sys.path.insert(0, str(code_path))
from hatlab_nwb_functions import get_sorted_units_and_apparatus_kinematics_with_metadata   
from utils import choose_units_for_model

class timeline_data:
    def __init__(self):
        self.reaches = None
        self.app_visits = None   

    def get_reach_times(self, reaches, kin_module):
        
        first_event_key = [key for idx, key in enumerate(kin_module.data_interfaces.keys()) if idx == 0][0]
        dlc_scorer = kin_module.data_interfaces[first_event_key].scorer 
        
        if 'simple_joints_model' in dlc_scorer:
            wrist_label = 'hand'
        elif 'marmoset_model' in dlc_scorer and nwb.subject.subject_id == 'TY':
            wrist_label = 'l-wrist'
        elif 'marmoset_model' in dlc_scorer and nwb.subject.subject_id == 'MG':
            wrist_label = 'r-wrist'
        
        reach_start = []
        reach_stop  = []
        for reachNum, reach in reaches.iterrows():      
                    
            # get event data using container and ndx_pose names from segment_info table following form below:
            # nwb.processing['goal_directed_kinematics'].data_interfaces['moths_s_1_e_004_position']
            event_data      = kin_module.data_interfaces[reach.video_event]    
            timestamps          = event_data.pose_estimation_series[   wrist_label].timestamps[reach.start_idx:reach.stop_idx+1]
    
            reach_start.append(timestamps[0])
            reach_stop.append (timestamps[-1])
        
        self.reaches = pd.DataFrame(data=zip(reach_start, reach_stop), columns=['start', 'stop'])
        self.reaches['duration'] = self.reaches['stop'] - self.reaches['start']
        
    def get_app_visits(self, annotation, nwb):
        annotation_data = pd.read_csv(annotation)
        free_cams_timestamps = nwb.processing['video_event_timestamps_free'].data_interfaces['free_s_1_e_001_timestamps'].timestamps[:]
        self.session_duration = free_cams_timestamps[-1]

        class_num = int(annotation_data.loc[annotation_data['class_options'] == 'Apparatus', 'class_num'].values[0])

        start_frames = annotation_data.loc[annotation_data['Class'] == class_num, 'Start'].values.astype(int) 
        stop_frames  = annotation_data.loc[annotation_data['Class'] == class_num, 'Stop'].values.astype(int)
        stop_frames[stop_frames == -1] = len(free_cams_timestamps)-1   
        
        start_times = free_cams_timestamps[start_frames]
        stop_times  = free_cams_timestamps[stop_frames]
        
        self.app_visits = pd.DataFrame(data=zip(start_times, stop_times), columns=['start', 'stop'])
        self.app_visits['duration'] = self.app_visits['stop'] - self.app_visits['start']
        
    def plot_timeline(self, figsize=None, title=None):
        fig, ax = plt.subplots(figsize=figsize)
        ax.axhline(0, c="black")
        visit_colors = ['gray', 'black']
        reach_colors = [(  0/255, 141/255, 208/255), (183/255, 219/255, 165/255)]
        for row, app_vis in self.app_visits.iterrows():
            ax.fill_between([app_vis.start, app_vis.stop], 0, 1, color=visit_colors[row % 2])
        for row, reach in self.reaches.iterrows():
            ax.fill_between([reach.start, reach.stop], 0, -1, color=reach_colors[row % 2])
        
        ax.set_title(title)
        sns.despine(left=True)
        plt.show()
        
                

if __name__ == "__main__":

    TY_timeline, MG_timeline = timeline_data(), timeline_data()    

    for marm, timeline in zip(['TY', 'MG'], [TY_timeline, MG_timeline]): 

        if marm == 'TY':
            nwb_infile   = data_path / 'TY' / 'TY20210211_freeAndMoths-003_resorted_20230612_DM_with_functional_networks.nwb'
            annotation   = data_path / 'TY' / 'spontaneous_behavior_annotation_TY20210211.csv'
            
        elif marm == 'MG':
            nwb_infile   = data_path / 'MG' / 'MG20230416_1505_mothsAndFree-002_processed_DM_with_functional_networks.nwb'
            annotation   = data_path / 'MG' / 'spontaneous_behavior_annotation_MG20230416.csv'

        with NWBHDF5IO(nwb_infile, 'r') as io_in:
            nwb = io_in.read()
    
            reaches_key = [key for key in nwb.intervals.keys() if 'reaching_segments' in key][0]
            
            units, reaches, kin_module = get_sorted_units_and_apparatus_kinematics_with_metadata(nwb, reaches_key, plot=False)
            
            timeline.get_reach_times(reaches, kin_module)
        
            timeline.get_app_visits(annotation, nwb)
        
        timeline.plot_timeline(figsize=(10, 2), title=marm)
    
        print(f'''
              Marmoset {marm}:
                  visits        = {timeline.app_visits.shape[0]} (avg {round(timeline.app_visits["duration"].mean()/60, 2)} minutes, total {round(timeline.app_visits["duration"].sum()/60, 2)} minutes)
                  reaches       = {timeline.reaches.shape[0]} (avg {round(timeline.reaches["duration"].mean(), 2)} seconds, total {round(timeline.reaches["duration"].sum()/60, 2)} minutes)
                  reaches/visit = {round(timeline.reaches.shape[0]/timeline.app_visits.shape[0], 2)}
                  session duration = {round(timeline.session_duration / 60, 2)} minutes
              ''')