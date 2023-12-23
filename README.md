# marmoset-trajectory_and_network_encoding_model

# Overview

Contains code for analysis and visualization of data for a manuscript in final stages of preparation. Some of the code 
is complete, but some requires cleanup prior to publishing. 

# Software Requirements

This code has been tested in a fresh conda environment using python>=3.11.

## Python dependencies
    
    numpy
    pandas
    pytables
    scikit-learn
    scipy
    pynwb
    ndx_pose
    neo
    elephant
    viziphant
    quantities
    dill
    statsmodels
    seaborn
    probeinterface
    

# Installation (10-30 minutes)

Create conda environment and install dependencies.

    conda create -n marm_encoding_model python=3.11 numpy pandas pytables scikit-learn seaborn 
    conda activate marm_encoding_model
    pip install -U dill neo elephant viziphant statsmodels probeinterface pynwb ndx-pose
    
Download or clone this repository.

Download sample data and place it in "USERPATH/marmoset-trajectory_and_network_encoding_model/demo/data". Within "data" you should see the folders called "original" and "demo".

For reviewers: data will be supplied via a link from the editor.

For other users: the nwb file for running the demo is located at **Insert DANDI link**

# Demo the code

This should take ~XX minutes on a standard computer.

Open the in a text editor or python IDE: [/demo/MG_subset_demo/code/run_demo.py](/demo/MG_subset_demo/code/run_demo.py)

This code will execute all the scripts in order and generate plots, which will populate in "marmoset-trajectory_and_network_encoding_model/demo/MG_subset_demo/plots". You can also open and run each code independently, in the order prescribed in run_demo.py.

The demo analyzes all units and reach samples available in the MG dataset. However, only one lead/lag set of [-100, +300]ms is analyzed and only 5 train/test splits are completed (instead of the full 500). A few results that can't be replicated with only 5 train/test splits are disabled.

# Reproduce MG data 

This should take less than 5 minutes on a standard computer.

Open in a text editor or python IDE: [/demo/MG_original_data_reproduction/code/reproduce_figures.py](/demo/MG_original_data_reproduction/code/reproduce_figures.py)

This code will execute the figure reproduction scripts in order and generate plots, which will populate in "marmoset-trajectory_and_network_encoding_model/demo/MG_original_data_reproduction/plots". You can also open and run each code independently, in the order prescribed in reproduce_figures.py.

# Basic Workflow 
The full code is located in /clean_final_analysis/. This code is intended to run on the slurm job management system. Both python files and the associated sbatch job submission files are linked below. 

1.	Generate functional networks and peri-event time histograms to assess unit modulation.

	-	[generate_functional_networks.py](/clean_final_analysis/python/generate_functional_networks.py)
	-	[check_movement_peth_by_unit.py](/clean_final_analysis/python/check_movement_peth_by_unit.py)

2.	Collect/organize neural spiking data and behavioral data

	-	[sample_trajectory_models.sbatch](/analysis_with_nwb_format/sample_trajectory_models.sbatch)
	-	[collect_trajectory_samples_TY.py](/analysis_with_nwb_format/collect_trajectory_samples_TY.py)
    
3.	Train and test trajectory encoding models **(highly parallelized)** 

	-	[create_trajectory_models.sbatch](/clean_final_analysis/sbatch/create_trajectory_models.sbatch)
	-	[create_trajectory_models_TY.py](/clean_final_analysis/python/create_trajectory_models_TY.py)
    
4.	Analyze trajectory/kinematics models and produce some plots

	-	[analyze_trajectory_only_models.py](/clean_final_analysis/python/analyze_trajectory_only_models.py)
    
5.	Train and test full network models with kinematics and functional interaction terms **(highly parallelized)**

	-	[create_network_models_TY.sbatch](/clean_final_analysis/sbatch/create_network_models_TY.sbatch)
    -	[create_network_models_TY.py](/clean_final_analysis/python/create_network_models_TY.py) 
    
6.	Analyze full network models and produce rest of plots

	-	[analyze_models_with_network_terms.py](/clean_final_analysis/python/analyze_models_with_network_terms.py)
