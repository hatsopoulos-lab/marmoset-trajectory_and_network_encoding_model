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
    

# Installation

    conda create -n paper_demo python=3.11 numpy pandas pytables scikit-learn seaborn 
    conda activate paper_demo
    pip install -U dill neo elephant viziphant statsmodels probeinterface pynwb
    



### Basic Steps
1.	Generate functional networks and peri-event time histograms to assess unit modulation.

	-	[generate_functional_networks.py](/analysis_with_nwb_format/generate_functional_networks.py)
	-	[check_movement_psth_by_unit.py](/analysis_with_nwb_format/check_movement_psth_by_unit.py)

2.	Collect/organize neural spiking data and behavioral data

	-	[sample_trajectory_models.sbatch](/analysis_with_nwb_format/sample_trajectory_models.sbatch)
	-	[collect_trajectory_samples_TY.py](/analysis_with_nwb_format/collect_trajectory_samples_TY.py)
    
3.	Train and test trajectory encoding models **(highly parallelized)** 

	-	[create_trajectory_models.sbatch](/analysis_with_nwb_format/create_trajectory_models.sbatch)
	-	[create_trajectory_models_TY.py](/analysis_with_nwb_format/create_trajectory_models_TY.py)
    
4.	Analyze trajectory/kinematics models and produce some plots

	-	[analyze_trajectory_only_models.py](/analysis_with_nwb_format/analyze_trajectory_only_models.py)
    
5.	Train and test full network models with kinematics and functional interaction terms **(highly parallelized)**

	-	[create_network_models_TY.sbatch](/analysis_with_nwb_format/create_network_models_TY.sbatch)
    -	[create_network_models_TY.py](/analysis_with_nwb_format/create_network_models_TY.py) 
    
6.	Analyze full network models and produce rest of plots

	-	[analyze_models_with_network_terms.py](/analysis_with_nwb_format/analyze_models_with_network_terms.py)
