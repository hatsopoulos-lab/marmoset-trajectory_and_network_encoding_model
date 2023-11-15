#!/bin/bash
#SBATCH --job-name=sample_for_encoding_models_TY
#SBATCH --output=/project/nicho/projects/dalton/network_encoding_paper/clean_final_analysis/job_log_files/sample_for_encoding_models_TY.out
#SBATCH --error=/project/nicho/projects/dalton/network_encoding_paper/clean_final_analysis/job_log_files/sample_for_encoding_models_TY.err
#SBATCH --time=16:00:00
#SBATCH --partition=caslake
#SBATCH --mem=128G
#SBATCH --account=pi-nicho

#------------Load the appropriate applications---------------------------------------#
module load python/anaconda-2022.05 
source activate /beagle3/nicho/environments/nwb_and_neuroconv

#---------------------Execute the programs-------------------------------------------#
python /project/nicho/projects/marmosets/code_database/analysis/trajectory_encoding_model/clean_final_analysis/python/collect_trajectory_samples_TY.py
