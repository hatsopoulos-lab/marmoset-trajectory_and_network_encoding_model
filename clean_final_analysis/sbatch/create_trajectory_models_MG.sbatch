#!/bin/bash
#SBATCH --job-name=create_trajectory_models_MG
#SBATCH --output=/project/nicho/projects/dalton/network_encoding_paper/clean_final_analysis/job_log_files/create_trajectory_models_MG_%a.out
#SBATCH --error=/project/nicho/projects/dalton/network_encoding_paper/clean_final_analysis/job_log_files/create_trajectory_models_MG_%a.err
#SBATCH --time=5:00:00
#SBATCH --partition=caslake
#SBATCH --array=0-237     
#SBATCH --mem=64G
#SBATCH --account=pi-nicho

#------------Load the appropriate applications---------------------------------------#
module load python/anaconda-2022.05 
source activate /beagle3/nicho/environments/nwb_and_neuroconv

#---------------------Execute the programs-------------------------------------------#
python /project/nicho/projects/marmosets/code_database/analysis/trajectory_encoding_model/clean_final_analysis/python/create_trajectory_models_MG.py
