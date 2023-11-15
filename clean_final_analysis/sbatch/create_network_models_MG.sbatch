#!/bin/bash
#SBATCH --job-name=MG_create_network_models
#SBATCH --output=/project/nicho/projects/dalton/network_encoding_paper/clean_final_analysis/job_log_files/create_network_models_MG_%a.out
#SBATCH --error=/project/nicho/projects/dalton/network_encoding_paper/clean_final_analysis/job_log_files/create_network_models_MG_%a.err
#SBATCH --time=36:00:00
#SBATCH --partition=caslake
#SBATCH --mem=128G
#SBATCH --array=0-15
#SBATCH --account=pi-nicho

#9 11 21 23
#0-23
#------------Load the appropriate applications---------------------------------------#
module load python/anaconda-2022.05 
source activate /beagle3/nicho/environments/nwb_and_neuroconv

#---------------------Execute the programs-------------------------------------------#
python /project/nicho/projects/marmosets/code_database/analysis/trajectory_encoding_model/clean_final_analysis/python/create_network_models_MG.py