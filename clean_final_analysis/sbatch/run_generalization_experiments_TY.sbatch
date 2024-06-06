#!/bin/bash
#SBATCH --job-name=TY_network_generalization_experiments
#SBATCH --output=/project/nicho/projects/dalton/network_encoding_paper/clean_final_analysis/job_log_files/TY_network_generalization_experiments_%a.out
#SBATCH --error=/project/nicho/projects/dalton/network_encoding_paper/clean_final_analysis/job_log_files/TY_network_generalization_experiments_%a.err
#SBATCH --time=18:00:00
#SBATCH --partition=beagle3
#SBATCH --mem=64G
#SBATCH --array=0-7
#SBATCH --account=pi-nicho

#9 11 21 23
#0-23
#------------Load the appropriate applications---------------------------------------#
module load python/anaconda-2022.05 
source activate /beagle3/nicho/environments/nwb_and_neuroconv

#---------------------Execute the programs-------------------------------------------#
python /project/nicho/projects/marmosets/code_database/analysis/trajectory_encoding_model/clean_final_analysis/python/generalization_experiments_TY.py
