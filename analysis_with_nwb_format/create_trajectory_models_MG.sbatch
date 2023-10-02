#!/bin/bash
#SBATCH --job-name=create_trajectory_models_MG
#SBATCH --output=/project/nicho/projects/dalton/job_log_files/create_trajectory_models_MG_%a.out
#SBATCH --error=/project/nicho/projects/dalton/job_log_files/create_trajectory_models_MG_%a.err
#SBATCH --time=00:50:00
#SBATCH --partition=caslake
#SBATCH --array=0-254     
#SBATCH --mem=64G
#SBATCH --account=pi-nicho

# 0-764
#0-713
#0-83

#------------Load the appropriate applications---------------------------------------#
module load python/anaconda-2022.05 
source activate /project/nicho/environments/spyder

#---------------------Execute the programs-------------------------------------------#
python /project/nicho/projects/marmosets/code_database/analysis/trajectory_encoding_model/analysis_with_nwb_format/create_trajectory_models_MG.py
