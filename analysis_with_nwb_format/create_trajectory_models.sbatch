#!/bin/bash
#SBATCH --job-name=create_trajectory_models
#SBATCH --output=/project/nicho/projects/dalton/job_log_files/create_trajectory_models_%a.out
#SBATCH --error=/project/nicho/projects/dalton/job_log_files/create_trajectory_models_%a.err
#SBATCH --time=00:40:00
#SBATCH --partition=caslake
#SBATCH --array=0-713
#SBATCH --mem=64G
#SBATCH --account=pi-nicho

#------------Load the appropriate applications---------------------------------------#
module load python/anaconda-2022.05 
source activate /project/nicho/environments/spyder

#---------------------Execute the programs-------------------------------------------#
python /project/nicho/projects/marmosets/code_database/analysis/trajectory_encoding_model/analysis_with_nwb_format/create_trajectory_only_models_in_full_parallel.py