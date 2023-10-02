#!/bin/bash
#SBATCH --job-name=create_network_models_MG
#SBATCH --output=/project/nicho/projects/dalton/job_log_files/create_network_models_MG_%a.out
#SBATCH --error=/project/nicho/projects/dalton/job_log_files/create_network_models_MG_%a.err
#SBATCH --time=36:00:00
#SBATCH --partition=caslake
#SBATCH --mem=128G
#SBATCH --array=0-23
#SBATCH --account=pi-nicho

#9 11 21 23
#0-23
#------------Load the appropriate applications---------------------------------------#
module load python/anaconda-2022.05 
source activate /project/nicho/environments/spyder

#---------------------Execute the programs-------------------------------------------#
python /project/nicho/projects/marmosets/code_database/analysis/trajectory_encoding_model/analysis_with_nwb_format/create_network_models_MG.py