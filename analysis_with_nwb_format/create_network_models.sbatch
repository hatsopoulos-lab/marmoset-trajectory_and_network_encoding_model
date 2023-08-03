#!/bin/bash
#SBATCH --job-name=create_network_models_cortical_areas
#SBATCH --output=/project/nicho/projects/dalton/job_log_files/create_network_models_cortical_areas_%a.out
#SBATCH --error=/project/nicho/projects/dalton/job_log_files/create_network_models_cortical_areas_%a.err
#SBATCH --time=36:00:00
#SBATCH --partition=caslake
#SBATCH --mem=128G
#SBATCH --array=0,9,11,21,23
#SBATCH --account=pi-nicho

#------------Load the appropriate applications---------------------------------------#
module load python/anaconda-2022.05 
source activate /project/nicho/environments/spyder

#---------------------Execute the programs-------------------------------------------#
python /project/nicho/projects/marmosets/code_database/analysis/trajectory_encoding_model/analysis_with_nwb_format/create_models_with_networks_and_cortical_area_subnetworks.py