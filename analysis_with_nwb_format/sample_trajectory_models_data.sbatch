#!/bin/bash
#SBATCH --job-name=sample_for_encoding_models
#SBATCH --output=/project/nicho/projects/dalton/job_log_files/sample_for_encoding_models.out
#SBATCH --error=/project/nicho/projects/dalton/job_log_files/sample_for_encoding_models.err
#SBATCH --time=24:00:00
#SBATCH --partition=caslake
#SBATCH --mem=128G
#SBATCH --account=pi-nicho

#------------Load the appropriate applications---------------------------------------#
module load python/anaconda-2022.05 
source activate /project/nicho/environments/spyder

#---------------------Execute the programs-------------------------------------------#
python /project/nicho/projects/marmosets/code_database/analysis/trajectory_encoding_model/analysis_with_nwb_format/collect_trajectory_samples_and_test_regularization.py
