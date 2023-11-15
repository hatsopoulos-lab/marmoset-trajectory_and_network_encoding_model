#!/bin/bash
#SBATCH --job-name=MG_regtest
#SBATCH --output=/project/nicho/projects/dalton/job_log_files/regtest_MG_%a.out
#SBATCH --error=/project/nicho/projects/dalton/job_log_files/regtest_MG_%a.err
#SBATCH --time=2:00:00
#SBATCH --partition=caslake
#SBATCH --mem=128G
#SBATCH --array=0-51
#SBATCH --account=pi-nicho

#------------Load the appropriate applications---------------------------------------#
module load python/anaconda-2022.05 
source activate /beagle3/nicho/environments/nwb_and_neuroconv

#---------------------Execute the programs-------------------------------------------#
python /project/nicho/projects/marmosets/code_database/analysis/trajectory_encoding_model/analysis_with_nwb_format/regularization_sweep_kinFN_model.py
