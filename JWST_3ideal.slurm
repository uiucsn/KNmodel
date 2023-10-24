#!/bin/bash
#SBATCH --time=6-15:30:00
#SBATCH --job-name="jwst_3ideal"
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=vedgs2@illinois.edu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=9
#SBATCH --output=jwst_3ideal.out
#SBATCH --error=jwst_3ideal.err
#SBATCH --partition=caps

module load anaconda/2023-Mar/3
source activate /projects/caps/uiucsn/ved

python jwst_sim_3ideal.py --trials_dir jwst_3ideal --ntry 1000