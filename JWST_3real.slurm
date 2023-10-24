#!/bin/bash
#SBATCH --time=6-15:30:00
#SBATCH --job-name="jwst_3real"
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=vedgs2@illinois.edu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=9
#SBATCH --output=jwst_3real.out
#SBATCH --error=jwst_3real.err
#SBATCH --partition=caps

module load anaconda/2023-Mar/3
source activate /projects/caps/uiucsn/ved

python jwst_sim_3real.py --trials_dir jwst_3real --ntry 1000