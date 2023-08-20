#!/bin/bash
#SBATCH --job-name="Ligo_O4_Nitz"
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=vedgs2@illinois.edu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --output=Ligo_O4_Abbott.out
#SBATCH --error=Ligo_O4_Abbott.err
#SBATCH --partition=secondary

module load anaconda/2023-Mar/3
source activate KN-rates

python monte_carlo_sims.py --trials_dir O4-DECam-r-23mag-Nitz --rate_model 'Nitz23'