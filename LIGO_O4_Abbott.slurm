#!/bin/bash
#SBATCH --time=6-15:30:00
#SBATCH --job-name="Ligo_O4_Abbott"
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=vedgs2@illinois.edu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --output=Ligo_O4_Abbott.out
#SBATCH --error=Ligo_O4_Abbott.err
#SBATCH --partition=caps

export PYTHONHOME='/projects/caps/uiucsn/ved'

module load anaconda/2023-Mar/3
source activate KN-rates

python monte_carlo_sims.py --trials_dir O4-DECam-r-23mag-Abbott  --ntry 1000