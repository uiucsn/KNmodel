#!/bin/bash
#SBATCH --time=6-15:30:00
#SBATCH --job-name="Ligo_O5_Nitz"
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=vedgs2@illinois.edu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --output=Ligo_O5_Nitz.out
#SBATCH --error=Ligo_O5_Nitz.err
#SBATCH --partition=caps

export PYTHONHOME='/projects/caps/uiucsn/ved'

module load anaconda/2023-Mar/3
source activate KN-rates

python monte_carlo_sims.py --trials_dir O5-LSST-r-24mag-Nitz --ligo_run O5 --detection_passband lsstr --detection_threshold 24 --rate_model Nitz23 --ntry 1000