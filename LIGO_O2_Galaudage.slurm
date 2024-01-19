#!/bin/bash
#SBATCH --time=6-15:30:00
#SBATCH --job-name="Ligo_O2"
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=vedgs2@illinois.edu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=Ligo_O2.out
#SBATCH --error=Ligo_O2.err
#SBATCH --partition=caps

module load anaconda/2023-Mar/3
source activate /projects/caps/uiucsn/ved

python monte_carlo_sims.py --trials_dir O2-DECam-r-Galaudage --ligo_run O2  --ntry 1000