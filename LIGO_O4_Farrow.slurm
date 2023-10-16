#!/bin/bash
#SBATCH --time=6-15:30:00
#SBATCH --job-name="Ligo_O4_Farrow"
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=vedgs2@illinois.edu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --output=Ligo_O4_farrow.out
#SBATCH --error=Ligo_O4_farrow.err
#SBATCH --partition=caps

module load anaconda/2023-Mar/3
source activate /projects/caps/uiucsn/ved

python monte_carlo_sims.py --trials_dir O4-DECam-r-23mag-Farrow --ntry 1000  --mass_distrib Farrow19