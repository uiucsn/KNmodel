#!/bin/bash
#SBATCH --time=6-15:30:00
#SBATCH --job-name="Ligo_O4_ztf"
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=vedgs2@illinois.edu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --output=Ligo_O4_ztf.out
#SBATCH --error=Ligo_O4_ztf.err
#SBATCH --partition=caps

module load anaconda/2023-Mar/3
source activate /projects/caps/uiucsn/ved

python monte_carlo_sims.py --trials_dir O4-ztf-r --ntry 1000 --detection_passband ztfr --detection_threshold 21.4