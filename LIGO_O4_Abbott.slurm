#!/bin/bash
#SBATCH --qos=shared
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=vedgs2@illinois.edu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --output=Ligo_O4_Abbott.out
#SBATCH --error=Ligo_O4_Abbott.err

module load anaconda/2023-Mar/3
source activate KN-rates

python monte_carlo_sims.py --trials_dir O4-DECam-r-23mag-Abbott --ntry 20