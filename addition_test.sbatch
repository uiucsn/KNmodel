#!/bin/sh
# 
#SBATCH --time=00:03:00
#SBATCH --job-name="p_test"
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=haillep2@illinois.edu
#SBATCH --nodes=1
#SBATCH --output=add-p.txt
#SBATCH --error=add-p-e.txt
#SBATCH --partition=caps
#SBATCH --account=caps

srun python afterglow_distribution.py
###conda run -n KNenv python afterglow_params.py