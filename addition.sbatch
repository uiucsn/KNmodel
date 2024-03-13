#!/bin/sh
#SBATCH --time=10:00:00
#SBATCH --job-name="distr"
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=haillep2@illinois.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --output=add_dist.txt
#SBATCH --error=add_distr-e.txt
#SBATCH --partition=caps
#SBATCH --account=caps

### srun python afterglow_distribution.py
conda run -n KNenv python afterglow_distribution.py