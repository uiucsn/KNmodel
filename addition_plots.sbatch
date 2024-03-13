#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --job-name="aft_plot"
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=haillep2@illinois.edu
#SBATCH --nodes=1
#SBATCH --output=add-f.txt
#SBATCH --error=add-f-e.txt
#SBATCH --partition=caps
srun python finesstplot.py