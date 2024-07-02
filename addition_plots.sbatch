#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --job-name="aft_plot"
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=haillep2@illinois.edu
#SBATCH --nodes=1
#SBATCH --output=add-f.txt
#SBATCH --error=add-f-e.txt
#SBATCH --partition=caps

###srun python finesstplot.py
srun python afterglow_detection.py
###srun python afterglow_params.py
###srun python afterglow_plots.py --plot
###srun python waveforms.py