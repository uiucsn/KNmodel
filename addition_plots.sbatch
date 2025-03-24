#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --job-name="det_params"
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=haillep2@illinois.edu
#SBATCH --nodes=1
#SBATCH --output=add-f.txt
#SBATCH --error=add-f-e.txt
#SBATCH --partition=caps
#SBATCH --account=caps

### python afterglow_testPlot.py
### python afterglow_plots.py --plot
python afterglow_detection.py
###python afterglow_params.py
###python afterglow_distribution.py --plot
###python waveforms.py
###conda run -n KNenv python bulla_fitter.py