#!/bin/bash
#SBATCH --time=6-15:30:00
#SBATCH --job-name="o4vso5"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vedgs2@illinois.edu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=9
#SBATCH --mem-per-cpu=45G
#SBATCH --output=o4vso5.out
#SBATCH --error=o4vso5.err
#SBATCH --partition=caps

module load anaconda/2023-Mar/3
source activate /projects/caps/uiucsn/ved

python O4-O5-critical.py --ntry 1000