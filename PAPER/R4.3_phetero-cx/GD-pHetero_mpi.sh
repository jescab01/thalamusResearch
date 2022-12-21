#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --partition=debug
#SBATCH --job-name=GD
#SBATCH --ntasks=25
#SBATCH --time=00:24:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=F_r_e@hotmail.es
#SBATCH --error=logs/err-%j.log
#SBATCH --output=logs/out-%j.log
##------------------------ End job description ------------------------

module purge && module load Python/3.9.6-GCCcore-11.2.0

srun python GD-pHetero_mpi.py


