#!/bin/bash
#===================================================#
#SBATCH --cluster=merlin6
#SBATCH --partition=hourly
#SBATCH -V
#SBATCH --job-name=graphene
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --array=1
#SBATCH --error=error/job_%A_%a.err
#SBATCH --output=slurm/BlackP_bz_scan_%A_%a.out
#SBATCH --time=01:00:00
#===================================================#

/psi/home/pulgar_j/dynamics-w90/exe/wann_calc.x ./inp/BlackP_grid_list_nk32x32x32.inp ./out/BlackP_New_Ham

