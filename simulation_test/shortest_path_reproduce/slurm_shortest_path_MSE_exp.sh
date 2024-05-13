#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=3:30:00
#SBATCH --mail-user=yongpeng@usc.edu
#SBATCH --account=vayanou_651
#SBATCH --output=./outputs/slurmlogs/%x_%j.out
#SBATCH --error=./outputs/slurmlogs/%x_%j.err

#

module purge
module load gcc 

eval "$(conda shell.bash hook)"
conda activate pyepo_dsl

eval "$1"