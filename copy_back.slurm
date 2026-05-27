#!/bin/bash

#SBATCH --job-name=ga_copy
#SBATCH --partition=hype
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

cp -r "$SCRATCH/paralela/" "$HOME/ga_job/"
