#!/bin/bash

#SBATCH --job-name=ga
#SBATCH --partition=hype
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=2:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

cd "$SCRATCH"

if [ -d "paralela" ]; then rm -fr paralela; fi

git clone  --recurse-submodules https://github.com/Jubarte27/paralela
cd paralela

./init.sh
source .venv/bin/activate
python3 generate_design.py
srun ./run_all.sh

cp -r "$SCRATCH/paralela" "$HOME/ga_job/"

