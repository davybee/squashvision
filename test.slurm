#!/bin/bash
#SBATCH --job-name=detects
#SBATCH --gres=gpu:1
#SBATCH -o std_out_err/%j.out
#SBATCH -e std_out_err/%j.err
#SBATCH --time 03:59:00
#SBATCH -c 10
#SBATCH --mem=64G

module load anaconda3/2025.6
conda activate label

python testing.py