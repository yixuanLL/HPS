#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=out_test
#SBATCH --gres=gpu:1
#SBATCH --mem=24GB

source /local/scratch/yliu270/anaconda3/bin/activate flamby
dir_path=$(dirname $(pwd))
echo "${dir_path}"

output=`python ${dir_path}/privAmp/evaluate_bounds.py`
echo "${output}"
