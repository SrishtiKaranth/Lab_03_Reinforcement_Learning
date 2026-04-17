#!/bin/bash
#SBATCH --job-name=mario-a3c
#SBATCH --account=project_2018838
#SBATCH --partition=small
#SBATCH --time=14:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/scratch/project_2018838/ml-mario-rl/logs/a3c_%j.out
#SBATCH --error=/scratch/project_2018838/ml-mario-rl/logs/a3c_%j.err

mkdir -p /scratch/project_2018838/ml-mario-rl/logs

module purge
module load pytorch/2.3
source mario_env/bin/activate

# Go to working dir
cd /scratch/project_2018838/ml-mario-rl

python mario_a3c_v-1.py

