#!/bin/bash
#SBATCH --job-name=PrepareFlantoScratch                       # Specify a name for your job
#SBATCH --output=slurm-logs/out-prepflan-%j.log               # Specify the output log file
#SBATCH --error=slurm-logs/err-prepflan-%j.log                # Specify the error log file
#SBATCH --nodes=1                                             # Number of nodes to request
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1                                     # Number of CPU cores per task
#SBATCH --time=04:00:00                                       # Maximum execution time (HH:MM:SS)
#SBATCH --qos=default                                         # Specify the partition (queue) you want to use
#SBATCH --gres=gpu:rtxa5000:0                                 # Number of GPUs per node
#SBATCH --mem=32G                                             # Memory per node


# Project setup
proj_root="/nfshomes/jrober23/Documents/lit-gpt"
scratch_root="/fs/nexus-scratch/jrober23"

# set up PyEnvironment
source $scratch_root/PyEnvs/FMMAttention/bin/activate
module load Python3

# run script
python3 $proj_root/scripts/prepare_flan.py --destination_path $scratch_root/data/flan --checkpoint_dir $scratch_root/checkpoints/meta-llama/Llama-2-7b-hf
