#!/bin/bash

#SBATCH --job-name=FineTiny_Full                              # Specify a name for your job

#SBATCH --output=slurm-logs/out-fine-tiny-full-%j.log         # Specify the output log file

#SBATCH --error=slurm-logs/err-fine-tiny-full-%j.log          # Specify the error log file

#SBATCH --nodes=1                                             # Number of nodes to request

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=1                                     # Number of CPU cores per task

#SBATCH --time=12:00:00                                       # Maximum execution time (HH:MM:SS)

#SBATCH --qos=high                                            # Specify the partition (queue) you want to use

#SBATCH --gres=gpu:rtxa6000:2                                 # Number of GPUs per node

#SBATCH --mem=32G                                             # Memory per node



# Project setup

gpt_root="/nfshomes/jrober23/Documents/lit-gpt"

scratch_root="/fs/nexus-scratch/jrober23"



# set up PyEnvironment

source /fs/nexus-scratch/jrober23/PyEnvs/FMMAttention/bin/activate

module load Python3



# run finetune, saving to gpt_root so nexus backs up fine tuned models but leaving base models on scratch

python3 finetune/full.py --data_dir $scratch_root/data/alpaca --checkpoint_dir $scratch_root/checkpoints/TinyLlama/TinyLlama-1.1B-Chat-v1.0 --out_dir $scratch_root/out/full_weight/TinyLlama --resume True
