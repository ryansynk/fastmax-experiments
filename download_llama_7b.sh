#!/bin/bash
#SBATCH --job-name=DwnldTiny2Scratch                       # Specify a name for your job
#SBATCH --output=slurm-logs/out-dwnld-tiny-%j.log               # Specify the output log file
#SBATCH --error=slurm-logs/err-dwnld-tiny-%j.log                # Specify the error log file
#SBATCH --nodes=1                                             # Number of nodes to request
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1                                     # Number of CPU cores per task
#SBATCH --time=04:00:00                                       # Maximum execution time (HH:MM:SS)
#SBATCH --qos=default                                         # Specify the partition (queue) you want to use
#SBATCH --gres=gpu:rtxa5000:0                                 # Number of GPUs per node
#SBATCH --mem=32G                                             # Memory per node

# Project setup
gpt_root="/nfshomes/jrober23/Documents/lit-gpt"
scratch_root="/fs/nexus-scratch/jrober23"

# set up PyEnvironment
source $scratch_root/PyEnvs/Lit-GPT/bin/activate

# make sure your environment has the huggingface_hub[hr_transfer] (see lit-gpt tutorials for pip command)
# (this is taken care of in requirements-cmsc720.txt)

python3 $gpt_root/scripts/download.py --repo_id meta-llama/Llama-2-7b-hf --access_token ***YOUR TOKEN HERE*** --checkpoint_dir $scratch_root/checkpoints

# convert the checkpoint to lit-gpt format
python3 $gpt_root/scripts/convert_hf_checkpoint.py --checkpoint_dir $scratch_root/checkpoints/meta-llama/Llama-2-7b-hf
