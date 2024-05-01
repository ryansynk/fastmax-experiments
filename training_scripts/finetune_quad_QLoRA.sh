#!/bin/bash
#SBATCH --job-name=LoraQuadFlan                             # Specify a name for your job
#SBATCH --output=slurm-logs/out-flan-quad-%j.log         # Specify the output log file
#SBATCH --error=slurm-logs/err-flan-quad-%j.log          # Specify the error log file
#SBATCH --nodes=1                                             # Number of nodes to request
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1                                     # Number of CPU cores per task
#SBATCH --time=24:00:00                                       # Maximum execution time (HH:MM:SS)
#SBATCH --qos=high                                            # Specify the partition (queue) you want to use
#SBATCH --gres=gpu:rtxa6000:1                                # Number of GPUs per node
#SBATCH --mem=32G                                             # Memory per node

# Project setup

gpt_root="/nfshomes/jrober23/Documents/lit-gpt"
scratch_root="/fs/nexus-scratch/jrober23"
WANDB_API_KEY="0ac25eead122887ad5f73d917b545b9ee5e2f3b5"

# set up PyEnvironment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $scratch_root/LitGPT/
wandb login $WANDB_API_KEY

# run finetune, saving to gpt_root so nexus backs up fine tuned models but leaving base models on scratch
python3 finetune/lora.py --io.train_data_dir $scratch_root/data/flan --io.val_data_dir $scratch_root/data/flan --io.checkpoint_dir $scratch_root/checkpoints/meta-llama/Llama-2-7b-hf --io.out_dir $scratch_root/out/lora_weights/Llama2-Flan-Quad-NoQuant --train.micro_batch_size 1 --precision bf16-true --devices 1
