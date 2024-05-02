#!/bin/bash
#SBATCH --job-name=PretrainTinyLlama # Specify a name for your job
#SBATCH --output=../slurm-logs/pretrain_tinyllama-%j.log         # Specify the output log file
#SBATCH --error=../slurm-logs/pretrain_tinyllama-%j.log          # Specify the error log file
#SBATCH --nodes=1                                             # Number of nodes to request
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4                                     # Number of CPU cores per task
#SBATCH --time=24:00:00                                       # Maximum execution time (HH:MM:SS)
#SBATCH --qos=high                                            # Specify the partition (queue) you want to use
#SBATCH --gres=gpu:rtxa5000:1                                # Number of GPUs per node
#SBATCH --mem=64G                                             # Memory per node

# Project setup

# Project setup
proj_root="/fs/nexus-projects/FAST_Attention"
data_root="/fs/nexus-projects/FAST_Attention/data"

# set up PyEnvironment
source ~/.bashrc
conda activate fast
wandb login
cd ..

# run finetune, saving to gpt_root so nexus backs up fine tuned models but leaving base models on scratch
python pretrain/openwebtext.py \
    --io.train_data_dir $data_root/openwebtext \
    --io.val_data_dir $data_root/openwebtext \
    --model_name "tiny-llama-1.1b" \
    --train.micro_batch_size 1 \
    --train.global_batch_size 1 \
    --io.out_dir $proj_root/out
