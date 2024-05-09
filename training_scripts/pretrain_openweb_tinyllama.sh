#!/bin/bash
#SBATCH --job-name=PretrainTinyLlama # Specify a name for your job
#SBATCH --output=../slurm-logs/pretrain_tinyllama-%j.log         # Specify the output log file
#SBATCH --error=../slurm-logs/pretrain_tinyllama-%j.log          # Specify the error log file
#SBATCH --partition=tron
#SBATCH --qos=medium
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:rtxa6000:2                                # Number of GPUs per node
#SBATCH --mem=32gb

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
    --train.micro_batch_size 4 \
    --train.global_batch_size 8 \
    --train.epoch_size 8000000000 \
    --train.save_interval 1000 \
    --train.max_seq_length 2048 \
    --train.learning_rate 4e-4 \
    --train.min_lr 4e-5 \
    --train.beta2 0.999 \
    --devices 2 \
    --io.out_dir $proj_root/out/tinyllama_openwebtext
