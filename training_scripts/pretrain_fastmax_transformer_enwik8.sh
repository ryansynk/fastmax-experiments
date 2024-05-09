#!/bin/bash
#SBATCH --job-name=PrepareOpenWebText                       # Specify a name for your job
#SBATCH --output=slurm-logs/out-prepflan-%j.log               # Specify the output log file
#SBATCH --error=slurm-logs/out-prepflan-%j.log                # Specify the error log file
#SBATCH --nodes=1                                             # Number of nodes to request
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16                                     # Number of CPU cores per task
#SBATCH --time=04:00:00                                       # Maximum execution time (HH:MM:SS)
#SBATCH --qos=high                                         # Specify the partition (queue) you want to use
# #SBATCH --gres=gpu:rtxa5000:1                                 # Number of GPUs per node
#SBATCH --mem=64G                                             # Memory per node

data_root="/fs/nexus-projects/FAST_Attention/data"
proj_root="/fs/nexus-projects/FAST_Attention"
proj_root="/fs/nexus-scratch/ryansynk/fastmax-experiments"

source ~/.bashrc
conda activate fast

#TODO: Edit hyperparams
python pretrain/enwik8.py \
    --io.train_data_dir /fs/nexus-projects/FAST_Attention/data \
    --io.val_data_dir /fs/nexus-projects/FAST_Attention/data \
    --model_name "easy-transformer-fastmax" \
    --train.micro_batch_size 1 \
    --train.global_batch_size 4 \
    --train.learning_rate 1e-3 \
    --train.min_lr 1e-5 \
    --train.beta2 0.999 \
    --eval.interval 100 \
    --io.out_dir /fs/nexus-projects/FAST_Attention/out/easytransformerfastmax_enwik8
