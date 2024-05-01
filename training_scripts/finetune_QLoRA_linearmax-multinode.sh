#!/bin/bash
#SBATCH --job-name=LoraLinearMaxFlan                             # Specify a name for your job
#SBATCH --output=slurm-logs/out-flan-linearmax-%j.log         # Specify the output log file
#SBATCH --nodes=1                                             # Number of nodes to request
#SBATCH --ntasks-per-node=8          # crucial - only 1 task per dist per node!
#SBATCH --gres=gpu:rtxa6000:8   # Number of GPUs to request and specify the GPU type
#SBATCH --time=24:00:00            # Maximum execution time (HH:MM:SS)
#SBATCH --partition=cbcb-heng   # Partition name
#SBATCH --account=cbcb-heng
#SBATCH --qos=huge-long
#SBATCH --mem=32G                # Memory per node

scratch_root="/fs/nexus-scratch/rezashkv/research/projects/lit-gpt-CMSC720Proj"

source ~/.bashrc
conda activate litgpt
cd $scratch_root || exit
# run finetune, saving to gpt_root so nexus backs up fine tuned models but leaving base models on scratch
# python3 finetune/lora.py --io.train_data_dir $scratch_root/data/flan --io.val_data_dir $scratch_root/data/flan --io.checkpoint_dir $scratch_root/checkpoints/meta-llama/Llama-2-7b-hf --io.out_dir $scratch_root/out/lora_weights/Llama2-Flan-Quad-NoQuant --train.micro_batch_size 1 --precision bf16-true --devices 1
srun python3 finetune/lora.py --attn_alg linearmax --io.train_data_dir $scratch_root/data/flan --io.val_data_dir $scratch_root/data/flan --io.checkpoint_dir $scratch_root/checkpoints/meta-llama/Llama-2-7b-hf --io.out_dir $scratch_root/out/lora_weights/Llama2-Flan-linearmax-NoQuant --train.micro_batch_size 1 --precision bf16-true --devices 8
