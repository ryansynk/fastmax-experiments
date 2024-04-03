#!/bin/bash
#SBATCH --job-name=full_quad_alpaca                             # Specify a name for your job
#SBATCH --output=slurm-logs/full_quad_alpaca -%j.log         # Specify the output log file
#SBATCH --nodes=1                                             # Number of nodes to request
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2                                     # Number of CPU cores per task
#SBATCH --time=24:00:00                                       # Maximum execution time (HH:MM:SS)
#SBATCH --qos=high                                            # Specify the partition (queue) you want to use
#SBATCH --gres=gpu:rtxa6000:1                                # Number of GPUs per node
#SBATCH --mem=64G                                             # Memory per node

# Project setup

# Project setup
# scratch_root="/fs/nexus-scratch/jrober23"
scratch_root="/vulcanscratch/mhoover4/code/lit-gpt-CMSC720Proj"

# set up PyEnvironment
# source $scratch_root/PyEnvs/Lit-GPT/bin/activate
source $(conda info --base)/etc/profile.d/conda.sh          # Use if conda is already on your path but you still need to run "conda init <shell_name>"       
conda activate cmsc720

# run finetune, saving to gpt_root so nexus backs up fine tuned models but leaving base models on scratch
# python3 finetune/lora.py --io.train_data_dir $scratch_root/data/flan --io.val_data_dir $scratch_root/data/flan --io.checkpoint_dir $scratch_root/checkpoints/meta-llama/Llama-2-7b-hf --io.out_dir $scratch_root/out/lora_weights/Llama2-Flan-Quad-NoQuant --train.micro_batch_size 1 --precision bf16-true --devices 1
python3 finetune/full.py --attn_alg quadratic --io.train_data_dir $scratch_root/data/alpaca --io.val_data_dir $scratch_root/data/alpaca --io.checkpoint_dir $scratch_root/checkpoints/meta-llama/Llama-2-7b-hf --io.out_dir $scratch_root/out/full_weights/Llama2-alpaca-linearmax-NoQuant --train.micro_batch_size 1 --precision bf16-true --devices 1
