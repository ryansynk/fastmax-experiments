#!/bin/bash
#SBATCH --job-name=generate                             # Specify a name for your job
#SBATCH --output=slurm-logs/out-gen-%j.log         # Specify the output log file
#SBATCH --error=slurm-logs/err-gen-%j.log          # Specify the error log file
#SBATCH --nodes=1                                             # Number of nodes to request
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1                                     # Number of CPU cores per task
#SBATCH --time=24:00:00                                       # Maximum execution time (HH:MM:SS)
#SBATCH --qos=high                                            # Specify the partition (queue) you want to use
#SBATCH --gres=gpu:rtxa5000:1                                 # Number of GPUs per node
#SBATCH --mem=32G                                             # Memory per node

gpt_root="/nfshomes/jrober23/Documents/lit-gpt"
scratch_root="/fs/nexus-scratch/jrober23"
WANDB_API_KEY="0ac25eead122887ad5f73d917b545b9ee5e2f3b5"

# set up PyEnvironment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $scratch_root/LitGPT/

python3 generate/lora.py --prompt "Read me a children bedtime story" --lora_path "/fs/nexus-scratch/jrober23/out/lora_weights/Llama2-Flan-Quad/lit_model_lora_finetuned.pth" --checkpoint_dir "/fs/nexus-scratch/jrober23/checkpoints/meta-llama/Llama-2-7b-hf/" --quantize "bnb.nf4" --attn_alg linearmax
