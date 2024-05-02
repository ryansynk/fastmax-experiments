#!/bin/bash
#SBATCH --job-name=PretrainPythia                             # Specify a name for your job
#SBATCH --output=../slurm-logs/pretrain_pythia-%j.log         # Specify the output log file
#SBATCH --error=../slurm-logs/pretrain_pythia-%j.log          # Specify the error log file
#SBATCH --nodes=1                                             # Number of nodes to request
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4                                     # Number of CPU cores per task
#SBATCH --time=24:00:00                                       # Maximum execution time (HH:MM:SS)
#SBATCH --qos=high                                            # Specify the partition (queue) you want to use
#SBATCH --gres=gpu:rtxa5000:1                                # Number of GPUs per node
#SBATCH --mem=32G                                             # Memory per node

# Project setup

# Project setup
proj_root="/fs/nexus-projects/FAST_Attention"

# set up PyEnvironment
source $(conda info --base)/etc/profile.d/conda.sh    # Use if conda is already on your path but you still need to run "conda init <shell_name>"       
conda activate cmsc720
wandb login

# run finetune, saving to gpt_root so nexus backs up fine tuned models but leaving base models on scratch
cd ..
python3 pretrain/openwebtext.py \
    --io.train_data_dir $data_root/data/openwebtext \
    --io.val_data_dir $data_root/data/openwebtext \
    --io.out_dir $proj_root/pythia-14m_openwebtext/out \
    --model_name "pythia-14m"
