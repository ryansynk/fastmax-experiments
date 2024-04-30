#!/bin/bash
#SBATCH --job-name=PrepareOpenWebText                       # Specify a name for your job
#SBATCH --output=slurm-logs/out-prepOpenWeb-%j.log               # Specify the output log file
#SBATCH --error=slurm-logs/out-prepOpenWeb-%j.log                # Specify the error log file
#SBATCH --nodes=1                                             # Number of nodes to request
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16                                     # Number of CPU cores per task
#SBATCH --time=04:00:00                                       # Maximum execution time (HH:MM:SS)
#SBATCH --qos=high                                         # Specify the partition (queue) you want to use
# #SBATCH --gres=gpu:rtxa5000:0                                 # Number of GPUs per node
#SBATCH --mem=64G                                             # Memory per node


# Project setup
data_root="/fs/nexus-scratch/mhoover4/"
proj_root="/vulcanscratch/mhoover4/code/fastmax-experiments/"

# set up PyEnvironment
source $(conda info --base)/etc/profile.d/conda.sh    # Use if conda is already on your path but you still need to run "conda init <shell_name>"       
conda activate cmsc720


# run script
python3 $proj_root/scripts/prepare_openwebtext.py \
    --destination_path $data_root/data/openwebtext \
    --checkpoint_dir $proj_root/checkpoints/EleutherAI/pythia-14m

