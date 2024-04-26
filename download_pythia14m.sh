#!/bin/bash
#SBATCH --job-name=DwnldPythia                      # Specify a name for your job
#SBATCH --output=slurm-logs/dwnld-pythia-%j.log               # Specify the output log file
#SBATCH --error=slurm-logs/dwnld-pythia-%j.log                # Specify the error log file
#SBATCH --nodes=1                                             # Number of nodes to request
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4                                     # Number of CPU cores per task
#SBATCH --time=04:00:00                                       # Maximum execution time (HH:MM:SS)
#SBATCH --qos=default                                         # Specify the partition (queue) you want to use
#SBATCH --gres=gpu:rtxa5000:0                                 # Number of GPUs per node
#SBATCH --mem=32G                                             # Memory per node

# set up PyEnvironment
# source $scratch_root/PyEnvs/Lit-GPT/bin/activate
source $(conda info --base)/etc/profile.d/conda.sh          # Use if conda is already on your path but you still need to run "conda init <shell_name>"       
conda activate cmsc720

# make sure your environment has the huggingface_hub[hr_transfer] (see lit-gpt tutorials for pip command)
# (this is taken care of in requirements-cmsc720.txt)

python scripts/download.py --repo_id EleutherAI/pythia-14m

python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/EleutherAI/pythia-14m