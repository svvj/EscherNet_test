#!/bin/bash
#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --nodes=1
#SBATCH --job-name="bg_removal"
#SBATCH --account="punim2482"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mail-user=seungwon.jeong@student.unimelb.edu.au
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --time=0-00:40:00
#SBATCH --array=49-100%10  # Runs 1 to 100 jobs in batches of 10

# Check that the script is launched with sbatch
if [ -z "$SLURM_JOB_ID" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Navigate to working directory
cd /data/gpfs/projects/punim2482/workspace/EscherNet_test

# Load necessary modules
module load CUDA/11.8.0
module load Anaconda3/2022.10
eval "$(conda shell.bash hook)"
conda activate eschernet  # Change this if a different env is needed

# Set the image index based on SLURM job array ID
IMAGE_IDX=$SLURM_ARRAY_TASK_ID

# Define input and output directories
INPUT_DIR="/data/gpfs/projects/punim2482/workspace/EscherNet_test/logs_6DoF/NeRF_ours/N${IMAGE_IDX}M100"
# INPUT_DIR="/data/gpfs/projects/punim2482/workspace/EscherNet_test/sampled/N1M100"
OUTPUT_DIR="/data/gpfs/projects/punim2482/workspace/EscherNet_test/removal/NeRF_ours/upsampled"
# OUTPUT_DIR="/data/gpfs/projects/punim2482/workspace/EscherNet_test/sampled/N1M100/removed"

# Run the background removal script
python remove_background.py --image_path "$INPUT_DIR" --save_path "$OUTPUT_DIR/N${IMAGE_IDX}M100"

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s
