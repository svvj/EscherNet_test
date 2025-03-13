#!/bin/bash
#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --nodes=1
#SBATCH --job-name="eschernet"
#SBATCH --account="punim2482"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mail-user=seungwon.jeong@student.unimelb.edu.au
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --time=0-00:30:00
#SBATCH --array=1-100%10  # Runs jobs with T_in values from 11 to 99, 10 at a time

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
conda activate eschernet

# Set T_in based on job array index
T_in=$SLURM_ARRAY_TASK_ID

# Run evaluation script
bash eval_eschernet.sh $T_in

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s
