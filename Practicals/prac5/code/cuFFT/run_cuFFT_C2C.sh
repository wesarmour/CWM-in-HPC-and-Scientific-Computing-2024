#!/bin/bash
# set the number of nodes and processes per node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=short

# set max wallclock time
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1

# set name of job
#SBATCH --job-name=cuFFT-C2C

# Use our reservation
#SBATCH --reservation=training


module purge
module load CUDA/11.4.1-GCC-10.3.0 

./cuFFT_C2C
