#!/bin/bash
#SBATCH -J start-time-sweep         # job name
#SBATCH -o launcher.o%j             # output and error file name (%j expands to jobID)
#SBATCH -N 1                        # number of nodes requested
#SBATCH -n 5                        # total number of tasks to run in parallel
#SBATCH -p development              # queue (partition) 
#SBATCH -t 00:30:00                 # run time (hh:mm:ss) 
#SBATCH -A GravSearches             # Allocation name to charge job against

module load launcher

export OMP_NUM_THREADS=4

export LAUNCHER_WORKDIR=${pwd}
export LAUNCHER_JOB_FILE=taskfile 

${LAUNCHER_DIR}/paramrun