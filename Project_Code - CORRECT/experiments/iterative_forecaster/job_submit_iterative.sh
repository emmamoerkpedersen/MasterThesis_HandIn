#!/bin/sh
### General options
### specify queue
#BSUB -q hpc 
## Set job name
#BSUB -J "EPQQ_job"
### Ask for number of cores (default 1)
#BSUB -n 1
### Specify that we want the job to get killed if it exceeds 3 GB memory per core/slot
#BSUB -M 20GB
### Set walltime limit: hh:mm
#BSUB -W 20:00
### user email adress
#BSUB s194463@student.dtu.dk
### Send notification at completion
#BSUB -N
### specify the output and error file. %J is the job ID
#BSUB -o "Output_iterative_CPFS+TF%J.out"
#BSUB -e "Error_iterative_CPFS+TF.%J.err"

pwd
module load python3/3.12.9

python3 "run_alternating_model.py"

