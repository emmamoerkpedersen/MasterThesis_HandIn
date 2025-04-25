#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J EPQQ_GPU_JOB

### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 15:00
# request 10GB of system-memory
#BSUB -R "rusage[mem=30GB]"
#BSUB -R "select[gpu32gb]"
### -- user email address --
#BSUB -u s194463@student.dtu.dk
### -- send notification at completion --
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu-%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --
# Load the cuda module



pwd
module load python3/3.12.9

python3 "RNN_Model.py"