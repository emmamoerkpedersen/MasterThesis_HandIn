#!\bin\sh
### General options
### specify queue
#BSUB -q hpc 
## Set job name
#BSUB -J "EPQQ_job"
### Ask for number of cores (default 1)
#BSUB -n 1
### Specify that we want the job to get killed if it exceeds 3 GB memory per core/slot
#BSUB -M 5GB
### Set walltime limit: hh:mm
#BSUB -W 00:15
### user email adress
#BSUB s194463@student.dtu.dk
### Send notification at completion
#BSUB -N
### specify the output and error file. %J is the job ID
#BSUB -o "Output.%J.out"
#BSUB -e "Error.%J.err"

module load python3/3.10.11
module load cuda/11.8.0
module load cudnn/8.6.0
module load numpy/1.26.0
module load pandas/2.2.3
module load matplotlib/3.10.1 


python3 "main.py"

