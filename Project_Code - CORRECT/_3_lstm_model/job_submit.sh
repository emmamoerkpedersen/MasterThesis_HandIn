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
#BSUB -W 01:00
### user email adress
#BSUB s194463@student.dtu.dk
### Send notification at completion
#BSUB -N
### specify the output and error file. %J is the job ID
#BSUB -o "Output_LSTM.%J.out"
#BSUB -e "Error_LSTM.%J.err"

module load python3/3.9.21
module load cuda/11.8
module load cudnn/8.6.0
module load numpy/1.24.3
module load pandas/2.1.1
module load matplotlib/3.7.2


python3 "lstm_forecaster.py"

