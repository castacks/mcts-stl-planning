#!/bin/sh

. /jet/home/jaypat/.bashrc

set -x

conda activate pytorch

python play.py --model_weights goalGAIL1b_60.pt > log.txt

## to debug
##interact -p GPU-shared -t 4:00:00 -n 5 --gres=gpu:1

##to submit 
##  sbatch -p GPU-shared -t 24:00:00 -n 5 --gres=gpu:1 sbatch_run.sh 
