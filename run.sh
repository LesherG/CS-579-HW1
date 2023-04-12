#!/bin/bash
#SBATCH -J Assignment 1 test						  # name of job
#SBATCH -A cs479-579	  # name of my sponsored account, e.g. class or research group, NOT ONID!
#SBATCH -o A1t.out				  # name of output file for this submission script
#SBATCH -e A1e.err				  # name of error file for this submission script

python3.8 train.py LeNet
python3.8 train.py VGG16
python3.8 train.py ResNet18