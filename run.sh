#!/bin/bash
#SBATCH -J Assignment_1	
#SBATCH -A cs479-579	        
#SBATCH -p class      
#SBATCH --gres=gpu:1            
#SBATCH -o A1console.out				
#SBATCH -e A1error.err				
#SBATCH --mem=10G

module load python3/3.8

#python3 train.py Rotation1 MNIST
#python3 train.py Rotation2 CIFAR
#python3 train.py Flip1 MNIST
#python3 train.py Flip2 CIFAR
python3 train.py Opti1 MNIST
python3 train.py Opti2 CIFAR
#python3 train.py Batch1 MNIST
#python3 train.py Batch2 CIFAR
#python3 train.py Rate1 MNIST
python3 train.py Rate2 CIFAR
