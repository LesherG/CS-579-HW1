#!/bin/bash
#SBATCH -J Assignment_1	
#SBATCH -A cs479-579	        
#SBATCH -p class      
#SBATCH --gres=gpu:1            
#SBATCH -o A1console.out				
#SBATCH -e A1error.err				
#SBATCH --mem=10G

module load python3/3.8

python3 train.py LeNet MNIST
python3 train.py LeNet CIFAR
python3 train.py VGG16 MNIST
python3 train.py VGG16 CIFAR
python3 train.py ResNet18 MNIST
python3 train.py ResNet18 CIFAR
