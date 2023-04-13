#!/bin/bash
#SBATCH -J Assignment_1	
#SBATCH -A cs479-579	        
#SBATCH -p class      
#SBATCH --gres=gpu:1            
#SBATCH -o A1console.out				
#SBATCH -e A1error.err				

module load python3/3.8

for i in "LeNet" "VGG16" "ResNet18"
do
    for j in "MINST" "CIFAR"
    do
        python3 train.py "$i" "$j"
    done
done

