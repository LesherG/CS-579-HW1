import sys
import os
from LeNet import LeNet
from ResNet import ResNet
import torch
from train import trainModel
import test
from torch import nn
from torch import optim
import torchvision
import torchvision.transforms as transforms

if torch.cuda.is_available():
    dev = "cuda"
else: 
    dev = "cpu"
device = torch.device(dev)


#
# Gavin Lesher
# 05/25/2023
#
#

def main():
    torch.manual_seed(28)
    preamble = "base"

    model = None
    dataset = None

    transform = [transforms.ToTensor()]
    imageDimensions = None
    imageChannels = None


    epochs = 50
    batchSize = 200
    learningRate = 1e-3
    opti = "Adam"

    # Parse arguements
    for i in range(len(sys.argv)):
        if(sys.argv[i][:2] == "--"):
            arg = sys.argv[i][2:]
            param = sys.argv[i + 1]
            if(arg == "model"):
                model = param
            elif(arg == "dataset"):
                dataset = param
                if(param == "CIFAR"):
                    imageDimensions = 32*32
                    imageChannels = 3
                    transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
                elif(param == "MNIST"):
                    imageDimensions = 28*28
                    imageChannels = 1
            elif(arg == "lr"):
                learningRate = float(param)
            elif(arg == "opti"):
                opti = param
            elif(arg == "batch"):
                batchSize = int(param)
            elif(arg == "epochs"):
                epochs = int(param)
            elif(arg == "extra"):
                if(param == "rotation"):
                    transform.append(transforms.RandomRotation(degrees=(0, 359)))
                elif(param == "flip"):
                    transform.append(transforms.RandomHorizontalFlip())
            elif(arg == "preamble"):
                preamble = param

    # Create logger before we replace the model string
    logger = test.Logger(model, dataset, preamble=preamble)

    # Create model
    if(model == "LeNet"):
        model = LeNet(datasize=imageDimensions, inputChannels=imageChannels)
    elif(model == "ResNet"):
        model = ResNet(datasize=imageDimensions, inputChannels=imageChannels)
    model.to(device)
            
    # Create datasets with transforms from settings
    transform = transforms.Compose(transform)
    if(dataset == "MNIST"):
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=1)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=1)
    elif(dataset == "CIFAR"):
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=1)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=1)

    # Train model
    trainModel(model, trainloader, testloader, opti, learningRate, epochs, logger)

    # -------------------------------------
    # | any post-training processing here |
    # -------------------------------------




    # -------------------------------------
    # Any cleanup here
    model.save(dataset + "_" + preamble + ".pt")

if(__name__ == "__main__"):
    main()