import sys
import os
import models as M
import plotting
import torch
from torch import nn
from torch import optim
import torchvision
import torchvision.transforms as transforms

if torch.cuda.is_available():
    dev = "cuda"
else: 
    dev = "cpu"
device = torch.device(dev)


class Logger():
    def __init__(self, nn, data, preamble = ""):
        self.nn = nn
        self.data = data

        self.directory = "./results/" + preamble + "/" + nn + "/" + data + "/"
        os.makedirs(self.directory, exist_ok=True)
        
        self.trf = open(self.directory + "trainData.csv", "w+")
        self.trf.write("EPOCH,ACC,LOSS\n")

        self.tef = open(self.directory + "testData.csv", "w+")
        self.tef.write("EPOCH,ACC,LOSS\n")
    
    def writeTr(self, str):
        self.trf.write(str)

    def writeTe(self, str):
        self.tef.write(str)

    def __del__(self):
        self.trf.close()
        self.tef.close()

    def makeGraph(self):   
        self.trf.close()
        self.tef.close()

        plotting.plot(self.directory)

        self.trf = open(self.directory + "trainData.csv", "a+")
        self.tef = open(self.directory + "testData.csv", "a+")


def testModel(epoch, model, trainData, testData, lossFunction, logger):
    accuracies = []
    losses = []
    for i, data in enumerate(trainData, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        logits = model.forward(inputs)
        loss = lossFunction(logits, labels)

        y_pred = torch.argmax(logits, dim = 1)
        accuracies.append((y_pred == labels).float().mean().item())
        losses.append(loss.item())

    logger.writeTr(str(epoch) + "," + str(sum(accuracies)/len(accuracies)) + "," + str(sum(losses)/len(losses)) + "\n")


    accuracies = []
    losses = []
    for i, data in enumerate(testData, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        logits = model.forward(inputs)
        loss = lossFunction(logits, labels)

        y_pred = torch.argmax(logits, dim = 1)
        accuracies.append((y_pred == labels).float().mean().item())
        losses.append(loss.item())

    logger.writeTe(str(epoch) + "," + str(sum(accuracies)/len(accuracies)) + "," + str(sum(losses)/len(losses)) + "\n")
    logger.makeGraph()
