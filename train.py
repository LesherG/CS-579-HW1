import sys
import os
from time import time
from test import testModel
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



def trainModel(model, trainset, testset, opti, lera, epochs, logger=None):
    loss_function = nn.CrossEntropyLoss()
    if(opti == "Adam"):
        optimizer = optim.Adam(params = model.parameters(), lr = lera)
    elif(opti == "SGD"):
        optimizer = optim.SGD(params=model.parameters(), lr=lera, momentum=0.9)

    startTime = time()

    for e in range(epochs):
        if(e % 5 == 0 or e == epochs - 1):
            testModel(e, model, trainset, testset, loss_function, logger)


        running_loss = 0.0
        for i, data in enumerate(trainset, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model.forward(inputs)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 50 == 49:
                timeTo = time() - startTime
                print(f'{timeTo // 3600:02.0f}:{(timeTo % 3600) // 60:02.0f}:{timeTo % 60:02.0f} : [{e + 1}, {i + 1:5d}] loss: {running_loss / 50:.3f}')
                running_loss = 0.0



