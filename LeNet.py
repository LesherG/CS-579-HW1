import os
import math

import torch
from torch import nn

if torch.cuda.is_available():
    dev = "cuda"
else: 
    dev = "cpu"
cuda = torch.device(dev)


class LeNet(nn.Module):
    def __init__(self, numClasses = 10, datasize = 28*28, inputChannels = 1):
        super().__init__()
        
        convOut = int((math.sqrt(datasize) / 2) / 2 - 2)
        self.features = nn.Sequential(
            # C1
            nn.Conv2d(inputChannels, 6, 5, padding = 2),
            nn.Sigmoid(),
            nn.AvgPool2d(2),
            
            # C3
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.AvgPool2d(2),
            nn.Flatten(),

            # Dense 
            nn.Linear(16 * convOut * convOut, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, numClasses)
        )

    def forward(self, x):
        logits = self.features(x)
        return logits

    def predict(self, x):
        logits = self.features(x)
        pred = nn.Softmax(logits)
        y_pred = torch.argmax(pred)
        return(pred[y_pred])

    def save(self, file):
        os.makedirs("./models/LeNet/", exist_ok=True)
        torch.save(self, "./models/LeNet/" + file)