import os

import torch
from torch import nn

if torch.cuda.is_available():
    dev = "cuda"
else: 
    dev = "cpu"
cuda = torch.device(dev)

class VGG16(nn.Module):
    def __init__(self, numClasses = 10, datasize = 28*28, inputChannels = 1):
        super().__init__()

        self.features = nn.Sequential(
            # Conv Set 1
            nn.Conv2d(inputChannels, 64, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Conv Set 2
            nn.Conv2d(64, 128, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Conv Set 3
            nn.Conv2d(128, 256, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Conv Set 4
            nn.Conv2d(256, 512, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Conv Set 5
            nn.Conv2d(512, 512, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            #FC layers
            nn.Linear(512, 4096), #In an actual implementation this should not be 512 * 1. 1 should be variable depending on the input size
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, numClasses)
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
        os.makedirs("./models/VGG16/", exist_ok=True)
        torch.save(self, "./models/VGG16/" + file)