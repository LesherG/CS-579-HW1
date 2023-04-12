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
        torch.save(self, "./models/LeNet/" + file)

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
        torch.save(self, "./models/VGG16/" + file)



class ResNet18(nn.Module):
    def __init__(self, numClasses = 10, datasize = 28*28, inputChannels = 1):
        super().__init__()
        
        self.in_channels = 64

        self.init = nn.Sequential(
            nn.Conv2d(inputChannels, 64, 7, stride = 2, padding = 3),    
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride  = 2, padding = 1)
        )

        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.close = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512, numClasses)
        )


    def _make_layer(self, out_channels, blocks, stride = 1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, 
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False 
                ),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(
            BasicBlock(
                self.in_channels, out_channels, stride, downsample
            )
        )
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(BasicBlock(
                self.in_channels,
                out_channels
            ))
        return nn.Sequential(*layers)



    def forward(self, x):
        x = self.init(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.close(x)

        return x

    def predict(self, x):
        logits = self.features(x)
        pred = nn.Softmax(logits)
        y_pred = torch.argmax(pred)
        return(pred[y_pred])

    def save(self, file):
        torch.save(self, "./models/ResNet18/" + file)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(BasicBlock, self).__init__()
        # Multiplicative factor for the subsequent conv2d layer's output channels.
        # It is 1 for ResNet18 and ResNet34.
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels, 
            kernel_size=3, 
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return  out