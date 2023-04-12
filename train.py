import sys
import models as M
import dataloading as D
import torch
from torch import nn
from torch import optim
import torchvision
import torchvision.transforms as transforms

EPOCHS = 50
BATCH_SIZE = 200
LEARNING_RATE = 1e-3


if torch.cuda.is_available():
    dev = "cuda"
else: 
    dev = "cpu"
device = torch.device(dev)



def main():
    
    if(sys.argv[1] == "LeNet"):
        model = M.LeNet(datasize=32*32, inputChannels=3)
    elif(sys.argv[1] == "VGG16"):
        model = M.VGG16(datasize=32*32, inputChannels=3)
    elif(sys.argv[1] == "ResNet18"):
        model = M.ResNet18(datasize=32*32, inputChannels=3)

    model.to(device)
    
    print(model)



    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    train_model(model, trainloader)
    model.save("test.pt")


    


def train_model(model, dataset):
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params = model.parameters(), lr = LEARNING_RATE)


    for e in range(EPOCHS):
        # Train set

        running_loss = 0.0
        for i, data in enumerate(dataset, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model.forward(inputs)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 50 == 49:    # print every 2000 mini-batches
                print(f'[{e + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

if(__name__ == "__main__"):
    main()