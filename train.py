import sys
import os
import models as M
import dataloading as D
import torch
from torch import nn
from torch import optim
import torchvision
import torchvision.transforms as transforms

EPOCHS = 51
BATCH_SIZE = 200
LEARNING_RATE = 1e-3


if torch.cuda.is_available():
    dev = "cuda"
else: 
    dev = "cpu"
print("\n" + dev + "\n")
device = torch.device(dev)



class Logger():
    def __init__(self, nn, data):
        self.nn = nn
        self.data = data
        os.makedirs("./results/" + nn + "/" + data + "/", exist_ok=True)
        self.trf = open("./results/" + nn + "/" + data + "/" + "trainData.csv", "w+")
        self.trf.write("ACC,LOSS\n")
        self.tef = open("./results/" + nn + "/" + data + "/" + "testData.csv", "w+")
        self.tef.write("ACC,LOSS\n")
    
    def writeTr(self, str):
        self.trf.write(str)

    def writeTe(self, str):
        self.tef.write(str)

    def __del__(self):
        self.trf.close()
        self.tef.close()




def main():
    if(len(sys.argv) != 3):
        print("invalid parameters.")
        return
    for arg in sys.argv:
        if arg == "-h" or arg == "--help":
            print("python3 train.py [NN] [data]")
            break
    
    

    transform = None
    trainset = None
    trainloader = None
    testset = None
    testloader = None

    imageDimensions = None
    imageChannels = None

    if(sys.argv[2] == "CIFAR"):
        imageDimensions = 32
        imageChannels = 3
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

    elif(sys.argv[2] == "MNIST"):
        imageDimensions = 28
        imageChannels = 1
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
    else:
        print("No dataset specified: Aborting")
        return

    model = None

    if(sys.argv[1] == "LeNet"):
        model = M.LeNet(datasize=imageDimensions * imageDimensions, inputChannels=imageChannels)
    elif(sys.argv[1] == "VGG16"):
        model = M.VGG16(datasize=imageDimensions * imageDimensions, inputChannels=imageChannels)
    elif(sys.argv[1] == "ResNet18"):
        model = M.ResNet18(datasize=imageDimensions * imageDimensions, inputChannels=imageChannels)
    else:
        print("No neural net specified: Aborting")
        return

    model.to(device)
    print(model)

    logger = Logger(sys.argv[1], sys.argv[2])

    train_model(model, trainloader, testloader, logger)
    model.save(sys.argv[2] + ".pt")


    


def train_model(model, trainset, testset, logger):
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params = model.parameters(), lr = LEARNING_RATE)


    for e in range(EPOCHS):
        # Train set


	if e % 5 == 0:
	    accuracies = []
	    losses = []
	    print(len(trainset) * BATCH_SIZE)
	    for i, data in enumerate(trainset, 0):
		inputs, labels = data
		inputs = inputs.to(device)
		labels = labels.to(device)
		
		logits = model.forward(inputs)
		loss = loss_function(logits, labels)

		y_pred = torch.argmax(logits, dim = 1)
		accuracies.append((y_pred == labels).float().mean().item())
		losses.append(loss.item())
		
	    logger.writeTr(str(sum(accuracies) / len(accuracies)) + "," + str(sum(losses) / len(losses)) + "\n")

	    accuracies = []
	    losses = []
	    for i, data in enumerate(testset, 0):
		inputs, labels = data
		inputs = inputs.to(device)
		labels = labels.to(device)
		
		logits = model.forward(inputs)
		loss = loss_function(logits, labels)

		y_pred = torch.argmax(logits, dim = 1)
		accuracies.append((y_pred == labels).float().mean().item())
		losses.append(loss.item())
		
	    logger.writeTe(str(sum(accuracies) / len(accuracies)) + "," + str(sum(losses) / len(losses)) + "\n")
    
	    
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
	    if i % 50 == 49:    # print every 2000 mini-batches
		print(f'[{e + 1}, {i + 1:5d}] loss: {running_loss / 50:.3f}')
		running_loss = 0.0


if(__name__ == "__main__"):
    main()
