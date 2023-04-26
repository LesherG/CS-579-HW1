import pandas as pd
import matplotlib.pyplot as plt
import sys
import os




def plot(directory, title = "test"):
    train = pd.read_csv(directory + "trainData.csv")
    trAcc = train.ACC.values
    trLoss = train.LOSS.values

    test = pd.read_csv(directory + "testData.csv")
    teAcc = test.ACC.values
    teLoss = test.LOSS.values

    epochs = train.EPOCH.values



    fig, axs = plt.subplots(1, 2)
    fig.suptitle(title)
    fig.set_figwidth(10)


    axs[0].plot(epochs, trAcc, label="Training")
    axs[0].plot(epochs, teAcc, label="Testing")
    axs[0].set_xlabel("Epochs")
    axs[0].set_xlim([-2, max(epochs) + 2])
    axs[0].set_xticks([x for x in range(0, max(epochs), 10)])
    axs[0].set_ylim([0,1])
    axs[0].legend()

    axs[1].plot(epochs, trLoss, label="Training")
    axs[1].plot(epochs, teLoss, label="Testing")
    axs[1].set_xlabel("Epochs")
    axs[1].set_xlim([-2, max(epochs) + 2])
    axs[1].set_xticks([x for x in range(0, max(epochs), 10)])
    axs[1].set_ylim([0, 5])
    axs[1].legend()

    axs[0].set_title("Accuracy")
    axs[1].set_title("Loss")

    #plt.show()
    os.makedirs("./graphs/" + directory[10:], exist_ok=True)
    plt.savefig("./graphs/" + directory[10:] + "Graph.png")
    print(f"\t-- Created file at: " + "./graphs/" + directory[10:] + "Graph.png")