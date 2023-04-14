import pandas as pd
import matplotlib.pyplot as plt
import sys
import os



headers = ["ACC", "LOSS"]

test = pd.read_csv("./results/" + sys.argv[1] + "/" + sys.argv[2] + "/testData.csv")
teAcc = test.ACC.values
teLoss = test.LOSS.values

train = pd.read_csv("./results/" + sys.argv[1] + "/" + sys.argv[2] + "/trainData.csv")
trAcc = train.ACC.values
trLoss = train.LOSS.values

count = list(range(0, 51, 5))



fig, axs = plt.subplots(1, 2)
fig.suptitle(sys.argv[1] + " " + sys.argv[2])
fig.set_figwidth(10)


axs[0].plot(count, trAcc, label="Training")
axs[0].plot(count, teAcc, label="Testing")
axs[0].set_xlabel("Epochs")
axs[0].set_xlim([-2, 52])
axs[0].set_xticks([x for x in range(0, 51, 10)])
axs[0].set_ylim([0,1])
axs[0].legend()

axs[1].plot(count, trLoss, label="Training")
axs[1].plot(count, teLoss, label="Testing")
axs[1].set_xlabel("Epochs")
axs[1].set_xlim([-2, 52])
axs[1].set_xticks([x for x in range(0, 51, 10)])
axs[1].set_ylim([0, 5])
axs[1].legend()

axs[0].set_title("Accuracy")
axs[1].set_title("Loss")

#plt.show()
os.makedirs("./graphs/" + sys.argv[1] + "/", exist_ok=True)
plt.savefig("./graphs/" + sys.argv[1] + "/" + sys.argv[2] + "Graph.png")
