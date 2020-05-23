# -*- : coding: utf-8 -*
import sys
import torch.nn as nn
import torch.nn.functional as F


#converlutional neural network
class CNN(nn.Module):
    def __init__(self, inChannels, kernels, kernelSize, outSize):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(inChannels, kernels, kernelSize)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(kernels, 16, kernelSize)
        self.fc1 = nn.Linear(outSize, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 50)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #print(x.size())
        #sys.exit()
        x = x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# unsupervised deep learning
class AutoEncoder(nn.Module):
    def __init__(self, ):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(3 * 7 * 1000, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 3 * 7 * 1000),
            nn.Sigmoid(),  # compress to a range (0, 1)
        )
        self.classfier = nn.Sequential(
            nn.Linear(3, 128),
            nn.Tanh(),
            nn.Linear(128, 5),
            nn.Sigmoid(),
        )

    # 无监督前向传播过程
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        label = self.classfier(encoded)
        return (encoded, decoded, label)


#Fully Connected Network
class FCN(nn.Module):
    def __init__(self, rowNum,colNum):
        super(FCN, self).__init__()
        #self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(rowNum*colNum, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 50)

    def forward(self, x):
        x = x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
