# -*- : coding: utf-8 -*
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# converlutional neural network
from torch.autograd import Variable


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
        # print(x.size())
        # sys.exit()
        x = x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# unsupervised deep learning
class AutoEncoder(nn.Module):
    def __init__(self, zn,xn,yn):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(zn * xn * yn, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 16),
            #nn.Tanh(),
            #nn.Linear(12, 16),
        )
        self.decoder = nn.Sequential(
            #nn.Linear(16, 12),
            #nn.Tanh(),
            nn.Linear(16, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, zn * xn * yn),
            nn.Sigmoid(),  # compress to a range (0, 1)
        )
        self.classfier = nn.Sequential(
            #nn.Linear(3, 12),
            #nn.Tanh(),
            nn.Linear(16, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 16),
            nn.Sigmoid(),
        )

    # 无监督前向传播过程
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        label = self.classfier(encoded)
        return (encoded, decoded, label)


'''
# Autoencoder Definition
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        c = capacity
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=c, kernel_size=4, stride=2, padding=1)  # out:c*14*14
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c * 2, kernel_size=4, stride=2, padding=1)  # out:c*2*7*7
        self.fc = nn.Linear(in_features=c * 2 * 7 * 7, out_features=latent_dims)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # flatten batch of multi-channel feature maps to a batch of feature vectors
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        c = capacity
        self.fc = nn.Linear(in_features=latent_dims, out_features=c * 2 * 7 * 7)
        self.conv2 = nn.ConvTranspose2d(in_channels=c * 2, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=c, out_channels=1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), capacity * 2, 7,
                   7)  # unflatten batch of feature vectors to a  batch of multi-channel feature maps
        x = F.relu(self.conv2(x))
        x = torch.tanh(
            self.conv1(x))  # last layer before output is tanh ,since the images are normalized and 0-centered
        return x


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        latent = self.encoder(x)
        x_recon = self.decoder(latent)
        return x_recon
'''


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


# Fully Connected Network
class FCN(nn.Module):
    def __init__(self, rowNum, colNum):
        super(FCN, self).__init__()
        # self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(rowNum * colNum, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 50)

    def forward(self, x):
        x = x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
