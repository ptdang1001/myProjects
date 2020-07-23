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
    def __init__(self, xn, yn):
        super(AutoEncoder, self).__init__()
        self.inSize = xn * yn
        self.encoder = nn.Sequential(
            nn.Linear(self.inSize, int(self.inSize / 2)),
            nn.Tanh(),
            nn.Linear(int(self.inSize / 2), int(self.inSize / 4)),
            nn.Tanh(),
            nn.Linear(int(self.inSize / 4), int(self.inSize / 8)),
        )
        self.decoder = nn.Sequential(
            nn.Linear(int(self.inSize / 8), int(self.inSize / 4)),
            nn.Tanh(),
            nn.Linear(int(self.inSize / 4), int(self.inSize / 2)),
            nn.Tanh(),
            nn.Linear(int(self.inSize / 2), self.inSize),
            nn.Sigmoid(),  # compress to a range (0, 1)
        )
        self.classfier = nn.Sequential(
            nn.Linear(int(self.inSize / 8), int(self.inSize / 4)),
            nn.Tanh(),
            nn.Linear(int(self.inSize / 4), int(self.inSize / 2)),
            nn.Tanh(),
            nn.Linear(int(self.inSize / 2), 16),
            nn.Sigmoid(),  # compress to a range (0, 1)
        )

    # 无监督前向传播过程
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        label = self.classfier(encoded)
        return (encoded, decoded, label)


# Autoencoder with cnn
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


class AutoEncoder_CNN(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        latent = self.encoder(x)
        x_recon = self.decoder(latent)
        return x_recon


class VAE_old(nn.Module):
    def __init__(self, xn, yn):
        super(VAE, self).__init__()
        # encoder layers
        self.inSize = xn * yn
        self.fc1 = nn.Linear(self.inSize, int(self.inSize / 2))
        self.fc2 = nn.Linear(int(self.inSize / 2), int(self.inSize / 4))
        self.fc3 = nn.Linear(int(self.inSize / 4), int(self.inSize / 8))
        self.fc4 = nn.Linear(int(self.inSize / 8), int(self.inSize / 16))
        self.fc51 = nn.Linear(int(self.inSize / 16), 2)
        self.fc52 = nn.Linear(int(self.inSize / 16), 2)
        # decoder layers
        self.fc6 = nn.Linear(2, int(self.inSize / 16))
        self.fc7 = nn.Linear(int(self.inSize / 16), int(self.inSize / 8))
        self.fc8 = nn.Linear(int(self.inSize / 8), int(self.inSize / 4))
        self.fc9 = nn.Linear(int(self.inSize / 4), int(self.inSize / 2))
        self.fc101 = nn.Linear(int(self.inSize / 2), xn * yn)

    def encode(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return (self.fc51(x), self.fc52(x))

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        z = F.relu(self.fc6(z))
        z = F.relu(self.fc7(z))
        z = F.relu(self.fc8(z))
        z = F.relu(self.fc9(z))
        return (torch.sigmoid(self.fc101(z)))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        label = torch.softmax(z, dim = 1)
        return (z, self.decode(z), label)

class VAE(nn.Module):
    def __init__(self, xn, yn):
        super(VAE, self).__init__()
        # encoder layers
        self.inSize = xn * yn
        self.fc1 = nn.Linear(self.inSize, int(self.inSize / 2))
        self.fc2 = nn.Linear(int(self.inSize / 2), int(self.inSize / 4))
        self.fc31 = nn.Linear(int(self.inSize / 4), 4)
        self.fc32 = nn.Linear(int(self.inSize / 4), 4)
        # decoder layers
        self.fc4 = nn.Linear(4, int(self.inSize / 4))
        self.fc5 = nn.Linear(int(self.inSize / 4), int(self.inSize / 2))
        self.fc61 = nn.Linear(int(self.inSize / 2), xn * yn)

    def encode(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return (self.fc31(x), self.fc32(x))

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        z = F.relu(self.fc4(z))
        z = F.relu(self.fc5(z))
        return (torch.sigmoid(self.fc61(z)))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        label = torch.softmax(z, dim = 1)
        return (z, self.decode(z), label)

# Fully Connected Network
class FCN(nn.Module):
    def __init__(self, rowNum, colNum):
        super(FCN, self).__init__()
        # self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(rowNum * colNum, 12)
        self.fc2 = nn.Linear(12, 6)
        self.fc3 = nn.Linear(6, 3)

    def forward(self, x):
        x = x.view(-1, x.size()[1] * x.size()[2])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return (x)


# GAN
# Generator
class Generator(nn.Module):
    def __init__(self, xn, yn):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, xn * yn),
        )

    def forward(self, z):
        x = self.model(z)
        return (x)


# Didcriminator
class Discriminator(nn.Module):
    def __init__(self, xn, yn):
        super(Discriminator, self).__init__()
        self.inSize = xn * yn
        self.fc1 = nn.Linear(self.inSize, int(self.inSize / 2))
        self.fc2 = nn.Linear(int(self.inSize / 2), int(self.inSize / 4))

        # discriminator layer
        self.fc31 = nn.Linear(int(self.inSize / 4), 1)

        # classifier layer
        self.fc32 = nn.Linear(int(self.inSize / 4), int(self.inSize / 8))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        score = torch.sigmoid(F.relu(self.fc31(x)))
        label = torch.sigmoid(F.relu(self.fc32(x)))
        return (score, label)
