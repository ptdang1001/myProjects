import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class VAE(nn.Module):
    def __init__(self, xn, yn):
        super(VAE, self).__init__()
        # encoder layers
        self.inSize = xn * yn
        self.fc1 = nn.Linear(self.inSize, int(self.inSize/2))
        self.fc2 = nn.Linear(int(self.inSize/2), int(self.inSize/4))
        self.fc3 = nn.Linear(int(self.inSize/4), int(self.inSize/8))
        self.fc4 = nn.Linear(int(self.inSize/8), int(self.inSize/16))
        self.fc5 = nn.Linear(int(self.inSize/16), int(self.inSize/32))
        self.fc6 = nn.Linear(int(self.inSize/32), int(self.inSize/64))
        self.fc7 = nn.Linear(int(self.inSize/64), int(self.inSize/128))
        self.fc8 = nn.Linear(int(self.inSize/128), int(self.inSize/256))
        self.fc9 = nn.Linear(int(self.inSize/256), int(self.inSize/512))
        self.fc10 = nn.Linear(int(self.inSize/512), int(self.inSize/1024))
        self.fc11 = nn.Linear(int(self.inSize/1024), int(self.inSize/2048))
        self.fc12 = nn.Linear(int(self.inSize/2048), int(self.inSize/4096))
        self.fc131 = nn.Linear(int(self.inSize/4096), 2)
        self.fc132 = nn.Linear(int(self.inSize/4096), 2)
        # classifier layer
        #self.fc133 = nn.Linear(int(self.inSize/4096), 4)

        # decoder layers
        self.fc14 = nn.Linear(2, int(self.inSize/4096))
        self.fc15 = nn.Linear(int(self.inSize/4096), int(self.inSize/2048))
        self.fc16 = nn.Linear(int(self.inSize/2048), int(self.inSize/1024))
        self.fc17 = nn.Linear(int(self.inSize/1024), int(self.inSize/512))
        self.fc18 = nn.Linear(int(self.inSize/512), int(self.inSize/256))
        self.fc19 = nn.Linear(int(self.inSize/256), int(self.inSize/128))
        self.fc20 = nn.Linear(int(self.inSize/128), int(self.inSize/64))
        self.fc21 = nn.Linear(int(self.inSize/64), int(self.inSize/32))
        self.fc22 = nn.Linear(int(self.inSize/32), int(self.inSize/16))
        self.fc23 = nn.Linear(int(self.inSize/16), int(self.inSize/8))
        self.fc24 = nn.Linear(int(self.inSize/8), int(self.inSize/4))
        self.fc25 = nn.Linear(int(self.inSize/4), int(self.inSize/2))
        self.fc26 = nn.Linear(int(self.inSize/2), self.inSize)

    def encode(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        x = F.relu(self.fc10(x))
        x = F.relu(self.fc11(x))
        x = F.relu(self.fc12(x))
        return (self.fc131(x), self.fc132(x))

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        z = F.relu(self.fc14(z))
        z = F.relu(self.fc15(z))
        z = F.relu(self.fc16(z))
        z = F.relu(self.fc17(z))
        z = F.relu(self.fc18(z))
        z = F.relu(self.fc19(z))
        z = F.relu(self.fc20(z))
        z = F.relu(self.fc21(z))
        z = F.relu(self.fc22(z))
        z = F.relu(self.fc23(z))
        z = F.relu(self.fc24(z))
        z = F.relu(self.fc25(z))
        return (torch.sigmoid(self.fc26(z)))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        label = torch.softmax(z, dim=1)
        return (z, self.decode(z), label)


class VAE_old(nn.Module):
    def __init__(self, xn, yn):
        super(VAE_old, self).__init__()
        # encoder layers
        self.inSize = xn * yn
        self.fc1 = nn.Linear(self.inSize, int(self.inSize / 2))
        self.fc2 = nn.Linear(int(self.inSize / 2), int(self.inSize / 4))
        self.fc31 = nn.Linear(int(self.inSize / 4), 2)
        self.fc32 = nn.Linear(int(self.inSize / 4), 2)
        # decoder layers
        self.fc4 = nn.Linear(2, int(self.inSize / 4))
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
