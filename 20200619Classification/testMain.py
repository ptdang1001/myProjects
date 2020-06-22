# -*- coding: utf8 -*

# system lib
import argparse
import sys
import time
import os
from multiprocessing import Pool

# third part lib
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from cvae import cvae 
from sklearn.cluster import KMeans

# my libs
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(".."))

import myModules
import myTrainTest
import myData

# parameters
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1)
if torch.cuda.is_available():
    parser.add_argument("--batch_size", type=int, default=1)
else:
    parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--n_cpu", type=int, default=os.cpu_count())
parser.add_argument("--minusMean", type=int, default=1)
parser.add_argument("--stdBias", type=int, default=0)
parser.add_argument("--numThreshold", type=int, default=30)
parser.add_argument("--xn", type=int, default=300)
parser.add_argument("--crType", type=str, default="norm")
parser.add_argument("--sampleNum", type=int, default=300)
parser.add_argument("--baseTimes", type=int, default=1)
parser.add_argument("--errorStdBias", type=int, default=0 / 10)
runPams = parser.parse_args()


def getFCNPams(rowNum, colNum, device, lr):
    fcn = Predicte.myModules.FCN(rowNum=rowNum, colNum=colNum)
    fcn = fcn.to(device)
    optimizer = torch.optim.Adam(fcn.parameters(), lr=lr)
    lossFunc = nn.CrossEntropyLoss()
    return (fcn, optimizer, lossFunc)


def getCNNPams(zn, xn, yn, device, lr):
    cnnXout = Predicte.myUtils.myData.getCNNOutSize(xn, 3, 2)
    cnnYout = Predicte.myUtils.myData.getCNNOutSize(yn, 3, 2)
    cnn = Predicte.myModules.CNN(inChannels=zn,
                                 kernels=6,
                                 kernelSize=2,
                                 outSize=16 * cnnXout * cnnYout)
    cnn = cnn.to(device)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)
    lossFunc = nn.CrossEntropyLoss()
    return (cnn, optimizer, lossFunc)


# end

def getAEPams(xn, yn, device, lr):
    AE = Predicte.myModules.AutoEncoder(xn, yn)
    AE = AE.to(device)
    optimizer = torch.optim.Adam(AE.parameters(), lr=lr)
    lossFunc = nn.MSELoss()
    return (AE, optimizer, lossFunc)


# end

def getVAEPams(xn, yn, device, lr):
    VAE = myModules.VAE(xn, yn)
    VAE = VAE.to(device)
    optimizer = torch.optim.Adam(VAE.parameters(), lr=lr)
    lossFunc = nn.MSELoss()
    return (VAE, optimizer, lossFunc)


# end

def getGANPams(xn, yn, device, lr):
    # G parts
    G = Predicte.myModules.Generator(xn, yn)
    G = G.to(device)
    G_optimizer = torch.optim.Adam(G.parameters(), lr=lr)

    # D parts
    D = Predicte.myModules.Discriminator(xn, yn)
    D = D.to(device)
    D_optimizer = torch.optim.Adam(D.parameters(), lr=lr)
    lossFunc = nn.BCELoss()
    return (G, G_optimizer, D, D_optimizer, lossFunc)


# end



def main():
    x = pd.read_csv("/N/project/zhangclab/pengtao/myProjectsDataRes/20200619Classification/data/salmonE74cDNA_counts_baseline.csv", index_col=0)
    x = x.T
    x = x + 1
    x = x.apply(np.log2)
    x = x.values 
    #embedder = cvae.CompressionVAE(x.values)
    
    # train
    x = torch.rand(100,4096)
    data = torch.tensor(x)
    data = data.view(data.size()[0],data.size()[1],1)
    # choose cpu or gpu automatically
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # optFeatureMap data
    net, optimizer, lossFunc = getVAEPams(
        data.size()[1],
        data.size()[2],
        device,
        runPams.lr)

    z = myTrainTest.train_test_VAE(
        data, device, lossFunc, runPams, net, optimizer)
    print(z.shape)
    [print(l) for l in z]
    sys.exit()
    kmeans = KMeans(n_clusters=4, random_state=0).fit(z)
    print(KMeans.labels_)
    #embedder.visualize(z, labels=[int(label) for label in kmeans.labels_], filename="t.png")
    return ()


# run main
if __name__ == "__main__":
    main()
