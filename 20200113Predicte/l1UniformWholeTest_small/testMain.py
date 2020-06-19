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

# my libs
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(".."))

import Predicte.myModules
import Predicte.myUtils.myTrainTest
import Predicte.myUtils.myData

# parameters
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1)
if torch.cuda.is_available():
    parser.add_argument("--batch_size", type=int, default=5)
else:
    parser.add_argument("--batch_size", type=int, default=5)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--n_cpu", type=int, default=os.cpu_count())
parser.add_argument("--minusMean", type=int, default=1)
parser.add_argument("--stdBias", type=int, default=0)
parser.add_argument("--numThreshold", type=int, default=30)
parser.add_argument("--xn", type=int, default=300)
parser.add_argument("--crType", type=str, default="uniform")
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
    VAE = Predicte.myModules.VAE(xn, yn)
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
    # parameters
    mean = 0
    minusMean = 1
    blockNum = 1
    replace = 1
    zn = 1
    yn = runPams.xn
    totalRow = 1000
    totalCol = totalRow
    overlap = 0
    probType = "l1"
    # partitions

    labels_mateDatas = Predicte.myUtils.myData.getL1SpeBaseData(runPams.crType, minusMean, runPams.errorStdBias,
                                                                blockNum, runPams.baseTimes, zn, runPams.xn, yn,
                                                                totalRow, totalCol, overlap,
                                                                replace)
    mateDatas = list(labels_mateDatas[-1][-1])

    mateData = pd.DataFrame(mateDatas[0])
    # mateData = pd.DataFrame(np.random.rand(1000, 1000))

    parts = Predicte.myUtils.myData.mateData2Parts(mateData.copy())
    res = list()
    with Pool(os.cpu_count()) as p:
        res = p.map(Predicte.myUtils.myData.getSamplesRowColStd, parts)
    # splite samples, samplesArr, rowStdArr, colStdArr from res
    samples = list()
    samplesArr = list()
    rowStdArr = list()
    # colStdArr = list()
    for r in res:
        samples.append(r[0])
        samplesArr.append(r[1])
        rowStdArr.append(r[2])
        # colStdArr.append(r[3])
    samplesArr = np.stack(samplesArr)
    #rowStdArr = np.stack(rowStdArr)
    # colStdArr = np.stack(colStdArr)

    # get bases matrix
    basesMtrx, baseTypeNumAfterKmean, baseIdAfterKMeans = Predicte.myUtils.myData.getBasesMtrxAfterKmean()
    # get row and col feature map: (7*7) * (7*1652)
    rowFeatureMap = np.matmul(samplesArr, (basesMtrx.iloc[:, 0:7].values.T))
    # colFeatureMap = np.matmul(samplesArr.transpose((0, 1, 3, 2)), (basesMtrx.iloc[:, 0:7].values.T))

    # normalize row and col by std from original 50*50's row and col std
    # rowFeatureMap = np.true_divide(rowFeatureMap, rowStdArr)
    rowFeatureMap = -np.sort(-rowFeatureMap, axis=2)
    tmp = rowFeatureMap

    # normalize col by std from original 50*50' col
    # colFeatureMap = np.true_divide(colFeatureMap, colStdArr)
    # colFeatureMap = -np.sort(-colFeatureMap, axis=2)

    # resort them by their mean
    rowFeatureMap = Predicte.myUtils.myData.getResortMeanFeatureMap(rowFeatureMap[:,:,0:7,:])
    # colFeatureMap = Predicte.myUtils.myData.getResortMeanFeatureMap(colFeatureMap)

    # row and col max pooling 7*1652 -> 7*16
    rowFeatureMap = Predicte.myUtils.myData.myMaxPooling(rowFeatureMap, baseTypeNumAfterKmean)
    # colFeatureMap = Predicte.myUtils.myData.myMaxPooling(colFeatureMap, baseTypeNumAfterKmean)
    # featureMap = np.stack((rowFeatureMap, colFeatureMap), axis=2)

    # sort the rows
    rowFeatureMap = -np.sort(-rowFeatureMap, axis=3)[:,:,:,:]

    rowFeatureMap = torch.tensor(rowFeatureMap)
    rowFeatureMap = rowFeatureMap.view(rowFeatureMap.size()[0] * rowFeatureMap.size()[1], rowFeatureMap.size()[2],
                                       rowFeatureMap.size()[3])
    # choose cpu or gpu automatically
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # optFeatureMap data
    G_net, G_optimizer, D_net, D_optimizer, lossFunc = getGANPams(
        rowFeatureMap.size()[1],
        rowFeatureMap.size()[2],
        device,
        runPams.lr)

    predLabels = Predicte.myUtils.myTrainTest.train_test_GAN(
        rowFeatureMap, device, lossFunc, runPams, G_net, G_optimizer, D_net, D_optimizer)

    predLabels = np.array(predLabels)
    n = len(predLabels)
    n2 = len(predLabels[0])
    print(','.join(str(predLabels[i][j]) for i in range(n)  for j in range(n2)))
    # predLabels = np.resize(predLabels,(len(samples),len(samples[0]),1))
    predLabels = np.unique(predLabels)
    print(predLabels)
    sys.exit()
    # predLabels = np.random.randint(0, 16, (20, 500, 1))
    # get update row and col indices
    # initial the new empty samples list
    allNewSamples = list()
    for _ in range(16):
        allNewSamples.append([])
    # re generate the samples by their label
    partNum = len(predLabels)
    samplesNum = len(predLabels[0])
    for p in range(partNum):
        for s in range(samplesNum):
            label = predLabels[p][s]
            allNewSamples[label.item()].append(samples[p][s])

    # get new samples from mateData
    # test = Predicte.myUtils.myData.getNewPart(allNewSamples[0], mateData)

    p = Pool(os.cpu_count())
    results = list()
    for samples in allNewSamples:
        results.append(p.apply_async(Predicte.myUtils.myData.getNewPart, args=(samples, mateData)))
    p.close()
    p.join()
    newParts = list()
    for res in results:
        newParts.append(res.get())

    matchLabel = list()
    for newPart in newParts:
        if len(newPart) == 0:
            matchLabel.append("Nan")
            continue
        matchRowLen = np.sum(list(map(lambda x: x < yn, newPart.index)))
        matchColLen = np.sum(list(map(lambda x: x < yn, newPart.columns)))
        accuracy = ((matchRowLen * matchColLen) / (yn * yn)) * 100
        matchLabel.append(accuracy)
    matchLabel = ','.join(str(l) for l in matchLabel)
    print(matchLabel)
    return ()


# run main
if __name__ == "__main__":
    main()
