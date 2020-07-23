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
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# my libs
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(".."))

import Predicte.myModules
import Predicte.myUtils.myTrainTest
import Predicte.myUtils.myData

# parameters
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--n_inner_epochs", type=int, default=3)
if torch.cuda.is_available():
    parser.add_argument("--batch_size", type=int, default=500)
else:
    parser.add_argument("--batch_size", type=int, default=500)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--n_cpu", type=int, default=os.cpu_count())
parser.add_argument("--minusMean", type=int, default=1)
parser.add_argument("--xn", type=int, default=200)
parser.add_argument("--crType", type=str, default="uniform")
<<<<<<< HEAD
parser.add_argument("--baseTimes", type=int, default=1)
parser.add_argument("--errorStdBias", type=int, default=2)
=======
parser.add_argument("--baseTimes", type=int, default=10)
parser.add_argument("--errorStdBias", type=int, default=0 / 10)
>>>>>>> bc27f9ef45248b5553a0507017e7bb0f57f45632
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
    totalCol = 1000
    overlap = 0
    probType = "l1"
    runPams.errorStdBias = runPams.errorStdBias / 10
    # choose cpu or gpu automatically
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #torch.manual_seed(16)
    # partitions
    st = time.time()
    labels_mateDatas = Predicte.myUtils.myData.getL1SpeBaseData(runPams.crType, minusMean, runPams.errorStdBias,
                                                                blockNum, runPams.baseTimes, zn, runPams.xn, yn,
                                                                totalRow, totalCol, overlap,
                                                                replace)
    mateDatas = list(labels_mateDatas[-1][-1])

    mateData = pd.DataFrame(mateDatas[0])
    mateData_noshuffle = mateData.copy()
    mateData = shuffle(mateData)
    mateData = shuffle(mateData.T)
    mateData = mateData.T
<<<<<<< HEAD
=======
    mateData = mateData_noshuffle
>>>>>>> bc27f9ef45248b5553a0507017e7bb0f57f45632
    # mateData = pd.DataFrame(np.random.rand(1000, 1000))

    # get partitions
    parts = Predicte.myUtils.myData.mateData2Parts(mateData.copy())
<<<<<<< HEAD
    #parts = [ part.abs() for part in parts]
    for _ in range(runPams.n_inner_epochs):
        res = list()
        with Pool(os.cpu_count()) as p:
            res = p.map(Predicte.myUtils.myData.getSamplesRowColStd, parts)
        # splite samples, samplesArr, rowStdArr, colStdArr from res
        samples = list()
        samplesArr = list()
        # rowStdArr = list()
        # colStdArr = list()
        for r in res:
            samples.append(r[0])
            samplesArr.append(r[1])
            # rowStdArr.append(r[2])
            # colStdArr.append(r[3])
        labels = Predicte.myUtils.myData.getLabelFrSamples(samples, runPams.xn, runPams.xn)
    
        samplesArr = np.stack(samplesArr)
        # rowStdArr = np.stack(rowStdArr)
        # colStdArr = np.stack(colStdArr)
    
        # get bases matrix
        basesMtrx, baseTypeNumAfterKmean, baseIdAfterKMeans = Predicte.myUtils.myData.getBasesMtrxAfterKmean()
        # get row and col feature map: (7*7) * (7*1652)
        rowFeatureMap = np.matmul(samplesArr, (basesMtrx.iloc[:, 0:7].values.T))
        # colFeatureMap = np.matmul(samplesArr.transpose((0, 1, 3, 2)), (basesMtrx.iloc[:, 0:7].values.T))

        # normalize row and col by std from original 50*50's row and col std
        # rowFeatureMap = np.true_divide(rowFeatureMap, rowStdArr)
        rowFeatureMap = -np.sort(-rowFeatureMap, axis=2)

        # normalize col by std from original 50*50' col
        # colFeatureMap = np.true_divide(colFeatureMap, colStdArr)
        # colFeatureMap = -np.sort(-colFeatureMap, axis=2)

        # resort them by their mean
        rowFeatureMap = Predicte.myUtils.myData.getResortMeanFeatureMap(rowFeatureMap[:, :, 2:7, :])
        # colFeatureMap = Predicte.myUtils.myData.getResortMeanFeatureMap(colFeatureMap)

        # row and col max pooling 7*1652 -> 7*16
        rowFeatureMap = Predicte.myUtils.myData.myMaxPooling(rowFeatureMap, baseTypeNumAfterKmean)
        # colFeatureMap = Predicte.myUtils.myData.myMaxPooling(colFeatureMap, baseTypeNumAfterKmean)
        # featureMap = np.stack((rowFeatureMap, colFeatureMap), axis=2)

        # sort the rows
        rowFeatureMap = -np.sort(-rowFeatureMap, axis=3)[:, :, :, :]
        rowFeatureMap = rowFeatureMap[:, :, :, 0:5] - rowFeatureMap[:, :, :, 11:16]
        '''
        featureMean = np.mean(rowFeatureMap, axis=3)
        zn, xn, yn = featureMean.shape
        featureMean = np.reshape(featureMean, (zn, xn, yn, 1))
        featureStd = np.std(rowFeatureMap, axis=3)
        zn, xn, yn = featureStd.shape
        featureStd = np.reshape(featureStd, (zn, xn, yn, 1))
        rowFeatureMap = np.true_divide(rowFeatureMap, featureMean)
        '''
        rowFeatureMap_np = rowFeatureMap.copy()
        labels = torch.tensor(labels).long()
        rowFeatureMap = torch.tensor(rowFeatureMap).float()
        rowFeatureMap = rowFeatureMap.view(rowFeatureMap.size()[0] * rowFeatureMap.size()[1], rowFeatureMap.size()[2],
                                           rowFeatureMap.size()[3])

        # rowFeatureMap = torch.rand(100, 7, 16)

        # optFeatureMap data
        net, optimizer, lossFunc = getFCNPams(
            rowFeatureMap.size()[1],
            rowFeatureMap.size()[2],
            device,
            runPams.lr)
        '''
        z, predLabels = Predicte.myUtils.myTrainTest.train_test_FCN(
            rowFeatureMap, labels, net, device, optimizer, lossFunc, runPams)
        '''
        acc, predLabelsByFCN = Predicte.myUtils.myTrainTest.train_test_FCN(rowFeatureMap, labels, net, device, optimizer,
                                                                 lossFunc, runPams)
        print(predLabelsByFCN)
        print(acc)
        #sys.exit()
        '''
        kmeans_estimator = KMeans(n_clusters=2, random_state=0).fit(z)
        kmeans_label_pred = kmeans_estimator.labels_
        predLabels = kmeans_label_pred
        print(predLabels)
        '''
        predLabels = [pl[1] for pl in predLabelsByFCN]
        predLabels = np.concatenate(predLabels)
        print(np.unique(predLabels))
        labelType = np.unique(predLabels)
        classNum = len(labelType)
        predLabels = np.resize(predLabels, (len(samples), len(samples[0]), 1))

        # get update row and col indices
        # initial the new empty samples list
        allNewSamples = list()
        for _ in range(classNum):
            allNewSamples.append([])

        # re generate the samples by their generated label
        sampleSetNum = len(samples)
        samplesNum = len(samples[0])
        for i in range(sampleSetNum):
            for j in range(samplesNum):
                label = predLabels[i][j]
                idx = np.where(labelType == label.item())[0][0]
                allNewSamples[idx].append(samples[i][j])
    
        # get new expand samples from mateData
        # test = Predicte.myUtils.myData.getNewPart(allNewSamples[0], mateData)

        pool = Pool(os.cpu_count())
        tmpResults = list()
        for samples in allNewSamples:
            tmpResults.append(pool.apply_async(Predicte.myUtils.myData.getNewPart, args=(samples, mateData, runPams.xn)))
        pool.close()
        pool.join()

        # get new partitions
        newParts = list()
        for res in tmpResults:
            newParts.append(res.get())
        parts = newParts
    # caculate the match degree
=======

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
    # rowStdArr = np.stack(rowStdArr)
    # colStdArr = np.stack(colStdArr)

    # get bases matrix
    basesMtrx, baseTypeNumAfterKmean, baseIdAfterKMeans = Predicte.myUtils.myData.getBasesMtrxAfterKmean()
    # get row and col feature map: (7*7) * (7*1652)
    rowFeatureMap = np.matmul(samplesArr, (basesMtrx.iloc[:, 0:7].values.T))
    # colFeatureMap = np.matmul(samplesArr.transpose((0, 1, 3, 2)), (basesMtrx.iloc[:, 0:7].values.T))

    # normalize row and col by std from original 50*50's row and col std
    # rowFeatureMap = np.true_divide(rowFeatureMap, rowStdArr)
    rowFeatureMap = -np.sort(-rowFeatureMap, axis=2)

    # normalize col by std from original 50*50' col
    # colFeatureMap = np.true_divide(colFeatureMap, colStdArr)
    # colFeatureMap = -np.sort(-colFeatureMap, axis=2)

    # resort them by their mean
    rowFeatureMap = Predicte.myUtils.myData.getResortMeanFeatureMap(rowFeatureMap[:, :, 2:7, :])
    # colFeatureMap = Predicte.myUtils.myData.getResortMeanFeatureMap(colFeatureMap)

    # row and col max pooling 7*1652 -> 7*16
    rowFeatureMap = Predicte.myUtils.myData.myMaxPooling(rowFeatureMap, baseTypeNumAfterKmean)
    # colFeatureMap = Predicte.myUtils.myData.myMaxPooling(colFeatureMap, baseTypeNumAfterKmean)
    # featureMap = np.stack((rowFeatureMap, colFeatureMap), axis=2)

    # sort the rows
    rowFeatureMap = -np.sort(-rowFeatureMap, axis=3)[:, :, :, :5]
    rowFeatureMap_np = rowFeatureMap.copy()
    rowFeatureMap = torch.tensor(rowFeatureMap)
    rowFeatureMap = rowFeatureMap.view(rowFeatureMap.size()[0] * rowFeatureMap.size()[1], rowFeatureMap.size()[2],
                                       rowFeatureMap.size()[3])

    # rowFeatureMap = torch.rand(100, 7, 16)
    # choose cpu or gpu automatically
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # optFeatureMap data
    net, optimizer, lossFunc = getVAEPams(
        rowFeatureMap.size()[1],
        rowFeatureMap.size()[2],
        device,
        runPams.lr)

    z, predLabels = Predicte.myUtils.myTrainTest.train_test_VAE(
        rowFeatureMap, net, device, optimizer, lossFunc, runPams)
    #kmeans_estimator = KMeans(n_clusters=2, random_state=0).fit(z)
    #kmeans_label_pred = kmeans_estimator.labels_

    labelType = np.unique(predLabels)
    classNum = len(labelType)
    predLabels = np.resize(predLabels, (len(samples), len(samples[0]), 1))


    # get update row and col indices
    # initial the new empty samples list
    allNewSamples = list()
    for _ in range(classNum):
        allNewSamples.append([])

    # re generate the samples by their generated label
    sampleSetNum = len(samples)
    samplesNum = len(samples[0])
    for i in range(sampleSetNum):
        for j in range(samplesNum):
            label = predLabels[i][j]
            idx = np.where(labelType == label.item())[0][0]
            allNewSamples[idx].append(samples[i][j])

    # get new expand samples from mateData
    # test = Predicte.myUtils.myData.getNewPart(allNewSamples[0], mateData)

    pool = Pool(os.cpu_count())
    tmpResults = list()
    for samples in allNewSamples:
        tmpResults.append(pool.apply_async(Predicte.myUtils.myData.getNewPart, args=(samples, mateData)))
    pool.close()
    pool.join()

    # get new partitions
    newParts = list()
    for res in tmpResults:
        newParts.append(res.get())

    #caculate the match degree
>>>>>>> bc27f9ef45248b5553a0507017e7bb0f57f45632
    matchLabel = list()
    for newPart in newParts:
        if len(newPart) == 0:
            matchLabel.append("Nan")
            continue
        matchRowLen = np.sum(list(map(lambda x: x < yn, newPart.index)))
        matchColLen = np.sum(list(map(lambda x: x < yn, newPart.columns)))
        accuracy = ((matchRowLen * matchColLen) / (yn * yn)) * 100
        accuracy = np.around(accuracy, decimals=2)
        matchLabel.append(accuracy)
<<<<<<< HEAD
    # matchLabel = ','.join(str(l) for l in matchLabel)
=======
    #matchLabel = ','.join(str(l) for l in matchLabel)
>>>>>>> bc27f9ef45248b5553a0507017e7bb0f57f45632

    # output the results
    res = list()
    res.append(runPams.xn)
    res.append(runPams.baseTimes)
    res.append(runPams.errorStdBias)
    for label in matchLabel:
        res.append(label)
    res = pd.DataFrame(res)
    res = res.T
<<<<<<< HEAD
    print(res)
    sys.exit()
    pathName = "C:/Users/pdang/Desktop/"
    fileName = pathName + "finalRes_U_20200708.csv"
    res.to_csv(fileName, mode="a", index=False, header=False)
    print("end")
    print("end")
=======
    pathName = "C:/Users/pdang/Desktop/"
    fileName = pathName + "finalRes_U_20200707.csv"
    res.to_csv(fileName, mode="a", index=False, header=False)
>>>>>>> bc27f9ef45248b5553a0507017e7bb0f57f45632
    return ()


# run main
if __name__ == "__main__":
    main()
