# -*- coding: utf8 -*

# system lib
import argparse
import sys
import time
import os

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
def getData(runPams):
    # parameters
    mean = 0
    minusMean = 1
    blockNum = 1
    replace = 0
    zn = 100
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

    labels_parts = list(map(Predicte.myUtils.myData.mateData2Parts, mateDatas))
    labels = list()
    parts = list()
    for i in range(len(labels_parts)):
        for j in range(len(labels_parts[0])):
            labels.append(labels_parts[i][j][0])
            parts.append(labels_parts[i][j][1])
    featureMap_optFeatureMap = list(map(Predicte.myUtils.myData.get3dMap, parts))
    featureMap = list()
    optFeatureMap = list()
    for fo in featureMap_optFeatureMap:
        featureMap.append(fo[0])
        optFeatureMap.append(fo[1])
    featureMap=torch.stack(featureMap)
    optFeatureMap = torch.stack(optFeatureMap)
    parts = np.stack(parts)
    parts = torch.from_numpy(parts).view(parts.shape[0], 1, parts.shape[1], parts.shape[2])
    return (labels, parts, featureMap, optFeatureMap)


# end

def main(runPams):
    timeStam = str(int(time.time()))
    # saveExcelPath = "C:\\Users\\pdang\\Desktop\\" + timeStam + ".xlsx"
    saveExcelPath = "/N/project/zhangclab/pengtao/myProjectsDataRes/20200113Predicte/results/l1UniformWholeTest_small/block1/excelRes/" + timeStam + ".xlsx"
    # st = time.time()
    # get parts, featureMap, optFeatureMap
    olabel, parts, featureMap, optFeatureMap = getData(runPams)

    # choose spu or gpu automatically
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # parts data
    net, optimizer, lossFunc = getCNNPams(
        parts.size()[1],
        parts.size()[2],
        parts.size()[3],
        device,
        runPams.lr)
    press, sytrue_ypred = Predicte.myUtils.myTrainTest.train_test(
        olabel, parts, net, device, optimizer, lossFunc, runPams)

    # featureMap data
    net, optimizer, lossFunc = getCNNPams(
        featureMap.size()[1],
        featureMap.size()[2],
        featureMap.size()[3],
        device,
        runPams.lr)
    fres, fytrue_ypred = Predicte.myUtils.myTrainTest.train_test(
        olabel, featureMap, net, device, optimizer, lossFunc, runPams)
    # optFeatureMap data
    net, optimizer, lossFunc = getCNNPams(
        optFeatureMap.size()[1],
        optFeatureMap.size()[2],
        optFeatureMap.size()[3],
        device,
        runPams.lr)
    ores, oytrue_ypred = Predicte.myUtils.myTrainTest.train_test(
        olabel, optFeatureMap, net, device, optimizer, lossFunc, runPams)
    # prepare results

    res = list()
    if runPams.minusMean == 1:
        res.append("c*r-E")
    else:
        res.append("c*r")
    res.append(runPams.xn)
    res.append("N(0-" + str(runPams.stdBias / 10) + ")")
    res.append(runPams.baseTimes)
    res.append(runPams.numThreshold)
    res.append("50*" + str(parts.size()[2]))
    res.append(press)
    res.append('*'.join(str(i) for i in featureMap.size()))
    res.append(fres)
    res.append("*".join(str(i) for i in optFeatureMap.size()))
    res.append(ores)
    # save data to excel
    resDF = pd.DataFrame(res)
    resDF.columns = ["res"]
    sytrue_ypred = pd.DataFrame(sytrue_ypred)
    sytrue_ypred.columns = ["true", "pred"]
    cytrue_ypred = pd.DataFrame(fytrue_ypred)
    cytrue_ypred.columns = ["true", "pred"]
    oytrue_ypred = pd.DataFrame(oytrue_ypred)
    oytrue_ypred.columns = ["true", "pred"]

    writer = pd.ExcelWriter(saveExcelPath)  # 写入Excel文件
    resDF.to_excel(writer, index=False)
    sytrue_ypred.to_excel(writer, startcol=2, index=False)
    cytrue_ypred.to_excel(writer, startcol=5, index=False)
    oytrue_ypred.to_excel(writer, startcol=8, index=False)
    writer.save()
    writer.close()
    # output data
    res = ','.join(str(i) for i in res)
    print(res)
    return ()


# run main
if __name__ == "__main__":
    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=10)
    if torch.cuda.is_available():
        parser.add_argument("--batch_size", type=int, default=64)
    else:
        parser.add_argument("--batch_size", type=int, default=os.cpu_count())
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--n_cpu", type=int, default=os.cpu_count())
    parser.add_argument("--minusMean", type=int, default=1)
    parser.add_argument("--stdBias", type=int, default=0)
    parser.add_argument("--numThreshold", type=int, default=30)
    parser.add_argument("--xn", type=int, default=50)
    parser.add_argument("--crType", type=str, default="uniform")
    parser.add_argument("--sampleNum", type=int, default=500)
    parser.add_argument("--baseTimes", type=int, default=3)
    parser.add_argument("--errorStdBias", type=int, default=0 / 10)
    runPams = parser.parse_args()

    main(runPams)
