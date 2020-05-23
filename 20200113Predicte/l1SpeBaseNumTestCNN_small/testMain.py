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

# my libs
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(".."))

import getSpeBaseL1_small
import Predicte.myModules
import Predicte.myUtils.myTrainTest
import Predicte.myUtils.myData

def getFCNPams(rowNum, colNum, device, lr):
    fcn = Predicte.myModules.FCN(rowNum=rowNum, colNum=colNum)
    fcn = fcn.to(device)
    optimizer = torch.optim.Adam(fcn.parameters(), lr=lr)
    lossFunc = nn.CrossEntropyLoss()
    return (fcn, optimizer, lossFunc)

def getCNNPams(rowNum, colNum,device, lr):
    cnnXout = Predicte.myUtils.myData.getCNNOutSize(rowNum.size()[2], 3, 2)
    cnnYout = Predicte.myUtils.myData.getCNNOutSize(colNum.size()[3], 3, 2)
    cnn = Predicte.myModules.CNN(inChannels=2,
                         kernels=6,
                         kernelSize=2,
                         outSize=16 * cnnXout * cnnYout)
    cnn = cnn.to(device)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)
    lossFunc = nn.CrossEntropyLoss()
    return(cnn,optimizer,lossFunc)
# end


def main(runPams):
    # get samples, featureMap, optFeatureMap
    olabel, samples, featureMap, optFeatureMap = getSpeBaseL1_small.main(
        runPams)
    '''
    [print(olabel[i],samples[i]) for i in range(len(samples))]
    print("------------------------------------------------------")
    [print(olabel[i],featureMap[i]) for i in range(len(featureMap))]
    print("--------------------------------------------------")
    [print(olabel[i],optFeatureMap[i]) for i in range(len(optFeatureMap))]
    print("---------------------------------------------------------------")
    '''
    print(olabel.size())
    print(samples.size())
    print(featureMap.size())
    print(optFeatureMap.size())

    # choose spu or gpu automatically
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # samples data
    fcn, optimizer, lossFunc = getFCNPams(
        samples.size()[2],
        samples.size()[3],
        device,
        runPams.lr)
    sres, sytrue_ypred = Predicte.myUtils.myTrainTest.train_test(
        olabel, samples, fcn, device, optimizer, lossFunc, runPams)

    # featureMap data
    cnn, optimizer, lossFunc = getCNNPams(
        featureMap.size()[2],
        featureMap.size()[3],
        device,
        runPams.lr)
    fres, fytrue_ypred = Predicte.myUtils.myTrainTest.train_test(
        olabel, featureMap, cnn, device, optimizer, lossFunc, runPams)

    # optFeatureMap data
    cnn, optimizer, lossFunc = getCNNPams(
        optFeatureMap.size()[2],
        optFeatureMap.size()[3],
        device,
        runPams.lr)
    ores, oytrue_ypred = Predicte.myUtils.myTrainTest.train_test(
        olabel, optFeatureMap, cnn, device, optimizer, lossFunc, runPams)
    # prepare results

    res = list()
    if runPams.minusMean == 1:
        res.append("c*r-E")
    else:
        res.append("c*r")
    res.append("N(0-" + str(runPams.stdBias/10) + ")")
    res.append(runPams.xn)
    res.append(runPams.numThreshold)
    res.append("7*" + str(samples.size()[2]))
    res.append(sres)
    res.append("7*" + str(featureMap.size()[3]))
    res.append(fres)
    res.append("7*" + str(optFeatureMap.size()[3]))
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
    timeStam = str(int(time.time()))
    #filePath = "C:\\Users\\pdang\\Desktop\\" + timeStam + ".xlsx"
    filePath = "/N/project/zhangclab/pengtao/myProjectsDataRes/20200113Predicte/results/l1SpeBaseNumTestCNN/block1_small/excelRes/" + timeStam + ".xlsx"
    writer = pd.ExcelWriter(filePath)  # 写入Excel文件
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
    )
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--n_cpu", type=int, default=os.cpu_count())
    parser.add_argument("--minusMean", type=int, default=0)
    parser.add_argument("--stdBias", type=int, default=0)
    parser.add_argument("--numThreshold", type=int, default=7)
    parser.add_argument("--xn", type=int, default=20)
    parser.add_argument("--crType", type=str, default="norm")
    runPams = parser.parse_args()
    main(runPams)
