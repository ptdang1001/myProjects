# -*- coding: utf8 -*

# system lib
import pandas as pd
import torch.nn as nn
import torch
import argparse
import sys
import time
import os

path = os.path.abspath("..")
sys.path.append(path)
import myUtils.getSpeBaseL1
import myUtils.myTrainTest
import myUtils.myData
import myModules


# third part libs

# my libs


def main(opt):
    # data parameters to get data
    runPams = {}
    runPams["baseAddNorm"] = opt.baseAddNorm
    runPams["minusMean"] = opt.minusMean
    runPams["xn"] = opt.xn
    runPams["stdBias"] = opt.stdBias/10
    runPams["consisThreshold"] = opt.consisThreshold
    '''
    runPams["baseAddNorm"] = 0
    runPams["minusMean"] = 0
    runPams["xn"] = 20
    runPams["stdBias"] = 0
    runPams["consisThreshold"] = 40
    '''
    # l1 bases
    baseLen = 7
    olabel, samples, colFeatures, optColFeatures = myUtils.getSpeBaseL1.main(runPams)
    '''
    #[print(olabel[i], samples[i]) for i in range(len(olabel))]
    print(optColFeatures.size())
    print(colFeatures.size())
    print(samples.size())
    print(olabel.size())
    sys.exit()
    '''
    # networks
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # samples networks
    sfcn = myModules.FCN(size=baseLen)
    sfcn = sfcn.to(device)
    soptimizer = torch.optim.Adam(sfcn.parameters(), lr=opt.lr)
    slossFunc = nn.CrossEntropyLoss()

    # colFeature network
    ccnnXout = myUtils.myData.getCNNOutSize(colFeatures.size()[2], 3, 2)
    ccnnYout = myUtils.myData.getCNNOutSize(colFeatures.size()[3], 3, 2)
    ccnn = myModules.CNN(inChannels=1,
                         kernels=6,
                         kernelSize=2,
                         outSize=16 * ccnnXout * ccnnYout)
    ccnn = ccnn.to(device)
    coptimizer = torch.optim.Adam(ccnn.parameters(), lr=opt.lr)
    clossFunc = nn.CrossEntropyLoss()

    # optColFeature network
    ocnnXout = myUtils.myData.getCNNOutSize(optColFeatures.size()[2], 3, 2)
    ocnnYout = myUtils.myData.getCNNOutSize(optColFeatures.size()[3], 3, 2)
    ocnn = myModules.CNN(inChannels=1,
                         kernels=6,
                         kernelSize=2,
                         outSize=16 * ocnnXout * ocnnYout)
    ocnn = ocnn.to(device)
    ooptimizer = torch.optim.Adam(ocnn.parameters(), lr=opt.lr)
    olossFunc = nn.CrossEntropyLoss()

    # samples data
    sres, sytrue_ypred = myUtils.myTrainTest.train_test(olabel, samples, sfcn, device,
                                                        soptimizer, slossFunc, opt)

    # colFeature data
    cres, cytrue_ypred = myUtils.myTrainTest.train_test(olabel, colFeatures, ccnn, device,
                                                        coptimizer, clossFunc, opt)

    # optColFeature data
    ores, oytrue_ypred = myUtils.myTrainTest.train_test(olabel, optColFeatures, ocnn, device,
                                                        ooptimizer, olossFunc, opt)
    # prepare results

    res = list()
    res.append(runPams["baseAddNorm"])
    if runPams["minusMean"] == 1:
        res.append("c*r-E")
    else:
        res.append("c*r")
    res.append("N(0-" + str(runPams["stdBias"]) + ")")
    res.append(runPams["xn"])
    res.append(baseLen)
    res.append(runPams["consisThreshold"])
    res.append(sres)
    res.append(colFeatures.size()[3])
    res.append(cres)
    res.append(optColFeatures.size()[3])
    res.append(ores)
    #save data to excel
    resDF = pd.DataFrame(res)
    resDF.columns = ["res"]
    sytrue_ypred = pd.DataFrame(sytrue_ypred)
    sytrue_ypred.columns = ["true", "pred"]
    cytrue_ypred = pd.DataFrame(cytrue_ypred)
    cytrue_ypred.columns = ["true", "pred"]
    oytrue_ypred = pd.DataFrame(oytrue_ypred)
    oytrue_ypred.columns = ["true", "pred"]
    timeStam = str(int(time.time()))
    filePath = "C:\\Users\\pdang\\Desktop\\" + timeStam + ".xlsx"
    # filePath = "/N/project/zhangclab/pengtao/myProjectsDataRes/20200113Predicte/results/l1SpeBaseTest/block1/excelRes/" + tm + ".xlsx"
    writer = pd.ExcelWriter(filePath)  # 写入Excel文件
    resDF.to_excel(writer, index=False)
    sytrue_ypred.to_excel(writer, startcol=2, index=False)
    cytrue_ypred.to_excel(writer, startcol=5, index=False)
    oytrue_ypred.to_excel(writer, startcol=8, index=False)
    writer.save()
    writer.close()
    #output data
    res = ','.join(str(i) for i in res)
    print(res)
    return ()


if __name__ == "__main__":
    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10, )
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--n_cpu", type=int, default=os.cpu_count())
    parser.add_argument("--minusMean", type=int)
    parser.add_argument("--stdBias", type=int)
    parser.add_argument("--consisThreshold", type=int)
    parser.add_argument("--xn", type=int)
    parser.add_argument("--baseAddNorm", type=int)
    opt = parser.parse_args()
    main(opt)
