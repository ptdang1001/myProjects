# -*- coding: utf-8 -*

# system libs
import os
import sys
import platform

# 3rd libs
import torch
import numpy as np
from itertools import permutations

# my libs
path = ""
if platform.system() == "Windows":
    path = os.path.abspath("./Predicte")  #windos system
else:
    path = os.path.abspath("..")  #linux system
sys.path.append(path)
import myUtils.myData


def main(runPams):
    # parameters
    replace = runPams[0]
    minusMean = runPams[1]
    xn = runPams[2]
    normBias = runPams[3]
    l1 = 1
    lk = 2
    zn = 1000
    yn = xn
    num = 3
    totalRow = 50
    totalCol = totalRow
    overlap = 0
    probType = "lall"

    # partitions
    labels_datas_l1c = myUtils.myData.getL1CNormData(normBias, minusMean, num,
                                                     zn, xn, yn, totalRow,
                                                     totalCol, overlap,
                                                     replace)
    labels_datas_l1 = myUtils.myData.getLkNormData(l1, normBias, minusMean,
                                                   num, zn, xn, yn, totalRow,
                                                   totalCol, overlap, replace)
    labels_datas_lk = myUtils.myData.getLkNormData(lk, normBias, minusMean,
                                                   num, zn, xn, yn, totalRow,
                                                   totalCol, overlap, replace)

    labels_datas = list()
    labels_datas.append(labels_datas_l1c)
    labels_datas.append(labels_datas_l1)
    labels_datas.append(labels_datas_lk)
    labels = list()
    datas = list()
    for i in range(len(labels_datas)):
        for j in range(num):
            label = torch.tensor([labels_datas[i][j][0]] * zn).long()
            labels.append(label)
            data = labels_datas[i][j][1]
            datas.append(data)
    labels = torch.cat(labels)
    datas = torch.cat(datas)
    #[print(labels[i], datas[i]) for i in range(len(labels))]
    #sys.exit()
    # shuffle data
    datas = list(map(myUtils.myData.shuffleData, datas))
    datas = torch.stack(datas)
    #get ssvd data
    ssvdDatas = list(map(myUtils.myData.ssvd, datas))
    ssvdDatas = torch.stack(ssvdDatas).float()
    # get 3d map
    mapData = myUtils.myData.get3dMap(probType, totalRow, totalCol, datas)
    _, mapData = myUtils.myData.addNumGaussianNoise(mapData, labels,
                                                    int(mapData.size()[0] / 3))
    # add noise to labels, datas
    datas = datas.view(len(datas), 1, totalRow, totalCol)
    labels, datas = myUtils.myData.addNumGaussianNoise(
        datas, labels, int(datas.size()[0] / 3))

    # add noise to ssvddatas ,labels
    ssvdDatas = ssvdDatas.view(len(ssvdDatas), 1, totalRow, totalCol)
    _, ssvdDatas = myUtils.myData.addNumGaussianNoise(
        ssvdDatas, labels, int(ssvdDatas.size()[0] / 3))
    '''
    [print(labels[i], datas[i]) for i in range(len(labels))]
    print(mapData.size())
    print(ssvdDatas.size())
    print(datas.size())
    print(labels.size())
    '''
    return (labels, datas, ssvdDatas, mapData)


if __name__ == "__main__":
    runPams = [0, 0, 7, 0]
    main(runPams)
