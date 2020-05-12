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
    lk = 2
    zn = 1000
    yn = xn
    num = 3
    totalRow = 50
    totalCol = totalRow
    overlap = 0
    probType = "l1"

    # partitions
    labels_datas = myUtils.myData.getLkNormData(lk, normBias, minusMean, num,
                                                zn, xn, yn, totalRow, totalCol,
                                                overlap, replace)
    labels = torch.cat(
        [torch.tensor([labels_datas[i][0]] * zn).long() for i in range(num)],
        dim=0)
    datas = torch.cat([labels_datas[i][1] for i in range(num)])
    #[print(labels[i], datas[i]) for i in range(len(labels))]
    #sys.exit()
    # shuffle data
    datas = list(map(myUtils.myData.shuffleData, datas))
    datas = torch.stack(datas)
    #print(datas[0])

    ssvdDatas = list(map(myUtils.myData.ssvd, datas))
    ssvdDatas = torch.stack(ssvdDatas).float()

    # get 3d map
    mapData = myUtils.myData.get3dMap(probType, totalRow, totalCol, datas)

    _, mapData = myUtils.myData.addNumGaussianNoise(mapData, labels,
                                                    int(mapData.size()[0] / 3))
    # add noise to labels, datas
    datas = datas.view(zn * num, 1, totalRow, totalCol)
    labels, datas = myUtils.myData.addNumGaussianNoise(
        datas, labels, int(datas.size()[0] / 3))

    # add noise to ssvddatas ,labels
    ssvdDatas = ssvdDatas.view(zn * num, 1, totalRow, totalCol)
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
