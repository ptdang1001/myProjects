# -*- coding: utf-8 -*

# system libs
import os
import sys
import time
import platform

# 3rd libs
import torch
import numpy as np
from itertools import permutations

# my libs
if platform.system() == "Windows":
    path = os.path.abspath("./Predicte")  #windos system
else:
    path = os.path.abspath("../Predicte")  #linux system
import myData


def get3dMap(totalRow, totalCol, noiseMean, noiseMbias, noiseStdbias, labels,
             datas):
    # l1 bases
    b1 = [1, 1, 1, -1, -1, -1, 0]
    b2 = [1, 1, -1, -1, 0, 0, 0]
    b3 = [1, -1, 0, 0, 0, 0, 0]
    l1Bases = [b1, b2, b3]

    baseTypeNum, basesMtrx = myData.getBasesMtrxs(l1Bases)
    randomRowColIdx = myData.getRandomRowColIdx(low=0,
                                                hight=totalCol - 1,
                                                row=2,
                                                col=len(b1),
                                                number=500)

    # mtx2map entity
    mtx2map = myData.Mtrx23dMap(baseTypeNum, basesMtrx, totalRow, totalCol,
                                randomRowColIdx)

    mapDatas = list(map(mtx2map.main, datas))
    mapDatas = torch.stack(mapDatas, 0)

    # add noise to mapDatas
    _, mapDatas = myData.addNumMeanNoise(mapDatas, labels,
                                         int(mapDatas.size()[0] / 3),
                                         noiseMean, noiseMbias, noiseStdbias)
    return (mapDatas)


def main(npms):
    # parameters
    mean = 10.0
    stdBias = 0.0
    noiseNorm = npms[3]
    noiseMbias = npms[1]
    noiseStdbias = npms[2]
    zn = 500
    xn = npms[0]
    yn = xn
    num = 3
    totalRow = 50
    totalCol = totalRow
    overlap = 1

    # partitions
    noiseMean, labels_datas = myData.getL1MeanData(mean, stdBias, noiseNorm,
                                                   noiseMbias, noiseStdbias,
                                                   num, zn, xn, yn, totalRow,
                                                   totalCol, overlap)
    #print(noiseMean - noiseMbias, noiseMean)
    labels = torch.cat(
        [torch.tensor([labels_datas[i][0]] * zn).long() for i in range(num)],
        dim=0)
    datas = torch.cat([labels_datas[i][1] for i in range(num)])
    #[print(labels[i], datas[i]) for i in range(len(labels))]
    #sys.exit()
    # shuffle data
    datas = list(map(myData.shuffleData, datas))
    datas = torch.stack(datas)
    # print(datas[100])

    ssvdDatas = list(map(myData.ssvd, datas))
    ssvdDatas = torch.stack(ssvdDatas).float()

    # get 3d map
    mapData = get3dMap(totalRow, totalCol, noiseMean, noiseMbias, noiseStdbias,
                       labels, datas)

    # add noise to labels, datas
    datas = datas.view(zn * num, 1, totalRow, totalCol)
    labels, datas = myData.addNumMeanNoise(datas, labels,
                                           int(datas.size()[0] / 3), noiseMean,
                                           noiseMbias, noiseStdbias)

    # add noise to ssvddatas ,labels
    ssvdDatas = ssvdDatas.view(zn * num, 1, totalRow, totalCol)
    _, ssvdDatas = myData.addNumMeanNoise(ssvdDatas, labels,
                                          int(ssvdDatas.size()[0] / 3), mean,
                                          noiseMbias, noiseStdbias)
    '''
    labelsArr = np.array(labels)
    datasArr = np.array(datas)
    ssvdDatasArr = np.array(ssvdDatas)
    np.save("Predicte/data/L1/l1labels1.npy", labelsArr)
    np.save("Predicte/data/L1/l1datas1.npy", datasArr)
    np.save("Predicte/data/L1/l1ssvdDatas1.npy", ssvdDatasArr)
    
    print(mapData.size())
    print(ssvdDatas.size())
    print(datas.size())
    print(labels.size())
    '''
    return (labels, datas, ssvdDatas, mapData)


if __name__ == "__main__":
    main()
