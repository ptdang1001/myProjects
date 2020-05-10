# -*- coding: utf-8 -*

# system libs
import sys

# 3rd libs
import torch
import numpy as np

# my libs
sys.path.append(
    "C:/users/pdang/Desktop/DATA/data.d/myGithubRepositories/myProjects/20200113Predicte/Predicte"
)
import myUtils.myData


def main(runPams):
    # parameters
    minusMean = runPams[0]
    xn = runPams[1]
    normalBias = runPams[2]
    overlap = 0
    zn = 3
    yn = xn
    num = 3
    totalRow = 50
    totalCol = totalRow
    probType = "l1c"

    # partitions
    labels_datas = myUtils.myData.getL1CNormalData(normalBias, minusMean, num,
                                                   zn, xn, yn, totalRow,
                                                   totalCol, overlap)
    labels, datas = myUtils.myData.combineLabelData(labels_datas, zn, num)
    #[print(labels[i], datas[i]) for i in range(len(labels))]
    #sys.exit()
    # shuffle data
    datas = list(map(myUtils.myData.shuffleData, datas))
    datas = torch.stack(datas)

    # get ssvd datas
    ssvdDatas = list(map(myUtils.myData.ssvd, datas))
    ssvdDatas = torch.stack(ssvdDatas).float()

    # get 3d map
    mapData = myUtils.myData.get3dMap(probType, totalRow, totalCol, datas)
    # add number guassian noise to 3dmap data
    _, mapData = myUtils.myData.addNumGaussianNoise(mapData, labels,
                                                    int(mapData.size()[0] / 3))

    # add gaussian noise to labels, datas
    datas = datas.view(zn * num, 1, totalRow, totalCol)
    labels, datas = myUtils.myData.addNumGaussianNoise(
        datas, labels, int(datas.size()[0] / 3))

    # add guassian noise to ssvddatas ,labels
    ssvdDatas = ssvdDatas.view(zn * num, 1, totalRow, totalCol)
    _, ssvdDatas = myUtils.myData.addNumGaussianNoise(
        ssvdDatas, labels, int(ssvdDatas.size()[0] / 3))
    '''
    print(mapData.size())
    print(ssvdDatas.size())
    print(datas.size())
    print(labels.size())
    '''
    return (labels, datas, ssvdDatas, mapData)


if __name__ == "__main__":
    main()
