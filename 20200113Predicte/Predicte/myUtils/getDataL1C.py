# -*- coding: utf-8 -*

# system libs
import sys

# 3rd libs
import torch
import numpy as np


# my libs
sys.path.append(
    "/geode2/home/u070/pdang/Carbonate/projects/20200113Predicte/Predicte"
)
import myData


def main(dpms):
    #parameters
    mean = 100
    stdBias = 0
    noiseNorm = dpms[3] 
    noiseMbias = dpms[1]
    noiseStdbias = dpms[2]
    zn = 500
    xn = dpms[0]
    yn = xn
    num = 3
    totalRow = 50
    totalCol = totalRow
    overlap = 1
    probType = "l1c"

    # partitions
    labels_datas = myData.getL1CMeanData(mean, stdBias, noiseNorm, noiseMbias,
                                                 noiseStdbias, num, zn, xn, yn,
                                                 totalRow, totalCol, overlap)

    labels, datas = myData.combineLabelData(labels_datas, zn, num)
    #[print(labels[i], datas[i]) for i in range(len(labels))]
    #sys.exit()
    #shuffle data
    datas = list(map(myData.shuffleData, datas))
    datas = torch.stack(datas)
    
    #get ssvd datas
    ssvdDatas = list(map(myData.ssvd, datas))
    ssvdDatas = torch.stack(ssvdDatas).float()

    #get 3d map
    mapData = myData.get3dMap(probType, totalRow, totalCol, mean,
                                      noiseMbias, noiseStdbias, labels, datas)
    #add number noise to 3dmap data
    _, mapData = myData.addNumMeanNoise(mapData, labels,
                                                int(mapData.size()[0] / 3),
                                                mean, noiseMbias, noiseStdbias)
    
    #add noise to labels, datas
    datas = datas.view(zn * num, 1, totalRow, totalCol)
    labels, datas = myData.addNumMeanNoise(datas, labels,
                                                   int(datas.size()[0] / 3),
                                                   mean, noiseMbias,
                                                   noiseStdbias)
    
    # add noise to ssvddatas ,labels
    ssvdDatas = ssvdDatas.view(zn * num, 1, totalRow, totalCol)
    _, ssvdDatas = myData.addNumMeanNoise(ssvdDatas, labels,
                                                  int(ssvdDatas.size()[0] / 3),
                                                  mean, noiseMbias,
                                                  noiseStdbias)
    '''
    labelsArr = np.array(labels)
    datasArr = np.array(datas)
    ssvdDatasArr = np.array(ssvdDatas)
    mapDataArr = np.array(mapData)
    np.save("Predicte/data/L1C/l1cmapdata1.npy", mapDataArr)
    np.save("Predicte/data/L1C/l1clabels1.npy", labelsArr)
    np.save("Predicte/data/L1C/l1cdatas1.npy", datasArr)
    np.save("Predicte/data/L1C/l1cssvdDatas1.npy", ssvdDatas)
    
    print(mapData.size())
    print(ssvdDatas.size())
    print(datas.size())
    print(labels.size())
    '''
    return (labels,datas,ssvdDatas,mapData)


if __name__ == "__main__":
    main()
