# -*- coding: utf-8 -*

# system libs
import os
import sys
import platform

# 3rd libs
import torch

# my libs
path = os.path.abspath('./Predicte')
sys.path.append(path)
import myUtils.myData


def main(runPams):
    # parameters
    minusMean = runPams[0]
    xn = runPams[1]
    basesNum=runPams[2]
    normBias = 0
    replace = 0
    lk = 1
    zn = 1
    yn = xn
    blockNum = 1
    totalRow = 50
    totalCol = totalRow
    overlap = 0
    probType = "l1"

    # partitions
    labels_datas = myUtils.myData.getLkNormData(lk, normBias, minusMean, blockNum, zn,
                                                xn, yn, totalRow, totalCol, overlap,
                                                replace)
    datas = torch.cat([labels_datas[i][1] for i in range(blockNum)])
    # [print(labels[i], datas[i]) for i in range(len(labels))]
    # sys.exit()
    # get samples
    samples, baseFeature, inconBaseFeature = myUtils.myData.getSamplesFeature(probType, datas, totalRow, totalCol, basesNum)
    labels, samples = myUtils.myData.getSamplesLabels(samples)
    # [print(labels[i],samples[i]) for i in range(len(samples))]
    # sys.exit()
    # shuffle samples
    samples = list(map(myUtils.myData.shuffleData, samples))
    samples = torch.stack(samples)
    # process samples
    samples = samples.view(samples.size()[0], 1, samples.size()[1], samples.size()[2])
    labels, samples = myUtils.myData.addNumGaussianNoise(samples, labels, int(len(samples) / 3))
    _, baseFeature = myUtils.myData.addNumGaussianNoise(baseFeature, labels, int(len(baseFeature) / 3))
    _, inconBaseFeature = myUtils.myData.addNumGaussianNoise(inconBaseFeature, labels, int(len(inconBaseFeature) / 3))
    #[print(labels[i],samples[i]) for i in range(len(samples))]

    print(inconBaseFeature.size())
    print(baseFeature.size())
    print(samples.size())
    print(labels.size())

    return (labels, samples, baseFeature, inconBaseFeature)


if __name__ == "__main__":
    runPams = [0, 25,50]
    main(runPams)
