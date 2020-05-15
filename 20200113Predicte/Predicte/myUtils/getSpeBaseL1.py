# -*- coding: utf-8 -*

# system libs
import os
import sys
import platform

# 3rd libs
import torch

# my libs

import myUtils.myData


def main(runPams):
    # parameters
    blockNum = runPams[0]
    minusMean = runPams[1]
    normBias = runPams[2]
    baseNumThreshold = runPams[3]
    xn = runPams[4]
    inconThreshold = -7
    replace = 0
    zn = 6
    yn = xn
    totalRow = 56
    totalCol = totalRow
    overlap = 0
    probType = "l1Spe"

    # partitions
    labels_datas = myUtils.myData.getL1BaseData(normBias, minusMean, blockNum, zn,
                                                xn, yn, totalRow, totalCol, overlap,
                                                replace)
    datas = labels_datas[-1][-1]
    #[print(labels[i], datas[i]) for i in range(len(labels))]
    # sys.exit()
    # get samples
    samples, baseFeature, inconBaseFeature = myUtils.myData.getSamplesFeature(probType, datas, totalRow, totalCol,
                                                                              inconThreshold, baseNumThreshold)
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
    return (labels, samples, baseFeature, inconBaseFeature)


if __name__ == "__main__":
    main()
