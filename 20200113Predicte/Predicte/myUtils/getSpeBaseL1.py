# -*- coding: utf-8 -*

# system libs
import os
import sys

# 3rd libs
import torch
import pysnooper
# my libs

import myUtils.myData

@pysnooper.snoop()
def main(runPams):
    # parameters
    baseAddNorm = runPams[0]
    minusMean = runPams[1]
    xn = runPams[2]
    stdBias = runPams[3]
    baseNumThreshold = runPams[4]
    mean = 0
    blockNum = 1
    conThreshold = 0
    replace = 0
    zn = 1
    yn = xn
    totalRow = 50
    totalCol = totalRow
    overlap = 0
    probType = "l1Spe"

    # partitions
    labels_datas = myUtils.myData.getL1SpeBaseData(baseAddNorm, minusMean, blockNum, zn,
                                                   xn, yn, totalRow, totalCol, overlap,
                                                   replace)
    datas = labels_datas[-1][-1]
    # shuffle data
    datas = list(map(myUtils.myData.shuffleData, datas))
    datas = torch.stack(datas)
    # get samples
    samples, rowFeatures, colFeatures, basesConsisScores = myUtils.myData.getSamplesFeature(probType, datas, totalRow,
                                                                                            totalCol)

    colFeature_basesConsisScores = [[f, basesConsisScores, conThreshold] for f in colFeatures]
    optColFeatures = list(map(myUtils.myData.getOptFeature, colFeature_basesConsisScores))
    sys.exit()
    labels, samples = myUtils.myData.getSamplesLabels(samples)
    # [print(labels[i],samples[i]) for i in range(len(samples))]
    # sys.exit()
    # add data error
    samples = myUtils.myData.addDataError(samples, mean, stdBias)
    rowFeature = myUtils.myData.addDataError(rowFeature, mean, stdBias)
    colFeature = myUtils.myData.addDataError(colFeature, mean, stdBias)
    # [print(labels[i], samples[i]) for i in range(len(samples))]
    # sys.exit()
    # process samples
    samples = samples.view(
        samples.size()[0],
        1,
        samples.size()[1],
        samples.size()[2])
    # add num mean noise to samples
    labels, samples = myUtils.myData.addNumMeanNoise(
        samples, labels, int(len(samples) / 3), mean, stdBias)
    # add num mean noise to samples
    _, rowFeature = myUtils.myData.addNumMeanNoise(
        rowFeature, labels, int(len(baseFeature) / 3), mean, stdBias)
    # add num mean noise to samples
    _, colFeature = myUtils.myData.addNumMeanNoise(
        colFeature, labels, int(
            len(colFeature) / 3), mean, stdBias)
    return (labels, samples, rowFeature, colFeature)
