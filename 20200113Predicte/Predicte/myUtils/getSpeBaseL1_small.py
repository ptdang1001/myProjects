# -*- coding: utf-8 -*

# system libs
import os
import sys

# 3rd libs
import torch
import pysnooper
# my libs

import myUtils.myData


#@pysnooper.snoop()
def main(runPams):
    # parameters
    baseAddNorm = runPams["baseAddNorm"]
    minusMean = runPams["minusMean"]
    xn = runPams["xn"]
    stdBias = runPams["stdBias"]
    consisThreshold = runPams["consisThreshold"]
    mean = 0
    blockNum = 1
    replace = 0
    zn = 2
    yn = xn
    totalRow = 50
    totalCol = totalRow
    overlap = 0
    probType = "l1Spe"

    # partitions
    labels_datas = myUtils.myData.getL1SpeBaseData(baseAddNorm, minusMean,
                                                   blockNum, zn, xn, yn,
                                                   totalRow, totalCol, overlap,
                                                   replace)
    datas = labels_datas[-1][-1]
    # shuffle data
    datas = torch.stack(list(map(myUtils.myData.shuffleData, datas)))
    # get samples
    samples, rowFeatures, colFeatures, basesConsisScores = myUtils.myData.getSamplesFeature(
        probType, datas, totalRow, totalCol)
    colFeature_basesConsisScores_conThreshold = [
        [f, basesConsisScores, consisThreshold] for f in colFeatures
    ]
    optColFeatures = list(
        map(myUtils.myData.getOptFeature,
            colFeature_basesConsisScores_conThreshold))
    optColFeatures = myUtils.myData.completeData2SameColLen(optColFeatures)
    # get labels and samples
    labels, samples = myUtils.myData.getSamplesLabels(samples)
    # add data error
    samples = myUtils.myData.addDataError(samples, mean, stdBias)
    colFeatures = myUtils.myData.addDataError(colFeatures, mean, stdBias)
    optColFeatures = myUtils.myData.addDataError(optColFeatures, mean, stdBias)

    # reshape sample to 4D data
    samples = samples.view(samples.size()[0], 1,
                           samples.size()[1],
                           samples.size()[2])
    #reshape colFeature to 4D data
    colFeatures = colFeatures.view(colFeatures.size()[0], 1,
                                   colFeatures.size()[1],
                                   colFeatures.size()[2])
    #reshape optColFeature to 4D data
    optColFeatures = optColFeatures.view(optColFeatures.size()[0], 1,
                                         optColFeatures.size()[1],
                                         optColFeatures.size()[2])
    # add num mean noise to samples
    labels, samples = myUtils.myData.addNumMeanNoise(samples, labels,
                                                     int(len(samples) / 3),
                                                     mean, stdBias)
    # add num mean noise to col Feature
    _, colFeatures = myUtils.myData.addNumMeanNoise(colFeatures, labels,
                                                    int(len(colFeatures) / 3),
                                                    mean, stdBias)
    # add num mean noise to opt col feature
    _, optColFeatures = myUtils.myData.addNumMeanNoise(
        optColFeatures, labels, int(len(optColFeatures) / 3), mean, stdBias)
    '''
    [print(labels[i], samples[i]) for i in range(len(samples))]
    [print(labels[i], colFeatures[i]) for i in range(len(colFeatures))]
    [print(labels[i], optColFeatures[i]) for i in range(len(optColFeatures))]
    sys.exit()
    '''
    return (labels, samples, colFeatures, optColFeatures)
