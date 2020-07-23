# -*- coding: utf-8 -*

# system libs
import sys
import os

# 3rd libs
import torch
import pysnooper
# my libs
#add current folder to sys.path
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(".."))

import Predicte.myUtils.myData


#@pysnooper.snoop()
def main(runPams):
    # parameters
    mean = 0
    blockNum = 1
    replace = 0
    zn = 20
    yn = runPams.xn
    totalRow = 50
    totalCol = totalRow
    overlap = 0
    probType = "l1Spe"
    # partitions
    labels_datas = Predicte.myUtils.myData.getL1SpeBaseData(runPams.crType, runPams.minusMean,
                                           blockNum, zn, runPams.xn, yn,
                                           totalRow, totalCol, overlap,
                                           replace)
    datas = labels_datas[-1][-1]
    # shuffle data
    datas = torch.stack(list(map(Predicte.myUtils.myData.shuffleData, datas)))
    # get samples, featureMap, optFeatureMap
    labels, samples, featureMap, optFeatureMap = Predicte.myUtils.myData.getSamplesFeature(
        probType, mean, runPams.stdBias/10, runPams.numThreshold, datas, totalRow, totalCol)
    return (labels, samples, featureMap, optFeatureMap)
