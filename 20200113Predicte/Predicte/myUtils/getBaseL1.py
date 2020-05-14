# -*- coding: utf-8 -*

# system libs
import os
import sys
import platform

# 3rd libs
import torch

# my libs
'''
path = ""
if platform.system() == "Windows":
    path = os.path.abspath("./Predicte")  #windos system
else:
    path = os.path.abspath("..")  #linux system
sys.path.append(path)
print(sys.path)
'''
import myData


def main(runPams):
    # parameters
    minusMean = runPams[0]
    xn = runPams[1]
    normBias = runPams[2]
    replace = 0
    lk = 1
    zn = 2
    yn = xn
    blockNum = 1
    totalRow = 50
    totalCol = totalRow
    overlap = 0
    probType = "l1"

    # partitions
    labels_datas = myData.getLkNormData(lk, normBias, minusMean, blockNum, zn,
                                        xn, yn, totalRow, totalCol, overlap,
                                        replace)
    datas = torch.cat([labels_datas[i][1] for i in range(blockNum)])
    #[print(labels[i], datas[i]) for i in range(len(labels))]
    #sys.exit()

    # get samples
    samples, baseFeature = myData.getSamplesFeature(probType, datas, totalRow, totalCol)
    labels, samples = myData.getSamplesLabels(samples)
    print(baseFeature.size())
    return (labels, samples)


if __name__ == "__main__":
    runPams = [0, 25, 0]
    main(runPams)
