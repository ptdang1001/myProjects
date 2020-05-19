# -*- coding: utf8 -*

# system lib

import sys
import argparse
import os

# third part libs
import torch
import torch.nn as nn
import pandas as pd

# my libs
path = os.path.abspath("..")
sys.path.append(path)
import myUtils.getSpeBaseL1
import myUtils.myTrainTest
import myUtils.myData
import myModules

# parameters
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs",
                    type=int,
                    default=10,
                    help="number of epochs of training")
parser.add_argument("--batch_size",
                    type=int,
                    default=10,
                    help="size of the batches")
parser.add_argument("--lr",
                    type=float,
                    default=0.0002,
                    help="adam: learning rate")
parser.add_argument(
    "--n_cpu",
    type=int,
    default=os.cpu_count(),
    help="number of cpu threads to use during batch generation")

parser.add_argument("--replace", type=int, help="data+noise or data>noise")
parser.add_argument("--minusMean", type=int, help="if minus mean")
parser.add_argument("--stdBias", type=int, help="std")
parser.add_argument(
    "--baseNumThreshold",
    type=int,
    help="base number threshold")
parser.add_argument("--xn", type=int, help="number of rows and cols")
parser.add_argument("--baseAddNorm", type=int, help="if add norm to data")

opt = parser.parse_args()


def main():
    # data parameters
    runPams = list()

    baseAddNorm = opt.baseAddNorm
    minusMean = opt.minusMean
    xn = opt.xn
    stdBias = opt.stdBias
    stdBias = stdBias / 10
    baseNumThreshold = opt.baseNumThreshold
    '''
    baseAddNorm=0
    minusMean = 0
    xn = 20
    stdBias = 5
    stdBias=stdBias/10
    baseNumThreshold = 40
    '''
    runPams.append(baseAddNorm)
    runPams.append(minusMean)
    runPams.append(xn)
    runPams.append(stdBias)
    runPams.append(baseNumThreshold)
    # l1 bases
    baseLen = 7
    olabel, samples, baseFeatures, inconBaseFeatures = myUtils.getSpeBaseL1.main(
        runPams)
    '''
    [print(olabel[i], samples[i]) for i in range(len(olabel))]
    print(inconBaseFeatures.size())
    print(baseFeatures.size())
    print(samples.size())
    print(olabel.size())
    sys.exit()
    '''
    # networks
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # samples networks
    sfcn = myModules.FCN(size=baseLen)
    sfcn = sfcn.to(device)
    soptimizer = torch.optim.Adam(sfcn.parameters(), lr=opt.lr)
    slossFunc = nn.CrossEntropyLoss()

    # baseFeature network
    bcnnXout = myUtils.myData.getCNNOutSize(baseFeatures.size()[2], 3, 2)
    bcnnYout = myUtils.myData.getCNNOutSize(baseFeatures.size()[3], 3, 2)
    bcnn = myModules.CNN(inChannels=2,
                         kernels=6,
                         kernelSize=2,
                         outSize=16 * bcnnXout * bcnnYout)
    bcnn = bcnn.to(device)
    boptimizer = torch.optim.Adam(bcnn.parameters(), lr=opt.lr)
    blossFunc = nn.CrossEntropyLoss()

    # inconBaseFeature network
    icnnXout = myUtils.myData.getCNNOutSize(inconBaseFeatures.size()[2], 3, 2)
    icnnYout = myUtils.myData.getCNNOutSize(inconBaseFeatures.size()[3], 3, 2)
    icnn = myModules.CNN(inChannels=2,
                         kernels=6,
                         kernelSize=2,
                         outSize=16 * icnnXout * icnnYout)
    icnn = icnn.to(device)
    ioptimizer = torch.optim.Adam(icnn.parameters(), lr=opt.lr)
    ilossFunc = nn.CrossEntropyLoss()

    # samples data
    sres = myUtils.myTrainTest.train_test(olabel, samples, sfcn, device,
                                                        soptimizer, slossFunc, opt)

    # baseFeature data
    bres = myUtils.myTrainTest.train_test(olabel, baseFeatures, bcnn, device,
                                                        boptimizer, blossFunc, opt)

    # inconBaseFeature data
    ires = myUtils.myTrainTest.train_test(olabel, inconBaseFeatures, icnn, device,
                                                        ioptimizer, ilossFunc, opt)
    # prepare results

    res = list()
    res.append(baseAddNorm)
    if minusMean == 1:
        res.append("c*r-E")
    else:
        res.append("c*r")
    res.append("N(0-" + str(stdBias) + ")")
    res.append(xn)
    res.append(baseLen)
    res.append(sres)
    res.append(baseFeatures.size()[3])
    res.append(bres)
    res.append(inconBaseFeatures.size()[3])
    res.append(ires)
    res = ','.join(str(i) for i in res)
    print(res)
    return ()


if __name__ == "__main__":
    main()
