# -*- coding: utf8 -*

# system lib
import sys
import argparse
import os

# third part libs
import torch
import torch.nn as nn

# my libs
path = os.path.abspath("..")
sys.path.append(path)
import myModules
import myUtils.myData
import myUtils.myTrainTest
#import myUtils.myDraw
import myUtils.getBaseL1

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
parser.add_argument("--n_print",
                    type=int,
                    default=200,
                    help="how many times training print one result")
parser.add_argument("--replace", type=int, help="data+noise or data>noise")
parser.add_argument("--minusMean", type=int, help="if minus mean")
parser.add_argument("--normBias", type=int, help="data plus bias")
parser.add_argument("--xn", type=int, help="number of rows and cols")
opt = parser.parse_args()


def main():
    # data parameters
    runPams = list()
    '''
    minusMean = opt.minusMean
    xn = opt.xn
    '''
    minusMean = 0
    xn = 30
    baseSize = 7
    runPams.append(minusMean)
    runPams.append(xn)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # samples networks
    sfcn = myModules.FCN(size=baseSize)
    sfcn = sfcn.to(device)
    soptimizer = torch.optim.Adam(sfcn.parameters(), lr=opt.lr)
    slossFunc = nn.CrossEntropyLoss()

    # baseFeature network
    bcnn = myModules.CNN(inChannels=2,
                         kernels=6,
                         kernelSize=2,
                         outSize=16 * 1 * 97)
    bcnn = bcnn.to(device)
    boptimizer = torch.optim.Adam(bcnn.parameters(), lr=opt.lr)
    blossFunc = nn.CrossEntropyLoss()

    # inconBaseFeature network
    icnn = myModules.CNN(inChannels=2,
                         kernels=6,
                         kernelSize=2,
                         outSize=16 * 1 * 11)
    icnn = icnn.to(device)
    ioptimizer = torch.optim.Adam(icnn.parameters(), lr=opt.lr)
    ilossFunc = nn.CrossEntropyLoss()

    # l1 bases

    olabel, samples, baseFeatures, inconBaseFeatures  = myUtils.getBaseL1.main(runPams)
    #[print(olabel[i], odata[i]) for i in range(12)]


    # samples data
    sres = myUtils.myTrainTest.train_test(olabel, samples, sfcn, device,
                                          soptimizer, slossFunc, opt)

    # baseFeature data
    bres = myUtils.myTrainTest.train_test(olabel, baseFeatures, bcnn, device,
                                          boptimizer, blossFunc, opt)

    # inconBaseFeature data
    ires = myUtils.myTrainTest.train_test(olabel, inconBaseFeatures, icnn, device,
                                          ioptimizer, ilossFunc, opt)
    #prepare results

    res = list()
    if minusMean == 1:
        res.append("X-E")
    else:
        res.append("X")
    res.append(xn)
    res.append(baseSize)
    res.append("N(01)-E")
    res.append(sres)
    res.append(bres)
    res.append(ires)
    res = ','.join(str(i) for i in res)
    print(res)
    return ()


if __name__ == "__main__":
    main()
