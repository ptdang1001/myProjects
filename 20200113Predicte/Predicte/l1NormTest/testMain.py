# -*- coding: utf8 -*

# system lib
import sys
import argparse
import os

# third part libs
import torch
import torch.nn as nn

# my libs
path = os.path.abspath("./Predicte")
sys.path.append(path)
import myModules
import myUtils.myData
import myUtils.myTrainTest
#import myUtils.myDraw
import myUtils.getNormL1

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
    # ocnn networks
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ocnn = myModules.CNN(inChannels=1,
                         kernels=6,
                         kernelSize=7,
                         outSize=16 * 8 * 8)
    ocnn = ocnn.to(device)
    ooptimizer = torch.optim.Adam(ocnn.parameters(), lr=opt.lr)
    olossFunc = nn.CrossEntropyLoss()

    # map data
    mcnn = myModules.CNN(inChannels=3,
                         kernels=6,
                         kernelSize=2,
                         outSize=16 * 1 * 249)
    mcnn = mcnn.to(device)
    moptimizer = torch.optim.Adam(mcnn.parameters(), lr=opt.lr)
    mlossFunc = nn.CrossEntropyLoss()

    # ssvd data
    scnn = myModules.CNN(inChannels=1,
                         kernels=6,
                         kernelSize=7,
                         outSize=16 * 8 * 8)
    scnn = scnn.to(device)
    soptimizer = torch.optim.Adam(scnn.parameters(), lr=opt.lr)
    slossFunc = nn.CrossEntropyLoss()

    #data parameters
    runPams = list()
    '''
    replace = opt.replace
    minusMean = opt.minusMean
    xn = opt.xn
    normBias = opt.normBias
    '''
    replace = 0
    minusMean = 0
    xn = 7
    normBias = 0

    runPams.append(replace)
    runPams.append(minusMean)
    runPams.append(xn)
    runPams.append(normBias)
    # l1c normal

    olabel, odata, ssvdData, mapData = myUtils.getNormL1.main(runPams)
    #[print(olabel[i], odata[i]) for i in range(12)]

    # original 2d data
    ores = myUtils.myTrainTest.train_test(olabel, odata, ocnn, device,
                                          ooptimizer, olossFunc, opt)

    # mapdata train and test
    mres = myUtils.myTrainTest.train_test(olabel, mapData, mcnn, device,
                                          moptimizer, mlossFunc, opt)

    # ssvddata train and test
    sres = myUtils.myTrainTest.train_test(olabel, ssvdData, scnn, device,
                                          soptimizer, slossFunc, opt)
    #prepare results

    res = list()
    if replace == 0:
        res.append("x+z")
    else:
        res.append("x->z")
    if minusMean == 0:
        res.append("c*r+" + str(normBias))
    else:
        res.append("c*r-E+" + str(normBias))
    res.append(xn)
    res.append("N(0,1)-E")
    res.append(ores)
    res.append(mres)
    res.append(sres)
    res = ','.join(str(i) for i in res)
    print(res)
    return ()


if __name__ == "__main__":
    main()
