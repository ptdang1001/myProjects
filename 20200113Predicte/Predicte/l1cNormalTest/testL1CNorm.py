# -*- coding: utf8 -*

# system lib
import sys
import argparse
import os

# third part libs
import torch
import torch.nn as nn

# my libs
sys.path.append("C:/users/pdang/Desktop/DATA/data.d/myGithubRepositories/myProjects/20200113Predicte/Predicte")
import myModules
import myUtils.myData
import myTrainTest
#import myUtils.myDraw
import myUtils.getNormalL1C



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
parser.add_argument("--xn",
                    type=int,
                    help="number of rows and cols")
parser.add_argument("--noiseMBias",
                    type=int,
                    help="noiseMBias")
parser.add_argument("--noiseStdBias",
                    type=int,
                    help="noiseStdBias")
parser.add_argument("--noiseNorm",
                    type=int,
                    help="noiseNorm")
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
    dpms = list()
    
    xn=opt.xn
    noiseMBias=opt.noiseMBias
    noiseStdBias=opt.noiseStdBias
    noiseNorm=opt.noiseNorm

    dpms.append(xn)
    dpms.append(noiseMBias)
    dpms.append(noiseStdBias)
    dpms.append(noiseNorm)
    # l1c
    
    olabel, odata, ssvdData, mapData= data.l1c.l1cdata.main(dpms)

    # original 2d data
    ores=myTrainTest.train_test(
        olabel, odata, ocnn, device, ooptimizer, olossFunc, opt)
    
    # mapdata train and test
    mres=myTrainTest.train_test(
        olabel, mapData, mcnn, device, moptimizer, mlossFunc, opt)

    # ssvddata train and test
    sres=myTrainTest.train_test(
        olabel, ssvdData, scnn, device, soptimizer, slossFunc, opt)
    #prepare results
    
    res = list()
    [res.append(dpms[i]) for i in range(len(dpms))]
    res.append(ores)
    res.append(mres)
    res.append(sres)
    res = ','.join(str(i) for i in res)
    print(res)
    return()

if __name__ == "__main__":
    main()
