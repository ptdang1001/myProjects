# -*- coding: utf8 -*

# system lib
import argparse
import sys
#import time
import os
#from multiprocessing import Pool

# third part lib
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
#from sklearn.utils import shuffle
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt

#my libs

import myData
import myModule

# parameters
parser = argparse.ArgumentParser()
if torch.cuda.is_available():
    parser.add_argument("--batch_size", type=int, default=5)
else:
    parser.add_argument("--batch_size", type=int, default=5)
parser.add_argument("--lr", type=float, default=0.0005)
parser.add_argument("--n_cpu", type=int, default=os.cpu_count())
parser.add_argument("--k", type=int, default=2)
runPams = parser.parse_args()


def getVAEPams(xn, yn, device, lr):
    VAE = myModule.VAE(xn, yn)
    VAE = VAE.to(device)
    optimizer = torch.optim.Adagrad(VAE.parameters(), lr=lr)
    lossFunc = nn.MSELoss()
    return (VAE, optimizer, lossFunc)

def getNewJ(grads, set_c, times=5):
    # dropout the weights multi times and average the G
    grads = grads * set_c
    cn = len(set_c)
    sum_grads = torch.zeros_like(grads)

    for _ in range(times):
        tmpGrads = grads.clone()
        rdmIdx = torch.randint(0, 2, (1, cn))
        tmpGrads = tmpGrads * rdmIdx
        sum_grads = sum_grads + tmpGrads
    tmpGrads = sum_grads / times
    tmpGradsMax = torch.norm(tmpGrads, p=2, dim=0, keepdim=True)
    maxIdx = torch.argmax(tmpGradsMax)
    return (maxIdx)

def main():
    
    #pre process data
    '''
    x = pd.read_csv(
        "/N/project/zhangclab/pengtao/myProjectsDataRes/20200619Classification/data/salmonE74cDNA_counts_baseline.csv",
        index_col=0)
    x = x.T
    data = x.values.copy()
    '''
    data = np.random.rand(10, 200)
    xn, yn = data.shape
    data = np.reshape(data, (xn, 1, yn))
    data = np.insert(data, 0, 1, axis=2)
    zn, xn, yn = data.shape
    # set s
    set_s = np.zeros(xn * yn)
    set_s[0] = 1

    # set c
    set_c = np.ones(xn * yn)
    set_c[0] = 0

    # dataLoader
    dataSet = myData.MyDataset(data, data)
    dataLoader = DataLoader(dataset=dataSet,
                            batch_size=zn,
                            shuffle=False,
                            num_workers=runPams.n_cpu,
                            pin_memory=torch.cuda.is_available())

    # choose cpu or gpu automatically
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net, optimizer, lossFunc = getVAEPams(xn, yn, device, runPams.lr)

    #np->tensor or gpu
    set_s = torch.tensor(set_s).float().to(device)
    set_c = torch.tensor(set_c).float().to(device)

    # train
    while torch.sum(set_s == 1) <= (runPams.k+1):
        for step, (x, _) in enumerate(dataLoader):
            b_x = Variable(x.view(-1, xn * yn).float().to(device))
            b_y = Variable(x.view(-1, xn * yn).float().to(device))

            # initialize the weight of set c to be zero and of set s to be normal
            net.fc1.weight.data = net.fc1.weight.data * (set_s)

            # network
            encoded, decoded, label = net(b_x)
            loss = lossFunc(decoded, b_y)  # mean square error
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward() # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            #get new J
            newJ = getNewJ(net.fc1.weight.grad.clone(), set_c)


            # initialize the weight of node J by xavier
            tmpWeight = torch.rand(1, net.fc1.out_features)
            tmpWeight = nn.init.xavier_normal_(tmpWeight)
            net.fc1.weight.data[:, newJ] = tmpWeight

            # update set s and aet C
            set_s[newJ] = torch.tensor(1).to(device)
            set_c[newJ] = torch.tensor(0).to(device)
            '''
            if step % 10 == 9:
                print('Epoch: ', count, '| train loss: %.4f' % loss.data.numpy())
            '''

    # test

    predLabels = list()
    features = list()
    for (x, y) in dataLoader:
        b_x = Variable(x.view(-1, xn * yn).float().to(device))  # batch x (data)
        feature, _, predicted = net(b_x)
        features.append([feature.cpu().detach().numpy()])
        predicted = torch.max(predicted.data, 1)[1].cpu().numpy()
        predLabels.append(predicted)
    # test end

    features = np.hstack(features)
    zn, xn, yn = features.shape
    features = np.reshape(features, (xn, yn))
    features = np.array(features)

    estimator = KMeans(n_clusters=4, random_state=0).fit(features)
    label_pred = estimator.labels_
    # get figures
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    # 这里'or'代表中的'o'代表画圈，'r'代表颜色为红色，后面的依次类推
    for i in range(len(label_pred)):
        plt.plot([features[i, 0]], [features[i, 1]], mark[label_pred[i]], markersize=5)
    #save data
    pathName = "C:\\Users\\pdang\\Desktop\\"
    fileName = pathName + str(runPams.k) + ".png"
    plt.savefig(fileName)

    fileName = pathName + str(runPams.k) + ".csv"
    setS = pd.DataFrame(set_s, index=None)
    setS = setS.T
    setS.to_csv(fileName)
    #plt.show()
    return()

if __name__ == "__main__":
    main()