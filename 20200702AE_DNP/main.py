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
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE 
import umap
from sklearn import cluster 
#my libs

import myData
import myModule

# parameters
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=1)
if torch.cuda.is_available():
    parser.add_argument("--batch_size", type=int, default=745)
else:
    parser.add_argument("--batch_size", type=int, default=745)
parser.add_argument("--lr", type=float, default=0.005)
parser.add_argument("--n_cpu", type=int, default=os.cpu_count())
parser.add_argument("--k", type=int, default=3)
parser.add_argument("--n_top_gene", type=int, default=500)
runPams = parser.parse_args()


def getVAEPams(xn, yn, device, lr):
    VAE = myModule.VAE(xn, yn)
    VAE = VAE.to(device)
    optimizer = torch.optim.Adagrad(VAE.parameters(), lr=lr)
    lossFunc = nn.MSELoss()
    return (VAE, optimizer, lossFunc)

def getNewJ(grads, set_c,  device, times=8):
    # dropout the weights multi times and average the G
    grads = grads * set_c 
    cn = len(set_c)
    sum_grads = torch.zeros_like(grads).to(device)

    for _ in range(times):
        tmpGrads = grads.clone()
        rdmIdx = torch.randint(0, 2, (1, cn)).to(device)
        tmpGrads = tmpGrads * rdmIdx
        sum_grads = sum_grads + tmpGrads
    tmpGrads = (sum_grads / times)
    #tmpGradsMax = torch.norm(tmpGrads, p=2, dim=0, keepdim=True)
    tmpGradsMax = torch.sum(tmpGrads ** 2, 1).sqrt()
    maxIdx = torch.argmax(tmpGradsMax)
    return (maxIdx)

def main():
    #select cpu or gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(16)

    #pre process data

    #annotation
    an = pd.read_csv("/N/project/zhangclab/pengtao/myProjectsDataRes/20200619Classification/data/annotation.csv", index_col = 0)
    tmp = an["loh_percent"].copy()
    for i in range(len(tmp)):
        if tmp[i] >= 0.05:
            tmp[i] = 1
        else:
            if tmp[i] < 0.05:
                tmp[i] = 0
    an["loh_percent"] = tmp


    tmp2 = an["mutations_per_mb"].copy()
    for i in range(len(tmp2)):
        if tmp2[i] >= 28:
            tmp2[i] = 1
        else:
            if tmp2[i] < 28:
                tmp2[i] = 0
    an["mutations_per_mb"] = tmp2

    #data 
    x = pd.read_csv(
        "/N/project/zhangclab/pengtao/myProjectsDataRes/20200619Classification/data/salmonE74cDNA_counts_baseline.csv",
        index_col=0)
    x = x.T
    x = (x + 1).apply(np.log2)
    #test = np.median(x, axis=0)
    x_std = np.std(x, axis=0)
    top_gene = runPams.n_top_gene
    top_gene_idx=x_std.argsort()[::-1][0:top_gene]
    data = x.iloc[:, top_gene_idx]
    data = data.values.copy()
    top_gene_names = list(x.columns[top_gene_idx])
    top_gene_names = np.insert(top_gene_names, 0, "bias")
    
    #data = np.random.rand(10, 200)
    xn, yn = data.shape
    
    # umap + kmeans
    pams = str(runPams.k)+"_"+str(runPams.n_top_gene)
    pathName = "/N/project/zhangclab/pengtao/myProjectsDataRes/20200619Classification/results/"+pams+"/imgs_UMAP/"
    # umap
    reducer = umap.UMAP()
    z = reducer.fit_transform(data)
    # kmeans
    kmeans = KMeans(n_clusters=4, random_state=0).fit(z)
    
    imgName = "kmeans.png"
    myData.myDraw(z, kmeans.labels_, pathName, imgName)
    
    
    # Hierarchical Clustering
    clst=cluster.AgglomerativeClustering(n_clusters=4)

    imgName = "Hierarchical_Clustering.png"
    myData.myDraw(z, clst.fit_predict(z), pathName, imgName)


    for i in range(1, len(an.columns)):
        a = an.columns[i]
        imgName = str(a)+".png"
        myData.myDraw(z, an[a], pathName, imgName)

    # TSNE 
    pathName = "/N/project/zhangclab/pengtao/myProjectsDataRes/20200619Classification/results/"+pams+"/imgs_TSNE/"

    # T-sNE
    z = TSNE(n_components=2).fit_transform(data)
    # kmeans
    kmeans = KMeans(n_clusters=4, random_state=0).fit(z)
    
    imgName = "kmeans.png"
    myData.myDraw(z, kmeans.labels_, pathName, imgName)
    
    
    # Hierarchical Clustering
    clst=cluster.AgglomerativeClustering(n_clusters=4)

    imgName = "Hierarchical_Clustering.png"
    myData.myDraw(z, clst.fit_predict(z), pathName, imgName)


    for i in range(1, len(an.columns)):
        a = an.columns[i]
        imgName = str(a)+".png"
        myData.myDraw(z, an[a], pathName, imgName)
    
    # vae+dnp
    #data = np.random.rand(10, 2000)
    #xn, yn = data.shape
    data = np.reshape(data, (xn, 1, yn))
    data = np.insert(data, 0, 1, axis=2)
    #data = data[:,:,:5000]
    zn, xn, yn = data.shape
    # set s
    set_s = np.zeros(xn * yn)
    set_s[0] = 1

    # set c
    set_c = np.ones(xn * yn)
    set_c[0] = 0
    
    # np 2 tensor 
    data = torch.tensor(data)
    # dataLoader
    dataSet = myData.MyDataset(data, data)
    dataLoader = DataLoader(dataset=dataSet,
                            batch_size=runPams.batch_size,
                            shuffle=False,
                            num_workers=runPams.n_cpu,
                            pin_memory=torch.cuda.is_available())


    net, optimizer, lossFunc = getVAEPams(xn, yn, device, runPams.lr)

    #np->tensor or gpu
    set_s = torch.tensor(set_s).float().to(device)
    set_c = torch.tensor(set_c).float().to(device)

    # train
    while torch.sum(set_s == 1).item() < (runPams.k+1):
        print(torch.sum(set_s==1).item())
        for _ in range(runPams.epoch):
            for step, (x, _) in enumerate(dataLoader):
                b_x = Variable(x.view(-1, xn * yn).float().to(device))
                b_y = Variable(x.view(-1, xn * yn).float().to(device))
    
                # initialize the weight of set c to be zero and of set s to be normal
                net.fc1.weight.data = net.fc1.weight.data * (set_s)
    
                # network
                _, decoded, _ = net(b_x)
                loss = lossFunc(decoded, b_y)  # mean square error
                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward() # backpropagation, compute gradients
                optimizer.step()  # apply gradients 
                print(net.fc1.weight.grad)

        #get new J
        newJ = getNewJ(net.fc1.weight.grad.clone(), set_c, device).item()
        print(newJ)

        # initialize the weight of node J by xavier
        tmpWeight = torch.rand(1, net.fc1.out_features)
        tmpWeight = nn.init.xavier_normal_(tmpWeight)
        net.fc1.weight.data[:, newJ] = tmpWeight

        # update set s and aet C
        set_s[newJ] = torch.tensor(1)
        set_c[newJ] = torch.tensor(0)

    # test
    #sys.exit()
    predLabelsByVAE = list()
    features = list()
    for (x, _) in dataLoader:
        b_x = Variable(x.view(-1, xn * yn).float().to(device))  # batch x (data)
        feature, _, predicted = net(b_x)
        features.append([feature.cpu().detach().numpy()])
        predicted = torch.max(predicted.data, 1)[1].cpu().numpy()
        predLabelsByVAE.append(predicted)
    # test end

    features = np.hstack(features)
    zn, xn, yn = features.shape
    features = np.reshape(features, (xn, yn))
    features = np.array(features)
    z = features
    pams = str(runPams.k)+"_"+str(runPams.n_top_gene)
    pathName = "/N/project/zhangclab/pengtao/myProjectsDataRes/20200619Classification/results/"+pams+"/imgs_VAE+DNP/"   

    # kmeans
    kmeans = KMeans(n_clusters=4, random_state=0).fit(z)
     
    imgName = "kmeans.png"
    myData.myDraw(z, kmeans.labels_, pathName, imgName)
     
     
    # Hierarchical Clustering
    clst=cluster.AgglomerativeClustering(n_clusters=4)
 
    imgName = "Hierarchical_Clustering.png"
    myData.myDraw(z, clst.fit_predict(z), pathName, imgName)
 
 
    for i in range(1, len(an.columns)):
        a = an.columns[i]
        imgName = str(a)+".png"
        myData.myDraw(z, an[a], pathName, imgName)
    

    # save gene names
    pathName = "/N/project/zhangclab/pengtao/myProjectsDataRes/20200619Classification/results/"+pams+"/genes_selected.csv"
    genes = pd.DataFrame(set_s.cpu().detach().numpy())
    genes = genes.T
    genes.columns = top_gene_names
    genes.to_csv(pathName)
    '''
    kmeans_estimator = KMeans(n_clusters=4, random_state=0).fit(features)
    labelByVAEKmeans = kmeans_estimator.labels_ 
    # get figures
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    # 这里'or'代表中的'o'代表画圈，'r'代表颜色为红色，后面的依次类推
    for i in range(len(labelByVAEKmeans)):
        plt.plot([features[i, 0]], [features[i, 1]], mark[label_pred[i]], markersize=5)
    #save data
    pathName = "/N/project/zhangclab/pengtao/myProjectsDataRes/20200702AE_DNP/results/csv_img_res/"
    fileName = pathName + str(runPams.k) + ".png"
    plt.savefig(fileName)

    fileName = pathName + str(runPams.k) + ".csv"
    setS = pd.DataFrame(set_s.cpu().detach().numpy())
    setS = setS.T
    setS.to_csv(fileName)
    #plt.show()
    '''
    return()

if __name__ == "__main__":
    main()
