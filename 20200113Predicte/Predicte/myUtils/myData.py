# -*- coding: utf-8 -*

# sys libs
import sys
import csv

# 3rd libs
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from itertools import permutations
import itertools
from scipy.sparse.linalg import svds, eigs
import pysnooper
import numba as nb


# my libs


def test():
    print("hello")


# end


# gene dataset loader
class GeneDataset(Dataset):
    def __init__(self, data, label, transform=None):
        self.data = data
        self.label = label
        self.transform = transform

    # end

    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    # end

    def __len__(self):
        return len(self.data)
    # end


class Mtrx23dMap():
    def __init__(self, baseTypeNum, bases, totalRow, totalCol,
                 randomRowColIdx):
        self.baseTypeNum = baseTypeNum
        self.bases = bases
        self.totalRow = totalRow
        self.totalCol = totalCol
        self.randomRowColIdx = randomRowColIdx

    # end

    def getMaxSlice(self, baseFeature, startIdx, endIdx):
        maxSlice = torch.max(baseFeature[:, :, startIdx:endIdx], 2)
        maxSlice = maxSlice[0]  # get values
        z = maxSlice.size()[0]
        x = maxSlice.size()[1]
        y = 1
        maxSlice = maxSlice.reshape(z, x, y)
        return (maxSlice)

    # end

    def getBaseMap(self, samples):
        rowFeature = torch.matmul(samples, self.bases.float())
        colFeature = torch.matmul(samples.permute(0, 2, 1), self.bases.float())
        baseFeature = torch.cat((rowFeature, colFeature), 0)  # 1000x7xm
        # max pooling
        finalMap = [
            self.getMaxSlice(baseFeature, self.baseTypeNum[i],
                             self.baseTypeNum[i + 1])
            for i in range(len(self.baseTypeNum))
            if i < len(self.baseTypeNum) - 1
        ]
        finalMap = torch.cat(finalMap, 2)  # 1000x7xlen(bases)
        return (finalMap)

    # end

    def getBaseMapNoMaxPooling(self, samples):
        rowFeature = torch.matmul(samples, self.bases.float())
        colFeature = torch.matmul(samples.permute(0, 2, 1), self.bases.float())
        # baseFeature = torch.stack((rowFeature, colFeature), 1)  # 1000*2*7*m
        # baseFeature = baseFeature.permute(0, 3, 2, 1)
        return (rowFeature, colFeature)

    # end

    def getSamples(self, partition):
        samples = [
            partition[np.ix_(rowIdx, colIdx)]
            for rowIdx, colIdx in self.randomRowColIdx
        ]
        samples = torch.stack(samples)
        return (samples)

    # end

    def getFeatureMap(self, partition):
        samples = self.getSamples(partition)
        baseMap = self.getBaseMap(samples)
        featureMap = baseMap.permute(2, 1, 0)
        return (featureMap)

    # end

    def main(self, partition):
        featureMap = self.getFeatureMap(partition)
        return (featureMap)
    # end


# end

def getSamplesFeature(probType, partitions, totalRow, totalCol):
    # l1 bases
    bases = list()
    if probType == "l1c":
        b1 = [1, 1, 1, 1, 1, 1, 0]
        b2 = [1, 1, 1, 1, 0, 0, 0]
        b3 = [1, 1, 0, 0, 0, 0, 0]
        bases = [b1, b2, b3]
    elif probType == "l1":
        b1 = [1, 1, 1, -1, -1, -1, 0]
        b2 = [1, 1, -1, -1, 0, 0, 0]
        b3 = [1, -1, 0, 0, 0, 0, 0]
        bases = [b1, b2, b3]
    elif probType == "lall":
        b1 = [1, 1, 1, 1, 1, 1, 0]
        b2 = [1, 1, 1, 1, 0, 0, 0]
        b3 = [1, 1, 0, 0, 0, 0, 0]
        b4 = [1, 1, 1, -1, -1, -1, 0]
        b5 = [1, 1, -1, -1, 0, 0, 0]
        b6 = [1, -1, 0, 0, 0, 0, 0]
        bases = [b1, b2, b3, b4, b5, b6]
    else:
        base1 = [1, 1, 1, -1, -1, -1, 0]
        base2 = [1, 1, -1, -1, 0, 0, 0, ]
        base3 = [1, -1, 0, 0, 0, 0, 0]
        base4 = [2, 1, 1, -1, -1, -2, 0]
        bases = [base1, base2, base3, base4]
    baseTypeNum, basesMtrx = getBasesMtrxs(bases)
    randomRowColIdx = getRandomRowColIdx(low=0,
                                         hight=totalCol - 1,
                                         row=2,
                                         col=len(bases[0]),
                                         number=5)
    # get all base mtrx
    consisBasesScores = getConsistencyScoreMtrx(basesMtrx)
    mtx2map = Mtrx23dMap(baseTypeNum, basesMtrx, totalRow, totalCol, randomRowColIdx)
    samples = list(map(mtx2map.getSamples, partitions))
    samples = torch.cat(samples, 0)
    rowFeature, colFeature = mtx2map.getBaseMapNoMaxPooling(samples)
    return (samples, rowFeature, colFeature, consisBasesScores)


# end


def nonZeroNum(sample):
    if len(torch.nonzero(sample)) >= 6:
        return (len(torch.nonzero(sample)))
    else:
        return (0)


# end


def getSamplesLabels(samples):
    labels = list(map(nonZeroNum, samples))
    labels = torch.tensor(labels)
    return (labels, samples)


# end


def get3dMap(probType, totalRow, totalCol, datas):
    # l1 bases
    bases = list()
    if probType == "l1c":
        b1 = [1, 1, 1, 1, 1, 1, 0]
        b2 = [1, 1, 1, 1, 0, 0, 0]
        b3 = [1, 1, 0, 0, 0, 0, 0]
        bases = [b1, b2, b3]
    elif probType == "l1":
        b1 = [1, 1, 1, -1, -1, -1, 0]
        b2 = [1, 1, -1, -1, 0, 0, 0]
        b3 = [1, -1, 0, 0, 0, 0, 0]
        bases = [b1, b2, b3]
    elif probType == "lall":
        b1 = [1, 1, 1, 1, 1, 1, 0]
        b2 = [1, 1, 1, 1, 0, 0, 0]
        b3 = [1, 1, 0, 0, 0, 0, 0]
        b4 = [1, 1, 1, -1, -1, -1, 0]
        b5 = [1, 1, -1, -1, 0, 0, 0]
        b6 = [1, -1, 0, 0, 0, 0, 0]
        bases = [b1, b2, b3, b4, b5, b6]
    else:
        base1 = [1, 1, 1, -1, -1, -1, 0]
        base2 = [1, 1, -1, -1, 0, 0, 0, ]
        base3 = [1, -1, 0, 0, 0, 0, 0]
        base4 = [2, 1, 1, -1, -1, -2, 0]
        bases = [base1, base2, base3, base4]
    randomRowColIdx = getRandomRowColIdx(low=0,
                                         hight=totalCol - 1,
                                         row=2,
                                         col=len(bases[0]),
                                         number=500)

    # mtx2map entity
    mtx2map = Mtrx23dMap(baseTypeNum, basesMtrx, totalRow, totalCol,
                         randomRowColIdx)

    mapDatas = list(map(mtx2map.main, datas))
    mapDatas = torch.stack(mapDatas, 0)
    return (mapDatas)


# end

def delFeature(fTmp, idx, scoreTmp, conThreshold):
    colNum = scoreTmp.size()[1]
    saveIdx = [i for i in range(colNum) if idx != i and fTmp[idx][i] < conThreshold]
    fTmp = fTmp[saveIdx]
    scoreTmp = scoreTmp[:, saveIdx]
    return (fTmp, scoreTmp)


# end

@pysnooper.snoop()
def getOptFeature(f_b_c):
    feature, basesConsisScores, conThreshold = f_b_c[0], f_b_c[1], f_b_c[2]
    fSort = feature.t().clone()  # 7*M->M*7
    fSort = torch.sort(fSort)
    fSort = fSort[0]
    # rowNum = fSort.size()[0]
    colNum = fSort.size()[1]
    optFeatures = list()
    for c in range(colNum):
        fMaxMeans = list()
        fTmp = fSort.clone()
        scoreTmp = basesConsisScores.clone()
        while len(fTmp) > 1:
            fMean = torch.mean(fTmp, dim=1, keepdim=True)
            fMaxMean, fMaxIdx = torch.max(fMean, 0)
            fMaxMeans.append(fMaxMean)
            fTmp, scoreTmp = delFeature(fTmp, fMaxIdx, scoreTmp, conThreshold)
        optFeatures.append(fMaxMeans)
        if c == colNum:
            break
        fSort = fSort[:, c + 1:]
    maxLen = max([(len(f), f) for f in optFeatures])
    for f in optFeatures:
        f.extend(0.0 for _ in range(maxLen - len(f)))
    optFeatures = torch.stack(optFeatures).t()
    sys.exit()
    return (optFeatures)


# end

# sparse ssvd
def ssvd(x, r=1):
    x = np.array(x)
    sptop = svds(x, k=r)
    sptoptmp = np.zeros((sptop[0].shape[1], sptop[2].shape[0]))
    sptoptmp[np.diag_indices(np.min(sptoptmp.shape))] = sptop[1]
    sptopreconstruct = np.dot(sptop[0], np.dot(sptoptmp, sptop[2]))
    sptopreconstruct = torch.tensor(sptopreconstruct)
    return (sptopreconstruct)


# end


# transtorm 3 1x7 1d bases vectors to 1 7xm 2d matrix
def getBaseMtrx(base):
    basePermsMtrx = list(set(list(permutations(base, len(base)))))
    basePermsMtrx = np.stack(basePermsMtrx).T
    return (basePermsMtrx)


# end


def getBasesMtrxs(bases):
    basesMtrxs = [torch.tensor(getBaseMtrx(base)) for base in bases]
    baseTypeNum = [basesMtrx.shape[1] for basesMtrx in basesMtrxs]
    basesMtrx = torch.cat(basesMtrxs, 1)  # to 7x(m1+m2+m3)
    baseTypeNum = np.insert(baseTypeNum, 0, 0)
    baseTypeNum = list(itertools.accumulate(baseTypeNum))
    return (baseTypeNum, basesMtrx)


# end

def getConsistencyScoreMtrx(basesMtrx):
    conBasesMtrx = basesMtrx.t().clone()
    rowNum = conBasesMtrx.size()[0]
    colNum = conBasesMtrx.size()[1]
    consisScoreMtrx = torch.zeros(rowNum, rowNum)
    for i1 in range(rowNum - 1):
        for i2 in range(i1 + 1, rowNum):
            score = 0
            for j in range(colNum):
                if conBasesMtrx[i1][j] == conBasesMtrx[i2][j]:
                    score = score + 1
                else:
                    score = score - 1
            consisScoreMtrx[i1][i2] = consisScoreMtrx[i2][i1] = score
    return (conBasesMtrx)


# end

def getInconsistencyBasesMtrxs(basesMtrx, threshold):
    inconBasesMtrx = basesMtrx.t().clone()
    rowNum = inconBasesMtrx.size()[0]
    colNum = inconBasesMtrx.size()[1]
    ind = list()
    for i1 in range(rowNum - 1):
        if i1 in ind:
            continue
        for i2 in range(i1 + 1, rowNum):
            if i2 in ind:
                continue
            score = 0
            for j in range(colNum):
                if inconBasesMtrx[i1][j] == inconBasesMtrx[i2][j]:
                    score = score + 1
                else:
                    score = score - 1
            if score <= threshold:
                ind.append(i2)
    ind.insert(-1, 0)
    inconBasesMtrx = inconBasesMtrx[ind, :]
    inconBasesMtrx = inconBasesMtrx.t()
    return (inconBasesMtrx)


# end


# get all row index list and col index list randomly
def getRandomRowColIdx(low=0, hight=49, row=2, col=7, number=10):
    randomRowIdx = [
        np.sort(np.random.choice(range(low, hight), col, replace=False))
        for _ in range(number)
    ]
    randomColIdx = [
        np.sort(np.random.choice(range(low, hight), col, replace=False))
        for _ in range(number)
    ]
    randomRowColIdx = list(zip(randomRowIdx, randomColIdx))
    return (randomRowColIdx)


# get all row index list and col index list randomly


# merge 2 3d matries
def merge2Mtrx(smallMtrx, bigMtrx, r, c, replace=0):
    # the size of a and b can be different
    z = 0
    if replace == 0:
        bigMtrx[z:smallMtrx.shape[0], r:r + smallMtrx.shape[1], c:c +
                                                                  smallMtrx.shape[2]] += smallMtrx
    else:
        bigMtrx[z:smallMtrx.shape[0], r:r + smallMtrx.shape[1], c:c +
                                                                  smallMtrx.shape[2]] = smallMtrx
    return (bigMtrx)


# get l1c mean blocks data background is zero
def getL1CMeanData(mean,
                   stdBias,
                   noiseNorm,
                   noiseMbias,
                   noiseStdbias,
                   num,
                   zn,
                   xn,
                   yn,
                   totalRow,
                   totalCol,
                   overlap=0):
    std = torch.ones(zn, xn, yn).add_(stdBias)
    # noise parameters
    noiseMean = mean + noiseMbias
    noisestd = torch.ones(zn, totalRow, totalCol).add_(noiseStdbias)
    noiseData = torch.normal(noiseMean, noisestd)

    # prepare data
    labels_datas = list()
    for i in range(1, num + 1):
        label = i
        blocks = list()
        blocks = [
            (torch.normal(mean, std) + (torch.randn(zn, xn, yn) * noiseNorm))
            for k in range(i)
        ]

        addNoiseRes = merge2Mtrx(blocks[0], noiseData, 0, 0)
        if i == 1:
            labels_datas.append([label, addNoiseRes.clone()])
            continue

        if overlap == 0:
            r, c = xn, yn
            for j in range(1, i):
                addNoiseRes = merge2Mtrx(blocks[j], addNoiseRes, r, c)

                r, c = r + xn, c + yn
            labels_datas.append([label, addNoiseRes.clone()])
        else:
            r, c = int(xn / 2) + 1, int(yn / 2) + 1
            for j in range(1, i):
                addNoiseRes = merge2Mtrx(blocks[j], addNoiseRes, r, c)
                r, c = r + int(xn / 2) + 1, c + int(yn / 2) + 1
            labels_datas.append([label, addNoiseRes.clone()])
    return (labels_datas)


# get l1c normal blocks data
def getL1CNormData(normalBias, minusMean, num, zn, xn, yn, totalRow, totalCol,
                   overlap, replace):
    # noise parameters
    gaussianNoise = torch.randn(zn, totalRow, totalCol)
    gaussianNoise = gaussianNoise - torch.mean(gaussianNoise)
    # gaussianNoise = torch.zeros(zn, totalRow, totalCol)  # zero noise
    # prepare normal distribution data
    labels_datas = list()
    for i in range(1, num + 1):
        label = i
        blocks = list()
        if minusMean == 0:
            blocks = [torch.randn(zn, xn, yn) + normalBias for _ in range(i)]
        else:
            for _ in range(i):
                block = torch.randn(zn, xn, yn)
                block = block - torch.mean(block) + normalBias
                blocks.append(block)

        addNoiseRes = merge2Mtrx(blocks[0], gaussianNoise, 0, 0, replace)
        if i == 1:
            labels_datas.append([label, addNoiseRes.clone()])
            continue
        if overlap == 0:
            r, c = xn, yn
            for j in range(1, i):
                addNoiseRes = merge2Mtrx(blocks[j], addNoiseRes, r, c, replace)
                r, c = r + xn, c + yn
            labels_datas.append([label, addNoiseRes.clone()])
        else:
            r, c = int(xn / 2) + 1, int(yn / 2) + 1
            for j in range(1, i):
                addNoiseRes = merge2Mtrx(blocks[j], addNoiseRes, r, c, 1)
                r, c = r + int(xn / 2) + 1, c + int(yn / 2) + 1
            labels_datas.append([label, addNoiseRes.clone()])
    return (labels_datas)


#  get l1 mean blocks data background is 0
def getL1MeanData(mean,
                  stdBias,
                  noiseNorm,
                  noiseMbias,
                  noiseStdbias,
                  num,
                  zn,
                  xn,
                  yn,
                  totalRow,
                  totalCol,
                  overlap=0):
    std = torch.ones(1, yn).add_(stdBias)
    blocksNum = num * zn
    blocks = [((torch.normal(mean, std).t() * torch.normal(mean, std)) +
               (torch.randn(xn, yn) * noiseNorm)) for _ in range(blocksNum)]
    blocks = torch.stack(blocks).view(num, zn, xn, yn)

    # noise parameters
    noiseMean = int(torch.mean(blocks) + noiseMbias)
    noisestd = torch.ones(zn, totalRow, totalCol).add_(noiseStdbias)
    noiseData = torch.normal(noiseMean, noisestd)
    # noiseData = torch.zeros(zn, totalRow, totalCol)
    labels_datas = list()
    for i in range(1, num + 1):
        label = i
        addNoiseRes = merge2Mtrx(blocks[0], noiseData, 0, 0)
        if i == 1:
            labels_datas.append([label, addNoiseRes.clone()])
            continue
        if overlap == 0:
            r, c = xn, yn
            for j in range(1, i):
                addNoiseRes = merge2Mtrx(blocks[j], addNoiseRes, r, c)
                r, c = r + xn, c + yn
            labels_datas.append([label, addNoiseRes.clone()])

        else:
            r, c = int(xn / 2) + 1, int(yn / 2) + 1
            for j in range(1, i):
                addNoiseRes = merge2Mtrx(blocks[j], addNoiseRes, r, c)
                r, c = r + int(xn / 2) + 1, c + int(yn / 2) + 1
            labels_datas.append([label, addNoiseRes.clone()])

    return (noiseMean, labels_datas)


#  get lk normal blocks data background noise is gaussian noise
def getLkNormData(lk, normalBias, minusMean, num, zn, xn, yn, totalRow,
                  totalCol, overlap, replace):
    blocksNum = num * zn
    blocks = list()
    if minusMean == 0:
        blocks = [
            torch.matmul(torch.randn(xn, lk), torch.randn(lk, xn)) + normalBias
            for _ in range(blocksNum)
        ]
    else:
        for _ in range(blocksNum):
            block = torch.matmul(torch.randn(xn, lk), torch.randn(lk, xn))
            block = block - torch.mean(block) + normalBias
            blocks.append(block)
    blocks = torch.stack(blocks).view(num, zn, xn, yn)
    # gaussian noise parameters
    gaussianNoise = torch.randn(zn, totalRow, totalCol)
    gaussianNoise = gaussianNoise - torch.mean(gaussianNoise)
    gaussianNoise = torch.zeros(zn, totalRow, totalCol)  # zero background noise

    labels_datas = list()
    for i in range(1, num + 1):
        label = i
        addNoiseRes = merge2Mtrx(blocks[0], gaussianNoise, 0, 0, replace)
        if i == 1:
            labels_datas.append([label, addNoiseRes.clone()])
            continue
        if overlap == 0:
            r, c = xn, yn
            for j in range(1, i):
                addNoiseRes = merge2Mtrx(blocks[j], addNoiseRes, r, c, replace)
                r, c = r + xn, c + yn
            labels_datas.append([label, addNoiseRes.clone()])

        else:
            r, c = int(xn / 2) + 1, int(yn / 2) + 1
            for j in range(1, i):
                addNoiseRes = merge2Mtrx(blocks[j], addNoiseRes, r, c, 1)
                r, c = r + int(xn / 2) + 1, c + int(yn / 2) + 1
            labels_datas.append([label, addNoiseRes.clone()])
    return (labels_datas)


#  get l1 base blocks data background noise is gaussian noise
def getL1SpeBaseData(baseAddNorm, minusMean, num, zn, xn, yn, totalRow,
                     totalCol, overlap, replace):
    base = []
    if baseAddNorm == 0:
        base1 = [1, ] * int(xn / 2)
        base2 = [-1, ] * (xn - int(xn / 2))
        bases = base1 + base2
    else:
        base1 = [1, ] * int(xn / 3)
        base2 = [-1, ] * int(xn / 3)
        base3 = torch.randn(1, xn - 2 * int(xn / 3))
        bases = base1 + base2 + base3
    base = torch.tensor(bases)
    blocksNum = num * zn
    blocks = list()
    if minusMean == 0:
        for _ in range(blocksNum):
            randomIdx = torch.tensor(np.random.choice(range(xn), xn, replace=False)).long()
            c = base[randomIdx].view(xn, 1).float()
            r = torch.randn(1, xn)
            blocks.append(torch.matmul(c, r))
    else:
        for _ in range(blocksNum):
            randomIdx = torch.tensor(np.random.choice(range(xn), xn, replace=False)).long()
            c = base[randomIdx].view(xn, 1).float()
            r = torch.randn(1, xn)
            block = torch.matmul(c, r)
            block = block - torch.mean(block)
            blocks.append(block)
    blocks = torch.stack(blocks).view(num, zn, xn, yn)
    # zero noise parameters
    # gaussianNoise = torch.randn(zn, totalRow, totalCol)
    # gaussianNoise = gaussianNoise - torch.mean(gaussianNoise)
    zeroNoise = torch.zeros(zn, totalRow, totalCol)  # zero background noise

    labels_datas = list()
    for i in range(1, num + 1):
        label = i
        addNoiseRes = merge2Mtrx(blocks[0], zeroNoise, 0, 0, replace)
        if i == 1:
            labels_datas.append([label, addNoiseRes.clone()])
            continue
        if overlap == 0:
            r, c = xn, yn
            for j in range(1, i):
                addNoiseRes = merge2Mtrx(blocks[j], addNoiseRes, r, c, replace)
                r, c = r + xn, c + yn
            labels_datas.append([label, addNoiseRes.clone()])

        else:
            r, c = int(xn / 2) + 1, int(yn / 2) + 1
            for j in range(1, i):
                addNoiseRes = merge2Mtrx(blocks[j], addNoiseRes, r, c, 1)
                r, c = r + int(xn / 2) + 1, c + int(yn / 2) + 1
            labels_datas.append([label, addNoiseRes.clone()])
    return (labels_datas)


# add many mean noise to the whole data
def addNumMeanNoise(data, label, num, mean, stdBias):
    std = torch.zeros(num,
                      data.size()[1],
                      data.size()[2],
                      data.size()[3]).add_(stdBias)
    noise = torch.normal(mean, std)
    noiseLabel = torch.tensor([
                                  0.0,
                              ] * num).long()
    data = torch.cat((data, noise), 0)
    label = torch.cat((label, noiseLabel), 0)
    return (label, data)


# add many mean noise to the whole data
def addNumGaussianNoise(data, label, num):
    gaussianNoise = torch.randn_like(data)
    gaussianNoise = gaussianNoise[0:num]
    gaussianNoise = gaussianNoise - torch.mean(gaussianNoise)
    noiseLabel = torch.tensor([
                                  0.0,
                              ] * num).long()
    data = torch.cat((data, gaussianNoise), 0)
    label = torch.cat((label, noiseLabel), 0)
    return (label, data)


# combine labels and data
def combineLabelData(labels_datas, zn, num):
    labels = torch.cat(
        [torch.tensor([labels_datas[i][0]] * zn).long() for i in range(num)])
    datas = torch.cat([labels_datas[i][1] for i in range(num)])
    return (labels, datas)


# shuffle data
def shuffleData(inputData):
    rowNum = inputData.shape[0]
    colNum = inputData.shape[1]
    rowNum = torch.randperm(rowNum)
    colNum = torch.randperm(colNum)
    inputData = inputData[rowNum[:, None], colNum]
    idx = torch.randperm(inputData.nelement())
    return (inputData.view(-1)[idx].view(inputData.size()))


# separate data into train and test
def separateData(labels, datas, sep):
    trainData = list()
    trainLabel = list()
    testData = list()
    testLabel = list()

    dataLen = datas.size()[0]
    for i in range(dataLen):
        if i % sep == 0:
            testData.append(datas[i])
            testLabel.append(labels[i])
        else:
            trainData.append(datas[i])
            trainLabel.append(labels[i])
    trainData = torch.stack(trainData)
    trainLabel = torch.stack(trainLabel)
    testData = torch.stack(testData)
    testLabel = torch.stack(testLabel)
    return (trainData, trainLabel, testData, testLabel)


def getCNNOutSize(inDim, layerNum, kernelSize):
    for i in range(layerNum - 1):
        inDim = int((inDim - (kernelSize - 1)) / 2)
    return (inDim)


def addDataError(data, mean, stdBias):
    std = torch.zeros_like(data).add_(stdBias)
    noise = torch.normal(mean, std)
    noise = noise - torch.mean(noise)
    return (data + noise)
