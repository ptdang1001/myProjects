# -*- coding: utf-8 -*

# sys libs

# 3rd libs
import torch
import numpy as np
from numba import jit
from numba.typed import List
# my libs

# gene dataset loader
from torch.utils.data import Dataset


class MyDataset(Dataset):
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


# end

def getMaxIdx(corectEf, num):
    maxIdx = np.argpartition(corectEf.ravel(), -num)[-num:]
    maxIdx = np.unravel_index(maxIdx, corectEf.shape)
    maxIdx = List(set(np.append(maxIdx[0], maxIdx[1])))
    return (maxIdx)


@jit(nopython=True)
def addNantoEf(corectEf):
    for i in range(corectEf.shape[1]):
        for j in range(i, corectEf.shape[0]):
            corectEf[j][i] = np.nan
    return (corectEf)


# end

@jit(nopython=True)
def getLabelFrRCIdx(rowIdx, colIdx, patternLen):
    rn = 0
    cn = 0
    for i in rowIdx:
        if i < patternLen:
            rn += 1
    for i in colIdx:
        if i < patternLen:
            cn += 1
    return (rn * cn)


# @jit
def mateData2Parts(mateData):
    partRowNum = 50
    patternLen = 50
    partColNum = partRowNum
    labels_parts = list()
    loopNum = int(mateData.shape[0] / partRowNum)
    for i in range(loopNum):
        colCorectEf = np.corrcoef(mateData[0:5, :].T)
        colCorectEf[np.tril_indices(colCorectEf.shape[0], 0)] = np.nan
        colMaxIdx = getMaxIdx(colCorectEf, partColNum - 1)
        rowCorectEf = np.corrcoef(mateData[:, colMaxIdx])
        rowCorectEf[np.tril_indices(rowCorectEf.shape[0], 0)] = np.nan
        # rowCorectEf = addNantoEf(rowCorectEf)
        rowMaxIdx = getMaxIdx(rowCorectEf, partRowNum - 1)
        part = mateData[np.ix_(rowMaxIdx, colMaxIdx)]
        label = getLabelFrRCIdx(rowMaxIdx, colMaxIdx, patternLen)
        if len(part) == partRowNum:
            labels_parts.append([label, part])
        mateData = np.delete(mateData, rowMaxIdx, axis=0)
        mateData = np.delete(mateData, colMaxIdx, axis=1)
    return (labels_parts)


# end

class Part23dMap():
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
        rowFeature = samples * self.bases.float()
        colFeature = samples.transpose(0, 2, 1) * self.bases.float()
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

    def getRowColFeatureMapNoMaxPooling(self, samples):
        rowFeatureMap = np.matmul(samples, self.bases.astype(np.float64))
        colFeatureMap = np.matmul(samples.transpose(0, 2, 1), self.bases.astype(np.float64))
        # rowColFeatureMap = torch.stack((rowFeatureMap, colFeatureMap),1)  # 2*500*7*M->500*2*7*M
        return (rowFeatureMap, colFeatureMap)

    # end

    def getSamples(self, partition):
        samples = [
            partition[np.ix_(rowIdx, colIdx)]
            for rowIdx, colIdx in self.randomRowColIdx
        ]
        samples = np.stack(samples)
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


def getSamplesFeature(probType, mean, sampleNum, stdBias, numThreshold, partitions,
                      totalRow, totalCol):
    # l1 bases
    bases = []
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
        base2 = [1, 1, -1, -1, 0, 0, 0]
        base3 = [1, -1, 0, 0, 0, 0, 0]
        base4 = [2, 1, 1, -1, -1, -2, 0]
        bases = [base1, base2, base3, base4]
    baseTypeNum, basesMtrx = getBasesMtrxs(bases)
    randomRowColIdx = getRandomRowColIdx(low=0,
                                         hight=totalCol - 1,
                                         row=2,
                                         col=len(bases[0]),
                                         number=sampleNum)
    # get all base mtrx
    # consisBasesScoresL1 = getConsistencyScoreMtrx(basesMtrx)
    # consisBasesScoresL1 = np.array(consisBasesScoresL1)
    # np.save("C:\\Users\\pdang\\Desktop\\consisBasesScoresL1.npy",consisBasesScoresL1)

    # scorePath = "/N/project/zhangclab/pengtao/myProjectsDataRes/20200113Predicte/data/consisBasesScoresL1.npy"
    scorePath = "C:\\Users\\pdang\\Desktop\\consisBasesScoresL1.npy"
    consisBasesScores = torch.tensor(np.load(scorePath))
    mtx2map = Part23dMap(baseTypeNum, basesMtrx, totalRow, totalCol,
                         randomRowColIdx)
    samples = torch.cat(list(map(mtx2map.getSamples, partitions)))
    labels = torch.tensor(list(map(nonZeroNum, samples)))
    samples = addDataError(samples, mean, stdBias)
    zn, xn, yn = samples.size()
    samples = samples.view(zn, 1, xn, yn)
    labels, samples = addNumMeanNoise(samples, labels, int(len(samples) / 20),
                                      mean, stdBias)
    zn, _, xn, yn = samples.size()
    rowFeatureMap, colFeatureMap = mtx2map.getRowColFeatureMapNoMaxPooling(samples.view(zn, xn, yn))
    # combine row and col

    # get opt row and col
    rFM_cBS_nT = [[rfm, consisBasesScores, numThreshold] for rfm in rowFeatureMap]
    cFM_cBS_nT = [[cfm, consisBasesScores, numThreshold] for cfm in colFeatureMap]

    optRowFeatureMap = list(map(getOptFeatureMap, rFM_cBS_nT))
    optColFeatureMap = list(map(getOptFeatureMap, cFM_cBS_nT))
    # complete row and col to the same col number
    optRowFeatureMap = completeData2SameColLen(optRowFeatureMap)
    optColFeatureMap = completeData2SameColLen(optColFeatureMap)
    # combine opt row and col to one
    # reshape optFeatureMap data
    optFeatureMap = torch.stack((optRowFeatureMap, optColFeatureMap), dim=1)
    # reshap featureMap data
    featureMap = torch.stack((rowFeatureMap, colFeatureMap), dim=1)
    return (labels, samples, featureMap, optFeatureMap)


# end

def nonZeroNum(sample):
    if len(torch.nonzero(sample)) >= 6:
        return (len(torch.nonzero(sample)))
    else:
        return (0)


# end


def getLabelsFrmSamples(samples):
    labels = torch.tensor(list(map(nonZeroNum, samples)))
    return (labels)


# end


def get3dMap(part, numThreshold=30, probType="l1", partRow=50, partCol=50):
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
        base2 = [1, 1, -1, -1, 0, 0, 0]
        base3 = [1, -1, 0, 0, 0, 0, 0]
        base4 = [2, 1, 1, -1, -1, -2, 0]
        bases = [base1, base2, base3, base4]
    baseTypeNum, basesMtrx = getBasesMtrxs(bases)

    randomRowColIdx = getRandomRowColIdx(low=0,
                                         hight=partRow - 1,
                                         row=2,
                                         col=7,
                                         number=500)

    # mtx2map entity
    part2map = Part23dMap(baseTypeNum, basesMtrx, partRow, partCol,
                          randomRowColIdx)
    samples = part2map.getSamples(part)

    # get all base mtrx
    # consisBasesScoresL1 = getConsistencyScoreMtrx(basesMtrx)
    # consisBasesScoresL1 = np.array(consisBasesScoresL1)
    # np.save("C:\\Users\\pdang\\Desktop\\consisBasesScoresL1.npy",consisBasesScoresL1)
    scorePath = "/N/project/zhangclab/pengtao/myProjectsDataRes/20200113Predicte/data/consisBasesScoresL1.npy"
    #scorePath = "C:\\Users\\pdang\\Desktop\\consisBasesScoresL1.npy"
    consisBasesScores = np.load(scorePath)
    rowFeatureMap, colFeatureMap = part2map.getRowColFeatureMapNoMaxPooling(samples)
    # combine row and col

    # get opt row and col
    rFM_cBS_nT = [[rfm, consisBasesScores, numThreshold] for rfm in rowFeatureMap]
    cFM_cBS_nT = [[cfm, consisBasesScores, numThreshold] for cfm in colFeatureMap]

    optRowFeatureMap = list(map(getOptFeatureMap, rFM_cBS_nT))
    optColFeatureMap = list(map(getOptFeatureMap, cFM_cBS_nT))
    # complete row and col to the same col number
    # optRowFeatureMap = completeData2SameColLen(optRowFeatureMap)
    # optColFeatureMap = completeData2SameColLen(optColFeatureMap)
    # combine opt row and col to one
    # reshape optFeatureMap data
    optFeatureMap = torch.from_numpy(np.vstack((optRowFeatureMap, optColFeatureMap)))
    # reshap featureMap data
    featureMap = torch.from_numpy(np.vstack((rowFeatureMap, colFeatureMap)))
    return ([featureMap, optFeatureMap])


# end
@jit
def delFeature(fTmp, idx, scoreTmp, conThreshold):
    saveIdx = list()
    for j in np.arange(scoreTmp.shape[1]):
        if scoreTmp[idx][j] <= conThreshold:
            saveIdx.append(j)
    fTmp = fTmp[saveIdx]
    scoreTmp = scoreTmp[np.ix_(saveIdx, saveIdx)]
    return (fTmp, scoreTmp)


# end

def getOptFeatureMap_old(f_b_n):
    feature, basesConsisScores, numThreshold = f_b_n[0], f_b_n[1], f_b_n[2]
    fSort = feature.T.copy()  # 7*M->M*7
    fSort = np.sort(fSort)
    fSort = fSort[0]
    # rowNum = fSort.size()[0]
    colNum = fSort.size()[1]
    optFeatures = list()
    for c in range(colNum):
        fMaxMeans = list()
        fTmp = fSort.clone()
        scoreTmp = basesConsisScores.clone()
        while len(fTmp) >= 1:
            fMean = torch.mean(fTmp, dim=1, keepdim=True)
            fMaxMean, fMaxIdx = torch.max(fMean, 0)
            fMaxMeans.append(fMaxMean)
            if len(fMaxMeans) == numThreshold:
                break
            fMaxIdx = fMaxIdx.item()
            fTmp, scoreTmp = delFeature(fTmp, fMaxIdx, scoreTmp, 2)
        optFeatures.append(fMaxMeans)
        if c == colNum:
            break
        fSort = fSort[:, 1:]
    # maxLen = max([(len(f)) for f in optFeatures])
    # for f in optFeatures:
    #  f.extend(0.0 for _ in range(maxLen - len(f)))
    return (optFeatures)


# end
@jit
def getOptFeatureMap(f_b_n):
    feature, basesConsisScores, numThreshold = f_b_n[0], f_b_n[1], f_b_n[2]
    fSort = feature.T.copy()  # 7*M->M*7
    fSort = np.sort(fSort)
    colNum = fSort.shape[1]
    optFeatures = list()
    for c in np.arange(colNum):
        fMaxMeans = List()
        fTmp = fSort.copy()
        scoreTmp = basesConsisScores.copy()
        for i in np.arange(1000):
            fMean = np.mean(fTmp, axis=1)
            fMaxMean, fMaxIdx = np.max(fMean, 0), np.argmax(fMean, 0)
            fMaxMeans.append(fMaxMean)
            if len(fMaxMeans) == numThreshold:
                break
            fTmp, scoreTmp = delFeature(fTmp, fMaxIdx, scoreTmp, 2)
        optFeatures.append(fMaxMeans)
        if c == colNum:
            break
        fSort = fSort[..., 1:]
    return (optFeatures)


# sparse ssvd
def ssvd(x, r=1):
    from scipy.sparse.linalg import svds
    x = np.array(x)
    sptop = svds(x, k=r)
    sptoptmp = np.zeros((sptop[0].shape[1], sptop[2].shape[0]))
    sptoptmp[np.diag_indices(np.min(sptoptmp.shape))] = sptop[1]
    sptopreconstruct = np.dot(sptop[0], np.dot(sptoptmp, sptop[2]))
    sptopreconstruct = torch.tensor(sptopreconstruct)
    return (sptopreconstruct)


# end

# end


# transtorm 3 1x7 1d bases vectors to 1 7xm 2d matrix
def getBaseMtrx(base):
    from itertools import permutations
    basePermsMtrx = list(set(list(permutations(base, len(base)))))
    basePermsMtrx = np.stack(basePermsMtrx).T
    return (basePermsMtrx)


# end


def getBasesMtrxs(bases):
    from itertools import accumulate
    basesMtrxs = [getBaseMtrx(base) for base in bases]
    baseTypeNum = [basesMtrx.shape[1] for basesMtrx in basesMtrxs]
    basesMtrx = np.hstack(basesMtrxs)  # to 7x(m1+m2+m3)
    baseTypeNum = np.insert(baseTypeNum, 0, 0)
    baseTypeNum = list(accumulate(baseTypeNum))
    return (baseTypeNum, basesMtrx)


# end

def getConsistencyScoreMtrx(basesMtrx):
    conBasesMtrx = basesMtrx.t().clone()
    rowNum = conBasesMtrx.size()[0]
    colNum = conBasesMtrx.size()[1]
    consisScoreMtrx = torch.diag(torch.tensor([7] * rowNum), diagonal=0)
    for i1 in range(rowNum - 1):
        for i2 in range(i1 + 1, rowNum):
            score = 0
            for j in range(colNum):
                if conBasesMtrx[i1][j] == conBasesMtrx[i2][j]:
                    score = score + 1
                else:
                    score = score - 1
            consisScoreMtrx[i1][i2] = consisScoreMtrx[i2][i1] = score
    return (consisScoreMtrx)


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
    # gaussianNoise = torch.zeros(zn, totalRow, totalCol)  # zero background noise

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
def getL1SpeBaseData(crType, minusMean, errorStdBias, blockNum, baseTimes, zn, xn, yn, totalRow, totalCol,
                     overlap, replace):
    blocksNum = blockNum * zn
    mean = 0
    errorStd = torch.zeros(xn, yn).add_(errorStdBias)
    error = torch.normal(mean, errorStd)
    blocks = list()
    bases = list()
    if crType == "uniform":
        if minusMean == 1:
            for _ in range(blocksNum):
                c = torch.rand(xn, 1)
                r = torch.rand(1, xn)
                block = torch.matmul(c, r) * baseTimes
                block = block - torch.mean(block)
                blocks.append((block + error).numpy())
        else:
            for _ in range(blocksNum):
                c = torch.rand(xn, 1)
                r = torch.rand(1, xn)
                blocks.append((torch.matmul(c, r) * baseTimes + error).numpy())
    elif crType == "norm":
        if minusMean == 1:
            for _ in range(blocksNum):
                c = torch.randn(xn, 1)
                r = torch.randn(1, xn)
                block = torch.matmul(c, r) * baseTimes
                block = block - torch.mean(block)
                blocks.append((block + error).numpy())
        else:
            for _ in range(blocksNum):
                c = torch.randn(xn, 1)
                r = torch.randn(1, xn)
                blocks.append((torch.matmul(c, r) * baseTimes + error).numpy())
    else:
        base1 = [1]
        base2 = [-1]
        base3 = [0]
        bases = [base1, base2, base3]
        if minusMean == 0:
            for _ in range(blocksNum):
                randomIdx = torch.tensor(np.random.choice(range(xn), xn, replace=False)).long()
                c = bases[randomIdx].view(xn, 1).float()
                r = torch.randn(1, xn)
                blocks.append(torch.matmul(c, r))
            else:
                for _ in range(blocksNum):
                    randomIdx = torch.tensor(np.random.choice(range(xn), xn, replace=False)).long()
                c = bases[randomIdx].view(xn, 1).float()
                r = torch.randn(1, xn)
                block = torch.matmul(c, r)
                block = block - torch.mean(block)
                blocks.append(block)
    blocks = np.stack(blocks)
    blocks = np.reshape(blocks, (blockNum, zn, xn, yn))
    # noise parameters
    gaussianNoise = np.random.randn(zn, totalRow, totalCol)
    gaussianNoise = gaussianNoise - np.mean(gaussianNoise)
    # zeroNoise = torch.zeros(zn, totalRow, totalCol)  # zero background noise
    labels_datas = list()
    for i in range(1, blockNum + 1):
        label = i
        addNoiseRes = merge2Mtrx(blocks[0], gaussianNoise, 0, 0, replace)
        if i == 1:
            labels_datas.append([label, addNoiseRes.copy()])
            continue
        if overlap == 0:
            r, c = xn, yn
            for j in range(1, i):
                addNoiseRes = merge2Mtrx(blocks[j], addNoiseRes, r, c, replace)
                r, c = r + xn, c + yn
            labels_datas.append([label, addNoiseRes.copy()])
        else:
            r, c = int(xn / 2) + 1, int(yn / 2) + 1
            for j in range(1, i):
                addNoiseRes = merge2Mtrx(blocks[j], addNoiseRes, r, c, 1)
                r, c = r + int(xn / 2) + 1, c + int(yn / 2) + 1
        labels_datas.append([label, addNoiseRes.copy()])
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


# end


def completeData2SameColLen(data):
    maxColLen = 0
    for d in data:
        tmpLen = len(d[0])
        maxColLen = tmpLen if tmpLen > maxColLen else maxColLen
    for d in data:
        for col in d:
            col.extend(0.0 for _ in range(maxColLen - len(col)))
    data = torch.tensor(data)
    return (data)

# end
