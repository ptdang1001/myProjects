# -*- coding: utf8 -*

# system lib
import sys
import os

# third part libs
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

# my libs
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import myEvaluation
import myData


def train_test(label, data, Net, device, optimizer, lossFunc, opt):
    trainData, trainLabel, testData, testLabel = myData.separateData(label,
                                                                     data,
                                                                     sep=5)
    # -------------------------------train-------------------------------------
    dataSet = myData.MyDataset(trainData, trainLabel)
    dataLoader = DataLoader(dataset=dataSet,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=opt.n_cpu,
                            pin_memory=torch.cuda.is_available())
    # train start
    for epoch in range(opt.n_epochs):
        for step, (x, y) in enumerate(dataLoader):
            b_x = Variable(x.to(device))  # batch data
            b_y = Variable(y.to(device))  # batch label
            output = Net(b_x)
            loss = lossFunc(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    #train end

    # ----------------------------------------------------------test-----------------------------------------
    #test start
    dataSet = myData.MyDataset(testData, testLabel)
    dataLoader = DataLoader(dataset=dataSet,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=opt.n_cpu,
                            pin_memory=torch.cuda.is_available())
    rmse = 0.0  # root mean square error
    acc = 0.0  # Accuracy
    SE = 0.0  # Sensitivity (Recall)
    PC = 0.0  # Precision
    F1 = 0.0  # F1 Score
    JS = 0.0  # Jaccard Similarity
    ytrue_ypred = list()
    length = 0
    for (x, y) in dataLoader:
        b_x = Variable(x.to(device))  # batch x (data)
        b_y = Variable(y.to(device))  # batch y (label)
        outputs = torch.sigmoid(Net(b_x))
        predicted = torch.max(outputs.data, 1)[1].cpu()
        b_y = b_y.cpu()
        ytrue_ypred.append([b_y.numpy(), predicted.numpy()])
        rmse += myEvaluation.get_RMSE(b_y, predicted)
        acc += myEvaluation.get_accuracy(b_y, predicted)
        SE += myEvaluation.get_sensitivity(b_y, predicted)
        PC += myEvaluation.get_precision(b_y, predicted)
        F1 += myEvaluation.get_F1(b_y, predicted)
        JS += myEvaluation.get_JS(b_y, predicted)
        length += 1
    #test end

    res = [rmse, acc, SE, PC, F1, JS]
    res = [round(r / length, 2) for r in res]
    # res = ','.join(str(i) for i in res)
    return (res, ytrue_ypred)
