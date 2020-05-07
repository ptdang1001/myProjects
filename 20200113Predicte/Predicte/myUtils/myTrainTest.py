# -*- coding: utf8 -*

# system lib
import sys
import argparse
import os


# third part libs
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

# my libs
sys.path.append(
    "/N/u/pdang/Carbonate/projects/20200113Predicte/Predicte/myutils"
)

import mydata
import myevaluation

def train_test(label, data, Net, device, optimizer, lossFunc, opt):
    trainData, trainLabel, testData, testLabel = mydata.separateData(
        label, data, sep=5) 
    #-------------------------------train-------------------------------------
    dataSet = mydata.GeneDataset(trainData, trainLabel)
    dataLoader = DataLoader(dataset=dataSet,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=opt.n_cpu,
                            pin_memory=torch.cuda.is_available())
    # training
    #LossList = list()
    for epoch in range(opt.n_epochs):
        running_loss = 0.0
        for step, (x, y) in enumerate(dataLoader):
            b_x = Variable(x.to(device))  # batch data
            b_y = Variable(y.to(device))  # batch label
            output = Net(b_x)
            loss = lossFunc(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            '''
            running_loss += loss.data.cpu().item()
            if step % opt.n_print == (opt.n_print -
                                      1):  # print every 200 mini-batches
                print("[%d, %5d] loss: %.3f" %
                      (epoch + 1, step + 1, running_loss / opt.n_print))
                running_loss = 0.0
        # LossList.append(running_loss)
    #print("Train Finished!")
    #myutils.mydraw.draw(LossList, opt.n_epochs)
            '''
    # ----------------------------------------------------------test-----------------------------------------
    dataSet = mydata.GeneDataset(testData, testLabel)
    dataLoader = DataLoader(dataset=dataSet,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=opt.n_cpu,
                            pin_memory=torch.cuda.is_available())
    acc = 0.    # Accuracy
    SE = 0.     # Sensitivity (Recall)
    #SP = 0.     # Specificity
    PC = 0.     # Precision
    F1 = 0.     # F1 Score
    JS = 0.     # Jaccard Similarity
    #DC = 0.     # Dice Coefficient
    length=0
    for (x, y) in dataLoader:
        b_x = Variable(x.to(device))  # batch x (data)
        b_y = Variable(y.to(device))  # batch y (label)
        outputs = torch.sigmoid(Net(b_x))
        predicted = torch.max(outputs.data, 1)[1].cpu()
        #correct += (predicted == GT).sum().item()
        b_y=b_y.cpu()
        acc += myevaluation.get_accuracy(b_y, predicted)

        SE += myevaluation.get_sensitivity(b_y, predicted)
        #SP += myevaluation.get_specificity(b_y, predicted)
        PC += myevaluation.get_precision(b_y, predicted)
        F1 += myevaluation.get_F1(b_y, predicted)
        JS += myevaluation.get_JS(b_y, predicted)
        #DC += myevaluation.get_DC(b_y, predicted)
        length += 1

                    
    acc = round(100*acc/length,3)
    SE = round(SE/length,3)
    #SP = round(SP/length,4)
    PC = round(PC/length,3)
    F1 = round(F1/length,3)
    JS = round(JS/length,3)
    #DC = DC/length
    res = list()
    res.append(acc)
    res.append(SE)
    #res.append(SP)
    res.append(PC)
    res.append(F1)
    res.append(JS)
    #res.append(DC)
    res = ','.join(str(i) for i in res)

    return(res)
