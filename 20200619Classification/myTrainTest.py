# -*- coding: utf8 -*

# system lib
import sys
import os

# third part libs
import numpy as np
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
    # train end

    # ----------------------------------------------------------test-----------------------------------------
    # test start
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
    # test end

    res = [rmse, acc, SE, PC, F1, JS]
    res = [round(r / length, 2) for r in res]
    # res = ','.join(str(i) for i in res)
    return (res, ytrue_ypred)


# end

def train_test_AE(data, net, device, optimizer, lossFunc, opt):
    zn, xn, yn = data.size()
    dataSet = myData.MyDataset(data, data)
    dataLoader = DataLoader(dataset=dataSet,
                            batch_size=opt.batch_size,
                            shuffle=False,
                            num_workers=opt.n_cpu,
                            pin_memory=torch.cuda.is_available())
    # train start
    for epoch in range(opt.n_epochs):
        for step, (x, _) in enumerate(dataLoader):
            b_x = Variable(x.view(-1, xn * yn).float().to(device))  # batch data
            encoded, decoded, _ = net(b_x)
            loss = lossFunc(decoded, b_x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            '''
            if step % 10 == 9:
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())
            '''
    # train end

    # ----------------------------------------------------------test-----------------------------------------
    # test start

    predLabels = list()
    for (x, y) in dataLoader:
        b_x = Variable(x.view(-1, xn * yn).float().to(device))  # batch x (data)
        _, _, label = net(b_x)
        predicted = torch.max(label.data, 1)[1].cpu()
        predLabels.append([predicted.numpy()])

    # test end
    predLabels = np.concatenate(predLabels, axis=1)
    # res = ','.join(str(i) for i in res)
    return (predLabels)


# end

def train_test_VAE(data, device, lossFunc, opt, net, optimizer):
    zn, xn, yn = data.size()
    dataSet = myData.MyDataset(data, data)
    dataLoader = DataLoader(dataset=dataSet,
                            batch_size=opt.batch_size,
                            shuffle=False,
                            num_workers=opt.n_cpu,
                            pin_memory=torch.cuda.is_available())
    # train start
    for epoch in range(opt.n_epochs):
        for step, (x, _) in enumerate(dataLoader):
            b_x = Variable(x.view(-1, xn * yn).float().to(device))  # batch data
            _, decoded = net(b_x)
            loss = lossFunc(decoded, b_x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 100 == 9:
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy())
            
    # train end

    # ----------------------------------------------------------test-----------------------------------------
    # test start
    predLabels = list()
    for (x, _) in dataLoader:
        b_x = Variable(x.view(-1, xn * yn).float().to(device))  # batch x (data)
        z, _ = net(b_x)
        predLabels.append([z.cpu().detach().numpy()])

    # test end
    predLabels = np.vstack(predLabels)
    zn, xn, yn = predLabels.shape
    predLabels = np.reshape(predLabels, (zn ,yn))
    return (predLabels)


# end


def train_test_GAN(data, device, lossFunc, opt, net_G, g_optimizer, net_D, d_optimizer, d_steps=16, g_steps=8):
    zn, xn, yn = data.size()
    dataSet = myData.MyDataset(data, data)
    dataLoader = DataLoader(dataset=dataSet,
                            batch_size=opt.batch_size,
                            shuffle=False,
                            num_workers=opt.n_cpu,
                            pin_memory=torch.cuda.is_available())

    for epoch in range(opt.n_epochs):
        for d_index in range(d_steps):
            for i, (x, _) in enumerate(dataLoader):
                # 1. Train D on real+fake
                net_D.zero_grad()

                #  1A: Train D on real
                d_real_data = Variable(x.view(-1, xn*yn).float().to(device))
                d_real_decision, _ = net_D(d_real_data)
                d_real_loss = lossFunc(d_real_decision, Variable(torch.ones_like(d_real_decision).to(device)))  # ones = true
                d_real_loss.backward()  # compute/store gradients, but don't change params

                #  1B: Train D on fake
                d_gen_input = Variable(torch.randn(100).to(device))
                d_fake_data = net_G(d_gen_input).detach()  # detach to avoid training G on these labels
                d_fake_decision, _ = net_D(d_fake_data)
                d_fake_loss = lossFunc(d_fake_decision, Variable(torch.zeros_like(d_fake_decision)).to(device))  # zeros = fake
                d_fake_loss.backward()
                d_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()

        for g_index in range(g_steps):
            # 2. Train G on D's response (but DO NOT train D on these labels)
            net_G.zero_grad()

            gen_input = Variable(torch.randn(100).to(device))
            g_fake_data = net_G(gen_input)
            dg_fake_decision, _ = net_D(g_fake_data)
            g_loss = lossFunc(dg_fake_decision, Variable(torch.ones_like(dg_fake_decision).to(device)))  # we want to fool, so pretend it's all genuine
            g_loss.backward()
            g_optimizer.step()  # Only optimizes G's parameters

    # train end

    # ----------------------------------------------------------test-----------------------------------------
    # test start

    predLabels = list()
    for (x, _) in dataLoader:
        x = Variable(x.view(-1, xn*yn).float().to(device))  # batch x (data)
        _, label = net_D(x)
        predicted = torch.max(label.data, 1)[1].cpu().numpy()
        predLabels.append([predicted])
        # test end
    predLabels = np.concatenate(predLabels, axis=1)
    return (predLabels)
# end
