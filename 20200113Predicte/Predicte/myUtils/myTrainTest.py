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

def train_test_VAE(data, net, device, optimizer, lossFunc, opt):
    n, zn, xn, yn = data.size()
    dataSet = myData.MyDataset(data, data)
    dataLoader = DataLoader(dataset=dataSet,
                            batch_size=opt.batch_size,
                            shuffle=False,
                            num_workers=opt.n_cpu,
                            pin_memory=torch.cuda.is_available())
    # train start
    for epoch in range(opt.n_epochs):
        for step, (x, _) in enumerate(dataLoader):
            b_x = Variable(x.view(-1, zn * xn * yn).float().to(device))  # batch data
            decoded, _, _, _ = net(b_x)
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
        b_x = Variable(x.view(-1, zn * xn * yn).float().to(device))  # batch x (data)
        _, label, _, _ = net(b_x)
        predicted = torch.max(label.data, 1)[1].cpu()
        predLabels.append([predicted.numpy()])

    # test end
    predLabels = np.concatenate(predLabels, axis=1)
    # res = ','.join(str(i) for i in res)
    return (predLabels)


# end


def train_test_GD(data, net_G, net_D, device, optimizer_G, optimizer_D, lossFunc, opt):
    zn, xn, yn = data.size()
    dataSet = myData.MyDataset(data, data)
    dataLoader = DataLoader(dataset=dataSet,
                            batch_size=opt.batch_size,
                            shuffle=False,
                            num_workers=opt.n_cpu,
                            pin_memory=torch.cuda.is_available())
    for i, (x, _) in enumerate(dataLoader):
        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_x = Variable(x).float().to(device)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(torch.randn(xn, yn)).float().to(device)

        # Generate a batch of images
        gen_x = net_G(z)

        # Loss measures generator's ability to fool the discriminator
        score, _ = net_D(gen_x)
        g_loss = lossFunc(score, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        score, _ = net_D(real_x)
        real_loss = lossFunc(score, valid)
        score, _ = net_D(gen_x.detach())
        fake_loss = lossFunc(score, fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()
    for epoch in range(num_epochs):
        for d_index in range(d_steps):
            # 1. Train D on real+fake
            D.zero_grad()

            #  1A: Train D on real
            d_real_data = Variable(d_sampler(d_input_size))
            d_real_decision = D(preprocess(d_real_data))
            d_real_error = criterion(d_real_decision, Variable(torch.ones(1)))  # ones = true
            d_real_error.backward()  # compute/store gradients, but don't change params

            #  1B: Train D on fake
            d_gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
            d_fake_data = G(d_gen_input).detach()  # detach to avoid training G on these labels
            d_fake_decision = D(preprocess(d_fake_data.t()))
            d_fake_error = criterion(d_fake_decision, Variable(torch.zeros(1)))  # zeros = fake
            d_fake_error.backward()
            d_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()

        for g_index in range(g_steps):
            # 2. Train G on D's response (but DO NOT train D on these labels)
            G.zero_grad()

            gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
            g_fake_data = G(gen_input)
            dg_fake_decision = D(preprocess(g_fake_data.t()))
            g_error = criterion(dg_fake_decision,
                                Variable(torch.ones(1)))  # we want to fool, so pretend it's all genuine

            g_error.backward()
            g_optimizer.step()  # Only optimizes G's parameters
    # train end

    # ----------------------------------------------------------test-----------------------------------------
    # test start

    predLabels = list()
    for (x, _) in dataLoader:
        x = Variable(x).float().to(device)  # batch x (data)
        _, label = net_D(x)
        predicted = torch.max(label.data, 1)[1].cpu()
        predLabels.append([predicted.numpy()])

        # test end
        predLabels = np.concatenate(predLabels, axis=1)
        # res = ','.join(str(i) for i in res)
    return (predLabels)
# end
