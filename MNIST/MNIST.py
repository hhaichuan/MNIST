#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-5-29 上午11:45
# @Author  : hhaichuan
# @Site    : 
# @File    : MNIST.py
# @Product: PyCharm Community Edition

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import gzip
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def readData(filename, normParam):
    with open(filename, 'rb') as f:
        print('Extracting', f.name)
        with gzip.GzipFile(fileobj=f) as bytestream:
            dt = np.dtype(np.uint32).newbyteorder('>')
            magic = np.frombuffer(bytestream.read(4), dtype=dt)[0]
            num_images = np.frombuffer(bytestream.read(4), dtype=dt)[0]
            rows = np.frombuffer(bytestream.read(4), dtype=dt)[0]
            cols = np.frombuffer(bytestream.read(4), dtype=dt)[0]
            buf = bytestream.read(rows * cols * num_images)
            data = np.frombuffer(buf, dtype=np.uint8)
            read_data = data.reshape(num_images, 1, rows, cols)/255.0
            read_data = (read_data - normParam[0])/normParam[1]
    return read_data

def readLabels(filename):
    with open(filename, 'rb') as f:
        print('Extracting', f.name)
        with gzip.GzipFile(fileobj=f) as bytestream:
            dt = np.dtype(np.uint32).newbyteorder('>')
            magic = np.frombuffer(bytestream.read(4), dtype=dt)[0]
            num_items = np.frombuffer(bytestream.read(4), dtype=dt)[0]
            buf = bytestream.read(num_items)
            read_data = np.frombuffer(buf, dtype=np.uint8).squeeze()
    return read_data

def getTrainTensor(batch_N, normParam):
    local_file = './data/train-images-idx3-ubyte.gz'
    trainData = readData(local_file, normParam)
    local_file = './data/train-labels-idx1-ubyte.gz'
    trainLabels = np.array(readLabels(local_file),dtype='int')
    trainSet = torch.utils.data.TensorDataset(torch.FloatTensor(trainData), torch.LongTensor(trainLabels))
    return DataLoader(trainSet, batch_N, shuffle=True)

def getTestTensor(batch_N, normParam):
    local_file = './data/t10k-images-idx3-ubyte.gz'
    testData = readData(local_file, normParam)
    local_file = './data/t10k-labels-idx1-ubyte.gz'
    testLabels = np.array(readLabels(local_file), dtype='int')
    testSet = torch.utils.data.TensorDataset(torch.FloatTensor(testData), torch.LongTensor(testLabels))
    return DataLoader(testSet, batch_N)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.Conv1 = nn.Conv2d(1,6,5)
        self.Pool1 = nn.MaxPool2d(2,2)
        self.Conv2 = nn.Conv2d(6,16,3)
        self.Pool2 = nn.MaxPool2d(2,2)
        self.Linear1 = nn.Linear(16*5*5,120)
        self.Linear2 = nn.Linear(120, 84)
        self.Linear3 = nn.Linear(84,10)

    def forward(self, x):
        x = self.Pool1(F.relu(self.Conv1(x)))
        x = self.Pool2(F.relu(self.Conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.Linear1(x))
        x = F.relu(self.Linear2(x))
        x = self.Linear3(x)
        return x

def trainNet(cycles, batch_N):
    net = Net()
    net.cuda()
    normParam = [0.5, 0.5]
    trainSet = getTrainTensor(batch_N, normParam)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(cycles):
        optimizer = optim.SGD(net.parameters(), lr=0.05 / (np.log(epoch + 1.0) + 1.0), momentum=0.9)
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainSet, 0):
            dataSet, labelArray = data
            inputs, labels = Variable(dataSet.cuda()), Variable(labelArray.cuda())
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
            total += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted.cpu().numpy().squeeze() == labelArray.numpy()).sum()
            print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss ))
            print("TrainAccuray = %.3f %%" %(100.0* correct / total))
    print('Finished Training')
    torch.save(net.state_dict(), './data/net_params.pkl')

def testNet(batch_N):
    normParam = [0.5, 0.5]
    net = Net()
    net.load_state_dict(torch.load('./data/net_params.pkl'))
    testSet = getTestTensor(batch_N, normParam)
    total = 0.0
    correct = 0.0
    faultSet = []
    correctLabels = []
    faultLabels = []
    for data, labels in testSet:
        inputs, labelsArr = Variable(data), Variable(labels)
        outputs = net(inputs)
        value, predict = torch.max(outputs.data, 1)
        judgement = predict.numpy().squeeze() == labelsArr.data.numpy().squeeze()
        correct += judgement.sum()
        total += labelsArr.size()[0]
        if not judgement.all():
            fault = ~ judgement
            wrongdata = data.numpy()[fault]
            trueLabels = labels.numpy()[fault]
            wrongLables = predict.numpy().squeeze()[fault]
            l = len(trueLabels)
            for i in range(l):
                faultSet.append(wrongdata[i])
                correctLabels.append(trueLabels[i])
                faultLabels.append(wrongLables[i])
    faultSet = (np.array(faultSet).squeeze() * 0.5 + 0.5) * 255
    correctLabels = np.array(correctLabels).squeeze()
    faultLabels = np.array(faultLabels).squeeze()
    print("TestAccuray = %.3f %%" % (100.0 * correct / total))
    return faultSet, correctLabels, faultLabels


class animate(object):
    def __init__(self, text, img):
        self.fig = plt.figure()
        self.ax = self.fig.add_axes([0,0,1,1])
        self.text = text
        self.img = img
        self.left, self.width = .25, .5
        self.bottom, self.height = .25, .5
        self.right = self.left + self.width
        self.top = self.bottom + self.height
        self.length = len(text)
        self.im = self.ax.imshow(self.img[0], animated=True)
        self.tx = self.ax.text(self.right, self.top, text[0], fontsize=20, color='yellow')

    def init(self):
        im = self.ax.imshow(self.img[1], animated=True)
        tx = self.ax.text(self.right, self.top, self.text[1], fontsize=20, color='yellow')
        return im, tx

    def update(self, i):
        self.im.set_array(self.img[i])
        self.tx.set_text(self.text[i])
        return  self.im, self.tx

    def disp(self):
        ani = animation.FuncAnimation(self.fig, self.update, np.arange(2, self.length), interval=200, blit=True, init_func=self.init)
        plt.show()

trainNet(50, 100)
