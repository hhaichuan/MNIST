#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-6-3 下午3:12
# @Author  : hhaichuan
# @Site    : 
# @File    : trainedNetTest.py
# @Product: PyCharm Community Edition

import MNIST

faultSet, correctLabels, faultLabels = MNIST.testNet(100)
text = []
for i in range(len(faultLabels)):
    text.append('True: ' + str(correctLabels[i]) + ' Test: ' + str(faultLabels[i]))

an = MNIST.animate(text, faultSet)
an.disp()
