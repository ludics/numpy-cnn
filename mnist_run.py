#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by: ludi
# Created on: 2020/1/8


import numpy as np
import os
import time
from numpy_cnn.utils.load_mnist import get_mnist, BatchIterator
from numpy_cnn.utils.mylog import Logger
from numpy_cnn.utils.metric import accuracy
import numpy_cnn.layers as layers
from numpy_cnn.loss import CrossEntropyLoss
from numpy_cnn.optimizer import SGD, Adam
from numpy_cnn.net import Net
from numpy_cnn.model import Model


if not os.path.exists('./logs'):
    os.makedirs('./logs')
log = Logger('./logs/train.log',level='debug').logger

if __name__ == "__main__":
    data = get_mnist('./data/mnist')
    x_train, y_train, x_test, y_test = data
    x_train, x_test = x_train / 256, x_test / 256
    y_train = np.eye(10)[y_train.astype(np.int).reshape(-1)]
    y_test = np.eye(10)[y_test.astype(np.int).reshape(-1)]
    # net = Net([
    #     layers.Linear(784, 200),
    #     layers.ReLU(),
    #     layers.Linear(200, 100),
    #     layers.ReLU(),
    #     layers.Linear(100, 70),
    #     layers.ReLU(),
    #     layers.Linear(70, 30),
    #     layers.ReLU(),
    #     layers.Linear(30, 10)
    # ])
    net = Net([
        layers.Linear(784, 400),
        layers.ReLU(),
        layers.Linear(400, 100),
        layers.ReLU(),
        layers.Linear(100, 10)
    ])
    print(net.parameters)
    loss = CrossEntropyLoss()
    model = Model(net, loss, Adam, 1e-3)
    batch_size = 128
    iterator = BatchIterator(batch_size)
    for epoch in range(20):
        t_start = time.time()
        for inputs, labels in iterator(x_train, y_train):
            preds = model.forward(inputs)
            loss = model.backward(preds, labels)
        print("Epoch %d time cost: %.4f" % (epoch, time.time() - t_start))
        test_pred = model.forward(x_test)
        test_pred_idx = np.argmax(test_pred, axis=1)
        test_y_idx = np.argmax(y_test, axis=1)
        res = accuracy(test_pred_idx, test_y_idx)
        print(res)
        # for param in net.parameters:
        #     print('data')
        #     print(param.data)
        #     print('grad')
        #     print(param.grad)





    log.info('Just test')

