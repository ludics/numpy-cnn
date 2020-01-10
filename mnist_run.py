#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by: ludi
# Created on: 2020/1/8


import numpy as np
import os
import time
from numpy_cnn.utils.dataset import load_mnist, batch_iterator
from numpy_cnn.utils.mylog import Logger
import numpy_cnn.layers as layers
from numpy_cnn.loss import CrossEntropyLoss
from numpy_cnn.optimizer import SGD, Adam
from numpy_cnn.net import Net


if not os.path.exists('./logs'):
    os.makedirs('./logs')
log = Logger('./logs/train.log',level='debug').logger


def accuracy(preds, labels):
    preds_idx = np.argmax(preds, axis=1)
    labels_idx = np.argmax(labels, axis=1)
    total_num = len(preds_idx)
    hit_num = int(np.sum(preds_idx == labels_idx))
    return {"total_num": total_num,
            "hit_num": hit_num,
            "acc": 1.0 * hit_num / total_num}


def train(net, data, num_epochs, batch_size, lr=1e-3):
    x_train, y_train, x_test, y_test = data
    x_train, x_test = x_train / 256, x_test / 256
    y_train = np.eye(10)[y_train.astype(np.int).reshape(-1)]
    y_test = np.eye(10)[y_test.astype(np.int).reshape(-1)]
    optim = Adam(net.parameters, lr=lr)
    loss = CrossEntropyLoss()
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        if epoch % 50 == 0:
            lr *= 0.1
        for inputs, labels in batch_iterator(x_train, y_train, batch_size):
            preds = net.forward(inputs)
            l = loss.loss(preds, labels)
            grad = loss.grad(preds, labels)
            net.backward(grad)
            optim.update()
            train_l_sum += l
            train_acc_sum += accuracy(preds, labels)['hit_num']
            n += labels.shape[0]
            batch_count += 1
        test_preds = net.forward(x_test)
        test_res = accuracy(test_preds, y_test)
        log.info('epoch %3d, lr %.6f, loss %.6f, train acc %.6f, test acc %.6f, time %.1f sec'
              % (epoch + 1, lr, train_l_sum / n, train_acc_sum / n, test_res['acc'], time.time() - start))


if __name__ == "__main__":
    data = load_mnist('./data/mnist')

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
    # log.info("Net parameters: " + str(net.parameters))
    log.info(net)
    batch_size = 128
    num_epochs = 20
    train(net, data, num_epochs, batch_size, lr=1e-3)
    log.info('Train Complete')

