#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by: ludi
# Created on: 2020/1/8


import numpy as np
from .layers import Layer


class Model(Layer):
    def __init__(self, net, loss, optim, lr=1e-3):
        super().__init__("Model")
        self.net = net
        self.loss = loss
        self.optim = optim(self.net.parameters, lr)

    def forward(self, inputs):
        return self.net.forward(inputs)

    def backward(self, preds, targets):
        # print('loss grad')
        loss = self.loss.loss(preds, targets)
        # print(loss)
        grad = self.loss.grad(preds, targets)
        # print(grad)
        self.net.backward(grad)
        self.optim.update()
        return loss