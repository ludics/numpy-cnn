#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by: ludi
# Created on: 2020/1/8

from .layer import Layer
import numpy as np


class ReLU(Layer):
    def __init__(self):
        super().__init__("ReLU")
        self.x = None

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad):
        grad[self.x < 0] = 0
        return grad

class Sigmoid(Layer):
    def __init__(self):
        super().__init__("Sigmoid")
        self.y = None

    def forward(self, x):
        self.y = 1 / (1 + np.exp(-x))
        return self.y

    def backward(self, grad):
        return self.y * (1 - self.y) * grad


# class Softmax(Layer):
#     def forward(self, x):
#         v = np.exp(x - x.max(axis=-1, keepdims=True))
